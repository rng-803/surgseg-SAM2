# surgseg-SAM2
# SAM 2 (+ Tiny Refiner) — Surgical Video Segmentation Pipeline

This README is a practical roadmap to set up, train, and deploy a **SAM 2 + tiny refiner head** for surgical **video** segmentation with a focus on **precision** and **real‑time/near‑real‑time inference**.

---

## Table of Contents
1. [Goals & Assumptions](#goals--assumptions)
2. [Dataset Preparation](#dataset-preparation)
3. [Offline SAM2 Prior Generation](#offline-sam2-prior-generation)
4. [Refiner Head Model & Training](#refiner-head-model--training)
5. [Optional Temporal Consistency (Train-Only)](#optional-temporal-consistency-train-only)
6. [End-to-End Streaming Inference](#end-to-end-streaming-inference)
7. [Export & Deployment](#export--deployment)
8. [Latency/Accuracy Sweeps](#latencyaccuracy-sweeps)
9. [Monitoring, QA & Regression](#monitoring-qa--regression)
10. [Deliverables & Folder Structure](#deliverables--folder-structure)
11. [High-Level Checklist](#high-level-checklist)

---

## Goals & Assumptions
- **Goal:** Surgical **video** segmentation with high temporal stability and low latency.
- **Approach:** Use **SAM 2** for streaming/temporal memory and a **tiny refiner head** trained on your labeled frames to sharpen class boundaries and reduce drift.
- **Hardware Assumption:** 1× NVIDIA GPU (e.g., RTX 4070/4090) for training. Deployment will target **FP16** on desktop GPUs; edge/mobile later.
- **Current Data:** 101 pairs of PNG images + hand-annotated masks, 2 categories.

**Why this architecture?** SAM 2 provides robust video memory and generalization; a small refiner specializes to your domain’s edges/textures with minimal latency overhead (~1–3 ms).

---

## Dataset Preparation
Your refiner learns to correct a **noisy prior** (SAM2 prediction) into a clean segmentation. Prepare the data so we can train on triplets: **(image, prior, GT)**.

### A) Standardize Masks & Index
- Convert masks to **single-channel 8-bit** with IDs `{0: background, 1..C: classes}` (binary → `{0,1}` if applicable).
- Create an index file (`dataset.jsonl` or CSV) with fields:
  - `image_path`, `mask_path`, `video_id?`, `frame_id?`, `classes_present`.
- If frames belong to specific surgeries, **split by video/patient** to avoid leakage.

### B) Deterministic Prompt Generation (from GT)
- For each annotated image, synthesize prompts to **simulate inference**:
  - **Positive point(s):** sample near the mask centroid.
  - **Negative points:** 1–2 just outside the boundary.
  - **Bounding box:** tight GT bbox.
- Save prompts alongside each image (e.g., `img_000123.prompts.json`).

### C) (Optional but Recommended) Precompute SAM2 Priors
- Run SAM 2 once per image with the saved prompts; write **prior logits/probabilities** per class to disk (`.npy`) or binary masks (`.png`).
- Keep the **same resolution** you’ll use for refiner training/inference.
- Result: triplets **(image, prior, GT)** accelerate refiner training and ensure reproducibility.

### D) Augmentations & Balancing
- Oversample underrepresented classes (if any).
- Apply **video-like, consistent** augmentations (even on single frames):
  - Geometric: mild resize/crop, rotation ±5°.
  - Photometric: brightness/contrast jitter, Gaussian noise.
  - Motion/defocus blur; occasional specular highlight synthesis.
- Maintain a **fixed inference resolution** (e.g., 736×1280 or 720p). Use letterboxing to preserve aspect ratio.

### E) Splits
- Preferred: **train/val/test by video/patient**. If frames are standalone, use an 80/10/10 split but avoid leaking multiple crops of the same image across splits.

### F) Targets & Losses (Refiner)
- The refiner learns `f(image ⊕ prior) → GT`.
- Use **Cross-Entropy + Dice**, optionally **Boundary loss** for crisper edges.

---

## Offline SAM2 Prior Generation
1. **Environment:** Install SAM 2 dependencies; verify GPU/AMP.
2. **Runner Script:** For each image index row:
   - Load `image`, `prompts.json` (points + box).
   - Run SAM 2 to obtain **prior logits/probabilities** (or binary mask).
   - Save prior as `.npy` (preferred) or `.png` under `sam2_prior/` mirroring dataset layout.
3. **Logging:** Log empty/failure cases; write a small summary (counts, class coverage).

**Tip:** Priors saved as logits make the refiner more powerful (it can learn calibration and boundary corrections effectively).

---

## Refiner Head Model & Training

### Architecture (tiny & fast)
- **Inputs:** `image (3ch)` concatenated with `prior (Cch)` → total `(3+C)` channels.
- **Backbone:** 3–5 **depthwise-separable** conv blocks or a **ConvNeXt-Tiny‑lite** variant.
- **Heads:**
  - Main segmentation head → `C` logits.
  - Optional boundary head (train-time) supervised by an edge map to encourage crisp borders.
- **Params:** keep small (<2–5M) so the latency overhead is ~1–3 ms on desktop GPUs.

### Loss & Optimization
- `Loss = CE + 0.5 * Dice (+ 0.2 * Boundary)` (tune weights as needed).
- **Optimizer:** AdamW; `lr=3e-4`, `weight_decay=0.05`, cosine decay; warmup 1–3k iters.
- **AMP:** use mixed precision.
- **Batching:** set effective batch 8–16 (use grad accumulation if VRAM-limited).

### Dataloader
- Loads triplets `(image, prior, GT)`; applies **consistent** spatial transforms (same resize/crop for all three).
- Normalization: standard ImageNet stats for images; scale priors to `[0,1]` if probabilities, or one-hot if masks.

### Validation
- Metrics: **Dice/IoU per class + mean**, **Boundary F-score**.
- Qualitative: export overlay PNGs to `val_vis/` for quick visual checks.

---

## Optional Temporal Consistency (Train-Only)
If you can assemble small clips around labeled frames:
- Add a **temporal EMA** consistency regularizer on logits for adjacent frames (no optical flow), or
- Use a lightweight flow (e.g., RAFT-small) to **warp** predictions `t→t+1` and penalize disagreement (train-time only).
- Keep inference unchanged; this helps flicker.

---

## End-to-End Streaming Inference
1. **t=0 Seeding:**
   - Use a simple detector or user click to generate a **box + point** prompt.
   - Run **SAM 2 → prior**; push through **refiner → mask**.
2. **t>0 Auto-prompting:**
   - From the **previous refined mask**, derive a positive point (inside mask) and a box (bbox of mask) to prompt SAM 2.
   - **SAM 2 → prior → refiner → mask**.
3. **Memory Cadence:**
   - Write to SAM 2’s memory every **K=5–8** frames (tune for quality vs speed).
   - Prune long-past memory entries to cap compute/memory.
4. **Light Post-Processing:**
   - **Temporal EMA** on logits (`α≈0.7`) to reduce flicker.
   - Optional small hole-fill / morphology for tiny artifacts.

**Throughput Knobs:** increase K (fewer writes), prune memory, reduce resolution slightly, or use frame-pruning heuristics if available.

---

## Export & Deployment
1. **Model Export:**
   - Keep **SAM 2** as **PyTorch FP16**.
   - Export **Refiner** to **ONNX** (static H×W) → **TensorRT FP16** engine for minimal latency.
2. **Runtime:**
   - A small C++/Python service: capture frame → (optional preproc) → SAM 2 (Torch) → TRT(refiner) → postproc → overlay.
3. **Calibration (optional):**
   - If using INT8 for the refiner, collect ~200 diverse frames for calibrating quantization.

---

## Latency/Accuracy Sweeps
Run a grid to find the **Pareto front**:
- Variables: **resolution**, **K (memory write interval)**, **memory size (read count)**.
- Metrics: **Dice/IoU**, **Boundary F-score**, **ms/frame** (end-to-end, including SAM 2 + refiner + postproc).
- Save results to `sweeps/` with JSON + plots.

---

## Monitoring, QA & Regression
- Maintain a **Difficult Frames Set** (smoke, occlusions, glare, fast motion) for quick regression tests.
- Track **confidence** (max softmax) and flag frames with low confidence or large diffs vs `t-1`.
- Log attention/activation snapshots periodically for debugging.

---

## Deliverables & Folder Structure
```
project/
  data/
    images/                   # your PNGs
    masks/                    # single-channel 8-bit IDs
    dataset.jsonl             # index of items
    prompts/                  # one JSON per image
  sam2_prior/                 # logits or masks produced by SAM2 (offline)
  refiner/
    models/                   # tiny refiner implementation
    train.py                  # training loop for refiner
    val_vis/                  # qualitative overlays
    checkpoints/
  runtime/
    stream_infer.py           # end-to-end streaming loop
    utils/
  export/
    export_onnx.py            # refiner → ONNX
    build_trt.py              # ONNX → TensorRT
    sweep.py                  # latency/accuracy sweeps
  docs/
    README.md                 # this file
```

---

## High-Level Checklist
- [ ] **Env ready** (PyTorch + SAM 2 deps; albumentations, opencv, timm, torchmetrics, onnx, tensorrt, etc.)
- [ ] **Masks standardized** to 8-bit single-channel with correct IDs
- [ ] **Index** (`dataset.jsonl`) built; **splits** by video/patient where possible
- [ ] **Prompts generated** (points + box) and saved per image
- [ ] **SAM2 priors precomputed** (logits/masks) to `sam2_prior/`
- [ ] **Refiner implemented** (tiny, depthwise/ConvNeXt-lite); losses and dataloader wired
- [ ] **Training** complete with AMP + logging + qualitative `val_vis/`
- [ ] **(Optional) Temporal regularization** added for clip-based batches
- [ ] **Streaming inference** pipeline with auto-prompting and memory cadence K
- [ ] **Postproc** (temporal EMA, small morphology) enabled
- [ ] **Export** refiner → ONNX → TensorRT FP16; SAM 2 kept as Torch FP16
- [ ] **Latency/accuracy sweeps** run; Pareto configuration chosen for deployment
- [ ] **Monitoring/QA** suite in place; difficult frames tracked for regression

---

### Notes
- Start with a conservative resolution (e.g., 736×1280) and **K=5**. Verify quality and latency, then explore K=8 and a slightly lower resolution if needed.
- Keep the **refiner small**; the bulk of compute stays in SAM 2. This preserves real-time throughput while still improving boundary quality.
- If you later obtain short clips around labeled frames, consider adding **temporal consistency** terms during training for improved stability at deployment.

