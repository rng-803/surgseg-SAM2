# surgseg-SAM2

## RunPod Quickstart (A5000 / A4000 / RTX 4090)

The steps below assume a RunPod template with Ubuntu + CUDA 12.x. All commands run under `/workspace`.

### 1. Clone the repo and set up Python

```bash
cd /workspace
git clone https://github.com/rng-803/surgseg-SAM2.git
cd surgseg_SAM2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Gotcha:** Always invoke scripts as modules (e.g., `python -m refiner.train`) or `export PYTHONPATH=/workspace/surgseg_SAM2`; otherwise `import refiner` fails.

### 2. Download your dataset

Use the Kaggle CLI or your preferred method to pull the dataset, then place files so they match the expected layout:

```
dataset/
  images/*.jpg
data/
  masks/*.png
  dataset.jsonl        # index (image_path/mask_path fields)
  prompts/*.prompts.json
```

Example Kaggle download (requires `kaggle` CLI and API token):

```bash
mkdir -p data dataset
kaggle datasets download -d <owner/dataset> -p /workspace/tmp_dataset --unzip
cp /workspace/tmp_dataset/images/*.jpg dataset/images/
cp /workspace/tmp_dataset/masks/*.png data/masks/
cp /workspace/tmp_dataset/dataset.jsonl data/
cp /workspace/tmp_dataset/prompts/*.json data/prompts/
```

Update paths in `data/dataset.jsonl` if your folder names differ, or pass `--project-root /workspace/surgseg_SAM2`.

### 3. Download the SAM2 checkpoint

`pip install -r requirements.txt` installs the SAM2 **code**; you still need the weights:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="facebook/sam2-hiera-tiny",
    filename="sam2_hiera_tiny.pt",
    local_dir="checkpoints/sam2",
    local_dir_use_symlinks=False,
)
print("Checkpoint saved to:", path)
PY
```

> **Gotchas:**  
> • If the download repo is gated, set `HF_TOKEN` before running the script.  
> • If you rename the checkpoint file, pass `--sam2-config configs/sam2/sam2_hiera_t.yaml` explicitly when generating priors.

### 4. Generate SAM2 priors (GPU recommended)

```bash
python -m scripts.generate_priors \
  --mode sam2 \
  --index-path data/dataset.jsonl \
  --prompts-dir data/prompts \
  --project-root . \
  --output-dir sam2_prior/logits \
  --num-classes 3 \
  --sam2-checkpoint checkpoints/sam2/sam2_hiera_tiny.pt \
  --sam2-config configs/sam2/sam2_hiera_t.yaml \
  --device cuda
```

This produces `.npy` logits mirrored under `sam2_prior/logits/...`.

### 5. Kick off training

```bash
python -m refiner.train \
  --index-path data/dataset.jsonl \
  --project-root . \
  --priors-dir sam2_prior/logits \
  --batch-size 4 \
  --epochs 30 \
  --workers 4 \
  --amp \
  --device cuda \
  --val-fraction 0.2 \
  --vis-samples 4
```

Use `--train-size 736` or similar if you want to resize to fit memory; otherwise omit to train at native resolution.

### 6. Common pitfalls


- **Missing priors:** If `sam2_prior/logits` is empty, the loader falls back to mask-derived one-hots, so confirm priors exist before training.  
- **Path mismatches:** Every entry in `data/dataset.jsonl` is resolved relative to `--project-root` (default `.`). Keep the repo root consistent or override the flag.  
- **CUDA wheel conflicts:** RunPod images sometimes ship older torch wheels; re-installing via `pip install -r requirements.txt` inside the venv ensures torch/torchvision match.
- **Mask preprocessing:** Raw PNG masks from many tools use RGB colors (e.g., 0/51/102) rather than contiguous IDs. Before training or generating priors, run the standardization script (or the full `scripts/run_preprocessing.py` pipeline) to convert them into single-channel `{0..C}` maps:

  ```bash
  python -m scripts.standardize_masks \
    --input-dir dataset/masks_raw \
    --output-dir data/masks \
    --class-map configs/class_map.json
  ```

  Adjust `class_map.json` to map each raw color/intensity to your target label IDs. The `scripts/run_preprocessing.py` helper will standardize masks, rebuild `data/dataset.jsonl`, and regenerate prompts in one go if you prefer a single command.
