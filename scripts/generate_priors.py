"""Offline SAM2 prior generation script.

This runner iterates over the dataset index, loads the corresponding prompts,
and writes prior logits/probabilities to `sam2_prior/` so refiner training can
consume triplets (image, prior, mask).

Two modes are supported:
1. `mock` (default) — generate priors directly from the ground-truth mask. This
   is useful for smoke-testing the pipeline before wiring up SAM2.
2. `sam2` — dynamically import a user-provided SAM2 helper module that exposes
   `build_predictor(checkpoint: Path, device: str)` and
   `predict(predictor, image: np.ndarray, prompts: dict) -> np.ndarray`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

LOGGER = logging.getLogger("sam2_prior")


def _load_index(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_prompts(prompts_dir: Path, stem: str) -> Dict[str, Any]:
    prompt_path = prompts_dir / f"{stem}.prompts.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file {prompt_path} is missing")
    return json.loads(prompt_path.read_text())


def _load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _mask_to_prior(mask: np.ndarray, num_classes: int, blur_sigma: float) -> np.ndarray:
    h, w = mask.shape
    logits = np.zeros((num_classes, h, w), dtype=np.float32)
    for cls in range(num_classes):
        cls_mask = (mask == cls).astype(np.float32)
        if cls_mask.sum() == 0:
            continue
        if blur_sigma > 0:
            cls_mask = gaussian_filter(cls_mask, sigma=blur_sigma)
        logits[cls] = cls_mask
    denom = np.sum(logits, axis=0, keepdims=True) + 1e-6
    probs = logits / denom
    return probs.astype(np.float32, copy=False)


class Sam2ModuleRunner:
    """Bridge to a user-provided SAM2 helper module."""

    def __init__(self, module_path: str, checkpoint: Path, device: str):
        module = importlib.import_module(module_path)
        if not hasattr(module, "build_predictor") or not hasattr(module, "predict"):
            raise AttributeError(
                f"{module_path} must define `build_predictor` and `predict` functions"
            )
        self.module = module
        self.predictor = module.build_predictor(checkpoint=checkpoint, device=device)

    def __call__(self, image: np.ndarray, prompts: Dict[str, Any]) -> np.ndarray:
        logits = self.module.predict(self.predictor, image=image, prompts=prompts)
        return np.asarray(logits, dtype=np.float32)


def generate_priors(args: argparse.Namespace) -> None:
    project_root = args.project_root
    records = list(_load_index(args.index_path))
    LOGGER.info("Loaded %d records from %s", len(records), args.index_path)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    runner = None
    if args.mode == "sam2":
        if not args.sam2_module or not args.sam2_checkpoint:
            raise ValueError("`--sam2-module` and `--sam2-checkpoint` are required for mode=sam2")
        runner = Sam2ModuleRunner(args.sam2_module, args.sam2_checkpoint, args.device)

    written = 0
    skipped = 0
    for record in records:
        image_rel = Path(record["image_path"])
        stem = image_rel.stem
        image_path = project_root / image_rel
        mask_path = project_root / record["mask_path"]
        prompt_data = _load_prompts(args.prompts_dir, stem)
        image = _load_image(image_path)
        if args.mode == "mock":
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            prior = _mask_to_prior(mask, args.num_classes, args.blur_sigma)
        else:
            prior = runner(image, prompt_data)
        if prior.shape[0] != args.num_classes:
            LOGGER.warning(
                "Prior class dimension %s != num_classes (%d) for %s",
                prior.shape,
                args.num_classes,
                stem,
            )
        dest = output_root / image_rel.with_suffix(".npy")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, prior)
        written += 1
    LOGGER.info("Generated %d priors (skipped %d)", written, skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("sam2_prior/logits"))
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--mode", choices=["mock", "sam2"], default="mock")
    parser.add_argument("--sam2-module", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--blur-sigma", type=float, default=1.5, help="Gaussian smoothing for mock priors")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    generate_priors(args)


if __name__ == "__main__":
    main()
