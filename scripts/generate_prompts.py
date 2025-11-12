"""Generate deterministic SAM-style prompts from ground-truth masks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

LOGGER = logging.getLogger("prompt_gen")


def _load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8, copy=False)


def _bbox(coords: np.ndarray) -> List[int]:
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # Format [x0, y0, x1, y1]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def _sample_negative(mask: np.ndarray, cls_mask: np.ndarray, rng: np.random.Generator, count: int) -> List[List[int]]:
    border = binary_dilation(cls_mask, iterations=2) & (~cls_mask)
    coords = np.column_stack(np.nonzero(border))
    if coords.size == 0:
        background = np.column_stack(np.nonzero(mask == 0))
        coords = background
    if coords.size == 0:
        return []
    idx = rng.choice(len(coords), size=min(count, len(coords)), replace=False)
    sampled = coords[idx]
    return [[int(x), int(y)] for y, x in sampled]


def _sample_positive(coords: np.ndarray, rng: np.random.Generator, count: int) -> List[List[int]]:
    if len(coords) == 0:
        return []
    if len(coords) < count:
        idx = np.arange(len(coords))
    else:
        idx = rng.choice(len(coords), size=count, replace=False)
    sampled = coords[idx]
    return [[int(x), int(y)] for y, x in sampled]


def _build_prompt(mask: np.ndarray, class_id: int, rng: np.random.Generator, pos_count: int, neg_count: int) -> Dict:
    cls_mask = mask == class_id
    coords = np.column_stack(np.nonzero(cls_mask))
    if coords.size == 0:
        return {}
    bbox = _bbox(coords)
    positives = _sample_positive(coords, rng, pos_count)
    negatives = _sample_negative(mask, cls_mask, rng, neg_count)
    return {
        "class_id": int(class_id),
        "positive_points": positives,
        "negative_points": negatives,
        "bbox": bbox,
    }


def generate_prompts(masks_dir: Path, images_dir: Path, output_dir: Path, project_root: Path, pos_points: int = 1, neg_points: int = 2) -> None:
    mask_paths = sorted(masks_dir.rglob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found in {masks_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in mask_paths:
        mask = _load_mask(mask_path)
        rng = np.random.default_rng(abs(hash(mask_path.stem)) % (2**32))
        image_candidate = images_dir / mask_path.relative_to(masks_dir)
        image_candidate = image_candidate.with_suffix(".jpg")
        record = {
            "image_path": str(image_candidate.relative_to(project_root)),
            "mask_path": str(mask_path.relative_to(project_root)),
            "prompts": [],
        }
        for class_id in sorted(np.unique(mask)):
            if class_id == 0:
                continue
            prompt = _build_prompt(mask, int(class_id), rng, pos_points, neg_points)
            if prompt:
                record["prompts"].append(prompt)
        if not record["prompts"]:
            LOGGER.warning("No foreground classes found in %s", mask_path)
            continue
        output_path = output_dir / f"{mask_path.stem}.prompts.json"
        output_path.write_text(json.dumps(record, indent=2))
        LOGGER.debug("Wrote prompts for %s", mask_path)
    LOGGER.info("Prompts saved to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--positive-points", type=int, default=1)
    parser.add_argument("--negative-points", type=int, default=2)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    generate_prompts(
        args.masks_dir,
        args.images_dir,
        args.output_dir,
        args.project_root,
        args.positive_points,
        args.negative_points,
    )


if __name__ == "__main__":
    main()
