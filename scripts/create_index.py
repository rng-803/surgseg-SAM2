"""Build a dataset index (JSONL) with paths, metadata, and class coverage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("dataset_index")


def _load_mask(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8, copy=False)


def _derive_video_frame(stem: str) -> tuple[str, str]:
    if "__" in stem:
        video_id, frame_id = stem.split("__", maxsplit=1)
    else:
        video_id, frame_id = stem, "0"
    return video_id, frame_id


def _build_record(image_path: Path, mask_path: Path, root: Path) -> Dict:
    mask = _load_mask(mask_path)
    classes = np.unique(mask).astype(int).tolist()
    video_id, frame_id = _derive_video_frame(image_path.stem)
    return {
        "image_path": str(image_path.relative_to(root)),
        "mask_path": str(mask_path.relative_to(root)),
        "video_id": video_id,
        "frame_id": frame_id,
        "classes_present": classes,
    }


def create_index(images_dir: Path, masks_dir: Path, output_path: Path, project_root: Path) -> List[Dict]:
    image_paths = sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")
    records: List[Dict] = []
    missing_masks = 0
    for image_path in image_paths:
        mask_candidate = masks_dir / image_path.relative_to(images_dir)
        mask_candidate = mask_candidate.with_suffix(".png")
        if not mask_candidate.exists():
            missing_masks += 1
            LOGGER.warning("Missing mask for %s", image_path)
            continue
        record = _build_record(image_path, mask_candidate, project_root)
        records.append(record)
    LOGGER.info("Indexed %d items (%d missing masks)", len(records), missing_masks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")
    LOGGER.info("Wrote %s", output_path)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/images"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    create_index(args.images_dir, args.masks_dir, args.output, args.project_root)


if __name__ == "__main__":
    main()
