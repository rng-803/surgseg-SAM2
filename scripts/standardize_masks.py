"""Standardize raw mask assets into single-channel 8-bit IDs.

The script converts RGB masks into grayscale label maps, optionally applying a
user-provided class mapping, and writes the results into `data/masks` (or a
custom destination). The output directory mirrors the relative layout of the
input directory to keep downstream indexing simple.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("mask_standardizer")


def _load_class_map(path: str | None) -> Dict[int, int] | None:
    if path is None:
        return None
    mapping = json.loads(Path(path).read_text())
    sanitized: Dict[int, int] = {}
    for key, value in mapping.items():
        sanitized[int(key)] = int(value)
    return sanitized


def _apply_class_map(arr: np.ndarray, mapping: Dict[int, int] | None) -> np.ndarray:
    if mapping is None:
        return arr
    remapped = arr.copy()
    for src_id, dst_id in mapping.items():
        remapped[arr == src_id] = np.uint8(dst_id)
    return remapped.astype(np.uint8, copy=False)


def _standardize_mask(src: Path, dst: Path, mapping: Dict[int, int] | None) -> Iterable[int]:
    image = Image.open(src)
    arr = np.array(image)
    if arr.ndim == 3:
        # Assume RGB mask; take the first channel which carries the ID.
        arr = arr[..., 0]
    arr = arr.astype(np.uint8, copy=False)
    arr = _apply_class_map(arr, mapping)
    unique_vals = np.unique(arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(dst)
    return unique_vals.tolist()


def standardize_directory(input_dir: Path, output_dir: Path, mapping_path: str | None = None, suffix: str = ".png") -> None:
    mapping = _load_class_map(mapping_path)
    mask_paths = sorted(input_dir.rglob(f"*{suffix}"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks with suffix {suffix} found under {input_dir}")
    LOGGER.info("Processing %d mask files", len(mask_paths))
    for mask_path in mask_paths:
        rel_path = mask_path.relative_to(input_dir)
        dst_path = output_dir / rel_path
        unique_vals = _standardize_mask(mask_path, dst_path, mapping)
        LOGGER.debug("%s -> %s | classes=%s", mask_path, dst_path, unique_vals)
    LOGGER.info("Masks written to %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/masks"), help="Path to raw mask directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/masks"), help="Destination directory for standardized masks")
    parser.add_argument("--class-map", type=str, default=None, help="Optional JSON file mapping raw IDs to contiguous IDs")
    parser.add_argument("--suffix", type=str, default=".png", help="Mask file suffix to look for")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    standardize_directory(args.input_dir, args.output_dir, args.class_map, args.suffix)


if __name__ == "__main__":
    main()
