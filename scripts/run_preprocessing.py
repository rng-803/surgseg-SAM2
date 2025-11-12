"""Convenience wrapper that runs the full dataset preprocessing chain."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.create_index import create_index
from scripts.generate_prompts import generate_prompts
from scripts.standardize_masks import standardize_directory

LOGGER = logging.getLogger("prep_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--raw-masks", type=Path, default=Path("dataset/masks"))
    parser.add_argument("--std-mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--prompts-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--index-path", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--class-map", type=Path, default=None, help="Optional JSON file mapping raw mask IDs to contiguous IDs")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    LOGGER.info("Standardizing masks...")
    standardize_directory(args.raw_masks, args.std_mask_dir, mapping_path=str(args.class_map) if args.class_map else None)
    LOGGER.info("Building dataset index...")
    create_index(args.raw_images, args.std_mask_dir, args.index_path, args.project_root)
    LOGGER.info("Generating prompts...")
    generate_prompts(args.std_mask_dir, args.raw_images, args.prompts_dir, args.project_root)
    LOGGER.info("Preprocessing complete")


if __name__ == "__main__":
    main()
