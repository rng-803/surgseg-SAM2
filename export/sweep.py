"""Latency/accuracy sweep harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def run_sweep(args: argparse.Namespace) -> None:
    # Placeholder: this function should launch inference at different resolutions / K intervals
    # and write metrics to disk. We keep it simple for now so the scaffolding exists.
    configs: List[Dict] = []
    for resolution in args.resolutions:
        for mem_interval in args.memory_intervals:
            configs.append(
                {
                    "resolution": resolution,
                    "memory_interval": mem_interval,
                    "dice": None,
                    "latency_ms": None,
                }
            )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(configs, indent=2))
    print(f"Sweep template saved to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", default=["720p", "960p"], help="List of resolutions to evaluate")
    parser.add_argument("--memory-intervals", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--output", type=Path, default=Path("export/sweeps/placeholder.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
