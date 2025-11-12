"""Export the trained refiner to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from refiner.models import TinyRefiner, TinyRefinerConfig


def export_onnx(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRefiner(
        TinyRefinerConfig(
            in_channels=args.in_channels,
            num_classes=args.num_classes,
            width_mult=args.width_mult,
            num_stages=args.num_stages,
            hidden_channels=args.hidden_channels,
            boundary_head=False,
        )
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.to(device).eval()
    dummy = torch.randn(1, args.in_channels, args.height, args.width, device=device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["inputs"],
        output_names=["logits", "boundary_logits"] if model.boundary_head is not None else ["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"inputs": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Saved ONNX model to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("export/refiner.onnx"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--in-channels", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()
