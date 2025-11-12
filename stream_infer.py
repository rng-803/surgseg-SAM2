"""End-to-end streaming inference loop skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from refiner.models import TinyRefiner, TinyRefinerConfig
from runtime.utils.postprocess import temporal_ema


class Sam2Interface:
    """Placeholder for the actual SAM2 model runner."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        # TODO: load the real SAM2 weights here.

    def run(self, image: np.ndarray, prompts: dict) -> torch.Tensor:
        raise NotImplementedError("Hook up the official SAM2 inference code here")


class StreamInferenceService:
    def __init__(
        self,
        sam2_ckpt: Path,
        refiner_ckpt: Path,
        num_classes: int,
        device: torch.device,
        ema_alpha: float = 0.7,
    ):
        self.device = device
        self.sam2 = Sam2Interface(sam2_ckpt)
        self.refiner = TinyRefiner(
            TinyRefinerConfig(in_channels=3 + num_classes, num_classes=num_classes)
        )
        state = torch.load(refiner_ckpt, map_location="cpu")
        self.refiner.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        self.refiner.to(device).eval()
        self.prev_logits: Optional[torch.Tensor] = None
        self.ema_alpha = ema_alpha

    def _prepare_inputs(self, image: np.ndarray, prior: torch.Tensor) -> torch.Tensor:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        prior = prior.to(self.device).unsqueeze(0)
        return torch.cat([tensor, prior], dim=1)

    def _prompts_from_mask(self, mask: np.ndarray) -> dict:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return {"positive_points": [], "negative_points": [], "bbox": None}
        centroid = [float(xs.mean()), float(ys.mean())]
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return {"positive_points": [centroid], "negative_points": [], "bbox": bbox}

    def step(self, image: np.ndarray, prev_mask: Optional[np.ndarray] = None) -> np.ndarray:
        prompts = self._prompts_from_mask(prev_mask) if prev_mask is not None else {}
        prior = self.sam2.run(image, prompts)
        inputs = self._prepare_inputs(image, prior)
        with torch.no_grad():
            logits = self.refiner(inputs)["logits"]
            logits = temporal_ema(self.prev_logits, logits, self.ema_alpha)
            self.prev_logits = logits
            prediction = torch.argmax(logits, dim=1)
        return prediction.squeeze(0).cpu().numpy().astype(np.uint8)


def run_stream(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    service = StreamInferenceService(args.sam2_ckpt, args.refiner_ckpt, args.num_classes, device, args.ema_alpha)
    image = np.array(Image.open(args.frame_path).convert("RGB"))
    mask = service.step(image)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(out_path)
    print(f"Saved mask to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-ckpt", type=Path, required=True)
    parser.add_argument("--refiner-ckpt", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runtime/output.png"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stream(args)


if __name__ == "__main__":
    main()
