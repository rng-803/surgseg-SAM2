"""End-to-end streaming inference loop built on SAM2 + Tiny Refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from refiner.models import TinyRefiner, TinyRefinerConfig
from runtime.sam2_helper import build_predictor, predict
from runtime.utils.postprocess import temporal_ema

PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 176, 246],
        [234, 92, 92],
        [120, 195, 20],
        [255, 214, 10],
        [196, 147, 214],
    ],
    dtype=np.uint8,
)


class Sam2Interface:
    """Thin wrapper around runtime.sam2_helper for streaming usage."""

    def __init__(
        self,
        checkpoint: Path | None,
        num_classes: int,
        device: str,
        config_path: Optional[str],
        model_id: Optional[str],
        apply_postprocessing: bool,
    ):
        self.predictor = build_predictor(
            checkpoint=checkpoint,
            device=device,
            config_path=config_path,
            model_id=model_id,
            apply_postprocessing=apply_postprocessing,
        )
        self.num_classes = num_classes

    def run(self, image: np.ndarray, prompts: dict) -> torch.Tensor:
        priors = predict(
            self.predictor,
            image=image,
            prompts=prompts,
            num_classes=self.num_classes,
            multimask_output=False,
        )
        return torch.from_numpy(priors)


class StreamInferenceService:
    def __init__(
        self,
        sam2_ckpt: Path,
        refiner_ckpt: Path,
        num_classes: int,
        device: torch.device,
        ema_alpha: float = 0.7,
        sam2_config: Optional[str] = None,
        sam2_model_id: Optional[str] = None,
        disable_sam2_postprocess: bool = False,
    ):
        self.device = device
        self.num_classes = num_classes
        self.sam2 = Sam2Interface(
            sam2_ckpt,
            num_classes,
            str(device),
            sam2_config,
            sam2_model_id,
            apply_postprocessing=not disable_sam2_postprocess,
        )
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
        prompts: List[Dict] = []
        for cls in range(1, self.num_classes):
            coords = np.column_stack(np.nonzero(mask == cls))
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0)[::-1]  # (x, y)
            bbox = [
                int(coords[:, 1].min()),
                int(coords[:, 0].min()),
                int(coords[:, 1].max()),
                int(coords[:, 0].max()),
            ]
            prompts.append(
                {
                    "class_id": cls,
                    "positive_points": [[float(centroid[0]), float(centroid[1])]],
                    "negative_points": [],
                    "bbox": bbox,
                }
            )
        return {"prompts": prompts}

    def step(
        self,
        image: np.ndarray,
        prompt_payload: Optional[dict] = None,
        prev_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if prompt_payload is None:
            if prev_mask is None:
                raise ValueError("Initial prompts are required for the first frame.")
            prompt_payload = self._prompts_from_mask(prev_mask)
        prior = self.sam2.run(image, prompt_payload)
        inputs = self._prepare_inputs(image, prior)
        with torch.no_grad():
            logits = self.refiner(inputs)["logits"]
            logits = temporal_ema(self.prev_logits, logits, self.ema_alpha)
            self.prev_logits = logits
            prediction = torch.argmax(logits, dim=1)
        return prediction.squeeze(0).cpu().numpy().astype(np.uint8)


def run_stream(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    service = StreamInferenceService(
        args.sam2_ckpt,
        args.refiner_ckpt,
        args.num_classes,
        device,
        args.ema_alpha,
        args.sam2_config,
        args.sam2_model_id,
        args.disable_sam2_postprocess,
    )
    init_payload = json.loads(Path(args.init_prompts).read_text())
    if isinstance(init_payload, dict) and "prompts" in init_payload:
        init_prompts = {"prompts": init_payload["prompts"]}
    elif isinstance(init_payload, list):
        init_prompts = {"prompts": init_payload}
    else:
        raise ValueError("init-prompts must be a JSON file containing either a 'prompts' list or a list of prompt dicts.")

    if not args.frame_path and not args.video_path:
        raise ValueError("Provide either --frame-path or --video-path")

    if args.video_path:
        run_video_inference(args, service, init_prompts)
    else:
        image = np.array(Image.open(args.frame_path).convert("RGB"))
        mask = service.step(image, prompt_payload=init_prompts)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(out_path)
        print(f"Saved mask to {out_path}")


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    max_label = int(mask.max())
    if max_label < PALETTE.shape[0]:
        palette = PALETTE
    else:
        reps = int(np.ceil((max_label + 1) / PALETTE.shape[0]))
        palette = np.tile(PALETTE, (reps, 1))
    idx = np.clip(mask, 0, palette.shape[0] - 1)
    return palette[idx]


def run_video_inference(args: argparse.Namespace, service: StreamInferenceService, init_prompts: dict) -> None:
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {args.video_path}")
    fps = args.output_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    frame_idx = 0
    prev_mask: Optional[np.ndarray] = None
    prompt_payload = init_prompts
    png_dir = Path(args.output_dir) if args.output_dir else None
    if png_dir:
        png_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = service.step(frame_rgb, prompt_payload=prompt_payload, prev_mask=prev_mask)
        prev_mask = mask
        prompt_payload = None  # subsequent frames derive prompts from mask
        colored = _mask_to_rgb(mask)
        writer.write(cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
        if png_dir:
            Image.fromarray(mask).save(png_dir / f"frame_{frame_idx:06d}.png")
        frame_idx += 1
    writer.release()
    cap.release()
    print(f"Wrote {frame_idx} masks to {out_path}")
    if png_dir:
        print(f"Per-frame masks saved under {png_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2-ckpt", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, default=None)
    parser.add_argument("--sam2-model-id", type=str, default=None)
    parser.add_argument("--disable-sam2-postprocess", action="store_true")
    parser.add_argument("--refiner-ckpt", type=Path, required=True)
    parser.add_argument("--frame-path", type=Path, help="Single frame path for inference")
    parser.add_argument("--video-path", type=Path, help="Optional video for streaming inference")
    parser.add_argument("--output", type=Path, default=Path("runtime/output.png"))
    parser.add_argument("--output-video", type=Path, default=Path("runtime/output.mp4"))
    parser.add_argument("--output-dir", type=Path, help="Optional directory to dump per-frame masks")
    parser.add_argument("--output-fps", type=float, default=None)
    parser.add_argument("--init-prompts", type=Path, required=True, help="JSON prompts for the first frame")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stream(args)


if __name__ == "__main__":
    main()
