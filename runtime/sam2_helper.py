"""Utility helpers to run SAM 2 for offline prior generation and runtime inference.

The script-based pipeline expects this module to expose two functions:

- ``build_predictor``: loads a SAM 2 image predictor (either from a local checkpoint
  or from the Hugging Face hub).
- ``predict``: consumes a numpy image + prompt dictionary and returns
  ``(C, H, W)`` priors suitable for training the tiny refiner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sam2.build_sam import build_sam2, build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as exc:  # pragma: no cover - import guard for environments without SAM 2
    raise ImportError(
        "SAM 2 is not installed. Install it via "
        "'pip install git+https://github.com/facebookresearch/segment-anything-2'."
    ) from exc

LOGGER = logging.getLogger("sam2_helper")

_CHECKPOINT_HINTS: Tuple[Tuple[str, str], ...] = (
    ("sam2_hiera_tiny", "configs/sam2/sam2_hiera_t.yaml"),
    ("sam2_hiera_small", "configs/sam2/sam2_hiera_s.yaml"),
    ("sam2_hiera_base_plus", "configs/sam2/sam2_hiera_b+.yaml"),
    ("sam2_hiera_large", "configs/sam2/sam2_hiera_l.yaml"),
    ("sam2.1_hiera_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ("sam2.1_hiera_small", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    ("sam2.1_hiera_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    ("sam2.1_hiera_large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
)


def _infer_config_from_checkpoint(checkpoint: Path | str) -> str:
    """Best-effort mapping from checkpoint names to Hydra config paths."""
    name = Path(checkpoint).name.lower()
    for key, cfg in _CHECKPOINT_HINTS:
        if key in name:
            LOGGER.info("Inferred SAM2 config '%s' from checkpoint name '%s'", cfg, name)
            return cfg
    raise ValueError(
        f"Unable to infer SAM2 config from checkpoint name '{name}'. "
        "Pass --sam2-config explicitly when invoking scripts.generate_priors."
    )


def build_predictor(
    checkpoint: Path | str | None,
    *,
    device: str = "cuda",
    config_path: Optional[str] = None,
    model_id: Optional[str] = None,
    apply_postprocessing: bool = True,
    mask_threshold: float = 0.0,
    max_hole_area: float = 0.0,
    max_sprinkle_area: float = 0.0,
) -> SAM2ImagePredictor:
    """Construct a SAM 2 image predictor."""
    predictor_kwargs = dict(
        mask_threshold=mask_threshold,
        max_hole_area=max_hole_area,
        max_sprinkle_area=max_sprinkle_area,
    )
    if model_id:
        LOGGER.info("Loading SAM2 from Hugging Face model_id='%s'", model_id)
        predictor = SAM2ImagePredictor.from_pretrained(
            model_id,
            device=device,
            mode="eval",
            apply_postprocessing=apply_postprocessing,
            **predictor_kwargs,
        )
        return predictor
    if checkpoint is None:
        raise ValueError("checkpoint is required when `model_id` is not provided")
    config_to_use = config_path or _infer_config_from_checkpoint(checkpoint)
    LOGGER.info("Loading SAM2 from checkpoint=%s with config=%s", checkpoint, config_to_use)
    sam_model = build_sam2(
        config_file=config_to_use,
        ckpt_path=str(checkpoint),
        device=device,
        mode="eval",
        apply_postprocessing=apply_postprocessing,
    )
    return SAM2ImagePredictor(sam_model, **predictor_kwargs)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _stack_points(positive: Sequence[Sequence[float]], negative: Sequence[Sequence[float]]):
    coords: List[Sequence[float]] = []
    labels: List[int] = []
    if positive:
        coords.extend(positive)
        labels.extend([1] * len(positive))
    if negative:
        coords.extend(negative)
        labels.extend([0] * len(negative))
    if not coords:
        return None, None
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def predict(
    predictor: SAM2ImagePredictor,
    *,
    image: np.ndarray,
    prompts: Dict,
    num_classes: int,
    multimask_output: bool = False,
) -> np.ndarray:
    """Generate per-class logits/probabilities from SAM 2 given prompts."""
    if image.ndim != 3:
        raise ValueError("image must be an HxWx3 array")
    predictor.set_image(image)
    h, w, _ = image.shape
    priors = np.zeros((num_classes, h, w), dtype=np.float32)
    prompt_list: Iterable[Dict] = prompts.get("prompts", [])
    if isinstance(prompt_list, dict):
        prompt_list = prompt_list.values()
    for prompt in prompt_list:
        cls = int(prompt.get("class_id", 1))
        if cls >= num_classes:
            LOGGER.warning("Skipping class_id=%d (> num_classes=%d)", cls, num_classes)
            continue
        pos = prompt.get("positive_points", []) or []
        neg = prompt.get("negative_points", []) or []
        coords, labels = _stack_points(pos, neg)
        box = np.asarray(prompt["bbox"], dtype=np.float32) if prompt.get("bbox") else None
        if coords is None and box is None:
            LOGGER.warning("Prompt for class %d has neither points nor bbox, skipping", cls)
            continue
        masks, _, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            multimask_output=multimask_output,
            return_logits=True,
            normalize_coords=False,
        )
        logits = masks[0] if masks.ndim == 3 else masks
        prob = _sigmoid(logits).astype(np.float32, copy=False)
        priors[cls] = np.maximum(priors[cls], prob)
    # background channel
    fg = priors[1:].sum(axis=0, keepdims=False)
    priors[0] = np.clip(1.0 - fg, 0.0, 1.0)
    denom = priors.sum(axis=0, keepdims=True)
    priors /= np.clip(denom, 1e-6, None)
    return priors


__all__ = ["build_predictor", "predict"]
