"""Post-processing utilities for streaming inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_ema(prev_logits: torch.Tensor | None, current_logits: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend logits with an exponential moving average to reduce flicker."""
    if prev_logits is None:
        return current_logits
    return alpha * prev_logits + (1 - alpha) * current_logits


def boundary_refine(mask: torch.Tensor) -> torch.Tensor:
    """Simple morphological smoothing using average pooling."""
    mask = mask.float().unsqueeze(1)
    smoothed = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return (smoothed > 0.5).to(mask.dtype).squeeze(1)
