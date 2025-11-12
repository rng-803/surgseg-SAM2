"""Tiny refiner head that sharpens SAM2 priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


@dataclass
class TinyRefinerConfig:
    in_channels: int
    num_classes: int
    width_mult: float = 1.0
    num_stages: int = 4
    hidden_channels: int = 64
    boundary_head: bool = True


class TinyRefiner(nn.Module):
    """Simple encoder-decoder style head for refining SAM priors."""

    def __init__(self, config: TinyRefinerConfig):
        super().__init__()
        self.config = config
        hidden = int(config.hidden_channels * config.width_mult)
        stages = []
        in_ch = config.in_channels
        for idx in range(config.num_stages):
            out_ch = hidden * (2 ** (idx // 2))
            stages.append(DepthwiseSeparableConv(in_ch, out_ch))
            in_ch = out_ch
        self.backbone = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, config.num_classes, 1),
        )
        if config.boundary_head:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(in_ch, hidden // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden // 2, 1, 1),
            )
        else:
            self.boundary_head = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.head(features)
        outputs = {"logits": logits}
        if self.boundary_head is not None:
            outputs["boundary_logits"] = self.boundary_head(features)
        return outputs


__all__ = ["TinyRefiner", "TinyRefinerConfig"]
