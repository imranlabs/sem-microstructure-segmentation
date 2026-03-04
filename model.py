"""
model.py
--------
Provides two UNet variants for semantic image segmentation:

1. UNetScratch  — lightweight UNet built from scratch (supports any number of input channels)
2. build_smp_unet — pretrained encoder UNet via segmentation_models_pytorch

No device logic or global model instantiation here.

Usage
-----
Scratch model:
    from model import UNetScratch
    model = UNetScratch(in_channels=1, num_classes=5).to(device)

Pretrained model:
    from model import build_smp_unet, PretrainedUNetConfig
    cfg = PretrainedUNetConfig(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, num_classes=5)
    model = build_smp_unet(cfg).to(device)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # only required when using pretrained models


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive (Conv2d → BN → ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d(2) followed by DoubleConv — halves spatial dims."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d upsample → concat skip connection → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # in_ch is the number of channels BEFORE upsampling
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Center-crop skip if spatial sizes differ (e.g. odd input dimensions)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = self._center_crop(skip, x.shape[-2], x.shape[-1])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

    @staticmethod
    def _center_crop(t: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, _, h, w = t.shape
        top  = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return t[:, :, top:top + target_h, left:left + target_w]


# ---------------------------------------------------------------------------
# Scratch UNet
# ---------------------------------------------------------------------------

class UNetScratch(nn.Module):
    """
    5-level UNet trained from scratch.

    Args:
        in_channels: number of input image channels (e.g. 1 for grayscale, 3 for RGB).
        num_classes:  number of segmentation classes.
        base:         base feature-map width (doubles at each encoder level).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5, base: int = 32) -> None:
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, base)
        self.down1 = Down(base,      base * 2)
        self.down2 = Down(base * 2,  base * 4)
        self.down3 = Down(base * 4,  base * 8)
        self.down4 = Down(base * 8,  base * 16)  # bottleneck

        # Decoder
        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8,  base * 4)
        self.up3 = Up(base * 4,  base * 2)
        self.up4 = Up(base * 2,  base)

        # Output projection
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        return self.outc(x)


# ---------------------------------------------------------------------------
# Pretrained UNet via segmentation_models_pytorch
# ---------------------------------------------------------------------------

@dataclass
class PretrainedUNetConfig:
    """Configuration for a pretrained SMP UNet."""
    encoder_name:    str           = "resnet34"
    encoder_weights: Optional[str] = "imagenet"  # set to None to use random weights
    in_channels:     int           = 1
    num_classes:     int           = 5


def build_smp_unet(cfg: PretrainedUNetConfig) -> nn.Module:
    """
    Build a UNet with a pretrained encoder from segmentation_models_pytorch.

    Args:
        cfg: PretrainedUNetConfig instance.

    Returns:
        nn.Module — raw logits (no activation applied).
    """
    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is not installed. "
            "Run: pip install segmentation-models-pytorch"
        )

    return smp.Unet(
        encoder_name    = cfg.encoder_name,
        encoder_weights = cfg.encoder_weights,
        in_channels     = cfg.in_channels,
        classes         = cfg.num_classes,
        activation      = None,  # raw logits; loss_fn handles softmax internally
    )
