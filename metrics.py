"""
metrics.py
Segmentation metrics for SEM multi-class masks.

- Hard IoU / Dice from argmax predictions
- Confusion matrix (counts + row-normalized percent)
- Boundary F1 (per class) with tolerance radius (no scipy)

All functions are torch-first and safe on CPU/MPS/CUDA.
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def preds_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits: (B,C,H,W) -> preds: (B,H,W) int64"""
    return torch.argmax(logits, dim=1)


@torch.no_grad()
def iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """preds/targets: (B,H,W) -> (C,)"""
    out = []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        out.append((inter + eps) / (union + eps))
    return torch.stack(out)


@torch.no_grad()
def dice_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """preds/targets: (B,H,W) -> (C,)"""
    out = []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        out.append((2.0 * inter + eps) / (denom + eps))
    return torch.stack(out)


@torch.no_grad()
def hard_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor | float]:
    """
    Returns:
      mean_iou, mean_dice (float)
      iou_per_class, dice_per_class (cpu tensors)
      preds (B,H,W) on same device as logits (handy for confusion)
    """
    preds = preds_from_logits(logits)
    iou = iou_per_class(preds, targets, num_classes, eps=eps)
    dice = dice_per_class(preds, targets, num_classes, eps=eps)
    return {
        "preds": preds,
        "mean_iou": float(iou.mean().item()),
        "mean_dice": float(dice.mean().item()),
        "iou_per_class": iou.detach().cpu(),
        "dice_per_class": dice.detach().cpu(),
    }


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    preds/targets: (B,H,W) int64
    Returns: (C,C) counts, rows=true, cols=pred
    """
    p = preds.reshape(-1)
    t = targets.reshape(-1)
    k = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    t = t[k]
    p = p[k]
    cm = torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def confusion_percent(cm: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize cm to percent."""
    cm = cm.float()
    row_sum = cm.sum(dim=1, keepdim=True).clamp_min(eps)
    return (cm / row_sum) * 100.0


# -----------------------
# Boundary F1 (per class)
# -----------------------

def _dilate_bool(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """mask: (B,1,H,W) bool -> dilated bool"""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    x = mask.float()
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)
    return x > 0.5


def _mask_to_boundary(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B,1,H,W) bool
    boundary = mask - erode(mask)
    erosion implemented via maxpool on inverted mask
    """
    x = mask.float()
    inv = 1.0 - x
    eroded_inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - eroded_inv
    boundary = (x - eroded) > 0.5
    return boundary


@torch.no_grad()
def boundary_f1_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    radius: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    preds/targets: (B,H,W) int64
    Returns: (C,) boundary F1 per class (CPU)
    """
    f1s = []
    for c in range(num_classes):
        p = (preds == c).unsqueeze(1)  # (B,1,H,W)
        t = (targets == c).unsqueeze(1)

        pb = _mask_to_boundary(p)
        tb = _mask_to_boundary(t)

        tb_d = _dilate_bool(tb, radius)
        pb_d = _dilate_bool(pb, radius)

        tp_p = (pb & tb_d).sum().float()
        tp_t = (tb & pb_d).sum().float()

        prec = tp_p / (pb.sum().float() + eps)
        rec = tp_t / (tb.sum().float() + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)
        f1s.append(f1)

    return torch.stack(f1s).cpu()
