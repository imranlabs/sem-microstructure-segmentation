"""
evaluate.py
-----------
Evaluation and testing helpers for semantic segmentation.

Compatible with the existing train.py loss_fn (returns a scalar, not a tuple).
Loads checkpoints saved by train.train_model() with keys:
    "model_state", "optimizer_state", "epoch", "best_score", "monitor"

Key design decisions:
- IoU / Dice computed from the global confusion matrix (not batch-averaged)
  to avoid bias from unequal batch sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from metrics import confusion_matrix, confusion_percent, boundary_f1_per_class


# ---------------------------------------------------------------------------
# Module-level loss components (mirrors train.py)
# ---------------------------------------------------------------------------

_dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)


def _compute_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    device: torch.device,
) -> float:
    weight = class_weight.clone().detach().to(dtype=torch.float32, device=device)
    return float(nn.CrossEntropyLoss(weight=weight)(logits, targets).detach())


def _compute_dice(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(_dice_loss(logits, targets).detach())


# ---------------------------------------------------------------------------
# Helpers — metrics from confusion matrix
# ---------------------------------------------------------------------------

def _iou_from_cm(cm: torch.Tensor, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    cm = cm.float()
    inter  = cm.diag()
    union  = cm.sum(1) + cm.sum(0) - inter
    iou_pc = ((inter + eps) / (union + eps)).numpy()
    return iou_pc, float(iou_pc.mean())


def _dice_from_cm(cm: torch.Tensor, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    cm = cm.float()
    inter   = cm.diag()
    denom   = cm.sum(1) + cm.sum(0)
    dice_pc = ((2 * inter + eps) / (denom + eps)).numpy()
    return dice_pc, float(dice_pc.mean())


# ---------------------------------------------------------------------------
# Core evaluate — works with scalar loss_fn from train.py
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn,
    class_weight: torch.Tensor,
    num_classes: int,
    boundary_radius: Optional[int] = None,
) -> Dict:
    """
    Evaluate model on a dataloader using the existing scalar loss_fn from train.py.

    Args:
        model:           nn.Module.
        loader:          DataLoader yielding (imgs, masks, *).
        device:          inference device.
        loss_fn:         train.loss_fn — returns a scalar tensor.
        class_weight:    1-D per-class weight tensor.
        num_classes:     number of segmentation classes.
        boundary_radius: if set, compute boundary F1 with this dilation radius.

    Returns:
        Dict with keys:
            loss, ce, dice,
            mean_iou, mean_dice,
            iou_per_class, dice_per_class   (np.ndarray[C])
            cm                              (torch.Tensor[C,C] counts)
            cm_percent                      (np.ndarray[C,C] row-normalised %)
            bf1_per_class, mean_bf1         (only if boundary_radius is set)
    """
    model.eval()

    total_loss = 0.0
    total_ce   = 0.0
    total_dice = 0.0
    n_batches  = 0

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    bf1_acc = [] if boundary_radius is not None else None

    for batch in loader:
        imgs , masks = batch[0].to(device), batch[1].to(device)
        #imgs  = imgs.to(device, non_blocking=True)
        #masks = masks.to(device, non_blocking=True)

        logits = model(imgs)

        # scalar loss — compatible with existing train.py loss_fn
        loss = loss_fn(logits, masks, class_weight=class_weight, device=device)

        # CE and Dice logged separately for diagnostics
        total_ce   += _compute_ce(logits, masks, class_weight, device)
        total_dice += _compute_dice(logits, masks)

        preds = torch.argmax(logits, dim=1)
        cm += confusion_matrix(preds.cpu(), masks.cpu(), num_classes=num_classes)

        total_loss += float(loss.detach())
        n_batches  += 1

        if boundary_radius is not None:
            bf1 = boundary_f1_per_class(preds.cpu(), masks.cpu(), num_classes=num_classes, radius=boundary_radius)
            bf1_acc.append(bf1.numpy())

    # IoU / Dice from global CM — unbiased across batch sizes
    iou_pc,  mean_iou  = _iou_from_cm(cm)
    dice_pc, mean_dice = _dice_from_cm(cm)

    out = {
        "loss":           total_loss / max(n_batches, 1),
        "ce":             total_ce   / max(n_batches, 1),
        "dice":           total_dice / max(n_batches, 1),
        "mean_iou":       mean_iou,
        "mean_dice":      mean_dice,
        "iou_per_class":  iou_pc,
        "dice_per_class": dice_pc,
        "cm":             cm,
        "cm_percent":     confusion_percent(cm).numpy(),
    }

    if boundary_radius is not None and bf1_acc:
        bf1_pc = np.mean(np.stack(bf1_acc), axis=0)
        out["bf1_per_class"] = bf1_pc
        out["mean_bf1"]      = float(bf1_pc.mean())

    return out


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load a checkpoint saved by train.train_model().

    Expected keys: "model_state", "optimizer_state", "epoch", "best_score", "monitor"
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if "model_state" not in ckpt:
        raise KeyError(
            f"Checkpoint at {ckpt_path} has no 'model_state' key. "
            f"Found keys: {list(ckpt.keys())}"
        )

    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None:
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("Warning: no 'optimizer_state' in checkpoint — optimizer not restored.")

    return ckpt


# ---------------------------------------------------------------------------
# test_model
# ---------------------------------------------------------------------------

def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn,
    class_weight: torch.Tensor,
    num_classes: int,
    ckpt_path: Optional[Path] = None,
    boundary_radius: Optional[int] = None,
) -> Dict:
    """
    Optionally load a checkpoint, run evaluate(), and print a summary.

    Args:
        model:           nn.Module (architecture already instantiated).
        test_loader:     DataLoader for the test split.
        device:          inference device.
        loss_fn:         train.loss_fn — returns a scalar tensor.
        class_weight:    1-D per-class weight tensor.
        num_classes:     number of segmentation classes.
        ckpt_path:       if provided, load weights from this checkpoint.
        boundary_radius: optional boundary F1 dilation radius.

    Returns:
        Stats dict from evaluate().
    """
    if ckpt_path is not None:
        ckpt       = load_checkpoint(Path(ckpt_path), model)
        epoch      = ckpt.get("epoch", "?")
        best_score = ckpt.get("best_score", "?")
        monitor    = ckpt.get("monitor", "?")
        print(f"Loaded checkpoint  epoch={epoch}  best_{monitor}={best_score}")

    model.to(device)
    stats = evaluate(
        model           = model,
        loader          = test_loader,
        device          = device,
        loss_fn         = loss_fn,
        class_weight    = class_weight,
        num_classes     = num_classes,
        boundary_radius = boundary_radius,
    )

    print(f"\nTEST RESULTS")
    print(f"  loss={stats['loss']:.4f}  CE={stats['ce']:.4f}  Dice={stats['dice']:.4f}")
    print(f"  mIoU={stats['mean_iou']:.4f}  mDice={stats['mean_dice']:.4f}")
    print(f"  IoU  per class:  {np.round(stats['iou_per_class'],  4)}")
    print(f"  Dice per class:  {np.round(stats['dice_per_class'], 4)}")
    if "mean_bf1" in stats:
        print(f"  mBF1={stats['mean_bf1']:.4f}  BF1 per class: {np.round(stats['bf1_per_class'], 4)}")

    return stats
@torch.no_grad()
def test_evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    boundary_radius: Optional[int] = None,
) -> Dict:
    model.eval()

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    bf1_acc = [] if boundary_radius is not None else None

    for imgs, masks, _ in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        preds  = torch.argmax(logits, dim=1)

        cm += confusion_matrix(preds.cpu(), masks.cpu(), num_classes=num_classes)

        if boundary_radius is not None:
            bf1 = boundary_f1_per_class(preds.cpu(), masks.cpu(), num_classes=num_classes, radius=boundary_radius)
            bf1_acc.append(bf1.numpy())

    iou_pc,  mean_iou  = _iou_from_cm(cm)
    dice_pc, mean_dice = _dice_from_cm(cm)

    out = {
        "mean_iou":       mean_iou,
        "mean_dice":      mean_dice,
        "iou_per_class":  iou_pc,
        "dice_per_class": dice_pc,
        "cm":             cm,
        "cm_percent":     confusion_percent(cm).numpy(),
    }

    if boundary_radius is not None and bf1_acc:
        bf1_pc = np.mean(np.stack(bf1_acc), axis=0)
        out["bf1_per_class"] = bf1_pc
        out["mean_bf1"]      = float(bf1_pc.mean())

    return out


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    ckpt_path: Optional[Path] = None,
    boundary_radius: Optional[int] = None,
) -> Dict:
    if ckpt_path is not None:
        ckpt       = load_checkpoint(Path(ckpt_path), model)
        epoch      = ckpt.get("epoch", "?")
        best_score = ckpt.get("best_score", "?")
        monitor    = ckpt.get("monitor", "?")
        print(f"Loaded checkpoint  epoch={epoch}  best_{monitor}={best_score}")

    model.to(device)

    stats = test_evaluate(
        model           = model,
        loader          = test_loader,
        device          = device,
        num_classes     = num_classes,
        boundary_radius = boundary_radius,
    )

    print(f"\nTEST RESULTS")
    print(f"  mIoU={stats['mean_iou']:.4f}  mDice={stats['mean_dice']:.4f}")
    print(f"  IoU  per class:  {np.round(stats['iou_per_class'],  4)}")
    print(f"  Dice per class:  {np.round(stats['dice_per_class'], 4)}")
    if "mean_bf1" in stats:
        print(f"  mBF1={stats['mean_bf1']:.4f}  BF1 per class: {np.round(stats['bf1_per_class'], 4)}")

    return stats
