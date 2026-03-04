"""
train.py
--------
Training and evaluation utilities for semantic segmentation.

Public API
----------
get_device()
build_scheduler(optimizer, scheduler_type, epochs) -> (scheduler, is_plateau)
make_optimizer(model, lr) -> AdamW optimizer
loss_fn(logits, targets, class_weight, device) -> scalar loss
compute_iou(preds, targets, num_classes) -> (mean_iou, iou_per_class)
compute_dice(preds, targets, num_classes) -> (mean_dice, dice_per_class)
train_one_epoch(model, dataloader, optimizer, device, class_weight, num_classes) -> train_loss
evaluate(model, dataloader, device, class_weight, num_classes) -> (val_loss, mean_iou, mean_dice, iou_pc, dice_pc)
train_model(...) -> (history, ckpt_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available accelerator (MPS / CUDA / CPU)."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Optional[str],
    epochs: int,
) -> tuple:
    """
    Build a LR scheduler.

    Returns:
        (scheduler | None, is_plateau: bool)
        is_plateau=True means caller must pass a metric to scheduler.step(metric).
    """
    scheduler_type = (scheduler_type or "").lower().strip()

    if scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1), False
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs), False
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        ), True
    else:
        return None, False


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """AdamW over trainable parameters only."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# DiceLoss instance is stateless — safe to share across calls
_dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)


def loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weight: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Combined Cross-Entropy + Dice loss (0.6 / 0.4 split).

    Args:
        logits:       (N, C, H, W) raw model output.
        targets:      (N, H, W) integer class labels.
        class_weight: 1-D tensor of per-class weights (already a tensor).
        device:       target device for class_weight.

    Returns:
        Scalar loss tensor.
    """
    weight = class_weight.clone().detach().to(dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=weight)(logits, targets)
    dice = _dice_loss(logits, targets)
    return 0.6 * ce + 0.4 * dice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> tuple[float, torch.Tensor]:
    """
    Compute mean IoU and per-class IoU.

    Args:
        preds:       (N, H, W) int64 predictions.
        targets:     (N, H, W) int64 ground-truth labels.
        num_classes: number of classes C.
        eps:         smoothing term to avoid division by zero.

    Returns:
        (mean_iou: float, iou_per_class: Tensor[C])
    """
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        targ_c = targets == c
        inter = (pred_c & targ_c).sum().float()
        union = (pred_c | targ_c).sum().float()
        ious.append((inter + eps) / (union + eps))

    iou_per_class = torch.stack(ious)
    return iou_per_class.mean().item(), iou_per_class.cpu()


@torch.no_grad()
def compute_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> tuple[float, torch.Tensor]:
    """
    Compute mean Dice and per-class Dice.

    Args:
        preds:       (N, H, W) int64 predictions.
        targets:     (N, H, W) int64 ground-truth labels.
        num_classes: number of classes C.
        eps:         smoothing term to avoid division by zero.

    Returns:
        (mean_dice: float, dice_per_class: Tensor[C])
    """
    dices = []
    for c in range(num_classes):
        pred_c = preds == c
        targ_c = targets == c
        inter = (pred_c & targ_c).sum().float()
        denom = pred_c.sum().float() + targ_c.sum().float()
        dices.append((2 * inter + eps) / (denom + eps))

    dice_per_class = torch.stack(dices)
    return dice_per_class.mean().item(), dice_per_class.cpu()


# ---------------------------------------------------------------------------
# Train / Eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weight: torch.Tensor,
) -> float:
    """Run one full training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for imgs, masks, _ in dataloader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = loss_fn(logits, masks, class_weight=class_weight, device=device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_weight: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """
    Evaluate model on a dataloader.

    Returns:
        (val_loss, mean_iou, mean_dice, iou_per_class, dice_per_class)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = loss_fn(logits, masks, class_weight=class_weight, device=device)
            total_loss += loss.item()

            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mean_iou,  iou_pc  = compute_iou(all_preds,  all_targets, num_classes)
    mean_dice, dice_pc = compute_dice(all_preds, all_targets, num_classes)

    return total_loss / len(dataloader), mean_iou, mean_dice, iou_pc, dice_pc


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    device: torch.device,
    class_weight: torch.Tensor,
    num_classes: int,
    num_epochs: int = 15,
    freeze_epochs: int = 4,
    lr_frozen: float = 1e-3,
    lr_unfrozen: float = 1e-4,
    scheduler_type: Optional[str] = None,   # "step" | "cosine" | "plateau" | None
    early_stop_patience: int = 3,
    monitor: str = "miou",                  # "miou" | "val_loss"
    ckpt_dir: str | Path = "./checkpoints",
    ckpt_name: str = "best_model.pth",
    verbose: bool = True,
) -> tuple[list[dict], Path]:
    """
    Train with optional encoder freeze → unfreeze and early stopping.

    Args:
        model:                nn.Module to train (moved to device by caller).
        train_dl / val_dl:    DataLoaders yielding (imgs, masks, *).
        device:               training device.
        class_weight:         1-D tensor of per-class CE weights.
        num_classes:          number of segmentation classes.
        num_epochs:           total training epochs.
        freeze_epochs:        epochs to keep encoder frozen (0 to skip).
        lr_frozen:            LR while encoder is frozen.
        lr_unfrozen:          LR after encoder is unfrozen.
        scheduler_type:       "step" | "cosine" | "plateau" | None.
        early_stop_patience:  epochs with no improvement before stopping.
        monitor:              "miou" (maximize) or "val_loss" (minimize).
        ckpt_dir / ckpt_name: checkpoint save location.
        verbose:              print per-epoch summary.

    Returns:
        (history: list of dicts, ckpt_path: Path)
    """
    ckpt_dir  = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name

    # --- freeze encoder if model has one (e.g. SMP pretrained) ---
    if freeze_epochs > 0 and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    optimizer = make_optimizer(model, lr_frozen)
    scheduler, scheduler_is_plateau = build_scheduler(optimizer, scheduler_type, num_epochs)

    # --- early-stopping setup ---
    monitor = monitor.lower()
    if monitor == "miou":
        best_score = -np.inf
        is_better = lambda score, best: score > best
    elif monitor == "val_loss":
        best_score = np.inf
        is_better = lambda score, best: score < best
    else:
        raise ValueError(f"monitor must be 'miou' or 'val_loss', got '{monitor}'")

    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(num_epochs):

        # --- unfreeze encoder after freeze_epochs ---
        if epoch == freeze_epochs and hasattr(model, "encoder"):
            if verbose:
                print(f"[Epoch {epoch + 1}] Unfreezing encoder, switching to lr={lr_unfrozen:.2e}")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = make_optimizer(model, lr_unfrozen)
            scheduler, scheduler_is_plateau = build_scheduler(
                optimizer, scheduler_type, num_epochs - epoch
            )

        # --- train + validate ---
        train_loss = train_one_epoch(model, train_dl, optimizer, device, class_weight)
        val_loss, mean_iou, mean_dice, iou_pc, dice_pc = evaluate(
            model, val_dl, device, class_weight, num_classes
        )

        lr_now = optimizer.param_groups[0]["lr"]

        # --- scheduler step ---
        if scheduler is not None:
            scheduler.step(val_loss) if scheduler_is_plateau else scheduler.step()

        # --- checkpoint & early stopping ---
        score   = mean_iou if monitor == "miou" else val_loss
        is_best = is_better(score, best_score)

        if is_best:
            best_score = score
            bad_epochs = 0
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score":      best_score,
                    "monitor":         monitor,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch":      epoch + 1,
                "lr":         lr_now,
                "train_loss": float(train_loss),
                "val_loss":   float(val_loss),
                "mean_iou":   float(mean_iou),
                "mean_dice":  float(mean_dice),
                "is_best":    bool(is_best),
                "bad_epochs": int(bad_epochs),
            }
        )

        if verbose:
            flag = "  ← best" if is_best else ""
            print(
                f"Epoch {epoch + 1:02d}/{num_epochs}  lr={lr_now:.2e} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
                f"mIoU={mean_iou:.3f}  mDice={mean_dice:.3f}{flag}"
            )

        if bad_epochs >= early_stop_patience:
            if verbose:
                print(
                    f"\nEarly stopping after {bad_epochs} epochs without improvement. "
                    f"Best {monitor}={best_score:.4f}  →  saved to {ckpt_path}"
                )
            break

    return history, ckpt_path
