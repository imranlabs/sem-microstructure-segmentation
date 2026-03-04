"""
utils.py
Utilities for SEM segmentation project:
- reproducibility seed
- class frequency computation
- plotting: confusion matrix (%), class frequency vs IoU
- overlay visualization: image + GT overlay + pred overlay

Matplotlib only (no seaborn).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_class_frequencies(
    loader: Iterable,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Pixel fraction per class over a loader.
    Returns (C,) array that sums to ~1.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total = 0.0

    for _, masks, _ in loader:
        masks = masks.to(device)
        total += masks.numel()
        for c in range(num_classes):
            counts[c] += (masks == c).sum()

    frac = (counts / max(total, 1.0)).detach().cpu().numpy()
    return frac


def plot_confusion_matrix_percent(
    cm_percent: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix (%)",
    save_path: Optional[Path] = None,
) -> None:
    C = cm_percent.shape[0]
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_percent, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(C)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    for i in range(C):
        for j in range(C):
            plt.text(j, i, f"{cm_percent[i, j]:.2f}", ha="center", va="center")

    plt.ylabel("True class")
    plt.xlabel("Pred class")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_freq_vs_iou(
    class_freq: np.ndarray,
    iou_per_class: np.ndarray,
    class_names: Sequence[str],
    title: str = "Class Frequency vs IoU",
    save_path: Optional[Path] = None,
) -> None:
    C = len(class_names)
    x = np.arange(C)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, class_freq, marker="o", label="Pixel fraction")
    plt.plot(x, iou_per_class, marker="o", label="IoU")
    plt.xticks(x, class_names, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


@torch.no_grad()
def visualize_overlays(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    num_samples: int = 6,
    alpha: float = 0.35,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Creates image + GT overlay + pred overlay triplets.
    Uses first channel of img as grayscale background.
    """
    model.eval()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    shown = 0
    for imgs, masks, fnames in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        imgs_cpu = imgs.detach().cpu()
        masks_cpu = masks.detach().cpu()
        preds_cpu = preds.detach().cpu()

        B = imgs_cpu.shape[0]
        for b in range(B):
            if shown >= num_samples:
                return

            img = imgs_cpu[b, 0].numpy()
            gt = masks_cpu[b].numpy()
            pr = preds_cpu[b].numpy()
            fname = fnames[b]

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(img, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("GT overlay")
            plt.imshow(img, cmap="gray")
            plt.imshow(gt, alpha=alpha)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Pred overlay")
            plt.imshow(img, cmap="gray")
            plt.imshow(pr, alpha=alpha)
            plt.axis("off")

            plt.suptitle(str(fname))
            plt.tight_layout()

            if save_dir is not None:
                out = save_dir / f"overlay_{shown:03d}_{Path(fname).stem}.png"
                plt.savefig(out, dpi=200)

            plt.show()
            shown += 1

def maskinfo(mask_path:str):
    """ 
    args : mask file path 
    returns : Maks image mode(grayscale or RGB), 
             image shape, unique label values.
    """
    m = Image.open(mask_path)
    arr = np.array(m)
    print("PIL mode:", m.mode)
    print("Shape:", arr.shape, "dtype:", arr.dtype)

    # Case A : Single-channel mask (H, W)
    if arr.ndim == 2:
        uniq = np.unique(arr)
        print("Unique label value:", uniq)
        print("num unique:", len(uniq))

    # Case B: Color-coded mask (H,W,3)

    elif arr.ndim == 3 and arr.shape[2] in (3,4):
        rgb = arr[...,:3] # ignore alpha if present
        colors = np.unique(rgb.reshape(-1,3),axis=0)
        print("num unique colors:",len(colors))
        print("first 20 colors:\n", colors[:20])
    else:
        print("unexpected mask format.")


def mask_ids(mask_dir):
    """
    Builds a mask id dictionary
    args : mask dir path
    return : dictionary . keys :file names; values: unique ids
    
    
    """
    id_dict = {}
    for path in Path(mask_dir).glob("*.bmp"):
        m = Image.open(path)
        uniq = np.unique(np.array(m))
        #id_dict[path.stem] = uniq 
        id_dict[str(path)] = uniq 
    return id_dict


def show_sample(img_t:Optional[torch.Tensor], mask_t:Optional[torch.Tensor], title=""):
    """ img_t: (1,H,W), mask_t: (H,W) """
    
    img = img_t.squeeze(0).cpu().numpy()
    mask = mask_t.cpu().numpy()
   
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Mask (remapped IDs 0..4)")
    plt.imshow(mask)
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, alpha=0.35)
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()