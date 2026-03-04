"""
dataset.py
- Label remap LUT builder (raw IDs -> 0..C-1)
- PyTorch Dataset for SEM segmentation
- Example usage: 
  Scratch model (1-chanel)
    train_ds = SEMSegDataset(IMAGES_DIR, MASKS_DIR, train_files, to_rgb=False)
   Pretrained ResNet34 encoder (needs 3-channel):  
    train_ds = SEMSegDataset(IMAGES_DIR, MASKS_DIR, train_files, to_rgb=True)

- base_key(fname:str)
- make_grouped_split(filenames, n_train=300, n_val=50, n_test=55, seed=42):


"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import os, random, glob
from collections import defaultdict
import torch
from PIL import Image
from torch.utils.data import Dataset

# Raw label IDs in mask -> contiguous train IDs
ID_MAP: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
NUM_CLASSES: int = len(ID_MAP)
INVALID_LABEL: int = 255  # value used for unmapped/invalid raw IDs


def build_remap_lut(id_map: Dict[int, int], invalid: int = INVALID_LABEL) -> np.ndarray:
    """
    Build a 256-entry LUT for fast remapping of uint8 masks.
    Any raw ID not in id_map maps to `invalid`.
    """
    lut = np.full(256, invalid, dtype=np.uint8)
    for src, dst in id_map.items():
        lut[int(src)] = int(dst)
    return lut


REMAP: np.ndarray = build_remap_lut(ID_MAP)


class SEMSegDataset(Dataset):
    """
    Returns:
      img:  (C,H,W) float32 in [0,1]   (C=1 by default)
      mask: (H,W)   int64 with values 0..NUM_CLASSES-1
      fname: str
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        filenames: List[str],
        to_rgb: bool = False,  # set True if you want (3,H,W) by repeating grayscale
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.filenames = list(filenames)
        self.to_rgb = to_rgb

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname = self.filenames[idx]

        img_path = self.images_dir / fname
        msk_path = self.masks_dir / fname

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not msk_path.exists():
            raise FileNotFoundError(f"Mask not found: {msk_path}")

        # --- image: grayscale -> float32 [0,1], shape (1,H,W) ---
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        img = img / 255.0
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)

        if self.to_rgb:
            img_t = img_t.repeat(3, 1, 1)  # (3,H,W)

        # --- mask: uint8 raw IDs -> int64 0..C-1 ---
        msk = np.array(Image.open(msk_path), dtype=np.uint8)
        msk2 = REMAP[msk].astype(np.int64)

        # Safety: ensure all labels are mapped
        if (msk2 == INVALID_LABEL).any():
            bad = np.unique(msk[msk2 == INVALID_LABEL])
            raise ValueError(f"Unmapped label values in {fname}: {bad.tolist()}")

        mask_t = torch.from_numpy(msk2)  # (H,W) int64

        return img_t, mask_t, fname
    
def base_key(fname:str) -> str:
    """ 
    Extracts basekey from filename 
    args : filenae
    return : basekey
    sem700_x0_y288_ElasticTransform.bmp -> sem700_x0_y288
    
    """
    parts = fname.split("_")
    return "_".join(parts[:3]) # sem{mag}_x{}_y{}

def make_grouped_split(filenames, n_train=300, n_val=50, n_test=55, seed=42):
    groups = defaultdict(list)
    for f in filenames:
        # base_key(f) creates "key". 
        # If the base_key is new: defaultdict instantly creates groups[key] = []. 
        # Then, .append(f) adds the filename to that brand-new list.
        # If the base_key already exists: It simply finds the existing list for that key and adds the filename to the end of it. 
        groups[base_key(f)].append(f) 

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    # allocate by number of *files* roughly but keep groups intact
    splits = {"train":[],"val":[],"test":[]}
    targets = {"train":n_train,"val":n_val,"test":n_test}

    for k in keys:
        # choose the split that is most under its target
        best = min(targets.keys(), key=lambda s:len(splits[s])/max(targets[s],1))
        splits[best].extend(groups[k])

    return splits, groups

