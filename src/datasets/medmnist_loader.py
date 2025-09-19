# src/datasets/medmnist_loader.py
import os
import math
import torch
import medmnist
from medmnist import INFO
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---- small utilities ----
class RandomGamma(torch.nn.Module):
    """Gamma in [min_gamma, max_gamma] with prob p (before ToTensor)."""
    def __init__(self, min_gamma: float = 0.9, max_gamma: float = 1.1, p: float = 0.5):
        super().__init__()
        self.min_gamma, self.max_gamma, self.p = min_gamma, max_gamma, p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            g = float(torch.empty(1).uniform_(self.min_gamma, self.max_gamma))
            return TF.adjust_gamma(img, gamma=g)
        return img


def _square_resize_keep_ar(img_size: int) -> transforms.Compose:
    """
    Keep aspect ratio, then center-crop to a square.
    - Resize(img_size): short side == img_size
    - CenterCrop(img_size): square crop
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ])


def _build_transforms(img_size: int, augment: str = "none") -> transforms.Compose:
    """
    Compose transforms that match the paper's 'medically conservative' policy:
      - AR preserving resize -> center crop to square
      - Grayscale->3ch for ImageNet-pretrained backbones
      - Light aug: HFlip(0.5) + mild ColorJitter(±10%) + RandomGamma(0.9–1.1)
      - Heavy aug: RandAugment(N=2, M=7) fallback로 소폭 기하/광학 강화
    """
    t = [
        _square_resize_keep_ar(img_size),
        transforms.Grayscale(num_output_channels=3),  # 3ch backbone 호환
    ]

    if augment in ("light", "heavy"):
        if augment == "light":
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                RandomGamma(0.9, 1.1, p=0.5),
            ]
        else:  # "heavy"
            try:
                from torchvision.transforms import RandAugment
                aug = [RandAugment(num_ops=2, magnitude=7)]
            except Exception:
                aug = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
        t.extend(aug)

    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(t)


def _class_prevalence(labels) -> Optional[list]:
    """
    Compute per-class positive rate for multilabel; else return None.
    MedMNIST labels are numpy arrays.
    """
    try:
        import numpy as np
        arr = labels
        if hasattr(labels, "numpy"):  # just in case
            arr = labels.numpy()
        if arr.ndim == 2:  # multilabel
            n = arr.shape[0]
            pos = arr.sum(axis=0)
            prev = (pos / max(1, n)).astype(float).tolist()
            return prev
    except Exception:
        pass
    return None


def get_medmnist_loaders(
    name: str,
    batch_size: int,
    img_size: int,
    augment: str = "none",
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train/val/test loaders for a MedMNIST dataset with paper-aligned transforms.
    Returns (train_loader, val_loader, test_loader, meta)
      - meta: {task, n_classes, label_names, prevalence, split_sizes}
    """
    if name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset name: {name}. "
                         f"Valid keys include: {sorted(INFO.keys())[:8]} ...")

    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    # num_workers: (cores-1) with floor 2
    if num_workers is None:
        ncpu = os.cpu_count() or 2
        num_workers = max(2, ncpu - 1)

    # pin_memory: CUDA에서만 True (MPS/CPU는 False 권장)
    pin = torch.cuda.is_available()
    common_kwargs = dict(num_workers=num_workers, pin_memory=pin, drop_last=False)
    if num_workers > 0:
        common_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))

    # transforms
    train_tf = _build_transforms(img_size, augment)
    eval_tf = _build_transforms(img_size, "none")

    # datasets
    train_ds = DataClass(split="train", transform=train_tf, download=True)
    val_ds   = DataClass(split="val",   transform=eval_tf,  download=True)
    test_ds  = DataClass(split="test",  transform=eval_tf,  download=True)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common_kwargs)

    # meta
    # n_classes: from INFO if present; fallback to label shapes
    if "n_classes" in info:
        n_classes = int(info["n_classes"])
    else:
        lab = train_ds.labels
        n_classes = int(lab.shape[1]) if lab.ndim > 1 else int(lab.max()) + 1

    label_names = None
    if "label" in info and isinstance(info["label"], dict):
        # dict: {idx: name}; make list ordered by idx
        try:
            # INFO["label"] keys may be strings; sort by int(key)
            label_names = [info["label"][k] for k in sorted(info["label"].keys(), key=lambda x: int(x))]
        except Exception:
            label_names = list(info["label"].values())

    prevalence = _class_prevalence(train_ds.labels)
    split_sizes = {
        "train": int(len(train_ds)),
        "val":   int(len(val_ds)),
        "test":  int(len(test_ds)),
    }

    meta = {
        "task": info["task"],                # e.g., "multi-label, binary-class"
        "n_classes": n_classes,
        "label_names": label_names,          # may be None for some datasets
        "prevalence": prevalence,            # list for multilabel; else None
        "split_sizes": split_sizes,
        "img_size": int(img_size),
        "augment": augment,
    }
    return train_loader, val_loader, test_loader, meta
