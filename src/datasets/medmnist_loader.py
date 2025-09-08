# src/datasets/medmnist_loader.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import medmnist
from medmnist import INFO

def get_medmnist_loaders(
    name: str = "chestmnist",
    batch_size: int = 128,
    img_size: int = 224,
    download: bool = True,
    num_workers: int = 2,
    augment: str = "none",  # ← "none" | "light"
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:

    name = name.lower()
    assert name in INFO, f"{name} is not a valid MedMNIST dataset."
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    if augment == "light":
        train_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # "none"
        train_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = DataClass(split="train", transform=train_tfms, download=download)
    val_ds   = DataClass(split="val",   transform=eval_tfms, download=download)
    test_ds  = DataClass(split="test",  transform=eval_tfms, download=download)

    # MPS(mac) 로더 옵션
    use_mps = torch.backends.mps.is_available()
    if use_mps:
        nw = 0; pin = False; persistent = False
    else:
        nw = num_workers; pin = True; persistent = nw > 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=persistent)

    meta = {"task": info["task"], "n_classes": len(info["label"]), "desc": info["description"], "name": name}
    return train_loader, val_loader, test_loader, meta
