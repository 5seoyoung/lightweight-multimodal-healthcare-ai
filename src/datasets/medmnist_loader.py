# src/datasets/medmnist_loader.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_medmnist_loaders(
    name: str = "chestmnist",
    batch_size: int = 128,
    img_size: int = 224,
    download: bool = True,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    name 예시:
      - chestmnist (multi-label 14 classes, 흉부 X-ray)
      - pneumoniamnist (binary)
      - pathmnist, bloodmnist, dermamnist 등
    """
    name = name.lower()
    assert name in INFO, f"{name} is not a valid MedMNIST dataset. Valid keys: {list(INFO.keys())[:8]} ..."
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    # grayscale → 3채널로, 사이즈 통일
    common_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_ds = DataClass(split="train", transform=common_tfms, download=download)
    val_ds   = DataClass(split="val",   transform=common_tfms, download=download)
    test_ds  = DataClass(split="test",  transform=common_tfms, download=download)

    # --- MPS(macOS) 최적화: pin_memory 경고/속도 이슈 회피 ---
    use_mps = torch.backends.mps.is_available()
    if use_mps:
        nw = 0
        pin = False
        persistent = False
    else:
        nw = num_workers
        pin = True
        persistent = nw > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=pin, persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=persistent
    )

    meta = {
        "task": info["task"],                  # "multi-label", "multi-class", "binary-class"
        "n_classes": len(info["label"]),
        "desc": info["description"],
        "name": name,
    }
    return train_loader, val_loader, test_loader, meta
