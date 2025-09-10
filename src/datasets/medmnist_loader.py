# src/datasets/medmnist_loader.py
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

def _build_transforms(img_size: int, augment: str = "none"):
    """img_size에 맞춰 공통 전처리 + augment 옵션 적용"""
    t = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),  # MobileNet 등 3채널 모델 호환
    ]

    if augment in ("light", "heavy"):
        if augment == "light":
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        else:  # heavy
            try:
                from torchvision.transforms import RandAugment
                aug = [RandAugment(num_ops=2, magnitude=7)]
            except Exception:
                aug = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomResizedCrop((img_size, img_size), scale=(0.9, 1.0)),
                ]
        t.extend(aug)

    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(t)


def get_medmnist_loaders(
    name: str,
    batch_size: int,
    img_size: int,
    augment: str = "none",         # ✅ 추가된 인자
    num_workers: int = 4,
):
    """MedMNIST 표준 train/val/test 로더와 메타데이터 반환"""
    assert name in INFO, f"Unknown MedMNIST dataset: {name}"
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])

    # train에는 augment, val/test는 항상 eval transform
    train_tf = _build_transforms(img_size, augment)
    eval_tf  = _build_transforms(img_size, "none")

    train_ds = DataClass(split="train", transform=train_tf, download=True)
    val_ds   = DataClass(split="val",   transform=eval_tf,  download=True)
    test_ds  = DataClass(split="test",  transform=eval_tf,  download=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # 메타정보
    # n_classes는 INFO에 있으면 사용, 없으면 라벨 모양에서 추정(멀티라벨 대응)
    n_classes = info.get("n_classes", None)
    if n_classes is None:
        try:
            n_classes = train_ds.labels.shape[1] if train_ds.labels.ndim > 1 else int(train_ds.labels.max()) + 1
        except Exception:
            n_classes = 1

    meta = {"task": info["task"], "n_classes": int(n_classes)}
    return train_loader, val_loader, test_loader, meta
