# src/train.py
import argparse, os, json, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAUROC, MulticlassAUROC, BinaryAUROC,
    MultilabelAveragePrecision, MulticlassAveragePrecision, BinaryAveragePrecision
)
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets.medmnist_loader import get_medmnist_loaders
from models.baseline_cnn import BaselineCNN


def build_metrics(task: str, n_classes: int, device):
    if task == "multi-label":
        auroc = MultilabelAUROC(num_labels=n_classes).to(device)
        auprc = MultilabelAveragePrecision(num_labels=n_classes).to(device)
    elif task == "multi-class":
        auroc = MulticlassAUROC(num_classes=n_classes, average="macro").to(device)
        auprc = MulticlassAveragePrecision(num_classes=n_classes, average="macro").to(device)
    else:  # binary-class
        auroc = BinaryAUROC().to(device)
        auprc = BinaryAveragePrecision().to(device)
    return auroc, auprc


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, task):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)  # medmnist 라벨 shape 보정
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        if task == "multi-class":
            y = y.long()
            loss = criterion(logits, y)
        elif task == "binary-class":
            # 🔧 BCE 대신 CE로 2-class 학습
            y = y.long()
            loss = criterion(logits, y)
        else:  # multi-label
            y = y.float()
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device, task, n_classes):
    model.eval()
    auroc, auprc = build_metrics(task, n_classes, device)
    losses, y_true_all, y_pred_all = 0.0, [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)
        logits = model(x)

        # 확률/로스/예측 산출
        if task == "multi-class":
            probs = torch.softmax(logits, dim=1)
            y_int = y.long()
            loss = criterion(logits, y_int)
            y_pred = probs.argmax(1).detach().cpu()
            y_true = y_int.detach().cpu()

            # torchmetrics 업데이트
            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

        elif task == "binary-class":
            # 🔧 CE 기반 2-class 로짓 → softmax 확률
            probs2 = torch.softmax(logits, dim=1)   # [B, 2]
            p1 = probs2[:, 1]                       # 양성 확률 [B]
            y_int = y.long()
            loss = criterion(logits, y_int)

            y_pred = probs2.argmax(1).detach().cpu()
            y_true = y_int.detach().cpu()

            # torchmetrics는 binary에서 [N] 확률/타깃 기대
            auroc.update(p1.to(device), y.to(device))
            auprc.update(p1.to(device), y.to(device))

        else:  # multi-label
            probs = torch.sigmoid(logits)
            y_float = y.float()
            loss = criterion(logits, y_float)
            y_pred = (probs > 0.5).long().detach().cpu()
            y_true = y_float.long().detach().cpu()

            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

        losses += loss.item() * x.size(0)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    avg_loss = losses / len(loader.dataset)
    y_true_all = torch.cat(y_true_all)
    y_pred_all = torch.cat(y_pred_all)

    # F1 (macro) for readability
    if task == "multi-class":
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy(), average="macro")
    elif task == "binary-class":
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy())
    else:
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy(), average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "f1_macro": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="chestmnist")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size
    )
    task, n_classes = meta["task"], meta["n_classes"]
    multi_label = (task == "multi-label")

    model = BaselineCNN(
        n_classes=n_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        multi_label=multi_label
    ).to(device)

    # 🔧 손실함수 선택: binary도 CE 사용
    if task == "multi-class":
        criterion = nn.CrossEntropyLoss()
    elif task == "binary-class":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val = -1
    ckpt_path = os.path.join(args.outdir, "checkpoints", f"{args.dataset}_{args.backbone}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, task)
        val_metrics = evaluate(model, val_loader, criterion, device, task, n_classes)
        dt = time.time() - t0

        log = {
            "epoch": ep,
            "train_loss": tr_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "sec": round(dt, 1)
        }
        print(json.dumps(log, ensure_ascii=False))

        score = val_metrics["auroc"]  # 선택 기준
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), ckpt_path)

    # 최종 테스트
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)  # 안전하게 생성
    test_metrics = evaluate(model, test_loader, criterion, device, task, n_classes)
    with open(os.path.join(args.outdir, "logs", f"{args.dataset}_{args.backbone}_test.json"), "w") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print("TEST:", json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
