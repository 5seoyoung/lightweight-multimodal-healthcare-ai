# src/distill_train.py
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
from models.distill import DistillLoss
from utils.class_freq import estimate_pos_weight


def build_metrics(task: str, n_classes: int, device):
    if task == "multi-label":
        auroc = MultilabelAUROC(num_labels=n_classes).to(device)
        auprc = MultilabelAveragePrecision(num_labels=n_classes).to(device)
    elif task == "multi-class":
        auroc = MulticlassAUROC(num_classes=n_classes, average="macro").to(device)
        auprc = MulticlassAveragePrecision(num_classes=n_classes, average="macro").to(device)
    else:
        auroc = BinaryAUROC().to(device)
        auprc = BinaryAveragePrecision().to(device)
    return auroc, auprc


def param_count(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def load_teacher_ckpt(model, ckpt_path, device):
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(json.dumps({"msg": "loaded_teacher_ckpt", "path": ckpt_path}))
    else:
        print(json.dumps({"msg": "teacher_ckpt_not_found_or_skipped", "path": ckpt_path}))
    return model


def train_one_epoch(student, teacher, loader, loss_fn, optimizer, device, task):
    student.train(); teacher.eval()
    running = 0.0
    for x, y in tqdm(loader, desc="Train(distill)", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)

        with torch.no_grad():
            t_logits = teacher(x)

        s_logits = student(x)
        loss = loss_fn(s_logits, t_logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, task, n_classes, device):
    model.eval()
    auroc, auprc = build_metrics(task, n_classes, device)
    losses, y_true_all, y_pred_all = 0.0, [], []

    if task in ("multi-class", "binary-class"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # 🔧 추가: 멀티라벨 최적 threshold 산출용 버퍼
    val_probs_list, val_targets_list = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)
        logits = model(x)

        if task == "multi-class":
            probs = torch.softmax(logits, dim=1)
            y_int = y.long()
            loss = criterion(logits, y_int)
            y_pred = probs.argmax(1).detach().cpu()
            y_true = y_int.detach().cpu()
            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

        elif task == "binary-class":
            probs2 = torch.softmax(logits, dim=1)  # [B,2]
            p1 = probs2[:, 1]
            y_int = y.long()
            loss = criterion(logits, y_int)
            y_pred = probs2.argmax(1).detach().cpu()
            y_true = y_int.detach().cpu()
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

            # 🔧 수집: 최적 threshold 계산용
            val_probs_list.append(probs.detach().cpu())
            val_targets_list.append(y_float.detach().cpu())

        losses += loss.item() * x.size(0)
        y_true_all.append(y_true); y_pred_all.append(y_pred)

    avg_loss = losses / len(loader.dataset)
    y_true_all = torch.cat(y_true_all); y_pred_all = torch.cat(y_pred_all)

    if task == "multi-class":
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy(), average="macro")
    elif task == "binary-class":
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy())
    else:
        f1 = f1_score(y_true_all.numpy(), y_pred_all.numpy(), average="macro", zero_division=0)

    # 🔧 멀티라벨: 검증 세트에서 클래스별 최적 임계값으로 F1 재계산
    if task == "multi-label" and len(val_probs_list) > 0:
        import numpy as np  # noqa: F401 (일부 환경에서 utils가 numpy 의존)
        from utils.thresholds import optimal_thresholds
        probs_all = torch.cat(val_probs_list).numpy()
        targ_all = torch.cat(val_targets_list).numpy()
        ths = optimal_thresholds(probs_all, targ_all, steps=50)  # shape: [n_classes]
        preds_opt = (probs_all >= ths[None, :]).astype(int)
        f1_opt = f1_score(targ_all, preds_opt, average="macro", zero_division=0)
    else:
        ths, f1_opt = None, None

    out = {
        "loss": avg_loss,
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "f1_macro": float(f1),
    }
    if task == "multi-label":
        out["f1_macro_opt"] = float(f1_opt) if f1_opt is not None else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="pneumoniamnist")
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--teacher_backbone", type=str, default="resnet50")
    ap.add_argument("--student_backbone", type=str, default="resnet18")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--teacher_ckpt", type=str, default="", help="path to pre-trained teacher checkpoint (.pt)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size
    )

    # 🔧 task 문자열 정규화 및 클래스 수 추출을 먼저 수행
    raw_task = meta["task"]
    if "multi-label" in raw_task:
        task = "multi-label"
    else:
        task = raw_task
    n_classes = meta["n_classes"]

    # 🔧 (순서 수정) pos_weight 계산은 task/n_classes 확정 후에
    pos_weight = None
    if task == "multi-label":
        pos_weight = estimate_pos_weight(train_loader, n_classes).to(device)

    teacher = BaselineCNN(n_classes, backbone=args.teacher_backbone, pretrained=True,
                          multi_label=(task == "multi-label")).to(device)
    teacher = load_teacher_ckpt(teacher, args.teacher_ckpt, device)

    student = BaselineCNN(n_classes, backbone=args.student_backbone, pretrained=args.pretrained,
                          multi_label=(task == "multi-label")).to(device)

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    distill_loss = DistillLoss(task=task, alpha=args.alpha, tau=args.tau, pos_weight=pos_weight)
    optimizer = AdamW(student.parameters(), lr=args.lr)

    print(json.dumps({
        "teacher_params_M": round(param_count(teacher), 3),
        "student_params_M": round(param_count(student), 3),
        "task": task, "n_classes": n_classes
    }))

    best_val = -1.0
    ckpt = os.path.join(
        args.outdir, "checkpoints",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}.pt"
    )

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(student, teacher, train_loader, distill_loss, optimizer, device, task)
        val_metrics = evaluate(student, val_loader, task, n_classes, device)
        dt = round(time.time() - t0, 1)

        log = {"epoch": ep, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_metrics.items()}, "sec": dt}
        print(json.dumps(log, ensure_ascii=False))

        if val_metrics["auroc"] > best_val:
            best_val = val_metrics["auroc"]
            torch.save(student.state_dict(), ckpt)

    student.load_state_dict(torch.load(ckpt, map_location=device))
    test_metrics = evaluate(student, test_loader, task, n_classes, device)
    with open(os.path.join(
        args.outdir, "logs",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}_test.json"
    ), "w") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print("TEST:", json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
