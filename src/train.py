# src/train.py
import argparse, os, json, time, random, platform
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (
    MultilabelAUROC, MulticlassAUROC, BinaryAUROC,
    MultilabelAveragePrecision, MulticlassAveragePrecision, BinaryAveragePrecision
)
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from datasets.medmnist_loader import get_medmnist_loaders
from models.baseline_cnn import BaselineCNN
from utils.class_freq import estimate_pos_weight
from utils.thresholds import optimal_thresholds


# -------------------------
# Reproducibility helpers
# -------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_deterministic(flag: bool = True):
    torch.backends.cudnn.deterministic = flag
    torch.backends.cudnn.benchmark = not flag
    # for CUDA reproducibility (has no effect on CPU/MPS, harmless otherwise)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# -------------------------
# Metrics builders
# -------------------------
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


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)

        logits = model(x)
        if task == "multi-class":
            loss = criterion(logits, y.long())
        elif task == "binary-class":  # 2-class CE
            loss = criterion(logits, y.long())
        else:  # multi-label
            loss = criterion(logits, y.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task, n_classes, thresholds=None, compute_thresholds=False):
    """
    thresholds: 멀티라벨에서 클래스별 임계값(list or np.ndarray). None이면 0.5 고정.
    compute_thresholds=True이면 검증셋에서 F1-opt 기준의 클래스별 임계값을 탐색해 함께 반환.
    """
    model.eval()
    auroc, auprc = build_metrics(task, n_classes, device)
    losses, y_true_all, y_pred_all = 0.0, [], []
    probs_buf, targs_buf = [], []

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
            probs = torch.sigmoid(logits)  # [B,C]
            y_float = y.float()
            loss = criterion(logits, y_float)

            if thresholds is None:
                y_pred = (probs > 0.5).long().detach().cpu()
            else:
                th = torch.tensor(thresholds, device=probs.device).view(1, -1)
                y_pred = (probs >= th).long().detach().cpu()
            y_true = y_float.long().detach().cpu()

            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

            if compute_thresholds:
                probs_buf.append(probs.detach().cpu())
                targs_buf.append(y_float.detach().cpu())

        losses += loss.item() * x.size(0)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    avg_loss = losses / len(loader.dataset)
    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()

    if task == "multi-class":
        f1 = f1_score(y_true_all, y_pred_all, average="macro")
    elif task == "binary-class":
        f1 = f1_score(y_true_all, y_pred_all)
    else:
        f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    out = {
        "loss": avg_loss,
        "auroc": float(auroc.compute().item()),
        "auprc": float(auprc.compute().item()),
        "f1_macro": float(f1),
    }

    ths = None
    # 멀티라벨: 검증셋에서 클래스별 임계값 탐색 + per-class AUC 리포트
    if task == "multi-label" and compute_thresholds and len(probs_buf) > 0:
        probs_all = torch.cat(probs_buf).numpy()
        targs_all = torch.cat(targs_buf).numpy()
        ths = optimal_thresholds(probs_all, targs_all, steps=100)
        preds_opt = (probs_all >= ths[None, :]).astype(int)
        out["f1_macro_opt"] = float(f1_score(targs_all, preds_opt, average="macro", zero_division=0))
        out["thresholds"] = ths.tolist()

        # per-class AUC (디버깅/분석용)
        per_class_auc = []
        for c in range(n_classes):
            try:
                per_class_auc.append(float(roc_auc_score(targs_all[:, c], probs_all[:, c])))
            except ValueError:
                per_class_auc.append(float("nan"))
        out["per_class_auc"] = per_class_auc

    return out, ths


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="chestmnist")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--aug", type=str, default="light", choices=["none", "light"])
    ap.add_argument("--use_pos_weight", action="store_true")
    ap.add_argument("--patience", type=int, default=3)  # Early stopping patience (AUPRC 기준)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp_name", type=str, default=None, help="결과 디렉토리 이름 override")
    args = ap.parse_args()

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Seed & determinism
    set_global_seed(args.seed)
    set_deterministic(True)

    # Experiment id & output skeleton
    exp_id = args.exp_name or f"{args.dataset}_{args.backbone}_res{args.img_size}_aug{args.aug}_ep{args.epochs}_seed{args.seed}"
    exp_dir = os.path.join(args.outdir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    runlog_path = os.path.join(exp_dir, "run.log")           # JSONL
    metrics_path = os.path.join(exp_dir, "metrics.json")     # summary
    ths_path = os.path.join(exp_dir, "thresholds.json")      # per-class thresholds (val-opt)
    ckpt_path = os.path.join(exp_dir, "ckpt.pt")             # best-on-val(AUPRC)
    test_json_path = os.path.join(exp_dir, "test.json")      # final test metrics

    # 1) 데이터 로더 + 메타
    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size, augment=args.aug
    )
    raw_task = meta["task"]  # 예: "multi-label, binary-class"
    task = "multi-label" if "multi-label" in raw_task else raw_task
    n_classes = meta["n_classes"]

    # 2) pos_weight (멀티라벨 & 옵션 ON일 때)
    pos_weight = None
    if task == "multi-label" and args.use_pos_weight:
        pos_weight = estimate_pos_weight(train_loader, n_classes).to(device)

    # 3) 모델
    model = BaselineCNN(
        n_classes=n_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        multi_label=(task == "multi-label"),
    ).to(device)

    # 4) 손실함수
    if task == "multi-class":
        criterion = nn.CrossEntropyLoss()
    elif task == "binary-class":
        criterion = nn.CrossEntropyLoss()  # 2-class CE
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auprc = -1.0
    best_val_ths = None
    best_val_summary = None
    bad = 0

    # 5) 학습 루프 (간단 워밍업 2ep)
    for ep in range(1, args.epochs + 1):
        # Linear warmup (ep 1~2)
        if ep <= 2:
            warmup_scale = ep / 2.0
            for g in optimizer.param_groups:
                g["lr"] = args.lr * warmup_scale

        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, task)
        val_metrics, ths = evaluate(
            model, val_loader, criterion, device, task, n_classes,
            thresholds=None, compute_thresholds=(task == "multi-label")
        )
        dt = round(time.time() - t0, 3)
        scheduler.step()

        # epoch log (JSONL)
        epoch_log = {
            "epoch": ep,
            "train_loss": tr_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "sec": dt,
        }
        with open(runlog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log, ensure_ascii=False) + "\n")

        # 모델 선택 + 임계값 스냅샷 (AUPRC 기준)
        score = val_metrics["auprc"]
        if score > best_val_auprc:
            best_val_auprc = score
            best_val_ths = ths
            best_val_summary = val_metrics
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
            # 임계값 저장(멀티라벨일 때만 존재)
            if best_val_ths is not None:
                with open(ths_path, "w", encoding="utf-8") as f:
                    json.dump({"thresholds": best_val_ths.tolist()}, f, ensure_ascii=False, indent=2)
        else:
            bad += 1
            if bad >= args.patience:
                break

    # 6) 테스트 (검증에서 얻은 임계값을 적용)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics, _ = evaluate(
        model, test_loader, criterion, device, task, n_classes,
        thresholds=best_val_ths, compute_thresholds=False
    )
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # 7) 요약 메타 + 환경 + 하이퍼 기록
    summary = {
        "exp_id": exp_id,
        "dataset": args.dataset,
        "task": task,
        "n_classes": n_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "backbone": args.backbone,
        "pretrained": bool(args.pretrained),
        "augmentation": args.aug,
        "use_pos_weight": bool(args.use_pos_weight),
        "seed": args.seed,
        "device": str(device),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "selection_metric": "val_auprc",
        "best_on_val": best_val_summary,
        "test": test_metrics,
        "artifacts": {
            "runlog": os.path.relpath(runlog_path, start=args.outdir),
            "thresholds": os.path.relpath(ths_path, start=args.outdir) if os.path.exists(ths_path) else None,
            "checkpoint": os.path.relpath(ckpt_path, start=args.outdir),
            "test_json": os.path.relpath(test_json_path, start=args.outdir),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("TEST:", json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
