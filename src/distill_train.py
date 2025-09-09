# src/distill_train.py
import argparse, os, json, time, math
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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


# ----------------------------
# Metrics builders
# ----------------------------
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


# ----------------------------
# Train / Eval
# ----------------------------
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

    # 멀티라벨 최적 threshold 산출용 버퍼
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

            # 최적 threshold 계산용 수집
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

    # 멀티라벨: 검증 세트에서 클래스별 최적 임계값으로 F1 재계산
    if task == "multi-label" and len(val_probs_list) > 0:
        import numpy as np  # noqa
        from utils.thresholds import optimal_thresholds
        probs_all = torch.cat(val_probs_list).numpy()
        targ_all = torch.cat(val_targets_list).numpy()
        ths = optimal_thresholds(probs_all, targ_all, steps=50)  # [n_classes]
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


# ----------------------------
# Schedulers
# ----------------------------
def build_warmup_cosine(optimizer, num_epochs, warmup_epochs=2):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        t = (ep - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return LambdaLR(optimizer, lr_lambda)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="pneumoniamnist")
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--teacher_backbone", type=str, default="resnet50")
    ap.add_argument("--student_backbone", type=str, default="resnet18")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--teacher_ckpt", type=str, default="", help="path to pre-trained teacher checkpoint (.pt)")
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs)")
    ap.add_argument("--select_metric", type=str, default="", choices=["", "auroc", "auprc"],
                    help="validation metric to select best checkpoint; default: auprc (multi-label) / auroc (others)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size
    )

    # task 문자열 정규화 및 클래스 수
    raw_task = meta["task"]
    if "multi-label" in raw_task:
        task = "multi-label"
    else:
        task = raw_task
    n_classes = meta["n_classes"]

    # pos_weight (멀티라벨에서만)
    pos_weight = None
    if task == "multi-label":
        pos_weight = estimate_pos_weight(train_loader, n_classes).to(device)

    # teacher / student
    teacher = BaselineCNN(n_classes, backbone=args.teacher_backbone, pretrained=True,
                          multi_label=(task == "multi-label")).to(device)
    teacher = load_teacher_ckpt(teacher, args.teacher_ckpt, device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = BaselineCNN(n_classes, backbone=args.student_backbone, pretrained=args.pretrained,
                          multi_label=(task == "multi-label")).to(device)

    distill_loss = DistillLoss(task=task, alpha=args.alpha, tau=args.tau, pos_weight=pos_weight).to(device)
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    # 파라미터/태스크 정보 출력
    print(json.dumps({
        "teacher_params_M": round(param_count(teacher), 3),
        "student_params_M": round(param_count(student), 3),
        "task": task, "n_classes": n_classes
    }))

    # 선택 지표: 기본은 멀티라벨=auprc, 그 외=auroc (명시되면 우선)
    if args.select_metric:
        select_metric = args.select_metric
    else:
        select_metric = "auprc" if task == "multi-label" else "auroc"
    print(json.dumps({"selection_metric": select_metric}))

    best_val = -1.0
    bad_epochs = 0
    ckpt = os.path.join(
        args.outdir, "checkpoints",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}.pt"
    )
    warmup_ep = max(1, args.warmup_epochs)

    for ep in range(1, args.epochs + 1):
        # linear warmup for alpha (3ep 고정 워밍업은 이전 호환 유지)
        alpha_now = args.alpha * min(1.0, ep / 3.0)
        distill_loss.alpha = alpha_now  # 런타임 갱신

        t0 = time.time()
        tr_loss = train_one_epoch(student, teacher, train_loader, distill_loss, optimizer, device, task)
        val_metrics = evaluate(student, val_loader, task, n_classes, device)
        scheduler.step()
        dt = round(time.time() - t0, 1)

        # 에폭 로그(+ 현재 lr/alpha)
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch": ep,
            "train_loss": tr_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "alpha_now": float(alpha_now),
            "lr": float(cur_lr),
            "sec": dt
        }
        print(json.dumps(log, ensure_ascii=False))

        # 체크포인트 선택 + 얼리스탑
        score = val_metrics[select_metric]
        if score > best_val:
            best_val = score
            torch.save(student.state_dict(), ckpt)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(json.dumps({"early_stop": True, "stopped_epoch": ep}))
                break

    # best ckpt 로드 후 테스트
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
