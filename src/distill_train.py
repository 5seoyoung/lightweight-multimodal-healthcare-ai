# src/distill_train.py
import argparse, os, json, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from utils.thresholds import optimal_thresholds

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}

# -------------- metrics --------------
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

# -------------- train / eval --------------
def train_one_epoch(student, teacher, loader, loss_fn, optimizer, device, task, ema: EMA = None):
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

        if ema is not None:
            ema.update(student)

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

    ths, f1_opt = None, None
    if task == "multi-label" and len(val_probs_list) > 0:
        probs_all = torch.cat(val_probs_list).numpy()
        targ_all = torch.cat(val_targets_list).numpy()
        ths = optimal_thresholds(probs_all, targ_all, steps=50)  # [n_classes]
        preds_opt = (probs_all >= ths[None, :]).astype(int)
        f1_opt = f1_score(targ_all, preds_opt, average="macro", zero_division=0)

    out = {
        "loss": avg_loss,
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "f1_macro": float(f1),
        "thresholds": ths.tolist() if ths is not None else None,
        "f1_macro_opt": float(f1_opt) if f1_opt is not None else None,
    }
    return out

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="pneumoniamnist")
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--teacher_backbone", type=str, default="resnet50")
    ap.add_argument("--student_backbone", type=str, default="resnet18")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--teacher_ckpt", type=str, default="")
    ap.add_argument("--selection_metric", type=str, default="auprc", choices=["auprc","auroc"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--warmup_ep", type=int, default=3)
    ap.add_argument("--early_patience", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "logs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers
    )

    # task / n_classes
    raw_task = meta["task"]
    task = "multi-label" if "multi-label" in raw_task else raw_task
    n_classes = meta["n_classes"]

    # pos_weight (multi-label만)
    pos_weight = None
    if task == "multi-label":
        pos_weight = estimate_pos_weight(train_loader, n_classes).to(device)

    teacher = BaselineCNN(n_classes, backbone=args.teacher_backbone, pretrained=True,
                          multi_label=(task == "multi-label")).to(device)
    teacher = load_teacher_ckpt(teacher, args.teacher_ckpt, device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = BaselineCNN(n_classes, backbone=args.student_backbone, pretrained=args.pretrained,
                          multi_label=(task == "multi-label")).to(device)

    distill_loss = DistillLoss(task=task, alpha=args.alpha, tau=args.tau, pos_weight=pos_weight)
    optimizer = AdamW(student.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ema = EMA(student, decay=args.ema_decay)

    print(json.dumps({
        "teacher_params_M": round(param_count(teacher), 3),
        "student_params_M": round(param_count(student), 3),
        "task": task, "n_classes": n_classes
    }))
    print(json.dumps({"selection_metric": args.selection_metric}))

    # 체크포인트 경로
    ckpt = os.path.join(
        args.outdir, "checkpoints",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}.pt"
    )
    ths_path = os.path.join(
        args.outdir, "logs",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}_val_thresholds.json"
    )

    best_metric = -1.0
    best_ep = 0
    no_improve = 0
    stopped = False

    for ep in range(1, args.epochs + 1):
        # alpha warmup
        alpha_now = args.alpha * min(1.0, ep / max(1, args.warmup_ep))
        distill_loss.alpha = alpha_now

        t0 = time.time()
        tr_loss = train_one_epoch(student, teacher, train_loader, distill_loss, optimizer, device, task, ema=ema)

        # EMA 가중치로 평가
        ema.apply(student)
        val_metrics = evaluate(student, val_loader, task, n_classes, device)
        ema.restore(student)

        scheduler.step()
        dt = round(time.time() - t0, 1)
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr

        log = {
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": val_metrics["loss"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_macro_opt": val_metrics.get("f1_macro_opt"),
            "alpha_now": alpha_now,
            "lr": lr_now,
            "sec": dt
        }
        print(json.dumps(log, ensure_ascii=False))

        # 선택 지표로 베스트 갱신
        sel = val_metrics[args.selection_metric]
        if sel > best_metric:
            best_metric = sel
            best_ep = ep
            no_improve = 0
            torch.save(student.state_dict(), ckpt)
            # 최적 threshold 저장 (multi-label일 때만)
            if task == "multi-label" and val_metrics.get("thresholds") is not None:
                with open(ths_path, "w") as f:
                    json.dump(val_metrics["thresholds"], f, indent=2)
        else:
            no_improve += 1
            if no_improve >= args.early_patience:
                print(json.dumps({"early_stop": True, "stopped_epoch": ep}))
                stopped = True
                break

    # best 로드 후 테스트 (EMA 적용하여 평가)
    student.load_state_dict(torch.load(ckpt, map_location=device))
    ema.apply(student)
    test_metrics = evaluate(student, test_loader, task, n_classes, device)
    ema.restore(student)

    # 테스트 로그 저장
    test_log_path = os.path.join(
        args.outdir, "logs",
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}_test.json"
    )
    with open(test_log_path, "w") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("TEST:", json.dumps(test_metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
