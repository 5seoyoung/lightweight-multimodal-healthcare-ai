# src/distill_train.py
import argparse, os, json, time, random, platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (
    MultilabelAUROC, MulticlassAUROC, BinaryAUROC,
    MultilabelAveragePrecision, MulticlassAveragePrecision, BinaryAveragePrecision
)
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets.medmnist_loader import get_medmnist_loaders
from models.baseline_cnn import BaselineCNN
from utils.class_freq import estimate_pos_weight
from utils.thresholds import optimal_thresholds


# ------------------------ Reproducibility ------------------------
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def set_deterministic(flag: bool = True):
    torch.backends.cudnn.deterministic = flag
    torch.backends.cudnn.benchmark = not flag
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# ------------------------ Metrics builders ------------------------
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


# ------------------------ KD loss (multilabel) ------------------------
class KDCombinedLoss(nn.Module):
    """
    Multi-label:
      L = (1-α) * BCEWithLogits (per-class) +
          α * τ^2 * BCEWithLogits( s_logits/τ , sigmoid(t_logits/τ).detach() ) [class-weighted]
    Multi-class/Binary: CE + KL on softmax with temperature (변경 없음).
    """
    def __init__(self, task: str, alpha: float, tau: float,
                 pos_weight: torch.Tensor | None = None,
                 kd_class_weights: torch.Tensor | None = None):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.tau = tau
        self.kd_w = kd_class_weights  # [C] or None

        if task == "multi-label":
            # supervised term: per-class BCEWithLogits
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, targets):
        if self.task == "multi-label":
            # 1) supervised (hard) term
            bce_mat = self.bce(s_logits, targets.float())  # [B, C]
            sup = bce_mat.mean()

            # 2) soft KD term (teacher probs detached)
            tau = self.tau
            with torch.no_grad():
                t_soft = torch.sigmoid(t_logits / tau)  # [B, C], no grad

            # per-element BCE-with-logits in logit space (안정적, MPS 안전)
            kd_elem = F.binary_cross_entropy_with_logits(
                s_logits / tau, t_soft, reduction="none"
            )  # [B, C]

            if self.kd_w is not None:
                w = self.kd_w.view(1, -1).to(kd_elem.device)
                kd = (kd_elem * w).mean()
            else:
                kd = kd_elem.mean()

            return (1 - self.alpha) * sup + self.alpha * (tau ** 2) * kd

        elif self.task == "multi-class":
            sup = self.ce(s_logits, targets.long())
            tau = self.tau
            with torch.no_grad():
                t_prob = torch.softmax(t_logits / tau, dim=1)
            s_log_prob = torch.log_softmax(s_logits / tau, dim=1)
            kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            return (1 - self.alpha) * sup + self.alpha * (tau ** 2) * kl

        else:  # binary-class (2-way CE + KD on softmax)
            sup = self.ce(s_logits, targets.long())
            tau = self.tau
            with torch.no_grad():
                t_prob = torch.softmax(t_logits / tau, dim=1)
            s_log_prob = torch.log_softmax(s_logits / tau, dim=1)
            kl = F.kl_div(s_log_prob, t_prob, reduction="batchmean")
            return (1 - self.alpha) * sup + self.alpha * (tau ** 2) * kl



# ------------------------ Feature losses ------------------------
def attention_map(feat: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> [B, 1, H, W], channel energy
    am = feat.pow(2).mean(1, keepdim=True)
    # L2 normalize per-sample
    b = am.shape[0]
    am = am.view(b, -1)
    am = F.normalize(am, p=2, dim=1)
    return am.view(b, 1, -1)  # flattened spatial with channel=1

def feature_distill_loss(student_feats, teacher_feats, mode: str = "at") -> torch.Tensor:
    """
    student_feats / teacher_feats: list of feature maps (deepest last)
    mode: 'at' (attention transfer) or 'hint' (FitNets-style)
    """
    m = min(len(student_feats), len(teacher_feats))
    loss = 0.0
    for i in range(m):
        fs, ft = student_feats[i], teacher_feats[i].detach()
        # spatial align
        if fs.dim() == 2:  # safety for rare 2D heads
            fs = fs.unsqueeze(-1).unsqueeze(-1)
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1).unsqueeze(-1)
        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)

        if mode == "at":
            # build attention maps and compare
            am_s = attention_map(fs)  # [B,1,H*W]
            am_t = attention_map(ft)  # [B,1,H*W]
            loss = loss + F.mse_loss(am_s, am_t)
        else:  # hint
            # channel align with 1x1 conv projection (on-the-fly)
            if fs.shape[1] != ft.shape[1]:
                w = torch.empty((ft.shape[1], fs.shape[1], 1, 1), device=fs.device, dtype=fs.dtype)
                nn.init.kaiming_uniform_(w, a=1.0)
                fs = F.conv2d(fs, w)
            loss = loss + F.mse_loss(fs, ft)
    return loss / max(1, m)


# ------------------------ train / eval ------------------------
def train_one_epoch(student, teacher, loader, kd_loss, optimizer, device, task,
                    scaler=None, feat_mode: str = "none", lambda_feat: float = 0.0):
    student.train(); teacher.eval()
    running = 0.0

    use_feat = (lambda_feat > 0.0) and (feat_mode in {"at", "hint"})
    use_amp = (scaler is not None)

    for x, y in tqdm(loader, desc="Train(distill)", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                if use_feat:
                    with torch.no_grad():
                        t_logits, t_feats = teacher.forward_with_features(x)
                    s_logits, s_feats = student.forward_with_features(x)
                    loss = kd_loss(s_logits, t_logits, y)
                    loss_feat = feature_distill_loss(s_feats, t_feats, mode=feat_mode)
                    loss = loss + lambda_feat * loss_feat
                else:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    s_logits = student(x)
                    loss = kd_loss(s_logits, t_logits, y)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_feat:
                with torch.no_grad():
                    t_logits, t_feats = teacher.forward_with_features(x)
                s_logits, s_feats = student.forward_with_features(x)
                loss = kd_loss(s_logits, t_logits, y)
                loss_feat = feature_distill_loss(s_feats, t_feats, mode=feat_mode)
                loss = loss + lambda_feat * loss_feat
            else:
                with torch.no_grad():
                    t_logits = teacher(x)
                s_logits = student(x)
                loss = kd_loss(s_logits, t_logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        running += float(loss.item()) * x.size(0)

    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, task, n_classes, device, thresholds=None, compute_thresholds=False):
    model.eval()
    auroc, auprc = build_metrics(task, n_classes, device)
    losses, y_true_all, y_pred_all = 0.0, [], []
    if task in ("multi-class", "binary-class"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    probs_buf, targs_buf = [], []

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.squeeze().to(device)
        logits = model(x)

        if task == "multi-class":
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, y.long())
            y_pred = probs.argmax(1).detach().cpu()
            y_true = y.long().detach().cpu()
            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

        elif task == "binary-class":
            probs2 = torch.softmax(logits, dim=1)
            p1 = probs2[:, 1]
            loss = criterion(logits, y.long())
            y_pred = probs2.argmax(1).detach().cpu()
            y_true = y.long().detach().cpu()
            auroc.update(p1.to(device), y.to(device))
            auprc.update(p1.to(device), y.to(device))

        else:  # multi-label
            probs = torch.sigmoid(logits)
            loss = criterion(logits, y.float())

            if thresholds is None:
                y_pred = (probs > 0.5).long().detach().cpu()
            else:
                th = torch.tensor(thresholds, device=probs.device).view(1, -1)
                y_pred = (probs >= th).long().detach().cpu()
            y_true = y.long().detach().cpu()

            auroc.update(probs.to(device), y.to(device))
            auprc.update(probs.to(device), y.to(device))

            if compute_thresholds:
                probs_buf.append(probs.detach().cpu())
                targs_buf.append(y.float().detach().cpu())

        losses += float(loss.item()) * x.size(0)
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
    if task == "multi-label" and compute_thresholds and len(probs_buf) > 0:
        probs_all = torch.cat(probs_buf).numpy()
        targs_all = torch.cat(targs_buf).numpy()
        ths = optimal_thresholds(probs_all, targs_all, steps=100)
        preds_opt = (probs_all >= ths[None, :]).astype(int)
        out["f1_macro_opt"] = float(f1_score(targs_all, preds_opt, average="macro", zero_division=0))
        out["thresholds"] = ths.tolist()
    return out, ths


# ------------------------ Helpers ------------------------
def make_class_weights(train_loader, n_classes, mode: str, beta: float):
    """
    mode in {"inverse", "effective"}; returns tensor [C] or None
    """
    if mode not in {"inverse", "effective"}:
        return None
    pos = torch.zeros(n_classes)
    for _, y in train_loader:
        y = y.squeeze()
        if y.ndim == 1:
            continue
        pos += y.sum(dim=0)
    pos = pos.clamp(min=1.0)
    if mode == "inverse":
        w = (1.0 / pos)
        w = w / w.mean()
        return w
    else:
        n = pos
        w = (1 - beta) / (1 - beta ** n)
        w = w / w.mean()
        return w

def schedule_alpha_tau(sched: str, t: float, a_min: float, a_max: float, tau_min: float, tau_max: float):
    """
    t in [0,1]; sched: "kd2ce" (front-load KD), "ce2kd" (back-load KD), "const"
    """
    if sched == "kd2ce":
        a = a_max + (a_min - a_max) * t
        tau = tau_max + (tau_min - tau_max) * t
    elif sched == "ce2kd":
        a = a_min + (a_max - a_min) * t
        tau = tau_min
    else:
        a, tau = a_max, tau_max
    return float(a), float(tau)


# ------------------------ Main ------------------------
def main():
    ap = argparse.ArgumentParser()
    # data/protocol
    ap.add_argument("--dataset", type=str, default="chestmnist")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--aug", type=str, default="light", choices=["none", "light"])
    # models
    ap.add_argument("--teacher_backbone", type=str, default="resnet18")
    ap.add_argument("--student_backbone", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--teacher_ckpt", type=str, default="", help="optional pretrained teacher checkpoint")
    ap.add_argument("--pretrained_student", action="store_true")
    # KD hyper
    ap.add_argument("--sched", type=str, default="kd2ce", choices=["kd2ce", "ce2kd", "const"])
    ap.add_argument("--alpha_min", type=float, default=0.1)
    ap.add_argument("--alpha_max", type=float, default=0.4)
    ap.add_argument("--tau_min", type=float, default=3.0)
    ap.add_argument("--tau_max", type=float, default=5.0)
    ap.add_argument("--cw_kd", type=str, default="none", choices=["none", "inverse", "effective"])
    ap.add_argument("--cw_beta", type=float, default=0.99)
    # feature distill
    ap.add_argument("--feat", type=str, default="none", choices=["none", "at", "hint"])
    ap.add_argument("--lambda_feat", type=float, default=0.0)
    # opt/sched
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eta_min", type=float, default=1e-6)
    ap.add_argument("--early_patience", type=int, default=3)
    ap.add_argument("--warmup_ep", type=int, default=2)
    ap.add_argument("--selection_metric", type=str, default="auprc", choices=["auprc", "auroc"])
    ap.add_argument("--amp", action="store_true")
    # reproducibility & io
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    set_global_seed(args.seed); set_deterministic(True)

    exp_id = args.exp_name or (
        f"distill_{args.dataset}_{args.teacher_backbone}_to_{args.student_backbone}"
        f"_res{args.img_size}_sched{args.sched}_a{args.alpha_min}-{args.alpha_max}"
        f"_t{args.tau_min}-{args.tau_max}_cw{args.cw_kd}_feat{args.feat}"
        f"_ep{args.epochs}_seed{args.seed}"
    )
    exp_dir = os.path.join(args.outdir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    runlog_path = os.path.join(exp_dir, "run.log")
    metrics_path = os.path.join(exp_dir, "metrics.json")
    ths_path = os.path.join(exp_dir, "thresholds.json")
    ckpt_path = os.path.join(exp_dir, "ckpt.pt")
    test_json_path = os.path.join(exp_dir, "test.json")

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        name=args.dataset, batch_size=args.batch_size, img_size=args.img_size, augment=args.aug
    )
    raw_task = meta["task"]
    task = "multi-label" if "multi-label" in raw_task else raw_task
    n_classes = meta["n_classes"]

    # supervised pos_weight (multilabel only)
    pos_weight = None
    if task == "multi-label":
        pos_weight = estimate_pos_weight(train_loader, n_classes).to(device)

    # class-weighted KD vector
    kd_w = None
    if task == "multi-label" and args.cw_kd != "none":
        kd_w = make_class_weights(train_loader, n_classes, args.cw_kd, args.cw_beta).to(device)

    # teacher / student
    teacher = BaselineCNN(
        n_classes, backbone=args.teacher_backbone, pretrained=True,
        multi_label=(task == "multi-label")
    ).to(device)
    if args.teacher_ckpt and os.path.isfile(args.teacher_ckpt):
        sd = torch.load(args.teacher_ckpt, map_location=device)
        try:
            teacher.load_state_dict(sd, strict=False)
        except Exception:
            teacher.load_state_dict(sd, strict=True)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    student = BaselineCNN(
        n_classes, backbone=args.student_backbone, pretrained=args.pretrained_student,
        multi_label=(task == "multi-label")
    ).to(device)

    # KD loss / optimizer / scheduler
    kd_loss = KDCombinedLoss(task=task, alpha=args.alpha_max, tau=args.tau_max,
                             pos_weight=pos_weight, kd_class_weights=kd_w)
    optimizer = AdamW(student.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    # AMP scaler (CUDA only; MPS/CPU는 비활성)
    scaler = None
    if args.amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda', enabled=True)

    # training
    best_sel = -1.0
    best_val = None
    best_ths = None
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        # warm-up
        if ep <= max(1, args.warmup_ep):
            warm = ep / float(max(1, args.warmup_ep))
            for g in optimizer.param_groups:
                g["lr"] = args.lr * warm

        # α/τ schedule
        t = (ep - 1) / max(1, (args.epochs - 1))
        a_now, tau_now = schedule_alpha_tau(args.sched, t, args.alpha_min, args.alpha_max, args.tau_min, args.tau_max)
        kd_loss.alpha = a_now
        kd_loss.tau = tau_now

        t0 = time.time()
        tr_loss = train_one_epoch(
            student, teacher, train_loader,
            kd_loss, optimizer, device, task,
            scaler=scaler, feat_mode=args.feat, lambda_feat=float(args.lambda_feat)
        )

        val_metrics, ths = evaluate(
            student, val_loader, task, n_classes, device,
            compute_thresholds=(task == "multi-label"), thresholds=None
        )
        dt = round(time.time() - t0, 3)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr

        # epoch log
        epoch_log = {
            "epoch": ep,
            "train_loss": tr_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "alpha": a_now,
            "tau": tau_now,
            "lr": lr_now,
            "sec": dt,
        }
        with open(runlog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log, ensure_ascii=False) + "\n")
        print(json.dumps(epoch_log, ensure_ascii=False))

        sel = val_metrics[args.selection_metric]
        if sel > best_sel:
            best_sel = sel
            best_val = val_metrics
            best_ths = ths
            no_improve = 0
            torch.save(student.state_dict(), ckpt_path)
            if task == "multi-label" and best_ths is not None:
                with open(ths_path, "w", encoding="utf-8") as f:
                    json.dump({"thresholds": best_ths.tolist()}, f, ensure_ascii=False, indent=2)
        else:
            no_improve += 1
            if no_improve >= args.early_patience:
                break

    # test with frozen thresholds
    student.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics, _ = evaluate(
        student, test_loader, task, n_classes, device,
        compute_thresholds=False, thresholds=best_ths
    )
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    print("TEST:", json.dumps(test_metrics, ensure_ascii=False, indent=2))

    # summary
    summary = {
        "exp_id": os.path.basename(exp_dir),
        "dataset": args.dataset,
        "task": task,
        "n_classes": n_classes,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "augmentation": args.aug,
        "teacher_backbone": args.teacher_backbone,
        "student_backbone": args.student_backbone,
        "teacher_ckpt_used": bool(args.teacher_ckpt and os.path.isfile(args.teacher_ckpt)),
        "sched": args.sched,
        "alpha_min": args.alpha_min, "alpha_max": args.alpha_max,
        "tau_min": args.tau_min, "tau_max": args.tau_max,
        "cw_kd": args.cw_kd, "cw_beta": args.cw_beta,
        "feat": args.feat, "lambda_feat": args.lambda_feat,
        "optimizer": "AdamW", "lr": args.lr, "eta_min": args.eta_min,
        "selection_metric": args.selection_metric,
        "early_patience": args.early_patience,
        "amp": bool(args.amp),
        "seed": args.seed,
        "device": str(device),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "best_on_val": best_val,
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


if __name__ == "__main__":
    main()
