#!/usr/bin/env python
# scripts/plot_curves.py
"""
Plot training curves, per-class thresholds, PR/ROC curves, and reliability diagrams.

Inputs (all optional; provide what you have):
  --run-log        : JSONL file with epoch-wise records (e.g., results/seed_runs/*/run.log)
                     expected keys: val_auprc, val_auroc, val_f1_macro, val_f1_macro_opt (optional), epoch
  --thresholds     : thresholds.json produced at validation (list of per-class thresholds)
  --preds          : predictions file to draw PR/ROC/ECE; formats:
                        - NPZ with arrays: 'probs' [N,C], 'targets' [N,C] (multilabel) or [N] (multiclass/binary)
                        - JSON with keys:  'probs', 'targets' (same shapes as above)
                        - CSV with columns like p_c0,...,p_c{C-1}, y_c0,...,y_c{C-1} (multilabel)
  --class-names    : optional text file with one class name per line (for thresholds bar labels)
  --outdir         : output directory for figures (default: results/figures)
  --title          : title prefix for plots
  --bins           : number of bins for ECE (default: 15)
  --no-show        : do not display, only save

Examples:
  python scripts/plot_curves.py --run-log results/seed_runs/kd_.../run.log \
    --thresholds results/thresholds/kd_...json --outdir results/figures

  python scripts/plot_curves.py --preds results/preds/test_preds.npz --title "ChestMNIST (test)"
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless; will still show if --no-show not set and backend allows
import matplotlib.pyplot as plt

# Optional: sklearn for curves and metrics
try:
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        roc_curve,
        auc,
    )
    _SK_OK = True
except Exception:
    _SK_OK = False


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str):
    xs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                xs.append(json.loads(line))
            except Exception:
                # allow lines that start with 'TEST:' etc.
                if line.startswith("{") and line.endswith("}"):
                    xs.append(json.loads(line))
    return xs


def load_thresholds(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    # allow raw list or wrapped dict
    if isinstance(data, dict) and "thresholds" in data:
        arr = np.array(data["thresholds"], dtype=float)
    else:
        arr = np.array(data, dtype=float)
    return arr


def load_class_names(path: Optional[str], C: int) -> list:
    if path is None or not os.path.isfile(path):
        return [f"c{idx}" for idx in range(C)]
    with open(path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != C:
        # pad or trim
        names = (names + [f"c{idx}" for idx in range(C)])[:C]
    return names


def try_load_preds(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probs, targets).

    - NPZ: expects arrays 'probs' and 'targets'
    - JSON: expects keys 'probs' and 'targets'
    - CSV: expects columns p_c0..p_c{C-1} and y_c0..y_c{C-1} (multilabel),
           or p, y for binary; flexible but must be well-labeled.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        d = np.load(path)
        probs = d["probs"]
        targets = d["targets"]
        return probs, targets
    if ext == ".json":
        with open(path, "r") as f:
            d = json.load(f)
        return np.array(d["probs"]), np.array(d["targets"])
    if ext == ".csv":
        import csv
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("Empty CSV")
        # detect keys
        keys = rows[0].keys()
        p_keys = sorted([k for k in keys if k.startswith("p_c")])
        y_keys = sorted([k for k in keys if k.startswith("y_c")])
        if p_keys and y_keys and len(p_keys) == len(y_keys):
            C = len(p_keys)
            P = np.zeros((len(rows), C), dtype=float)
            Y = np.zeros((len(rows), C), dtype=float)
            for i, r in enumerate(rows):
                for c in range(C):
                    P[i, c] = float(r[p_keys[c]])
                    Y[i, c] = float(r[y_keys[c]])
            return P, Y
        # try binary as 'p' and 'y'
        if "p" in keys and "y" in keys:
            P = np.array([float(r["p"]) for r in rows], dtype=float)
            Y = np.array([float(r["y"]) for r in rows], dtype=float)
            return P, Y
        raise ValueError("CSV must contain p_c*, y_c* (multilabel) or p,y (binary)")
    raise ValueError(f"Unsupported predictions file extension: {ext}")


# ---------- Plot helpers ----------

def plot_training_curves(logrecs, title: str, outdir: str, no_show: bool):
    if not logrecs:
        print("[plot_curves] No log records to plot.")
        return
    epochs = [r.get("epoch", i+1) for i, r in enumerate(logrecs)]
    keys = [("val_auprc", "Val AUPRC"),
            ("val_auroc", "Val AUROC"),
            ("val_f1_macro", "Val F1 (macro)"),
            ("val_f1_macro_opt", "Val F1 (macro, opt)")]
    for k, label in keys:
        vals = [r[k] for r in logrecs if k in r]
        es   = [r.get("epoch", i+1) for i, r in enumerate(logrecs) if k in r]
        if not vals:
            continue
        plt.figure(figsize=(6.4, 4.0))
        plt.plot(es, vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(label)
        plt.title(f"{title} – {label}" if title else label)
        plt.grid(True, alpha=0.3)
        _ensure_outdir(outdir)
        out = os.path.join(outdir, f"curve_{k}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        if not no_show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close()
        print(f"[plot_curves] saved: {out}")


def plot_thresholds_bar(ths: np.ndarray, class_names: list, title: str, outdir: str, no_show: bool):
    C = len(ths)
    x = np.arange(C)
    plt.figure(figsize=(max(6.4, C * 0.4 + 2), 4.0))
    plt.bar(x, ths, width=0.8)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Threshold")
    ttl = "Per-class thresholds"
    if title:
        ttl = f"{title} – {ttl}"
    plt.title(ttl)
    plt.grid(axis="y", alpha=0.3)
    _ensure_outdir(outdir)
    out = os.path.join(outdir, "thresholds_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    if not no_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    print(f"[plot_curves] saved: {out}")


def _as_multilabel(probs, targets) -> bool:
    # probs [N,C], targets [N,C] -> multilabel; probs [N], targets [N] -> binary
    return probs.ndim == 2 and targets.ndim == 2


def plot_pr_roc(probs: np.ndarray, targets: np.ndarray, title: str, outdir: str, no_show: bool):
    if not _SK_OK:
        print("[plot_curves] sklearn not available; skip PR/ROC plots.")
        return

    _ensure_outdir(outdir)
    multilabel = _as_multilabel(probs, targets)

    # PR curve
    plt.figure(figsize=(6.4, 4.0))
    if multilabel:
        # micro-average PR
        P = probs.ravel()
        Y = targets.ravel()
        prec, rec, _ = precision_recall_curve(Y, P)
        ap_micro = average_precision_score(Y, P)
        plt.plot(rec, prec, label=f"micro-PR (AP={ap_micro:.3f})")
    else:
        prec, rec, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        plt.plot(rec, prec, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} – PR curve" if title else "PR curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    out = os.path.join(outdir, "pr_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    if not no_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    print(f"[plot_curves] saved: {out}")

    # ROC curve
    plt.figure(figsize=(6.4, 4.0))
    if multilabel:
        # micro-average ROC
        fpr, tpr, _ = roc_curve(targets.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"micro-ROC (AUC={roc_auc:.3f})")
    else:
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title} – ROC curve" if title else "ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    out = os.path.join(outdir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    if not no_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    print(f"[plot_curves] saved: {out}")


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """
    ECE for binary or multilabel (micro-averaged over all labels).
    probs: [N] or [N,C] in [0,1]
    targets: same shape with {0,1}
    """
    p = probs.ravel()
    y = targets.ravel()
    n = len(p)
    # equal-frequency bins: sort by confidence
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    # split into bins with roughly equal counts
    edges = np.linspace(0, n, bins + 1, dtype=int)
    ece = 0.0
    for i in range(bins):
        s, t = edges[i], edges[i + 1]
        if t <= s:
            continue
        conf = p_sorted[s:t].mean()
        acc = y_sorted[s:t].mean()
        w = (t - s) / n
        ece += w * abs(acc - conf)
    return float(ece)


def plot_reliability(probs: np.ndarray, targets: np.ndarray, bins: int, title: str, outdir: str, no_show: bool):
    # micro-averaged reliability diagram
    p = probs.ravel()
    y = targets.ravel()

    # equal-frequency bins (to match our ECE calc)
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(p_sorted)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    confs, accs = [], []
    for i in range(bins):
        s, t = edges[i], edges[i + 1]
        if t <= s:
            continue
        confs.append(float(p_sorted[s:t].mean()))
        accs.append(float(y_sorted[s:t].mean()))

    ece = expected_calibration_error(probs, targets, bins=bins)

    plt.figure(figsize=(6.4, 4.0))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    plt.scatter(confs, accs, s=18, label=f"empirical (ECE={ece:.3f})")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} – Reliability diagram" if title else "Reliability diagram")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    _ensure_outdir(outdir)
    out = os.path.join(outdir, "reliability.png")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    if not no_show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    print(f"[plot_curves] saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-log", type=str, default="", help="path to JSONL epoch-wise log (run.log)")
    ap.add_argument("--thresholds", type=str, default="", help="path to thresholds.json")
    ap.add_argument("--preds", type=str, default="", help="path to predictions file (npz/json/csv)")
    ap.add_argument("--class-names", type=str, default="", help="optional text file with class names (one per line)")
    ap.add_argument("--bins", type=int, default=15, help="bins for ECE/reliability")
    ap.add_argument("--outdir", type=str, default="results/figures")
    ap.add_argument("--title", type=str, default="", help="title prefix for figures")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    _ensure_outdir(args.outdir)

    # 1) training curves
    if args.run_log and os.path.isfile(args.run_log):
        logs = load_jsonl(args.run_log)
        plot_training_curves(logs, args.title, args.outdir, args.no_show)
    else:
        if args.run_log:
            print(f"[plot_curves] run-log not found: {args.run_log}")

    # 2) thresholds bar
    if args.thresholds and os.path.isfile(args.thresholds):
        ths = load_thresholds(args.thresholds)
        names = load_class_names(args.class_names, len(ths))
        plot_thresholds_bar(ths, names, args.title, args.outdir, args.no_show)
    else:
        if args.thresholds:
            print(f"[plot_curves] thresholds file not found: {args.thresholds}")

    # 3) PR/ROC + Reliability (if predictions are provided)
    if args.preds and os.path.isfile(args.preds):
        try:
            probs, targets = try_load_preds(args.preds)
            # sanity clamp
            probs = np.clip(probs, 0.0, 1.0)
            targets = (targets > 0.5).astype(float)
            plot_pr_roc(probs, targets, args.title, args.outdir, args.no_show)
            if _SK_OK:
                plot_reliability(probs, targets, bins=args.bins, title=args.title, outdir=args.outdir, no_show=args.no_show)
            else:
                print("[plot_curves] sklearn missing; skip reliability.")
        except Exception as e:
            print(f"[plot_curves] failed to load preds: {e}")
    else:
        if args.preds:
            print(f"[plot_curves] preds file not found: {args.preds}")

    print("[plot_curves] done.")


if __name__ == "__main__":
    main()
