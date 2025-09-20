# src/cli.py
"""
Unified command-line entrypoint for the project.

Subcommands
-----------
1) train        : run supervised baseline training (wraps src/train.py)
2) distill      : run knowledge distillation training (wraps src/distill_train.py)
3) summarize    : aggregate JSON logs into a CSV at results/summary/runs.csv
4) ls-logs      : list discovered result logs for quick inspection

Examples
--------
# Supervised baseline (ChestMNIST, MobileNetV3-Small, 160px)
python -m src.cli train \
  --dataset chestmnist --img-size 160 --batch-size 64 --epochs 12 \
  --backbone mobilenetv3_small_100 --aug light --use-pos-weight --seed 0

# Distillation (ResNet-18 -> MobileNetV3-Small, 128px)
python -m src.cli distill \
  --dataset chestmnist --img-size 128 --batch-size 64 --epochs 12 \
  --teacher-backbone resnet18 --student-backbone mobilenetv3_small_100 \
  --alpha 0.1 --tau 5.0 --selection-metric auprc --seed 0

# Summarize test logs into CSV
python -m src.cli summarize
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import runpy
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Local utils (no extra deps)
try:
    from utils.seed import seed_everything
except Exception:
    seed_everything = None  # graceful fallback


# -------------------------------
# Helpers
# -------------------------------
def _echo(msg: str) -> None:
    print(f"[cli] {msg}", flush=True)


def _as_bool(v: bool) -> str:
    return "true" if v else "false"


def _cwd_root() -> str:
    return os.path.abspath(os.getcwd())


def _ensure_dirs() -> None:
    for p in [
        "results",
        os.path.join("results", "logs"),
        os.path.join("results", "checkpoints"),
        os.path.join("results", "summary"),
        os.path.join("results", "figures"),
        os.path.join("results", "thresholds"),
    ]:
        os.makedirs(p, exist_ok=True)


def _run_module_as_main(mod: str, argv: List[str]) -> None:
    """
    Run a module's __main__ with a temporary sys.argv.
    This allows us to reuse src/train.py and src/distill_train.py without refactoring.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [mod] + argv
        _echo(f"exec: python -m {mod} " + " ".join(argv))
        # Note: run_module will execute if __name__ == '__main__' block in the target
        runpy.run_module(mod, run_name="__main__")
    finally:
        sys.argv = old_argv


def _default_outdir() -> str:
    return "results"


# -------------------------------
# Subcommand: train (supervised)
# -------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    _ensure_dirs()
    # Seeding
    if seed_everything is not None and args.seed is not None:
        info = seed_everything(args.seed)
        _echo(f"seed set: {json.dumps(info)}")
    else:
        _echo("seed not set (seed module not found or seed=None)")

    # Map high-level args -> src/train.py CLI
    argv = [
        f"--dataset={args.dataset}",
        f"--img_size={args.img_size}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--backbone={args.backbone}",
        f"--outdir={args.outdir}",
        f"--aug={'light' if args.aug == 'light' else 'none'}",
        f"--patience={args.patience}",
    ]
    if args.pretrained:
        argv.append("--pretrained")
    if args.use_pos_weight:
        argv.append("--use_pos_weight")

    # Dispatch
    _run_module_as_main("src.train", argv)


# -------------------------------
# Subcommand: distill
# -------------------------------
def cmd_distill(args: argparse.Namespace) -> None:
    _ensure_dirs()
    # Seeding
    if seed_everything is not None and args.seed is not None:
        info = seed_everything(args.seed)
        _echo(f"seed set: {json.dumps(info)}")
    else:
        _echo("seed not set (seed module not found or seed=None)")

    argv = [
        f"--dataset={args.dataset}",
        f"--img_size={args.img_size}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--teacher_backbone={args.teacher_backbone}",
        f"--student_backbone={args.student_backbone}",
        f"--alpha={args.alpha}",
        f"--tau={args.tau}",
        f"--outdir={args.outdir}",
        f"--selection_metric={args.selection_metric}",
        f"--num_workers={args.num_workers}",
        f"--ema_decay={args.ema_decay}",
        f"--warmup_ep={args.warmup_ep}",
        f"--early_patience={args.early_patience}",
    ]
    if args.pretrained:
        argv.append("--pretrained")
    if args.teacher_ckpt:
        argv.append(f"--teacher_ckpt={args.teacher_ckpt}")

    _run_module_as_main("src.distill_train", argv)


# -------------------------------
# Subcommand: summarize logs -> CSV
# -------------------------------
def _parse_test_log_name(path: str) -> Dict[str, str]:
    """
    Parse filename patterns like:
      - chestmnist_resnet18_test.json
      - distill_chestmnist_resnet18_to_mobilenetv3_small_100_test.json
    """
    fn = os.path.basename(path)
    rec: Dict[str, str] = {"task": "", "dataset": "", "teacher": "", "student": ""}

    # distill pattern
    m = re.match(r"distill_(?P<dataset>[^_]+)_(?P<teacher>.+?)_to_(?P<student>.+?)_test\.json$", fn)
    if m:
        rec["task"] = "distill"
        rec["dataset"] = m.group("dataset")
        rec["teacher"] = m.group("teacher")
        rec["student"] = m.group("student")
        return rec

    # baseline pattern
    m = re.match(r"(?P<dataset>[^_]+)_(?P<backbone>.+?)_test\.json$", fn)
    if m:
        rec["task"] = "supervised"
        rec["dataset"] = m.group("dataset")
        rec["student"] = m.group("backbone")
        return rec

    rec["task"] = "unknown"
    rec["dataset"] = "unknown"
    return rec


def _read_json_safely(path: str) -> Optional[Dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        _echo(f"warn: failed to read {path}: {e}")
        return None


def cmd_summarize(args: argparse.Namespace) -> None:
    _ensure_dirs()
    logs_glob = os.path.join("results", "logs", "*_test.json")
    paths = sorted(glob.glob(logs_glob))
    if not paths:
        _echo("no *_test.json found under results/logs")
        return

    rows: List[Dict[str, str]] = []
    for p in paths:
        meta = _parse_test_log_name(p)
        data = _read_json_safely(p)
        if not data:
            continue

        rows.append(
            {
                "timestamp": datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
                "file": os.path.basename(p),
                "task": meta.get("task", ""),
                "dataset": meta.get("dataset", ""),
                "teacher": meta.get("teacher", ""),
                "student": meta.get("student", ""),
                "loss": f"{data.get('loss', float('nan')):.6f}",
                "auprc": f"{data.get('auprc', float('nan')):.6f}",
                "auroc": f"{data.get('auroc', float('nan')):.6f}",
                "f1_macro": f"{data.get('f1_macro', float('nan')):.6f}",
                "f1_macro_opt": f"{data.get('f1_macro_opt', float('nan')):.6f}"
                if "f1_macro_opt" in data
                else "",
            }
        )

    out_csv = os.path.join("results", "summary", "runs.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "file",
                "task",
                "dataset",
                "teacher",
                "student",
                "loss",
                "auprc",
                "auroc",
                "f1_macro",
                "f1_macro_opt",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    _echo(f"wrote summary: {out_csv} ({len(rows)} rows)")


# -------------------------------
# Subcommand: list logs
# -------------------------------
def cmd_ls_logs(args: argparse.Namespace) -> None:
    logs_glob = os.path.join("results", "logs", "*_test.json")
    for p in sorted(glob.glob(logs_glob)):
        meta = _parse_test_log_name(p)
        _echo(f"{os.path.basename(p)} | task={meta.get('task')} dataset={meta.get('dataset')} "
              f"teacher={meta.get('teacher','-')} student={meta.get('student','-')}")


# -------------------------------
# Argparse wiring
# -------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description="Project CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="run supervised training (wraps src/train.py)")
    pt.add_argument("--dataset", type=str, default="chestmnist")
    pt.add_argument("--img-size", type=int, default=160)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--epochs", type=int, default=12)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    pt.add_argument("--pretrained", action="store_true")
    pt.add_argument("--outdir", type=str, default=_default_outdir())
    pt.add_argument("--aug", type=str, default="light", choices=["none", "light"])
    pt.add_argument("--use-pos-weight", action="store_true")
    pt.add_argument("--patience", type=int, default=3)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    # distill
    pd = sub.add_parser("distill", help="run knowledge distillation (wraps src/distill_train.py)")
    pd.add_argument("--dataset", type=str, default="chestmnist")
    pd.add_argument("--img-size", type=int, default=128)
    pd.add_argument("--batch-size", type=int, default=64)
    pd.add_argument("--epochs", type=int, default=12)
    pd.add_argument("--lr", type=float, default=3e-4)
    pd.add_argument("--teacher-backbone", type=str, default="resnet18")
    pd.add_argument("--student-backbone", type=str, default="mobilenetv3_small_100")
    pd.add_argument("--alpha", type=float, default=0.1)
    pd.add_argument("--tau", type=float, default=5.0)
    pd.add_argument("--pretrained", action="store_true")
    pd.add_argument("--outdir", type=str, default=_default_outdir())
    pd.add_argument("--teacher-ckpt", type=str, default="")
    pd.add_argument("--selection-metric", type=str, default="auprc", choices=["auprc", "auroc"])
    pd.add_argument("--num-workers", type=int, default=4)
    pd.add_argument("--ema-decay", type=float, default=0.999)
    pd.add_argument("--warmup-ep", type=int, default=3)
    pd.add_argument("--early-patience", type=int, default=3)
    pd.add_argument("--seed", type=int, default=0)
    pd.set_defaults(func=cmd_distill)

    # summarize
    ps = sub.add_parser("summarize", help="aggregate logs into results/summary/runs.csv")
    ps.set_defaults(func=cmd_summarize)

    # list logs
    pll = sub.add_parser("ls-logs", help="list available *_test.json under results/logs")
    pll.set_defaults(func=cmd_ls_logs)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
