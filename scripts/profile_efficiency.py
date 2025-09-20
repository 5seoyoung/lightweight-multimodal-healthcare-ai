#!/usr/bin/env python
# scripts/profile_efficiency.py
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# repo local import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.baseline_cnn import BaselineCNN  # noqa: E402

# Optional deps
try:
    import psutil  # for RSS on CPU/MPS
except Exception:
    psutil = None

# try THOP for FLOPs
_THOP_OK = True
try:
    from thop import profile as thop_profile  # type: ignore
except Exception:
    _THOP_OK = False


def get_devices(selection: str):
    selection = selection.lower()
    available = []
    if selection in ("auto", "all"):
        available.append("cpu")
        if torch.backends.mps.is_available():  # Apple Silicon
            available.append("mps")
        if torch.cuda.is_available():
            available.append("cuda")
    else:
        for d in selection.split(","):
            d = d.strip()
            if d == "cpu":
                available.append("cpu")
            elif d == "mps" and torch.backends.mps.is_available():
                available.append("mps")
            elif d == "cuda" and torch.cuda.is_available():
                available.append("cuda")
    # de-dup, keep order
    out = []
    for d in available:
        if d not in out:
            out.append(d)
    return out


def param_count_m(model: torch.nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


@torch.no_grad()
def measure_flops_gmacs(model, img_size: int, device: str) -> float:
    """Return GMACs if THOP is available, otherwise NaN."""
    if not _THOP_OK:
        return float("nan")
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    if device in ("cuda", "mps"):
        dummy = dummy.to(device)
    # thop returns macs (multiply-adds) and params
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    # convert to GMACs
    return float(macs / 1e9)


def _synchronize(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


@torch.no_grad()
def measure_latency_ms(model, img_size: int, device: str, iters: int, warmup: int):
    """Return (mean_ms, std_ms) excluding warmup iterations."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    if device in ("cuda", "mps"):
        x = x.to(device)

    # warmup
    for _ in range(max(0, warmup)):
        _ = model(x)
    _synchronize(device)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        _synchronize(device)
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)

    arr = np.asarray(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)


@torch.no_grad()
def measure_peak_memory(model, img_size: int, device: str, iters: int = 20):
    """
    CUDA: bytes via torch.cuda.max_memory_allocated (precise).
    MPS/CPU: fall back to process RSS delta if psutil is available, else NaN.
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    if device in ("cuda", "mps"):
        x = x.to(device)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(iters):
            _ = model(x)
        _synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated()
        return int(peak_bytes), "bytes (cuda_alloc)"
    else:
        # Approximate via RSS delta
        if psutil is None:
            return None, "unavailable (psutil not installed)"
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss
        for _ in range(iters):
            _ = model(x)
        _synchronize(device)
        rss_after = proc.memory_info().rss
        return int(max(0, rss_after - rss_before)), "bytes (rss_delta)"


def build_model(backbone: str, n_classes: int, multi_label: bool, pretrained: bool, device: str):
    model = BaselineCNN(n_classes=n_classes, backbone=backbone, pretrained=pretrained, multi_label=multi_label)
    if device in ("cuda", "mps"):
        model = model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="chestmnist", help="used to decide n_classes if --n-classes not set")
    ap.add_argument("--n-classes", type=int, default=None, help="override class count; if None, infer from dataset")
    ap.add_argument("--multi-label", action="store_true", help="set if the dataset is multilabel")
    ap.add_argument("--backbone", type=str, default="mobilenetv3_small_100")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--devices", type=str, default="auto", help="cpu,mps,cuda or 'auto'")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="results/summary")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # infer n_classes when not provided
    n_classes = args.n_classes
    if n_classes is None:
        try:
            from medmnist import INFO  # lightweight import; no download
            info = INFO[args.dataset]
            n_classes = int(info.get("n_classes", 1))
            # basic heuristic: ChestMNIST is multilabel
            if "multi-label" in info.get("task", ""):
                args.multi_label = True
        except Exception:
            print(json.dumps({"warn": "failed_to_infer_n_classes_from_medmnist_INFO; set --n-classes"}))
            n_classes = 14  # safe fallback for ChestMNIST

    devices = get_devices(args.devices)
    results = {
        "dataset": args.dataset,
        "n_classes": n_classes,
        "multi_label": bool(args.multi_label),
        "backbone": args.backbone,
        "img_size": args.img_size,
        "pretrained": bool(args.pretrained),
        "iters": args.iters,
        "warmup": args.warmup,
        "env": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "thop_available": _THOP_OK,
            "psutil_available": psutil is not None,
        },
        "per_device": {},
    }

    for dev in devices:
        model = build_model(
            backbone=args.backbone,
            n_classes=n_classes,
            multi_label=args.multi_label,
            pretrained=args.pretrained,
            device=dev,
        )

        # FLOPs
        flops_gmacs = measure_flops_gmacs(model, args.img_size, dev)

        # latency
        mean_ms, std_ms = measure_latency_ms(model, args.img_size, dev, args.iters, args.warmup)

        # memory
        peak_bytes, mem_note = measure_peak_memory(model, args.img_size, dev)

        # pack
        results["per_device"][dev] = {
            "param_M": round(param_count_m(model), 3),
            "flops_GMACs": (None if np.isnan(flops_gmacs) else round(flops_gmacs, 3)),
            "latency_ms_mean": round(mean_ms, 3),
            "latency_ms_sd": round(std_ms, 3),
            "peak_memory_bytes": peak_bytes,
            "peak_memory_note": mem_note,
        }

        # cleanups
        del model
        if dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # write file
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(
        args.outdir,
        f"efficiency_{args.dataset}_{args.backbone}_{args.img_size}_{ts}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"saved": out_path}, indent=2))
    # also echo a short table
    for dev, rec in results["per_device"].items():
        print(
            f"[{dev}] params(M)={rec['param_M']} | FLOPs(G)={rec['flops_GMACs']} | "
            f"latency(ms)={rec['latency_ms_mean']}±{rec['latency_ms_sd']} | "
            f"peak_mem(bytes)={rec['peak_memory_bytes']} ({rec['peak_memory_note']})"
        )


if __name__ == "__main__":
    main()
