# scripts/collect_metrics.py
import os, json, time, argparse
import torch
from glob import glob
from pathlib import Path

# optional FLOPs
def try_flops(model, img_size):
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
        model.eval()
        with torch.no_grad():
            x = torch.randn(1,3,img_size,img_size)
            flops = FlopCountAnalysis(model, x).total()
        return flops
    except Exception:
        return None

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def measure_latency(model, img_size, device, iters=50, warmup=10):
    model.eval().to(device)
    x = torch.randn(1,3,img_size,img_size, device=device)
    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    # measure
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms

def load_student(backbone, n_classes, pretrained):
    from src.models.baseline_cnn import BaselineCNN
    return BaselineCNN(n_classes=n_classes, backbone=backbone, pretrained=pretrained, multi_label=True)

ap = argparse.ArgumentParser()
ap.add_argument("--results_dir", default="results", type=str)
ap.add_argument("--out_csv", default="results/aggregate.csv", type=str)
args = ap.parse_args()

rows = []
for mpath in glob(os.path.join(args.results_dir, "**/metrics.json"), recursive=True):
    with open(mpath, "r") as f:
        meta = json.load(f)
    # basic
    dataset = meta.get("dataset")
    img_size = int(meta.get("img_size", 128))
    n_classes = int(meta.get("n_classes", 14))
    student_backbone = meta.get("student_backbone", meta.get("backbone", ""))
    teacher_backbone = meta.get("teacher_backbone", "")
    sched = meta.get("sched", "sup")
    cw_kd = meta.get("cw_kd", "none")
    pretrained_student = meta.get("pretrained_student", meta.get("pretrained", False))
    seed = meta.get("seed", None)

    # test metrics
    test = meta.get("test", {})
    auroc = test.get("auroc")
    auprc = test.get("auprc")

    # build student model to get params/flops/latency
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    try:
        model = load_student(student_backbone, n_classes, bool(pretrained_student))
        # ckpt가 있으면 로드
        ckpt_rel = meta.get("artifacts", {}).get("checkpoint", None)
        if ckpt_rel:
            ckpt_path = os.path.join(args.results_dir, ckpt_rel) if not os.path.isabs(ckpt_rel) else ckpt_rel
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location="cpu")
                try:
                    model.load_state_dict(sd, strict=False)
                except:
                    pass
        params = count_params(model)
        flops = try_flops(model, img_size)
        latency_ms = measure_latency(model, img_size, device)
    except Exception as e:
        params, flops, latency_ms = None, None, None

    rows.append(dict(
        file=mpath, dataset=dataset, img_size=img_size, seed=seed,
        teacher=teacher_backbone, student=student_backbone,
        sched=sched, cw_kd=cw_kd, pretrained_student=pretrained_student,
        auroc=auroc, auprc=auprc,
        params=params, flops=flops, latency_ms=latency_ms
    ))

# save CSV
import csv
Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
with open(args.out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"[OK] wrote {args.out_csv} ({len(rows)} rows)")
