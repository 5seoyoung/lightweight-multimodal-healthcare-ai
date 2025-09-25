# scripts/calibration_and_thresholds.py
import os, argparse, json, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# 프로젝트 로컬 임포트 (PYTHONPATH=. 필요)
from src.datasets.medmnist_loader import get_medmnist_loaders
from src.models.baseline_cnn import BaselineCNN
from src.utils.thresholds import optimal_thresholds

def reliability_points(probs, targets, bins=10, quantile=False):
    """
    Multilabel reliability: x=confidence, y=empirical positive rate.
    probs, targets: [N, C] numpy.
    """
    conf = probs.flatten()
    pos = (targets == 1).astype(np.float32).flatten()

    if quantile:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = 0.0, 1.0  # 안정화
    else:
        edges = np.linspace(0, 1, bins + 1)

    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(pos[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns), edges

def ece_from_points(xs, ys, ns, total):
    """Expected Calibration Error from binned points."""
    if len(xs) == 0:
        return float("nan")
    w = ns.astype(np.float64) / float(total)
    return float(np.sum(np.abs(xs - ys) * w))

def load_threshold_list(path, n_classes):
    with open(path, "r") as f:
        obj = json.load(f)
    # 호환: [..] 또는 {"thresholds":[..]} 둘 다 지원
    ths = obj.get("thresholds", obj)
    ths = np.array(ths, dtype=np.float32)
    if ths.ndim != 1 or len(ths) != n_classes:
        raise ValueError(f"Loaded thresholds shape mismatch: {ths.shape} vs n_classes={n_classes}")
    return ths

def save_threshold_list(path, ths):
    obj = {"thresholds": [float(x) for x in ths.tolist()]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[OK] saved per-class thresholds -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--quantilebins", action="store_true")
    ap.add_argument("--save_thresholds", type=str, default=None)
    ap.add_argument("--load_thresholds", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    _, val_loader, test_loader, meta = get_medmnist_loaders(
        args.dataset, batch_size=128, img_size=args.img_size, augment="none"
    )
    loader = val_loader if args.split == "val" else test_loader
    n_classes = meta["n_classes"]

    # 모델 & 체크포인트
    model = BaselineCNN(n_classes, backbone=args.backbone, pretrained=False, multi_label=True).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        model.load_state_dict(sd, strict=True)
    model.eval()

    # 예측 수집
    all_probs, all_targs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.squeeze().float()
            p = torch.sigmoid(model(x))
            all_probs.append(p.cpu())
            all_targs.append(y)
    probs = torch.cat(all_probs).numpy()
    targs = torch.cat(all_targs).numpy()

    # 임계값 결정: 로드가 있으면 사용, 없으면 현재 split에서 계산
    if args.load_thresholds is not None:
        ths = load_threshold_list(args.load_thresholds, n_classes)
        msg_th = f"[LOADED thresholds] {os.path.basename(args.load_thresholds)}"
    else:
        ths = optimal_thresholds(probs, targs, steps=100)
        msg_th = "[OPTIMIZED thresholds on current split]"
        if args.save_thresholds:
            save_threshold_list(args.save_thresholds, ths)

    preds = (probs >= ths[None, :]).astype(int)
    f1m = f1_score(targs, preds, average="macro", zero_division=0)

    xs, ys, ns, edges = reliability_points(
        probs, targs, bins=args.bins, quantile=args.quantilebins
    )
    ece = ece_from_points(xs, ys, ns, total=probs.size)

    alias = os.path.basename(os.path.dirname(args.ckpt))
    fig_out = f"results/reliability_{alias}_{args.split}.png"
    os.makedirs("results", exist_ok=True)

    # Plot
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Positive Rate")
    plt.title(f"Reliability ({args.split})")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200)

    if args.quantilebins:
        qs = np.linspace(0, 1, args.bins + 1)
        qvals = np.quantile(probs.flatten(), qs).round(3).tolist()
        print(f"[Probs quantiles] {qvals} (N={probs.size})")

    print(f"{msg_th}")
    print(f"[SPLIT] {args.split} | F1_macro@per-class-th = {f1m:.4f} | ECE = {ece:.4f}")
    print(f"Per-class thresholds: {np.round(ths, 3).tolist()}")
    print(f"[OK] saved {fig_out}")

if __name__ == "__main__":
    main()
