# scripts/calibration_and_thresholds.py
import argparse, json, os, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.datasets.medmnist_loader import get_medmnist_loaders
from src.models.baseline_cnn import BaselineCNN
from src.utils.thresholds import optimal_thresholds
from torch.utils.data import DataLoader

def multilabel_ece(probs, targets, bins=15, quantile=False):
    """ECE for multilabel: sum_k p(B_k)*|mean(conf)-mean(target)|."""
    conf = probs.flatten()
    targ = targets.flatten()
    if quantile:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(conf, qs)
        # 중복 edge 방지(동일값이 많은 경우)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
        # 너무 중복이면 fallback
        if len(edges) < 3:
            edges = np.linspace(0, 1, bins + 1)
    else:
        edges = np.linspace(0, 1, bins + 1)

    ece, accs, confs = 0.0, [], []
    for i in range(len(edges)-1):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum() == 0: 
            continue
        avg_conf = conf[m].mean()
        pos_rate = targ[m].mean()  # 실측 양성률
        ece += abs(avg_conf - pos_rate) * (m.mean())
        accs.append(pos_rate)
        confs.append(avg_conf)
    return float(ece), np.array(confs), np.array(accs)

def reliability_plot(confs, pos_rates, title, outp):
    if len(confs) == 0:
        print("[WARN] No non-empty bins; plot skipped.")
        return
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],'--')
    plt.plot(confs, pos_rates, marker='o')
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    print(f"[OK] saved {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--quantilebins", action="store_true",
                    help="Use quantile bins (권장).")
    ap.add_argument("--save_thresholds", default="results/thresholds.json")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 로더 받아오고, 멀티프로세싱 이슈 회피 위해 재래핑(num_workers=0)
    _, val_loader, test_loader, meta = get_medmnist_loaders(
        args.dataset, batch_size=128, img_size=args.img_size, augment="none")
    base_loader = val_loader if args.split == "val" else test_loader
    loader = DataLoader(base_loader.dataset, batch_size=128,
                        shuffle=False, num_workers=0, pin_memory=False)

    n_classes = meta["n_classes"]
    model = BaselineCNN(n_classes, backbone=args.backbone, pretrained=False, multi_label=True).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    try:
        model.load_state_dict(sd, strict=False)
    except:
        model.load_state_dict(sd, strict=True)
    model.eval()

    all_probs, all_targs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.squeeze().float()
            p = torch.sigmoid(model(x))
            all_probs.append(p.cpu()); all_targs.append(y)
    probs = torch.cat(all_probs).numpy()
    targs = torch.cat(all_targs).numpy()

    # --- thresholds ---
    if args.split == "val":
        ths = optimal_thresholds(probs, targs, steps=100)  # per-class
        with open(args.save_thresholds, "w") as f:
            json.dump({"thresholds": ths.tolist()}, f, indent=2)
        print("[OK] saved per-class thresholds ->", args.save_thresholds)
    else:
        ths = np.full((probs.shape[1],), 0.5, dtype=np.float32)
        if os.path.exists(args.save_thresholds):
            ths = np.array(json.load(open(args.save_thresholds))["thresholds"], dtype=np.float32)
            print("[OK] loaded thresholds from", args.save_thresholds)

    preds = (probs >= ths[None, :]).astype(int)
    f1m = f1_score(targs, preds, average="macro", zero_division=0)

    # --- ECE + Reliability (multilabel; pos-rate 기반) ---
    ece, confs, pos_rates = multilabel_ece(probs, targs, bins=args.bins, quantile=args.quantilebins)

    print(f"[SPLIT] {args.split} | F1_macro@per-class-th = {f1m:.4f} | ECE = {ece:.4f}")
    print("Per-class thresholds:", np.round(ths, 3).tolist())

    outp = f"results/reliability_{args.split}.png"
    reliability_plot(confs, pos_rates, f"Reliability ({args.split})", outp)

    # 분포 요약(디버깅용)
    flat = probs.flatten()
    qs = np.quantile(flat, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    print("[Probs quantiles]", np.round(qs, 3).tolist(), f"(N={flat.size})")

if __name__ == "__main__":
    main()
