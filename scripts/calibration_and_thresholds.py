# scripts/calibration_and_thresholds.py
import argparse, json, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from src.datasets.medmnist_loader import get_medmnist_loaders
from src.models.baseline_cnn import BaselineCNN
from src.utils.thresholds import optimal_thresholds

def ece_multilabel(probs, targets, bins=15):
    # probs, targets: [N, C], numpy
    conf = probs.flatten()
    corr = ((probs >= 0.5) == (targets==1)).astype(np.float32).flatten()
    bin_edges = np.linspace(0,1,bins+1)
    ece = 0.0
    for i in range(bins):
        m = (conf>=bin_edges[i]) & (conf<bin_edges[i+1])
        if m.sum()==0: continue
        acc = corr[m].mean()
        conf_mean = conf[m].mean()
        ece += abs(acc - conf_mean) * (m.mean())
    return float(ece)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="chestmnist")
ap.add_argument("--img_size", type=int, default=128)
ap.add_argument("--backbone", default="mobilenetv3_small_100")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", default="val", choices=["val","test"])
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
_, val_loader, test_loader, meta = get_medmnist_loaders(args.dataset, batch_size=128, img_size=args.img_size, augment="none")
loader = val_loader if args.split=="val" else test_loader
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
    for x,y in loader:
        x = x.to(device); y = y.squeeze().float()
        p = torch.sigmoid(model(x))
        all_probs.append(p.cpu()); all_targs.append(y)
probs = torch.cat(all_probs).numpy()
targs = torch.cat(all_targs).numpy()

# thresholds from val (for val split: recompute; for test split: 사용할 thresholds.json 있으면 로드)
ths = optimal_thresholds(probs, targs, steps=100)
preds = (probs >= ths[None,:]).astype(int)
f1m = f1_score(targs, preds, average="macro", zero_division=0)
ece = ece_multilabel(probs, targs)

print("[SPLIT]", args.split, "F1_macro@", "per-class th", f1m, " | ECE:", ece)
print("Per-class thresholds:", np.round(ths,3).tolist())

# reliability diagram
bins = np.linspace(0,1,11)
conf = probs.flatten(); corr = ((probs>=0.5)==(targs==1)).astype(np.float32).flatten()
accs, confs = [], []
for i in range(len(bins)-1):
    m = (conf>=bins[i]) & (conf<bins[i+1])
    if m.sum()==0: accs.append(np.nan); confs.append((bins[i]+bins[i+1])/2)
    else:
        accs.append(corr[m].mean()); confs.append(conf[m].mean())
plt.figure(figsize=(4,4))
plt.plot([0,1],[0,1],'--')
plt.plot(confs, accs, marker='o')
plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"Reliability ({args.split})")
plt.tight_layout(); outp=f"results/reliability_{args.split}.png"; plt.savefig(outp, dpi=200)
print(f"[OK] saved {outp}")
