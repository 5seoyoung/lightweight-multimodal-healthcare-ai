# scripts/fig2_reliability.py
import numpy as np, matplotlib.pyplot as plt, os

def ece(y_true, y_prob, n_bins=10):
    # multi-label: 클래스 축 평균
    eps = 1e-12
    bin_edges = np.linspace(0,1,n_bins+1)
    eces = []
    for c in range(y_true.shape[1]):
        t = y_true[:,c].astype(float)
        p = y_prob[:,c].astype(float)
        accs, confs, weights = [], [], []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b+1]
            idx = (p>=lo) & (p<hi) if b<n_bins-1 else (p>=lo) & (p<=hi)
            if idx.sum()==0:
                accs.append(0.0); confs.append((lo+hi)/2); weights.append(0)
            else:
                accs.append(t[idx].mean())
                confs.append(p[idx].mean())
                weights.append(idx.mean())  # bin weight
        eces.append(np.sum(np.array(weights)*np.abs(np.array(accs)-np.array(confs))))
    return float(np.mean(eces))

def plot_reliability(y_true, y_prob, title, outpath, n_bins=10):
    bin_edges = np.linspace(0,1,n_bins+1)
    accs, confs = [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b+1]
        idx = (y_prob>=lo) & (y_prob<hi) if b<n_bins-1 else (y_prob>=lo) & (y_prob<=hi)
        if idx.sum()==0:
            accs.append(0.0); confs.append((lo+hi)/2)
        else:
            accs.append(y_true[idx].mean()); confs.append(y_prob[idx].mean())
    fig, ax = plt.subplots(figsize=(3.4,3.2))
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1.0)
    ax.plot(confs, accs, marker="o", linewidth=1.8)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

def flatten_to_scalar(y_true, y_prob):
    # macro over classes → 벤 다이어그램용 스칼라로 전개
    return y_true.reshape(-1), y_prob.reshape(-1)

def main():
    os.makedirs("figures", exist_ok=True)
    pairs = [
        ("runs/student_mbv3_sup_s0/preds_test.npz", "Student (Supervised)", "figures/fig2_reliability_student.pdf"),
        ("runs/distill_ce2kd_inverse_s0/preds_test.npz", "KD (CE→KD + Inverse)", "figures/fig2_reliability_kd.pdf"),
    ]
    for npz, title, out in pairs:
        data = np.load(npz)
        y_true, y_prob = data["y_true"], data["y_prob"]
        e = ece(y_true, y_prob, n_bins=10)
        yt, yp = flatten_to_scalar(y_true, y_prob)
        plot_reliability(yt, yp, f"{title}  (ECE={e:.3f})", out)
        print(f"{title}: ECE={e:.4f} -> saved {out}")

if __name__ == "__main__":
    main()
