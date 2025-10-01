# scripts/fig3_training_dynamics.py
import json, os
import matplotlib.pyplot as plt

def read_log(path):
    ep, auprc, f1 = [], [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "epoch" in d and ("val_auprc" in d or "val_f1_macro" in d):
                ep.append(d["epoch"])
                auprc.append(d.get("val_auprc"))
                f1.append(d.get("val_f1_macro"))
    return ep, auprc, f1

def plot_metric(runs, metric_key, ylabel, outfile):
    fig, ax = plt.subplots(figsize=(3.6,3.2))
    for label, log in runs:
        ep, auprc, f1 = read_log(log)
        y = auprc if metric_key=="val_auprc" else f1
        ax.plot(ep, y, linewidth=1.8, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    print("Saved", outfile)

def main():
    runs = [
        ("Student (Supervised)", "runs/student_mbv3_sup_s0/run.log"),
        ("KD→CE + Effective",    "runs/distill_kd2ce_effective_s0/run.log"),
        ("CE→KD + Inverse",      "runs/distill_ce2kd_inverse_s0/run.log"),
    ]
    plot_metric(runs, "val_auprc", "Val AUPRC", "figures/fig3_val_auprc.pdf")
    plot_metric(runs, "val_f1_macro", "Val F1_macro", "figures/fig3_val_f1.pdf")

if __name__ == "__main__":
    main()
