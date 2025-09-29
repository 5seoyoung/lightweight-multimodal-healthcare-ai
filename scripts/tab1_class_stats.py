# scripts/tab1_class_stats.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from medmnist import ChestMNIST
from medmnist import INFO

def load_split(split):
    ds = ChestMNIST(split=split, download=True, transform=None, as_rgb=False)
    imgs, labels = ds.imgs, ds.labels  # labels shape: [N, 14]
    return labels

def prevalence(labels):
    n = labels.shape[0]
    pos = labels.sum(axis=0)
    return pos, n

def main():
    info = INFO["chestmnist"]
    class_names = [c.replace("_", " ") for c in info["label"]]

    y_tr = load_split("train")
    y_val = load_split("val")
    y_te = load_split("test")

    pos_tr, n_tr = prevalence(y_tr)
    pos_val, n_val = prevalence(y_val)
    pos_te, n_te = prevalence(y_te)

    df = pd.DataFrame({
        "Class": class_names,
        "Train Pos(%)": np.round(100*pos_tr/n_tr, 2),
        "Val Pos(%)":   np.round(100*pos_val/n_val, 2),
        "Test Pos(%)":  np.round(100*pos_te/n_te, 2),
        "#Train": pos_tr.astype(int),
        "#Val":   pos_val.astype(int),
        "#Test":  pos_te.astype(int),
    })

    df.to_csv("figures/tab1_class_stats.csv", index=False)

    # Render as vector table
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    plt.savefig("figures/tab1_class_stats.pdf", bbox_inches="tight")
    plt.savefig("figures/tab1_class_stats.png", bbox_inches="tight", dpi=300)
    print("Saved CSV and PDF/PNG in figures/")

if __name__ == "__main__":
    main()
