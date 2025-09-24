# scripts/plot_pareto.py
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/aggregate.csv")
ap.add_argument("--out", default="results/pareto.png")
ap.add_argument("--metric", default="auprc", choices=["auprc","auroc"])
args = ap.parse_args()

df = pd.read_csv(args.csv)

# 그룹( student, sched, cw_kd, pretrained )로 묶어 시드 평균±표준편차
grp_cols = ["student","sched","cw_kd","pretrained_student","img_size","dataset","teacher"]
agg = df.groupby(grp_cols).agg(
    auprc_mean=("auprc","mean"), auprc_std=("auprc","std"),
    auroc_mean=("auroc","mean"), auroc_std=("auroc","std"),
    params=("params","mean"), flops=("flops","mean"), latency=("latency_ms","mean"),
    n=("auprc","count")
).reset_index()

y_mean = agg[f"{args.metric}_mean"].values
y_std  = agg[f"{args.metric}_std"].values

def label_row(r):
    tag = f"{r.student}"
    if r.sched!="sup": tag += f" | {r.sched}"
    if r.cw_kd!="none": tag += f" | {r.cw_kd}"
    if bool(r.pretrained_student): tag += " | pre"
    return tag

labels = [label_row(r) for _,r in agg.iterrows()]

fig, axes = plt.subplots(1,3, figsize=(16,4))
pairs = [("params","Params"),("flops","FLOPs"),("latency","Latency(ms)")]

for ax,(xcol,xtitle) in zip(axes,pairs):
    x = agg[xcol].values
    m = ~np.isnan(x)
    ax.errorbar(x[m], y_mean[m], yerr=y_std[m], fmt='o')
    ax.set_xlabel(xtitle); ax.set_ylabel(args.metric.upper())
    ax.set_title(f"{xtitle} vs {args.metric.upper()}")
    # 간단한 라벨
    for i,(xx,yy) in enumerate(zip(x[m], y_mean[m])):
        ax.annotate(labels[i], (xx,yy), fontsize=8, xytext=(5,3), textcoords='offset points')

fig.tight_layout()
plt.savefig(args.out, dpi=200)
print(f"[OK] saved {args.out}")
