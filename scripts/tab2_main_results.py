# scripts/tab2_main_results.py
import json, glob, re, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def parse_method(run_name):
    name = os.path.basename(run_name)
    if name.startswith("teacher_resnet18"):
        return "Teacher ResNet-18", "✗", "–"
    if name.startswith("student_mbv3_sup"):
        return "Student MobileNetV3", "✓", "–"
    if "distill_ce2kd_inverse" in name:
        return "KD: CE→KD + Inverse", "✓", "ce2kd/inverse"
    if "distill_kd2ce_effective" in name:
        return "KD: KD→CE + Effective", "✓", "kd2ce/effective"
    if "distill_ce2kd_none" in name or "ce2kd/none" in name:
        return "KD: CE→KD + None", "✓", "ce2kd/none"
    return "Unknown", "?", "?"

def collect():
    rows = []
    for test_json in glob.glob("runs/*_s*/test.json"):
        run = os.path.dirname(test_json)
        with open(test_json) as f:
            d = json.load(f)
        method, pre, sched = parse_method(run)
        seed = re.findall(r"_s(\d+)", run)
        seed = int(seed[0]) if seed else -1
        rows.append({
            "Run": run, "Method": method, "Pretrained": pre, "Schedule/Weight": sched,
            "Seed": seed,
            "AUROC": d.get("auroc"), "AUPRC": d.get("auprc"),
            "F1_macro": d.get("f1_macro"), "ECE": d.get("ece", None)
        })
    return pd.DataFrame(rows)

def agg_table(df):
    gcols = ["Method","Pretrained","Schedule/Weight"]
    mcols = ["AUROC","AUPRC","F1_macro","ECE"]
    out = df.groupby(gcols)[mcols].agg(["mean","std"]).reset_index()
    # pretty format mean±std
    disp = out[gcols].copy()
    for c in mcols:
        disp[c] = out[(c,"mean")].map(lambda x: f"{x:.4f}") + " ± " + out[(c,"std")].map(lambda s: f"{(0 if pd.isna(s) else s):.4f}")
    return disp

def render_pdf_table(df):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.2)
    plt.savefig("figures/tab2_main_results.pdf", bbox_inches="tight")
    plt.savefig("figures/tab2_main_results.png", bbox_inches="tight", dpi=300)

def main():
    os.makedirs("figures", exist_ok=True)
    df = collect()
    if df.empty:
        raise RuntimeError("No runs/*_s*/test.json found.")
    disp = agg_table(df)
    disp.to_csv("figures/tab2_main_results.csv", index=False)
    render_pdf_table(disp)
    print("Saved figures/tab2_main_results.[csv|pdf|png]")

if __name__ == "__main__":
    main()
