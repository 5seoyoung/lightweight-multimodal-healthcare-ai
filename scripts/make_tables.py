#!/usr/bin/env python
# scripts/make_tables.py
import argparse
import json
import os
import re
import sys
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def _find_test_logs(results_dir: str) -> List[str]:
    paths = []
    paths += glob(os.path.join(results_dir, "logs", "*_test.json"))
    paths += glob(os.path.join(results_dir, "seed_runs", "*", "logs", "*_test.json"))
    # de-dup and sort
    paths = sorted(list(dict.fromkeys(paths)))
    return paths


def _parse_exp_from_filename(fname: str) -> Dict[str, Optional[str]]:
    """
    파일명 규칙:
      - baseline:  {dataset}_{backbone}_test.json
      - distill :  distill_{dataset}_{teacher}_to_{student}_test.json
    반환 키: regimen, dataset, student_backbone, teacher_backbone
    """
    base = os.path.basename(fname)
    name = base.replace("_test.json", "")

    if name.startswith("distill_"):
        # distill_chestmnist_resnet18_to_mobilenetv3_small_100
        m = re.match(r"^distill_([^_]+)_(.+?)_to_(.+)$", name)
        if not m:
            return dict(regimen=None, dataset=None, student_backbone=None, teacher_backbone=None)
        dataset, teacher, student = m.groups()
        return dict(
            regimen="distill",
            dataset=dataset,
            student_backbone=student,
            teacher_backbone=teacher,
        )
    else:
        # chestmnist_mobilenetv3_small_100
        m = re.match(r"^([^_]+)_(.+)$", name)
        if not m:
            return dict(regimen=None, dataset=None, student_backbone=None, teacher_backbone=None)
        dataset, backbone = m.groups()
        return dict(
            regimen="baseline",
            dataset=dataset,
            student_backbone=backbone,
            teacher_backbone=None,
        )


def _extract_group_tag(path: str) -> Optional[str]:
    """
    시드 반복 폴더(ex. results/seed_runs/baseline128/...)일 경우 group 태그 추출
    없으면 None
    """
    # .../seed_runs/<group>/logs/file.json
    m = re.search(r"seed_runs/([^/]+)/logs/", path.replace("\\", "/"))
    return m.group(1) if m else None


def _read_json(fp: str) -> Optional[dict]:
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON: {fp} ({e})", file=sys.stderr)
        return None


def _bootstrap_ci(x: np.ndarray, n_boot: int = 2000, ci: float = 0.95, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    if x.size == 1:
        return (x[0], x[0])

    means = []
    n = len(x)
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def collect_runs(results_dir: str) -> pd.DataFrame:
    files = _find_test_logs(results_dir)
    rows = []
    for fp in files:
        meta = _parse_exp_from_filename(fp)
        if meta["regimen"] is None:
            continue
        data = _read_json(fp)
        if not isinstance(data, dict):
            continue

        group = _extract_group_tag(fp)  # e.g., baseline128, kd_r18_to_mbv3_128_a01_t5_e12
        # best effort로 해상도 추출(폴더명에 128/160이 들어있다면)
        res = None
        if group:
            m = re.search(r"(1[0-9]{2}|2[0-9]{2}|[3-9][0-9])", group)  # 96,128,160,224...
            if m:
                res = int(m.group(1))

        rows.append(dict(
            path=fp,
            group=group,
            resolution=res,
            regimen=meta["regimen"],
            dataset=meta["dataset"],
            student_backbone=meta["student_backbone"],
            teacher_backbone=meta["teacher_backbone"],
            auprc=data.get("auprc", np.nan),
            auroc=data.get("auroc", np.nan),
            f1_macro=data.get("f1_macro", np.nan),
            f1_macro_opt=data.get("f1_macro_opt", np.nan),
            loss=data.get("loss", np.nan),
        ))
    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "group", "path"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    그룹 기준:
      - dataset, regimen, student_backbone, teacher_backbone, resolution (있으면)
      - seed_runs/<group> 안의 여러 파일이 하나의 그룹을 이룬다고 가정 (없으면 group=None)
      - group이 다르더라도 위 핵심 키가 같으면 같은 실험으로 묶어 평균을 낼 수 있음
    """
    if df.empty:
        return df

    keys = ["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution"]
    df["_cluster"] = df[keys].astype(str).agg("|".join, axis=1)

    recs = []
    for cid, g in df.groupby("_cluster"):
        # seed-run별로 묶여 있다면 그대로 평균; 아니면 파일 하나라도 요약 생성
        n = len(g)
        for metric in ["auprc", "auroc", "f1_macro", "f1_macro_opt", "loss"]:
            x = g[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(x)) if x.size else np.nan
            sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
            lo, hi = _bootstrap_ci(x, n_boot=2000, ci=0.95) if x.size else (np.nan, np.nan)

            rec = {
                "dataset": g["dataset"].iloc[0],
                "regimen": g["regimen"].iloc[0],
                "student_backbone": g["student_backbone"].iloc[0],
                "teacher_backbone": g["teacher_backbone"].iloc[0],
                "resolution": g["resolution"].iloc[0],
                "n_runs": int(n),
                "metric": metric,
                "mean": mean,
                "sd": sd,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
            recs.append(rec)

    out = pd.DataFrame(recs).sort_values(
        ["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "metric"]
    )
    return out


def _fmt_mean_ci(m: float, lo: float, hi: float, digits: int = 3) -> str:
    if np.isnan(m):
        return "-"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def to_markdown(summary: pd.DataFrame) -> str:
    """
    regimen x metric 피벗해 읽기 좋은 표로 변환
    """
    if summary.empty:
        return "# Summary\n\nNo results found.\n"

    # 핵심 메트릭만 표시(필요시 수정)
    keep = summary[summary["metric"].isin(["auprc", "auroc", "f1_macro"])]
    # 포맷 문자열 생성
    keep["mean_ci"] = keep.apply(lambda r: _fmt_mean_ci(r["mean"], r["ci95_lo"], r["ci95_hi"]), axis=1)

    tbl = keep.pivot_table(
        index=["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "n_runs"],
        columns="metric",
        values="mean_ci",
        aggfunc="first",
        fill_value="-"
    ).reset_index()

    # 열 순서 정리
    col_order = ["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "n_runs",
                 "auprc", "auroc", "f1_macro"]
    for c in col_order:
        if c not in tbl.columns:
            tbl[c] = "-"
    tbl = tbl[col_order]

    # markdown
    md = []
    md.append("# Summary (mean [95% CI])")
    md.append("")
    md.append(tbl.to_markdown(index=False))
    md.append("")
    md.append("_Note_: Values are mean [95% CI] across runs in each group; `n_runs` shows the number of test logs.")
    return "\n".join(md)


def to_latex(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "% No results"

    keep = summary[summary["metric"].isin(["auprc", "auroc", "f1_macro"])]
    keep["mean_ci"] = keep.apply(lambda r: _fmt_mean_ci(r["mean"], r["ci95_lo"], r["ci95_hi"]), axis=1)

    tbl = keep.pivot_table(
        index=["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "n_runs"],
        columns="metric",
        values="mean_ci",
        aggfunc="first",
        fill_value="-"
    ).reset_index()

    col_order = ["dataset", "regimen", "student_backbone", "teacher_backbone", "resolution", "n_runs",
                 "auprc", "auroc", "f1_macro"]
    for c in col_order:
        if c not in tbl.columns:
            tbl[c] = "-"
    tbl = tbl[col_order]

    # LaTeX table
    return tbl.to_latex(index=False, escape=True, longtable=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results", help="results root directory")
    ap.add_argument("--out-dir", type=str, default=None, help="output dir (default: <results-dir>/summary)")
    ap.add_argument("--no-md", action="store_true", help="do not write Markdown table")
    ap.add_argument("--no-tex", action="store_true", help="do not write LaTeX table")
    ap.add_argument("--no-csv", action="store_true", help="do not write CSV files")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "summary")
    os.makedirs(out_dir, exist_ok=True)

    df_runs = collect_runs(args.results_dir)
    df_runs.sort_values("path").to_csv(os.path.join(out_dir, "run_list.csv"), index=False)

    summary = summarize(df_runs)
    if not args.no_csv:
        summary.to_csv(os.path.join(out_dir, "summary_by_group.csv"), index=False)

    if not args.no_md:
        md = to_markdown(summary)
        with open(os.path.join(out_dir, "summary_by_group.md"), "w") as f:
            f.write(md)

    if not args.no_tex:
        tex = to_latex(summary)
        with open(os.path.join(out_dir, "summary_by_group.tex"), "w") as f:
            f.write(tex)

    # 콘솔에도 간단 요약 출력
    print(f"[make_tables] collected {len(df_runs)} test logs")
    if not summary.empty:
        print(summary.head(12).to_string(index=False))
    else:
        print("[make_tables] no summary generated (no logs found?)")


if __name__ == "__main__":
    main()
