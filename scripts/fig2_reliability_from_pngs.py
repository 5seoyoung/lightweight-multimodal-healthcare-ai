import os, glob, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs("figures", exist_ok=True)

# 우리가 방금 만든 결과 PNG 경로들(존재하는 것만 사용)
cands = [
    ("Student Sup s0", "results/reliability_student_mbv3_sup_s0_test.png"),
    ("Student Sup s1", "results/reliability_student_mbv3_sup_s1_test.png"),
    ("Student Sup s2", "results/reliability_student_mbv3_sup_s2_test.png"),
    ("Distill CE→KD(inv) s0", "results/reliability_distill_ce2kd_inverse_s0_test.png"),
    ("Distill CE→KD(inv) s1", "results/reliability_distill_ce2kd_inverse_s1_test.png"),
    ("Distill CE→KD(inv) s2", "results/reliability_distill_ce2kd_inverse_s2_test.png"),
]
items = [(t,p) for t,p in cands if os.path.exists(p)]
if not items:
    raise SystemExit("No reliability_*_test.png found in results/")

# 그리드 크기 잡기 (3열 기준)
n = len(items)
cols = min(3, n)
rows = (n + cols - 1)//cols

fig, axes = plt.subplots(rows, cols, figsize=(cols*4.2, rows*4.2))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])
elif cols == 1:
    axes = np.array([[ax] for ax in axes])

# 이미지 배치
for idx, (title, path) in enumerate(items):
    r, c = divmod(idx, cols)
    ax = axes[r, c]
    img = np.array(Image.open(path))
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

# 빈 칸 처리
for idx in range(n, rows*cols):
    r, c = divmod(idx, cols)
    axes[r, c].axis('off')

plt.tight_layout()
for ext in ("png", "pdf"):
    out = f"figures/fig2_reliability.{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved {out}")
