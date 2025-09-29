# scripts/fig1_pipeline.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
import numpy as np

def box(ax, xy, w, h, text, lw=1.5):
    x,y = xy
    rect = FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=lw, edgecolor="black", facecolor="white")
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center")

def arrow(ax, xy1, xy2, lw=1.2):
    ax.add_patch(FancyArrow(xy1[0], xy1[1], xy2[0]-xy1[0], xy2[1]-xy1[1],
                            width=0.004, head_width=0.035, head_length=0.05,
                            length_includes_head=True, color="black", linewidth=lw))

def main():
    fig = plt.figure(figsize=(8.0, 3.6))
    ax = fig.add_axes([0.04,0.15,0.58,0.8]); ax.axis("off")

    # Main flow (left to right)
    box(ax, (0.03,0.40), 0.12, 0.22, "Input\n(ChestMNIST)")
    box(ax, (0.20,0.40), 0.18, 0.22, "Student\nMobileNetV3-Small")
    box(ax, (0.44,0.40), 0.14, 0.22, "logits_s")
    box(ax, (0.20,0.75), 0.18, 0.22, "Teacher\nResNet-18")
    box(ax, (0.44,0.75), 0.14, 0.22, "logits_t")
    box(ax, (0.72,0.40), 0.20, 0.22, "Per-class\nThresholding\n(val-optimized)")
    box(ax, (0.72,0.75), 0.20, 0.22, "Losses\nBCE + KL\nα(t), τ(t)")

    # arrows
    arrow(ax, (0.15,0.51),(0.20,0.51))           # Input -> Student
    arrow(ax, (0.38,0.51),(0.44,0.51))           # Student -> logits_s
    arrow(ax, (0.38,0.86),(0.44,0.86))           # Teacher -> logits_t
    arrow(ax, (0.58,0.86),(0.72,0.86))           # logits_t -> Loss
    arrow(ax, (0.58,0.51),(0.72,0.51))           # logits_s -> Thresholding
    # KD arrow (logits_t to logits_s via loss block)
    arrow(ax, (0.51,0.75),(0.72,0.75))
    ax.text(0.615,0.78,"KL(σ(z_t/τ) || σ(z_s/τ))", ha="center", va="center", fontsize=9)

    # Inset curves for α(t) and τ(t)
    ax2 = fig.add_axes([0.70,0.12,0.25,0.22])
    epochs = np.arange(1,13)
    alpha = 0.1 + (0.7-0.1)*(epochs-1)/(len(epochs)-1)    # 0.1 -> 0.7
    tau   = 2.0 + (5.0-2.0)*(epochs-1)/(len(epochs)-1)    # 2   -> 5
    ax2.plot(epochs, alpha, linewidth=1.8, label=r"$\alpha(t)$")
    ax2.plot(epochs, tau,   linewidth=1.8, label=r"$\tau(t)$")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Value")
    ax2.set_title("Schedule (CE→KD)")
    ax2.grid(True, linewidth=0.5, alpha=0.6)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    for ext in ("pdf","png"):
        plt.savefig(f"figures/fig1_pipeline.{ext}", bbox_inches="tight", dpi=300)
    print("Saved to figures/fig1_pipeline.[pdf|png]")

if __name__ == "__main__":
    main()
