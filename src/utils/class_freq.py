# src/utils/class_freq.py
import torch
from typing import Tuple

@torch.no_grad()
def estimate_pos_weight(
    loader,
    n_classes: int,
    cap: float = 1e2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate per-class pos_weight for BCEWithLogitsLoss in multi-label setting.

    Formula (per class c):
        pos_weight_c = N_neg_c / (N_pos_c + eps)
    where N_neg_c = N_total - N_pos_c (N_total is #samples)

    Args:
        loader: DataLoader yielding (x, y). y may be [B, C], [B, 1, C], or [B].
        n_classes: number of labels/classes C.
        cap: upper bound to avoid numerically extreme weights on ultra-rare classes.
        eps: small constant to avoid division-by-zero.

    Returns:
        torch.Tensor of shape [C], dtype=float32 (CPU tensor).
    """
    pos = torch.zeros(n_classes, dtype=torch.float64)
    total = 0

    for _, y in loader:
        # y could be [B, C], [B, 1, C], or [B]
        y = y.squeeze()  # collapse possible singleton dims
        if y.ndim == 1:
            # binary to [B, 1] for consistency
            y = y.unsqueeze(1)
        # ensure [B, C]
        y = y.reshape(y.size(0), -1).to(dtype=torch.float64)

        if y.size(1) != n_classes:
            raise ValueError(f"Label dimension mismatch: got {y.size(1)} != n_classes({n_classes})")

        pos += y.sum(dim=0)
        total += y.size(0)

    neg = total - pos
    pw = (neg / (pos + eps)).clamp(max=cap)
    return pw.to(dtype=torch.float32)


@torch.no_grad()
def estimate_class_prevalence(
    loader,
    n_classes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Utility for logging/reporting: per-class prevalence pi_c = N_pos_c / N_total.

    Returns:
        torch.Tensor of shape [C], float32 on CPU.
    """
    pos = torch.zeros(n_classes, dtype=torch.float64)
    total = 0

    for _, y in loader:
        y = y.squeeze()
        if y.ndim == 1:
            y = y.unsqueeze(1)
        y = y.reshape(y.size(0), -1).to(dtype=torch.float64)
        if y.size(1) != n_classes:
            raise ValueError(f"Label dimension mismatch: got {y.size(1)} != n_classes({n_classes})")
        pos += y.sum(dim=0)
        total += y.size(0)

    prev = pos / (total + eps)
    return prev.to(dtype=torch.float32)
