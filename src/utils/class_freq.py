# src/utils/class_freq.py
import torch

def estimate_pos_weight(loader, n_classes: int) -> torch.Tensor:
    """
    pos_weight_c = N_neg_c / N_pos_c
    """
    pos = torch.zeros(n_classes, dtype=torch.float64)
    total = 0
    for _, y in loader:
        y = y.squeeze().float()  # [B, C]
        pos += y.sum(dim=0)
        total += y.size(0)
    neg = total - pos
    pw = (neg / (pos + 1e-9)).clamp(max=1e3)  # 과도한 가중 상한
    return pw.float()
