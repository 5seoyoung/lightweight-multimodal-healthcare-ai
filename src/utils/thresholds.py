# src/utils/thresholds.py
import numpy as np

def optimal_thresholds(probs: np.ndarray, targets: np.ndarray, steps: int = 50):
    """
    probs: (N, C) sigmoid 확률, targets: (N, C) {0,1}
    클래스별 F1 최대화 threshold 탐색
    """
    N, C = probs.shape
    ths = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p = probs[:, c]
        t = targets[:, c]
        best_f1, best_th = -1.0, 0.5
        for th in np.linspace(0.05, 0.95, steps):
            pred = (p >= th).astype(int)
            tp = (pred & (t == 1)).sum()
            fp = (pred & (t == 0)).sum()
            fn = ((1 - pred) & (t == 1)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        ths[c] = best_th
    return ths
