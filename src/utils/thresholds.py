# src/utils/thresholds.py
import numpy as np

def optimal_thresholds(probs: np.ndarray, targets: np.ndarray, steps: int = 50):
    """
    멀티라벨에서 클래스별 F1을 최대화하는 임계값(0~1)을 간단 그리드서치로 찾음.
    probs: [N, C] sigmoid 확률
    targets: [N, C] {0,1}
    returns: [C] per-class threshold
    """
    eps = 1e-12
    n_classes = probs.shape[1]
    ths = np.zeros(n_classes, dtype=np.float32)

    # 간격 균일 grid (edge값 포함)
    grid = np.linspace(0.0, 1.0, steps+1)

    for c in range(n_classes):
        p = probs[:, c]
        y = targets[:, c]
        best_f1, best_t = -1.0, 0.5
        # 빠른 누적 계산을 위해 정렬 기반 접근은 생략(간단/안전 우선)
        for t in grid:
            pred = (p >= t).astype(np.int32)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            precision = tp / (tp + fp + eps)
            recall    = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        ths[c] = best_t
    return ths
