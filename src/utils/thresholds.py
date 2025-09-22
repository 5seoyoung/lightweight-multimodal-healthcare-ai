# src/utils/thresholds.py
import numpy as np
from typing import Optional

def optimal_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    steps: Optional[int] = None,
    n_steps: Optional[int] = None,
    grid: Optional[np.ndarray] = None,
):
    """
    멀티라벨에서 클래스별 F1을 최대화하는 임계값을 탐색.
    - probs: [N, C] (시그모이드 확률)
    - targets: [N, C] ({0,1})
    - steps / n_steps: 0~1 균일 그리드 개수(기본 50). 둘 중 아무거나 써도 됨.
    - grid: 직접 임계값 배열(np.ndarray)을 넘길 수도 있음(예: np.linspace(0,1,101)).

    returns: [C] per-class threshold (float32)
    """
    assert probs.ndim == 2 and targets.ndim == 2, "probs/targets must be 2D [N, C]"
    assert probs.shape == targets.shape, "probs and targets must have same shape"
    n_classes = probs.shape[1]

    # 우선순위: grid > steps/n_steps > default(50)
    if grid is not None:
        grid = np.asarray(grid, dtype=np.float32)
        assert grid.ndim == 1 and grid.size >= 2, "grid must be 1-D with >=2 points"
    else:
        k = steps if steps is not None else (n_steps if n_steps is not None else 50)
        # edge 포함 균일 그리드
        grid = np.linspace(0.0, 1.0, int(k) + 1, dtype=np.float32)

    eps = 1e-12
    ths = np.zeros(n_classes, dtype=np.float32)

    # 클래스별 그리드 서치
    for c in range(n_classes):
        p = probs[:, c].astype(np.float32, copy=False)
        y = targets[:, c].astype(np.int32, copy=False)

        best_f1, best_t = -1.0, 0.5
        # 간단/안전 우선 구현 (정렬 기반 최적화는 생략)
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
