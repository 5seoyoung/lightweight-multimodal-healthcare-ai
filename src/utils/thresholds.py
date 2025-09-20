# src/utils/thresholds.py
import numpy as np
from typing import Tuple, Dict, Any


def _best_threshold_from_sorted(scores_desc: np.ndarray,
                                labels_sorted: np.ndarray,
                                beta: float = 1.0) -> Tuple[float, float, float, float]:
    """
    단일 클래스에 대해 내림차순 정렬된 점수(scores_desc)와 동일 순서 라벨(labels_sorted)을 받아
    F_beta가 최대가 되는 임계값과 해당 (F, P, R)을 반환.
    임계값은 p_k와 p_{k+1}의 중간값으로 정의(경계는 1.0 또는 0.0 처리를 포함).
    """
    eps = 1e-12
    n = scores_desc.shape[0]
    P = labels_sorted.sum()  # 양성 총개수

    # 양성이 하나도 없는 클래스: 어떤 임계값도 F1은 0 → 예측 없음(1.0) 권고
    if P == 0:
        return 1.0, 0.0, 0.0, 0.0

    # 누적 TP (상위 k개를 양성으로 예측)
    cum_TP = np.cumsum(labels_sorted)  # k=1..n
    k = np.arange(1, n + 1, dtype=np.float64)
    TP = cum_TP.astype(np.float64)
    FP = k - TP
    FN = float(P) - TP

    precision = TP / (k + eps)
    recall = TP / (float(P) + eps)

    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall + eps)

    # k=0(전부 음성 예측)일 때의 F_beta=0도 고려하지만 최대가 될 일은 없음
    idx = int(np.argmax(fbeta))
    best_f = float(fbeta[idx])
    best_p = float(precision[idx])
    best_r = float(recall[idx])

    # 임계값은 p_idx와 p_{idx+1}의 중간값(같은 값 연속 시 mid도 동일)
    if idx < n - 1:
        t = float((scores_desc[idx] + scores_desc[idx + 1]) / 2.0)
    else:
        # 모든 샘플을 양성으로 예측하는 k=n일 때는 다음 값이 없으므로 낮은 쪽으로 살짝
        t = float((scores_desc[idx] + 0.0) / 2.0)

    # 수치 안정화: [0,1] 클램프
    t = max(0.0, min(1.0, t))
    return t, best_f, best_p, best_r


def optimal_thresholds(probs: np.ndarray,
                       targets: np.ndarray,
                       beta: float = 1.0) -> np.ndarray:
    """
    멀티라벨에서 '클래스별 F_beta'를 최대화하는 임계값을 정확히 산출(정렬 기반, 그리드 불필요).
    probs:  [N, C] (시그모이드 확률)
    targets:[N, C] ({0,1})
    beta:   F_beta의 beta (기본 1.0 → F1)

    returns:
      ths: [C] per-class threshold
    """
    assert probs.ndim == 2 and targets.ndim == 2, "probs/targets must be [N, C]"
    assert probs.shape == targets.shape, "probs and targets must have same shape"
    N, C = probs.shape

    ths = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p = probs[:, c].astype(np.float64)
        y = targets[:, c].astype(np.int8)

        # 정렬(내림차순)
        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y[order]

        t, _, _, _ = _best_threshold_from_sorted(p_sorted, y_sorted, beta=beta)
        ths[c] = t

    return ths


def optimal_thresholds_with_stats(probs: np.ndarray,
                                  targets: np.ndarray,
                                  beta: float = 1.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    임계값 + 부가 통계(F_beta, 정밀도, 재현율)를 함께 반환.
    returns:
      ths: [C]
      stats: {
        "per_class": [{"t":..,"F":..,"P":..,"R":..}, ...],
        "macro_F": float, "macro_P": float, "macro_R": float
      }
    """
    assert probs.ndim == 2 and targets.ndim == 2 and probs.shape == targets.shape
    N, C = probs.shape

    ths = np.zeros(C, dtype=np.float32)
    info = []
    Fs, Ps, Rs = [], [], []
    for c in range(C):
        p = probs[:, c].astype(np.float64)
        y = targets[:, c].astype(np.int8)
        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y[order]
        t, F, P, R = _best_threshold_from_sorted(p_sorted, y_sorted, beta=beta)
        ths[c] = t
        Fs.append(F); Ps.append(P); Rs.append(R)
        info.append({"t": float(t), "F": float(F), "P": float(P), "R": float(R)})

    stats = {
        "per_class": info,
        "macro_F": float(np.nanmean(Fs) if len(Fs) else 0.0),
        "macro_P": float(np.nanmean(Ps) if len(Ps) else 0.0),
        "macro_R": float(np.nanmean(Rs) if len(Rs) else 0.0),
    }
    return ths, stats
