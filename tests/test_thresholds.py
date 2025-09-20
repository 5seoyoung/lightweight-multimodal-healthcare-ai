# tests/test_thresholds.py
import os
import sys
import numpy as np
import pytest

# --- import path 설정: 프로젝트 루트와 src 를 sys.path 에 추가 ---
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
for p in (_ROOT_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.thresholds import optimal_thresholds  # noqa: E402


def _f1_at_threshold(p: np.ndarray, y: np.ndarray, t: float, eps: float = 1e-12) -> float:
    """단일 클래스 확률/정답 벡터에 대해 threshold t 적용 시 F1 계산."""
    pred = (p >= t).astype(np.int32)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float(2 * precision * recall / (precision + recall + eps))


def test_shape_and_bounds_random():
    np.random.seed(42)
    N, C = 137, 7
    probs = np.clip(np.random.rand(N, C), 0.0, 1.0)
    targets = (np.random.rand(N, C) > 0.7).astype(np.int32)

    ths = optimal_thresholds(probs, targets, steps=50)
    assert isinstance(ths, np.ndarray)
    assert ths.shape == (C,)
    assert np.all(ths >= 0.0) and np.all(ths <= 1.0)


def test_perfect_separation_threshold_within_gap():
    """
    음성은 0.10, 양성은 0.90 근방으로 분리된 경우,
    반환된 threshold 가 (max_neg, min_pos] 구간에 떨어지는지 확인.
    """
    np.random.seed(0)
    N, C = 200, 3
    y = (np.random.rand(N, C) > 0.7).astype(np.int32)  # 약 30% 양성
    neg_noise = np.random.uniform(-0.01, 0.01, size=(N, C))
    pos_noise = np.random.uniform(-0.01, 0.01, size=(N, C))
    p = np.where(y == 1, 0.90 + pos_noise, 0.10 + neg_noise)
    p = np.clip(p, 0.0, 1.0)

    ths = optimal_thresholds(p, y, steps=100)
    for c in range(C):
        max_neg = float(p[y[:, c] == 0, c].max(initial=0.0))
        min_pos = float(p[y[:, c] == 1, c].min(initial=1.0))
        # 최적 F1 을 내는 어느 t 도 (max_neg, min_pos] 안에 존재
        assert max_neg < ths[c] <= min_pos


def test_all_negative_and_all_positive_classes():
    """
    모든 샘플이 0 인 클래스, 1 인 클래스에 대한 방어적 동작 확인.
    - all-negative: F1=0 이므로 루프 첫 지점(th=0.0)을 선택하는 구현 → 0.0 기대
    - all-positive: th=0.0 에서 F1=1 이므로 0.0 선택 기대
    """
    N = 50
    probs = np.concatenate([
        np.random.uniform(0.0, 0.3, size=(N, 1)),  # c0: all-negative 라벨에 대응하는 확률 (임의)
        np.random.uniform(0.7, 1.0, size=(N, 1)),  # c1: all-positive 라벨에 대응하는 확률 (임의)
    ], axis=1)
    targets = np.concatenate([
        np.zeros((N, 1), dtype=np.int32),          # c0: 모두 0
        np.ones((N, 1), dtype=np.int32),           # c1: 모두 1
    ], axis=1)

    ths = optimal_thresholds(probs, targets, steps=20)
    assert ths.shape == (2,)
    # 구현 특성상 둘 다 0.0 을 반환
    assert ths[0] == pytest.approx(0.0, abs=1e-9)
    assert ths[1] == pytest.approx(0.0, abs=1e-9)

    # 반환 임계값에서의 F1 이, 동일 그리드에서의 최대 F1 과 동일한지 확인
    grid = np.linspace(0.0, 1.0, 21)
    for c in range(2):
        f1_ret = _f1_at_threshold(probs[:, c], targets[:, c], ths[c])
        f1_grid_max = max(_f1_at_threshold(probs[:, c], targets[:, c], t) for t in grid)
        assert f1_ret == pytest.approx(f1_grid_max, rel=0, abs=1e-12)


def test_matches_grid_argmax_first_max_ok():
    """
    랜덤 확률/라벨에 대해 steps=25 그리드로 계산 시,
    반환된 threshold 에서의 F1 이 그리드 최대 F1 과 일치하는지 확인.
    (동률 시 구현은 '처음 만나는 최대'를 선택하지만, 여기선 값 일치만 검증)
    """
    np.random.seed(123)
    N, C = 120, 5
    probs = np.random.beta(a=2.0, b=5.0, size=(N, C))  # 좌우 치우친 분포
    targets = (np.random.rand(N, C) > 0.75).astype(np.int32)

    steps = 25
    ths = optimal_thresholds(probs, targets, steps=steps)
    grid = np.linspace(0.0, 1.0, steps + 1)

    for c in range(C):
        f1_ret = _f1_at_threshold(probs[:, c], targets[:, c], ths[c])
        f1_grid = np.array([_f1_at_threshold(probs[:, c], targets[:, c], t) for t in grid])
        f1_max = float(f1_grid.max())
        assert f1_ret == pytest.approx(f1_max, rel=0, abs=1e-12)


def test_known_boundary_threshold():
    """
    인위적으로 타깃이 'p >= 0.7 → 1, else 0' 로 생성된 경우,
    steps=10 그리드에서는 0.7 이 포함되므로 최적 threshold 가 0.7 과 일치해야 한다.
    """
    N = 200
    p = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    y = (p >= 0.7).astype(np.int32)
    ths = optimal_thresholds(p, y, steps=10)  # grid: 0.0, 0.1, ..., 1.0
    assert ths.shape == (1,)
    assert ths[0] == pytest.approx(0.7, abs=1e-12)

    # 해당 임계값에서 F1 이 그리드 최대와 동일
    grid = np.linspace(0.0, 1.0, 11)
    f1_ret = _f1_at_threshold(p[:, 0], y[:, 0], ths[0])
    f1_max = max(_f1_at_threshold(p[:, 0], y[:, 0], t) for t in grid)
    assert f1_ret == pytest.approx(f1_max, rel=0, abs=1e-12)
