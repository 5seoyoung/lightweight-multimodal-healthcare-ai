# tests/test_class_freq.py
import os
import sys
import pytest
import torch

# --- import path: 프로젝트 루트 및 src 추가 ---
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
for p in (_ROOT_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.class_freq import estimate_pos_weight  # noqa: E402


class DummyLoader:
    """(x, y) 배치 쌍을 순차적으로 내보내는 아주 단순한 로더.
    - x는 쓰이지 않으므로 더미 텐서로 대체
    - y는 [B, C]의 {0,1} 라벨 텐서
    """
    def __init__(self, y_batches):
        self.y_batches = y_batches

    def __iter__(self):
        for y in self.y_batches:
            x = torch.zeros((y.size(0), 1))  # placeholder
            yield x, y


def test_balanced_two_classes_weights_are_one():
    # B=4, C=2, 각 클래스 양성 2개/음성 2개 → pos_weight = 2/2 = 1
    y = torch.tensor([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], dtype=torch.float32)
    loader = DummyLoader([y])
    pw = estimate_pos_weight(loader, n_classes=2)
    assert pw.shape == (2,)
    assert pw.dtype == torch.float32
    assert torch.allclose(pw, torch.tensor([1.0, 1.0], dtype=torch.float32), atol=1e-6)


def test_imbalanced_classes_match_neg_over_pos_ratio():
    # 총 4개 샘플, C=2
    # c0: pos=1, neg=3 → 3/1 = 3.0
    # c1: pos=3, neg=1 → 1/3 ≈ 0.3333
    y = torch.tensor([
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 0],
    ], dtype=torch.float32)
    loader = DummyLoader([y])
    pw = estimate_pos_weight(loader, n_classes=2)
    exp = torch.tensor([3.0, 1.0 / 3.0], dtype=torch.float32)
    assert torch.allclose(pw, exp, rtol=0, atol=1e-6)


def test_multi_batch_aggregation_not_per_batch():
    # 두 배치를 합산해서 집계해야 함.
    # 배치1: B=2, y=[[1,0],[0,0]]
    # 배치2: B=3, y=[[0,1],[1,0],[0,0]]
    # total=5
    # c0: pos=2 → neg=3 → 3/2 = 1.5
    # c1: pos=1 → neg=4 → 4/1 = 4.0
    y1 = torch.tensor([[1, 0],
                       [0, 0]], dtype=torch.float32)
    y2 = torch.tensor([[0, 1],
                       [1, 0],
                       [0, 0]], dtype=torch.float32)
    loader = DummyLoader([y1, y2])
    pw = estimate_pos_weight(loader, n_classes=2)
    exp = torch.tensor([1.5, 4.0], dtype=torch.float32)
    assert torch.allclose(pw, exp, rtol=0, atol=1e-6)


def test_zero_positive_class_is_clamped_to_1e3():
    # c0: all-negative → pos=0 → neg/pos → inf → clamp(<=1e3) → 1e3
    # c1: pos=2, neg=1 → 1/2 = 0.5
    y = torch.tensor([
        [0, 1],
        [0, 1],
        [0, 0],
    ], dtype=torch.float32)
    loader = DummyLoader([y])
    pw = estimate_pos_weight(loader, n_classes=2)
    assert pw[0].item() == pytest.approx(1_000.0, abs=1e-6)
    assert pw[1].item() == pytest.approx(0.5, abs=1e-6)


def test_returns_float32_tensor_with_correct_length_even_when_n_classes_mismatch():
    # n_classes 인자는 반환 길이를 결정하는 것이 아니라 검증 용도.
    # 함수는 y에서 집계하므로 n_classes는 y의 C와 일치해야 함.
    # 여기서는 올바른 C를 전달하고, dtype/shape를 다시 확인.
    y = torch.tensor([[1, 0, 1],
                      [0, 0, 0],
                      [0, 1, 0]], dtype=torch.float32)  # B=3, C=3
    loader = DummyLoader([y])
    pw = estimate_pos_weight(loader, n_classes=3)
    assert pw.dtype == torch.float32
    assert pw.shape == (3,)
    # 간단 sanity: c0 pos=1 → neg=2 → 2/1=2; c1 pos=1 → 2/1=2; c2 pos=1 → 2/1=2
    assert torch.allclose(pw, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32), atol=1e-6)
