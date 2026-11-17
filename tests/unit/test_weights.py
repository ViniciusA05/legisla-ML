"""Unit tests for Seam 3: Inverse-frequency class weight calculation and seed determinism."""

import numpy as np
import torch

from src.train_legislabl_bert import calculate_class_weights, set_seed


def test_inverse_frequency_minority_prioritization():
    """Minority class receives strictly higher weight and weighted sum equals N_total."""
    y = np.array([0] * 75 + [1] * 25)
    weights = calculate_class_weights(y, num_classes=2)

    assert weights.shape == (2,)
    assert weights.dtype == torch.float
    assert weights[1].item() > weights[0].item()
    assert np.isclose(weights[0].item(), 100 / (2 * 75))
    assert np.isclose(weights[1].item(), 100 / (2 * 25))
    assert np.isclose(weights[0].item() * 75 + weights[1].item() * 25, 100.0)


def test_balanced_classes_and_zero_division_guard():
    """Balanced classes yield 1.0 weights and missing class does not raise ZeroDivisionError."""
    y_bal = np.array([0, 1, 0, 1])
    w_bal = calculate_class_weights(y_bal, num_classes=2)
    assert torch.allclose(w_bal, torch.tensor([1.0, 1.0]))

    y_single = np.array([0, 0, 0, 0])
    w_single = calculate_class_weights(y_single, num_classes=2)
    assert torch.isfinite(w_single).all()


def test_set_seed_reproducibility():
    """Fixing seed produces identical random draws across PyTorch and NumPy."""
    set_seed(123)
    a_torch = torch.randn(4)
    a_np = np.random.rand(4)

    set_seed(123)
    b_torch = torch.randn(4)
    b_np = np.random.rand(4)

    assert torch.allclose(a_torch, b_torch)
    assert np.allclose(a_np, b_np)
