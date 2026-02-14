"""
test_metrics.py — Unit tests for core.metrics functions.
"""

import numpy as np
import pytest
from core.metrics import (
    compute_accuracy,
    compute_avg_confidence,
    compute_degradation_slope,
    compute_ece,
    compute_flip_rate,
)


class TestComputeAccuracy:
    def test_perfect(self):
        y = np.array([0, 1, 2, 3])
        assert compute_accuracy(y, y) == 1.0

    def test_zero(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([3, 2, 1, 0])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial(self):
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 0, 0])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_single_sample(self):
        assert compute_accuracy(np.array([1]), np.array([1])) == 1.0


class TestComputeAvgConfidence:
    def test_perfect_confidence(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert compute_avg_confidence(probs) == 1.0

    def test_uniform_confidence(self):
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert compute_avg_confidence(probs) == 0.5

    def test_range(self):
        rng = np.random.default_rng(42)
        raw = rng.random((100, 5))
        probs = raw / raw.sum(axis=1, keepdims=True)
        conf = compute_avg_confidence(probs)
        assert 0.0 <= conf <= 1.0


class TestComputeFlipRate:
    def test_no_flips(self):
        preds = np.array([0, 1, 2, 3])
        assert compute_flip_rate(preds, preds) == 0.0

    def test_all_flipped(self):
        baseline = np.array([0, 1, 2, 3])
        flipped = np.array([3, 2, 1, 0])
        assert compute_flip_rate(baseline, flipped) == 1.0

    def test_half_flipped(self):
        baseline = np.array([0, 1, 2, 3])
        partial = np.array([0, 1, 0, 0])
        assert compute_flip_rate(baseline, partial) == pytest.approx(0.5)


class TestComputeECE:
    def test_perfectly_calibrated(self):
        # Perfect predictions have low ECE
        y = np.array([0, 1, 0, 1])
        probs = np.array([
            [0.99, 0.01], [0.01, 0.99],
            [0.99, 0.01], [0.01, 0.99],
        ])
        ece = compute_ece(y, probs, n_bins=10)
        assert ece < 0.05  # near-zero

    def test_ece_range(self):
        rng = np.random.default_rng(42)
        raw = rng.random((100, 5))
        probs = raw / raw.sum(axis=1, keepdims=True)
        y = rng.integers(0, 5, size=100)
        ece = compute_ece(y, probs)
        assert 0.0 <= ece <= 1.0

    def test_empty_bins_handled(self):
        # All confidences in one bin
        probs = np.array([[0.95, 0.05]] * 10)
        y = np.zeros(10, dtype=int)
        ece = compute_ece(y, probs, n_bins=100)
        assert isinstance(ece, float)


class TestComputeDegradationSlope:
    def test_flat(self):
        sevs = [0.0, 0.5, 1.0]
        accs = [0.9, 0.9, 0.9]
        slope = compute_degradation_slope(sevs, accs)
        assert abs(slope) < 1e-10

    def test_negative_slope(self):
        sevs = [0.0, 0.5, 1.0]
        accs = [1.0, 0.5, 0.0]
        slope = compute_degradation_slope(sevs, accs)
        assert slope == pytest.approx(-1.0)

    def test_positive_slope(self):
        sevs = [0.0, 0.5, 1.0]
        accs = [0.0, 0.5, 1.0]
        slope = compute_degradation_slope(sevs, accs)
        assert slope == pytest.approx(1.0)
