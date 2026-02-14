"""
test_stress_runner.py — Integration tests for StressRunner.
"""

import numpy as np
import pytest
from core.model_wrapper import VisionModelWrapper, TimeSeriesModelWrapper
from core.perturbations import get_perturbations
from core.stress_runner import StressRunner


class TestStressRunnerVision:
    """Integration test with real VisionModelWrapper."""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(0)
        X = rng.random((10, 32, 32, 3)).astype(np.float32)
        y = rng.integers(0, 10, size=10)
        model = VisionModelWrapper(device="cpu", num_classes=10)
        perts = get_perturbations("vision")
        return X, y, model, perts

    def test_returns_required_keys(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42, severity_steps=3)
        results = runner.run(X, y)

        assert "severities" in results
        assert "perturbations" in results
        assert "worst_failures" in results

    def test_severities_count(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42, severity_steps=5)
        results = runner.run(X, y)
        assert len(results["severities"]) == 5

    def test_all_perturbations_present(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42, severity_steps=3)
        results = runner.run(X, y)
        for name, _ in perts:
            assert name in results["perturbations"]

    def test_metric_lists_correct_length(self, setup):
        X, y, model, perts = setup
        steps = 3
        runner = StressRunner(model, perts, seed=42, severity_steps=steps)
        results = runner.run(X, y)
        for name, metrics in results["perturbations"].items():
            assert len(metrics["accuracy"]) == steps
            assert len(metrics["avg_confidence"]) == steps
            assert len(metrics["flip_rate"]) == steps
            assert len(metrics["ece"]) == steps
            assert "degradation_slope" in metrics

    def test_worst_failures_max_20(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42)
        results = runner.run(X, y)
        assert len(results["worst_failures"]) <= 20

    def test_worst_failure_structure(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42)
        results = runner.run(X, y)
        if results["worst_failures"]:
            f = results["worst_failures"][0]
            assert "sample_idx" in f
            assert "true_label" in f
            assert "pred_label" in f
            assert "confidence" in f
            assert "perturbation" in f
            assert "severity" in f


class TestStressRunnerTimeSeries:
    """Integration test with TimeSeriesModelWrapper."""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(0)
        X = rng.random((10, 50)).astype(np.float32)
        y = rng.integers(0, 2, size=10)
        model = TimeSeriesModelWrapper(device="cpu", input_size=1, num_classes=2)
        perts = get_perturbations("timeseries")
        return X, y, model, perts

    def test_returns_required_keys(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42, severity_steps=3)
        results = runner.run(X, y)
        assert "severities" in results
        assert "perturbations" in results
        assert "worst_failures" in results

    def test_all_perturbations_present(self, setup):
        X, y, model, perts = setup
        runner = StressRunner(model, perts, seed=42, severity_steps=3)
        results = runner.run(X, y)
        for name, _ in perts:
            assert name in results["perturbations"]


class TestProgressCallback:
    """Verify the progress_callback is called correctly."""

    def test_callback_invoked(self):
        rng = np.random.default_rng(0)
        X = rng.random((5, 50)).astype(np.float32)
        y = rng.integers(0, 2, size=5)
        model = TimeSeriesModelWrapper(device="cpu", input_size=1, num_classes=2)
        perts = get_perturbations("timeseries")
        runner = StressRunner(model, perts, seed=42, severity_steps=3)

        calls = []
        def cb(step, total, msg):
            calls.append((step, total, msg))

        runner.run(X, y, progress_callback=cb)

        # 4 perturbations × 3 steps = 12 calls
        assert len(calls) == 12
        # Last call should be (12, 12, ...)
        assert calls[-1][0] == 12
        assert calls[-1][1] == 12
