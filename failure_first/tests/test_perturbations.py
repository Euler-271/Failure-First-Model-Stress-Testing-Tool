"""
test_perturbations.py — Tests for all perturbation functions.

Validates:
  - Shape preservation (single + batch)
  - severity=0 ≈ identity
  - Determinism with same seed
  - Value range for vision perturbations
"""

import numpy as np
import pytest
from core.perturbations import (
    gaussian_noise,
    random_occlusion,
    blur,
    resolution_downsample,
    ts_gaussian_noise,
    temporal_dropout,
    mean_shift,
    variance_shift,
    get_perturbations,
)


# -----------------------------------------------------------------------
# Vision perturbation tests
# -----------------------------------------------------------------------

class TestGaussianNoise:
    def test_shape_single(self, vision_single):
        out = gaussian_noise(vision_single, 0.5, seed=0)
        assert out.shape == vision_single.shape

    def test_shape_batch(self, vision_batch):
        out = gaussian_noise(vision_batch, 0.5, seed=0)
        assert out.shape == vision_batch.shape

    def test_identity_at_zero(self, vision_batch):
        out = gaussian_noise(vision_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, vision_batch)

    def test_clipped_to_01(self, vision_batch):
        out = gaussian_noise(vision_batch, 1.0, seed=0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_deterministic(self, vision_batch):
        a = gaussian_noise(vision_batch, 0.5, seed=42)
        b = gaussian_noise(vision_batch, 0.5, seed=42)
        np.testing.assert_array_equal(a, b)


class TestRandomOcclusion:
    def test_shape_single(self, vision_single):
        out = random_occlusion(vision_single, 0.5, seed=0)
        assert out.shape == vision_single.shape

    def test_shape_batch(self, vision_batch):
        out = random_occlusion(vision_batch, 0.5, seed=0)
        assert out.shape == vision_batch.shape

    def test_identity_at_zero(self, vision_batch):
        out = random_occlusion(vision_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, vision_batch)

    def test_has_zero_patch(self, vision_single):
        out = random_occlusion(vision_single, 0.5, seed=0)
        # At severity=0.5, 50% of pixels should be black in the patch
        assert (out == 0.0).any()

    def test_deterministic(self, vision_batch):
        a = random_occlusion(vision_batch, 0.5, seed=42)
        b = random_occlusion(vision_batch, 0.5, seed=42)
        np.testing.assert_array_equal(a, b)


class TestBlur:
    def test_shape_single(self, vision_single):
        out = blur(vision_single, 0.5, seed=0)
        assert out.shape == vision_single.shape

    def test_shape_batch(self, vision_batch):
        out = blur(vision_batch, 0.5, seed=0)
        assert out.shape == vision_batch.shape

    def test_identity_at_zero(self, vision_batch):
        out = blur(vision_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, vision_batch)

    def test_clipped_to_01(self, vision_batch):
        out = blur(vision_batch, 1.0, seed=0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6


class TestResolutionDownsample:
    def test_shape_single(self, vision_single):
        out = resolution_downsample(vision_single, 0.5, seed=0)
        assert out.shape == vision_single.shape

    def test_shape_batch(self, vision_batch):
        out = resolution_downsample(vision_batch, 0.5, seed=0)
        assert out.shape == vision_batch.shape

    def test_identity_at_zero(self, vision_batch):
        out = resolution_downsample(vision_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, vision_batch)


# -----------------------------------------------------------------------
# Time-series perturbation tests
# -----------------------------------------------------------------------

class TestTsGaussianNoise:
    def test_shape_single(self, timeseries_single):
        out = ts_gaussian_noise(timeseries_single, 0.5, seed=0)
        assert out.shape == timeseries_single.shape

    def test_shape_batch(self, timeseries_batch):
        out = ts_gaussian_noise(timeseries_batch, 0.5, seed=0)
        assert out.shape == timeseries_batch.shape

    def test_identity_at_zero(self, timeseries_batch):
        out = ts_gaussian_noise(timeseries_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, timeseries_batch)

    def test_deterministic(self, timeseries_batch):
        a = ts_gaussian_noise(timeseries_batch, 0.5, seed=42)
        b = ts_gaussian_noise(timeseries_batch, 0.5, seed=42)
        np.testing.assert_array_equal(a, b)


class TestTemporalDropout:
    def test_shape(self, timeseries_batch):
        out = temporal_dropout(timeseries_batch, 0.5, seed=0)
        assert out.shape == timeseries_batch.shape

    def test_identity_at_zero(self, timeseries_batch):
        out = temporal_dropout(timeseries_batch, 0.0, seed=0)
        np.testing.assert_array_equal(out, timeseries_batch)

    def test_has_zeros(self, timeseries_batch):
        out = temporal_dropout(timeseries_batch, 0.99, seed=0)
        assert (out == 0.0).any()


class TestMeanShift:
    def test_shape(self, timeseries_batch):
        out = mean_shift(timeseries_batch, 0.5)
        assert out.shape == timeseries_batch.shape

    def test_identity_at_zero(self, timeseries_batch):
        out = mean_shift(timeseries_batch, 0.0)
        np.testing.assert_allclose(out, timeseries_batch, atol=1e-6)


class TestVarianceShift:
    def test_shape(self, timeseries_batch):
        out = variance_shift(timeseries_batch, 0.5)
        assert out.shape == timeseries_batch.shape

    def test_identity_at_zero(self, timeseries_batch):
        out = variance_shift(timeseries_batch, 0.0)
        np.testing.assert_allclose(out, timeseries_batch, atol=1e-6)


# -----------------------------------------------------------------------
# Registry tests
# -----------------------------------------------------------------------

class TestPerturbationRegistry:
    def test_vision_returns_list(self):
        perts = get_perturbations("vision")
        assert len(perts) == 4
        for name, fn in perts:
            assert callable(fn)

    def test_timeseries_returns_list(self):
        perts = get_perturbations("timeseries")
        assert len(perts) == 4
        for name, fn in perts:
            assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_perturbations("audio")
