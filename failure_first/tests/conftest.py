"""
conftest.py — Shared pytest fixtures for the stress testing framework.
"""

import numpy as np
import pytest


@pytest.fixture
def vision_batch():
    """Batch of 20 random 32x32 RGB images in [0,1]."""
    rng = np.random.default_rng(0)
    return rng.random((20, 32, 32, 3)).astype(np.float32)


@pytest.fixture
def vision_single():
    """Single 32x32 RGB image in [0,1]."""
    rng = np.random.default_rng(0)
    return rng.random((32, 32, 3)).astype(np.float32)


@pytest.fixture
def timeseries_batch():
    """Batch of 20 time-series of length 50."""
    rng = np.random.default_rng(0)
    return rng.random((20, 50)).astype(np.float32)


@pytest.fixture
def timeseries_single():
    """Single time-series of length 50."""
    rng = np.random.default_rng(0)
    return rng.random((50,)).astype(np.float32)


@pytest.fixture
def labels_binary():
    """Binary labels for 20 samples."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=20)


@pytest.fixture
def labels_10class():
    """10-class labels for 20 samples."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 10, size=20)


@pytest.fixture
def perfect_probs():
    """Probabilities that perfectly match labels (10 classes, 20 samples)."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=20)
    probs = np.zeros((20, 10), dtype=np.float32)
    probs[np.arange(20), labels] = 1.0
    return probs, labels


@pytest.fixture
def random_probs():
    """Random probability distributions (10 classes, 20 samples)."""
    rng = np.random.default_rng(0)
    raw = rng.random((20, 10)).astype(np.float32)
    probs = raw / raw.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 10, size=20)
    return probs, labels
