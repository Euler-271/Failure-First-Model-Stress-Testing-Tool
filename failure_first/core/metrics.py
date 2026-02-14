"""
metrics.py — Robustness metrics for the stress testing framework.

All functions accept numpy arrays and return plain Python scalars so they
serialise cleanly to JSON/CSV.
"""

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Basic classification metrics
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy (fraction of correct predictions)."""
    return float(np.mean(y_true == y_pred))


def compute_avg_confidence(probs: np.ndarray) -> float:
    """
    Mean of the maximum predicted probability across all samples.

    A confident model has this close to 1.0; confidence collapse drives it
    toward 1/num_classes.
    """
    max_probs = np.max(probs, axis=1)
    return float(np.mean(max_probs))


# ---------------------------------------------------------------------------
# Prediction stability
# ---------------------------------------------------------------------------

def compute_flip_rate(
    baseline_preds: np.ndarray,
    perturbed_preds: np.ndarray,
) -> float:
    """
    Fraction of samples whose predicted label changed relative to the
    baseline (severity=0) predictions.

    0.0 = perfectly stable, 1.0 = every prediction flipped.
    """
    return float(np.mean(baseline_preds != perturbed_preds))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (binned).

    Partitions predictions into *n_bins* equally-spaced confidence bins and
    computes the weighted average of |accuracy − confidence| per bin.
    """
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Samples falling into this confidence bin
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(np.mean(preds[mask] == y_true[mask]))
        bin_conf = float(np.mean(max_probs[mask]))
        bin_weight = mask.sum() / len(y_true)
        ece += abs(bin_acc - bin_conf) * bin_weight

    return float(ece)


# ---------------------------------------------------------------------------
# Degradation summary
# ---------------------------------------------------------------------------

def compute_degradation_slope(
    severities: List[float],
    accuracies: List[float],
) -> float:
    """
    Fit a simple linear regression of accuracy vs severity and return the
    slope.  A slope of 0 means the model is rock-solid; large negative
    slopes indicate rapid failure.
    """
    # Simple linear fit using numpy (avoids sklearn/scipy dependency)
    coeffs = np.polyfit(severities, accuracies, deg=1)
    return float(coeffs[0])  # coeffs[0] = slope
