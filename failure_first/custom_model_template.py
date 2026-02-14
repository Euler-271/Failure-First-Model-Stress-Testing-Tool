"""
custom_model_template.py — Template for custom model upload.

Implement the two functions below, then upload this file through the
Failure-First Stress Testing web dashboard.

Your file must define:
    predict(X)       -> np.ndarray of class labels, shape (N,)
    predict_proba(X) -> np.ndarray of probabilities, shape (N, num_classes)

X is a numpy array whose shape depends on your domain:
    Vision:      (N, H, W, C)  float32 in [0, 1]
    Time-series: (N, seq_len)  float32
"""

import numpy as np


def predict(X: np.ndarray) -> np.ndarray:
    """
    Return predicted class indices for the input batch.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape depends on your model domain.

    Returns
    -------
    np.ndarray
        Integer class labels, shape (N,).
    """
    # --- Replace with your model logic ---
    # Example: random predictions for 2 classes
    return np.random.randint(0, 2, size=X.shape[0])


def predict_proba(X: np.ndarray) -> np.ndarray:
    """
    Return class probabilities for the input batch.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape depends on your model domain.

    Returns
    -------
    np.ndarray
        Probabilities, shape (N, num_classes). Rows must sum to 1.
    """
    # --- Replace with your model logic ---
    # Example: random probabilities for 2 classes
    probs = np.random.rand(X.shape[0], 2)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
