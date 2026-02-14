"""
perturbations.py — Parameterized perturbation engine.

Every perturbation function follows the same contract:
    perturb(x, severity, seed=None) → x_perturbed

- x:        np.ndarray (single sample or batch)
- severity: float in [0, 1]  (0 = no change, 1 = maximum distortion)
- seed:     optional int for reproducibility
- returns:  np.ndarray same shape as x

Vision perturbations operate on (H, W, C) float images in [0, 1].
Time-series perturbations operate on (seq_len,) or (N, seq_len) arrays.
"""

from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy Gaussian blur helper (avoids scipy dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Create a normalised 1-D Gaussian kernel."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply separable Gaussian blur to an (H, W, C) image using numpy."""
    radius = int(np.ceil(3 * sigma))
    if radius == 0:
        return img.copy()
    kernel = _gaussian_kernel_1d(sigma, radius)

    # Pad → convolve rows → convolve cols, per channel
    padded = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    h, w, c = padded.shape
    # Horizontal pass
    temp = np.zeros_like(padded)
    for k in range(len(kernel)):
        temp[:, radius:w - radius, :] += padded[:, k:k + w - 2 * radius, :] * kernel[k]
    # Vertical pass
    result = np.zeros((img.shape[0], img.shape[1], c), dtype=img.dtype)
    for k in range(len(kernel)):
        result += temp[k:k + img.shape[0], radius:radius + img.shape[1], :] * kernel[k]
    return result


# ===================================================================
# Vision perturbations  (input: H×W×C float32 image in [0,1])
# ===================================================================

def gaussian_noise(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise with std = severity * 0.5."""
    rng = np.random.default_rng(seed)
    sigma = severity * 0.5  # max std = 0.5 at severity 1.0
    noise = rng.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return np.clip(x + noise, 0.0, 1.0)


def random_occlusion(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Black out a random rectangular patch whose area scales with severity.

    Accepts single (H,W,C) or batch (N,H,W,C) input.
    """
    rng = np.random.default_rng(seed)
    out = x.copy()

    # Handle batch vs single
    if x.ndim == 4:
        n, h, w = x.shape[0], x.shape[1], x.shape[2]
        patch_h = int(h * severity)
        patch_w = int(w * severity)
        if patch_h == 0 or patch_w == 0:
            return out
        for i in range(n):
            y0 = rng.integers(0, max(h - patch_h, 1))
            x0 = rng.integers(0, max(w - patch_w, 1))
            out[i, y0 : y0 + patch_h, x0 : x0 + patch_w] = 0.0
        return out

    h, w = x.shape[:2]
    patch_h = int(h * severity)
    patch_w = int(w * severity)
    if patch_h == 0 or patch_w == 0:
        return out
    y0 = rng.integers(0, max(h - patch_h, 1))
    x0 = rng.integers(0, max(w - patch_w, 1))
    out[y0 : y0 + patch_h, x0 : x0 + patch_w] = 0.0
    return out


def blur(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply Gaussian blur with sigma scaling with severity.

    Accepts single (H,W,C) or batch (N,H,W,C) input.
    """
    sigma = severity * 5.0
    if sigma < 1e-3:
        return x.copy()

    if x.ndim == 4:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.clip(_apply_gaussian_blur(x[i], sigma), 0.0, 1.0)
        return out

    return np.clip(_apply_gaussian_blur(x, sigma), 0.0, 1.0)


def resolution_downsample(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Downsample then upsample; at severity=1 resolution drops to 2×2.

    Accepts single (H,W,C) or batch (N,H,W,C) input.
    """
    if x.ndim == 4:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = _resolution_downsample_single(x[i], severity)
        return out
    return _resolution_downsample_single(x, severity)


def _resolution_downsample_single(
    x: np.ndarray, severity: float
) -> np.ndarray:
    """Single-image resolution downsample helper."""
    h, w = x.shape[:2]
    factor = max(1.0 - severity * 0.9, 0.1)
    new_h = max(int(h * factor), 2)
    new_w = max(int(w * factor), 2)

    if new_h >= h and new_w >= w:
        return x.copy()

    row_idx = np.linspace(0, h - 1, new_h).astype(int)
    col_idx = np.linspace(0, w - 1, new_w).astype(int)
    small = x[np.ix_(row_idx, col_idx)]

    row_idx_up = np.linspace(0, new_h - 1, h).astype(int)
    col_idx_up = np.linspace(0, new_w - 1, w).astype(int)
    restored = small[np.ix_(row_idx_up, col_idx_up)]
    return restored.astype(x.dtype)


# ===================================================================
# Time-series perturbations  (input: (seq_len,) or (N, seq_len))
# ===================================================================

def ts_gaussian_noise(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise scaled by severity × data std."""
    rng = np.random.default_rng(seed)
    sigma = severity * float(np.std(x) + 1e-8)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return x + noise


def temporal_dropout(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Zero-out a fraction of timesteps proportional to severity."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    mask = rng.random(x.shape) < severity  # True → drop
    out[mask] = 0.0
    return out


def mean_shift(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Shift the mean by severity × data std."""
    shift = severity * float(np.std(x) + 1e-8) * 2.0
    return x + shift


def variance_shift(
    x: np.ndarray,
    severity: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Scale variance — at severity=1 the data std doubles."""
    mu = np.mean(x)
    scale = 1.0 + severity  # 1.0 → 2.0
    return mu + (x - mu) * scale


# ===================================================================
# Registry helpers
# ===================================================================

# Each entry is (human-readable name, callable)
VISION_PERTURBATIONS: List[tuple[str, Callable]] = [
    ("gaussian_noise", gaussian_noise),
    ("random_occlusion", random_occlusion),
    ("blur", blur),
    ("resolution_downsample", resolution_downsample),
]

TIMESERIES_PERTURBATIONS: List[tuple[str, Callable]] = [
    ("ts_gaussian_noise", ts_gaussian_noise),
    ("temporal_dropout", temporal_dropout),
    ("mean_shift", mean_shift),
    ("variance_shift", variance_shift),
]


def get_perturbations(domain: str) -> List[tuple[str, Callable]]:
    """Return the perturbation list for a given domain ('vision' or 'timeseries')."""
    registry = {
        "vision": VISION_PERTURBATIONS,
        "timeseries": TIMESERIES_PERTURBATIONS,
    }
    if domain not in registry:
        raise ValueError(f"Unknown domain '{domain}'. Choose from {list(registry)}")
    return registry[domain]
