"""
model_wrapper.py — Model interface for the stress testing framework.

Provides an abstract ModelWrapper base class and concrete implementations
for vision (CNN) and time-series (LSTM) models. Each wrapper standardizes
predict() and predict_proba() so the stress runner can treat all models
identically.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class ModelWrapper(ABC):
    """
    Base interface every model must implement.

    predict(x)       → class labels   (np.ndarray of ints)
    predict_proba(x) → probabilities  (np.ndarray, shape [N, C])
    """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices for input batch *x*."""
        ...

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for input batch *x*."""
        ...


# ---------------------------------------------------------------------------
# Simple CNN for CIFAR-10 style images
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Lightweight 3-conv-layer CNN for 32×32 RGB images (10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                  # → 1×1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VisionModelWrapper(ModelWrapper):
    """
    Wraps a PyTorch image classifier.

    Expects numpy input of shape (N, H, W, C) with values in [0, 1].
    Internally converts to (N, C, H, W) tensors.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        num_classes: int = 10,
    ) -> None:
        self.device = torch.device(device)
        # Use supplied model or build a simple default CNN
        self.model = model if model is not None else SimpleCNN(num_classes)
        self.model.to(self.device).eval()

    # -- internal helper -----------------------------------------------------

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert (N,H,W,C) float numpy → (N,C,H,W) torch tensor."""
        if x.ndim == 3:
            x = x[np.newaxis, ...]            # single image → batch of 1
        # Channels-last → channels-first
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(self.device)
        return t

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (N, num_classes)."""
        logits = self.model(self._to_tensor(x))
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices, shape (N,)."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)


# ---------------------------------------------------------------------------
# Simple LSTM for univariate time-series classification
# ---------------------------------------------------------------------------

class SimpleLSTM(nn.Module):
    """Single-layer LSTM classifier for 1-D sequences."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)        # h_n: (1, batch, hidden)
        out = self.fc(h_n.squeeze(0))     # → (batch, num_classes)
        return out


class TimeSeriesModelWrapper(ModelWrapper):
    """
    Wraps a PyTorch time-series classifier.

    Expects numpy input of shape (N, seq_len) or (N, seq_len, 1).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
        input_size: int = 1,
        num_classes: int = 2,
    ) -> None:
        self.device = torch.device(device)
        self.model = (
            model if model is not None
            else SimpleLSTM(input_size=input_size, num_classes=num_classes)
        )
        self.model.to(self.device).eval()

    # -- internal helper -----------------------------------------------------

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Ensure shape (N, seq_len, 1) and convert to tensor."""
        if x.ndim == 1:
            x = x[np.newaxis, :, np.newaxis]
        elif x.ndim == 2:
            x = x[:, :, np.newaxis]
        return torch.from_numpy(x).float().to(self.device)

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.model(self._to_tensor(x))
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)
