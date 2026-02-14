"""
vision_example.py — Run a stress test on a simple CNN using CIFAR-10.

This example:
  1. Downloads the CIFAR-10 test set (or uses a cached copy).
  2. Wraps a lightweight, randomly-initialised CNN in VisionModelWrapper.
  3. Sweeps all four vision perturbations across severity 0→1.
  4. Generates a full report under  results/vision/.

Usage:
    python -m examples.vision_example
"""

import sys
import os

# Ensure project root is on the path so `core` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torchvision
import torchvision.transforms as T

from core.model_wrapper import VisionModelWrapper
from core.perturbations import get_perturbations
from core.stress_runner import StressRunner
from core.reporter import Reporter


def load_cifar10(num_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test images as float32 (N, 32, 32, 3) in [0, 1]."""
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=T.ToTensor()
    )
    images, labels = [], []
    for i in range(min(num_samples, len(ds))):
        img, lbl = ds[i]
        # ToTensor gives (C, H, W) — convert back to (H, W, C)
        images.append(img.permute(1, 2, 0).numpy())
        labels.append(lbl)
    return np.array(images, dtype=np.float32), np.array(labels)


def main(num_samples: int = 500, seed: int = 42) -> None:
    print("=" * 60)
    print("  Failure-First Stress Test  |  Vision (CIFAR-10)")
    print("=" * 60)

    # 1. Load data
    print(f"\n📦 Loading CIFAR-10 ({num_samples} samples)…")
    X, y = load_cifar10(num_samples)
    print(f"   Shape: {X.shape}  Labels: {y.shape}")

    # 2. Build model (randomly initialised — we're testing failure, not accuracy)
    print("🧠 Building SimpleCNN (random weights)…")
    model = VisionModelWrapper(device="cpu", num_classes=10)

    # 3. Perturbations
    perturbations = get_perturbations("vision")
    print(f"🔧 Perturbations: {[n for n, _ in perturbations]}")

    # 4. Run stress test
    runner = StressRunner(model, perturbations, seed=seed)
    print("\n🏃 Running stress sweep…")
    results = runner.run(X, y)

    # 5. Generate report
    reporter = Reporter(output_dir="results/vision")
    reporter.generate_report(results)


if __name__ == "__main__":
    main()
