"""
timeseries_example.py — Run a stress test on a simple LSTM with synthetic data.

This example:
  1. Generates a synthetic sine-wave classification dataset.
  2. Wraps a randomly-initialised LSTM in TimeSeriesModelWrapper.
  3. Sweeps all four time-series perturbations across severity 0→1.
  4. Generates a full report under  results/timeseries/.

Usage:
    python -m examples.timeseries_example
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.model_wrapper import TimeSeriesModelWrapper
from core.perturbations import get_perturbations
from core.stress_runner import StressRunner
from core.reporter import Reporter


def generate_synthetic_timeseries(
    num_samples: int = 500,
    seq_len: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a binary classification dataset from sine waves.

    Class 0 → low-frequency sine   (freq ~ 1 Hz)
    Class 1 → high-frequency sine  (freq ~ 3 Hz)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len)

    X, y = [], []
    for _ in range(num_samples):
        label = rng.integers(0, 2)
        freq = 1.0 if label == 0 else 3.0
        signal = np.sin(freq * t) + rng.normal(0, 0.1, seq_len)
        X.append(signal)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


def main(num_samples: int = 500, seed: int = 42) -> None:
    print("=" * 60)
    print("  Failure-First Stress Test  |  Time-Series (Synthetic)")
    print("=" * 60)

    # 1. Generate data
    print(f"\n📦 Generating synthetic time-series ({num_samples} samples)…")
    X, y = generate_synthetic_timeseries(num_samples, seed=seed)
    print(f"   Shape: {X.shape}  Labels: {y.shape}")

    # 2. Build model
    print("🧠 Building SimpleLSTM (random weights)…")
    model = TimeSeriesModelWrapper(device="cpu", input_size=1, num_classes=2)

    # 3. Perturbations
    perturbations = get_perturbations("timeseries")
    print(f"🔧 Perturbations: {[n for n, _ in perturbations]}")

    # 4. Run stress test
    runner = StressRunner(model, perturbations, seed=seed)
    print("\n🏃 Running stress sweep…")
    results = runner.run(X, y)

    # 5. Generate report
    reporter = Reporter(output_dir="results/timeseries")
    reporter.generate_report(results)


if __name__ == "__main__":
    main()
