#!/usr/bin/env python3
"""
stress_test.py — CLI entry point for the Failure-First Stress Testing Framework.

Usage examples:
    python stress_test.py --model vision   --dataset cifar10    --num-samples 200
    python stress_test.py --model timeseries --dataset synthetic --num-samples 300 --seed 123
"""

import argparse
import sys
from typing import Tuple

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Failure-First Model Stress Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python stress_test.py --model vision --dataset cifar10\n"
            "  python stress_test.py --model timeseries --dataset synthetic\n"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["vision", "timeseries"],
        required=True,
        help="Model type to stress-test.",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "synthetic"],
        required=True,
        help="Dataset to use (cifar10 for vision, synthetic for timeseries).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: results/<model>).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_cifar10(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 test images as float32 (N, 32, 32, 3) in [0, 1]."""
    import torchvision
    import torchvision.transforms as T

    ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=T.ToTensor()
    )
    images, labels = [], []
    for i in range(min(num_samples, len(ds))):
        img, lbl = ds[i]
        images.append(img.permute(1, 2, 0).numpy())
        labels.append(lbl)
    return np.array(images, dtype=np.float32), np.array(labels)


def load_synthetic_timeseries(
    num_samples: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic sine-wave binary classification dataset."""
    rng = np.random.default_rng(seed)
    seq_len = 50
    t = np.linspace(0, 2 * np.pi, seq_len)

    X, y = [], []
    for _ in range(num_samples):
        label = rng.integers(0, 2)
        freq = 1.0 if label == 0 else 3.0
        signal = np.sin(freq * t) + rng.normal(0, 0.1, seq_len)
        X.append(signal)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Failure-First Model Stress Testing Framework")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Samples:  {args.num_samples}")
    print(f"  Seed:     {args.seed}")
    print("=" * 60)

    # ---- Import core modules -----------------------------------------------
    from core.model_wrapper import VisionModelWrapper, TimeSeriesModelWrapper
    from core.perturbations import get_perturbations
    from core.stress_runner import StressRunner
    from core.reporter import Reporter

    # ---- Load dataset ------------------------------------------------------
    if args.model == "vision":
        if args.dataset != "cifar10":
            print(f"⚠️  Dataset '{args.dataset}' not supported for vision. Using cifar10.")
        print(f"\n📦 Loading CIFAR-10 ({args.num_samples} samples)…")
        X, y = load_cifar10(args.num_samples)
    else:
        if args.dataset != "synthetic":
            print(f"⚠️  Dataset '{args.dataset}' not supported for timeseries. Using synthetic.")
        print(f"\n📦 Generating synthetic time-series ({args.num_samples} samples)…")
        X, y = load_synthetic_timeseries(args.num_samples, args.seed)

    print(f"   Shape: {X.shape}  Labels: {y.shape}")

    # ---- Build model -------------------------------------------------------
    if args.model == "vision":
        print("🧠 Building SimpleCNN (random weights)…")
        model = VisionModelWrapper(device="cpu", num_classes=10)
    else:
        print("🧠 Building SimpleLSTM (random weights)…")
        model = TimeSeriesModelWrapper(device="cpu", input_size=1, num_classes=2)

    # ---- Perturbations -----------------------------------------------------
    perturbations = get_perturbations(args.model)
    print(f"🔧 Perturbations: {[n for n, _ in perturbations]}")

    # ---- Stress test -------------------------------------------------------
    runner = StressRunner(model, perturbations, seed=args.seed)
    print("\n🏃 Running stress sweep…")

    # tqdm progress bar via callback
    total_steps = len(perturbations) * 11  # 11 severity levels
    pbar = tqdm(total=total_steps, desc="Stress test", unit="step")

    def _progress(step, total, msg):
        pbar.set_postfix_str(msg)
        pbar.update(1)

    results = runner.run(X, y, progress_callback=_progress)
    pbar.close()

    # ---- Report ------------------------------------------------------------
    output_dir = args.output_dir or f"results/{args.model}"
    reporter = Reporter(output_dir=output_dir)
    reporter.generate_report(results)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
