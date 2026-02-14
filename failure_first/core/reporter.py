"""
reporter.py — Reporting system for stress test results.

Generates:
  • JSON log          — full structured results
  • CSV summary       — flattened table (severity × perturbation × metrics)
  • PNG plots         — accuracy curve, confidence drift, flip rate
  • Worst failures    — JSON of the top-20 highest-confidence misclassifications
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd


class Reporter:
    """Generates all output artefacts from a stress-test results dict."""

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self, results: Dict[str, Any]) -> None:
        """Run the full reporting pipeline."""
        self.save_json(results, self.output_dir / "stress_results.json")
        self.save_csv(results, self.output_dir / "stress_results.csv")
        self.plot_accuracy_curve(results, self.output_dir / "accuracy_curve.png")
        self.plot_confidence_drift(results, self.output_dir / "confidence_drift.png")
        self.plot_flip_rate(results, self.output_dir / "flip_rate.png")
        self.save_worst_failures(results, self.output_dir / "worst_failures.json")

        print(f"\n✅ Report saved to {self.output_dir.resolve()}/")

    # ------------------------------------------------------------------
    # Data exports
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(results: Dict[str, Any], path: os.PathLike) -> None:
        """Write the full results dict as pretty-printed JSON."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  📄 JSON log → {path}")

    @staticmethod
    def save_csv(results: Dict[str, Any], path: os.PathLike) -> None:
        """Flatten results into a tidy CSV: one row per (perturbation, severity)."""
        rows = []
        severities = results["severities"]
        for pert_name, metrics in results["perturbations"].items():
            for i, sev in enumerate(severities):
                rows.append({
                    "perturbation": pert_name,
                    "severity": sev,
                    "accuracy": metrics["accuracy"][i],
                    "avg_confidence": metrics["avg_confidence"][i],
                    "flip_rate": metrics["flip_rate"][i],
                    "ece": metrics["ece"][i],
                })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"  📊 CSV summary → {path}")

    @staticmethod
    def save_worst_failures(
        results: Dict[str, Any], path: os.PathLike
    ) -> None:
        """Save top-20 worst failure samples as JSON."""
        with open(path, "w") as f:
            json.dump(results.get("worst_failures", []), f, indent=2)
        print(f"  💀 Worst failures → {path}")

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _new_figure(self) -> tuple:
        """Return a fresh (fig, ax) pair with a clean dark-grid style."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.grid(True, alpha=0.3)
        return fig, ax

    def plot_accuracy_curve(
        self, results: Dict[str, Any], path: os.PathLike
    ) -> None:
        """Accuracy vs. severity for each perturbation."""
        fig, ax = self._new_figure()
        sev = results["severities"]
        for pert_name, metrics in results["perturbations"].items():
            slope = metrics.get("degradation_slope", 0)
            label = f"{pert_name}  (slope={slope:+.3f})"
            ax.plot(sev, metrics["accuracy"], marker="o", label=label)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Perturbation Severity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📈 Accuracy curve → {path}")

    def plot_confidence_drift(
        self, results: Dict[str, Any], path: os.PathLike
    ) -> None:
        """Average confidence vs. severity for each perturbation."""
        fig, ax = self._new_figure()
        sev = results["severities"]
        for pert_name, metrics in results["perturbations"].items():
            ax.plot(sev, metrics["avg_confidence"], marker="s", label=pert_name)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Average Confidence")
        ax.set_title("Confidence Drift Under Perturbation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  📉 Confidence drift → {path}")

    def plot_flip_rate(
        self, results: Dict[str, Any], path: os.PathLike
    ) -> None:
        """Prediction flip rate vs. severity for each perturbation."""
        fig, ax = self._new_figure()
        sev = results["severities"]
        for pert_name, metrics in results["perturbations"].items():
            ax.plot(sev, metrics["flip_rate"], marker="^", label=pert_name)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Flip Rate")
        ax.set_title("Prediction Instability (Flip Rate)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  🔄 Flip rate plot → {path}")
