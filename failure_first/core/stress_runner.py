"""
stress_runner.py — Stress testing engine.

StressRunner sweeps perturbation severity from 0.0 → 1.0, runs inference at
each level, and collects robustness metrics.  The output is a structured dict
ready for the Reporter.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import (
    compute_accuracy,
    compute_avg_confidence,
    compute_degradation_slope,
    compute_ece,
    compute_flip_rate,
)
from .model_wrapper import ModelWrapper


class StressRunner:
    """
    Orchestrates a full stress-test sweep for one model + dataset.

    Parameters
    ----------
    model : ModelWrapper
        The model to test.
    perturbations : list of (name, callable)
        Each callable has signature  fn(x, severity, seed) → x'.
    severity_steps : int
        Number of severity levels (default 11 → 0.0, 0.1, …, 1.0).
    seed : int or None
        Global seed for reproducibility.
    """

    def __init__(
        self,
        model: ModelWrapper,
        perturbations: List[Tuple[str, Callable]],
        severity_steps: int = 11,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.perturbations = perturbations
        self.severity_steps = severity_steps
        self.seed = seed

        # Severity levels: [0.0, 0.1, 0.2, …, 1.0]
        self.severities = [
            round(i / (severity_steps - 1), 2) for i in range(severity_steps)
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full sweep and return structured results.

        Parameters
        ----------
        X : np.ndarray   — input samples (images or time-series).
        y : np.ndarray   — ground-truth labels.
        progress_callback : callable, optional
            Called as callback(step, total_steps, message) after each
            (perturbation, severity) evaluation. Useful for progress bars.

        Returns
        -------
        dict with keys:
            "severities"   — list of float
            "perturbations" — dict[pert_name → dict of metric lists]
            "worst_failures" — list of dicts (top 20 worst samples)
        """
        # Baseline predictions (severity = 0 — unmodified data)
        baseline_probs = self.model.predict_proba(X)
        baseline_preds = np.argmax(baseline_probs, axis=1)

        results: Dict[str, Any] = {
            "severities": self.severities,
            "perturbations": {},
            "worst_failures": [],
        }

        # Collect worst-failure candidates across all perturbations/severities
        all_failure_records: List[Dict[str, Any]] = []

        # Total steps for progress tracking
        total_steps = len(self.perturbations) * len(self.severities)
        current_step = 0

        for pert_name, pert_fn in self.perturbations:
            metrics_log: Dict[str, List[float]] = {
                "accuracy": [],
                "avg_confidence": [],
                "flip_rate": [],
                "ece": [],
            }

            for sev in self.severities:
                # Apply perturbation to entire batch at once (vectorized)
                step_seed = (
                    None if self.seed is None
                    else abs(self.seed + hash(pert_name) + int(sev * 100)) % (2**31)
                )
                X_pert = pert_fn(X, sev, seed=step_seed)

                # Inference
                probs = self.model.predict_proba(X_pert)
                preds = np.argmax(probs, axis=1)

                # Metrics
                acc = compute_accuracy(y, preds)
                conf = compute_avg_confidence(probs)
                flip = compute_flip_rate(baseline_preds, preds)
                ece = compute_ece(y, probs)

                metrics_log["accuracy"].append(acc)
                metrics_log["avg_confidence"].append(conf)
                metrics_log["flip_rate"].append(flip)
                metrics_log["ece"].append(ece)

                # Collect failure samples (high-confidence misclassifications)
                self._collect_failures(
                    all_failure_records, preds, probs, y,
                    pert_name, sev,
                )

                # Progress callback
                current_step += 1
                if progress_callback is not None:
                    progress_callback(
                        current_step, total_steps,
                        f"{pert_name} @ severity {sev:.1f}",
                    )

            # Degradation slope for this perturbation
            slope = compute_degradation_slope(
                self.severities, metrics_log["accuracy"]
            )
            metrics_log["degradation_slope"] = slope  # type: ignore[assignment]

            results["perturbations"][pert_name] = metrics_log

        # Keep top-20 worst failures (highest confidence wrong predictions)
        all_failure_records.sort(key=lambda r: r["confidence"], reverse=True)
        results["worst_failures"] = all_failure_records[:20]

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    @staticmethod
    def _collect_failures(
        records: List[Dict[str, Any]],
        preds: np.ndarray,
        probs: np.ndarray,
        y_true: np.ndarray,
        pert_name: str,
        severity: float,
    ) -> None:
        """Record misclassified samples with their confidence."""
        wrong = preds != y_true
        wrong_indices = np.where(wrong)[0]
        for idx in wrong_indices:
            records.append({
                "sample_idx": int(idx),
                "true_label": int(y_true[idx]),
                "pred_label": int(preds[idx]),
                "confidence": float(np.max(probs[idx])),
                "perturbation": pert_name,
                "severity": severity,
            })
