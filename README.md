<p align="center">
  <h1 align="center">⚡ Failure-First Model Stress Testing</h1>
  <p align="center">
    A modular, model-agnostic framework that <strong>intentionally stress-tests ML models under controlled degradation</strong> and measures how they fail — not how they succeed.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?style=flat-square" />
  <img src="https://img.shields.io/badge/tests-67%20passed-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
</p>

---

## Why?

Most evaluations focus on best-case performance — clean data, standard benchmarks. But production models fail in the real world: noisy sensors, corrupted inputs, distribution shifts.

This framework answers:

- 📉 **How quickly does accuracy collapse** under increasing perturbation?
- 🎯 **Does the model stay calibrated** or become overconfident before failing?
- 💥 **Which perturbations cause the most damage?**
- 🔍 **Which samples fail with the highest confidence?** (worst-case analysis)

---

## Features

| Feature | Description |
|---------|-------------|
| **Web Dashboard** | Interactive UI with real-time progress bar, Plotly charts, and failure analysis |
| **CLI** | Scriptable stress tests with `tqdm` progress bars |
| **Custom Model Upload** | Upload your own TorchScript (`.pt`) or Python (`.py`) models via the web UI |
| **8 Perturbations** | 4 vision + 4 time-series perturbation functions with severity control |
| **5 Robustness Metrics** | Accuracy, confidence, flip rate, ECE, degradation slope |
| **Vectorized Engine** | Batch-level perturbation processing for fast sweeps |
| **67 Unit Tests** | Full test coverage across metrics, perturbations, runner, and wrappers |

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Failure-First-Model-Stress-Testing-Tool.git
cd Failure-First-Model-Stress-Testing-Tool/failure_first
pip install -r requirements.txt
```

### Web Dashboard

```bash
python app.py
# Open http://localhost:8080
```

The dashboard lets you:
- Select model type (Vision CNN, Time-Series LSTM, or Custom Upload)
- Choose perturbations and sample count
- Watch a **live progress bar** during stress tests
- Explore interactive charts and worst-case failure tables

### CLI

```bash
# Vision model on CIFAR-10
python stress_test.py --model vision --dataset cifar10 --num-samples 200

# Time-series model on synthetic data
python stress_test.py --model timeseries --dataset synthetic --num-samples 300

# Custom output directory
python stress_test.py --model vision --dataset cifar10 --output-dir my_results/
```

### Python API

```python
from core.model_wrapper import VisionModelWrapper
from core.perturbations import get_perturbations
from core.stress_runner import StressRunner
from core.reporter import Reporter

model = VisionModelWrapper(device="cpu", num_classes=10)
runner = StressRunner(model, get_perturbations("vision"), seed=42)
results = runner.run(X_test, y_test)

Reporter(output_dir="results/").generate_report(results)
```

---

## Custom Model Upload

Upload your own models through the web dashboard or API.

### TorchScript (`.pt`)

Save your model with `torch.jit.save()` and upload the `.pt` file.

### Python File (`.py`)

Create a `.py` file implementing two functions:

```python
import numpy as np

def predict(X: np.ndarray) -> np.ndarray:
    """Return class labels, shape (N,)."""
    ...

def predict_proba(X: np.ndarray) -> np.ndarray:
    """Return probabilities, shape (N, num_classes)."""
    ...
```

> ⚠️ **Security note:** Python file upload executes arbitrary code. Only use with trusted files.

### Dataset Format

Save your data as a `.npy` file:

```python
import numpy as np
np.save("data.npy", {"X": X_array, "y": y_array})
```

---

## Perturbations

### Vision

| Name | Effect |
|------|--------|
| `gaussian_noise` | Additive Gaussian noise (σ scales with severity) |
| `random_occlusion` | Random black-out rectangle |
| `blur` | Gaussian blur (kernel scales with severity) |
| `resolution_downsample` | Downsample → upsample to original size |

### Time-Series

| Name | Effect |
|------|--------|
| `ts_gaussian_noise` | Additive noise proportional to data std |
| `temporal_dropout` | Zero-out random timesteps |
| `mean_shift` | Shift signal mean |
| `variance_shift` | Scale signal variance |

All perturbations accept `severity ∈ [0, 1]` and an optional `seed` for reproducibility.

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | Top-1 accuracy at each severity level |
| **Avg Confidence** | Mean max predicted probability — detects confidence collapse |
| **Flip Rate** | Fraction of predictions changed vs. clean baseline |
| **ECE** | Expected Calibration Error — confidence vs. accuracy alignment |
| **Degradation Slope** | Linear slope of accuracy vs. severity; large negative = rapid failure |

---

## Output Files

| File | Description |
|------|-------------|
| `stress_results.json` | Full structured log of all metrics |
| `stress_results.csv` | Tidy CSV (one row per perturbation × severity) |
| `accuracy_curve.png` | Accuracy vs. perturbation severity |
| `confidence_drift.png` | Avg confidence vs. severity |
| `flip_rate.png` | Prediction instability rate vs. severity |
| `worst_failures.json` | Top 20 highest-confidence misclassifications |

---

## Project Structure

```
failure_first/
├── core/
│   ├── model_wrapper.py        # ModelWrapper base + CNN / LSTM wrappers
│   ├── custom_wrapper.py       # TorchScript & Python file wrappers
│   ├── perturbations.py        # 8 parameterized perturbation functions
│   ├── metrics.py              # 5 robustness metrics
│   ├── stress_runner.py        # Vectorized sweep engine + progress callback
│   └── reporter.py             # JSON / CSV / PNG report generation
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_metrics.py         # 16 metric tests
│   ├── test_perturbations.py   # 28 perturbation tests
│   ├── test_stress_runner.py   # 12 integration tests
│   └── test_custom_wrapper.py  # 11 wrapper tests
├── templates/
│   └── index.html              # Web dashboard (Plotly + SSE progress)
├── app.py                      # Flask server with SSE streaming
├── stress_test.py              # CLI entry point with tqdm
├── custom_model_template.py    # Template for custom Python models
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## CLI Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--model` | `vision`, `timeseries` | *required* | Model type |
| `--dataset` | `cifar10`, `synthetic` | *required* | Dataset to load |
| `--num-samples` | int | `500` | Number of test samples |
| `--seed` | int | `42` | Random seed |
| `--output-dir` | path | `results/<model>` | Output directory |

---

## License

MIT
