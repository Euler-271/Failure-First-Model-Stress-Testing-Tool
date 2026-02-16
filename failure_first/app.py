"""
app.py — Flask web server for the Failure-First Stress Testing Framework.

Serves a single-page dashboard and exposes API endpoints to run
stress tests with built-in or user-uploaded custom models.

Usage:
    python app.py
    # Then open http://localhost:8080
"""

import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, Response

# Ensure core modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_wrapper import VisionModelWrapper, TimeSeriesModelWrapper
from core.custom_wrapper import TorchScriptWrapper, PythonFileWrapper
from core.perturbations import get_perturbations
from core.stress_runner import StressRunner
from core.reporter import Reporter

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# Directory for uploaded files
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_MODEL_EXT = {".pt", ".pth", ".py"}
ALLOWED_DATASET_EXT = {".npy"}


def _safe_save(file_storage, allowed_exts: set) -> Path:
    """Save an uploaded file with a UUID name, validating extension."""
    original = file_storage.filename or ""
    ext = Path(original).suffix.lower()
    if ext not in allowed_exts:
        raise ValueError(f"Invalid file extension '{ext}'. Allowed: {allowed_exts}")
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name
    file_storage.save(str(dest))
    return dest


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
    """Generate synthetic sine-wave binary classification dataset."""
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


def load_npy_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a user-uploaded .npy dataset file.

    Expected format: np.save("data.npy", {"X": X_array, "y": y_array})

    WARNING: np.load(allow_pickle=True) can execute arbitrary code embedded
    in crafted .npy files. Only use in trusted environments.
    """
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        raise ValueError(f"Failed to load dataset: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Dataset .npy file must contain a dict with 'X' and 'y' keys.")
    if "X" not in data or "y" not in data:
        raise ValueError("Dataset dict must have 'X' and 'y' keys.")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"])

    # Validate shapes
    if X.ndim < 1 or y.ndim != 1:
        raise ValueError(f"Invalid shapes: X={X.shape}, y={y.shape}. y must be 1-D.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have same number of samples. Got {X.shape[0]} vs {y.shape[0]}.")
    if X.shape[0] > 10_000:
        raise ValueError(f"Too many samples ({X.shape[0]}). Maximum is 10,000.")

    return X, y


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard."""
    return render_template("index.html")


@app.route("/api/status")
def status():
    """Health check."""
    return jsonify({"status": "ok"})


@app.route("/api/template")
def download_template():
    """Download the custom model Python template."""
    template_path = Path(__file__).parent / "custom_model_template.py"
    return send_file(template_path, as_attachment=True)


@app.route("/api/run", methods=["POST"])
def run_stress_test():
    """
    Run a stress test with a built-in model.
    Returns an SSE stream with progress events and a final result event.
    """
    try:
        config = request.get_json()
        model_type = config.get("model", "vision")
        dataset = config.get("dataset", "cifar10")
        num_samples = max(1, min(int(config.get("num_samples", 200)), 1000))
        seed = int(config.get("seed", 42))
        selected_perts = config.get("perturbations", None)

        # Load data
        if model_type == "vision":
            X, y = load_cifar10(num_samples)
        else:
            X, y = load_synthetic_timeseries(num_samples, seed)

        # Build model
        if model_type == "vision":
            model = VisionModelWrapper(device="cpu", num_classes=10)
        else:
            model = TimeSeriesModelWrapper(device="cpu", input_size=1, num_classes=2)

        # Perturbations
        all_perts = get_perturbations(model_type)
        if selected_perts:
            all_perts = [(n, fn) for n, fn in all_perts if n in selected_perts]
        if not all_perts:
            return jsonify({"error": "No valid perturbations selected."}), 400

        def generate():
            q = queue.Queue()

            def _run_in_thread():
                try:
                    def progress_cb(step, total, msg):
                        q.put({"type": "progress", "step": step, "total": total, "message": msg})

                    runner = StressRunner(model, all_perts, seed=seed)
                    results = runner.run(X, y, progress_callback=progress_cb)

                    output_dir = f"results/{model_type}_web"
                    reporter = Reporter(output_dir=output_dir)
                    reporter.generate_report(results)

                    q.put({"type": "result", "payload": {
                        "success": True,
                        "config": {
                            "model": model_type,
                            "dataset": dataset,
                            "num_samples": num_samples,
                            "seed": seed,
                            "perturbations": [n for n, _ in all_perts],
                        },
                        "results": results,
                    }})
                except Exception as exc:
                    q.put({"type": "error", "message": str(exc)})
                finally:
                    q.put(None)  # sentinel

            threading.Thread(target=_run_in_thread, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                if item["type"] == "progress":
                    evt = json.dumps({"step": item["step"], "total": item["total"], "message": item["message"]})
                    yield f"event: progress\ndata: {evt}\n\n"
                elif item["type"] == "result":
                    yield f"event: result\ndata: {json.dumps(item['payload'])}\n\n"
                elif item["type"] == "error":
                    yield f"event: error\ndata: {json.dumps({'error': item['message']})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def run_custom_stress_test():
    """
    Run a stress test with a user-uploaded model and dataset.

    Expects multipart form data:
        model_file:   .pt (TorchScript) or .py (Python file)
        dataset_file: .npy file with {"X": array, "y": array}
        format:       "torchscript" | "python"
        domain:       "vision" | "timeseries"
        num_classes:  int
        seed:         int
        perturbations: comma-separated perturbation names (optional)
    """
    try:
        # --- Validate required fields ----------------------------------
        if "model_file" not in request.files:
            return jsonify({"error": "No model file uploaded."}), 400
        if "dataset_file" not in request.files:
            return jsonify({"error": "No dataset file uploaded."}), 400

        model_file = request.files["model_file"]
        dataset_file = request.files["dataset_file"]
        fmt = request.form.get("format", "torchscript")
        domain = request.form.get("domain", "vision")
        num_classes = int(request.form.get("num_classes", 10))
        seed = int(request.form.get("seed", 42))
        selected_perts = request.form.get("perturbations", "")

        # --- Save uploaded files (sanitised names) ---------------------
        model_path = _safe_save(model_file, ALLOWED_MODEL_EXT)
        dataset_path = _safe_save(dataset_file, ALLOWED_DATASET_EXT)

        # --- Load dataset ----------------------------------------------
        X, y = load_npy_dataset(str(dataset_path))
        num_samples = len(X)

        # --- Load model ------------------------------------------------
        if fmt == "torchscript":
            model = TorchScriptWrapper(
                model_path=str(model_path),
                num_classes=num_classes,
                device="cpu",
            )
        elif fmt == "python":
            model = PythonFileWrapper(py_path=str(model_path))
        else:
            return jsonify({"error": f"Unknown format: {fmt}"}), 400

        # --- Perturbations ---------------------------------------------
        all_perts = get_perturbations(domain)
        if selected_perts:
            names = [s.strip() for s in selected_perts.split(",") if s.strip()]
            if names:
                all_perts = [(n, fn) for n, fn in all_perts if n in names]
        if not all_perts:
            return jsonify({"error": "No valid perturbations selected."}), 400

        def generate():
            q = queue.Queue()

            def _run_in_thread():
                try:
                    def progress_cb(step, total, msg):
                        q.put({"type": "progress", "step": step, "total": total, "message": msg})

                    runner = StressRunner(model, all_perts, seed=seed)
                    results = runner.run(X, y, progress_callback=progress_cb)

                    output_dir = "results/custom_web"
                    reporter = Reporter(output_dir=output_dir)
                    reporter.generate_report(results)

                    q.put({"type": "result", "payload": {
                        "success": True,
                        "config": {
                            "model": f"custom ({fmt})",
                            "dataset": "uploaded",
                            "num_samples": num_samples,
                            "seed": seed,
                            "perturbations": [n for n, _ in all_perts],
                        },
                        "results": results,
                    }})
                except Exception as exc:
                    q.put({"type": "error", "message": str(exc)})
                finally:
                    q.put(None)
                    # Clean up uploaded files
                    for p in (model_path, dataset_path):
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass

            threading.Thread(target=_run_in_thread, daemon=True).start()

            while True:
                item = q.get()
                if item is None:
                    break
                if item["type"] == "progress":
                    evt = json.dumps({"step": item["step"], "total": item["total"], "message": item["message"]})
                    yield f"event: progress\ndata: {evt}\n\n"
                elif item["type"] == "result":
                    yield f"event: result\ndata: {json.dumps(item['payload'])}\n\n"
                elif item["type"] == "error":
                    yield f"event: error\ndata: {json.dumps({'error': item['message']})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Failure-First Stress Testing — Web Dashboard")
    print("  Open http://localhost:8080 in your browser")
    print("=" * 60)
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        host="127.0.0.1",
        port=8080,
        exclude_patterns=["*/uploads/*"],
    )
