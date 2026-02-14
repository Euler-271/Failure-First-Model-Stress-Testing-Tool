"""
test_custom_wrapper.py — Tests for TorchScript and Python file model wrappers.
"""

import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.custom_wrapper import PythonFileWrapper, TorchScriptWrapper


class TestPythonFileWrapper:
    """Tests for dynamically loaded Python file models."""

    @pytest.fixture
    def model_file(self, tmp_path):
        """Create a valid Python model file."""
        code = textwrap.dedent("""\
            import numpy as np

            def predict(X):
                means = X.mean(axis=tuple(range(1, X.ndim)))
                return (means > 0.5).astype(int)

            def predict_proba(X):
                means = X.mean(axis=tuple(range(1, X.ndim)))
                p1 = 1.0 / (1.0 + np.exp(-10 * (means - 0.5)))
                p0 = 1.0 - p1
                return np.column_stack([p0, p1])
        """)
        path = tmp_path / "test_model.py"
        path.write_text(code)
        return str(path)

    def test_load_valid_file(self, model_file):
        wrapper = PythonFileWrapper(model_file)
        assert wrapper is not None

    def test_predict_shape(self, model_file):
        wrapper = PythonFileWrapper(model_file)
        X = np.random.random((10, 32, 32, 3)).astype(np.float32)
        preds = wrapper.predict(X)
        assert preds.shape == (10,)

    def test_predict_proba_shape(self, model_file):
        wrapper = PythonFileWrapper(model_file)
        X = np.random.random((10, 32, 32, 3)).astype(np.float32)
        probs = wrapper.predict_proba(X)
        assert probs.shape == (10, 2)

    def test_predict_proba_sums_to_one(self, model_file):
        wrapper = PythonFileWrapper(model_file)
        X = np.random.random((10, 32, 32, 3)).astype(np.float32)
        probs = wrapper.predict_proba(X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_missing_predict_raises(self, tmp_path):
        path = tmp_path / "bad_model.py"
        path.write_text("import numpy as np\ndef predict_proba(X): return X\n")
        with pytest.raises(AttributeError, match="predict"):
            PythonFileWrapper(str(path))

    def test_missing_predict_proba_raises(self, tmp_path):
        path = tmp_path / "bad_model2.py"
        path.write_text("import numpy as np\ndef predict(X): return X\n")
        with pytest.raises(AttributeError, match="predict_proba"):
            PythonFileWrapper(str(path))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PythonFileWrapper("/nonexistent/model.py")


class TestTorchScriptWrapper:
    """Tests for TorchScript model loading."""

    @pytest.fixture
    def scripted_model_path(self, tmp_path):
        """Create a simple scripted model and save it."""
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 3)
            def forward(self, x):
                return self.fc(x)

        model = TinyModel()
        scripted = torch.jit.script(model)
        path = tmp_path / "tiny.pt"
        torch.jit.save(scripted, str(path))
        return str(path)

    def test_load(self, scripted_model_path):
        wrapper = TorchScriptWrapper(scripted_model_path, num_classes=3)
        assert wrapper is not None

    def test_predict_shape(self, scripted_model_path):
        wrapper = TorchScriptWrapper(scripted_model_path, num_classes=3)
        X = np.random.randn(5, 8).astype(np.float32)
        preds = wrapper.predict(X)
        assert preds.shape == (5,)

    def test_predict_proba_shape(self, scripted_model_path):
        wrapper = TorchScriptWrapper(scripted_model_path, num_classes=3)
        X = np.random.randn(5, 8).astype(np.float32)
        probs = wrapper.predict_proba(X)
        assert probs.shape == (5, 3)

    def test_predict_proba_sums_to_one(self, scripted_model_path):
        wrapper = TorchScriptWrapper(scripted_model_path, num_classes=3)
        X = np.random.randn(5, 8).astype(np.float32)
        probs = wrapper.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
