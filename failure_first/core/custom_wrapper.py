"""
custom_wrapper.py — Wrappers for user-uploaded custom models.

Supports two formats:
  1. TorchScript (.pt) — loaded via torch.jit.load()
  2. Python file (.py) — dynamically imported, must define predict() and predict_proba()
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from core.model_wrapper import ModelWrapper


class TorchScriptWrapper(ModelWrapper):
    """
    Wraps a TorchScript-serialized model.

    The user saves their model with:
        torch.jit.save(torch.jit.script(model), "model.pt")
    or:
        torch.jit.save(torch.jit.trace(model, example_input), "model.pt")

    The wrapper loads the .pt file and runs inference through it.
    The model must output raw logits of shape (N, num_classes).
    """

    def __init__(self, model_path: str, num_classes: int, device: str = "cpu"):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Run the model and return softmax probabilities."""
        tensor = torch.from_numpy(x).float().to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)


class PythonFileWrapper(ModelWrapper):
    """
    Wraps a user-supplied Python file as a model.

    The .py file must define two module-level functions:

        def predict(X: np.ndarray) -> np.ndarray:
            '''Return class labels, shape (N,).'''
            ...

        def predict_proba(X: np.ndarray) -> np.ndarray:
            '''Return probabilities, shape (N, num_classes).'''
            ...

    WARNING: This executes arbitrary user code. Only use in trusted
    environments (local machine, internal network).
    """

    def __init__(self, py_path: str):
        path = Path(py_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Python file not found: {path}")

        # Dynamically import the .py file as a module
        spec = importlib.util.spec_from_file_location("custom_model", str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_model"] = module
        spec.loader.exec_module(module)

        # Validate that required functions exist
        if not hasattr(module, "predict"):
            raise AttributeError(
                f"The file {path.name} must define a 'predict(X)' function."
            )
        if not hasattr(module, "predict_proba"):
            raise AttributeError(
                f"The file {path.name} must define a 'predict_proba(X)' function."
            )

        self._predict_fn = module.predict
        self._predict_proba_fn = module.predict_proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = self._predict_fn(x)
        return np.asarray(result)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result = self._predict_proba_fn(x)
        return np.asarray(result)
