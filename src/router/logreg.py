from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticRegressionModel:
    weights: np.ndarray  # (D,)
    bias: float
    mean: np.ndarray     # (D,)
    std: np.ndarray      # (D,)
    feature_names: List[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / self.std
        return _sigmoid(Xn @ self.weights + self.bias)

    def to_json(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.astype(float).tolist(),
            "bias": float(self.bias),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
            "feature_names": list(self.feature_names),
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "LogisticRegressionModel":
        return LogisticRegressionModel(
            weights=np.array(obj["weights"], dtype=np.float32),
            bias=float(obj["bias"]),
            mean=np.array(obj["mean"], dtype=np.float32),
            std=np.array(obj["std"], dtype=np.float32),
            feature_names=list(obj["feature_names"]),
        )


class LogisticRegression:
    def __init__(self, l2: float = 1e-3, lr: float = 0.1, epochs: int = 200, seed: int = 0):
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> LogisticRegressionModel:
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")

        rng = np.random.default_rng(self.seed)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        Xn = (X - mean) / std

        n, d = Xn.shape
        w = rng.normal(0.0, 0.01, size=(d,)).astype(np.float32)
        b = 0.0

        for _ in range(self.epochs):
            p = _sigmoid(Xn @ w + b)  # (n,)
            # gradients
            err = (p - y)  # (n,)
            gw = (Xn.T @ err) / n + self.l2 * w
            gb = float(err.mean())
            # update
            w = w - self.lr * gw
            b = b - self.lr * gb

        return LogisticRegressionModel(weights=w, bias=float(b), mean=mean, std=std, feature_names=feature_names)
