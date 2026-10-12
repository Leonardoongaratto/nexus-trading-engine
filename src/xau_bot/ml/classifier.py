"""Lightweight pure-Python Logistic Classifier with L2 Regularization for Meta-Labeling."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xau_bot.ml.evaluator import calculate_metrics, compute_log_loss
from xau_bot.ml.models import (
    MetaLabelDecision,
    MetaPrediction,
    ModelMetrics,
    StandardScalerParams,
    TradeFeatures,
)


class LogisticMetaClassifier:
    """Logistic regression classifier for setup meta-labeling with L2 weight regularization."""

    def __init__(
        self,
        weights: list[float] | None = None,
        bias: float = 0.0,
        scaler: StandardScalerParams | None = None,
        threshold: float = 0.50,
        version: str = "1.0.0",
        feature_names: tuple[str, ...] | None = None,
        trained_at: datetime | None = None,
        train_metrics: ModelMetrics | None = None,
        loss_history: list[float] | None = None,
    ) -> None:
        self.feature_names = feature_names or TradeFeatures.feature_names()
        self.weights = weights or [0.0] * len(self.feature_names)
        self.bias = bias
        self.scaler = scaler or StandardScalerParams(
            means=[0.0] * len(self.feature_names),
            stds=[1.0] * len(self.feature_names),
        )
        self.threshold = threshold
        self.version = version
        self.trained_at = trained_at
        self.train_metrics = train_metrics
        self.loss_history = loss_history or []

    @staticmethod
    def _sigmoid(z: float) -> float:
        clipped_z = max(-15.0, min(15.0, z))
        return 1.0 / (1.0 + math.exp(-clipped_z))

    def _fit_scaler(self, X: list[list[float]]) -> StandardScalerParams:
        n = len(X)
        if n == 0:
            return self.scaler
        dim = len(X[0])
        means: list[float] = [0.0] * dim
        stds: list[float] = [1.0] * dim

        for j in range(dim):
            col_vals = [X[i][j] for i in range(n)]
            mean = sum(col_vals) / n
            variance = sum((v - mean) ** 2 for v in col_vals) / n
            std = math.sqrt(variance)
            means[j] = mean
            stds[j] = std if std > 1e-9 else 1.0

        return StandardScalerParams(means=means, stds=stds)

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        r_values: list[float] | None = None,
        epochs: int = 250,
        lr: float = 0.1,
        l2: float = 0.01,
        threshold: float | None = None,
    ) -> ModelMetrics:
        """Trains the classifier using SGD with L2 penalty and adaptive learning rate decay."""
        n = len(X)
        if n == 0 or len(y) != n:
            raise ValueError("Training dataset must not be empty and X, y lengths must match.")

        dim = len(X[0])
        self.scaler = self._fit_scaler(X)
        X_norm = [self.scaler.transform(row) for row in X]

        self.weights = [0.0] * dim
        self.bias = 0.0
        if threshold is not None:
            self.threshold = threshold

        self.loss_history = []
        initial_probs = [self._sigmoid(self.bias) for _ in range(n)]
        self.loss_history.append(compute_log_loss(y, initial_probs))

        for epoch in range(epochs):
            current_lr = lr / (1.0 + 0.005 * epoch)
            for i in range(n):
                xi = X_norm[i]
                yi = float(y[i])

                z = self.bias
                for j in range(dim):
                    z += self.weights[j] * xi[j]
                p = self._sigmoid(z)
                error = p - yi

                for j in range(dim):
                    grad_w = error * xi[j] + l2 * self.weights[j]
                    self.weights[j] -= current_lr * grad_w
                self.bias -= current_lr * error

            epoch_probs = [self.predict_proba(row) for row in X]
            self.loss_history.append(compute_log_loss(y, epoch_probs))

        y_probs = [self.predict_proba(row) for row in X]
        self.trained_at = datetime.now(UTC)
        self.train_metrics = calculate_metrics(
            y_true=y,
            y_probs=y_probs,
            r_values=r_values,
            threshold=self.threshold,
        )
        return self.train_metrics

    def predict_proba(self, vector: list[float]) -> float:
        """Estimates positive class probability P(y=1|x)."""
        x_norm = self.scaler.transform(vector)
        z = self.bias
        for w, x in zip(self.weights, x_norm, strict=True):
            z += w * x
        return self._sigmoid(z)

    def decide(
        self,
        vector: list[float],
        threshold: float | None = None,
    ) -> MetaPrediction:
        """Evaluates trade features and returns binary ALLOW or VETO decision."""
        tau = threshold if threshold is not None else self.threshold
        p = self.predict_proba(vector)
        decision = MetaLabelDecision.ALLOW if p >= tau else MetaLabelDecision.VETO
        return MetaPrediction(
            probability=p,
            decision=decision,
            threshold=tau,
            is_fallback=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the classifier state to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "threshold": self.threshold,
            "weights": [float(w) for w in self.weights],
            "bias": float(self.bias),
            "scaler": self.scaler.to_dict(),
            "feature_names": list(self.feature_names),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "train_metrics": self.train_metrics.to_dict() if self.train_metrics else None,
            "loss_history": [round(float(loss), 6) for loss in self.loss_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticMetaClassifier:
        """Deserializes a classifier from dictionary payload."""
        trained_at = None
        if data.get("trained_at"):
            trained_at = datetime.fromisoformat(data["trained_at"])
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=UTC)

        train_metrics = None
        if data.get("train_metrics"):
            tm = data["train_metrics"]
            train_metrics = ModelMetrics(
                accuracy=float(tm["accuracy"]),
                precision=float(tm["precision"]),
                recall=float(tm["recall"]),
                roc_auc=float(tm["roc_auc"]),
                brier_score=float(tm["brier_score"]),
                delta_r=float(tm["delta_r"]),
                sample_size=int(tm["sample_size"]),
                positive_rate=float(tm["positive_rate"]),
            )

        weights = [float(w) for w in data["weights"]]
        scaler = StandardScalerParams.from_dict(data["scaler"])
        raw_features = data.get("feature_names")
        feature_names = (
            tuple(str(f) for f in raw_features) if raw_features else TradeFeatures.feature_names()
        )

        if len(weights) != len(scaler.means) or len(weights) != len(feature_names):
            raise ValueError(
                f"Model dimension mismatch: weights ({len(weights)}), "
                f"scaler ({len(scaler.means)}), features ({len(feature_names)})."
            )

        loss_hist = [float(x) for x in data.get("loss_history", [])]

        return cls(
            weights=weights,
            bias=float(data["bias"]),
            scaler=scaler,
            threshold=float(data.get("threshold", 0.50)),
            version=str(data.get("version", "1.0.0")),
            feature_names=feature_names,
            trained_at=trained_at,
            train_metrics=train_metrics,
            loss_history=loss_hist,
        )

    def save_to_file(self, path: Path) -> None:
        """Saves the serialized model payload to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> LogisticMetaClassifier:
        """Loads and instantiates a classifier from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
