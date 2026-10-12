"""Data models and value objects for the Machine Learning Meta-Labeling subpackage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MetaLabelDecision(StrEnum):
    """Binary decision emitted by the meta-labeling filter."""

    ALLOW = "ALLOW"
    VETO = "VETO"


@dataclass(frozen=True)
class TradeFeatures:
    """Normalized feature vector extracted from setup snapshots for meta-labeling."""

    displacement_atr: float
    fvg_size_atr: float
    spread_atr_ratio: float
    session_code: float
    news_distance_min: float
    confluences_count: float
    direction_code: float

    def to_vector(self) -> list[float]:
        """Converts feature attributes to an ordered numeric list."""
        return [
            self.displacement_atr,
            self.fvg_size_atr,
            self.spread_atr_ratio,
            self.session_code,
            self.news_distance_min,
            self.confluences_count,
            self.direction_code,
        ]

    @classmethod
    def feature_names(cls) -> tuple[str, ...]:
        """Returns canonical feature names matching vector order."""
        return (
            "displacement_atr",
            "fvg_size_atr",
            "spread_atr_ratio",
            "session_code",
            "news_distance_min",
            "confluences_count",
            "direction_code",
        )


@dataclass(frozen=True)
class StandardScalerParams:
    """Standardization parameters (mean and std) for feature vectors."""

    means: list[float]
    stds: list[float]

    def transform(self, vector: list[float]) -> list[float]:
        """Applies z-score scaling: (x - mean) / std."""
        scaled: list[float] = []
        for i, val in enumerate(vector):
            mean = self.means[i] if i < len(self.means) else 0.0
            std = self.stds[i] if i < len(self.stds) else 1.0
            divisor = std if abs(std) > 1e-9 else 1.0
            scaled.append((val - mean) / divisor)
        return scaled

    def to_dict(self) -> dict[str, Any]:
        """Serializes scaler parameters to a dictionary."""
        return {"means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardScalerParams:
        """Constructs scaler parameters from dictionary."""
        return cls(
            means=[float(x) for x in data["means"]],
            stds=[float(x) for x in data["stds"]],
        )


@dataclass(frozen=True)
class ModelMetrics:
    """Performance evaluation metrics for the meta-labeling model."""

    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    brier_score: float
    delta_r: float
    sample_size: int
    positive_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes metrics to a dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "roc_auc": round(self.roc_auc, 4),
            "brier_score": round(self.brier_score, 4),
            "delta_r": round(self.delta_r, 4),
            "sample_size": self.sample_size,
            "positive_rate": round(self.positive_rate, 4),
        }


@dataclass(frozen=True)
class MetaPrediction:
    """Inference output produced by the meta-labeling classifier."""

    probability: float
    decision: MetaLabelDecision
    threshold: float
    is_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializes prediction to a dictionary."""
        return {
            "probability": round(self.probability, 4),
            "decision": self.decision.value,
            "threshold": round(self.threshold, 4),
            "is_fallback": self.is_fallback,
        }
