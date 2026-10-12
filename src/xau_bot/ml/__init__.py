"""Machine Learning Meta-Labeling subpackage for XAU/USD trading bot."""

from __future__ import annotations

from xau_bot.ml.classifier import LogisticMetaClassifier
from xau_bot.ml.evaluator import calculate_metrics, calculate_roc_auc
from xau_bot.ml.features import FeatureExtractor
from xau_bot.ml.filter import MetaLabelFilter
from xau_bot.ml.models import (
    MetaLabelDecision,
    MetaPrediction,
    ModelMetrics,
    StandardScalerParams,
    TradeFeatures,
)

__all__ = [
    "FeatureExtractor",
    "LogisticMetaClassifier",
    "MetaLabelDecision",
    "MetaLabelFilter",
    "MetaPrediction",
    "ModelMetrics",
    "StandardScalerParams",
    "TradeFeatures",
    "calculate_metrics",
    "calculate_roc_auc",
]
