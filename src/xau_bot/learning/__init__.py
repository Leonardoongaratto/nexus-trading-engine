"""Learning and periodic review module (T-A6)."""

from __future__ import annotations

from xau_bot.learning.analyzer import LearningAnalyzer
from xau_bot.learning.human import HumanDiscretionAnalyzer
from xau_bot.learning.models import (
    CohortMetrics,
    FeatureStats,
    HumanDiscretionReport,
    HumanDiscretionVerdict,
    HumanSegmentStats,
    Hypothesis,
    HypothesisStatus,
    LearningReviewReport,
    ParameterVersion,
    ValidationMetrics,
    ValidationResult,
)
from xau_bot.learning.registry import ParameterRegistry
from xau_bot.learning.reporter import LearningReporter
from xau_bot.learning.shadow import ShadowDivergence, ShadowEngine, ShadowEvaluation
from xau_bot.learning.stats import (
    calculate_expectancy,
    calculate_win_loss_rates,
    fdr_benjamini_hochberg,
    two_proportion_z_test,
)
from xau_bot.learning.validator import HypothesisValidator

__all__ = [
    "CohortMetrics",
    "FeatureStats",
    "HumanDiscretionAnalyzer",
    "HumanDiscretionReport",
    "HumanDiscretionVerdict",
    "HumanSegmentStats",
    "Hypothesis",
    "HypothesisStatus",
    "HypothesisValidator",
    "LearningAnalyzer",
    "LearningReporter",
    "LearningReviewReport",
    "ParameterRegistry",
    "ParameterVersion",
    "ShadowDivergence",
    "ShadowEngine",
    "ShadowEvaluation",
    "ValidationMetrics",
    "ValidationResult",
    "calculate_expectancy",
    "calculate_win_loss_rates",
    "fdr_benjamini_hochberg",
    "two_proportion_z_test",
]
