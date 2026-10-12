"""Runtime filtering layer integrating Meta-Labeling decisions into setup evaluations."""

from __future__ import annotations

import logging
from pathlib import Path

from xau_bot.config import Settings
from xau_bot.domain.models import Direction
from xau_bot.journal.models import JournalEntry, SnapshotData
from xau_bot.ml.classifier import LogisticMetaClassifier
from xau_bot.ml.features import FeatureExtractor
from xau_bot.ml.models import MetaLabelDecision, MetaPrediction, TradeFeatures

logger = logging.getLogger(__name__)


class MetaLabelFilter:
    """Runtime filter evaluating trade opportunities using trained meta-labeling models."""

    def __init__(
        self,
        classifier: LogisticMetaClassifier | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.classifier = classifier
        if self.classifier is None and self.settings.ml_model_path:
            model_path = Path(self.settings.ml_model_path)
            if model_path.exists():
                try:
                    self.classifier = LogisticMetaClassifier.load_from_file(model_path)
                except Exception as e:
                    logger.warning(
                        "Falha ao carregar modelo de meta-labeling de %s: %s", model_path, e
                    )
        if (
            self.classifier is not None
            and self.classifier.feature_names != TradeFeatures.feature_names()
        ):
            logger.warning(
                "Modelo possui feature_names incompatíveis com TradeFeatures. Desativando filtro."
            )
            self.classifier = None

    @property
    def is_active(self) -> bool:
        """Indicates whether meta-labeling filtering is enabled and operational."""
        return self.settings.ml_meta_labeling_enabled and self.classifier is not None

    def evaluate_snapshot(
        self,
        snapshot: SnapshotData,
        direction: Direction,
    ) -> MetaPrediction:
        """Evaluates setup snapshot features and returns ALLOW or VETO decision."""
        if not self.is_active or self.classifier is None:
            return MetaPrediction(
                probability=1.0,
                decision=MetaLabelDecision.ALLOW,
                threshold=self.settings.ml_min_probability,
                is_fallback=True,
            )

        try:
            features = FeatureExtractor.extract_from_snapshot(snapshot, direction)
            return self.classifier.decide(
                features.to_vector(),
                threshold=self.settings.ml_min_probability,
            )
        except Exception as e:
            logger.warning("Falha na inferência de meta-labeling, aplicando fail-open: %s", e)
            return MetaPrediction(
                probability=1.0,
                decision=MetaLabelDecision.ALLOW,
                threshold=self.settings.ml_min_probability,
                is_fallback=True,
            )

    def evaluate_entry(self, entry: JournalEntry) -> MetaPrediction:
        """Convenience method to evaluate a full JournalEntry."""
        return self.evaluate_snapshot(entry.snapshot, entry.direction)

    @classmethod
    def from_settings(cls, settings: Settings) -> MetaLabelFilter:
        """Instantiates MetaLabelFilter configured from application Settings."""
        return cls(settings=settings)
