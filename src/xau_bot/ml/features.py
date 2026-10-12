"""Feature extraction pipeline for Meta-Labeling."""

from __future__ import annotations

from xau_bot.domain.models import Direction
from xau_bot.journal.models import JournalEntry, SnapshotData, TradeOutcome
from xau_bot.ml.models import TradeFeatures


class FeatureExtractor:
    """Extracts normalized numerical features from trading snapshots and journal entries."""

    @staticmethod
    def _map_session_code(session_name: str) -> float:
        norm = session_name.upper().replace("-", "_").strip()
        if "ASIA" in norm:
            return 0.0
        if "LONDON" in norm and "CLOSE" not in norm:
            return 1.0
        if "NY" in norm or "NEW_YORK" in norm:
            return 2.0
        if "CLOSE" in norm:
            return 3.0
        return 4.0

    @classmethod
    def extract_from_snapshot(
        cls,
        snapshot: SnapshotData,
        direction: Direction,
    ) -> TradeFeatures:
        """Extracts TradeFeatures from a single setup snapshot."""
        spread_ratio = snapshot.spread_price / snapshot.atr_m5 if snapshot.atr_m5 > 1e-9 else 0.0
        session_code = cls._map_session_code(snapshot.session)
        news_dist = snapshot.news_distance_min if snapshot.news_distance_min is not None else 240.0
        confluences_cnt = float(len(snapshot.confluences)) if snapshot.confluences else 0.0
        dir_code = 1.0 if direction == Direction.BULLISH else -1.0

        return TradeFeatures(
            displacement_atr=snapshot.displacement_atr,
            fvg_size_atr=snapshot.fvg_size_atr,
            spread_atr_ratio=spread_ratio,
            session_code=session_code,
            news_distance_min=news_dist,
            confluences_count=confluences_cnt,
            direction_code=dir_code,
        )

    @classmethod
    def extract_from_entry(cls, entry: JournalEntry) -> TradeFeatures:
        """Extracts TradeFeatures from a JournalEntry."""
        return cls.extract_from_snapshot(entry.snapshot, entry.direction)

    @classmethod
    def extract_dataset(
        cls,
        entries: list[JournalEntry],
    ) -> tuple[list[list[float]], list[int], list[float]]:
        """Extracts matrix X, labels y (1=Win, 0=Loss), and realized R multiples."""
        X: list[list[float]] = []
        y: list[int] = []
        r_values: list[float] = []

        valid_outcomes = (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN)
        for entry in entries:
            if entry.outcome not in valid_outcomes or entry.outcome_data is None:
                continue

            features = cls.extract_from_entry(entry)
            r = entry.outcome_data.realized_r
            label = 1 if (entry.outcome == TradeOutcome.WIN or r > 0.0) else 0

            X.append(features.to_vector())
            y.append(label)
            r_values.append(r)

        return X, y, r_values
