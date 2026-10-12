"""Historical Statistics Aggregator and Gate 1 Validation for Trade Setups."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xau_bot.backtest.models import BacktestTrade
from xau_bot.domain.models import TradeSetup

logger = logging.getLogger("xau_bot.advisor.stats")

GATE_1_MIN_SAMPLE_SIZE: int = 200
GATE_1_UNVALIDATED_WARNING: str = (
    "Estratégia não validada estatisticamente no Portão 1 (N < 200 ou sem vantagem estatística)"
)


@dataclass(frozen=True, slots=True)
class SetupHistoricalStats:
    """Historical trading statistics for a specific setup trigger or profile."""

    setup_type: str
    sample_size: int
    win_count: int
    loss_count: int
    win_rate: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float
    gate_1_validated: bool
    validation_warning: str | None = None

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be non-negative, got {self.sample_size}")
        if self.win_count < 0 or self.loss_count < 0:
            raise ValueError("win_count and loss_count must be non-negative")
        if self.win_count + self.loss_count != self.sample_size:
            raise ValueError(
                f"win_count ({self.win_count}) + loss_count ({self.loss_count}) "
                f"must equal sample_size ({self.sample_size})"
            )
        if not (0.0 <= self.win_rate <= 1.0):
            raise ValueError(f"win_rate must be between 0.0 and 1.0, got {self.win_rate}")

        for name, val in (
            ("expectancy_r", self.expectancy_r),
            ("avg_win_r", self.avg_win_r),
            ("avg_loss_r", self.avg_loss_r),
            ("profit_factor", self.profit_factor),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        expected_validated = (self.sample_size >= GATE_1_MIN_SAMPLE_SIZE) and (
            self.expectancy_r > 0.0
        )
        if self.gate_1_validated != expected_validated:
            raise ValueError(
                f"gate_1_validated invariant violation: expected {expected_validated}, "
                f"got {self.gate_1_validated}"
            )

        expected_warning = None if expected_validated else GATE_1_UNVALIDATED_WARNING
        if self.validation_warning != expected_warning:
            raise ValueError("validation_warning invariant violation")

    @classmethod
    def evaluate_gate_1(
        cls,
        setup_type: str,
        sample_size: int,
        win_count: int,
        loss_count: int,
        win_rate: float,
        expectancy_r: float,
        avg_win_r: float,
        avg_loss_r: float,
        profit_factor: float,
    ) -> SetupHistoricalStats:
        """Constructs stats evaluating Gate 1 statistical criteria (N >= 200 and expectancy > 0)."""
        r_exp = round(expectancy_r, 4)
        is_validated = (sample_size >= GATE_1_MIN_SAMPLE_SIZE) and (r_exp > 0.0)
        warning = None if is_validated else GATE_1_UNVALIDATED_WARNING
        return cls(
            setup_type=setup_type,
            sample_size=sample_size,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=round(win_rate, 4),
            expectancy_r=r_exp,
            avg_win_r=round(avg_win_r, 4),
            avg_loss_r=round(avg_loss_r, 4),
            profit_factor=round(profit_factor, 4),
            gate_1_validated=is_validated,
            validation_warning=warning,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes historical stats into a dictionary."""
        return {
            "setup_type": self.setup_type,
            "sample_size": self.sample_size,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "expectancy_r": self.expectancy_r,
            "avg_win_r": self.avg_win_r,
            "avg_loss_r": self.avg_loss_r,
            "profit_factor": self.profit_factor,
            "gate_1_validated": self.gate_1_validated,
            "validation_warning": self.validation_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SetupHistoricalStats:
        """Deserializes historical stats from a dictionary, strictly verifying Gate 1 invariants."""
        sample_size = int(data["sample_size"])
        win_count = int(data["win_count"])
        win_rate = (win_count / sample_size) if sample_size > 0 else 0.0
        return cls.evaluate_gate_1(
            setup_type=str(data["setup_type"]),
            sample_size=sample_size,
            win_count=win_count,
            loss_count=int(data["loss_count"]),
            win_rate=win_rate,
            expectancy_r=float(data["expectancy_r"]),
            avg_win_r=float(data["avg_win_r"]),
            avg_loss_r=float(data["avg_loss_r"]),
            profit_factor=float(data["profit_factor"]),
        )


class SetupStatsAggregator:
    """Aggregates trades into deterministic historical statistics per setup trigger."""

    def aggregate_trades(
        self,
        trades: Sequence[BacktestTrade],
        setup_type: str = "GLOBAL",
    ) -> SetupHistoricalStats:
        """Aggregates a sequence of backtest trades into a SetupHistoricalStats summary."""
        n = len(trades)
        if n == 0:
            return SetupHistoricalStats.evaluate_gate_1(
                setup_type=setup_type,
                sample_size=0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                expectancy_r=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
                profit_factor=0.0,
            )

        wins = [t.return_r for t in trades if t.return_r > 0]
        losses = [t.return_r for t in trades if t.return_r <= 0]
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / n
        expectancy_r = sum(t.return_r for t in trades) / n
        avg_win_r = sum(wins) / win_count if win_count > 0 else 0.0
        avg_loss_r = sum(losses) / loss_count if loss_count > 0 else 0.0

        total_loss_r = abs(sum(losses))
        if total_loss_r > 0:
            profit_factor = sum(wins) / total_loss_r
        else:
            profit_factor = 100.0 if win_count > 0 else 0.0

        return SetupHistoricalStats.evaluate_gate_1(
            setup_type=setup_type,
            sample_size=n,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            expectancy_r=expectancy_r,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            profit_factor=profit_factor,
        )

    def aggregate_by_trigger_type(
        self,
        trades: Sequence[BacktestTrade],
        setups_map: Mapping[str, TradeSetup],
    ) -> dict[str, SetupHistoricalStats]:
        """Groups trades by setup trigger type and computes stats per group."""
        groups: dict[str, list[BacktestTrade]] = defaultdict(list)
        for t in trades:
            trigger_key = "UNKNOWN"
            if t.setup_id and t.setup_id in setups_map:
                trigger_key = str(setups_map[t.setup_id].trigger_type.value)
            groups[trigger_key].append(t)

        result: dict[str, SetupHistoricalStats] = {}
        for trigger_key, group_trades in groups.items():
            result[trigger_key] = self.aggregate_trades(group_trades, setup_type=trigger_key)
        return result


class SetupStatsStore:
    """In-memory store and file persistence for setup historical statistics."""

    def __init__(self, stats: Mapping[str, SetupHistoricalStats] | None = None) -> None:
        self._stats: dict[str, SetupHistoricalStats] = dict(stats or {})

    def get(self, setup_type: str) -> SetupHistoricalStats | None:
        """Returns statistics for setup_type if registered."""
        return self._stats.get(setup_type)

    def get_or_default(self, setup_type: str) -> SetupHistoricalStats:
        """Returns registered statistics or a neutral unvalidated fallback."""
        if setup_type in self._stats:
            return self._stats[setup_type]
        return SetupHistoricalStats.evaluate_gate_1(
            setup_type=setup_type,
            sample_size=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            expectancy_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=0.0,
        )

    def set(self, setup_type: str, stats: SetupHistoricalStats) -> None:
        """Registers statistics for a setup type."""
        self._stats[setup_type] = stats

    def count(self) -> int:
        """Returns total registered setup categories."""
        return len(self._stats)

    def to_dict(self) -> dict[str, Any]:
        """Serializes all stored stats to dictionary."""
        return {k: v.to_dict() for k, v in self._stats.items()}

    def save_to_file(self, path: Path | str) -> None:
        """Persists stored statistics into a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path | str) -> SetupStatsStore:
        """Loads statistics from a JSON file with defensive fallbacks."""
        target = Path(path)
        if not target.exists():
            return cls()

        try:
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return cls()
            stats_map: dict[str, SetupHistoricalStats] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    stats_map[k] = SetupHistoricalStats.from_dict(v)
            return cls(stats_map)
        except Exception as exc:
            logger.warning("Falha ao carregar setup stats de %s: %s", target, exc)
            return cls()
