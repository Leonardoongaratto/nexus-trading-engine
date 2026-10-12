import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xau_bot.filters.sessions import TradeBlockReason


class NewsImpact(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    HOLIDAY = "HOLIDAY"


@dataclass(frozen=True, slots=True)
class NewsEvent:
    title: str
    currency: str
    impact: NewsImpact
    time: datetime
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.currency.strip():
            raise ValueError("currency must not be empty")
        if self.time.tzinfo is None:
            raise ValueError("time must be timezone-aware")

        for name, val in (
            ("actual", self.actual),
            ("forecast", self.forecast),
            ("previous", self.previous),
        ):
            if val is not None and not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")


@dataclass(frozen=True, slots=True)
class SpreadFilterConfig:
    max_spread_atr_ratio: float = 0.05
    max_spread_points: float | None = None
    point_size: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_spread_atr_ratio):
            raise ValueError(
                f"max_spread_atr_ratio must be finite, got {self.max_spread_atr_ratio}"
            )
        if self.max_spread_atr_ratio <= 0.0:
            raise ValueError(
                f"max_spread_atr_ratio must be positive, got {self.max_spread_atr_ratio}"
            )

        if not math.isfinite(self.point_size):
            raise ValueError(f"point_size must be finite, got {self.point_size}")
        if self.point_size <= 0.0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")

        if self.max_spread_points is not None:
            if not math.isfinite(self.max_spread_points):
                raise ValueError(f"max_spread_points must be finite, got {self.max_spread_points}")
            if self.max_spread_points <= 0.0:
                raise ValueError(
                    f"max_spread_points must be positive, got {self.max_spread_points}"
                )


@dataclass(frozen=True, slots=True)
class NewsFilterConfig:
    blackout_minutes_before: float = 30.0
    blackout_minutes_after: float = 30.0
    currencies: tuple[str, ...] = ("USD",)
    min_impact: NewsImpact = NewsImpact.HIGH
    max_age_hours: float | None = None

    def __post_init__(self) -> None:
        for name, val in (
            ("blackout_minutes_before", self.blackout_minutes_before),
            ("blackout_minutes_after", self.blackout_minutes_after),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if val < 0.0:
                raise ValueError(f"{name} cannot be negative, got {val}")

        if self.max_age_hours is not None:
            if not math.isfinite(self.max_age_hours):
                raise ValueError(f"max_age_hours must be finite, got {self.max_age_hours}")
            if self.max_age_hours <= 0.0:
                raise ValueError(f"max_age_hours must be positive, got {self.max_age_hours}")

        if not self.currencies:
            raise ValueError("currencies cannot be empty")

        normalized: list[str] = []
        for c in self.currencies:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("currency must be a non-empty string")
            normalized.append(c.strip().upper())
        object.__setattr__(self, "currencies", tuple(normalized))


@dataclass(frozen=True, slots=True)
class FilterCheckResult:
    can_trade: bool
    reason: TradeBlockReason | None = None
    details: str = ""
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.can_trade != (self.reason is None):
            raise ValueError(
                f"Incoherent FilterCheckResult: can_trade={self.can_trade} but reason={self.reason}"
            )
