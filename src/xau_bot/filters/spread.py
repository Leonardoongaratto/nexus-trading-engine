import math
from datetime import datetime

from xau_bot.filters.models import FilterCheckResult, SpreadFilterConfig
from xau_bot.filters.sessions import TradeBlockReason


class SpreadFilter:
    def __init__(self, config: SpreadFilterConfig | None = None) -> None:
        self.config: SpreadFilterConfig = config if config is not None else SpreadFilterConfig()

    def check(
        self,
        spread: float,
        atr: float,
        timestamp: datetime | None = None,
    ) -> FilterCheckResult:
        for name, val in (("spread", spread), ("atr", atr)):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if spread < 0.0:
            raise ValueError(f"spread cannot be negative, got {spread}")
        if atr <= 0.0:
            raise ValueError(f"atr must be positive, got {atr}")

        if timestamp is not None and timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")

        ratio = spread / atr
        if ratio > self.config.max_spread_atr_ratio:
            return FilterCheckResult(
                can_trade=False,
                reason=TradeBlockReason.SPREAD_ABOVE_THRESHOLD,
                details=(
                    f"Spread/ATR ratio {ratio:.4f} > {self.config.max_spread_atr_ratio:.4f} "
                    f"(spread={spread:.4f}, atr={atr:.4f})"
                ),
                timestamp=timestamp,
            )

        if self.config.max_spread_points is not None:
            spread_points = round(spread / self.config.point_size, 6)
            if spread_points > self.config.max_spread_points:
                return FilterCheckResult(
                    can_trade=False,
                    reason=TradeBlockReason.SPREAD_ABOVE_THRESHOLD,
                    details=(
                        f"Spread {spread_points:.1f} points ({spread:.4f} price) "
                        f"exceeds maximum points {self.config.max_spread_points:.1f}"
                    ),
                    timestamp=timestamp,
                )

        return FilterCheckResult(
            can_trade=True,
            reason=None,
            details=f"Spread/ATR ratio {ratio:.4f} within threshold",
            timestamp=timestamp,
        )
