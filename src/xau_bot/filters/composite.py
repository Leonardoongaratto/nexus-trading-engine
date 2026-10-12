from datetime import datetime

from xau_bot.filters.models import FilterCheckResult
from xau_bot.filters.news import NewsFilter
from xau_bot.filters.sessions import SessionFilter, TradeBlockReason
from xau_bot.filters.spread import SpreadFilter


class CompositeTradeFilter:
    def __init__(
        self,
        session_filter: SessionFilter | None = None,
        news_filter: NewsFilter | None = None,
        spread_filter: SpreadFilter | None = None,
    ) -> None:
        self.session_filter: SessionFilter = (
            session_filter if session_filter is not None else SessionFilter()
        )
        self.news_filter: NewsFilter | None = news_filter
        self.spread_filter: SpreadFilter | None = spread_filter

    def check(
        self,
        current_time: datetime,
        spread: float | None = None,
        atr: float | None = None,
    ) -> FilterCheckResult:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")

        session_can_trade, session_reason = self.session_filter.can_trade(current_time)
        if not session_can_trade:
            return FilterCheckResult(
                can_trade=False,
                reason=session_reason,
                details=f"Session filter blocked: {session_reason}",
                timestamp=current_time,
            )

        if self.news_filter is not None:
            news_res = self.news_filter.check(current_time)
            if not news_res.can_trade:
                return news_res

        if self.spread_filter is not None:
            if spread is None or atr is None:
                return FilterCheckResult(
                    can_trade=False,
                    reason=TradeBlockReason.MARKET_DATA_UNAVAILABLE,
                    details=(
                        f"Spread filter active but market data missing (spread={spread}, atr={atr})"
                    ),
                    timestamp=current_time,
                )
            spread_res = self.spread_filter.check(spread=spread, atr=atr, timestamp=current_time)
            if not spread_res.can_trade:
                return spread_res

        return FilterCheckResult(
            can_trade=True,
            reason=None,
            details="All trade filters passed",
            timestamp=current_time,
        )
