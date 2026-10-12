"""Domain models and data structures for the Trade Journal module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.domain.models import Direction


class TradeOutcome(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    UNEXECUTED = "UNEXECUTED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class LossCategory(StrEnum):
    NORMAL_VARIANCE = "NORMAL_VARIANCE"
    BAD_DECISION = "BAD_DECISION"


class LossRootCause(StrEnum):
    NORMAL_VARIANCE = "NORMAL_VARIANCE"
    COUNTER_BIAS = "COUNTER_BIAS"
    INSUFFICIENT_DISPLACEMENT = "INSUFFICIENT_DISPLACEMENT"
    NEWS_PROXIMITY = "NEWS_PROXIMITY"
    HIGH_SPREAD = "HIGH_SPREAD"
    TIGHT_SL = "TIGHT_SL"
    PREMATURE_EXIT = "PREMATURE_EXIT"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


@dataclass(frozen=True, slots=True)
class SnapshotData:
    timestamp: datetime
    session: str
    htf_bias: str
    swing_high: float
    swing_low: float
    poi_type: str
    poi_level: float
    swept_liquidity: bool
    displacement_atr: float
    fvg_size_atr: float
    choch_level: float
    spread_price: float
    atr_m5: float
    news_distance_min: float | None = None
    engine_version: str = "1.0.0"
    params_hash: str = ""
    user_decision: str = ""
    user_reason: str = ""
    confluences: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "session": self.session,
            "htf_bias": self.htf_bias,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "poi_type": self.poi_type,
            "poi_level": self.poi_level,
            "swept_liquidity": self.swept_liquidity,
            "displacement_atr": self.displacement_atr,
            "fvg_size_atr": self.fvg_size_atr,
            "choch_level": self.choch_level,
            "spread_price": self.spread_price,
            "atr_m5": self.atr_m5,
            "news_distance_min": self.news_distance_min,
            "engine_version": self.engine_version,
            "params_hash": self.params_hash,
            "user_decision": self.user_decision,
            "user_reason": self.user_reason,
            "confluences": list(self.confluences),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotData:
        raw_ts = data["timestamp"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            timestamp=ts,
            session=str(data.get("session", "")),
            htf_bias=str(data.get("htf_bias", "")),
            swing_high=float(data.get("swing_high", 0.0)),
            swing_low=float(data.get("swing_low", 0.0)),
            poi_type=str(data.get("poi_type", "")),
            poi_level=float(data.get("poi_level", 0.0)),
            swept_liquidity=bool(data.get("swept_liquidity", False)),
            displacement_atr=float(data.get("displacement_atr", 0.0)),
            fvg_size_atr=float(data.get("fvg_size_atr", 0.0)),
            choch_level=float(data.get("choch_level", 0.0)),
            spread_price=float(data.get("spread_price", 0.0)),
            atr_m5=float(data.get("atr_m5", 0.0)),
            news_distance_min=(
                float(data["news_distance_min"])
                if data.get("news_distance_min") is not None
                else None
            ),
            engine_version=str(data.get("engine_version", "1.0.0")),
            params_hash=str(data.get("params_hash", "")),
            user_decision=str(data.get("user_decision", "")),
            user_reason=str(data.get("user_reason", "")),
            confluences=tuple(data.get("confluences", ())),
        )


@dataclass(frozen=True, slots=True)
class OutcomeData:
    executed: bool
    entry_price: float
    stop_loss: float
    take_profit: float
    fill_price: float | None = None
    exit_price: float | None = None
    slippage_points: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    max_r: float = 0.0
    realized_r: float = 0.0
    pnl_usd: float = 0.0
    duration_seconds: float = 0.0
    exit_reason: str = ""
    post_exit_behavior: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "fill_price": self.fill_price,
            "exit_price": self.exit_price,
            "slippage_points": self.slippage_points,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "max_r": self.max_r,
            "realized_r": self.realized_r,
            "pnl_usd": self.pnl_usd,
            "duration_seconds": self.duration_seconds,
            "exit_reason": self.exit_reason,
            "post_exit_behavior": self.post_exit_behavior,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeData:
        return cls(
            executed=bool(data.get("executed", False)),
            entry_price=float(data.get("entry_price", 0.0)),
            stop_loss=float(data.get("stop_loss", 0.0)),
            take_profit=float(data.get("take_profit", 0.0)),
            fill_price=float(data["fill_price"]) if data.get("fill_price") is not None else None,
            exit_price=float(data["exit_price"]) if data.get("exit_price") is not None else None,
            slippage_points=float(data.get("slippage_points", 0.0)),
            mfe_r=float(data.get("mfe_r", 0.0)),
            mae_r=float(data.get("mae_r", 0.0)),
            max_r=float(data.get("max_r", 0.0)),
            realized_r=float(data.get("realized_r", 0.0)),
            pnl_usd=float(data.get("pnl_usd", 0.0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            exit_reason=str(data.get("exit_reason", "")),
            post_exit_behavior=str(data.get("post_exit_behavior", "")),
        )


@dataclass(slots=True)
class JournalEntry:
    id: str
    proposal_id: str
    snapshot: SnapshotData
    ticket: int | None = None
    symbol: str = "XAUUSD"
    direction: Direction = Direction.BULLISH
    status: ProposalStatus = ProposalStatus.PENDING_APPROVAL
    outcome: TradeOutcome = TradeOutcome.UNEXECUTED
    loss_cause: LossRootCause | None = None
    loss_category: LossCategory | None = None
    outcome_data: OutcomeData | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
