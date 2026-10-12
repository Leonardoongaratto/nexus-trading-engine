from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from xau_bot.domain.models import Candle, Timeframe


class DegradationReason(StrEnum):
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    TICK_STALE = "TICK_STALE"
    LOOP_STALE = "LOOP_STALE"
    TRADING_HALTED = "TRADING_HALTED"


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    max_tick_age_s: float = 30.0
    max_loop_silence_s: float = 120.0
    check_interval_s: float = 1.0
    ignore_stale_when_market_closed: bool = True
    auto_reconnect: bool = True
    reconnect_backoff_base_s: float = 1.0
    reconnect_backoff_max_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_tick_age_s <= 0.0:
            raise ValueError("max_tick_age_s must be positive")
        if self.max_loop_silence_s <= 0.0:
            raise ValueError("max_loop_silence_s must be positive")
        if self.check_interval_s <= 0.0:
            raise ValueError("check_interval_s must be positive")
        if self.reconnect_backoff_base_s <= 0.0:
            raise ValueError("reconnect_backoff_base_s must be positive")
        if self.reconnect_backoff_max_s < self.reconnect_backoff_base_s:
            raise ValueError("reconnect_backoff_max_s must be >= reconnect_backoff_base_s")


class RuntimeStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class WatchdogHealth:
    is_healthy: bool
    broker_connected: bool
    last_tick_age_s: float
    trading_halted: bool
    reasons: tuple[DegradationReason, ...]
    status_code: int
    status: str
    last_loop_beat_age_s: float
    timestamp: datetime


class HealthSyncTarget(Protocol):
    trading_halted: bool

    def apply_watchdog_health(self, health: WatchdogHealth) -> None: ...


@dataclass(frozen=True, slots=True)
class CandleCloseEvent:
    timeframe: Timeframe
    candle: Candle
    closed_series: tuple[Candle, ...]
    forming_candle: Candle | None
    closed_at: datetime
