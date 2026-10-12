import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class GuardianKillSwitchReason(StrEnum):
    DAILY_LOSS_BREACHED = "DAILY_LOSS_BREACHED"
    DRAWDOWN_BREACHED = "DRAWDOWN_BREACHED"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True, slots=True)
class HeartbeatPayload:
    timestamp: datetime
    status: str
    pid: int
    sequence: int
    trading_halted: bool = False

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.status.strip():
            raise ValueError("status must not be empty")
        if self.pid <= 0:
            raise ValueError(f"pid must be positive, got {self.pid}")
        if self.sequence < 0:
            raise ValueError(f"sequence must be non-negative, got {self.sequence}")

    def to_dict(self) -> dict[str, Any]:
        dt_utc = self.timestamp.astimezone(UTC)
        return {
            "timestamp": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": self.status,
            "pid": self.pid,
            "sequence": self.sequence,
            "trading_halted": self.trading_halted,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeartbeatPayload":
        raw_time = data["timestamp"]
        dt = raw_time if isinstance(raw_time, datetime) else datetime.fromisoformat(str(raw_time))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return cls(
            timestamp=dt.astimezone(UTC),
            status=str(data["status"]),
            pid=int(data["pid"]),
            sequence=int(data["sequence"]),
            trading_halted=bool(data.get("trading_halted", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "HeartbeatPayload":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON must decode to a dictionary")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class GuardianSignal:
    timestamp: datetime
    reason: GuardianKillSwitchReason | str
    current_loss_pct: float | None = None
    max_loss_pct: float | None = None
    positions_closed: int = 0
    orders_cancelled: int = 0
    details: str = ""

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.positions_closed < 0:
            raise ValueError(f"positions_closed cannot be negative, got {self.positions_closed}")
        if self.orders_cancelled < 0:
            raise ValueError(f"orders_cancelled cannot be negative, got {self.orders_cancelled}")
        if self.current_loss_pct is not None and not math.isfinite(self.current_loss_pct):
            raise ValueError("current_loss_pct must be finite")
        if self.max_loss_pct is not None and not math.isfinite(self.max_loss_pct):
            raise ValueError("max_loss_pct must be finite")

    def to_dict(self) -> dict[str, Any]:
        dt_utc = self.timestamp.astimezone(UTC)
        reason_val = self.reason.value if isinstance(self.reason, StrEnum) else str(self.reason)
        return {
            "timestamp": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reason": reason_val,
            "current_loss_pct": self.current_loss_pct,
            "max_loss_pct": self.max_loss_pct,
            "positions_closed": self.positions_closed,
            "orders_cancelled": self.orders_cancelled,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardianSignal":
        raw_time = data["timestamp"]
        dt = raw_time if isinstance(raw_time, datetime) else datetime.fromisoformat(str(raw_time))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        raw_reason = str(data["reason"])
        reason: GuardianKillSwitchReason | str
        try:
            reason = GuardianKillSwitchReason(raw_reason)
        except ValueError:
            reason = raw_reason

        return cls(
            timestamp=dt.astimezone(UTC),
            reason=reason,
            current_loss_pct=(
                float(data["current_loss_pct"])
                if data.get("current_loss_pct") is not None
                else None
            ),
            max_loss_pct=(
                float(data["max_loss_pct"]) if data.get("max_loss_pct") is not None else None
            ),
            positions_closed=int(data.get("positions_closed", 0)),
            orders_cancelled=int(data.get("orders_cancelled", 0)),
            details=str(data.get("details", "")),
        )

    @classmethod
    def from_json(cls, text: str) -> "GuardianSignal":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("JSON must decode to a dictionary")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class GuardianConfig:
    common_dir: Path
    heartbeat_filename: str = "xau_heartbeat.json"
    killswitch_filename: str = "xau_killswitch.json"
    calendar_filename: str = "calendar.json"
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 30.0
    calendar_max_age_h: float = 24.0
    magic_number: int = 202610

    def __post_init__(self) -> None:
        if self.heartbeat_interval_s <= 0.0 or not math.isfinite(self.heartbeat_interval_s):
            raise ValueError("heartbeat_interval_s must be positive and finite")
        if self.heartbeat_timeout_s <= 0.0 or not math.isfinite(self.heartbeat_timeout_s):
            raise ValueError("heartbeat_timeout_s must be positive and finite")
        if self.calendar_max_age_h <= 0.0 or not math.isfinite(self.calendar_max_age_h):
            raise ValueError("calendar_max_age_h must be positive and finite")
        if self.magic_number <= 0:
            raise ValueError("magic_number must be positive")
