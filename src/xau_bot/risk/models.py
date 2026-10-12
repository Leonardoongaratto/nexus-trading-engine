import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from xau_bot.domain.models import Direction


class RiskRejectReason(StrEnum):
    LOT_BELOW_MINIMUM = "LOT_BELOW_MINIMUM"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    INVALID_PRICES = "INVALID_PRICES"
    ZERO_OR_NEGATIVE_RISK = "ZERO_OR_NEGATIVE_RISK"
    STOPS_LEVEL_VIOLATION = "STOPS_LEVEL_VIOLATION"
    FREEZE_LEVEL_VIOLATION = "FREEZE_LEVEL_VIOLATION"


@dataclass(frozen=True, slots=True)
class SymbolRiskSpec:
    symbol: str
    contract_size: float
    tick_size: float
    tick_value: float
    lot_min: float
    lot_max: float
    lot_step: float
    point: float
    stops_level_points: float = 0.0
    freeze_level_points: float = 0.0
    commission_per_lot: float = 0.0
    spread_points: float = 0.0

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")

        for name, val in (
            ("contract_size", self.contract_size),
            ("tick_size", self.tick_size),
            ("tick_value", self.tick_value),
            ("lot_min", self.lot_min),
            ("lot_max", self.lot_max),
            ("lot_step", self.lot_step),
            ("point", self.point),
            ("stops_level_points", self.stops_level_points),
            ("freeze_level_points", self.freeze_level_points),
            ("commission_per_lot", self.commission_per_lot),
            ("spread_points", self.spread_points),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        for name, val in (
            ("contract_size", self.contract_size),
            ("tick_size", self.tick_size),
            ("tick_value", self.tick_value),
            ("lot_min", self.lot_min),
            ("lot_max", self.lot_max),
            ("lot_step", self.lot_step),
            ("point", self.point),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        for name, val in (
            ("stops_level_points", self.stops_level_points),
            ("freeze_level_points", self.freeze_level_points),
            ("commission_per_lot", self.commission_per_lot),
            ("spread_points", self.spread_points),
        ):
            if val < 0:
                raise ValueError(f"{name} cannot be negative, got {val}")

        if self.lot_min > self.lot_max:
            raise ValueError(f"lot_min cannot exceed lot_max: {self.lot_min} > {self.lot_max}")

        if self.lot_step > self.lot_min:
            raise ValueError(f"lot_step cannot exceed lot_min: {self.lot_step} > {self.lot_min}")


@dataclass(frozen=True, slots=True)
class AccountState:
    balance: float
    equity: float
    margin: float
    free_margin: float
    leverage: float

    def __post_init__(self) -> None:
        for name, val in (
            ("balance", self.balance),
            ("equity", self.equity),
            ("margin", self.margin),
            ("free_margin", self.free_margin),
            ("leverage", self.leverage),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if self.leverage < 1.0:
            raise ValueError(f"leverage must be at least 1.0, got {self.leverage}")

        if self.margin < 0:
            raise ValueError(f"margin cannot be negative, got {self.margin}")


@dataclass(frozen=True, slots=True)
class PositionRiskConfig:
    risk_per_trade_percent: float = 0.01
    max_risk_per_trade_usd: float | None = None
    fixed_risk_usd: float | None = None
    expected_slippage_points: float = 0.0
    be_trigger_r: float = 1.5
    be_offset_points: float = 0.0
    trailing_swing_buffer_points: float = 5.0
    min_trailing_step_points: float = 10.0
    trailing_atr_multiplier: float = 2.0
    include_commission_in_risk: bool = True
    include_spread_in_risk: bool = True
    partial_close_enabled: bool = True
    partial_close_trigger_r: float = 1.0
    partial_close_ratio: float = 0.5
    remove_tp_after_partial: bool = True
    smart_exit_enabled: bool = True
    smart_exit_min_r: float = 0.5

    def __post_init__(self) -> None:
        for name, val in (
            ("risk_per_trade_percent", self.risk_per_trade_percent),
            ("expected_slippage_points", self.expected_slippage_points),
            ("be_trigger_r", self.be_trigger_r),
            ("be_offset_points", self.be_offset_points),
            ("trailing_swing_buffer_points", self.trailing_swing_buffer_points),
            ("min_trailing_step_points", self.min_trailing_step_points),
            ("trailing_atr_multiplier", self.trailing_atr_multiplier),
            ("partial_close_trigger_r", self.partial_close_trigger_r),
            ("partial_close_ratio", self.partial_close_ratio),
            ("smart_exit_min_r", self.smart_exit_min_r),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if not (0.0 < self.risk_per_trade_percent <= 1.0):
            raise ValueError(
                f"risk_per_trade_percent must be between 0 and 1, got {self.risk_per_trade_percent}"
            )

        if self.expected_slippage_points < 0:
            raise ValueError(
                f"expected_slippage_points cannot be negative, got {self.expected_slippage_points}"
            )

        if self.be_trigger_r <= 0:
            raise ValueError(f"be_trigger_r must be positive, got {self.be_trigger_r}")

        if self.be_offset_points < 0:
            raise ValueError(f"be_offset_points cannot be negative, got {self.be_offset_points}")

        if self.trailing_swing_buffer_points < 0:
            raise ValueError(
                f"trailing_swing_buffer_points cannot be negative, "
                f"got {self.trailing_swing_buffer_points}"
            )

        if self.min_trailing_step_points < 0:
            raise ValueError(
                f"min_trailing_step_points cannot be negative, got {self.min_trailing_step_points}"
            )

        if self.trailing_atr_multiplier <= 0:
            raise ValueError(
                f"trailing_atr_multiplier must be positive, got {self.trailing_atr_multiplier}"
            )

        if self.partial_close_trigger_r <= 0:
            raise ValueError(
                f"partial_close_trigger_r must be positive, got {self.partial_close_trigger_r}"
            )

        if not (0.0 < self.partial_close_ratio < 1.0):
            raise ValueError(
                "partial_close_ratio must be strictly between 0 and 1, "
                f"got {self.partial_close_ratio}"
            )

        if self.smart_exit_min_r < 0:
            raise ValueError(f"smart_exit_min_r cannot be negative, got {self.smart_exit_min_r}")

        for name, opt_val in (
            ("max_risk_per_trade_usd", self.max_risk_per_trade_usd),
            ("fixed_risk_usd", self.fixed_risk_usd),
        ):
            if opt_val is not None:
                if not math.isfinite(opt_val):
                    raise ValueError(f"{name} must be finite, got {opt_val}")
                if opt_val <= 0:
                    raise ValueError(f"{name} must be positive, got {opt_val}")


@dataclass(frozen=True, slots=True)
class LotCalculationResult:
    success: bool
    lot: float
    raw_lot: float
    risk_amount_usd: float
    loss_at_sl_usd: float
    commission_usd: float
    spread_cost_usd: float
    price_distance: float
    slippage_cost_usd: float = 0.0
    reject_reason: RiskRejectReason | None = None
    details: str = ""

    def __post_init__(self) -> None:
        for name, val in (
            ("lot", self.lot),
            ("raw_lot", self.raw_lot),
            ("risk_amount_usd", self.risk_amount_usd),
            ("loss_at_sl_usd", self.loss_at_sl_usd),
            ("commission_usd", self.commission_usd),
            ("spread_cost_usd", self.spread_cost_usd),
            ("price_distance", self.price_distance),
            ("slippage_cost_usd", self.slippage_cost_usd),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")


@dataclass(frozen=True, slots=True)
class MarginValidationResult:
    is_valid: bool
    required_margin: float
    free_margin: float
    margin_level_after_trade: float | None = None
    reject_reason: RiskRejectReason | None = None

    def __post_init__(self) -> None:
        for name, val in (
            ("required_margin", self.required_margin),
            ("free_margin", self.free_margin),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        if self.margin_level_after_trade is not None and not math.isfinite(
            self.margin_level_after_trade
        ):
            raise ValueError(
                f"margin_level_after_trade must be finite, got {self.margin_level_after_trade}"
            )


@dataclass(frozen=True, slots=True)
class BreakevenResult:
    should_modify: bool
    new_stop_loss: float | None = None
    current_r_multiple: float = 0.0
    reason: str = ""
    reject_reason: RiskRejectReason | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.current_r_multiple):
            raise ValueError(f"current_r_multiple must be finite, got {self.current_r_multiple}")
        if self.new_stop_loss is not None:
            if not math.isfinite(self.new_stop_loss):
                raise ValueError(f"new_stop_loss must be finite, got {self.new_stop_loss}")
            if self.new_stop_loss <= 0:
                raise ValueError(f"new_stop_loss must be positive, got {self.new_stop_loss}")


@dataclass(frozen=True, slots=True)
class TrailingStopResult:
    should_modify: bool
    new_stop_loss: float | None = None
    delta_sl: float = 0.0
    reason: str = ""
    reject_reason: RiskRejectReason | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta_sl):
            raise ValueError(f"delta_sl must be finite, got {self.delta_sl}")
        if self.new_stop_loss is not None:
            if not math.isfinite(self.new_stop_loss):
                raise ValueError(f"new_stop_loss must be finite, got {self.new_stop_loss}")
            if self.new_stop_loss <= 0:
                raise ValueError(f"new_stop_loss must be positive, got {self.new_stop_loss}")


@dataclass(frozen=True, slots=True)
class PartialCloseResult:
    should_close: bool
    volume_to_close: float = 0.0
    remaining_volume: float = 0.0
    current_r_multiple: float = 0.0
    reason: str = ""
    reject_reason: RiskRejectReason | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.current_r_multiple):
            raise ValueError(f"current_r_multiple must be finite, got {self.current_r_multiple}")
        if not math.isfinite(self.volume_to_close) or self.volume_to_close < 0:
            raise ValueError(
                f"volume_to_close must be non-negative and finite, got {self.volume_to_close}"
            )
        if not math.isfinite(self.remaining_volume) or self.remaining_volume < 0:
            raise ValueError(
                f"remaining_volume must be non-negative and finite, got {self.remaining_volume}"
            )


@dataclass(frozen=True, slots=True)
class SmartExitDecision:
    should_exit: bool
    reason: str = ""
    reversal_type: str | None = None


@dataclass(frozen=True, slots=True)
class OpenPosition:
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    entry_price: float
    current_stop_loss: float
    initial_stop_loss: float
    open_time: datetime
    current_take_profit: float | None = None
    initial_take_profit: float | None = None

    def __post_init__(self) -> None:
        if self.ticket <= 0:
            raise ValueError(f"ticket must be positive, got {self.ticket}")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not isinstance(self.direction, Direction):
            raise ValueError(f"direction must be a Direction enum, got {self.direction!r}")
        if self.open_time.tzinfo is None:
            raise ValueError("open_time must be timezone-aware")

        for name, val in (
            ("volume", self.volume),
            ("entry_price", self.entry_price),
            ("current_stop_loss", self.current_stop_loss),
            ("initial_stop_loss", self.initial_stop_loss),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        for name, opt_val in (
            ("current_take_profit", self.current_take_profit),
            ("initial_take_profit", self.initial_take_profit),
        ):
            if opt_val is not None:
                if not math.isfinite(opt_val):
                    raise ValueError(f"{name} must be finite, got {opt_val}")
                if opt_val <= 0:
                    raise ValueError(f"{name} must be positive, got {opt_val}")


class GuardHaltReason(StrEnum):
    MAX_DAILY_LOSS_USD = "MAX_DAILY_LOSS_USD"
    MAX_DAILY_LOSS_PERCENT = "MAX_DAILY_LOSS_PERCENT"
    MAX_DAILY_DRAWDOWN_USD = "MAX_DAILY_DRAWDOWN_USD"
    MAX_DAILY_DRAWDOWN_PERCENT = "MAX_DAILY_DRAWDOWN_PERCENT"
    MAX_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    ACCOUNT_EQUITY_DEPLETED = "ACCOUNT_EQUITY_DEPLETED"
    MANUAL_HALT = "MANUAL_HALT"


@dataclass(frozen=True, slots=True)
class DailyRiskLimits:
    max_daily_loss_usd: float | None = None
    max_daily_loss_percent: float | None = None
    max_daily_drawdown_usd: float | None = None
    max_daily_drawdown_percent: float | None = None
    max_consecutive_losses: int | None = None
    reset_consecutive_losses_on_new_day: bool = False

    def __post_init__(self) -> None:
        for name, val in (
            ("max_daily_loss_usd", self.max_daily_loss_usd),
            ("max_daily_drawdown_usd", self.max_daily_drawdown_usd),
        ):
            if val is not None:
                if not math.isfinite(val):
                    raise ValueError(f"{name} must be finite, got {val}")
                if val <= 0:
                    raise ValueError(f"{name} must be positive, got {val}")

        for name, val in (
            ("max_daily_loss_percent", self.max_daily_loss_percent),
            ("max_daily_drawdown_percent", self.max_daily_drawdown_percent),
        ):
            if val is not None:
                if not math.isfinite(val):
                    raise ValueError(f"{name} must be finite, got {val}")
                if val <= 0.0 or val > 1.0:
                    raise ValueError(f"{name} must be between 0 and 1, got {val}")

        if self.max_consecutive_losses is not None and self.max_consecutive_losses < 1:
            raise ValueError(
                f"max_consecutive_losses must be at least 1, got {self.max_consecutive_losses}"
            )


@dataclass(frozen=True, slots=True)
class DailyTradingState:
    trading_day: date
    start_balance: float
    start_equity: float
    peak_equity: float
    current_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    consecutive_losses: int = 0
    trade_count: int = 0
    halted: bool = False
    halt_reason: GuardHaltReason | None = None
    halt_time: datetime | None = None

    def __post_init__(self) -> None:
        for name, val in (
            ("start_balance", self.start_balance),
            ("start_equity", self.start_equity),
            ("current_equity", self.current_equity),
            ("peak_equity", self.peak_equity),
            ("realized_pnl", self.realized_pnl),
            ("unrealized_pnl", self.unrealized_pnl),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        for name, val in (
            ("start_balance", self.start_balance),
            ("start_equity", self.start_equity),
            ("peak_equity", self.peak_equity),
        ):
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        if self.consecutive_losses < 0:
            raise ValueError(
                f"consecutive_losses cannot be negative, got {self.consecutive_losses}"
            )

        if self.trade_count < 0:
            raise ValueError(f"trade_count cannot be negative, got {self.trade_count}")

        if self.halt_time is not None and self.halt_time.tzinfo is None:
            raise ValueError("halt_time must be timezone-aware")

    @property
    def daily_pnl(self) -> float:
        return self.current_equity - self.start_equity

    @property
    def current_drawdown_usd(self) -> float:
        return max(0.0, self.peak_equity - self.current_equity)

    @property
    def current_drawdown_percent(self) -> float:
        return self.current_drawdown_usd / self.peak_equity


@dataclass(frozen=True, slots=True)
class GuardCheckResult:
    can_trade: bool
    halted: bool
    halt_reason: GuardHaltReason | None = None
    details: str = ""
    daily_pnl: float = 0.0
    current_drawdown_usd: float = 0.0
    current_drawdown_percent: float = 0.0
    consecutive_losses: int = 0

    def __post_init__(self) -> None:
        for name, val in (
            ("daily_pnl", self.daily_pnl),
            ("current_drawdown_usd", self.current_drawdown_usd),
            ("current_drawdown_percent", self.current_drawdown_percent),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")

        for name, val in (
            ("current_drawdown_usd", self.current_drawdown_usd),
            ("current_drawdown_percent", self.current_drawdown_percent),
        ):
            if val < 0:
                raise ValueError(f"{name} cannot be negative, got {val}")

        if self.consecutive_losses < 0:
            raise ValueError(
                f"consecutive_losses cannot be negative, got {self.consecutive_losses}"
            )
