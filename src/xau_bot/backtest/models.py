import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from xau_bot.domain.models import Direction
from xau_bot.risk.models import SymbolRiskSpec


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class OrderType(StrEnum):
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_MARKET = "BUY_MARKET"
    SELL_MARKET = "SELL_MARKET"


class TradeExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    BREAKEVEN = "BREAKEVEN"
    TRAILING_STOP = "TRAILING_STOP"
    KILL_SWITCH = "KILL_SWITCH"
    MANUAL = "MANUAL"
    END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True, slots=True)
class BacktestOrder:
    order_id: str
    direction: Direction
    order_type: OrderType
    volume: float
    limit_price: float
    stop_loss: float
    take_profit: float
    created_at: datetime
    setup_id: str | None = None
    expires_at: datetime | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_at: datetime | None = None
    fill_price: float | None = None
    cancellation_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must not be empty")

        if not math.isfinite(self.volume) or self.volume <= 0:
            raise ValueError(f"volume must be positive and finite, got {self.volume}")

        for name, val in (
            ("limit_price", self.limit_price),
            ("stop_loss", self.stop_loss),
            ("take_profit", self.take_profit),
        ):
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be positive and finite, got {val}")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        if self.expires_at is not None:
            if self.expires_at.tzinfo is None:
                raise ValueError("expires_at must be timezone-aware")
            if self.expires_at < self.created_at:
                raise ValueError(
                    f"expires_at must be >= created_at: {self.expires_at} < {self.created_at}"
                )

        if self.filled_at is not None and self.filled_at.tzinfo is None:
            raise ValueError("filled_at must be timezone-aware")

        if self.fill_price is not None and (
            not math.isfinite(self.fill_price) or self.fill_price <= 0
        ):
            raise ValueError(f"fill_price must be positive and finite, got {self.fill_price}")

        if self.order_type in (OrderType.BUY_LIMIT, OrderType.BUY_MARKET):
            if self.stop_loss >= self.limit_price:
                raise ValueError(
                    f"Buy order stop_loss ({self.stop_loss}) must be < "
                    f"limit_price ({self.limit_price})"
                )
            if self.take_profit <= self.limit_price:
                raise ValueError(
                    f"Buy order take_profit ({self.take_profit}) must be > "
                    f"limit_price ({self.limit_price})"
                )
        elif self.order_type in (OrderType.SELL_LIMIT, OrderType.SELL_MARKET):
            if self.stop_loss <= self.limit_price:
                raise ValueError(
                    f"Sell order stop_loss ({self.stop_loss}) must be > "
                    f"limit_price ({self.limit_price})"
                )
            if self.take_profit >= self.limit_price:
                raise ValueError(
                    f"Sell order take_profit ({self.take_profit}) must be < "
                    f"limit_price ({self.limit_price})"
                )


@dataclass(frozen=True, slots=True)
class BacktestTrade:
    trade_id: str
    order_id: str
    direction: Direction
    volume: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: TradeExitReason
    initial_sl: float
    final_sl: float
    tp: float
    commission: float
    swap: float
    gross_pnl: float
    net_pnl: float
    return_r: float
    mfe_price: float
    mae_price: float
    mfe_r: float
    mae_r: float
    duration_seconds: float
    setup_id: str | None = None

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise ValueError("trade_id must not be empty")
        if not self.order_id:
            raise ValueError("order_id must not be empty")

        if not math.isfinite(self.volume) or self.volume <= 0:
            raise ValueError(f"volume must be positive and finite, got {self.volume}")

        if self.entry_time.tzinfo is None:
            raise ValueError("entry_time must be timezone-aware")
        if self.exit_time.tzinfo is None:
            raise ValueError("exit_time must be timezone-aware")
        if self.exit_time < self.entry_time:
            raise ValueError(
                f"exit_time must be >= entry_time: {self.exit_time} < {self.entry_time}"
            )

        for name, val in (
            ("entry_price", self.entry_price),
            ("exit_price", self.exit_price),
            ("initial_sl", self.initial_sl),
            ("final_sl", self.final_sl),
            ("tp", self.tp),
            ("commission", self.commission),
            ("swap", self.swap),
            ("gross_pnl", self.gross_pnl),
            ("net_pnl", self.net_pnl),
            ("return_r", self.return_r),
            ("mfe_price", self.mfe_price),
            ("mae_price", self.mae_price),
            ("mfe_r", self.mfe_r),
            ("mae_r", self.mae_r),
            ("duration_seconds", self.duration_seconds),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    balance: float
    equity: float
    drawdown_usd: float
    drawdown_pct: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        for name, val in (
            ("balance", self.balance),
            ("equity", self.equity),
            ("drawdown_usd", self.drawdown_usd),
            ("drawdown_pct", self.drawdown_pct),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")


def _default_xau_risk_spec() -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=0.0,
        freeze_level_points=0.0,
        commission_per_lot=7.0,
        spread_points=25.0,
    )


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_balance: float = 10000.0
    leverage: float = 100.0
    symbol_spec: SymbolRiskSpec = field(default_factory=_default_xau_risk_spec)
    commission_per_lot: float | None = None
    slippage_points: float = 0.0
    fill_limit_on_touch: bool = True
    max_open_positions: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_balance) or self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {self.initial_balance}")
        if not math.isfinite(self.leverage) or self.leverage < 1.0:
            raise ValueError(f"leverage must be >= 1.0, got {self.leverage}")
        if self.commission_per_lot is None:
            object.__setattr__(self, "commission_per_lot", self.symbol_spec.commission_per_lot)
        elif not math.isfinite(self.commission_per_lot) or self.commission_per_lot < 0:
            raise ValueError(
                f"commission_per_lot cannot be negative, got {self.commission_per_lot}"
            )
        if not math.isfinite(self.slippage_points) or self.slippage_points < 0:
            raise ValueError(f"slippage_points cannot be negative, got {self.slippage_points}")
        if self.max_open_positions < 1:
            raise ValueError(
                f"max_open_positions must be at least 1, got {self.max_open_positions}"
            )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[EquityPoint, ...]
    initial_balance: float
    final_balance: float
    total_net_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    expectancy_r: float
    average_trade_duration_seconds: float
