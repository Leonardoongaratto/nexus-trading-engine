from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from xau_bot.domain.models import Direction


class OrderType(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: float
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    name: str
    bid: float
    ask: float
    spread_points: float = 0.25
    point_size: float = 0.01
    contract_size: float = 100.0
    min_volume: float = 0.01
    max_volume: float = 100.0
    volume_step: float = 0.01
    stops_level_points: float = 1.0
    freeze_level_points: float = 0.5
    tick_size: float = 0.01
    tick_value: float = 1.0


@dataclass(frozen=True, slots=True)
class OrderRequest:
    symbol: str
    order_type: OrderType
    volume: float
    price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    comment: str = ""
    magic: int = 0


@dataclass(frozen=True, slots=True)
class OrderResult:
    success: bool
    ticket: int | None = None
    price: float = 0.0
    volume: float = 0.0
    error_code: int = 0
    error_message: str = ""
    retcode: int = 10009


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    ticket: int
    symbol: str
    direction: Direction
    volume: float
    open_price: float
    open_time: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    current_price: float = 0.0
    floating_pnl: float = 0.0
    commission: float = 0.0
    comment: str = ""
    magic: int = 0


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    time_setup: datetime
    stop_loss: float | None = None
    take_profit: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    comment: str = ""
    magic: int = 0
