from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderStatus,
    OrderType,
    SymbolInfo,
)
from xau_bot.domain.models import Direction, Timeframe

TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408

TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8
TRADE_ACTION_CLOSE_BY = 10

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

ORDER_STATE_STARTED = 0
ORDER_STATE_PLACED = 1
ORDER_STATE_CANCELED = 2
ORDER_STATE_PARTIAL = 3
ORDER_STATE_FILLED = 4
ORDER_STATE_REJECTED = 5
ORDER_STATE_EXPIRED = 6

SYMBOL_FILLING_FOK = 1
SYMBOL_FILLING_IOC = 2
SYMBOL_FILLING_RETURN = 4

ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

RETCODE_REQUOTE = 10004
RETCODE_REJECT = 10006
RETCODE_CANCEL = 10007
RETCODE_PLACED = 10008
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
RETCODE_ERROR = 10011
RETCODE_TIMEOUT = 10012
RETCODE_INVALID = 10013
RETCODE_INVALID_VOLUME = 10014
RETCODE_INVALID_PRICE = 10015
RETCODE_INVALID_STOPS = 10016
RETCODE_TRADE_DISABLED = 10017
RETCODE_MARKET_CLOSED = 10018
RETCODE_NO_MONEY = 10019
RETCODE_PRICE_CHANGED = 10020
RETCODE_PRICE_OFF = 10021
RETCODE_INVALID_EXPIRATION = 10022
RETCODE_ORDER_CHANGED = 10023
RETCODE_TOO_MANY_REQUESTS = 10024
RETCODE_NO_CHANGES = 10025
RETCODE_SERVER_DISABLES_AT = 10026
RETCODE_CLIENT_DISABLES_AT = 10027
RETCODE_LOCKED = 10028
RETCODE_FROZEN = 10029
RETCODE_INVALID_FILL = 10030
RETCODE_CONNECTION = 10031


def resolve_filling_mode(filling_mode_mask: int) -> int:
    if filling_mode_mask & SYMBOL_FILLING_FOK:
        return ORDER_FILLING_FOK
    if filling_mode_mask & SYMBOL_FILLING_IOC:
        return ORDER_FILLING_IOC
    return ORDER_FILLING_RETURN


def map_order_type_to_mt5(order_type: OrderType) -> int:
    mapping = {
        OrderType.BUY: ORDER_TYPE_BUY,
        OrderType.SELL: ORDER_TYPE_SELL,
        OrderType.BUY_LIMIT: ORDER_TYPE_BUY_LIMIT,
        OrderType.SELL_LIMIT: ORDER_TYPE_SELL_LIMIT,
        OrderType.BUY_STOP: ORDER_TYPE_BUY_STOP,
        OrderType.SELL_STOP: ORDER_TYPE_SELL_STOP,
    }
    return mapping[order_type]


def map_mt5_to_order_type(mt5_type: int) -> OrderType:
    mapping = {
        ORDER_TYPE_BUY: OrderType.BUY,
        ORDER_TYPE_SELL: OrderType.SELL,
        ORDER_TYPE_BUY_LIMIT: OrderType.BUY_LIMIT,
        ORDER_TYPE_SELL_LIMIT: OrderType.SELL_LIMIT,
        ORDER_TYPE_BUY_STOP: OrderType.BUY_STOP,
        ORDER_TYPE_SELL_STOP: OrderType.SELL_STOP,
    }
    if mt5_type not in mapping:
        raise ValueError(f"Unknown MT5 order type: {mt5_type}")
    return mapping[mt5_type]


def map_mt5_to_order_status(state: int) -> OrderStatus:
    mapping = {
        ORDER_STATE_STARTED: OrderStatus.PENDING,
        ORDER_STATE_PLACED: OrderStatus.PENDING,
        ORDER_STATE_CANCELED: OrderStatus.CANCELED,
        ORDER_STATE_PARTIAL: OrderStatus.FILLED,
        ORDER_STATE_FILLED: OrderStatus.FILLED,
        ORDER_STATE_REJECTED: OrderStatus.REJECTED,
        ORDER_STATE_EXPIRED: OrderStatus.EXPIRED,
    }
    return mapping.get(state, OrderStatus.PENDING)


def map_mt5_account_info(info: Any) -> AccountInfo:
    return AccountInfo(
        balance=float(getattr(info, "balance", 0.0)),
        equity=float(getattr(info, "equity", 0.0)),
        margin=float(getattr(info, "margin", 0.0)),
        free_margin=float(getattr(info, "margin_free", 0.0)),
        margin_level=float(getattr(info, "margin_level", 0.0)),
        leverage=float(getattr(info, "leverage", 100.0)),
        currency=str(getattr(info, "currency", "USD")),
    )


def map_mt5_symbol_info(info: Any) -> SymbolInfo:
    point = float(getattr(info, "point", 0.01))
    stops_level_raw = float(getattr(info, "trade_stops_level", 0.0))
    freeze_level_raw = float(getattr(info, "trade_freeze_level", 0.0))
    spread_raw = float(getattr(info, "spread", 0.0))

    stops_level_points = stops_level_raw * point
    freeze_level_points = freeze_level_raw * point
    spread_points = spread_raw * point

    tick_size = float(getattr(info, "trade_tick_size", point))
    tick_value = float(getattr(info, "trade_tick_value", 1.0))

    return SymbolInfo(
        name=str(getattr(info, "name", "")),
        bid=float(getattr(info, "bid", 0.0)),
        ask=float(getattr(info, "ask", 0.0)),
        spread_points=spread_points,
        point_size=point,
        contract_size=float(getattr(info, "trade_contract_size", 100.0)),
        min_volume=float(getattr(info, "volume_min", 0.01)),
        max_volume=float(getattr(info, "volume_max", 100.0)),
        volume_step=float(getattr(info, "volume_step", 0.01)),
        stops_level_points=stops_level_points,
        freeze_level_points=freeze_level_points,
        tick_size=tick_size,
        tick_value=tick_value,
    )


def map_mt5_position(pos: Any, server_tz: str | ZoneInfo | None = None) -> BrokerPosition:
    pos_type = int(getattr(pos, "type", 0))
    direction = Direction.BULLISH if pos_type == POSITION_TYPE_BUY else Direction.BEARISH

    tz = ZoneInfo(server_tz) if isinstance(server_tz, str) else (server_tz or UTC)
    raw_time = getattr(pos, "time", 0)
    if isinstance(raw_time, datetime):
        open_time = raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=tz).astimezone(UTC)
    else:
        try:
            num_t = float(raw_time)
            if num_t > 0:
                open_time = datetime.fromtimestamp(num_t, tz=UTC).replace(tzinfo=tz).astimezone(UTC)
            else:
                open_time = datetime.now(tz=UTC)
        except (TypeError, ValueError):
            open_time = datetime.now(tz=UTC)

    sl = getattr(pos, "sl", None)
    tp = getattr(pos, "tp", None)
    sl_val = float(sl) if sl is not None and float(sl) > 0 else None
    tp_val = float(tp) if tp is not None and float(tp) > 0 else None

    return BrokerPosition(
        ticket=int(getattr(pos, "ticket", 0)),
        symbol=str(getattr(pos, "symbol", "")),
        direction=direction,
        volume=float(getattr(pos, "volume", 0.0)),
        open_price=float(getattr(pos, "price_open", 0.0)),
        open_time=open_time,
        stop_loss=sl_val,
        take_profit=tp_val,
        current_price=float(getattr(pos, "price_current", 0.0)),
        floating_pnl=float(getattr(pos, "profit", 0.0)),
        commission=float(getattr(pos, "commission", 0.0)),
        comment=str(getattr(pos, "comment", "")),
        magic=int(getattr(pos, "magic", 0)),
    )


def map_mt5_order(ord_obj: Any, server_tz: str | ZoneInfo | None = None) -> BrokerOrder:
    raw_type = int(getattr(ord_obj, "type", 0))
    order_type = map_mt5_to_order_type(raw_type)

    raw_state = int(getattr(ord_obj, "state", 0))
    status = map_mt5_to_order_status(raw_state)

    tz = ZoneInfo(server_tz) if isinstance(server_tz, str) else (server_tz or UTC)
    raw_time = getattr(ord_obj, "time_setup", 0)
    if isinstance(raw_time, datetime):
        time_setup = raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=tz).astimezone(UTC)
    else:
        try:
            num_t = float(raw_time)
            if num_t > 0:
                time_setup = (
                    datetime.fromtimestamp(num_t, tz=UTC).replace(tzinfo=tz).astimezone(UTC)
                )
            else:
                time_setup = datetime.now(tz=UTC)
        except (TypeError, ValueError):
            time_setup = datetime.now(tz=UTC)

    sl = getattr(ord_obj, "sl", None)
    tp = getattr(ord_obj, "tp", None)
    sl_val = float(sl) if sl is not None and float(sl) > 0 else None
    tp_val = float(tp) if tp is not None and float(tp) > 0 else None

    volume = float(getattr(ord_obj, "volume_current", getattr(ord_obj, "volume_initial", 0.0)))
    price = float(getattr(ord_obj, "price_open", 0.0))

    return BrokerOrder(
        ticket=int(getattr(ord_obj, "ticket", 0)),
        symbol=str(getattr(ord_obj, "symbol", "")),
        order_type=order_type,
        volume=volume,
        price=price,
        time_setup=time_setup,
        stop_loss=sl_val,
        take_profit=tp_val,
        status=status,
        comment=str(getattr(ord_obj, "comment", "")),
        magic=int(getattr(ord_obj, "magic", 0)),
    )


def map_timeframe_to_mt5(timeframe: Timeframe) -> int:
    mapping = {
        Timeframe.M1: TIMEFRAME_M1,
        Timeframe.M5: TIMEFRAME_M5,
        Timeframe.M15: TIMEFRAME_M15,
        Timeframe.H1: TIMEFRAME_H1,
        Timeframe.H4: TIMEFRAME_H4,
        Timeframe.D1: TIMEFRAME_D1,
    }
    return mapping[timeframe]
