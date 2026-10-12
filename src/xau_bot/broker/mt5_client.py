import importlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, cast, runtime_checkable

from xau_bot.broker.mt5_mapping import (
    ORDER_STATE_PLACED,
    ORDER_TYPE_BUY,
    ORDER_TYPE_BUY_LIMIT,
    POSITION_TYPE_BUY,
    POSITION_TYPE_SELL,
    RETCODE_CONNECTION,
    RETCODE_DONE,
    RETCODE_INVALID,
    RETCODE_INVALID_VOLUME,
    RETCODE_NO_MONEY,
    RETCODE_PLACED,
    RETCODE_TRADE_DISABLED,
    SYMBOL_FILLING_FOK,
    SYMBOL_FILLING_IOC,
    TRADE_ACTION_DEAL,
    TRADE_ACTION_MODIFY,
    TRADE_ACTION_PENDING,
    TRADE_ACTION_REMOVE,
    TRADE_ACTION_SLTP,
)


@runtime_checkable
class MT5ClientProtocol(Protocol):
    def initialize(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        portable: bool = False,
    ) -> bool: ...

    def shutdown(self) -> None: ...

    def terminal_info(self) -> Any | None: ...

    def account_info(self) -> Any | None: ...

    def symbol_info(self, symbol: str) -> Any | None: ...

    def symbol_info_tick(self, symbol: str) -> Any | None: ...

    def symbol_select(self, symbol: str, enable: bool = True) -> bool: ...

    def positions_get(self, symbol: str | None = None, ticket: int | None = None) -> Any | None: ...

    def orders_get(self, symbol: str | None = None, ticket: int | None = None) -> Any | None: ...

    def order_check(self, request: dict[str, Any]) -> Any | None: ...

    def order_send(self, request: dict[str, Any]) -> Any | None: ...

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: int,
        start_pos: int,
        count: int,
    ) -> Any | None: ...

    def last_error(self) -> tuple[int, str]: ...


@dataclass
class MockTerminalInfo:
    connected: bool = True
    trade_allowed: bool = True
    name: str = "MetaTrader 5 Mock Terminal"
    path: str = "C:\\Program Files\\MetaTrader 5"


@dataclass
class MockAccountInfo:
    login: int = 12345678
    balance: float = 10000.0
    equity: float = 10000.0
    margin: float = 0.0
    margin_free: float = 10000.0
    margin_level: float = 0.0
    leverage: float = 100.0
    currency: str = "USD"
    trade_allowed: bool = True


@dataclass
class MockSymbolInfo:
    name: str = "XAUUSD"
    bid: float = 2650.00
    ask: float = 2650.25
    spread: float = 25.0
    point: float = 0.01
    trade_contract_size: float = 100.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    trade_stops_level: float = 100.0
    trade_freeze_level: float = 50.0
    trade_tick_size: float = 0.01
    trade_tick_value: float = 1.00
    filling_mode: int = SYMBOL_FILLING_FOK | SYMBOL_FILLING_IOC
    visible: bool = True
    select: bool = True


@dataclass
class MockTick:
    time: int = 0
    bid: float = 2650.00
    ask: float = 2650.25
    last: float = 2650.10
    volume: int = 10


@dataclass
class MockTradePosition:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    time: int | float | datetime = 0
    sl: float = 0.0
    tp: float = 0.0
    price_current: float = 0.0
    profit: float = 0.0
    commission: float = 0.0
    comment: str = ""
    magic: int = 0


@dataclass
class MockTradeOrder:
    ticket: int
    symbol: str
    type: int
    volume_initial: float
    volume_current: float
    price_open: float
    time_setup: int | float | datetime = 0
    sl: float = 0.0
    tp: float = 0.0
    state: int = ORDER_STATE_PLACED
    comment: str = ""
    magic: int = 0


@dataclass
class MockOrderCheckResult:
    retcode: int = 0
    balance: float = 10000.0
    equity: float = 10000.0
    profit: float = 0.0
    margin: float = 265.0
    margin_free: float = 9735.0
    margin_level: float = 3773.5
    comment: str = "Done"


@dataclass
class MockOrderSendResult:
    retcode: int = RETCODE_DONE
    deal: int = 1001
    order: int = 2001
    volume: float = 0.1
    price: float = 2650.25
    bid: float = 2650.00
    ask: float = 2650.25
    comment: str = "Request executed"
    request: dict[str, Any] = field(default_factory=dict)


class MT5MockClient:
    def __init__(
        self,
        initial_connected: bool = True,
        initial_balance: float = 10000.0,
        fail_initialize: bool = False,
    ) -> None:
        self.connected = initial_connected
        self.fail_initialize = fail_initialize
        self._last_error = (0, "Success")
        self._account = MockAccountInfo(balance=initial_balance, margin_free=initial_balance)
        self._terminal = MockTerminalInfo(connected=initial_connected)
        self._symbols: dict[str, MockSymbolInfo] = {
            "XAUUSD": MockSymbolInfo(name="XAUUSD"),
        }
        self._positions: dict[int, MockTradePosition] = {}
        self._orders: dict[int, MockTradeOrder] = {}
        self._rates: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._next_ticket = 1000
        self.calls: list[str] = []
        self.last_init_kwargs: dict[str, Any] = {}
        self.fail_order_send_retcode: int | None = None
        self.fail_order_send_none: bool = False
        self.fail_positions_get: bool = False
        self.fail_positions_get_once: bool = False
        self.fail_positions_retcode: int = RETCODE_CONNECTION
        self.fail_orders_get: bool = False
        self.fail_orders_get_once: bool = False
        self.fail_orders_retcode: int = RETCODE_CONNECTION

    def initialize(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout: int | None = None,
        portable: bool = False,
        **extra_kwargs: Any,
    ) -> bool:
        self.calls.append("initialize")
        self.last_init_kwargs = dict(extra_kwargs)
        if path is not None:
            self.last_init_kwargs["path"] = path
        if login is not None:
            self.last_init_kwargs["login"] = login
        if password is not None:
            self.last_init_kwargs["password"] = password
        if server is not None:
            self.last_init_kwargs["server"] = server
        if timeout is not None:
            self.last_init_kwargs["timeout"] = timeout
        if portable:
            self.last_init_kwargs["portable"] = portable

        if self.fail_initialize:
            self._last_error = (-1, "Terminal not found or IPC failed")
            self.connected = False
            self._terminal.connected = False
            return False
        self.connected = True
        self._terminal.connected = True
        if login is not None:
            self._account.login = login
        return True

    def shutdown(self) -> None:
        self.calls.append("shutdown")
        self.connected = False
        self._terminal.connected = False

    def terminal_info(self) -> MockTerminalInfo | None:
        self.calls.append("terminal_info")
        if not self.connected:
            return None
        return self._terminal

    def account_info(self) -> MockAccountInfo | None:
        self.calls.append("account_info")
        if not self.connected:
            return None
        return self._account

    def symbol_info(self, symbol: str) -> MockSymbolInfo | None:
        self.calls.append(f"symbol_info:{symbol}")
        if not self.connected:
            return None
        return self._symbols.get(symbol)

    def symbol_info_tick(self, symbol: str) -> MockTick | None:
        self.calls.append(f"symbol_info_tick:{symbol}")
        sym = self._symbols.get(symbol)
        if not sym or not self.connected:
            return None
        return MockTick(
            time=int(datetime.now(tz=UTC).timestamp()),
            bid=sym.bid,
            ask=sym.ask,
            last=(sym.bid + sym.ask) / 2.0,
        )

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        self.calls.append(f"symbol_select:{symbol}:{enable}")
        if symbol not in self._symbols:
            return False
        self._symbols[symbol].select = enable
        return True

    def copy_rates_from_pos(
        self,
        symbol: str,
        timeframe: int,
        start_pos: int,
        count: int,
    ) -> Any | None:
        self.calls.append(f"copy_rates_from_pos:{symbol}:{timeframe}:{start_pos}:{count}")
        if not self.connected:
            return None
        key = (symbol, timeframe)
        if key in self._rates:
            stored = self._rates[key]
            return stored[start_pos : start_pos + count]

        now_ts = int(datetime.now(tz=UTC).timestamp())
        step = 300
        if timeframe == 1:
            step = 60
        elif timeframe == 5:
            step = 300
        elif timeframe == 15:
            step = 900
        elif timeframe == 16385:
            step = 3600
        elif timeframe == 16388:
            step = 14400
        elif timeframe == 16408:
            step = 86400

        rates = []
        base_p = 2650.0
        for i in range(count, 0, -1):
            rates.append(
                {
                    "time": now_ts - (i * step),
                    "open": base_p,
                    "high": base_p + 1.0,
                    "low": base_p - 1.0,
                    "close": base_p + 0.2,
                    "tick_volume": 100,
                    "spread": 25,
                    "real_volume": 100,
                }
            )
        return rates

    def positions_get(
        self, symbol: str | None = None, ticket: int | None = None
    ) -> tuple[MockTradePosition, ...] | None:
        self.calls.append("positions_get")
        if self.fail_positions_get_once:
            self.fail_positions_get_once = False
            self._last_error = (
                self.fail_positions_retcode,
                "Simulated transient positions_get failure",
            )
            return None
        if self.fail_positions_get:
            self._last_error = (self.fail_positions_retcode, "Simulated positions_get failure")
            return None
        if not self.connected:
            self._last_error = (RETCODE_CONNECTION, "Terminal not connected")
            return None
        res: list[MockTradePosition] = []
        for pos in self._positions.values():
            if ticket is not None and pos.ticket != ticket:
                continue
            if symbol is not None and pos.symbol != symbol:
                continue
            res.append(pos)
        return tuple(res)

    def orders_get(
        self, symbol: str | None = None, ticket: int | None = None
    ) -> tuple[MockTradeOrder, ...] | None:
        self.calls.append("orders_get")
        if self.fail_orders_get_once:
            self.fail_orders_get_once = False
            self._last_error = (self.fail_orders_retcode, "Simulated transient orders_get failure")
            return None
        if self.fail_orders_get:
            self._last_error = (self.fail_orders_retcode, "Simulated orders_get failure")
            return None
        if not self.connected:
            self._last_error = (RETCODE_CONNECTION, "Terminal not connected")
            return None
        res: list[MockTradeOrder] = []
        for ord_obj in self._orders.values():
            if ticket is not None and ord_obj.ticket != ticket:
                continue
            if symbol is not None and ord_obj.symbol != symbol:
                continue
            res.append(ord_obj)
        return tuple(res)

    def order_check(self, request: dict[str, Any]) -> MockOrderCheckResult | None:
        self.calls.append("order_check")
        if not self.connected:
            return None
        volume = float(request.get("volume", 0.0))
        if volume <= 0:
            return MockOrderCheckResult(retcode=RETCODE_INVALID_VOLUME, comment="Invalid volume")

        symbol_name = str(request.get("symbol", ""))
        sym = self._symbols.get(symbol_name)
        if not sym:
            return MockOrderCheckResult(retcode=RETCODE_TRADE_DISABLED, comment="Symbol not found")

        required_margin = (volume * sym.trade_contract_size * sym.ask) / self._account.leverage
        if required_margin > self._account.margin_free:
            return MockOrderCheckResult(
                retcode=RETCODE_NO_MONEY,
                comment="Not enough money",
                margin=required_margin,
                margin_free=self._account.margin_free,
            )

        return MockOrderCheckResult(retcode=0, comment="Done", margin=required_margin)

    def order_send(self, request: dict[str, Any]) -> MockOrderSendResult | None:
        self.calls.append("order_send")
        if self.fail_order_send_none:
            code = self.fail_order_send_retcode or RETCODE_CONNECTION
            self._last_error = (code, "Simulated order_send None failure")
            return None
        if self.fail_order_send_retcode is not None:
            return MockOrderSendResult(
                retcode=self.fail_order_send_retcode,
                comment="Simulated order_send retcode failure",
                request=request,
            )
        if not self.connected:
            self._last_error = (RETCODE_CONNECTION, "Terminal not connected")
            return None

        action = int(request.get("action", 0))
        symbol_name = str(request.get("symbol", "XAUUSD"))
        sym = self._symbols.get(symbol_name, MockSymbolInfo(name=symbol_name))

        if action == TRADE_ACTION_DEAL:
            pos_ticket = request.get("position")
            if pos_ticket is not None and int(pos_ticket) in self._positions:
                pos = self._positions[int(pos_ticket)]
                req_volume = float(request.get("volume", pos.volume))
                if req_volume >= pos.volume:
                    del self._positions[pos.ticket]
                else:
                    pos.volume -= req_volume
                return MockOrderSendResult(
                    retcode=RETCODE_DONE,
                    deal=self._next_ticket + 1,
                    order=self._next_ticket,
                    volume=req_volume,
                    price=float(request.get("price", sym.bid)),
                    comment="Position closed",
                    request=request,
                )

            ord_type = int(request.get("type", ORDER_TYPE_BUY))
            pos_type = POSITION_TYPE_BUY if ord_type == ORDER_TYPE_BUY else POSITION_TYPE_SELL
            price = float(
                request.get("price", sym.ask if pos_type == POSITION_TYPE_BUY else sym.bid)
            )
            volume = float(request.get("volume", 0.1))

            self._next_ticket += 1
            ticket = self._next_ticket
            pos = MockTradePosition(
                ticket=ticket,
                symbol=symbol_name,
                type=pos_type,
                volume=volume,
                price_open=price,
                time=int(datetime.now(tz=UTC).timestamp()),
                sl=float(request.get("sl", 0.0) or 0.0),
                tp=float(request.get("tp", 0.0) or 0.0),
                price_current=price,
                comment=str(request.get("comment", "")),
                magic=int(request.get("magic", 0)),
            )
            self._positions[ticket] = pos
            return MockOrderSendResult(
                retcode=RETCODE_DONE,
                deal=ticket + 5000,
                order=ticket,
                volume=volume,
                price=price,
                comment="Request executed",
                request=request,
            )

        if action == TRADE_ACTION_PENDING:
            self._next_ticket += 1
            ticket = self._next_ticket
            ord_obj = MockTradeOrder(
                ticket=ticket,
                symbol=symbol_name,
                type=int(request.get("type", ORDER_TYPE_BUY_LIMIT)),
                volume_initial=float(request.get("volume", 0.1)),
                volume_current=float(request.get("volume", 0.1)),
                price_open=float(request.get("price", 2640.0)),
                time_setup=int(datetime.now(tz=UTC).timestamp()),
                sl=float(request.get("sl", 0.0) or 0.0),
                tp=float(request.get("tp", 0.0) or 0.0),
                state=ORDER_STATE_PLACED,
                comment=str(request.get("comment", "")),
                magic=int(request.get("magic", 0)),
            )
            self._orders[ticket] = ord_obj
            return MockOrderSendResult(
                retcode=RETCODE_PLACED,
                order=ticket,
                volume=ord_obj.volume_current,
                price=ord_obj.price_open,
                comment="Order placed",
                request=request,
            )

        if action == TRADE_ACTION_SLTP:
            ticket = int(request.get("position", 0))
            if ticket not in self._positions:
                return MockOrderSendResult(
                    retcode=RETCODE_INVALID,
                    comment="Position not found",
                    request=request,
                )
            pos = self._positions[ticket]
            pos.sl = float(request.get("sl", pos.sl) or 0.0)
            pos.tp = float(request.get("tp", pos.tp) or 0.0)
            return MockOrderSendResult(
                retcode=RETCODE_DONE,
                order=ticket,
                comment="Stops modified",
                request=request,
            )

        if action == TRADE_ACTION_MODIFY:
            ticket = int(request.get("order", 0))
            if ticket not in self._orders:
                return MockOrderSendResult(
                    retcode=RETCODE_INVALID,
                    comment="Order not found",
                    request=request,
                )
            ord_obj = self._orders[ticket]
            if "price" in request:
                ord_obj.price_open = float(request["price"])
            if "sl" in request:
                ord_obj.sl = float(request["sl"] or 0.0)
            if "tp" in request:
                ord_obj.tp = float(request["tp"] or 0.0)
            return MockOrderSendResult(
                retcode=RETCODE_DONE,
                order=ticket,
                comment="Order modified",
                request=request,
            )

        if action == TRADE_ACTION_REMOVE:
            ticket = int(request.get("order", 0))
            if ticket not in self._orders:
                return MockOrderSendResult(
                    retcode=RETCODE_INVALID,
                    comment="Order not found",
                    request=request,
                )
            del self._orders[ticket]
            return MockOrderSendResult(
                retcode=RETCODE_DONE,
                order=ticket,
                comment="Order removed",
                request=request,
            )

        return MockOrderSendResult(
            retcode=RETCODE_INVALID,
            comment=f"Unknown action {action}",
            request=request,
        )

    def last_error(self) -> tuple[int, str]:
        return self._last_error


def get_default_mt5_client() -> MT5ClientProtocol:
    try:
        mod = importlib.import_module("MetaTrader5")
        return cast(MT5ClientProtocol, mod)
    except ImportError as e:
        raise RuntimeError(
            "MetaTrader5 package is not installed or not available on this platform. "
            "Please install it with `pip install MetaTrader5` on Windows, "
            "or inject an MT5ClientProtocol mock."
        ) from e
