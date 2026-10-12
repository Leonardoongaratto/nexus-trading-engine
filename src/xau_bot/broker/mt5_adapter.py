import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderType,
    SymbolInfo,
)
from xau_bot.broker.mt5_client import MT5ClientProtocol, get_default_mt5_client
from xau_bot.broker.mt5_mapping import (
    ORDER_TYPE_BUY,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_BUY_STOP,
    ORDER_TYPE_SELL,
    POSITION_TYPE_BUY,
    RETCODE_CONNECTION,
    RETCODE_DONE,
    RETCODE_DONE_PARTIAL,
    RETCODE_ERROR,
    RETCODE_INVALID,
    RETCODE_INVALID_STOPS,
    RETCODE_INVALID_VOLUME,
    RETCODE_PLACED,
    TRADE_ACTION_DEAL,
    TRADE_ACTION_MODIFY,
    TRADE_ACTION_PENDING,
    TRADE_ACTION_REMOVE,
    TRADE_ACTION_SLTP,
    map_mt5_account_info,
    map_mt5_order,
    map_mt5_position,
    map_mt5_symbol_info,
    map_order_type_to_mt5,
    map_timeframe_to_mt5,
    resolve_filling_mode,
)
from xau_bot.broker.ports import BrokerPort
from xau_bot.config import Settings
from xau_bot.domain.models import Candle, Timeframe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Mt5AdapterConfig:
    path: str | None = None
    login: int | None = None
    password: str | None = field(default=None, repr=False)
    server: str | None = None
    timeout_ms: int = 60000
    portable: bool = False
    server_tz: str = "UTC"
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    reconnect_backoff_factor: float = 2.0
    reconnect_jitter: float = 0.1
    max_reconnect_attempts: int = 5

    @classmethod
    def from_settings(cls, settings: Settings) -> "Mt5AdapterConfig":
        login_val: int | None = None
        raw_login = getattr(settings, "mt5_login", None)
        if raw_login is not None:
            val_str = (
                raw_login.get_secret_value()
                if hasattr(raw_login, "get_secret_value")
                else str(raw_login)
            )
            if val_str:
                try:
                    login_val = int(val_str)
                except ValueError as e:
                    raise ValueError(
                        "Configuration error: mt5_login must be a valid integer"
                    ) from e

        pwd_val: str | None = None
        raw_pwd = getattr(settings, "mt5_password", None)
        if raw_pwd is not None:
            pwd_val = (
                raw_pwd.get_secret_value() if hasattr(raw_pwd, "get_secret_value") else str(raw_pwd)
            )

        timeout_raw = getattr(settings, "mt5_timeout_ms", 60000)
        return cls(
            path=getattr(settings, "mt5_path", None),
            login=login_val,
            password=pwd_val,
            server=getattr(settings, "mt5_server", None),
            timeout_ms=int(timeout_raw),
            server_tz=getattr(settings, "broker_server_tz", "UTC"),
        )


class Mt5BrokerAdapter(BrokerPort):
    def __init__(
        self,
        client: MT5ClientProtocol | None = None,
        config: Mt5AdapterConfig | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._client = client
        self._config = config or Mt5AdapterConfig()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5_worker")
        self._connected = False
        self._ever_connected = False
        self._intentional_disconnect = False
        self._lock = asyncio.Lock()
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._rng = rng or random.Random()

    def _get_client(self) -> MT5ClientProtocol:
        if self._client is None:
            self._client = get_default_mt5_client()
        return self._client

    async def _run_in_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(self._executor, lambda: func(*args, **kwargs))
        return await loop.run_in_executor(self._executor, func, *args)

    async def connect(self) -> bool:
        async with self._lock:
            self._intentional_disconnect = False
            return await self._connect_unlocked()

    async def _connect_unlocked(self) -> bool:
        client = self._get_client()
        cfg = self._config

        def _init_sync() -> tuple[bool, tuple[int, str]]:
            kwargs: dict[str, Any] = {"portable": cfg.portable, "timeout": cfg.timeout_ms}
            if cfg.path is not None:
                kwargs["path"] = cfg.path
            if cfg.login is not None:
                kwargs["login"] = cfg.login
            if cfg.password is not None:
                kwargs["password"] = cfg.password
            if cfg.server is not None:
                kwargs["server"] = cfg.server

            ok = client.initialize(**kwargs)
            err = client.last_error() if not ok else (0, "")
            return bool(ok), err

        res_init: tuple[bool, tuple[int, str]] = await self._run_in_thread(_init_sync)
        success, (err_code, err_msg) = res_init
        self._connected = bool(success)
        if success:
            self._ever_connected = True

        if success:
            logger.info(
                "Connected to MetaTrader 5 terminal successfully",
                extra={"event": "mt5_connected", "server_configured": cfg.server is not None},
            )
        else:
            logger.error(
                "Failed to connect to MetaTrader 5 terminal",
                extra={
                    "event": "mt5_connect_failed",
                    "error_code": err_code,
                    "error_message": err_msg,
                },
            )
        return self._connected

    async def disconnect(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._run_in_thread(self._client.shutdown)
            self._connected = False
            self._intentional_disconnect = True
            logger.info("Disconnected from MetaTrader 5", extra={"event": "mt5_disconnected"})

    def is_connected(self) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            future = self._executor.submit(self._client.terminal_info)
            info = future.result(timeout=2.0)
            conn = bool(info and getattr(info, "connected", False))
            if not conn:
                self._connected = False
            return conn
        except TimeoutError:
            return self._connected
        except Exception:
            self._connected = False
            return False

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    async def aclose(self) -> None:
        await self.disconnect()
        self.close()

    async def reconnect(self) -> bool:
        async with self._lock:
            self._intentional_disconnect = False
            return await self._reconnect_unlocked()

    async def _reconnect_unlocked(self) -> bool:
        cfg = self._config
        delay = cfg.reconnect_initial_delay

        for attempt in range(1, cfg.max_reconnect_attempts + 1):
            logger.warning(
                "Attempting to reconnect to MetaTrader 5",
                extra={
                    "event": "mt5_reconnect_attempt",
                    "attempt": attempt,
                    "max": cfg.max_reconnect_attempts,
                },
            )
            success = await self._connect_unlocked()
            if success:
                logger.info(
                    "Reconnected to MetaTrader 5 successfully",
                    extra={"event": "mt5_reconnect_success", "attempt": attempt},
                )
                return True

            if attempt < cfg.max_reconnect_attempts:
                jitter_val = self._rng.uniform(-cfg.reconnect_jitter, cfg.reconnect_jitter)
                sleep_time = min(cfg.reconnect_max_delay, delay * (1.0 + jitter_val))
                await self._sleep_fn(sleep_time)
                delay = min(cfg.reconnect_max_delay, delay * cfg.reconnect_backoff_factor)

        logger.error(
            "Exceeded maximum reconnect attempts to MetaTrader 5",
            extra={"event": "mt5_reconnect_exhausted", "attempts": cfg.max_reconnect_attempts},
        )
        return False

    async def _ensure_connected(self) -> None:
        async with self._lock:
            await self._ensure_connected_unlocked()

    async def _ensure_connected_unlocked(self) -> None:
        if self._intentional_disconnect:
            raise RuntimeError("Broker is explicitly disconnected; automatic reconnection disabled")
        if not self._ever_connected:
            raise RuntimeError("Broker has never been connected; call connect() first")
        if not self._connected:
            logger.info("MT5 connection lost; triggering automatic reconnection")
            ok = await self._reconnect_unlocked()
            if not ok:
                raise RuntimeError("MT5 terminal disconnected and automatic reconnection failed")
            return

        client = self._get_client()

        def _check_terminal() -> bool:
            info = client.terminal_info()
            return bool(info and getattr(info, "connected", False))

        is_conn = await self._run_in_thread(_check_terminal)
        if not is_conn:
            self._connected = False
            logger.warning(
                "MT5 terminal disconnected; triggering automatic reconnection",
                extra={"event": "mt5_connection_lost"},
            )
            ok = await self._reconnect_unlocked()
            if not ok:
                raise RuntimeError("MT5 terminal disconnected and automatic reconnection failed")

    async def get_account_info(self) -> AccountInfo:
        await self._ensure_connected()
        client = self._get_client()

        def _get_acc() -> tuple[Any, tuple[int, str]]:
            acc = client.account_info()
            err = client.last_error() if acc is None else (0, "")
            return acc, err

        info, (err_code, err_msg) = await self._run_in_thread(_get_acc)
        if info is None and (err_code == RETCODE_CONNECTION or not self.is_connected()):
            self._connected = False
            await self._ensure_connected()
            info, (err_code, err_msg) = await self._run_in_thread(_get_acc)
        if info is None:
            raise RuntimeError("Failed to retrieve account info from MT5 terminal")
        return map_mt5_account_info(info)

    async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        await self._ensure_connected()
        client = self._get_client()

        def _get_sym() -> tuple[Any, tuple[int, str]]:
            client.symbol_select(symbol, True)
            sym = client.symbol_info(symbol)
            err = client.last_error() if sym is None else (0, "")
            return sym, err

        info, (err_code, err_msg) = await self._run_in_thread(_get_sym)
        if info is None and (err_code == RETCODE_CONNECTION or not self.is_connected()):
            self._connected = False
            await self._ensure_connected()
            info, (err_code, err_msg) = await self._run_in_thread(_get_sym)
        if info is None:
            return None
        return map_mt5_symbol_info(info)

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> tuple[Candle, ...]:
        await self._ensure_connected()
        client = self._get_client()

        def _fetch() -> tuple[Candle, ...]:
            client.symbol_select(symbol, True)
            mt5_tf = map_timeframe_to_mt5(timeframe)
            rates = client.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is None or len(rates) == 0:
                return ()
            candles: list[Candle] = []
            for r in rates:
                if isinstance(r, dict):
                    t_val = r["time"]
                    o_val = float(r["open"])
                    h_val = float(r["high"])
                    l_val = float(r["low"])
                    c_val = float(r["close"])
                    raw_v = (
                        r.get("tick_volume")
                        if r.get("tick_volume") is not None
                        else r.get("real_volume")
                    )
                    v_val = float(raw_v) if raw_v is not None else 0.0
                elif hasattr(r, "__getitem__"):
                    t_val = r["time"]
                    o_val = float(r["open"])
                    h_val = float(r["high"])
                    l_val = float(r["low"])
                    c_val = float(r["close"])
                    try:
                        raw_v = r["tick_volume"]
                    except (KeyError, IndexError):
                        try:
                            raw_v = r["real_volume"]
                        except (KeyError, IndexError):
                            raw_v = 0.0
                    v_val = float(raw_v) if raw_v is not None else 0.0
                else:
                    t_val = r.time
                    o_val = float(r.open)
                    h_val = float(r.high)
                    l_val = float(r.low)
                    c_val = float(r.close)
                    raw_v = getattr(r, "tick_volume", getattr(r, "real_volume", 0.0))
                    v_val = float(raw_v) if raw_v is not None else 0.0

                tz = ZoneInfo(self._config.server_tz) if self._config.server_tz else UTC
                if isinstance(t_val, datetime):
                    dt = (
                        t_val
                        if t_val.tzinfo is not None
                        else t_val.replace(tzinfo=tz).astimezone(UTC)
                    )
                else:
                    try:
                        numeric_t = float(t_val)
                        dt = (
                            datetime.fromtimestamp(numeric_t, tz=UTC)
                            .replace(tzinfo=tz)
                            .astimezone(UTC)
                        )
                    except (TypeError, ValueError) as err:
                        raise ValueError(f"Invalid timestamp value for candle: {t_val!r}") from err

                candles.append(
                    Candle(
                        time=dt,
                        open=o_val,
                        high=h_val,
                        low=l_val,
                        close=c_val,
                        volume=v_val,
                        timeframe=timeframe,
                    )
                )
            return tuple(candles)

        res: tuple[Candle, ...] = await self._run_in_thread(_fetch)
        return res

    def _validate_stops(
        self,
        is_buy: bool,
        ref_price: float,
        sl: float | None,
        tp: float | None,
        stops_level_points: float,
        freeze_level_points: float = 0.0,
    ) -> bool:
        if is_buy:
            if sl is not None and sl > 0.0:
                if sl >= ref_price:
                    return False
                if (ref_price - sl) < stops_level_points:
                    return False
                if freeze_level_points > 0 and (ref_price - sl) < freeze_level_points:
                    return False
            if tp is not None and tp > 0.0:
                if tp <= ref_price:
                    return False
                if (tp - ref_price) < stops_level_points:
                    return False
                if freeze_level_points > 0 and (tp - ref_price) < freeze_level_points:
                    return False
        else:
            if sl is not None and sl > 0.0:
                if sl <= ref_price:
                    return False
                if (sl - ref_price) < stops_level_points:
                    return False
                if freeze_level_points > 0 and (sl - ref_price) < freeze_level_points:
                    return False
            if tp is not None and tp > 0.0:
                if tp >= ref_price:
                    return False
                if (ref_price - tp) < stops_level_points:
                    return False
                if freeze_level_points > 0 and (ref_price - tp) < freeze_level_points:
                    return False
        return True

    async def send_order(self, request: OrderRequest) -> OrderResult:
        async with self._lock:
            try:
                await self._ensure_connected_unlocked()
            except RuntimeError as err:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_CONNECTION,
                    error_message=str(err),
                    retcode=RETCODE_CONNECTION,
                )
            client = self._get_client()

            def _query_symbol_sync() -> tuple[SymbolInfo | None, int]:
                client.symbol_select(request.symbol, True)
                raw_sym = client.symbol_info(request.symbol)
                if raw_sym is None:
                    return None, 0
                dto = map_mt5_symbol_info(raw_sym)
                filling = resolve_filling_mode(getattr(raw_sym, "filling_mode", 0))
                return dto, filling

            sym_dto, resolved_filling = await self._run_in_thread(_query_symbol_sync)
            if sym_dto is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Symbol {request.symbol} not found",
                    retcode=RETCODE_INVALID,
                )

            if request.volume < sym_dto.min_volume or request.volume > sym_dto.max_volume:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID_VOLUME,
                    error_message=f"Volume {request.volume} out of bounds",
                    retcode=RETCODE_INVALID_VOLUME,
                )

            mt5_type = map_order_type_to_mt5(request.order_type)
            is_market = request.order_type in (OrderType.BUY, OrderType.SELL)
            is_buy = request.order_type in (OrderType.BUY, OrderType.BUY_LIMIT, OrderType.BUY_STOP)

            action = TRADE_ACTION_DEAL if is_market else TRADE_ACTION_PENDING

            if is_market:
                ref_price = sym_dto.ask if is_buy else sym_dto.bid
                order_price = request.price if request.price is not None else ref_price
            else:
                if request.price is None or request.price <= 0:
                    return OrderResult(
                        success=False,
                        error_code=RETCODE_INVALID,
                        error_message="Pending orders require a valid positive price",
                        retcode=RETCODE_INVALID,
                    )
                order_price = request.price
                ref_price = order_price

            if request.stop_loss is not None or request.take_profit is not None:
                valid_stops = self._validate_stops(
                    is_buy=is_buy,
                    ref_price=ref_price,
                    sl=request.stop_loss,
                    tp=request.take_profit,
                    stops_level_points=sym_dto.stops_level_points,
                )
                if not valid_stops:
                    return OrderResult(
                        success=False,
                        error_code=RETCODE_INVALID_STOPS,
                        error_message="Stops violate stops_level_points or reference price",
                        retcode=RETCODE_INVALID_STOPS,
                    )

            mt5_req: dict[str, Any] = {
                "action": action,
                "symbol": request.symbol,
                "volume": request.volume,
                "type": mt5_type,
                "price": order_price,
                "type_filling": resolved_filling,
                "type_time": 0,
            }
            if request.stop_loss is not None:
                mt5_req["sl"] = request.stop_loss
            if request.take_profit is not None:
                mt5_req["tp"] = request.take_profit
            if request.comment:
                mt5_req["comment"] = request.comment
            if request.magic:
                mt5_req["magic"] = request.magic

            def _check_and_send_sync() -> tuple[str, Any, tuple[int, str]]:
                check = client.order_check(mt5_req)
                if check is None:
                    err = client.last_error()
                    return "check_none", None, err

                check_ret = getattr(check, "retcode", 0)
                if check_ret != 0 and check_ret != RETCODE_DONE:
                    return (
                        "check_failed",
                        check,
                        (check_ret, str(getattr(check, "comment", "order_check failed"))),
                    )

                send = client.order_send(mt5_req)
                if send is None:
                    err = client.last_error()
                    return "send_none", None, err
                return "sent", send, (0, "")

            status, res_obj, (err_code, err_msg) = await self._run_in_thread(_check_and_send_sync)

            if status == "check_none":
                if err_code == RETCODE_CONNECTION:
                    self._connected = False
                    await self._reconnect_unlocked()
                return OrderResult(
                    success=False,
                    error_code=err_code or RETCODE_ERROR,
                    error_message=err_msg or "order_check returned None",
                    retcode=RETCODE_CONNECTION if err_code == RETCODE_CONNECTION else RETCODE_ERROR,
                )

            if status == "check_failed":
                return OrderResult(
                    success=False,
                    error_code=err_code,
                    error_message=err_msg,
                    retcode=err_code,
                )

            if status == "send_none":
                if err_code == RETCODE_CONNECTION:
                    self._connected = False
                    await self._reconnect_unlocked()
                return OrderResult(
                    success=False,
                    error_code=err_code or RETCODE_ERROR,
                    error_message=err_msg or "order_send returned None",
                    retcode=RETCODE_CONNECTION if err_code == RETCODE_CONNECTION else RETCODE_ERROR,
                )

            retcode = int(getattr(res_obj, "retcode", RETCODE_ERROR))
            if retcode == RETCODE_CONNECTION:
                self._connected = False
                await self._reconnect_unlocked()
            success = retcode in (RETCODE_DONE, RETCODE_PLACED, RETCODE_DONE_PARTIAL)
            ticket_val = int(getattr(res_obj, "order", 0) or getattr(res_obj, "deal", 0))

            return OrderResult(
                success=success,
                ticket=ticket_val if ticket_val > 0 else None,
                price=float(getattr(res_obj, "price", order_price)),
                volume=float(getattr(res_obj, "volume", request.volume)),
                error_code=0 if success else retcode,
                error_message=str(getattr(res_obj, "comment", "")),
                retcode=retcode,
            )

    async def modify_order(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        async with self._lock:
            try:
                await self._ensure_connected_unlocked()
            except RuntimeError as err:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_CONNECTION,
                    error_message=str(err),
                    retcode=RETCODE_CONNECTION,
                )
            client = self._get_client()

            def _find_target() -> tuple[Any | None, SymbolInfo | None, bool]:
                positions = client.positions_get(ticket=ticket)
                target_obj: Any | None = None
                is_pos = False
                if positions and len(positions) > 0:
                    target_obj = positions[0]
                    is_pos = True
                else:
                    orders = client.orders_get(ticket=ticket)
                    if orders and len(orders) > 0:
                        target_obj = orders[0]
                        is_pos = False

                if target_obj is None:
                    return None, None, False

                s_name = str(getattr(target_obj, "symbol", ""))
                client.symbol_select(s_name, True)
                raw_s = client.symbol_info(s_name)
                dto = map_mt5_symbol_info(raw_s) if raw_s else None
                return target_obj, dto, is_pos

            target, sym_dto, is_position = await self._run_in_thread(_find_target)
            if target is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Ticket {ticket} not found in positions or orders",
                    retcode=RETCODE_INVALID,
                )

            sym_name = str(getattr(target, "symbol", ""))
            if sym_dto is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Symbol {sym_name} not found",
                    retcode=RETCODE_INVALID,
                )

            if is_position:
                pos_type = int(getattr(target, "type", 0))
                is_buy = pos_type == POSITION_TYPE_BUY
                ref_price = sym_dto.bid if is_buy else sym_dto.ask

                if stop_loss is not None or take_profit is not None:
                    valid_stops = self._validate_stops(
                        is_buy=is_buy,
                        ref_price=ref_price,
                        sl=stop_loss,
                        tp=take_profit,
                        stops_level_points=sym_dto.stops_level_points,
                        freeze_level_points=sym_dto.freeze_level_points,
                    )
                    if not valid_stops:
                        return OrderResult(
                            success=False,
                            error_code=RETCODE_INVALID_STOPS,
                            error_message="Stops violate stops_level_points or freeze_level_points",
                            retcode=RETCODE_INVALID_STOPS,
                        )

                mt5_req: dict[str, Any] = {
                    "action": TRADE_ACTION_SLTP,
                    "position": ticket,
                    "symbol": sym_name,
                    "sl": (
                        stop_loss
                        if stop_loss is not None
                        else float(getattr(target, "sl", 0.0) or 0.0)
                    ),
                    "tp": (
                        take_profit
                        if take_profit is not None
                        else float(getattr(target, "tp", 0.0) or 0.0)
                    ),
                }
            else:
                ord_type = int(getattr(target, "type", 0))
                is_buy = ord_type in (ORDER_TYPE_BUY, ORDER_TYPE_BUY_LIMIT, ORDER_TYPE_BUY_STOP)
                ref_price = float(getattr(target, "price_open", 0.0))

                if stop_loss is not None or take_profit is not None:
                    valid_stops = self._validate_stops(
                        is_buy=is_buy,
                        ref_price=ref_price,
                        sl=stop_loss,
                        tp=take_profit,
                        stops_level_points=sym_dto.stops_level_points,
                        freeze_level_points=sym_dto.freeze_level_points,
                    )
                    if not valid_stops:
                        return OrderResult(
                            success=False,
                            error_code=RETCODE_INVALID_STOPS,
                            error_message="Stops violate stops_level_points or freeze_level_points",
                            retcode=RETCODE_INVALID_STOPS,
                        )

                mt5_req = {
                    "action": TRADE_ACTION_MODIFY,
                    "order": ticket,
                    "price": ref_price,
                    "sl": (
                        stop_loss
                        if stop_loss is not None
                        else float(getattr(target, "sl", 0.0) or 0.0)
                    ),
                    "tp": (
                        take_profit
                        if take_profit is not None
                        else float(getattr(target, "tp", 0.0) or 0.0)
                    ),
                    "type_time": 0,
                }

            def _send_sync() -> tuple[Any, tuple[int, str]]:
                send = client.order_send(mt5_req)
                err = client.last_error() if send is None else (0, "")
                return send, err

            send_res, (err_code, err_msg) = await self._run_in_thread(_send_sync)
            if send_res is None:
                if err_code == RETCODE_CONNECTION:
                    self._connected = False
                    await self._reconnect_unlocked()
                return OrderResult(
                    success=False,
                    error_code=err_code or RETCODE_ERROR,
                    error_message=err_msg or "order_send returned None",
                    retcode=RETCODE_CONNECTION if err_code == RETCODE_CONNECTION else RETCODE_ERROR,
                )

            retcode = int(getattr(send_res, "retcode", RETCODE_ERROR))
            if retcode == RETCODE_CONNECTION:
                self._connected = False
                await self._reconnect_unlocked()
            success = retcode in (RETCODE_DONE, RETCODE_PLACED)
            return OrderResult(
                success=success,
                ticket=ticket,
                error_code=0 if success else retcode,
                error_message=str(getattr(send_res, "comment", "")),
                retcode=retcode,
            )

    async def cancel_order(self, ticket: int) -> OrderResult:
        async with self._lock:
            try:
                await self._ensure_connected_unlocked()
            except RuntimeError as err:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_CONNECTION,
                    error_message=str(err),
                    retcode=RETCODE_CONNECTION,
                )
            client = self._get_client()

            def _find_order() -> Any | None:
                orders = client.orders_get(ticket=ticket)
                return orders[0] if orders and len(orders) > 0 else None

            ord_obj = await self._run_in_thread(_find_order)
            if ord_obj is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Order {ticket} not found",
                    retcode=RETCODE_INVALID,
                )

            mt5_req = {
                "action": TRADE_ACTION_REMOVE,
                "order": ticket,
            }

            def _send_sync() -> tuple[Any, tuple[int, str]]:
                send = client.order_send(mt5_req)
                err = client.last_error() if send is None else (0, "")
                return send, err

            send_res, (err_code, err_msg) = await self._run_in_thread(_send_sync)
            if send_res is None:
                if err_code == RETCODE_CONNECTION:
                    self._connected = False
                    await self._reconnect_unlocked()
                return OrderResult(
                    success=False,
                    error_code=err_code or RETCODE_ERROR,
                    error_message=err_msg or "order_send returned None",
                    retcode=RETCODE_CONNECTION if err_code == RETCODE_CONNECTION else RETCODE_ERROR,
                )

            retcode = int(getattr(send_res, "retcode", RETCODE_ERROR))
            if retcode == RETCODE_CONNECTION:
                self._connected = False
                await self._reconnect_unlocked()
            success = retcode == RETCODE_DONE
            return OrderResult(
                success=success,
                ticket=ticket,
                error_code=0 if success else retcode,
                error_message=str(getattr(send_res, "comment", "")),
                retcode=retcode,
            )

    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult:
        async with self._lock:
            try:
                await self._ensure_connected_unlocked()
            except RuntimeError as err:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_CONNECTION,
                    error_message=str(err),
                    retcode=RETCODE_CONNECTION,
                )
            client = self._get_client()

            def _query_pos_sync() -> tuple[Any | None, SymbolInfo | None, int]:
                positions = client.positions_get(ticket=ticket)
                if not positions or len(positions) == 0:
                    return None, None, 0
                target_pos = positions[0]
                sym_name = str(getattr(target_pos, "symbol", ""))
                client.symbol_select(sym_name, True)
                raw_sym = client.symbol_info(sym_name)
                if raw_sym is None:
                    return target_pos, None, 0
                dto = map_mt5_symbol_info(raw_sym)
                filling = resolve_filling_mode(getattr(raw_sym, "filling_mode", 0))
                return target_pos, dto, filling

            pos, sym_dto, resolved_filling = await self._run_in_thread(_query_pos_sync)
            if pos is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Position {ticket} not found",
                    retcode=RETCODE_INVALID,
                )

            sym_name = str(getattr(pos, "symbol", ""))
            if sym_dto is None:
                return OrderResult(
                    success=False,
                    error_code=RETCODE_INVALID,
                    error_message=f"Symbol {sym_name} not found",
                    retcode=RETCODE_INVALID,
                )

            pos_vol = float(getattr(pos, "volume", 0.0))
            vol_to_close = min(volume, pos_vol) if volume is not None and volume > 0 else pos_vol

            pos_type = int(getattr(pos, "type", 0))
            is_buy = pos_type == POSITION_TYPE_BUY
            close_type = ORDER_TYPE_SELL if is_buy else ORDER_TYPE_BUY
            price = sym_dto.bid if is_buy else sym_dto.ask

            mt5_req = {
                "action": TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": sym_name,
                "volume": vol_to_close,
                "type": close_type,
                "price": price,
                "type_filling": resolved_filling,
                "type_time": 0,
            }

            def _send_sync() -> tuple[Any, tuple[int, str]]:
                send = client.order_send(mt5_req)
                err = client.last_error() if send is None else (0, "")
                return send, err

            send_res, (err_code, err_msg) = await self._run_in_thread(_send_sync)
            if send_res is None:
                if err_code == RETCODE_CONNECTION:
                    self._connected = False
                    await self._reconnect_unlocked()
                return OrderResult(
                    success=False,
                    error_code=err_code or RETCODE_ERROR,
                    error_message=err_msg or "order_send returned None",
                    retcode=RETCODE_CONNECTION if err_code == RETCODE_CONNECTION else RETCODE_ERROR,
                )

            retcode = int(getattr(send_res, "retcode", RETCODE_ERROR))
            if retcode == RETCODE_CONNECTION:
                self._connected = False
                await self._reconnect_unlocked()
            success = retcode in (RETCODE_DONE, RETCODE_DONE_PARTIAL)
            return OrderResult(
                success=success,
                ticket=ticket,
                price=float(getattr(send_res, "price", price)),
                volume=float(getattr(send_res, "volume", vol_to_close)),
                error_code=0 if success else retcode,
                error_message=str(getattr(send_res, "comment", "")),
                retcode=retcode,
            )

    async def get_open_positions(self) -> tuple[BrokerPosition, ...]:
        await self._ensure_connected()
        client = self._get_client()

        def _get_pos() -> tuple[Any, tuple[int, str]]:
            positions = client.positions_get()
            err = client.last_error() if positions is None else (0, "")
            return positions, err

        raw_positions, (err_code, err_msg) = await self._run_in_thread(_get_pos)
        if raw_positions is None and (err_code == RETCODE_CONNECTION or not self.is_connected()):
            self._connected = False
            await self._ensure_connected()
            raw_positions, (err_code, err_msg) = await self._run_in_thread(_get_pos)

        if raw_positions is None:
            raise RuntimeError(f"Failed to retrieve open positions from MT5: {err_msg or err_code}")

        return tuple(map_mt5_position(p, server_tz=self._config.server_tz) for p in raw_positions)

    async def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        await self._ensure_connected()
        client = self._get_client()

        def _get_ord() -> tuple[Any, tuple[int, str]]:
            orders = client.orders_get()
            err = client.last_error() if orders is None else (0, "")
            return orders, err

        raw_orders, (err_code, err_msg) = await self._run_in_thread(_get_ord)
        if raw_orders is None and (err_code == RETCODE_CONNECTION or not self.is_connected()):
            self._connected = False
            await self._ensure_connected()
            raw_orders, (err_code, err_msg) = await self._run_in_thread(_get_ord)

        if raw_orders is None:
            raise RuntimeError(f"Failed to retrieve open orders from MT5: {err_msg or err_code}")

        return tuple(map_mt5_order(o, server_tz=self._config.server_tz) for o in raw_orders)
