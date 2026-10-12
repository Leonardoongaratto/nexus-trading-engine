import asyncio
import logging
import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderStatus,
    OrderType,
    SymbolInfo,
)
from xau_bot.broker.ports import BrokerPort
from xau_bot.domain.models import Candle, Direction, Tick, Timeframe

logger = logging.getLogger(__name__)


def _calc_gross_pnl(
    direction: Direction,
    open_price: float,
    close_price: float,
    volume: float,
    contract_size: float,
) -> float:
    if direction == Direction.BULLISH:
        return (close_price - open_price) * volume * contract_size
    return (open_price - close_price) * volume * contract_size


def _validate_stops(
    direction: Direction,
    ref_sl_price: float,
    ref_tp_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    spec: SymbolInfo,
) -> OrderResult | None:
    min_dist = spec.stops_level_points * spec.point_size
    if stop_loss is not None:
        if stop_loss <= 0.0:
            return OrderResult(
                success=False,
                error_code=7,
                error_message="Stop Loss inválido: valor deve ser positivo",
                retcode=10016,
            )
        if direction == Direction.BULLISH and stop_loss > (ref_sl_price - min_dist + 1e-9):
            return OrderResult(
                success=False,
                error_code=7,
                error_message=f"Stop Loss inválido ou viola stops_level ({min_dist:.2f}) para BUY",
                retcode=10016,
            )
        if direction == Direction.BEARISH and stop_loss < (ref_sl_price + min_dist - 1e-9):
            return OrderResult(
                success=False,
                error_code=7,
                error_message=f"Stop Loss inválido ou viola stops_level ({min_dist:.2f}) para SELL",
                retcode=10016,
            )

    if take_profit is not None and take_profit != 0.0:
        if take_profit < 0.0:
            return OrderResult(
                success=False,
                error_code=7,
                error_message="Take Profit inválido: valor deve ser positivo",
                retcode=10016,
            )
        if direction == Direction.BULLISH and take_profit < (ref_tp_price + min_dist - 1e-9):
            return OrderResult(
                success=False,
                error_code=7,
                error_message=(
                    f"Take Profit inválido ou viola stops_level ({min_dist:.2f}) para BUY"
                ),
                retcode=10016,
            )
        if direction == Direction.BEARISH and take_profit > (ref_tp_price - min_dist + 1e-9):
            return OrderResult(
                success=False,
                error_code=7,
                error_message=(
                    f"Take Profit inválido ou viola stops_level ({min_dist:.2f}) para SELL"
                ),
                retcode=10016,
            )
    return None


class SimBrokerAdapter(BrokerPort):
    def __init__(
        self,
        initial_balance: float = 10000.0,
        leverage: float = 100.0,
        currency: str = "USD",
        commission_per_lot: float = 7.0,
        latency_ms: float = 0.0,
        rejection_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.balance: float = float(initial_balance)
        self.leverage: float = float(leverage)
        self.currency: str = currency
        self.commission_per_lot: float = float(commission_per_lot)
        self.latency_ms: float = max(0.0, float(latency_ms))
        self.rejection_rate: float = max(0.0, min(1.0, float(rejection_rate)))
        self._rng: random.Random = random.Random(seed)

        self._connected: bool = False
        self._next_ticket: int = 1000
        self._positions: dict[int, BrokerPosition] = {}
        self._orders: dict[int, BrokerOrder] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

        self._symbol_specs: dict[str, SymbolInfo] = {
            "XAUUSD": SymbolInfo(
                name="XAUUSD",
                bid=2650.0,
                ask=2650.25,
                spread_points=0.25,
                point_size=0.01,
                contract_size=100.0,
                min_volume=0.01,
                max_volume=100.0,
                volume_step=0.01,
                stops_level_points=1.0,
                freeze_level_points=0.5,
                tick_size=0.01,
                tick_value=1.0,
            )
        }
        self._candles: dict[tuple[str, Timeframe], list[Candle]] = {}
        self._custom_candles: set[tuple[str, Timeframe]] = set()

    def set_candles(
        self,
        timeframe: Timeframe,
        candles: Sequence[Candle],
        symbol: str = "XAUUSD",
    ) -> None:
        self._custom_candles.add((symbol, timeframe))
        self._candles[(symbol, timeframe)] = list(candles)

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> tuple[Candle, ...]:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        async with self._lock:
            key = (symbol, timeframe)
            if key in self._custom_candles:
                return tuple(self._candles.get(key, [])[-count:])

            now = datetime.now(UTC)
            tf_secs = max(1, int(timeframe.duration_seconds))
            now_ts = int(now.timestamp())
            aligned_ts = now_ts - (now_ts % tf_secs)
            aligned_now = datetime.fromtimestamp(aligned_ts, tz=UTC)
            duration = timedelta(seconds=tf_secs)

            existing = self._candles.get(key)
            if not existing:
                candles: list[Candle] = []
                base_price = 2650.0
                for i in range(count, -1, -1):
                    c_time = aligned_now - duration * i
                    candles.append(
                        Candle(
                            time=c_time,
                            open=base_price,
                            high=base_price + 1.0,
                            low=base_price - 1.0,
                            close=base_price + 0.2,
                            volume=100.0,
                            timeframe=timeframe,
                        )
                    )
                self._candles[key] = candles
                return tuple(candles[-count:])

            last_c_time = existing[-1].time
            if aligned_now > last_c_time:
                diff_bars = int((aligned_now - last_c_time).total_seconds() // tf_secs)
                base_price = 2650.0
                for step in range(1, diff_bars + 1):
                    new_time = last_c_time + duration * step
                    existing.append(
                        Candle(
                            time=new_time,
                            open=base_price,
                            high=base_price + 1.0,
                            low=base_price - 1.0,
                            close=base_price + 0.2,
                            volume=100.0,
                            timeframe=timeframe,
                        )
                    )
                if len(existing) > 500:
                    del existing[: len(existing) - 500]

            return tuple(existing[-count:])

    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        self._connected = True
        logger.info(
            "event=connected balance=%.2f currency=%s",
            self.balance,
            self.currency,
            extra={"event": "connected", "balance": self.balance},
        )
        return True

    async def disconnect(self) -> None:
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        self._connected = False
        logger.info("event=disconnected", extra={"event": "disconnected"})

    def set_fault_injection(self, rejection_rate: float = 0.0, latency_ms: float = 0.0) -> None:
        self.rejection_rate = max(0.0, min(1.0, rejection_rate))
        self.latency_ms = max(0.0, latency_ms)

    def on_tick(self, tick: Tick, symbol: str = "XAUUSD") -> None:
        spec = self._symbol_specs.get(symbol)
        if spec is not None:
            self._symbol_specs[symbol] = SymbolInfo(
                name=spec.name,
                bid=tick.bid,
                ask=tick.ask,
                spread_points=round(tick.ask - tick.bid, 4),
                point_size=spec.point_size,
                contract_size=spec.contract_size,
                min_volume=spec.min_volume,
                max_volume=spec.max_volume,
                volume_step=spec.volume_step,
                stops_level_points=spec.stops_level_points,
                freeze_level_points=spec.freeze_level_points,
                tick_size=spec.tick_size,
                tick_value=spec.tick_value,
            )

        contract_size = spec.contract_size if spec else 100.0

        filled_order_tickets: list[int] = []
        for ticket, order in list(self._orders.items()):
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            fill = False
            fill_price = order.price
            direction = Direction.BULLISH
            if order.order_type == OrderType.BUY_LIMIT and tick.ask <= order.price:
                fill = True
                fill_price = order.price
                direction = Direction.BULLISH
            elif order.order_type == OrderType.SELL_LIMIT and tick.bid >= order.price:
                fill = True
                fill_price = order.price
                direction = Direction.BEARISH

            if fill:
                filled_order_tickets.append(ticket)
                commission = order.volume * self.commission_per_lot
                self.balance -= commission
                curr_exit = tick.bid if direction == Direction.BULLISH else tick.ask
                init_floating = _calc_gross_pnl(
                    direction, fill_price, curr_exit, order.volume, contract_size
                )

                self._positions[ticket] = BrokerPosition(
                    ticket=ticket,
                    symbol=order.symbol,
                    direction=direction,
                    volume=order.volume,
                    open_price=fill_price,
                    open_time=tick.time,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    current_price=curr_exit,
                    floating_pnl=init_floating,
                    commission=commission,
                    comment=order.comment,
                    magic=order.magic,
                )
                logger.info(
                    "event=pending_order_filled ticket=%d symbol=%s price=%.2f",
                    ticket,
                    order.symbol,
                    fill_price,
                    extra={"event": "pending_order_filled", "ticket": ticket, "price": fill_price},
                )

        for ticket in filled_order_tickets:
            self._orders.pop(ticket, None)

        closed_tickets: list[int] = []
        for ticket, pos in list(self._positions.items()):
            if pos.symbol != symbol:
                continue

            if pos.direction == Direction.BULLISH:
                curr_price = tick.bid
                floating_pnl = _calc_gross_pnl(
                    pos.direction, pos.open_price, curr_price, pos.volume, contract_size
                )
                sl_hit = pos.stop_loss is not None and tick.bid <= pos.stop_loss
                tp_hit = pos.take_profit is not None and tick.bid >= pos.take_profit
                close_price = (
                    min(pos.stop_loss, tick.bid)
                    if (sl_hit and pos.stop_loss is not None)
                    else (pos.take_profit if tp_hit and pos.take_profit is not None else 0.0)
                )
            else:
                curr_price = tick.ask
                floating_pnl = _calc_gross_pnl(
                    pos.direction, pos.open_price, curr_price, pos.volume, contract_size
                )
                sl_hit = pos.stop_loss is not None and tick.ask >= pos.stop_loss
                tp_hit = pos.take_profit is not None and tick.ask <= pos.take_profit
                close_price = (
                    max(pos.stop_loss, tick.ask)
                    if (sl_hit and pos.stop_loss is not None)
                    else (pos.take_profit if tp_hit and pos.take_profit is not None else 0.0)
                )

            if sl_hit or tp_hit:
                realized_pnl = _calc_gross_pnl(
                    pos.direction, pos.open_price, close_price, pos.volume, contract_size
                )
                self.balance += realized_pnl
                closed_tickets.append(ticket)
                reason = "SL" if sl_hit else "TP"
                logger.info(
                    "event=position_closed ticket=%d reason=%s price=%.2f pnl=%.2f",
                    ticket,
                    reason,
                    close_price,
                    realized_pnl,
                    extra={
                        "event": "position_closed",
                        "ticket": ticket,
                        "reason": reason,
                        "price": close_price,
                        "pnl": realized_pnl,
                    },
                )
            else:
                self._positions[ticket] = BrokerPosition(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    volume=pos.volume,
                    open_price=pos.open_price,
                    open_time=pos.open_time,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    current_price=curr_price,
                    floating_pnl=floating_pnl,
                    commission=pos.commission,
                    comment=pos.comment,
                    magic=pos.magic,
                )

        for ticket in closed_tickets:
            self._positions.pop(ticket, None)

    async def get_account_info(self) -> AccountInfo:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            total_margin = 0.0
            total_floating_pnl = 0.0
            for pos in self._positions.values():
                spec = self._symbol_specs.get(pos.symbol)
                contract_size = spec.contract_size if spec else 100.0
                pos_margin = (pos.volume * contract_size * pos.open_price) / max(1.0, self.leverage)
                total_margin += pos_margin
                total_floating_pnl += pos.floating_pnl

            equity = self.balance + total_floating_pnl
            free_margin = equity - total_margin
            margin_level = (equity / total_margin * 100.0) if total_margin > 0.0 else float("inf")

            return AccountInfo(
                balance=self.balance,
                equity=equity,
                margin=total_margin,
                free_margin=free_margin,
                margin_level=margin_level,
                leverage=self.leverage,
                currency=self.currency,
            )

    async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)
            return self._symbol_specs.get(symbol)

    async def send_order(self, request: OrderRequest) -> OrderResult:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            if not self._connected:
                return OrderResult(
                    success=False,
                    error_code=1,
                    error_message="Corretora desconectada",
                    retcode=10015,
                )

            if self.rejection_rate > 0.0 and self._rng.random() < self.rejection_rate:
                return OrderResult(
                    success=False,
                    error_code=2,
                    error_message="Rejeição simulada de corretora (Requote / Falta de liquidez)",
                    retcode=10004,
                )

            spec = self._symbol_specs.get(request.symbol)
            if spec is None:
                return OrderResult(
                    success=False,
                    error_code=3,
                    error_message=f"Símbolo {request.symbol} não suportado",
                    retcode=10013,
                )

            if request.volume < spec.min_volume or request.volume > spec.max_volume:
                return OrderResult(
                    success=False,
                    error_code=4,
                    error_message=(
                        f"Volume {request.volume} inválido "
                        f"(mín: {spec.min_volume}, máx: {spec.max_volume})"
                    ),
                    retcode=10014,
                )

            rem = round((request.volume - spec.min_volume) % spec.volume_step, 6)
            if rem not in (0.0, spec.volume_step):
                return OrderResult(
                    success=False,
                    error_code=4,
                    error_message=f"Volume {request.volume} não respeita passo {spec.volume_step}",
                    retcode=10014,
                )

            if request.order_type in (OrderType.BUY, OrderType.SELL):
                direction = (
                    Direction.BULLISH if request.order_type == OrderType.BUY else Direction.BEARISH
                )
                fill_price = spec.ask if direction == Direction.BULLISH else spec.bid
                ref_sl = spec.bid if direction == Direction.BULLISH else spec.ask
                ref_tp = spec.ask if direction == Direction.BULLISH else spec.bid
                stops_err = _validate_stops(
                    direction, ref_sl, ref_tp, request.stop_loss, request.take_profit, spec
                )
                if stops_err is not None:
                    return stops_err
            elif request.order_type in (OrderType.BUY_LIMIT, OrderType.SELL_LIMIT):
                if request.price is None or request.price <= 0.0:
                    return OrderResult(
                        success=False,
                        error_code=6,
                        error_message="Preço obrigatório e positivo para ordens pendentes",
                        retcode=10015,
                    )
                direction = (
                    Direction.BULLISH
                    if request.order_type == OrderType.BUY_LIMIT
                    else Direction.BEARISH
                )
                if request.order_type == OrderType.BUY_LIMIT and request.price > spec.ask:
                    return OrderResult(
                        success=False,
                        error_code=7,
                        error_message="Preço de BUY_LIMIT deve ser menor ou igual ao Ask atual",
                        retcode=10016,
                    )
                if request.order_type == OrderType.SELL_LIMIT and request.price < spec.bid:
                    return OrderResult(
                        success=False,
                        error_code=7,
                        error_message="Preço de SELL_LIMIT deve ser maior ou igual ao Bid atual",
                        retcode=10016,
                    )
                stops_err = _validate_stops(
                    direction,
                    request.price,
                    request.price,
                    request.stop_loss,
                    request.take_profit,
                    spec,
                )
                if stops_err is not None:
                    return stops_err
                fill_price = request.price
            else:
                return OrderResult(
                    success=False,
                    error_code=6,
                    error_message=f"Tipo de ordem {request.order_type} não suportado",
                    retcode=10015,
                )

            commission = request.volume * self.commission_per_lot
            required_margin = (request.volume * spec.contract_size * fill_price) / max(
                1.0, self.leverage
            )

            current_margin = sum(
                (p.volume * self._symbol_specs.get(p.symbol, spec).contract_size * p.open_price)
                / max(1.0, self.leverage)
                for p in self._positions.values()
            )
            floating_pnl = sum(p.floating_pnl for p in self._positions.values())
            equity = self.balance + floating_pnl
            free_margin = equity - current_margin

            if required_margin + commission > free_margin:
                return OrderResult(
                    success=False,
                    error_code=5,
                    error_message=(
                        f"Margem insuficiente (requerida: US$ {required_margin:.2f} "
                        f"+ comissão {commission:.2f}, livre: US$ {free_margin:.2f})"
                    ),
                    retcode=10019,
                )

            self._next_ticket += 1
            ticket = self._next_ticket
            now = datetime.now(tz=UTC)

            if request.order_type in (OrderType.BUY, OrderType.SELL):
                self.balance -= commission
                curr_exit = spec.bid if direction == Direction.BULLISH else spec.ask
                init_floating = _calc_gross_pnl(
                    direction, fill_price, curr_exit, request.volume, spec.contract_size
                )
                pos = BrokerPosition(
                    ticket=ticket,
                    symbol=request.symbol,
                    direction=direction,
                    volume=request.volume,
                    open_price=fill_price,
                    open_time=now,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit,
                    current_price=curr_exit,
                    floating_pnl=init_floating,
                    commission=commission,
                    comment=request.comment,
                    magic=request.magic,
                )
                self._positions[ticket] = pos
                logger.info(
                    "event=order_placed ticket=%d symbol=%s type=%s volume=%.2f price=%.2f",
                    ticket,
                    request.symbol,
                    direction.value,
                    request.volume,
                    fill_price,
                    extra={
                        "event": "order_placed",
                        "ticket": ticket,
                        "symbol": request.symbol,
                        "type": direction.value,
                        "volume": request.volume,
                        "price": fill_price,
                    },
                )
                return OrderResult(
                    success=True,
                    ticket=ticket,
                    price=fill_price,
                    volume=request.volume,
                    retcode=10009,
                )

            order_price = request.price if request.price is not None else fill_price
            order = BrokerOrder(
                ticket=ticket,
                symbol=request.symbol,
                order_type=request.order_type,
                volume=request.volume,
                price=order_price,
                time_setup=now,
                stop_loss=request.stop_loss,
                take_profit=request.take_profit,
                status=OrderStatus.PENDING,
                comment=request.comment,
                magic=request.magic,
            )
            self._orders[ticket] = order
            logger.info(
                "event=pending_order_registered ticket=%d symbol=%s type=%s volume=%.2f price=%.2f",
                ticket,
                request.symbol,
                request.order_type.value,
                request.volume,
                order_price,
                extra={
                    "event": "pending_order_registered",
                    "ticket": ticket,
                    "symbol": request.symbol,
                    "type": request.order_type.value,
                    "volume": request.volume,
                    "price": order_price,
                },
            )
            return OrderResult(
                success=True,
                ticket=ticket,
                price=order_price,
                volume=request.volume,
                retcode=10009,
            )

    async def modify_order(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            if ticket in self._positions:
                pos = self._positions[ticket]
                spec = self._symbol_specs.get(pos.symbol)
                if spec is None:
                    return OrderResult(
                        success=False,
                        error_code=3,
                        error_message=f"Símbolo {pos.symbol} não suportado",
                        retcode=10013,
                    )

                ref_sl = spec.bid if pos.direction == Direction.BULLISH else spec.ask
                ref_tp = spec.ask if pos.direction == Direction.BULLISH else spec.bid
                stops_err = _validate_stops(
                    pos.direction, ref_sl, ref_tp, stop_loss, take_profit, spec
                )
                if stops_err is not None:
                    return stops_err

                new_sl = (
                    None
                    if stop_loss == 0.0
                    else (stop_loss if stop_loss is not None else pos.stop_loss)
                )
                new_tp = (
                    None
                    if take_profit == 0.0
                    else (take_profit if take_profit is not None else pos.take_profit)
                )

                self._positions[ticket] = BrokerPosition(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    volume=pos.volume,
                    open_price=pos.open_price,
                    open_time=pos.open_time,
                    stop_loss=new_sl,
                    take_profit=new_tp,
                    current_price=pos.current_price,
                    floating_pnl=pos.floating_pnl,
                    commission=pos.commission,
                    comment=pos.comment,
                    magic=pos.magic,
                )
                logger.info(
                    "event=position_modified ticket=%d SL=%s TP=%s",
                    ticket,
                    str(stop_loss),
                    str(take_profit),
                    extra={
                        "event": "position_modified",
                        "ticket": ticket,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    },
                )
                return OrderResult(success=True, ticket=ticket, retcode=10009)

            if ticket in self._orders:
                order = self._orders[ticket]
                spec = self._symbol_specs.get(order.symbol)
                if spec is None:
                    return OrderResult(
                        success=False,
                        error_code=3,
                        error_message=f"Símbolo {order.symbol} não suportado",
                        retcode=10013,
                    )

                direction = (
                    Direction.BULLISH
                    if order.order_type == OrderType.BUY_LIMIT
                    else Direction.BEARISH
                )
                stops_err = _validate_stops(
                    direction, order.price, order.price, stop_loss, take_profit, spec
                )
                if stops_err is not None:
                    return stops_err

                new_sl = (
                    None
                    if stop_loss == 0.0
                    else (stop_loss if stop_loss is not None else order.stop_loss)
                )
                new_tp = (
                    None
                    if take_profit == 0.0
                    else (take_profit if take_profit is not None else order.take_profit)
                )

                self._orders[ticket] = BrokerOrder(
                    ticket=order.ticket,
                    symbol=order.symbol,
                    order_type=order.order_type,
                    volume=order.volume,
                    price=order.price,
                    time_setup=order.time_setup,
                    stop_loss=new_sl,
                    take_profit=new_tp,
                    status=order.status,
                    comment=order.comment,
                    magic=order.magic,
                )
                logger.info(
                    "event=pending_order_modified ticket=%d SL=%s TP=%s",
                    ticket,
                    str(stop_loss),
                    str(take_profit),
                    extra={
                        "event": "pending_order_modified",
                        "ticket": ticket,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    },
                )
                return OrderResult(success=True, ticket=ticket, retcode=10009)

            return OrderResult(
                success=False,
                error_code=8,
                error_message=f"Ticket {ticket} não encontrado",
                retcode=10017,
            )

    async def cancel_order(self, ticket: int) -> OrderResult:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            if ticket in self._orders:
                self._orders.pop(ticket)
                logger.info(
                    "event=order_cancelled ticket=%d",
                    ticket,
                    extra={"event": "order_cancelled", "ticket": ticket},
                )
                return OrderResult(success=True, ticket=ticket, retcode=10009)

            return OrderResult(
                success=False,
                error_code=8,
                error_message=f"Ordem {ticket} não encontrada",
                retcode=10017,
            )

    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult:
        async with self._lock:
            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            if ticket not in self._positions:
                return OrderResult(
                    success=False,
                    error_code=8,
                    error_message=f"Posição {ticket} não encontrada",
                    retcode=10017,
                )

            pos = self._positions[ticket]
            if volume is not None and abs(volume - pos.volume) > 1e-6:
                return OrderResult(
                    success=False,
                    error_code=8,
                    error_message="Fechamento parcial não suportado no simulador",
                    retcode=10015,
                )

            self._positions.pop(ticket)
            spec = self._symbol_specs.get(pos.symbol)
            contract_size = spec.contract_size if spec else 100.0

            if spec is not None:
                close_price = spec.bid if pos.direction == Direction.BULLISH else spec.ask
            else:
                close_price = pos.current_price

            gross_pnl = _calc_gross_pnl(
                pos.direction, pos.open_price, close_price, pos.volume, contract_size
            )
            self.balance += gross_pnl
            logger.info(
                "event=position_closed ticket=%d price=%.2f pnl=%.2f",
                ticket,
                close_price,
                gross_pnl,
                extra={
                    "event": "position_closed",
                    "ticket": ticket,
                    "price": close_price,
                    "pnl": gross_pnl,
                },
            )

            return OrderResult(
                success=True,
                ticket=ticket,
                price=close_price,
                volume=pos.volume,
                retcode=10009,
            )

    async def get_open_positions(self) -> tuple[BrokerPosition, ...]:
        async with self._lock:
            return tuple(self._positions.values())

    async def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        async with self._lock:
            return tuple(self._orders.values())
