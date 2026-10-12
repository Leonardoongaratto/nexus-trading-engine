import asyncio
import logging
import threading
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import SecretStr

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderStatus,
    OrderType,
    SymbolInfo,
)
from xau_bot.broker.mt5_adapter import Mt5AdapterConfig, Mt5BrokerAdapter
from xau_bot.broker.mt5_client import (
    MockAccountInfo,
    MockSymbolInfo,
    MockTradeOrder,
    MockTradePosition,
    MT5MockClient,
    get_default_mt5_client,
)
from xau_bot.broker.mt5_mapping import (
    ORDER_FILLING_FOK,
    ORDER_FILLING_IOC,
    ORDER_FILLING_RETURN,
    ORDER_STATE_CANCELED,
    ORDER_STATE_EXPIRED,
    ORDER_STATE_FILLED,
    ORDER_STATE_PARTIAL,
    ORDER_STATE_PLACED,
    ORDER_STATE_REJECTED,
    ORDER_STATE_STARTED,
    ORDER_TYPE_BUY,
    ORDER_TYPE_BUY_LIMIT,
    ORDER_TYPE_BUY_STOP,
    ORDER_TYPE_SELL,
    ORDER_TYPE_SELL_LIMIT,
    ORDER_TYPE_SELL_STOP,
    POSITION_TYPE_BUY,
    POSITION_TYPE_SELL,
    RETCODE_CONNECTION,
    RETCODE_DONE,
    RETCODE_ERROR,
    RETCODE_INVALID,
    RETCODE_INVALID_STOPS,
    RETCODE_INVALID_VOLUME,
    RETCODE_NO_MONEY,
    RETCODE_PLACED,
    SYMBOL_FILLING_FOK,
    SYMBOL_FILLING_IOC,
    SYMBOL_FILLING_RETURN,
    TRADE_ACTION_MODIFY,
    TRADE_ACTION_REMOVE,
    TRADE_ACTION_SLTP,
    map_mt5_account_info,
    map_mt5_order,
    map_mt5_position,
    map_mt5_symbol_info,
    map_mt5_to_order_status,
    map_mt5_to_order_type,
    map_order_type_to_mt5,
    resolve_filling_mode,
)
from xau_bot.config import Settings
from xau_bot.domain.models import Direction, Timeframe


def test_resolve_filling_mode() -> None:
    assert resolve_filling_mode(SYMBOL_FILLING_FOK) == ORDER_FILLING_FOK
    assert resolve_filling_mode(SYMBOL_FILLING_FOK | SYMBOL_FILLING_IOC) == ORDER_FILLING_FOK
    assert resolve_filling_mode(SYMBOL_FILLING_IOC) == ORDER_FILLING_IOC
    assert resolve_filling_mode(SYMBOL_FILLING_RETURN) == ORDER_FILLING_RETURN
    assert resolve_filling_mode(0) == ORDER_FILLING_RETURN


def test_order_type_mappings() -> None:
    types = [
        (OrderType.BUY, ORDER_TYPE_BUY),
        (OrderType.SELL, ORDER_TYPE_SELL),
        (OrderType.BUY_LIMIT, ORDER_TYPE_BUY_LIMIT),
        (OrderType.SELL_LIMIT, ORDER_TYPE_SELL_LIMIT),
        (OrderType.BUY_STOP, ORDER_TYPE_BUY_STOP),
        (OrderType.SELL_STOP, ORDER_TYPE_SELL_STOP),
    ]
    for domain_t, mt5_t in types:
        assert map_order_type_to_mt5(domain_t) == mt5_t
        assert map_mt5_to_order_type(mt5_t) == domain_t

    with pytest.raises(ValueError, match="Unknown MT5 order type"):
        map_mt5_to_order_type(999)


def test_order_status_mappings() -> None:
    assert map_mt5_to_order_status(ORDER_STATE_STARTED) == OrderStatus.PENDING
    assert map_mt5_to_order_status(ORDER_STATE_PLACED) == OrderStatus.PENDING
    assert map_mt5_to_order_status(ORDER_STATE_CANCELED) == OrderStatus.CANCELED
    assert map_mt5_to_order_status(ORDER_STATE_PARTIAL) == OrderStatus.FILLED
    assert map_mt5_to_order_status(ORDER_STATE_FILLED) == OrderStatus.FILLED
    assert map_mt5_to_order_status(ORDER_STATE_REJECTED) == OrderStatus.REJECTED
    assert map_mt5_to_order_status(ORDER_STATE_EXPIRED) == OrderStatus.EXPIRED
    assert map_mt5_to_order_status(999) == OrderStatus.PENDING


def test_map_mt5_account_info() -> None:
    mock_acc = MockAccountInfo(
        balance=5000.0,
        equity=5100.0,
        margin=200.0,
        margin_free=4900.0,
        margin_level=2550.0,
        leverage=200.0,
        currency="USD",
    )
    dto = map_mt5_account_info(mock_acc)
    assert dto.balance == 5000.0
    assert dto.equity == 5100.0
    assert dto.margin == 200.0
    assert dto.free_margin == 4900.0
    assert dto.margin_level == 2550.0
    assert dto.leverage == 200.0
    assert dto.currency == "USD"


def test_map_mt5_symbol_info() -> None:
    mock_sym = MockSymbolInfo(
        name="XAUUSD",
        bid=2650.0,
        ask=2650.30,
        spread=30.0,
        point=0.01,
        trade_stops_level=50.0,
        trade_freeze_level=20.0,
        trade_tick_size=0.01,
        trade_tick_value=1.0,
    )
    dto = map_mt5_symbol_info(mock_sym)
    assert dto.name == "XAUUSD"
    assert dto.bid == 2650.0
    assert dto.ask == 2650.30
    assert dto.spread_points == pytest.approx(0.30)
    assert dto.stops_level_points == pytest.approx(0.50)
    assert dto.freeze_level_points == pytest.approx(0.20)
    assert dto.tick_size == 0.01
    assert dto.tick_value == 1.0


def test_map_mt5_position_and_order_timestamps() -> None:
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    pos_dt = MockTradePosition(
        ticket=101,
        symbol="XAUUSD",
        type=POSITION_TYPE_BUY,
        volume=0.2,
        price_open=2650.0,
        time=now,
    )
    dto_pos = map_mt5_position(pos_dt)
    assert dto_pos.open_time == now
    assert dto_pos.direction == Direction.BULLISH

    pos_epoch = MockTradePosition(
        ticket=102,
        symbol="XAUUSD",
        type=POSITION_TYPE_SELL,
        volume=0.3,
        price_open=2650.0,
        time=1759060800,
    )
    dto_pos2 = map_mt5_position(pos_epoch)
    assert dto_pos2.direction == Direction.BEARISH
    assert dto_pos2.open_time.tzinfo == UTC

    ord_epoch = MockTradeOrder(
        ticket=201,
        symbol="XAUUSD",
        type=ORDER_TYPE_BUY_LIMIT,
        volume_initial=0.1,
        volume_current=0.1,
        price_open=2640.0,
        time_setup=1759060800,
    )
    dto_ord = map_mt5_order(ord_epoch)
    assert dto_ord.order_type == OrderType.BUY_LIMIT
    assert dto_ord.time_setup.tzinfo == UTC


def test_get_default_mt5_client_raises_when_not_installed() -> None:
    import importlib.util

    if importlib.util.find_spec("MetaTrader5") is None:
        with pytest.raises(RuntimeError, match="MetaTrader5 package is not installed"):
            get_default_mt5_client()


def test_mt5_mock_client_lifecycle() -> None:
    client = MT5MockClient(initial_connected=False, fail_initialize=False)
    assert not client.connected
    assert client.terminal_info() is None
    assert client.account_info() is None

    assert client.initialize(login=999)
    assert client.connected
    assert client.terminal_info() is not None
    acc = client.account_info()
    assert acc is not None
    assert acc.login == 999

    client.shutdown()
    assert not client.connected

    client_fail = MT5MockClient(fail_initialize=True)
    assert not client_fail.initialize()
    code, msg = client_fail.last_error()
    assert code == -1


@pytest.mark.asyncio
async def test_adapter_connect_and_disconnect() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)

    assert not adapter.is_connected()
    ok = await adapter.connect()
    assert ok
    assert adapter.is_connected()

    await adapter.disconnect()
    assert not adapter.is_connected()


@pytest.mark.asyncio
async def test_adapter_connect_failure() -> None:
    client = MT5MockClient(fail_initialize=True)
    adapter = Mt5BrokerAdapter(client=client)

    ok = await adapter.connect()
    assert not ok
    assert not adapter.is_connected()


@pytest.mark.asyncio
async def test_adapter_reconnect_success() -> None:
    client = MT5MockClient(fail_initialize=True)
    cfg = Mt5AdapterConfig(
        reconnect_initial_delay=1.0,
        reconnect_max_delay=5.0,
        reconnect_backoff_factor=2.0,
        reconnect_jitter=0.0,
        max_reconnect_attempts=3,
    )
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        client.fail_initialize = False

    adapter = Mt5BrokerAdapter(client=client, config=cfg, sleep_fn=fake_sleep)
    ok = await adapter.reconnect()
    assert ok
    assert adapter.is_connected()
    assert len(sleeps) == 1
    assert sleeps[0] == 1.0


@pytest.mark.asyncio
async def test_adapter_reconnect_exhausted() -> None:
    client = MT5MockClient(fail_initialize=True)
    cfg = Mt5AdapterConfig(
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.02,
        max_reconnect_attempts=2,
    )
    adapter = Mt5BrokerAdapter(client=client, config=cfg)
    ok = await adapter.reconnect()
    assert not ok
    assert not adapter.is_connected()


@pytest.mark.asyncio
async def test_adapter_get_account_info() -> None:
    client = MT5MockClient(initial_balance=15000.0)
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    acc = await adapter.get_account_info()
    assert isinstance(acc, AccountInfo)
    assert acc.balance == 15000.0
    assert acc.free_margin == 15000.0


@pytest.mark.asyncio
async def test_adapter_get_account_info_disconnected_raises() -> None:
    client = MT5MockClient(initial_connected=False)
    adapter = Mt5BrokerAdapter(client=client)
    with pytest.raises(RuntimeError, match="Broker has never been connected"):
        await adapter.get_account_info()


@pytest.mark.asyncio
async def test_adapter_get_symbol_info() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    sym = await adapter.get_symbol_info("XAUUSD")
    assert sym is not None
    assert isinstance(sym, SymbolInfo)
    assert sym.name == "XAUUSD"
    assert sym.bid == 2650.00
    assert sym.ask == 2650.25

    unknown = await adapter.get_symbol_info("UNKNOWN")
    assert unknown is None


@pytest.mark.asyncio
async def test_adapter_send_market_buy_and_sell() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    req_buy = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        volume=0.1,
        stop_loss=2645.0,
        take_profit=2660.0,
        comment="test_buy",
        magic=123,
    )
    res_buy = await adapter.send_order(req_buy)
    assert res_buy.success
    assert res_buy.ticket is not None
    assert res_buy.price == 2650.25
    assert res_buy.volume == 0.1
    assert res_buy.retcode == RETCODE_DONE

    req_sell = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.SELL,
        volume=0.2,
        stop_loss=2655.0,
        take_profit=2640.0,
        comment="test_sell",
    )
    res_sell = await adapter.send_order(req_sell)
    assert res_sell.success
    assert res_sell.price == 2650.00
    assert res_sell.volume == 0.2


@pytest.mark.asyncio
async def test_adapter_send_pending_limit_orders() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    req_limit = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        volume=0.1,
        price=2640.0,
        stop_loss=2635.0,
        take_profit=2655.0,
    )
    res_limit = await adapter.send_order(req_limit)
    assert res_limit.success
    assert res_limit.retcode == RETCODE_PLACED
    assert res_limit.ticket is not None


@pytest.mark.asyncio
async def test_adapter_send_order_rejections() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res_sym = await adapter.send_order(
        OrderRequest(symbol="UNKNOWN", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_sym.success
    assert res_sym.retcode == RETCODE_INVALID

    res_vol = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.001)
    )
    assert not res_vol.success
    assert res_vol.retcode == RETCODE_INVALID_VOLUME

    res_no_price = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price=None)
    )
    assert not res_no_price.success
    assert res_no_price.retcode == RETCODE_INVALID

    res_stops = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            stop_loss=2650.0,
        )
    )
    assert not res_stops.success
    assert res_stops.retcode == RETCODE_INVALID_STOPS

    res_margin = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=50.0)
    )
    assert not res_margin.success
    assert res_margin.retcode == RETCODE_NO_MONEY


@pytest.mark.asyncio
async def test_adapter_modify_open_position_stops() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    ticket = res.ticket
    assert ticket is not None

    mod_res = await adapter.modify_order(
        ticket=ticket,
        stop_loss=2645.0,
        take_profit=2665.0,
    )
    assert mod_res.success
    assert mod_res.retcode == RETCODE_DONE

    bad_mod = await adapter.modify_order(
        ticket=ticket,
        stop_loss=2649.50,
    )
    assert not bad_mod.success
    assert bad_mod.retcode == RETCODE_INVALID_STOPS


@pytest.mark.asyncio
async def test_adapter_modify_pending_order() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price=2640.0,
            stop_loss=2635.0,
        )
    )
    ticket = res.ticket
    assert ticket is not None

    mod_res = await adapter.modify_order(ticket=ticket, stop_loss=2630.0, take_profit=2650.0)
    assert mod_res.success

    not_found = await adapter.modify_order(ticket=9999, stop_loss=2600.0)
    assert not not_found.success
    assert not_found.retcode == RETCODE_INVALID


@pytest.mark.asyncio
async def test_adapter_cancel_pending_order() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price=2640.0,
        )
    )
    ticket = res.ticket
    assert ticket is not None

    cancel_res = await adapter.cancel_order(ticket=ticket)
    assert cancel_res.success
    assert cancel_res.retcode == RETCODE_DONE

    cancel_again = await adapter.cancel_order(ticket=ticket)
    assert not cancel_again.success
    assert cancel_again.retcode == RETCODE_INVALID


@pytest.mark.asyncio
async def test_adapter_close_position_full_and_partial() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res_open = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.5)
    )
    ticket = res_open.ticket
    assert ticket is not None

    res_partial = await adapter.close_position(ticket=ticket, volume=0.2)
    assert res_partial.success
    assert res_partial.volume == 0.2

    positions = await adapter.get_open_positions()
    assert len(positions) == 1
    assert positions[0].volume == pytest.approx(0.3)

    res_full = await adapter.close_position(ticket=ticket)
    assert res_full.success
    assert res_full.volume == pytest.approx(0.3)

    positions_after = await adapter.get_open_positions()
    assert len(positions_after) == 0

    res_not_found = await adapter.close_position(ticket=9999)
    assert not res_not_found.success
    assert res_not_found.retcode == RETCODE_INVALID


@pytest.mark.asyncio
async def test_adapter_get_open_positions_and_orders() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    assert await adapter.get_open_positions() == ()
    assert await adapter.get_open_orders() == ()

    await adapter.send_order(OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1))
    await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL_LIMIT, volume=0.2, price=2660.0)
    )

    positions = await adapter.get_open_positions()
    assert len(positions) == 1
    assert isinstance(positions[0], BrokerPosition)
    assert positions[0].direction == Direction.BULLISH

    orders = await adapter.get_open_orders()
    assert len(orders) == 1
    assert isinstance(orders[0], BrokerOrder)
    assert orders[0].order_type == OrderType.SELL_LIMIT


def test_validate_stops_exhaustive() -> None:
    adapter = Mt5BrokerAdapter(client=MT5MockClient())

    assert adapter._validate_stops(
        is_buy=True,
        ref_price=2650.0,
        sl=2645.0,
        tp=2660.0,
        stops_level_points=1.0,
        freeze_level_points=0.5,
    )
    assert not adapter._validate_stops(
        is_buy=True, ref_price=2650.0, sl=2650.0, tp=2660.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=True, ref_price=2650.0, sl=2649.5, tp=2660.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=True,
        ref_price=2650.0,
        sl=2649.2,
        tp=2660.0,
        stops_level_points=0.5,
        freeze_level_points=1.0,
    )
    assert not adapter._validate_stops(
        is_buy=True, ref_price=2650.0, sl=2645.0, tp=2650.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=True, ref_price=2650.0, sl=2645.0, tp=2650.5, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=True,
        ref_price=2650.0,
        sl=2645.0,
        tp=2650.7,
        stops_level_points=0.5,
        freeze_level_points=1.0,
    )

    assert adapter._validate_stops(
        is_buy=False,
        ref_price=2650.0,
        sl=2655.0,
        tp=2640.0,
        stops_level_points=1.0,
        freeze_level_points=0.5,
    )
    assert not adapter._validate_stops(
        is_buy=False, ref_price=2650.0, sl=2650.0, tp=2640.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=False, ref_price=2650.0, sl=2650.5, tp=2640.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=False,
        ref_price=2650.0,
        sl=2650.6,
        tp=2640.0,
        stops_level_points=0.5,
        freeze_level_points=1.0,
    )
    assert not adapter._validate_stops(
        is_buy=False, ref_price=2650.0, sl=2655.0, tp=2650.0, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=False, ref_price=2650.0, sl=2655.0, tp=2649.5, stops_level_points=1.0
    )
    assert not adapter._validate_stops(
        is_buy=False,
        ref_price=2650.0,
        sl=2655.0,
        tp=2649.2,
        stops_level_points=0.5,
        freeze_level_points=1.0,
    )
    assert adapter._validate_stops(
        is_buy=True,
        ref_price=2650.0,
        sl=2645.0,
        tp=0.0,
        stops_level_points=1.0,
        freeze_level_points=0.5,
    )
    assert adapter._validate_stops(
        is_buy=False,
        ref_price=2650.0,
        sl=2655.0,
        tp=0.0,
        stops_level_points=1.0,
        freeze_level_points=0.5,
    )


@pytest.mark.asyncio
async def test_adapter_modify_order_clears_take_profit_with_zero() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    send_res = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            price=2650.0,
            stop_loss=2640.0,
            take_profit=2670.0,
        )
    )
    assert send_res.success
    ticket = send_res.ticket
    assert ticket is not None

    mod_res = await adapter.modify_order(ticket=ticket, take_profit=0.0)
    assert mod_res.success


@pytest.mark.asyncio
async def test_adapter_modify_pending_order_sell_limit() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            price=2660.0,
            stop_loss=2665.0,
        )
    )
    ticket = res.ticket
    assert ticket is not None

    bad_res = await adapter.modify_order(ticket=ticket, stop_loss=2660.5)
    assert not bad_res.success
    assert bad_res.retcode == RETCODE_INVALID_STOPS

    good_res = await adapter.modify_order(ticket=ticket, stop_loss=2670.0, take_profit=2650.0)
    assert good_res.success


@pytest.mark.asyncio
async def test_adapter_order_send_returns_none() -> None:
    class FailingSendClient(MT5MockClient):
        def order_send(self, request: dict[str, Any]) -> Any | None:
            self._last_error = (-5, "Network error")
            return None

    client = FailingSendClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res_send = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_send.success
    assert res_send.error_code == -5

    client._positions[333] = MockTradePosition(
        ticket=333,
        symbol="XAUUSD",
        type=POSITION_TYPE_BUY,
        volume=0.1,
        price_open=2650.0,
        time=int(datetime.now(tz=UTC).timestamp()),
    )
    mod_res = await adapter.modify_order(ticket=333, stop_loss=2640.0)
    assert not mod_res.success
    assert mod_res.error_code == -5

    client._orders[444] = MockTradeOrder(
        ticket=444,
        symbol="XAUUSD",
        type=ORDER_TYPE_BUY_LIMIT,
        volume_initial=0.1,
        volume_current=0.1,
        price_open=2640.0,
        time_setup=int(datetime.now(tz=UTC).timestamp()),
    )
    cancel_res = await adapter.cancel_order(ticket=444)
    assert not cancel_res.success
    assert cancel_res.error_code == -5

    close_res = await adapter.close_position(ticket=333)
    assert not close_res.success
    assert close_res.error_code == -5


@pytest.mark.asyncio
async def test_adapter_run_in_thread_kwargs() -> None:
    adapter = Mt5BrokerAdapter(client=MT5MockClient())

    def dummy(a: int, b: int = 10) -> int:
        return a + b

    val = await adapter._run_in_thread(dummy, 5, b=20)
    assert val == 25


def test_mt5_mock_client_edge_cases() -> None:
    client = MT5MockClient()

    assert client.symbol_info_tick("UNKNOWN") is None
    assert not client.symbol_select("UNKNOWN")

    res_chk = client.order_check({"volume": 0.1, "symbol": "UNKNOWN"})
    assert res_chk is not None and res_chk.retcode != 0

    res_send = client.order_send({"action": 999})
    assert res_send is not None and res_send.retcode == RETCODE_INVALID

    res_sltp = client.order_send({"action": TRADE_ACTION_SLTP, "position": 9999})
    assert res_sltp is not None and res_sltp.retcode == RETCODE_INVALID

    res_mod = client.order_send({"action": TRADE_ACTION_MODIFY, "order": 9999})
    assert res_mod is not None and res_mod.retcode == RETCODE_INVALID

    res_rem = client.order_send({"action": TRADE_ACTION_REMOVE, "order": 9999})
    assert res_rem is not None and res_rem.retcode == RETCODE_INVALID

    client.shutdown()
    assert client.symbol_info("XAUUSD") is None
    assert client.symbol_info_tick("XAUUSD") is None
    assert client.positions_get() is None
    assert client.orders_get() is None
    assert client.order_check({}) is None
    assert client.order_send({}) is None


@pytest.mark.asyncio
async def test_repro_m1_thread_affinity_all_mt5_calls_on_worker_thread() -> None:
    class ThreadTrackingClient(MT5MockClient):
        def __init__(self) -> None:
            super().__init__()
            self.thread_names: list[str] = []
            self.simulate_send_none: bool = False

        def _record_thread(self) -> None:
            self.thread_names.append(threading.current_thread().name)

        def initialize(self, *args: Any, **kwargs: Any) -> bool:
            self._record_thread()
            return super().initialize(*args, **kwargs)

        def shutdown(self) -> None:
            self._record_thread()
            super().shutdown()

        def terminal_info(self) -> Any:
            self._record_thread()
            return super().terminal_info()

        def account_info(self) -> Any:
            self._record_thread()
            return super().account_info()

        def symbol_info(self, symbol: str) -> Any:
            self._record_thread()
            return super().symbol_info(symbol)

        def symbol_select(self, symbol: str, enable: bool = True) -> bool:
            self._record_thread()
            return super().symbol_select(symbol, enable)

        def positions_get(self, *args: Any, **kwargs: Any) -> Any:
            self._record_thread()
            return super().positions_get(*args, **kwargs)

        def orders_get(self, *args: Any, **kwargs: Any) -> Any:
            self._record_thread()
            return super().orders_get(*args, **kwargs)

        def order_check(self, request: dict[str, Any]) -> Any:
            self._record_thread()
            return super().order_check(request)

        def order_send(self, request: dict[str, Any]) -> Any:
            self._record_thread()
            if getattr(self, "simulate_send_none", False):
                self._last_error = (10011, "simulated send failure")
                return None
            return super().order_send(request)

        def last_error(self) -> tuple[int, str]:
            self._record_thread()
            return super().last_error()

    client = ThreadTrackingClient()
    adapter = Mt5BrokerAdapter(client=client)

    await adapter.connect()
    connected = adapter.is_connected()
    assert connected
    await adapter.get_account_info()
    await adapter.get_symbol_info("XAUUSD")
    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert res.success and res.ticket is not None
    await adapter.modify_order(ticket=res.ticket, stop_loss=2640.0)
    await adapter.close_position(ticket=res.ticket)

    res_pending = await adapter.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price=2640.0,
        )
    )
    assert res_pending.success and res_pending.ticket is not None
    await adapter.cancel_order(ticket=res_pending.ticket)

    client.simulate_send_none = True
    res_fail = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_fail.success
    client.simulate_send_none = False

    await adapter.get_open_positions()
    await adapter.get_open_orders()
    await adapter.disconnect()
    adapter.close()

    assert len(client.thread_names) > 0
    for t_name in client.thread_names:
        assert t_name.startswith("mt5_worker"), f"Call executed on unexpected thread: {t_name}"


def test_repro_m2_settings_integration_and_omit_none_kwargs() -> None:
    settings = Settings(
        mt5_login=SecretStr("998877"),
        mt5_password=SecretStr("UltraSecretPass"),
        mt5_server="Brokers-Demo",
        mt5_path="C:\\Apps\\MT5\\terminal64.exe",
        mt5_timeout_ms=45000,
    )
    cfg = Mt5AdapterConfig.from_settings(settings)
    assert cfg.login == 998877
    assert cfg.password == "UltraSecretPass"
    assert cfg.server == "Brokers-Demo"
    assert cfg.path == "C:\\Apps\\MT5\\terminal64.exe"
    assert cfg.timeout_ms == 45000

    bad_settings = Settings(mt5_login=SecretStr("not-a-number"))
    with pytest.raises(ValueError, match="Configuration error: mt5_login must be a valid integer"):
        Mt5AdapterConfig.from_settings(bad_settings)

    client = MT5MockClient()
    cfg_sparse = Mt5AdapterConfig(login=12345, server=None, path=None)
    adapter = Mt5BrokerAdapter(client=client, config=cfg_sparse)
    asyncio.run(adapter.connect())
    assert "path" not in client.last_init_kwargs
    assert "server" not in client.last_init_kwargs
    assert client.last_init_kwargs.get("login") == 12345
    adapter.close()


@pytest.mark.asyncio
async def test_repro_m3_reconnect_backoff_and_jitter_exact_sequence() -> None:
    captured_delays: list[float] = []

    async def mock_sleep(d: float) -> None:
        captured_delays.append(d)

    client = MT5MockClient(fail_initialize=True)
    cfg = Mt5AdapterConfig(
        reconnect_initial_delay=2.0,
        reconnect_max_delay=5.0,
        reconnect_backoff_factor=2.0,
        reconnect_jitter=0.0,
        max_reconnect_attempts=4,
    )
    adapter = Mt5BrokerAdapter(client=client, config=cfg, sleep_fn=mock_sleep)

    ok = await adapter.reconnect()
    assert not ok
    assert captured_delays == [2.0, 4.0, 5.0]
    assert client.calls.count("initialize") == 4
    assert client.calls.count("initialize") == cfg.max_reconnect_attempts
    adapter.close()


@pytest.mark.asyncio
async def test_repro_m3_automatic_reconnection_on_disconnection() -> None:
    sleeps: list[float] = []

    async def mock_sleep(d: float) -> None:
        sleeps.append(d)

    client = MT5MockClient()
    cfg = Mt5AdapterConfig(
        reconnect_initial_delay=0.1,
        reconnect_max_delay=1.0,
        max_reconnect_attempts=3,
    )
    adapter = Mt5BrokerAdapter(client=client, config=cfg, sleep_fn=mock_sleep)
    await adapter.connect()
    assert adapter.is_connected()
    init_calls_before = client.calls.count("initialize")

    client.connected = False
    client._terminal.connected = False

    acc = await adapter.get_account_info()
    assert acc.balance == 10000.0
    assert adapter.is_connected()
    assert client.calls.count("initialize") == init_calls_before + 1

    init_calls_before = client.calls.count("initialize")
    client.fail_order_send_retcode = RETCODE_CONNECTION
    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res.success
    assert res.retcode == RETCODE_CONNECTION
    assert client.calls.count("initialize") == init_calls_before + 1

    client.fail_order_send_retcode = None
    res_ok = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert res_ok.success
    adapter.close()


@pytest.mark.asyncio
async def test_repro_m4_order_check_returns_none_fails_closed() -> None:
    class FailCheckClient(MT5MockClient):
        def order_check(self, request: dict[str, Any]) -> Any | None:
            self._last_error = (-10, "IPC timeout in check")
            return None

    client = FailCheckClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res.success
    assert res.retcode == RETCODE_ERROR
    assert "order_send" not in client.calls
    adapter.close()


@pytest.mark.asyncio
async def test_repro_m5_security_no_credentials_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MT5MockClient()
    cfg = Mt5AdapterConfig(
        login=998877,
        password="MySecretPassword123!",
        server="VerySecretBroker-Demo",
    )
    assert "MySecretPassword123!" not in repr(cfg)

    adapter = Mt5BrokerAdapter(client=client, config=cfg)
    with caplog.at_level(logging.DEBUG):
        await adapter.connect()
        await adapter.disconnect()

    assert "MySecretPassword123!" not in caplog.text
    assert "VerySecretBroker-Demo" not in caplog.text
    assert "998877" not in caplog.text
    adapter.close()


@pytest.mark.asyncio
async def test_adapter_close_and_aclose() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()
    assert adapter.is_connected()
    await adapter.aclose()
    assert not adapter.is_connected()


def test_map_mt5_symbol_info_zero_spread() -> None:
    mock_sym = MockSymbolInfo(name="XAUUSD", spread=0.0, point=0.01)
    dto = map_mt5_symbol_info(mock_sym)
    assert dto.spread_points == 0.0


@pytest.mark.asyncio
async def test_repro_m1_r3_explicit_disconnect_prevents_automatic_reconnect() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()
    assert adapter.is_connected()
    await adapter.disconnect()
    assert not adapter.is_connected()

    init_calls = client.calls.count("initialize")
    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res.success
    assert res.retcode == RETCODE_CONNECTION
    assert client.calls.count("initialize") == init_calls
    assert len(client._positions) == 0

    with pytest.raises(RuntimeError, match="explicitly disconnected"):
        await adapter.get_account_info()

    with pytest.raises(RuntimeError, match="explicitly disconnected"):
        await adapter.get_open_positions()

    with pytest.raises(RuntimeError, match="explicitly disconnected"):
        await adapter.get_open_orders()

    with pytest.raises(RuntimeError, match="explicitly disconnected"):
        await adapter.get_symbol_info("XAUUSD")

    adapter.close()


@pytest.mark.asyncio
async def test_repro_l_r3_1_never_connected_fails_closed() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res.success
    assert res.retcode == RETCODE_CONNECTION
    with pytest.raises(RuntimeError, match="never been connected"):
        await adapter.get_account_info()
    adapter.close()


@pytest.mark.asyncio
async def test_repro_m2_r3_positions_and_orders_fail_closed_on_error_or_disconnect() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    init_calls = client.calls.count("initialize")
    client.fail_positions_get_once = True
    client.fail_positions_retcode = RETCODE_CONNECTION

    positions = await adapter.get_open_positions()
    assert positions == ()
    assert client.calls.count("initialize") == init_calls + 1

    client.fail_positions_get = True
    client.fail_positions_retcode = -100
    with pytest.raises(RuntimeError, match="Failed to retrieve open positions"):
        await adapter.get_open_positions()
    client.fail_positions_get = False

    init_calls = client.calls.count("initialize")
    client.fail_orders_get_once = True
    client.fail_orders_retcode = RETCODE_CONNECTION

    orders = await adapter.get_open_orders()
    assert orders == ()
    assert client.calls.count("initialize") == init_calls + 1

    client.fail_orders_get = True
    client.fail_orders_retcode = -100
    with pytest.raises(RuntimeError, match="Failed to retrieve open orders"):
        await adapter.get_open_orders()
    client.fail_orders_get = False

    adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op_name", "fail_none"),
    [
        ("send", True),
        ("send", False),
        ("modify", True),
        ("modify", False),
        ("cancel", True),
        ("cancel", False),
        ("close", True),
        ("close", False),
    ],
)
async def test_repro_l_r3_2_and_3_write_operations_connection_error(
    op_name: str, fail_none: bool
) -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    client._positions[101] = MockTradePosition(
        ticket=101,
        symbol="XAUUSD",
        type=POSITION_TYPE_BUY,
        volume=0.1,
        price_open=2650.0,
    )
    client._orders[201] = MockTradeOrder(
        ticket=201,
        symbol="XAUUSD",
        type=ORDER_TYPE_BUY_LIMIT,
        volume_initial=0.1,
        volume_current=0.1,
        price_open=2640.0,
    )

    client.fail_order_send_none = fail_none
    client.fail_order_send_retcode = RETCODE_CONNECTION
    init_calls = client.calls.count("initialize")

    if op_name == "send":
        res = await adapter.send_order(
            OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
        )
    elif op_name == "modify":
        res = await adapter.modify_order(ticket=101, stop_loss=2640.0, take_profit=2660.0)
    elif op_name == "cancel":
        res = await adapter.cancel_order(ticket=201)
    else:
        res = await adapter.close_position(ticket=101)

    assert not res.success
    assert res.retcode == RETCODE_CONNECTION
    assert client.calls.count("initialize") == init_calls + 1
    assert client.calls.count("order_send") == 1
    adapter.close()


@pytest.mark.asyncio
async def test_repro_l_r3_2_write_operations_exhausted_reconnect_fail_closed() -> None:
    client = MT5MockClient()
    cfg = Mt5AdapterConfig(max_reconnect_attempts=2, reconnect_initial_delay=0.001)
    adapter = Mt5BrokerAdapter(client=client, config=cfg)
    await adapter.connect()
    client.connected = False
    client._terminal.connected = False
    client.fail_initialize = True

    res_send = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_send.success
    assert res_send.retcode == RETCODE_CONNECTION

    res_mod = await adapter.modify_order(ticket=101, stop_loss=2640.0)
    assert not res_mod.success
    assert res_mod.retcode == RETCODE_CONNECTION

    res_can = await adapter.cancel_order(ticket=201)
    assert not res_can.success
    assert res_can.retcode == RETCODE_CONNECTION

    res_clo = await adapter.close_position(ticket=101)
    assert not res_clo.success
    assert res_clo.retcode == RETCODE_CONNECTION
    adapter.close()


@pytest.mark.asyncio
async def test_repro_l_r4_1_reconnect_resets_intentional_disconnect() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()
    await adapter.disconnect()
    assert not adapter.is_connected()

    ok = await adapter.reconnect()
    assert ok
    assert adapter.is_connected()
    res = await adapter.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert res.success
    adapter.close()


@pytest.mark.asyncio
async def test_mt5_adapter_get_candles_success_and_empty() -> None:
    client = MT5MockClient()
    adapter = Mt5BrokerAdapter(client=client)
    await adapter.connect()

    candles = await adapter.get_candles("XAUUSD", Timeframe.M5, count=5)
    assert len(candles) == 5
    assert all(c.timeframe == Timeframe.M5 for c in candles)
    assert all(c.time.tzinfo == UTC for c in candles)
    assert all(candles[i].time < candles[i + 1].time for i in range(len(candles) - 1))

    class EmptyRatesClient(MT5MockClient):
        def copy_rates_from_pos(
            self, symbol: str, timeframe: int, start_pos: int, count: int
        ) -> Any | None:
            return None

    adapter_empty = Mt5BrokerAdapter(client=EmptyRatesClient())
    await adapter_empty.connect()
    assert await adapter_empty.get_candles("XAUUSD", Timeframe.M5) == ()
    adapter.close()
    adapter_empty.close()


@pytest.mark.asyncio
async def test_mt5_adapter_get_candles_converts_server_tz_to_utc() -> None:
    class ServerTzMockClient(MT5MockClient):
        def copy_rates_from_pos(
            self, symbol: str, timeframe: int, start_pos: int, count: int
        ) -> Any | None:
            return [
                {
                    "time": 1791194400,
                    "open": 2650.0,
                    "high": 2655.0,
                    "low": 2648.0,
                    "close": 2652.0,
                    "tick_volume": 100,
                }
            ]

    config = Mt5AdapterConfig(server_tz="Europe/Athens")
    adapter = Mt5BrokerAdapter(client=ServerTzMockClient(), config=config)
    await adapter.connect()
    candles = await adapter.get_candles("XAUUSD", Timeframe.M5, count=1)
    assert len(candles) == 1
    c = candles[0]
    assert c.time.tzinfo == UTC
    assert c.time.hour == 7
    adapter.close()


def test_map_mt5_position_and_order_server_tz() -> None:
    from xau_bot.broker.mt5_mapping import map_mt5_order, map_mt5_position

    pos_obj = MockTradePosition(
        ticket=101,
        symbol="XAUUSD",
        type=0,
        volume=0.1,
        price_open=2650.0,
        time=1791194400,
    )
    pos = map_mt5_position(pos_obj, server_tz="Europe/Athens")
    assert pos.open_time.tzinfo == UTC
    assert pos.open_time.hour == 7

    ord_obj = MockTradeOrder(
        ticket=201,
        symbol="XAUUSD",
        type=0,
        volume_initial=0.1,
        volume_current=0.1,
        price_open=2650.0,
        time_setup=1791194400,
    )
    order = map_mt5_order(ord_obj, server_tz="Europe/Athens")
    assert order.time_setup.tzinfo == UTC
    assert order.time_setup.hour == 7


@pytest.mark.asyncio
async def test_mt5_adapter_get_candles_handles_int_like_timestamp() -> None:
    class IntLike:
        def __init__(self, val: int) -> None:
            self.val = val

        def __float__(self) -> float:
            return float(self.val)

        def __int__(self) -> int:
            return self.val

    class IntLikeMockClient(MT5MockClient):
        def copy_rates_from_pos(
            self, symbol: str, timeframe: int, start_pos: int, count: int
        ) -> Any | None:
            return [
                {
                    "time": IntLike(1791194400),
                    "open": 2650.0,
                    "high": 2655.0,
                    "low": 2648.0,
                    "close": 2652.0,
                    "tick_volume": 100,
                }
            ]

    config = Mt5AdapterConfig(server_tz="UTC")
    adapter = Mt5BrokerAdapter(client=IntLikeMockClient(), config=config)
    await adapter.connect()
    candles = await adapter.get_candles("XAUUSD", Timeframe.M5, count=1)
    assert len(candles) == 1
    assert candles[0].time.tzinfo == UTC
    adapter.close()


@pytest.mark.asyncio
async def test_mt5_adapter_get_candles_handles_non_dict_getitem_record() -> None:
    class IntLike:
        def __init__(self, val: int) -> None:
            self.val = val

        def __float__(self) -> float:
            return float(self.val)

        def __int__(self) -> int:
            return self.val

    class RecRecord:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, item: str) -> Any:
            return self._data[item]

    class RecClient(MT5MockClient):
        def copy_rates_from_pos(
            self, symbol: str, timeframe: int, start_pos: int, count: int
        ) -> Any | None:
            return [
                RecRecord(
                    {
                        "time": IntLike(1791194400),
                        "open": 2650.0,
                        "high": 2655.0,
                        "low": 2648.0,
                        "close": 2652.0,
                        "tick_volume": 100,
                    }
                ),
                RecRecord(
                    {
                        "time": IntLike(1791194700),
                        "open": 2652.0,
                        "high": 2658.0,
                        "low": 2651.0,
                        "close": 2656.0,
                        "real_volume": 200,
                    }
                ),
            ]

    config = Mt5AdapterConfig(server_tz="UTC")
    adapter = Mt5BrokerAdapter(client=RecClient(), config=config)
    await adapter.connect()
    candles = await adapter.get_candles("XAUUSD", Timeframe.M5, count=2)
    assert len(candles) == 2
    assert candles[0].time.tzinfo == UTC
    assert candles[0].volume == 100.0
    assert candles[1].volume == 200.0
    adapter.close()
