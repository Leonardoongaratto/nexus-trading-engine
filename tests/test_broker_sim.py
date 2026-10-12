import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from xau_bot.broker import (
    AccountInfo,
    OrderRequest,
    OrderStatus,
    OrderType,
    SimBrokerAdapter,
    SymbolInfo,
)
from xau_bot.domain.models import Candle, Direction, Tick, Timeframe


def _create_sample_tick(
    price: float = 2650.0,
    spread: float = 0.25,
    timestamp: datetime | None = None,
) -> Tick:
    time_val = timestamp or datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    return Tick(
        time=time_val,
        bid=price,
        ask=price + spread,
        last=price,
        volume=1.0,
        flags=6,
    )


@pytest.mark.asyncio
async def test_ac_1401_sim_adapter_initialization_and_account_info() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, leverage=100.0)
    assert not broker.is_connected()

    connected = await broker.connect()
    assert connected
    assert broker.is_connected()

    acc = await broker.get_account_info()
    assert isinstance(acc, AccountInfo)
    assert acc.balance == 10000.0
    assert acc.equity == 10000.0
    assert acc.margin == 0.0
    assert acc.free_margin == 10000.0
    assert acc.leverage == 100.0
    assert acc.currency == "USD"

    symbol = await broker.get_symbol_info("XAUUSD")
    assert isinstance(symbol, SymbolInfo)
    assert symbol.name == "XAUUSD"
    assert symbol.contract_size == 100.0

    await broker.disconnect()
    assert not broker.is_connected()


@pytest.mark.asyncio
async def test_ac_1402_market_order_execution_and_margin() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, leverage=100.0, commission_per_lot=7.0)
    await broker.connect()

    tick = _create_sample_tick(price=2650.0, spread=0.30)
    broker.on_tick(tick)

    req = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        volume=1.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        comment="test buy",
    )
    res = await broker.send_order(req)
    assert res.success
    assert res.ticket is not None
    assert res.price == 2650.30
    assert res.volume == 1.0

    positions = await broker.get_open_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.ticket == res.ticket
    assert pos.direction == Direction.BULLISH
    assert pos.volume == 1.0
    assert pos.open_price == 2650.30
    assert pos.commission == 7.0

    acc = await broker.get_account_info()
    assert acc.margin == pytest.approx(2650.30, abs=1e-2)
    assert acc.free_margin == pytest.approx(7312.70, abs=1e-2)


@pytest.mark.asyncio
async def test_ac_1403_pending_limit_orders_and_tick_trigger() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, leverage=100.0)
    await broker.connect()

    start_time = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.30, timestamp=start_time))

    req_limit = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        volume=0.5,
        price=2640.0,
        stop_loss=2635.0,
        take_profit=2650.0,
    )
    res_limit = await broker.send_order(req_limit)
    assert res_limit.success
    assert res_limit.ticket is not None

    orders = await broker.get_open_orders()
    assert len(orders) == 1
    assert orders[0].ticket == res_limit.ticket
    assert orders[0].status == OrderStatus.PENDING

    broker.on_tick(
        _create_sample_tick(price=2645.0, spread=0.30, timestamp=start_time + timedelta(seconds=1))
    )
    assert len(await broker.get_open_orders()) == 1
    assert len(await broker.get_open_positions()) == 0

    broker.on_tick(
        _create_sample_tick(price=2639.50, spread=0.30, timestamp=start_time + timedelta(seconds=2))
    )
    assert len(await broker.get_open_orders()) == 0
    positions = await broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].ticket == res_limit.ticket
    assert positions[0].volume == 0.5
    assert positions[0].open_price == 2640.0


@pytest.mark.asyncio
async def test_ac_1404_close_position_realizes_pnl() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, leverage=100.0, commission_per_lot=7.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=1.0)
    )
    assert res.success and res.ticket is not None
    acc_open = await broker.get_account_info()
    assert acc_open.balance == pytest.approx(9993.0, abs=1e-4)

    broker.on_tick(
        _create_sample_tick(price=2660.0, spread=0.20, timestamp=t0 + timedelta(minutes=5))
    )

    positions = await broker.get_open_positions()
    assert positions[0].floating_pnl == pytest.approx(980.0, abs=1e-4)

    close_res = await broker.close_position(res.ticket)
    assert close_res.success
    assert len(await broker.get_open_positions()) == 0

    acc = await broker.get_account_info()
    assert acc.balance == pytest.approx(10973.0, abs=1e-4)
    assert acc.margin == 0.0
    assert acc.free_margin == pytest.approx(10973.0, abs=1e-4)


@pytest.mark.asyncio
async def test_ac_1405_modify_order_and_stops_levels() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=1.0)
    )
    assert res.success and res.ticket is not None

    mod_res = await broker.modify_order(res.ticket, stop_loss=2648.0, take_profit=2670.0)
    assert mod_res.success

    positions = await broker.get_open_positions()
    assert positions[0].stop_loss == 2648.0
    assert positions[0].take_profit == 2670.0

    invalid_mod = await broker.modify_order(res.ticket, stop_loss=2655.0)
    assert not invalid_mod.success
    assert "Stop Loss inválido" in invalid_mod.error_message


@pytest.mark.asyncio
async def test_ac_1406_order_rejection_insufficient_margin_and_fault_injection() -> None:
    broker = SimBrokerAdapter(initial_balance=1000.0, leverage=10.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    huge_req = OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=10.0)
    res_margin = await broker.send_order(huge_req)
    assert not res_margin.success
    assert res_margin.retcode == 10019
    assert "Margem insuficiente" in res_margin.error_message

    broker.set_fault_injection(rejection_rate=1.0)
    normal_req = OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.01)
    res_fault = await broker.send_order(normal_req)
    assert not res_fault.success
    assert "Rejeição simulada" in res_fault.error_message


@pytest.mark.asyncio
async def test_ac_1407_sl_tp_automatic_hit_on_tick() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY,
            volume=0.1,
            stop_loss=2645.0,
            take_profit=2660.0,
        )
    )
    assert res.success and res.ticket is not None
    assert len(await broker.get_open_positions()) == 1

    broker.on_tick(
        _create_sample_tick(price=2644.50, spread=0.20, timestamp=t0 + timedelta(seconds=10))
    )

    positions = await broker.get_open_positions()
    assert len(positions) == 0

    acc = await broker.get_account_info()
    assert acc.balance < 10000.0


@pytest.mark.asyncio
async def test_sell_market_order_and_close() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, commission_per_lot=5.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL, volume=1.0)
    )
    assert res.success and res.ticket is not None
    assert res.price == 2650.0

    broker.on_tick(
        _create_sample_tick(price=2640.0, spread=0.20, timestamp=t0 + timedelta(minutes=1))
    )
    positions = await broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].direction == Direction.BEARISH
    assert positions[0].floating_pnl == pytest.approx(980.0, abs=1e-4)

    close_res = await broker.close_position(res.ticket)
    assert close_res.success
    assert close_res.price == 2640.20

    acc = await broker.get_account_info()
    assert acc.balance == pytest.approx(10975.0, abs=1e-4)


@pytest.mark.asyncio
async def test_sell_limit_pending_order_and_fill() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_limit = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.SELL_LIMIT,
            volume=0.2,
            price=2660.0,
            stop_loss=2665.0,
            take_profit=2650.0,
        )
    )
    assert res_limit.success and res_limit.ticket is not None
    assert len(await broker.get_open_orders()) == 1

    broker.on_tick(
        _create_sample_tick(price=2660.50, spread=0.20, timestamp=t0 + timedelta(seconds=15))
    )
    assert len(await broker.get_open_orders()) == 0
    positions = await broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].direction == Direction.BEARISH
    assert positions[0].open_price == 2660.0


@pytest.mark.asyncio
async def test_modify_sell_position_and_pending_order() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_pos = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL, volume=0.1)
    )
    assert res_pos.success and res_pos.ticket is not None

    mod_res = await broker.modify_order(res_pos.ticket, stop_loss=2658.0, take_profit=2635.0)
    assert mod_res.success

    bad_mod = await broker.modify_order(res_pos.ticket, stop_loss=2645.0)
    assert not bad_mod.success
    assert "Stop Loss inválido" in bad_mod.error_message

    res_ord = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price=2630.0)
    )
    assert res_ord.success and res_ord.ticket is not None
    mod_ord_res = await broker.modify_order(res_ord.ticket, stop_loss=2620.0, take_profit=2640.0)
    assert mod_ord_res.success
    orders = await broker.get_open_orders()
    assert orders[0].stop_loss == 2620.0
    assert orders[0].take_profit == 2640.0

    missing_mod = await broker.modify_order(99999, stop_loss=2600.0)
    assert not missing_mod.success
    assert "Ticket 99999 não encontrado" in missing_mod.error_message


@pytest.mark.asyncio
async def test_cancel_order_and_close_missing_position() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_ord = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price=2630.0)
    )
    assert res_ord.success and res_ord.ticket is not None
    assert len(await broker.get_open_orders()) == 1

    cancel_res = await broker.cancel_order(res_ord.ticket)
    assert cancel_res.success
    assert len(await broker.get_open_orders()) == 0

    missing_cancel = await broker.cancel_order(99999)
    assert not missing_cancel.success
    assert "Ordem 99999 não encontrada" in missing_cancel.error_message

    missing_close = await broker.close_position(99999)
    assert not missing_close.success
    assert "Posição 99999 não encontrada" in missing_close.error_message


@pytest.mark.asyncio
async def test_send_order_validation_errors() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)

    res_disc = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_disc.success
    assert "Corretora desconectada" in res_disc.error_message

    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_sym = await broker.send_order(
        OrderRequest(symbol="PETR4", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_sym.success
    assert "Símbolo PETR4 não suportado" in res_sym.error_message

    res_vol_low = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.001)
    )
    assert not res_vol_low.success
    assert "Volume" in res_vol_low.error_message

    res_vol_high = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=150.0)
    )
    assert not res_vol_high.success
    assert "Volume" in res_vol_high.error_message

    res_no_price = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price=None)
    )
    assert not res_no_price.success
    assert "Preço obrigatório" in res_no_price.error_message

    res_high_limit = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_LIMIT, volume=0.1, price=2655.0)
    )
    assert not res_high_limit.success
    assert "BUY_LIMIT" in res_high_limit.error_message
    assert res_high_limit.retcode == 10016

    res_low_limit = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL_LIMIT, volume=0.1, price=2645.0)
    )
    assert not res_low_limit.success
    assert "SELL_LIMIT" in res_low_limit.error_message
    assert res_low_limit.retcode == 10016

    res_unsupported = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY_STOP, volume=0.1, price=2660.0)
    )
    assert not res_unsupported.success
    assert "não suportado" in res_unsupported.error_message


@pytest.mark.asyncio
async def test_sell_position_tp_and_sl_hit_on_tick() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.SELL,
            volume=0.1,
            stop_loss=2655.0,
            take_profit=2640.0,
        )
    )
    assert res.success and res.ticket is not None
    assert len(await broker.get_open_positions()) == 1

    broker.on_tick(
        _create_sample_tick(price=2655.0, spread=0.20, timestamp=t0 + timedelta(seconds=10))
    )
    assert len(await broker.get_open_positions()) == 0

    res_tp = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.SELL,
            volume=0.1,
            stop_loss=2660.0,
            take_profit=2640.0,
        )
    )
    assert res_tp.success and res_tp.ticket is not None
    assert len(await broker.get_open_positions()) == 1

    broker.on_tick(
        _create_sample_tick(price=2639.60, spread=0.20, timestamp=t0 + timedelta(seconds=20))
    )
    assert len(await broker.get_open_positions()) == 0


@pytest.mark.asyncio
async def test_sim_latency_and_custom_symbol() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, latency_ms=10.0)
    assert await broker.connect()

    sym = await broker.get_symbol_info("XAUUSD")
    assert sym is not None
    assert sym.name == "XAUUSD"

    acc = await broker.get_account_info()
    assert acc.balance == 10000.0

    await broker.disconnect()
    assert not broker.is_connected()


@pytest.mark.asyncio
async def test_bug_01_commission_accounting_exact_across_all_paths() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0, commission_per_lot=7.0, leverage=100.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_mkt = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=1.0)
    )
    assert res_mkt.success and res_mkt.ticket is not None
    acc1 = await broker.get_account_info()
    assert acc1.balance == pytest.approx(9993.00, abs=1e-4)

    positions = await broker.get_open_positions()
    assert positions[0].floating_pnl == pytest.approx(-20.00, abs=1e-4)
    assert acc1.equity == pytest.approx(9973.00, abs=1e-4)

    await broker.close_position(res_mkt.ticket)
    acc1_closed = await broker.get_account_info()
    assert acc1_closed.balance == pytest.approx(9973.00, abs=1e-4)
    assert acc1_closed.equity == pytest.approx(9973.00, abs=1e-4)

    res_limit = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            price=2645.0,
            volume=1.0,
            stop_loss=2640.0,
            take_profit=2660.0,
        )
    )
    assert res_limit.success and res_limit.ticket is not None
    acc2 = await broker.get_account_info()
    assert acc2.balance == pytest.approx(9973.00, abs=1e-4)

    broker.on_tick(
        _create_sample_tick(price=2644.60, spread=0.20, timestamp=t0 + timedelta(seconds=10))
    )
    assert len(await broker.get_open_orders()) == 0
    assert len(await broker.get_open_positions()) == 1
    acc2_filled = await broker.get_account_info()
    assert acc2_filled.balance == pytest.approx(9966.00, abs=1e-4)

    broker.on_tick(
        _create_sample_tick(price=2639.90, spread=0.20, timestamp=t0 + timedelta(seconds=20))
    )
    assert len(await broker.get_open_positions()) == 0
    acc2_sl = await broker.get_account_info()
    assert acc2_sl.balance == pytest.approx(9456.00, abs=1e-4)


@pytest.mark.asyncio
async def test_bug_02_get_symbol_info_is_pure_and_no_ghost_trades() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()

    sym = await broker.get_symbol_info("EURUSD")
    assert sym is None

    res = await broker.send_order(
        OrderRequest(symbol="EURUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res.success
    assert "não suportado" in res.error_message

    assert "EURUSD" not in broker._symbol_specs


@pytest.mark.asyncio
async def test_bug_03_stops_level_and_invalid_sl_tp_validation() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))
    broker._symbol_specs["XAUUSD"] = SymbolInfo(
        name="XAUUSD",
        bid=2650.0,
        ask=2650.20,
        spread_points=0.20,
        point_size=0.01,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        stops_level_points=20.0,
        freeze_level_points=0.5,
    )

    res_neg_tp = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, take_profit=-5.0)
    )
    assert not res_neg_tp.success
    assert res_neg_tp.retcode == 10016

    res_neg_sl = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, stop_loss=-1.0)
    )
    assert not res_neg_sl.success
    assert res_neg_sl.retcode == 10016

    res_close_tp = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, take_profit=2650.30)
    )
    assert not res_close_tp.success
    assert res_close_tp.retcode == 10016
    assert "stops_level" in res_close_tp.error_message

    res_close_sl = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, stop_loss=2649.90)
    )
    assert not res_close_sl.success
    assert res_close_sl.retcode == 10016
    assert "stops_level" in res_close_sl.error_message

    valid_buy = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert valid_buy.success and valid_buy.ticket is not None

    mod_close = await broker.modify_order(valid_buy.ticket, stop_loss=2649.90)
    assert not mod_close.success
    assert "stops_level" in mod_close.error_message

    mod_ok = await broker.modify_order(valid_buy.ticket, stop_loss=2649.70)
    assert mod_ok.success


@pytest.mark.asyncio
async def test_bug_04_concurrent_orders_race_condition_and_margin_exhaustion() -> None:
    broker = SimBrokerAdapter(initial_balance=8050.0, leverage=100.0, latency_ms=1.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    reqs = [OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=1.0) for _ in range(6)]
    results = await asyncio.gather(*(broker.send_order(r) for r in reqs))

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    assert len(successes) == 3
    assert len(failures) == 3
    for f in failures:
        assert f.retcode == 10019

    tickets = [s.ticket for s in successes]
    assert len(set(tickets)) == 3
    assert len(await broker.get_open_positions()) == 3


@pytest.mark.asyncio
async def test_close_position_partial_volume_rejection() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=1.0)
    )
    assert res.success and res.ticket is not None

    bad_vol = await broker.close_position(res.ticket, volume=0.5)
    assert not bad_vol.success
    assert "Fechamento parcial não suportado" in bad_vol.error_message

    good_close = await broker.close_position(res.ticket, volume=1.0)
    assert good_close.success


@pytest.mark.asyncio
async def test_rejection_leaves_state_uncorrupted() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    initial_balance = broker.balance
    initial_ticket = broker._next_ticket

    res1 = await broker.send_order(
        OrderRequest(symbol="INVALID", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res1.success
    assert broker.balance == initial_balance
    assert broker._next_ticket == initial_ticket
    assert len(await broker.get_open_positions()) == 0
    assert len(await broker.get_open_orders()) == 0

    res2 = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=100.0)
    )
    assert not res2.success
    assert broker.balance == initial_balance
    assert broker._next_ticket == initial_ticket
    assert len(await broker.get_open_positions()) == 0


def test_deterministic_random_seed() -> None:
    adapter1 = SimBrokerAdapter(rejection_rate=0.5, seed=42)
    adapter2 = SimBrokerAdapter(rejection_rate=0.5, seed=42)
    vals1 = [adapter1._rng.random() for _ in range(5)]
    vals2 = [adapter2._rng.random() for _ in range(5)]
    assert vals1 == vals2


@pytest.mark.asyncio
async def test_bug_05_market_orders_sl_tp_evaluated_against_exit_price_and_spread() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.25, timestamp=t0))
    broker._symbol_specs["XAUUSD"] = SymbolInfo(
        name="XAUUSD",
        bid=2650.0,
        ask=2650.25,
        spread_points=0.25,
        point_size=0.01,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        stops_level_points=20.0,
        freeze_level_points=0.5,
    )

    res_buy = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, stop_loss=2650.10)
    )
    assert not res_buy.success
    assert res_buy.retcode == 10016

    res_sell = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL, volume=0.1, stop_loss=2650.15)
    )
    assert not res_sell.success
    assert res_sell.retcode == 10016

    res_sell_tp = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL, volume=0.1, take_profit=2649.90)
    )
    assert not res_sell_tp.success
    assert res_sell_tp.retcode == 10016
    assert "Take Profit inválido ou viola stops_level" in res_sell_tp.error_message

    res_buy_tp = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, take_profit=2650.15)
    )
    assert not res_buy_tp.success
    assert res_buy_tp.retcode == 10016

    res_buy_exact = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1, stop_loss=2649.80)
    )
    assert res_buy_exact.success
    assert res_buy_exact.ticket is not None

    mod_buy_tp = await broker.modify_order(res_buy_exact.ticket, take_profit=2650.35)
    assert not mod_buy_tp.success
    assert mod_buy_tp.retcode == 10016
    assert "Take Profit" in mod_buy_tp.error_message

    res_sell_exact = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.SELL, volume=0.1, stop_loss=2650.45)
    )
    assert res_sell_exact.success
    assert res_sell_exact.ticket is not None


@pytest.mark.asyncio
async def test_margin_check_includes_commission() -> None:
    broker = SimBrokerAdapter(initial_balance=265.50, leverage=100.0, commission_per_lot=7.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.25, timestamp=t0))

    res_fail = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert not res_fail.success
    assert res_fail.retcode == 10019
    assert "comissão" in res_fail.error_message

    broker_zero_comm = SimBrokerAdapter(
        initial_balance=265.50, leverage=100.0, commission_per_lot=0.0
    )
    await broker_zero_comm.connect()
    broker_zero_comm.on_tick(_create_sample_tick(price=2650.0, spread=0.25, timestamp=t0))

    res_ok = await broker_zero_comm.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert res_ok.success
    assert res_ok.ticket is not None


@pytest.mark.asyncio
async def test_order_validation_and_unsupported_branches() -> None:
    broker = SimBrokerAdapter(initial_balance=10000.0)
    await broker.connect()
    t0 = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)
    broker.on_tick(_create_sample_tick(price=2650.0, spread=0.20, timestamp=t0))

    res_step = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.015)
    )
    assert not res_step.success
    assert res_step.retcode == 10014

    res_bl_stops = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price=2640.0,
            stop_loss=2645.0,
        )
    )
    assert not res_bl_stops.success
    assert res_bl_stops.retcode == 10016

    res_limit = await broker.send_order(
        OrderRequest(
            symbol="XAUUSD",
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            price=2640.0,
        )
    )
    assert res_limit.success and res_limit.ticket is not None
    mod_limit_bad = await broker.modify_order(res_limit.ticket, stop_loss=2645.0)
    assert not mod_limit_bad.success
    assert mod_limit_bad.retcode == 10016

    cancel_res = await broker.cancel_order(99999)
    assert not cancel_res.success
    assert cancel_res.retcode == 10017

    close_res = await broker.close_position(99999)
    assert not close_res.success
    assert close_res.retcode == 10017

    res_mkt = await broker.send_order(
        OrderRequest(symbol="XAUUSD", order_type=OrderType.BUY, volume=0.1)
    )
    assert res_mkt.success and res_mkt.ticket is not None
    broker._symbol_specs.pop("XAUUSD", None)
    mod_missing_spec = await broker.modify_order(res_mkt.ticket, stop_loss=2640.0)
    assert not mod_missing_spec.success
    assert mod_missing_spec.retcode == 10013


@pytest.mark.asyncio
async def test_sim_broker_get_candles_default_and_custom() -> None:
    broker = SimBrokerAdapter()
    await broker.connect()

    candles = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
    assert len(candles) == 10
    assert all(c.timeframe == Timeframe.M5 for c in candles)
    assert all(c.time.tzinfo == UTC for c in candles)
    assert all(candles[i].time < candles[i + 1].time for i in range(len(candles) - 1))

    now = datetime(2026, 10, 5, 8, 0, 0, tzinfo=UTC)
    custom = [
        Candle(
            time=now + timedelta(minutes=5 * i),
            open=2650.0 + i,
            high=2652.0 + i,
            low=2649.0 + i,
            close=2651.0 + i,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(5)
    ]
    broker.set_candles(Timeframe.M5, custom, symbol="XAUUSD")
    fetched = await broker.get_candles("XAUUSD", Timeframe.M5, count=3)
    assert len(fetched) == 3
    assert fetched[-1].close == 2655.0


@pytest.mark.asyncio
async def test_sim_broker_get_candles_alignment_and_no_duplicate_candle_events() -> None:
    from xau_bot.runtime.candle_loop import CandleLoop

    broker = SimBrokerAdapter()
    await broker.connect()

    batch1 = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
    batch2 = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)

    assert len(batch1) == len(batch2) == 10
    for c1, c2 in zip(batch1, batch2, strict=True):
        assert c1.time == c2.time
        assert c1.time.second == 0
        assert c1.time.minute % 5 == 0

    candle_loop = CandleLoop(timeframes=(Timeframe.M5,))
    events1 = candle_loop.feed_candles(Timeframe.M5, batch1, has_forming_candle=False)
    events2 = candle_loop.feed_candles(Timeframe.M5, batch2, has_forming_candle=False)

    assert len(events1) == 10
    assert len(events2) == 0


@pytest.mark.asyncio
async def test_sim_broker_advancing_feed_across_m5_boundary_appends_closed_bar() -> None:
    broker = SimBrokerAdapter()
    await broker.connect()

    t0 = datetime(2026, 10, 5, 12, 0, 15, tzinfo=UTC)
    t1 = datetime(2026, 10, 5, 12, 5, 30, tzinfo=UTC)
    t2 = t1 + timedelta(minutes=5 * 600)

    with patch("xau_bot.broker.sim_adapter.datetime") as mock_dt:
        mock_dt.now.return_value = t0
        mock_dt.fromtimestamp.side_effect = datetime.fromtimestamp

        batch1 = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
        assert len(batch1) == 10
        assert batch1[-1].time == datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

        mock_dt.now.return_value = t1
        batch2 = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
        assert len(batch2) == 10
        assert batch2[-1].time == datetime(2026, 10, 5, 12, 5, 0, tzinfo=UTC)
        assert batch2[-2].time == batch1[-1].time
        cached_count = len(broker._candles[("XAUUSD", Timeframe.M5)])
        assert cached_count == 12

        mock_dt.now.return_value = t2
        batch3 = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
        assert len(batch3) == 10
        assert len(broker._candles[("XAUUSD", Timeframe.M5)]) == 500


@pytest.mark.asyncio
async def test_sim_broker_modify_order_clears_take_profit_with_zero() -> None:
    broker = SimBrokerAdapter()
    await broker.connect()

    req = OrderRequest(
        symbol="XAUUSD",
        order_type=OrderType.BUY,
        volume=0.1,
        price=2650.25,
        stop_loss=2640.0,
        take_profit=2670.0,
    )
    result = await broker.send_order(req)
    assert result.success
    assert result.ticket is not None
    ticket = result.ticket

    positions = await broker.get_open_positions()
    assert len(positions) == 1
    assert positions[0].take_profit == 2670.0

    mod_res = await broker.modify_order(ticket=ticket, take_profit=0.0)
    assert mod_res.success

    positions_after = await broker.get_open_positions()
    assert len(positions_after) == 1
    assert positions_after[0].take_profit is None
    assert positions_after[0].stop_loss == 2640.0
