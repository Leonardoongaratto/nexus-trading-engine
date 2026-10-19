import asyncio
import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from xau_bot.broker.models import (
    BrokerOrder,
    BrokerPosition,
    OrderStatus,
    OrderType,
    SymbolInfo,
)
from xau_bot.broker.mt5_mapping import (
    RETCODE_CANCEL,
    RETCODE_CONNECTION,
    RETCODE_DONE,
    RETCODE_INVALID_STOPS,
    RETCODE_INVALID_VOLUME,
    RETCODE_MARKET_CLOSED,
    RETCODE_NO_MONEY,
    RETCODE_PLACED,
    RETCODE_PRICE_CHANGED,
    RETCODE_PRICE_OFF,
    RETCODE_REJECT,
    RETCODE_REQUOTE,
    RETCODE_TIMEOUT,
)
from xau_bot.domain.models import Direction, OrderIntent
from xau_bot.execution.executor import OrderExecutor
from xau_bot.execution.idempotency import InFlightTracker, generate_order_comment
from xau_bot.execution.models import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    RetcodeAction,
)
from xau_bot.execution.reconciler import OrderReconciler
from xau_bot.execution.retcodes import classify_retcode


def _make_intent(
    symbol: str = "XAUUSD",
    direction: Direction = Direction.BULLISH,
    order_type: str = "BUY",
    volume: float = 0.1,
    price: float = 2650.0,
    stop_loss: float = 2640.0,
    take_profit: float = 2670.0,
    magic: int = 123456,
    comment: str = "",
    expiration: datetime | None = None,
) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        direction=direction,
        order_type=order_type,
        volume=volume,
        price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        magic=magic,
        comment=comment,
        expiration=expiration,
    )


def _make_symbol_info(
    name: str = "XAUUSD",
    bid: float = 2650.0,
    ask: float = 2650.25,
) -> SymbolInfo:
    return SymbolInfo(
        name=name,
        bid=bid,
        ask=ask,
        spread_points=0.25,
        point_size=0.01,
        tick_size=0.01,
        tick_value=1.0,
    )


def test_generate_order_comment_length_and_determinism() -> None:
    intent1 = _make_intent(price=2650.0)
    comment1 = generate_order_comment(intent1, prefix="XAU")
    assert len(comment1) <= 31
    assert comment1.replace("_", "").isalnum()
    assert comment1.startswith("XAU_")

    comment1_again = generate_order_comment(intent1, prefix="XAU")
    assert comment1 == comment1_again

    intent2 = _make_intent(price=2651.0)
    comment2 = generate_order_comment(intent2, prefix="XAU")
    assert comment1 != comment2
    assert len(comment2) <= 31


def test_classify_retcode_success_actions() -> None:
    assert classify_retcode(RETCODE_DONE) == RetcodeAction.SUCCESS
    assert classify_retcode(RETCODE_PLACED) == RetcodeAction.SUCCESS
    assert classify_retcode(10010) == RetcodeAction.SUCCESS


def test_classify_retcode_fatal_abort_actions() -> None:
    fatal_codes = [
        RETCODE_REJECT,
        RETCODE_CANCEL,
        RETCODE_NO_MONEY,
        RETCODE_INVALID_VOLUME,
        RETCODE_INVALID_STOPS,
        RETCODE_MARKET_CLOSED,
        10011,
        10013,
        10015,
        10017,
        10022,
        10024,
        10025,
        10026,
        10027,
        10028,
        10029,
        10030,
        99999,
    ]
    for code in fatal_codes:
        assert classify_retcode(code) == RetcodeAction.FATAL_ABORT


def test_classify_retcode_retry_with_reconciliation_actions() -> None:
    retry_codes = [
        RETCODE_REQUOTE,
        RETCODE_TIMEOUT,
        RETCODE_PRICE_CHANGED,
        RETCODE_PRICE_OFF,
        RETCODE_CONNECTION,
    ]
    for code in retry_codes:
        assert classify_retcode(code) == RetcodeAction.RETRY_WITH_RECONCILIATION


@pytest.mark.asyncio
async def test_reconciler_finds_open_position() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    pos = BrokerPosition(
        ticket=101,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        magic=123456,
        comment="XAU_B_1234",
    )
    broker.get_open_positions.return_value = (pos,)
    broker.get_open_orders.return_value = ()

    reconciler = OrderReconciler(broker)
    res = await reconciler.reconcile_order(magic=123456, comment="XAU_B_1234")

    assert res.found is True
    assert res.is_position is True
    assert res.is_order is False
    assert res.ticket == 101
    assert res.position == pos

    res_ticket = await reconciler.reconcile_order(magic=0, comment="", ticket=101)
    assert res_ticket.found is True
    assert res_ticket.ticket == 101


@pytest.mark.asyncio
async def test_reconciler_finds_open_order() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    order = BrokerOrder(
        ticket=202,
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        volume=0.1,
        price=2645.0,
        time_setup=now,
        status=OrderStatus.PENDING,
        magic=123456,
        comment="XAU_BL_9876",
    )
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = (order,)

    reconciler = OrderReconciler(broker)
    res = await reconciler.reconcile_order(magic=123456, comment="XAU_BL_9876")

    assert res.found is True
    assert res.is_position is False
    assert res.is_order is True
    assert res.ticket == 202
    assert res.order == order

    res_ticket = await reconciler.reconcile_order(magic=0, comment="", ticket=202)
    assert res_ticket.found is True
    assert res_ticket.ticket == 202


@pytest.mark.asyncio
async def test_reconciler_returns_not_found_when_no_match() -> None:
    broker = AsyncMock()
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    reconciler = OrderReconciler(broker)
    res = await reconciler.reconcile_order(magic=123456, comment="UNKNOWN_COMMENT")

    assert res.found is False
    assert res.ticket is None
    assert res.position is None
    assert res.order is None


@pytest.mark.asyncio
async def test_executor_aborts_immediately_on_fatal_retcode() -> None:
    broker = AsyncMock()
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    fail_result = AsyncMock()
    fail_result.success = False
    fail_result.retcode = RETCODE_NO_MONEY
    fail_result.error_message = "No money"
    broker.send_order.return_value = fail_result

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=3))
    intent = _make_intent()

    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.ABORTED
    assert result.retcode == RETCODE_NO_MONEY
    assert result.retries_attempted == 1
    assert broker.send_order.call_count == 1
    assert broker.get_open_positions.call_count == 1
    assert broker.get_open_orders.call_count == 1


@pytest.mark.asyncio
async def test_executor_reconciles_and_prevents_double_fill_on_timeout() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    timeout_res = AsyncMock()
    timeout_res.success = False
    timeout_res.retcode = RETCODE_TIMEOUT
    timeout_res.error_message = "Network timeout"
    broker.send_order.return_value = timeout_res

    intent = _make_intent(magic=777, comment="XAU_TEST_TIMEOUT")

    pos = BrokerPosition(
        ticket=555,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        magic=777,
        comment="XAU_TEST_TIMEOUT",
    )
    broker.get_open_positions.side_effect = [(), (pos,)]
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=3, retry_delay_seconds=0.01))
    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.status == ExecutionStatus.RECONCILED
    assert result.reconciled is True
    assert result.ticket == 555
    assert broker.send_order.call_count == 1


@pytest.mark.asyncio
async def test_executor_reconciles_and_retries_when_order_not_found_after_timeout() -> None:
    broker = AsyncMock()

    timeout_res = AsyncMock()
    timeout_res.success = False
    timeout_res.retcode = RETCODE_TIMEOUT
    timeout_res.error_message = "Timeout 1"

    success_res = AsyncMock()
    success_res.success = True
    success_res.ticket = 999
    success_res.price = 2650.0
    success_res.volume = 0.1
    success_res.retcode = RETCODE_DONE

    broker.send_order.side_effect = [timeout_res, success_res]
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=3, retry_delay_seconds=0.01))
    intent = _make_intent()

    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.status == ExecutionStatus.FILLED
    assert result.ticket == 999
    assert result.retries_attempted == 2
    assert broker.send_order.call_count == 2


@pytest.mark.asyncio
async def test_executor_handles_broker_exception_and_retries() -> None:
    broker = AsyncMock()

    success_res = AsyncMock()
    success_res.success = True
    success_res.ticket = 999
    success_res.price = 2650.0
    success_res.volume = 0.1
    success_res.retcode = RETCODE_DONE

    broker.send_order.side_effect = [RuntimeError("Socket dropped"), success_res]
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=3, retry_delay_seconds=0.01))
    intent = _make_intent()

    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.status == ExecutionStatus.FILLED
    assert result.retries_attempted == 2


@pytest.mark.asyncio
async def test_executor_fails_when_max_retries_exceeded() -> None:
    broker = AsyncMock()

    timeout_res = AsyncMock()
    timeout_res.success = False
    timeout_res.retcode = RETCODE_TIMEOUT
    timeout_res.error_message = "Persistent timeout"

    broker.send_order.return_value = timeout_res
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=2, retry_delay_seconds=0.01))
    intent = _make_intent()

    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.FAILED
    assert result.retcode == RETCODE_TIMEOUT
    assert result.retries_attempted == 2
    assert broker.send_order.call_count == 2


@pytest.mark.asyncio
async def test_executor_handles_requote_and_updates_price() -> None:
    broker = AsyncMock()

    requote_res = AsyncMock()
    requote_res.success = False
    requote_res.retcode = RETCODE_REQUOTE
    requote_res.error_message = "Requote"

    done_res = AsyncMock()
    done_res.success = True
    done_res.ticket = 888
    done_res.price = 2650.50
    done_res.volume = 0.1
    done_res.retcode = RETCODE_DONE

    broker.send_order.side_effect = [requote_res, done_res]
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()
    broker.get_symbol_info.return_value = _make_symbol_info(ask=2650.50)

    executor = OrderExecutor(
        broker, ExecutionConfig(max_retries=3, max_slippage_points=100.0, retry_delay_seconds=0.01)
    )
    intent = _make_intent(price=2650.0)

    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.status == ExecutionStatus.FILLED
    assert result.ticket == 888
    assert broker.send_order.call_count == 2
    second_request = broker.send_order.call_args_list[1][0][0]
    assert second_request.price == 2650.50


@pytest.mark.asyncio
async def test_executor_handles_sell_requote_updates_bid_price() -> None:
    broker = AsyncMock()

    requote_res = AsyncMock()
    requote_res.success = False
    requote_res.retcode = RETCODE_REQUOTE
    requote_res.error_message = "Requote"

    done_res = AsyncMock()
    done_res.success = True
    done_res.ticket = 889
    done_res.price = 2649.50
    done_res.volume = 0.1
    done_res.retcode = RETCODE_DONE

    broker.send_order.side_effect = [requote_res, done_res]
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()
    broker.get_symbol_info.return_value = _make_symbol_info(bid=2649.50)

    executor = OrderExecutor(
        broker, ExecutionConfig(max_retries=3, max_slippage_points=100.0, retry_delay_seconds=0.01)
    )
    intent = _make_intent(direction=Direction.BEARISH, order_type="SELL", price=2650.0)

    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.ticket == 889
    second_request = broker.send_order.call_args_list[1][0][0]
    assert second_request.price == 2649.50


@pytest.mark.asyncio
async def test_executor_aborts_requote_if_symbol_info_none() -> None:
    broker = AsyncMock()

    requote_res = AsyncMock()
    requote_res.success = False
    requote_res.retcode = RETCODE_REQUOTE
    requote_res.error_message = "Requote"

    broker.send_order.return_value = requote_res
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()
    broker.get_symbol_info.return_value = None

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=3, retry_delay_seconds=0.01))
    intent = _make_intent(price=2650.0)

    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.FAILED
    assert "unavailable" in result.error_message.lower()


@pytest.mark.asyncio
async def test_executor_aborts_requote_if_slippage_exceeds_threshold() -> None:
    broker = AsyncMock()

    requote_res = AsyncMock()
    requote_res.success = False
    requote_res.retcode = RETCODE_PRICE_CHANGED
    requote_res.error_message = "Price changed"

    broker.send_order.return_value = requote_res
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()
    broker.get_symbol_info.return_value = _make_symbol_info(ask=2665.0)

    executor = OrderExecutor(
        broker, ExecutionConfig(max_retries=3, max_slippage_points=5.0, retry_delay_seconds=0.01)
    )
    intent = _make_intent(price=2650.0)

    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.ABORTED
    assert "slippage" in result.error_message.lower()
    assert broker.send_order.call_count == 1


@pytest.mark.asyncio
async def test_executor_returns_cached_completed_result() -> None:
    broker = AsyncMock()
    tracker = InFlightTracker()
    intent = _make_intent(magic=111, comment="CACHED_TEST")

    cached_res = ExecutionResult(
        success=True,
        status=ExecutionStatus.FILLED,
        ticket=777,
        price=2650.0,
        volume=0.1,
    )
    await tracker.mark_completed("111:CACHED_TEST", cached_res)

    executor = OrderExecutor(broker, tracker=tracker)
    result = await executor.execute_intent(intent)

    assert result == cached_res
    assert broker.send_order.call_count == 0


@pytest.mark.asyncio
async def test_executor_rejects_already_inflight_intent() -> None:
    broker = AsyncMock()
    tracker = InFlightTracker()
    intent = _make_intent(magic=111, comment="INFLIGHT_TEST")

    await tracker.acquire("111:INFLIGHT_TEST")

    executor = OrderExecutor(broker, tracker=tracker)
    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.REJECTED
    assert "already in-flight" in result.error_message


@pytest.mark.asyncio
async def test_inflight_tracker_blocks_concurrent_submissions() -> None:
    tracker = InFlightTracker()
    key = "123456:ORDER_001"

    acquired = await tracker.acquire(key)
    assert acquired is True

    acquired_again = await tracker.acquire(key)
    assert acquired_again is False

    await tracker.release(key)
    assert await tracker.is_in_flight(key) is False

    acquired_after_release = await tracker.acquire(key)
    assert acquired_after_release is True


@pytest.mark.asyncio
async def test_inflight_tracker_caches_and_returns_completed_result() -> None:
    tracker = InFlightTracker()
    key = "123456:ORDER_COMPLETED"

    completed = ExecutionResult(
        success=True,
        status=ExecutionStatus.FILLED,
        ticket=123,
        price=2650.0,
        volume=0.1,
    )
    await tracker.mark_completed(key, completed)

    cached = await tracker.get_completed(key)
    assert cached == completed


@pytest.mark.asyncio
async def test_executor_modify_position_success_and_reconciliation() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    success_res = AsyncMock()
    success_res.success = True
    success_res.ticket = 111
    success_res.retcode = RETCODE_DONE
    broker.modify_order.return_value = success_res

    executor = OrderExecutor(broker)
    res = await executor.modify_position(ticket=111, stop_loss=2645.0, take_profit=2680.0)
    assert res.success is True
    assert res.ticket == 111

    fatal_res = AsyncMock()
    fatal_res.success = False
    fatal_res.retcode = RETCODE_INVALID_STOPS
    fatal_res.error_message = "Invalid stops"
    broker.modify_order.return_value = fatal_res

    res_fatal = await executor.modify_position(ticket=111, stop_loss=2645.0, take_profit=2680.0)
    assert res_fatal.success is False
    assert res_fatal.status == ExecutionStatus.ABORTED

    fail_res = AsyncMock()
    fail_res.success = False
    fail_res.retcode = RETCODE_TIMEOUT
    fail_res.error_message = "Timeout"
    broker.modify_order.return_value = fail_res

    pos_updated = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2645.0,
        take_profit=2680.0,
    )
    broker.get_open_positions.return_value = (pos_updated,)

    res_rec = await executor.modify_position(ticket=111, stop_loss=2645.0, take_profit=2680.0)
    assert res_rec.success is True
    assert res_rec.status == ExecutionStatus.RECONCILED
    assert res_rec.reconciled is True

    pos_not_updated = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2630.0,
        take_profit=2660.0,
    )
    broker.get_open_positions.return_value = (pos_not_updated,)
    res_failed = await executor.modify_position(ticket=111, stop_loss=2645.0, take_profit=2680.0)
    assert res_failed.success is False
    assert res_failed.status == ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_executor_close_position_success_and_reconciliation() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    success_res = AsyncMock()
    success_res.success = True
    success_res.ticket = 111
    success_res.retcode = RETCODE_DONE
    broker.close_position.return_value = success_res

    executor = OrderExecutor(broker)
    res = await executor.close_position(ticket=111, volume=0.1)
    assert res.success is True

    fatal_res = AsyncMock()
    fatal_res.success = False
    fatal_res.retcode = RETCODE_MARKET_CLOSED
    fatal_res.error_message = "Market closed"
    broker.close_position.return_value = fatal_res

    res_fatal = await executor.close_position(ticket=111, volume=0.1)
    assert res_fatal.success is False
    assert res_fatal.status == ExecutionStatus.ABORTED

    fail_res = AsyncMock()
    fail_res.success = False
    fail_res.retcode = RETCODE_TIMEOUT
    fail_res.error_message = "Timeout"
    broker.close_position.return_value = fail_res

    broker.get_open_positions.return_value = ()

    res_rec = await executor.close_position(ticket=111, volume=0.1)
    assert res_rec.success is True
    assert res_rec.status == ExecutionStatus.RECONCILED
    assert res_rec.reconciled is True

    pos_initial = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.15,
        open_price=2650.0,
        open_time=now,
    )
    pos_partial = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.05,
        open_price=2650.0,
        open_time=now,
    )
    broker.get_open_positions.side_effect = [(pos_initial,), (pos_partial,)]
    res_partial = await executor.close_position(ticket=111, volume=0.1)
    assert res_partial.success is True
    assert res_partial.status == ExecutionStatus.RECONCILED

    pos_full = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.2,
        open_price=2650.0,
        open_time=now,
    )
    broker.get_open_positions.side_effect = [(pos_full,), (pos_full,)]
    res_failed = await executor.close_position(ticket=111, volume=0.1)
    assert res_failed.success is False
    assert res_failed.status == ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_executor_cancel_order_success_and_reconciliation() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    success_res = AsyncMock()
    success_res.success = True
    success_res.ticket = 333
    success_res.retcode = RETCODE_DONE
    broker.cancel_order.return_value = success_res

    executor = OrderExecutor(broker)
    res = await executor.cancel_order(ticket=333)
    assert res.success is True
    assert res.ticket == 333

    fatal_res = AsyncMock()
    fatal_res.success = False
    fatal_res.retcode = RETCODE_REJECT
    fatal_res.error_message = "Reject cancel"
    broker.cancel_order.return_value = fatal_res

    res_fatal = await executor.cancel_order(ticket=333)
    assert res_fatal.success is False
    assert res_fatal.status == ExecutionStatus.ABORTED

    timeout_res = AsyncMock()
    timeout_res.success = False
    timeout_res.retcode = RETCODE_TIMEOUT
    timeout_res.error_message = "Timeout"
    broker.cancel_order.return_value = timeout_res

    broker.get_open_orders.return_value = ()
    res_rec = await executor.cancel_order(ticket=333)
    assert res_rec.success is True
    assert res_rec.status == ExecutionStatus.RECONCILED
    assert res_rec.reconciled is True

    existing_order = BrokerOrder(
        ticket=333,
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        volume=0.1,
        price=2640.0,
        time_setup=now,
    )
    broker.get_open_orders.return_value = (existing_order,)
    res_failed = await executor.cancel_order(ticket=333)
    assert res_failed.success is False
    assert res_failed.status == ExecutionStatus.FAILED


@pytest.mark.asyncio
async def test_pre_reconciliation_detects_already_existing_order() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    intent = _make_intent(magic=999, comment="XAU_EXISTING")

    existing_pos = BrokerPosition(
        ticket=789,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        magic=999,
        comment="XAU_EXISTING",
    )
    broker.get_open_positions.return_value = (existing_pos,)
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker)
    result = await executor.execute_intent(intent)

    assert result.success is True
    assert result.status == ExecutionStatus.RECONCILED
    assert result.reconciled is True
    assert result.ticket == 789
    assert broker.send_order.call_count == 0


@pytest.mark.asyncio
async def test_repro_bug_t16_01_partial_close_reconciliation_with_030_position() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    pos_before = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.30,
        open_price=2650.0,
        open_time=now,
    )
    pos_after_success = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.20,
        open_price=2650.0,
        open_time=now,
    )

    fail_res = AsyncMock()
    fail_res.success = False
    fail_res.retcode = RETCODE_TIMEOUT
    fail_res.error_message = "Timeout"
    broker.close_position.return_value = fail_res
    broker.get_open_positions.side_effect = [(pos_before,), (pos_after_success,)]

    executor = OrderExecutor(broker)
    res = await executor.close_position(ticket=111, volume=0.10)
    assert res.success is True
    assert res.status == ExecutionStatus.RECONCILED
    assert res.reconciled is True

    broker.get_open_positions.side_effect = [(pos_before,), (pos_before,)]
    res_failed = await executor.close_position(ticket=111, volume=0.10)
    assert res_failed.success is False
    assert res_failed.status == ExecutionStatus.FAILED


def test_repro_bug_t16_02_distinct_expiration_produces_distinct_comments() -> None:
    now = datetime.now(tz=UTC)
    later = now.replace(minute=(now.minute + 15) % 60)
    intent1 = _make_intent(price=2650.0, comment="", expiration=now)
    intent2 = _make_intent(price=2650.0, comment="", expiration=later)

    c1 = generate_order_comment(intent1)
    c2 = generate_order_comment(intent2)
    assert c1 != c2


@pytest.mark.asyncio
async def test_repro_bug_t16_02_ttl_cache_expires() -> None:
    tracker = InFlightTracker(ttl_seconds=0.05)
    key = "123:TEST"
    res = ExecutionResult(success=True, status=ExecutionStatus.FILLED, ticket=10)
    await tracker.mark_completed(key, res)

    assert await tracker.get_completed(key) == res
    await asyncio.sleep(0.06)
    assert await tracker.get_completed(key) is None


@pytest.mark.asyncio
async def test_reconciler_and_executor_handle_reconciliation_exception_safely() -> None:
    broker = AsyncMock()
    timeout_res = AsyncMock()
    timeout_res.success = False
    timeout_res.retcode = RETCODE_TIMEOUT
    timeout_res.error_message = "Timeout"
    broker.send_order.return_value = timeout_res
    broker.get_open_positions.side_effect = [(), RuntimeError("MT5 broken")]
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=2, retry_delay_seconds=0.01))
    intent = _make_intent()

    result = await executor.execute_intent(intent)
    assert result.success is False
    assert result.status == ExecutionStatus.FAILED
    assert "reconciliation unavailable" in result.error_message.lower()
    assert broker.send_order.call_count == 1


@pytest.mark.asyncio
async def test_modify_close_cancel_handle_broker_exceptions_with_reconciliation() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    executor = OrderExecutor(broker)

    broker.modify_order.side_effect = RuntimeError("Transport error")
    pos_updated = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2645.0,
    )
    broker.get_open_positions.return_value = (pos_updated,)
    res_mod = await executor.modify_position(ticket=111, stop_loss=2645.0)
    assert res_mod.success is True
    assert res_mod.status == ExecutionStatus.RECONCILED

    broker.close_position.side_effect = RuntimeError("Transport error")
    pos_before = BrokerPosition(
        ticket=222,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
    )
    broker.get_open_positions.side_effect = [(pos_before,), ()]
    res_close = await executor.close_position(ticket=222)
    assert res_close.success is True
    assert res_close.status == ExecutionStatus.RECONCILED

    broker.cancel_order.side_effect = RuntimeError("Transport error")
    broker.get_open_orders.return_value = ()
    broker.get_open_positions.side_effect = None
    broker.get_open_positions.return_value = ()
    res_cancel = await executor.cancel_order(ticket=333)
    assert res_cancel.success is True
    assert res_cancel.status == ExecutionStatus.RECONCILED


def test_executor_sanitizes_custom_comment() -> None:
    intent_unsafe = _make_intent(comment="XAU / INVALID $&*")
    intent_safe = _make_intent(comment="XAU_SAFE_123")

    c_unsafe = OrderExecutor._sanitize_or_generate_comment(intent_unsafe, ExecutionConfig())
    c_safe = OrderExecutor._sanitize_or_generate_comment(intent_safe, ExecutionConfig())

    assert re.match(r"^[A-Za-z0-9_]{1,31}$", c_unsafe)
    assert c_safe == "XAU_SAFE_123"


@pytest.mark.asyncio
async def test_reconciled_result_uses_actual_position_and_order_price() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)
    timeout_res = AsyncMock(success=False, retcode=RETCODE_TIMEOUT, error_message="Timeout")
    broker.send_order.return_value = timeout_res

    pos = BrokerPosition(
        ticket=555,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2652.35,
        open_time=now,
        magic=777,
        comment="XAU_PRICE_TEST",
    )
    broker.get_open_positions.side_effect = [(), (pos,)]
    broker.get_open_orders.return_value = ()

    intent = _make_intent(price=2650.0, magic=777, comment="XAU_PRICE_TEST")
    executor = OrderExecutor(broker)
    res = await executor.execute_intent(intent)

    assert res.success is True
    assert res.status == ExecutionStatus.RECONCILED
    assert res.price == 2652.35


@pytest.mark.asyncio
async def test_concurrent_submissions_with_gather() -> None:
    broker = AsyncMock()

    async def slow_send_order(req: object) -> object:
        await asyncio.sleep(0.05)
        res = AsyncMock()
        res.success = True
        res.ticket = 999
        res.price = 2650.0
        res.volume = 0.1
        res.retcode = RETCODE_DONE
        return res

    broker.send_order.side_effect = slow_send_order
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker)
    intent = _make_intent(magic=888, comment="CONCURRENT_KEY")

    results = await asyncio.gather(
        executor.execute_intent(intent),
        executor.execute_intent(intent),
    )

    statuses = [r.status for r in results]
    assert ExecutionStatus.FILLED in statuses
    assert ExecutionStatus.REJECTED in statuses
    assert broker.send_order.call_count == 1


@pytest.mark.asyncio
async def test_executor_requote_on_limit_order_does_not_alter_limit_price() -> None:
    broker = AsyncMock()
    requote_res = AsyncMock(success=False, retcode=RETCODE_REQUOTE, error_message="Requote")
    done_res = AsyncMock(success=True, ticket=777, price=2645.0, volume=0.1, retcode=RETCODE_DONE)
    broker.send_order.side_effect = [requote_res, done_res]
    broker.get_open_positions.return_value = ()
    broker.get_open_orders.return_value = ()
    broker.get_symbol_info.return_value = _make_symbol_info(ask=2655.0)

    executor = OrderExecutor(broker, ExecutionConfig(max_retries=2, retry_delay_seconds=0.01))
    intent = _make_intent(order_type="BUY_LIMIT", price=2645.0)
    result = await executor.execute_intent(intent)

    assert result.success is True
    assert broker.send_order.call_count == 2
    second_request = broker.send_order.call_args_list[1][0][0]
    assert second_request.price == 2645.0


@pytest.mark.asyncio
async def test_repro_bug_t16_03_pre_reconciliation_failure_fails_closed_without_send() -> None:
    broker = AsyncMock()
    broker.get_open_positions.side_effect = RuntimeError("Broken network in pre-reconciliation")
    broker.get_open_orders.return_value = ()

    executor = OrderExecutor(broker)
    intent = _make_intent()

    result = await executor.execute_intent(intent)

    assert result.success is False
    assert result.status == ExecutionStatus.FAILED
    assert "pre-reconciliation unavailable" in result.error_message.lower()
    assert broker.send_order.call_count == 0


@pytest.mark.asyncio
async def test_repro_bug_t16_04_cancel_order_detects_triggered_position() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    fail_res = AsyncMock(success=False, retcode=RETCODE_TIMEOUT, error_message="Timeout")
    broker.cancel_order.return_value = fail_res
    broker.get_open_orders.return_value = ()

    pos_triggered = BrokerPosition(
        ticket=333,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2640.0,
        open_time=now,
    )
    broker.get_open_positions.return_value = (pos_triggered,)

    executor = OrderExecutor(broker)
    res = await executor.cancel_order(ticket=333)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert "triggered" in res.error_message.lower() or "position" in res.error_message.lower()


@pytest.mark.asyncio
async def test_modify_position_handles_10025_no_changes_when_already_applied() -> None:
    broker = AsyncMock()
    now = datetime.now(tz=UTC)

    no_changes_res = AsyncMock(success=False, retcode=10025, error_message="No changes")
    broker.modify_order.return_value = no_changes_res

    pos = BrokerPosition(
        ticket=111,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2645.0,
        take_profit=2680.0,
    )
    broker.get_open_positions.return_value = (pos,)

    executor = OrderExecutor(broker)
    res = await executor.modify_position(ticket=111, stop_loss=2645.0, take_profit=2680.0)

    assert res.success is True
    assert res.status == ExecutionStatus.RECONCILED


@pytest.mark.asyncio
async def test_execute_intent_propagates_deviation_to_order_request() -> None:
    broker = AsyncMock()
    broker.send_order.return_value = AsyncMock(
        success=True, retcode=RETCODE_DONE, ticket=999, price=2650.0, volume=0.1
    )

    cfg = ExecutionConfig(max_slippage_points=10.0)
    executor = OrderExecutor(broker, config=cfg)

    # 1. Intent with explicit deviation
    intent_with_dev = _make_intent(order_type="BUY")
    intent_with_dev = OrderIntent(
        symbol=intent_with_dev.symbol,
        direction=intent_with_dev.direction,
        order_type=intent_with_dev.order_type,
        volume=intent_with_dev.volume,
        price=intent_with_dev.price,
        stop_loss=intent_with_dev.stop_loss,
        take_profit=intent_with_dev.take_profit,
        magic=intent_with_dev.magic,
        comment="test-dev-1",
        deviation=25,
    )
    res1 = await executor.execute_intent(intent_with_dev)
    assert res1.success is True
    req1 = broker.send_order.call_args[0][0]
    assert req1.deviation == 25

    # 2. Intent without explicit deviation falls back to config.max_slippage_points
    intent_no_dev = _make_intent(order_type="SELL", comment="test-dev-2")
    res2 = await executor.execute_intent(intent_no_dev)
    assert res2.success is True
    req2 = broker.send_order.call_args[0][0]
    assert req2.deviation == 10
