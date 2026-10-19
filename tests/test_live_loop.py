"""Automated test suite for Live Trading Engine and LiveTradingLoopRunner."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from xau_bot.advisor.modes import (
    AdvisorDispatcher,
    ExecutionMode,
    ProposalManager,
    ProposalStatus,
)
from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    OrderType,
    SymbolInfo,
)
from xau_bot.broker.ports import BrokerPort
from xau_bot.broker.sim_adapter import SimBrokerAdapter
from xau_bot.domain.models import (
    Candle,
    Direction,
    OrderIntent,
    SetupStatus,
    SetupTriggerType,
    Swing,
    SwingType,
    Timeframe,
    TradeSetup,
)
from xau_bot.execution.executor import OrderExecutor
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import FilterCheckResult, NewsEvent, NewsImpact
from xau_bot.filters.news import NewsFilter
from xau_bot.filters.sessions import SessionFilter, TradeBlockReason
from xau_bot.journal import JournalRepository, LossCategory, LossRootCause, TradeOutcome
from xau_bot.runtime.candle_loop import CandleLoop
from xau_bot.runtime.live_loop import (
    LiveTradingEngine,
    LiveTradingLoopRunner,
)
from xau_bot.runtime.models import CandleCloseEvent
from xau_bot.runtime.watchdog import Watchdog


def _create_candle(
    time: datetime,
    tf: Timeframe = Timeframe.M5,
    open_: float = 2650.0,
    high: float = 2655.0,
    low: float = 2648.0,
    close: float = 2652.0,
    volume: float = 100.0,
) -> Candle:
    return Candle(
        time=time,
        timeframe=tf,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class StubBroker(BrokerPort):
    def __init__(self) -> None:
        self.connected = True
        self.open_positions: list[BrokerPosition] = []
        self.symbol_info = SymbolInfo(
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
        self.account_info = AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=100.0,
            leverage=100.0,
        )
        self.candles: dict[Timeframe, list[Candle]] = {}
        self.modified_orders: list[tuple[int, float | None, float | None]] = []
        self.sent_orders: list[OrderRequest] = []
        self.closed_positions: list[tuple[int, float | None]] = []

    async def connect(self) -> bool:
        self.connected = True
        return True

    async def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def get_account_info(self) -> AccountInfo:
        return self.account_info

    async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        if symbol == "XAUUSD":
            return self.symbol_info
        return None

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> tuple[Candle, ...]:
        stored = self.candles.get(timeframe, [])
        return tuple(stored[-count:])

    async def send_order(self, request: OrderRequest) -> OrderResult:
        self.sent_orders.append(request)
        return OrderResult(success=True, ticket=1001, price=request.price or 2650.0)

    async def modify_order(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        self.modified_orders.append((ticket, stop_loss, take_profit))
        return OrderResult(success=True, ticket=ticket)

    async def cancel_order(self, ticket: int) -> OrderResult:
        return OrderResult(success=True, ticket=ticket)

    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult:
        self.closed_positions.append((ticket, volume))
        return OrderResult(success=True, ticket=ticket)

    async def get_open_positions(self) -> tuple[BrokerPosition, ...]:
        return tuple(self.open_positions)

    async def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        return ()


@pytest.fixture
def base_components() -> dict[str, Any]:
    broker = StubBroker()
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    order_executor = OrderExecutor(broker=broker)
    telegram_service = MagicMock()
    telegram_service.send_proposal = AsyncMock(return_value=MagicMock(message_id=123))
    telegram_service.send_order_filled = AsyncMock()
    telegram_service.send_order_cancelled = AsyncMock()
    telegram_service.send_trade_closed = AsyncMock()
    telegram_service.send_break_even_activated = AsyncMock()
    telegram_service.send_trailing_stop_updated = AsyncMock()
    telegram_service.send_partial_close_executed = AsyncMock()
    telegram_service.send_smart_exit_triggered = AsyncMock()

    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
        executor=order_executor,
        telegram_service=telegram_service,
    )
    candle_loop = CandleLoop()
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        candle_loop=candle_loop,
        telegram_service=telegram_service,
    )
    return {
        "broker": broker,
        "proposal_manager": proposal_manager,
        "dispatcher": dispatcher,
        "candle_loop": candle_loop,
        "engine": engine,
        "executor": order_executor,
        "telegram_service": telegram_service,
    }


@pytest.mark.asyncio
async def test_ac_live_01_broker_port_get_candles(base_components: dict[str, Any]) -> None:
    broker = SimBrokerAdapter()
    await broker.connect()
    candles = await broker.get_candles("XAUUSD", Timeframe.M5, count=10)
    assert len(candles) == 10
    assert all(c.time.tzinfo == UTC for c in candles)
    assert list(candles) == sorted(candles, key=lambda c: c.time)


@pytest.mark.asyncio
async def test_ac_live_02_forming_candle_isolation_in_engine(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    now = datetime(2026, 10, 5, 8, 0, 0, tzinfo=UTC)

    raw_candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(26)]
    forming_candle = raw_candles[-1]

    detect_calls: list[tuple[Candle, ...]] = []

    def _mock_detect(**kwargs: Any) -> list[Any]:
        c_passed = tuple(kwargs.get("candles", ()))
        detect_calls.append(c_passed)
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("xau_bot.runtime.live_loop.detect_setups", _mock_detect)
        engine.candle_loop.register_handler(Timeframe.M5, engine.on_candle_close)
        events = engine.candle_loop.feed_candles(
            Timeframe.M5, raw_candles, has_forming_candle=True, dispatch=True
        )
        await asyncio.sleep(0.05)

    assert len(events) == 25
    assert engine.candle_loop.get_forming_candle(Timeframe.M5) == forming_candle
    assert len(detect_calls) > 0
    for candles_batch in detect_calls:
        assert forming_candle not in candles_batch


@pytest.mark.asyncio
async def test_ac_live_03_m5_candle_close_generates_and_dispatches_proposal(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_test_123",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    props = proposal_manager.get_proposals()
    assert len(props) == 1
    p = props[0]
    assert p.setup_id == "setup_test_123"
    assert p.status == ProposalStatus.PENDING_APPROVAL
    assert p.direction == Direction.BULLISH
    assert p.volume > 0.0


@pytest.mark.asyncio
async def test_ac_live_04_analyst_mode_advisory_proposal_no_orders(
    base_components: dict[str, Any],
) -> None:
    broker: StubBroker = base_components["broker"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.ANALYST,
    )
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
    )
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_analyst_1",
        direction=Direction.BEARISH,
        entry_price=2650.0,
        stop_loss=2655.0,
        take_profit=2635.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )
        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    props = proposal_manager.get_proposals()
    assert len(props) == 1
    assert props[0].status == ProposalStatus.ADVISORY
    assert props[0].mode == ExecutionMode.ANALYST
    assert len(base_components["broker"].sent_orders) == 0


@pytest.mark.asyncio
async def test_ac_live_05_semiauto_mode_pending_approval(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_semiauto_1",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )
        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    props = proposal_manager.get_proposals()
    assert len(props) == 1
    assert props[0].status == ProposalStatus.PENDING_APPROVAL
    assert props[0].expires_at > props[0].created_at
    assert base_components["telegram_service"].send_proposal.called


@pytest.mark.asyncio
async def test_ac_live_06_auto_mode_approves_and_executes_order(
    base_components: dict[str, Any],
) -> None:
    broker: StubBroker = base_components["broker"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    order_executor: OrderExecutor = base_components["executor"]
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.AUTO,
        executor=order_executor,
    )
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
    )
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_auto_1",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )
        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    stats = proposal_manager.get_stats()
    assert stats["dispatched"] == 1
    assert len(broker.sent_orders) == 1


@pytest.mark.asyncio
async def test_ac_live_07_invalidation_by_price(base_components: dict[str, Any]) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    fake_bull = TradeSetup(
        id="setup_inval_bull",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )
    prop_bull = proposal_manager.create_proposal(setup=fake_bull, mode=ExecutionMode.SEMIAUTO)

    fake_bear = TradeSetup(
        id="setup_inval_bear",
        direction=Direction.BEARISH,
        entry_price=2650.0,
        stop_loss=2655.0,
        take_profit=2635.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )
    prop_bear = proposal_manager.create_proposal(setup=fake_bear, mode=ExecutionMode.SEMIAUTO)

    await engine.maintain_active_proposals_and_positions(
        current_price=2644.0, ask_price=2652.0, current_time=now
    )
    p_bull = proposal_manager.get_proposal(prop_bull.id)
    p_bear = proposal_manager.get_proposal(prop_bear.id)
    assert p_bull is not None and p_bull.status == ProposalStatus.CANCELLED
    assert p_bear is not None and p_bear.status == ProposalStatus.PENDING_APPROVAL

    await engine.maintain_active_proposals_and_positions(
        current_price=2650.0, ask_price=2656.0, current_time=now
    )
    p_bear_after = proposal_manager.get_proposal(prop_bear.id)
    assert p_bear_after is not None and p_bear_after.status == ProposalStatus.CANCELLED


@pytest.mark.asyncio
async def test_ac_live_08_break_even_activation(base_components: dict[str, Any]) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=1001,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2640.0,
        take_profit=2680.0,
        current_price=2665.0,
    )
    broker.open_positions.append(open_pos)

    await engine.maintain_active_proposals_and_positions(current_price=2666.0, current_time=now)
    assert len(broker.modified_orders) == 1
    ticket, new_sl, tp = broker.modified_orders[0]
    assert ticket == 1001
    assert new_sl is not None and new_sl >= 2650.0


@pytest.mark.asyncio
async def test_ac_live_09_trailing_stop_update(base_components: dict[str, Any]) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=2002,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now - timedelta(hours=1),
        stop_loss=2650.5,
        take_profit=2700.0,
        current_price=2680.0,
    )
    broker.open_positions.append(open_pos)

    m5_candles = [
        _create_candle(
            time=now - timedelta(minutes=5 * (20 - i)),
            open_=2660.0 + i,
            high=2665.0 + i,
            low=2659.0 + i,
            close=2664.0 + i,
        )
        for i in range(20)
    ]
    engine.candle_loop.feed_candles(Timeframe.M5, m5_candles, has_forming_candle=False)

    fake_swing = Swing(
        time=now - timedelta(minutes=15),
        confirmed_time=now - timedelta(minutes=5),
        type=SwingType.LOW,
        price=2665.0,
        index=17,
    )

    engine.partially_closed_tickets.add(2002)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.find_swings",
            lambda c: [fake_swing],
        )
        await engine.maintain_active_proposals_and_positions(current_price=2680.0, current_time=now)

    assert len(broker.modified_orders) == 1
    ticket, new_sl, tp = broker.modified_orders[0]
    assert ticket == 2002
    assert new_sl is not None and new_sl > 2650.5


@pytest.mark.asyncio
async def test_ac_live_10_live_trading_loop_runner_lifecycle(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    broker.candles[Timeframe.M5] = [_create_candle(time=now)]
    watchdog = MagicMock()
    runner = LiveTradingLoopRunner(engine=engine, watchdog=watchdog, poll_interval_s=0.2)

    assert not runner.is_running
    runner.start()
    assert runner.is_running

    await asyncio.sleep(0.5)
    runner.stop(timeout_s=1.0)
    assert not runner.is_running
    assert watchdog.update_tick.called
    assert watchdog.beat_loop.called


@pytest.mark.asyncio
async def test_engine_skips_when_trading_halted(base_components: dict[str, Any]) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    engine.is_trading_halted_fn = lambda: True
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    detect_called = False

    def _mock_detect(**kwargs: Any) -> list[Any]:
        nonlocal detect_called
        detect_called = True
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("xau_bot.runtime.live_loop.detect_setups", _mock_detect)
        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    assert not detect_called


@pytest.mark.asyncio
async def test_engine_skips_when_composite_filter_blocks(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    mock_filter = MagicMock()
    mock_filter.check.return_value = FilterCheckResult(
        can_trade=False,
        reason=TradeBlockReason.OUTSIDE_KILL_ZONE,
        details="Outside kill zone",
        timestamp=datetime.now(UTC),
    )
    engine.composite_filter = mock_filter
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    detect_called = False

    def _mock_detect(**kwargs: Any) -> list[Any]:
        nonlocal detect_called
        detect_called = True
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("xau_bot.runtime.live_loop.detect_setups", _mock_detect)
        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    assert mock_filter.check.called
    assert not detect_called


@pytest.mark.asyncio
async def test_regression_m1_watchdog_not_updated_on_feed_failure() -> None:
    class FailingBroker(StubBroker):
        async def get_candles(
            self, symbol: str, timeframe: Timeframe, count: int = 100
        ) -> tuple[Candle, ...]:
            raise RuntimeError("Broker connection dropped")

        async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
            raise RuntimeError("Symbol info dropped")

    broker = FailingBroker()
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    order_executor = OrderExecutor(broker=broker)
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
        executor=order_executor,
    )
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        candle_loop=CandleLoop(),
    )
    mock_watchdog = MagicMock()
    runner = LiveTradingLoopRunner(engine=engine, watchdog=mock_watchdog, poll_interval_s=0.01)

    iterations = 0

    def on_beat(now: Any = None) -> None:
        nonlocal iterations
        iterations += 1
        if iterations >= 2:
            runner._stop_event.set()

    mock_watchdog.beat_loop.side_effect = on_beat
    await runner._run_async()

    assert iterations >= 2
    assert not mock_watchdog.update_tick.called


@pytest.mark.asyncio
async def test_regression_a1_watchdog_stops_updating_when_broker_drops_after_boot() -> None:
    class DroppingBroker(StubBroker):
        def __init__(self) -> None:
            super().__init__()
            self.healthy = True

        async def get_candles(
            self, symbol: str, timeframe: Timeframe, count: int = 100
        ) -> tuple[Candle, ...]:
            if not self.healthy:
                raise RuntimeError("Broker connection lost")
            return await super().get_candles(symbol, timeframe, count)

        async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
            if not self.healthy:
                raise RuntimeError("Symbol info connection lost")
            return await super().get_symbol_info(symbol)

    broker = DroppingBroker()
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    order_executor = OrderExecutor(broker=broker)
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
        executor=order_executor,
    )
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        candle_loop=CandleLoop(),
    )
    mock_watchdog = MagicMock()
    runner = LiveTradingLoopRunner(engine=engine, watchdog=mock_watchdog, poll_interval_s=0.01)

    iterations = 0

    def on_beat(now: Any = None) -> None:
        nonlocal iterations
        iterations += 1
        if iterations == 2:
            broker.healthy = False
        elif iterations >= 3:
            runner._stop_event.set()

    mock_watchdog.beat_loop.side_effect = on_beat
    await runner._run_async()

    assert iterations >= 3
    assert mock_watchdog.update_tick.call_count == 1


@pytest.mark.asyncio
async def test_regression_m5_breakeven_uses_initial_stop_loss_not_trailing_stop(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    pos = BrokerPosition(
        ticket=777,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2648.0,
        take_profit=2680.0,
        current_price=2655.0,
    )
    broker.open_positions.append(pos)

    engine.initial_stops[777] = 2640.0

    await engine.maintain_active_proposals_and_positions(current_price=2655.0, current_time=now)
    assert len(broker.modified_orders) == 0

    await engine.maintain_active_proposals_and_positions(current_price=2665.0, current_time=now)
    assert len(broker.modified_orders) == 1
    assert broker.modified_orders[0][0] == 777


@pytest.mark.asyncio
async def test_regression_a1_runner_watchdog_real_health_check() -> None:
    broker = SimBrokerAdapter()
    await broker.connect()
    session_filter = SessionFilter(server_tz="UTC")
    watchdog = Watchdog(broker=broker, session_filter=session_filter)
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=ProposalManager(),
        advisor_dispatcher=AdvisorDispatcher(
            proposal_manager=ProposalManager(),
            mode=ExecutionMode.SEMIAUTO,
            executor=OrderExecutor(broker=broker),
        ),
    )
    runner = LiveTradingLoopRunner(engine=engine, watchdog=watchdog, poll_interval_s=0.05)
    runner.start()
    try:
        await asyncio.sleep(0.2)
        report = watchdog.check_health()
        assert report.is_healthy
        assert report.status_code == 200
        assert not any(r.value == "TICK_STALE" for r in report.reasons)
    finally:
        runner.stop(timeout_s=1.0)


@pytest.mark.asyncio
async def test_maintain_active_proposals_and_positions_recovers_initial_stop_from_proposal(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2680.0,
        magic=123456,
        comment="test",
    )
    prop = engine.proposal_manager.create_proposal(
        setup=TradeSetup(
            id="setup_test_rec",
            direction=Direction.BULLISH,
            entry_price=2650.0,
            stop_loss=2640.0,
            take_profit=2680.0,
            time=now,
            confirmed_time=now,
            risk_reward_ratio=3.0,
            status=SetupStatus.PENDING,
            timeframe=Timeframe.M5,
        ),
        intent=intent,
        mode=ExecutionMode.SEMIAUTO,
    )
    prop.order_ticket = 777

    pos = BrokerPosition(
        ticket=777,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2648.0,
        take_profit=2680.0,
        current_price=2655.0,
    )
    broker.open_positions.append(pos)

    assert 777 not in engine.initial_stops
    await engine.maintain_active_proposals_and_positions(current_price=2655.0, current_time=now)
    assert engine.initial_stops.get(777) == 2640.0


@pytest.mark.asyncio
async def test_live_trading_engine_journal_integration() -> None:
    broker = StubBroker()
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    journal_repo = JournalRepository(":memory:")
    prop_mgr = ProposalManager()
    dispatcher = AdvisorDispatcher(
        proposal_manager=prop_mgr,
        mode=ExecutionMode.SEMIAUTO,
        executor=OrderExecutor(broker=broker),
    )
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=dispatcher,
        journal_repo=journal_repo,
    )

    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2680.0,
        magic=123456,
        comment="test",
        expiration=now + timedelta(hours=1),
    )
    setup = TradeSetup(
        id="setup_jrn_live",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2680.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )
    prop = prop_mgr.create_proposal(
        setup=setup,
        intent=intent,
        mode=ExecutionMode.SEMIAUTO,
        now=now,
    )

    # Entry should be saved in journal_repo
    entry = journal_repo.get_by_proposal_id(prop.id)
    assert entry is not None
    assert entry.proposal_id == prop.id
    assert entry.direction == Direction.BULLISH
    assert entry.status == ProposalStatus.PENDING_APPROVAL

    # Simulate proposal approval & dispatch with ticket 888
    prop.status = ProposalStatus.DISPATCHED
    prop.order_ticket = 888
    prop_mgr._notify_listeners(prop)

    entry = journal_repo.get_by_proposal_id(prop.id)
    assert entry is not None
    assert entry.status == ProposalStatus.DISPATCHED
    assert entry.ticket == 888

    # Simulate position active in broker
    pos = BrokerPosition(
        ticket=888,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2640.0,
        take_profit=2680.0,
        current_price=2660.0,
    )
    broker.open_positions.append(pos)
    await engine.maintain_active_proposals_and_positions(current_price=2660.0, current_time=now)
    assert 888 in engine.initial_stops

    # Position is closed (removed from broker.open_positions)
    broker.open_positions.clear()
    await engine.maintain_active_proposals_and_positions(
        current_price=2675.0,
        current_time=now + timedelta(minutes=30),
    )

    # Journal should now record outcome evaluated!
    closed_entry = journal_repo.get_by_proposal_id(prop.id)
    assert closed_entry is not None
    assert closed_entry.outcome_data is not None
    assert closed_entry.outcome_data.exit_price == 2675.0
    assert closed_entry.outcome == TradeOutcome.WIN
    assert closed_entry.outcome_data.realized_r > 0

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_invalidation_records_metrics(tmp_path: Path) -> None:
    broker = StubBroker()
    prop_mgr = ProposalManager(default_timeout_s=300.0)
    db_file = tmp_path / "test_inv.db"
    journal_repo = JournalRepository(db_path=str(db_file))

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=MagicMock(),
        journal_repo=journal_repo,
    )

    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=123456,
        comment="test",
        expiration=now + timedelta(hours=1),
    )
    setup = TradeSetup(
        id="setup_inv_01",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=2.0,
    )
    prop = prop_mgr.create_proposal(
        setup=setup,
        intent=intent,
        mode=ExecutionMode.SEMIAUTO,
        now=now,
    )

    # Invalidate proposal by price dropping to/below stop loss (2638.0 <= 2640.0)
    await engine.maintain_active_proposals_and_positions(
        current_price=2638.0,
        current_time=now + timedelta(minutes=5),
    )

    entry = journal_repo.get_by_proposal_id(prop.id)
    assert entry is not None
    assert entry.status == ProposalStatus.CANCELLED
    assert entry.outcome == TradeOutcome.INVALIDATED
    assert entry.outcome_data is not None
    assert entry.outcome_data.exit_price == 2640.0
    assert "invalidation" in entry.outcome_data.exit_reason.lower()

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_closed_position_duration_filter(tmp_path: Path) -> None:
    broker = StubBroker()
    prop_mgr = ProposalManager(default_timeout_s=300.0)
    db_file = tmp_path / "test_dur.db"
    journal_repo = JournalRepository(db_path=str(db_file))
    candle_loop = CandleLoop()

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=MagicMock(),
        candle_loop=candle_loop,
        journal_repo=journal_repo,
    )

    t0 = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    # Add historical candles from 2 hours ago that had extreme lows (2600.0)
    old_c = _create_candle(t0, open_=2650.0, high=2655.0, low=2600.0, close=2645.0)
    candle_loop._closed_history[Timeframe.M5].append(old_c)

    # Trade entered at t1 (11:00)
    t1 = datetime(2026, 10, 8, 11, 0, tzinfo=UTC)
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=123456,
        comment="test",
        expiration=t1 + timedelta(hours=1),
    )
    setup = TradeSetup(
        id="setup_filter_candles",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=t1,
        confirmed_time=t1,
        risk_reward_ratio=2.0,
    )
    prop = prop_mgr.create_proposal(
        setup=setup,
        intent=intent,
        mode=ExecutionMode.SEMIAUTO,
        now=t1,
    )
    prop.status = ProposalStatus.DISPATCHED
    prop.order_ticket = 999
    prop_mgr._notify_listeners(prop)

    # Position is open from t1 to t2, price only dipped to 2648.0
    c_trade = _create_candle(t1, open_=2650.0, high=2665.0, low=2648.0, close=2662.0)
    candle_loop._closed_history[Timeframe.M5].append(c_trade)

    pos = BrokerPosition(
        ticket=999,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=t1,
        stop_loss=2640.0,
        take_profit=2670.0,
        current_price=2665.0,
    )
    broker.open_positions.append(pos)
    await engine.maintain_active_proposals_and_positions(current_price=2665.0, current_time=t1)

    # Position closes at t2 with profit
    t2 = t1 + timedelta(minutes=15)
    broker.open_positions.clear()
    await engine.maintain_active_proposals_and_positions(current_price=2668.0, current_time=t2)

    entry = journal_repo.get_by_proposal_id(prop.id)
    assert entry is not None
    assert entry.outcome_data is not None
    # If old candle from t0 were included, MAE would be (2650 - 2600)/10 = 5.0R!
    # Duration filtering ensures MAE only reflects candles during position (low 2648 -> 0.2R)
    assert entry.outcome_data.mae_r <= 0.5

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_counter_bias_loss_classification(tmp_path: Path) -> None:
    broker = StubBroker()
    prop_mgr = ProposalManager(default_timeout_s=300.0)
    db_file = tmp_path / "test_cb_loss.db"
    journal_repo = JournalRepository(db_path=str(db_file))

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=MagicMock(),
        journal_repo=journal_repo,
    )
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=123456,
        comment="test",
        expiration=now + timedelta(hours=1),
    )
    # Trade setup has HTF_COUNTER confluence without touching private attributes
    setup = TradeSetup(
        id="setup_cb_live",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=2.0,
        confluences=("HTF_COUNTER",),
    )
    prop = prop_mgr.create_proposal(
        setup=setup,
        intent=intent,
        mode=ExecutionMode.SEMIAUTO,
        now=now,
    )
    prop.status = ProposalStatus.DISPATCHED
    prop.order_ticket = 777
    prop_mgr._notify_listeners(prop)

    pos = BrokerPosition(
        ticket=777,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2640.0,
        take_profit=2670.0,
        current_price=2648.0,
    )
    broker.open_positions.append(pos)
    await engine.maintain_active_proposals_and_positions(current_price=2648.0, current_time=now)

    # Position hits SL at 2639.0
    broker.open_positions.clear()
    await engine.maintain_active_proposals_and_positions(
        current_price=2639.0,
        current_time=now + timedelta(minutes=10),
    )

    entry = journal_repo.get_by_proposal_id(prop.id)
    assert entry is not None
    assert entry.outcome == TradeOutcome.LOSS
    assert entry.loss_cause == LossRootCause.COUNTER_BIAS
    assert entry.loss_category == LossCategory.BAD_DECISION

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_on_candle_close_captures_market_telemetry(
    tmp_path: Path,
) -> None:
    broker = StubBroker()
    broker.symbol_info = SymbolInfo(
        name="XAUUSD",
        bid=2650.0,
        ask=2650.35,
        spread_points=35.0,
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

    prop_mgr = ProposalManager()
    journal_repo = JournalRepository(db_path=tmp_path / "telemetry_test.db")
    candle_loop = CandleLoop(timeframes=[Timeframe.M5, Timeframe.H4])

    t0 = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    # Populate H4 candles indicating BEARISH bias
    h4_c0 = _create_candle(t0, tf=Timeframe.H4, open_=2680.0, high=2685.0, low=2650.0, close=2655.0)
    h4_c1 = _create_candle(
        t0 + timedelta(hours=4),
        tf=Timeframe.H4,
        open_=2655.0,
        high=2660.0,
        low=2630.0,
        close=2635.0,
    )
    candle_loop._closed_history[Timeframe.H4] = [h4_c0, h4_c1]

    # Populate 25 M5 candles to allow ATR calculation
    m5_candles: list[Candle] = []
    for i in range(25):
        m5_candles.append(
            _create_candle(
                t0 + timedelta(minutes=5 * i),
                tf=Timeframe.M5,
                open_=2640.0 + i * 0.1,
                high=2642.0 + i * 0.1,
                low=2639.0 + i * 0.1,
                close=2641.0 + i * 0.1,
            )
        )
    candle_loop._closed_history[Timeframe.M5] = m5_candles

    news_event = NewsEvent(
        title="FOMC Statement",
        currency="USD",
        impact=NewsImpact.HIGH,
        time=m5_candles[-1].time + timedelta(minutes=50),
    )
    news_filter = NewsFilter(events=[news_event])
    composite_filter = CompositeTradeFilter(news_filter=news_filter)

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=MagicMock(),
        candle_loop=candle_loop,
        composite_filter=composite_filter,
        journal_repo=journal_repo,
    )

    event = CandleCloseEvent(
        timeframe=Timeframe.M5,
        candle=m5_candles[-1],
        closed_series=tuple(m5_candles),
        forming_candle=None,
        closed_at=m5_candles[-1].time + timedelta(minutes=5),
    )

    await engine.on_candle_close(event)

    # Telemetry should be captured dynamically without private manipulation
    assert engine._last_spread_price == pytest.approx(0.35)
    assert engine._last_htf_bias == "BEARISH"
    assert engine._last_news_distance_min == pytest.approx(45.0)

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_evaluate_unexecuted_proposals_lifecycle(tmp_path: Path) -> None:
    broker = StubBroker()
    prop_mgr = ProposalManager()
    journal_repo = JournalRepository(db_path=tmp_path / "unexec_test.db")
    candle_loop = CandleLoop(timeframes=[Timeframe.M5])

    t0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=prop_mgr,
        advisor_dispatcher=MagicMock(),
        candle_loop=candle_loop,
        journal_repo=journal_repo,
    )

    # 1. Proposal that expires (unhit)
    intent1 = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2600.0,
        stop_loss=2590.0,
        take_profit=2620.0,
        magic=1,
        comment="p1",
        expiration=t0 + timedelta(hours=1),
    )
    s1 = TradeSetup(
        id="s1",
        direction=Direction.BULLISH,
        entry_price=2600.0,
        stop_loss=2590.0,
        take_profit=2620.0,
        time=t0,
        confirmed_time=t0,
        risk_reward_ratio=2.0,
    )
    prop_expired_unhit = prop_mgr.create_proposal(
        setup=s1, intent=intent1, mode=ExecutionMode.SEMIAUTO, now=t0
    )
    prop_expired_unhit.status = ProposalStatus.EXPIRED
    prop_mgr._notify_listeners(prop_expired_unhit)

    # 2. Proposal that was rejected and later hypothetical WIN
    intent2 = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=2,
        comment="p2",
        expiration=t0 + timedelta(hours=1),
    )
    s2 = TradeSetup(
        id="s2",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=t0,
        confirmed_time=t0,
        risk_reward_ratio=2.0,
    )
    prop_rejected_win = prop_mgr.create_proposal(
        setup=s2, intent=intent2, mode=ExecutionMode.SEMIAUTO, now=t0
    )
    prop_rejected_win.status = ProposalStatus.REJECTED
    prop_mgr._notify_listeners(prop_rejected_win)

    # 3. Active DISPATCHED proposal
    intent3 = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=3,
        comment="p3",
        expiration=t0 + timedelta(hours=1),
    )
    s3 = TradeSetup(
        id="s3",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=t0,
        confirmed_time=t0,
        risk_reward_ratio=2.0,
    )
    prop_dispatched = prop_mgr.create_proposal(
        setup=s3, intent=intent3, mode=ExecutionMode.SEMIAUTO, now=t0
    )
    prop_dispatched.status = ProposalStatus.DISPATCHED
    prop_mgr._notify_listeners(prop_dispatched)

    # 4. Active PENDING_APPROVAL proposal
    intent4 = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.1,
        price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        magic=4,
        comment="p4",
        expiration=t0 + timedelta(hours=1),
    )
    s4 = TradeSetup(
        id="s4",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=t0,
        confirmed_time=t0,
        risk_reward_ratio=2.0,
    )
    prop_pending = prop_mgr.create_proposal(
        setup=s4, intent=intent4, mode=ExecutionMode.SEMIAUTO, now=t0
    )

    # Feed M5 candles: touches entry (2650.0) and reaches TP (2670.0)
    m5_candles = [
        _create_candle(
            t0 + timedelta(minutes=5), open_=2655.0, high=2656.0, low=2649.0, close=2652.0
        ),
        _create_candle(
            t0 + timedelta(minutes=10), open_=2652.0, high=2665.0, low=2651.0, close=2664.0
        ),
        _create_candle(
            t0 + timedelta(minutes=15), open_=2664.0, high=2672.0, low=2663.0, close=2671.0
        ),
    ]
    candle_loop._closed_history[Timeframe.M5] = m5_candles

    evaluated = engine.evaluate_unexecuted_proposals()
    assert len(evaluated) == 1

    # Expired unhit proposal should preserve TradeOutcome.EXPIRED
    e1 = journal_repo.get_by_proposal_id(prop_expired_unhit.id)
    assert e1 is not None
    assert e1.outcome == TradeOutcome.EXPIRED

    # Rejected proposal should be evaluated as hypothetical WIN
    e2 = journal_repo.get_by_proposal_id(prop_rejected_win.id)
    assert e2 is not None
    assert e2.outcome == TradeOutcome.WIN
    assert e2.outcome_data is not None
    assert e2.outcome_data.executed is False

    # Dispatched and pending proposals must NOT be touched
    e3 = journal_repo.get_by_proposal_id(prop_dispatched.id)
    assert e3 is not None
    assert e3.outcome == TradeOutcome.UNEXECUTED
    assert e3.status == ProposalStatus.DISPATCHED

    e4 = journal_repo.get_by_proposal_id(prop_pending.id)
    assert e4 is not None
    assert e4.outcome == TradeOutcome.UNEXECUTED
    assert e4.status == ProposalStatus.PENDING_APPROVAL

    journal_repo.close()


@pytest.mark.asyncio
async def test_live_loop_partial_close_and_breakeven_execution(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=3001,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.20,
        open_price=2650.0,
        open_time=now - timedelta(minutes=30),
        stop_loss=2640.0,  # 10.0 pts risk
        take_profit=2700.0,
        current_price=2660.0,
    )
    broker.open_positions.append(open_pos)

    # Price hits 2660.0 (+10.0 pts = 1.0R trigger)
    await engine.maintain_active_proposals_and_positions(current_price=2660.0, current_time=now)

    # 1. 50% partial close should be dispatched (0.10 lots)
    assert len(broker.closed_positions) == 1
    ticket, vol_closed = broker.closed_positions[0]
    assert ticket == 3001
    assert vol_closed == 0.10

    # 2. SL should be modified to Break-Even (>= entry_price)
    assert len(broker.modified_orders) == 1
    mod_ticket, new_sl, tp = broker.modified_orders[0]
    assert mod_ticket == 3001
    assert new_sl is not None and new_sl >= 2650.0

    # 3. Position ticket marked as partially closed
    assert 3001 in engine.partially_closed_tickets

    # 4. Next tick at higher price does not trigger duplicate partial close
    await engine.maintain_active_proposals_and_positions(
        current_price=2665.0, current_time=now + timedelta(minutes=1)
    )
    assert len(broker.closed_positions) == 1


@pytest.mark.asyncio
async def test_live_loop_smart_exit_on_opposite_choch(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=3002,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        open_price=2650.0,
        open_time=now - timedelta(minutes=45),
        stop_loss=2640.0,  # 10.0 pts risk
        take_profit=2700.0,
        current_price=2656.0,  # +6.0 pts = 0.6R (in profit > 0.5R)
    )
    broker.open_positions.append(open_pos)

    engine.partially_closed_tickets.add(3002)

    # Confirmed swing low formed after entry
    swing_low = Swing(
        time=now - timedelta(minutes=20),
        confirmed_time=now - timedelta(minutes=10),
        type=SwingType.LOW,
        price=2655.0,
        index=15,
    )

    # Candle that closes below the swing low (ChoCH bearish)
    c_rev = _create_candle(
        time=now - timedelta(minutes=5),
        open_=2656.0,
        high=2656.5,
        low=2653.0,
        close=2653.5,  # below 2655.0
    )
    m5_candles = [
        _create_candle(
            time=now - timedelta(minutes=5 * (10 - i)),
            open_=2650.0 + i,
            high=2655.0 + i,
            low=2649.0 + i,
            close=2654.0 + i,
        )
        for i in range(9)
    ] + [c_rev]
    engine.candle_loop._closed_history[Timeframe.M5] = m5_candles

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("xau_bot.runtime.live_loop.find_swings", lambda c: [swing_low])
        await engine.maintain_active_proposals_and_positions(current_price=2656.0, current_time=now)

    # Smart Exit should close the position completely
    assert len(broker.closed_positions) == 1
    exit_ticket, exit_vol = broker.closed_positions[0]
    assert exit_ticket == 3002
    assert exit_vol is None  # full position closed
    assert 3002 in engine.smart_exited_tickets


@pytest.mark.asyncio
async def test_live_engine_multi_timeframe_initialization() -> None:
    broker = StubBroker()
    pm = ProposalManager()
    dispatcher = AdvisorDispatcher(proposal_manager=pm, mode=ExecutionMode.AUTO)

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=pm,
        advisor_dispatcher=dispatcher,
        trigger_timeframe=(Timeframe.M1, Timeframe.M5),
    )

    assert engine.trigger_timeframes == (Timeframe.M1, Timeframe.M5)
    assert engine.trigger_timeframe == Timeframe.M1
    assert Timeframe.M1 in engine.timeframes
    assert Timeframe.M5 in engine.timeframes


@pytest.mark.asyncio
async def test_live_engine_m1_and_m5_distinct_magic_and_comment() -> None:
    broker = StubBroker()
    pm = ProposalManager()
    dispatcher = AdvisorDispatcher(proposal_manager=pm, mode=ExecutionMode.AUTO)

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=pm,
        advisor_dispatcher=dispatcher,
        trigger_timeframe=(Timeframe.M1, Timeframe.M5),
        magic=1000,
    )

    now = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
    m1_candles = [
        _create_candle(time=now - timedelta(minutes=25 - i), tf=Timeframe.M1) for i in range(25)
    ]
    m15_candles = [
        _create_candle(time=now - timedelta(minutes=15 * (5 - i)), tf=Timeframe.M15)
        for i in range(5)
    ]
    engine.candle_loop._closed_history[Timeframe.M1] = m1_candles
    engine.candle_loop._closed_history[Timeframe.M15] = m15_candles

    setup = TradeSetup(
        id="setup_m1_test",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        risk_reward_ratio=2.0,
        confluences=("SWEEP",),
        valid_until=now + timedelta(minutes=10),
        status=SetupStatus.PENDING,
        trigger_type=SetupTriggerType.FVG_CE,
        time=now,
        confirmed_time=now,
        timeframe=Timeframe.M1,
    )

    event_m1 = CandleCloseEvent(
        timeframe=Timeframe.M1,
        candle=m1_candles[-1],
        closed_series=tuple(m1_candles),
        forming_candle=None,
        closed_at=now,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("xau_bot.runtime.live_loop.detect_setups", lambda **kwargs: [setup])
        mp.setattr("xau_bot.runtime.live_loop.get_active_setups", lambda setups, **kwargs: setups)
        await engine.on_candle_close(event_m1)

    # Verifica se a proposta gerada em M1 recebeu Magic + 1 e comentário SMC-M1-
    all_proposals = pm.get_all_proposals()
    assert len(all_proposals) == 1
    prop = all_proposals[0]
    assert prop.intent is not None
    assert prop.intent.magic == 1001
    assert prop.intent.comment.startswith("SMC-M1-")


@pytest.mark.asyncio
async def test_live_runner_registers_all_trigger_timeframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = StubBroker()
    pm = ProposalManager()
    dispatcher = AdvisorDispatcher(proposal_manager=pm, mode=ExecutionMode.AUTO)

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=pm,
        advisor_dispatcher=dispatcher,
        trigger_timeframe=(Timeframe.M1, Timeframe.M5),
    )

    runner = LiveTradingLoopRunner(engine=engine, poll_interval_s=0.1)
    registered_tfs: list[Timeframe] = []

    def mock_register(tf: Timeframe, handler: Any) -> None:
        registered_tfs.append(tf)

    monkeypatch.setattr(engine.candle_loop, "register_handler", mock_register)

    # Inicia e para rapidamente o runner worker async
    worker_task = asyncio.create_task(runner._run_async())
    runner._stop_event.set()
    await asyncio.sleep(0.05)
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task

    assert Timeframe.M1 in registered_tfs
    assert Timeframe.M5 in registered_tfs


@pytest.mark.asyncio
async def test_live_engine_cancels_expired_and_invalidated_pending_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = StubBroker()
    pm = ProposalManager()
    dispatcher = AdvisorDispatcher(proposal_manager=pm, mode=ExecutionMode.AUTO)
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=pm,
        advisor_dispatcher=dispatcher,
        symbol="XAUUSD",
    )
    now = datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC)

    setup_exp = TradeSetup(
        id="setup_exp",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=2.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )
    prop_expired = pm.create_proposal(
        setup=setup_exp,
        mode=ExecutionMode.AUTO,
        volume=0.1,
        risk_usd=100.0,
        risk_pct=1.0,
        timeout_s=60.0,
    )
    prop_expired.status = ProposalStatus.DISPATCHED
    prop_expired.order_ticket = 9001
    prop_expired.expires_at = now - timedelta(seconds=10)

    setup_tp = TradeSetup(
        id="setup_tp",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=2.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )
    prop_tp_hit = pm.create_proposal(
        setup=setup_tp,
        mode=ExecutionMode.AUTO,
        volume=0.1,
        risk_usd=100.0,
        risk_pct=1.0,
        timeout_s=600.0,
    )
    prop_tp_hit.status = ProposalStatus.DISPATCHED
    prop_tp_hit.order_ticket = 9002
    prop_tp_hit.expires_at = now + timedelta(minutes=10)

    cancelled_tickets: list[int] = []

    async def mock_get_open_orders() -> tuple[BrokerOrder, ...]:
        return (
            BrokerOrder(
                ticket=9001,
                symbol="XAUUSD",
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                price=2640.0,
                time_setup=now - timedelta(minutes=5),
            ),
            BrokerOrder(
                ticket=9002,
                symbol="XAUUSD",
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                price=2640.0,
                time_setup=now,
            ),
        )

    async def mock_cancel_order(ticket: int) -> OrderResult:
        cancelled_tickets.append(ticket)
        return OrderResult(success=True, ticket=ticket)

    monkeypatch.setattr(broker, "get_open_orders", mock_get_open_orders)
    monkeypatch.setattr(broker, "cancel_order", mock_cancel_order)

    await engine.maintain_active_proposals_and_positions(
        current_price=2665.0,
        ask_price=2665.25,
        current_time=now,
    )

    assert 9001 in cancelled_tickets
    assert 9002 in cancelled_tickets
    assert prop_expired.status == ProposalStatus.CANCELLED
    assert prop_tp_hit.status == ProposalStatus.CANCELLED


@pytest.mark.asyncio
async def test_maintain_active_proposals_notifies_telegram_on_fill_and_close(
    base_components: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    tg_service = base_components["telegram_service"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=9901,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        open_time=now,
        stop_loss=2640.0,
        take_profit=2670.0,
        current_price=2655.0,
    )

    async def mock_get_open_positions_open() -> tuple[BrokerPosition, ...]:
        return (open_pos,)

    monkeypatch.setattr(broker, "get_open_positions", mock_get_open_positions_open)

    # 1. Posição aberta pela primeira vez -> deve notificar send_order_filled
    await engine.maintain_active_proposals_and_positions(
        current_price=2655.0,
        ask_price=2655.25,
        current_time=now,
    )

    assert tg_service.send_order_filled.called
    assert tg_service.send_order_filled.call_args[1]["ticket"] == 9901
    assert tg_service.send_order_filled.call_args[1]["direction"].upper() == "BULLISH"

    # Segunda chamada com a mesma posição aberta não deve duplicar o alerta de fill
    tg_service.send_order_filled.reset_mock()
    await engine.maintain_active_proposals_and_positions(
        current_price=2656.0,
        ask_price=2656.25,
        current_time=now + timedelta(seconds=2),
    )
    assert not tg_service.send_order_filled.called

    # 2. Posição fechou no broker -> deve notificar send_trade_closed com WIN
    async def mock_get_open_positions_closed() -> tuple[BrokerPosition, ...]:
        return ()

    monkeypatch.setattr(broker, "get_open_positions", mock_get_open_positions_closed)

    await engine.maintain_active_proposals_and_positions(
        current_price=2670.0,
        ask_price=2670.25,
        current_time=now + timedelta(seconds=10),
    )

    assert tg_service.send_trade_closed.called
    assert tg_service.send_trade_closed.call_args[1]["ticket"] == 9901
    assert tg_service.send_trade_closed.call_args[1]["profit_usd"] > 0


@pytest.mark.asyncio
async def test_live_trading_loop_runner_multi_engine_lifecycle(
    base_components: dict[str, Any],
) -> None:
    engine1: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
    broker.candles[Timeframe.M5] = [_create_candle(time=now)]

    engine2 = LiveTradingEngine(
        broker=broker,
        proposal_manager=base_components["proposal_manager"],
        advisor_dispatcher=base_components["dispatcher"],
        symbol="BTCUSD",
        magic=123450,
    )

    watchdog = MagicMock()
    runner = LiveTradingLoopRunner(
        engines=[engine1, engine2],
        watchdog=watchdog,
        poll_interval_s=0.1,
    )

    assert runner.engine == engine1
    assert len(runner.engines) == 2
    assert not runner.is_running
    runner.start()
    assert runner.is_running

    await asyncio.sleep(0.3)
    runner.stop(timeout_s=1.0)
    assert not runner.is_running
    assert watchdog.beat_loop.called


@pytest.mark.asyncio
async def test_runner_unlimited_tp_after_partial_close(
    base_components: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    broker: StubBroker = base_components["broker"]
    tg_service = base_components["telegram_service"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    open_pos = BrokerPosition(
        ticket=7701,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        open_price=2650.0,
        open_time=now,
        stop_loss=2640.0,
        take_profit=2670.0,
        current_price=2660.0,
    )

    current_open_positions: list[BrokerPosition] = [open_pos]

    async def mock_get_open_positions() -> tuple[BrokerPosition, ...]:
        return tuple(current_open_positions)

    close_calls: list[tuple[int, float | None]] = []

    async def mock_close_position(ticket: int, volume: float | None = None) -> OrderResult:
        close_calls.append((ticket, volume))
        if volume is not None and current_open_positions:
            rem = current_open_positions[0].volume - volume
            current_open_positions[0] = BrokerPosition(
                ticket=ticket,
                symbol="XAUUSD",
                direction=Direction.BULLISH,
                volume=rem,
                open_price=2650.0,
                open_time=now,
                stop_loss=2640.0,
                take_profit=0.0,
                current_price=2660.0,
            )
        return OrderResult(success=True, ticket=ticket)

    modify_calls: list[dict[str, Any]] = []

    async def mock_modify_order(
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        modify_calls.append({"ticket": ticket, "stop_loss": stop_loss, "take_profit": take_profit})
        if current_open_positions:
            prev = current_open_positions[0]
            current_open_positions[0] = BrokerPosition(
                ticket=ticket,
                symbol="XAUUSD",
                direction=Direction.BULLISH,
                volume=prev.volume,
                open_price=2650.0,
                open_time=now,
                stop_loss=stop_loss if stop_loss is not None else prev.stop_loss,
                take_profit=take_profit if take_profit is not None else prev.take_profit,
                current_price=2660.0,
            )
        return OrderResult(success=True, ticket=ticket)

    monkeypatch.setattr(broker, "get_open_positions", mock_get_open_positions)
    monkeypatch.setattr(broker, "close_position", mock_close_position)
    monkeypatch.setattr(broker, "modify_order", mock_modify_order)

    # 1. Price is at 2660 (1R reached: 2650 + (2650-2640)*1.0 = 2660)
    await engine.maintain_active_proposals_and_positions(
        current_price=2660.0,
        ask_price=2660.25,
        current_time=now + timedelta(seconds=10),
    )

    # Partial close must have executed (0.05 closed)
    assert len(close_calls) == 1
    assert close_calls[0] == (7701, 0.05)
    assert 7701 in engine.partially_closed_tickets

    # modify_order must have been called with take_profit=0.0 (cleared!)
    assert len(modify_calls) == 1
    assert modify_calls[0]["ticket"] == 7701
    assert modify_calls[0]["take_profit"] == 0.0
    assert modify_calls[0]["stop_loss"] >= 2650.0  # Break-even
    assert engine.initial_tps[7701] is None

    # Telegram message must indicate runner active
    assert tg_service.send_partial_close_executed.called
    tg_kwargs = tg_service.send_partial_close_executed.call_args[1]
    assert tg_kwargs["ticket"] == 7701
    assert tg_kwargs["runner_active"] is True

    # 2. Subsequent call: Trailing Stop advances with take_profit=0.0
    engine.candle_loop.feed_candles(
        Timeframe.M5,
        [
            _create_candle(
                time=now + timedelta(minutes=5 * i),
                open_=2650.0 + i,
                high=2660.0 + i,
                low=2645.0 + i,
                close=2655.0 + i,
            )
            for i in range(20)
        ],
        has_forming_candle=False,
    )

    await engine.maintain_active_proposals_and_positions(
        current_price=2680.0,
        ask_price=2680.25,
        current_time=now + timedelta(minutes=20),
    )

    # If trailing stop modified order, take_profit must remain 0.0
    if len(modify_calls) > 1:
        assert modify_calls[-1]["take_profit"] == 0.0

    # 3. Position finally closes in broker
    current_open_positions.clear()
    await engine.maintain_active_proposals_and_positions(
        current_price=2675.0,
        ask_price=2675.25,
        current_time=now + timedelta(minutes=30),
    )

    assert tg_service.send_trade_closed.called
    assert 7701 not in engine.partially_closed_tickets


@pytest.mark.asyncio
async def test_ac_live_bullish_market_order_when_price_below_entry(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    broker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_bull_discount",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=2.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    # Current market ask is 2648.0 (cheaper than entry 2650.0, deep discount)
    broker.symbol_info = SymbolInfo(
        name="XAUUSD",
        bid=2647.70,
        ask=2648.00,
        point_size=0.01,
        spread_points=30.0,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        stops_level_points=1.0,
        freeze_level_points=0.5,
        tick_size=0.01,
        tick_value=1.0,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    props = proposal_manager.get_proposals()
    assert len(props) == 1
    p = props[0]
    assert p.setup_id == "setup_bull_discount"
    assert p.intent is not None
    assert p.intent.order_type == "BUY"
    assert p.intent.price == 2648.00


@pytest.mark.asyncio
async def test_ac_live_bearish_market_order_when_price_above_entry(
    base_components: dict[str, Any],
) -> None:
    engine: LiveTradingEngine = base_components["engine"]
    proposal_manager: ProposalManager = base_components["proposal_manager"]
    broker = base_components["broker"]
    now = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)

    candles = [_create_candle(time=now + timedelta(minutes=5 * i)) for i in range(25)]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_bear_premium",
        direction=Direction.BEARISH,
        entry_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=2.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    # Current market bid is 2652.0 (higher than entry 2650.0, higher premium)
    broker.symbol_info = SymbolInfo(
        name="XAUUSD",
        bid=2652.00,
        ask=2652.30,
        point_size=0.01,
        spread_points=30.0,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        stops_level_points=1.0,
        freeze_level_points=0.5,
        tick_size=0.01,
        tick_value=1.0,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    props = proposal_manager.get_proposals()
    assert len(props) == 1
    p = props[0]
    assert p.setup_id == "setup_bear_premium"
    assert p.intent is not None
    assert p.intent.order_type == "SELL"
    assert p.intent.price == 2652.00
    # Effective SL includes spread adjustment + buffer
    assert p.intent.stop_loss > 2660.00
