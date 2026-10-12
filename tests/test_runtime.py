import asyncio
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    SymbolInfo,
)
from xau_bot.domain.models import Candle, Timeframe
from xau_bot.filters.sessions import SessionFilter
from xau_bot.guardian.bridge import GuardianBridge
from xau_bot.guardian.models import GuardianConfig
from xau_bot.runtime.candle_loop import CandleLoop
from xau_bot.runtime.models import (
    CandleCloseEvent,
    DegradationReason,
    RuntimeStatus,
    WatchdogConfig,
    WatchdogHealth,
)
from xau_bot.runtime.runner import WatchdogRunner
from xau_bot.runtime.watchdog import Watchdog
from xau_bot.server import HealthStatusState, create_server


def _make_candle(
    time: datetime,
    tf: Timeframe = Timeframe.M5,
    open_: float = 2000.0,
    high: float = 2010.0,
    low: float = 1995.0,
    close: float = 2005.0,
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


class MockBroker:
    def __init__(self, connected: bool = True) -> None:
        self._connected = connected
        self.reconnect_called = 0

    def is_connected(self) -> bool:
        return self._connected

    def set_connected(self, connected: bool) -> None:
        self._connected = connected

    async def connect(self) -> bool:
        self.reconnect_called += 1
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def get_account_info(self) -> AccountInfo:
        raise NotImplementedError

    async def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        raise NotImplementedError

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> tuple[Candle, ...]:
        return ()

    async def send_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    async def modify_order(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult:
        raise NotImplementedError

    async def cancel_order(self, ticket: int) -> OrderResult:
        raise NotImplementedError

    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult:
        raise NotImplementedError

    async def get_open_positions(self) -> tuple[BrokerPosition, ...]:
        raise NotImplementedError

    async def get_open_orders(self) -> tuple[BrokerOrder, ...]:
        raise NotImplementedError


def test_candle_loop_ignores_forming_candle() -> None:
    loop = CandleLoop()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    t2 = t1 + timedelta(minutes=5)

    c0 = _make_candle(t0)
    c1 = _make_candle(t1)
    c2 = _make_candle(t2)

    events = loop.feed_candles(Timeframe.M5, [c0, c1, c2], has_forming_candle=True)

    assert len(events) == 2
    assert events[0].candle == c0
    assert events[1].candle == c1
    assert loop.get_forming_candle(Timeframe.M5) == c2
    assert loop.get_last_closed_candle(Timeframe.M5) == c1


def test_candle_loop_single_forming_candle_yields_no_events() -> None:
    loop = CandleLoop()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)

    events = loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=True)

    assert len(events) == 0
    assert loop.get_forming_candle(Timeframe.M5) == c0
    assert loop.get_last_closed_candle(Timeframe.M5) is None


def test_candle_loop_temporal_idempotency() -> None:
    loop = CandleLoop()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)

    c0 = _make_candle(t0)
    c1 = _make_candle(t1)

    events1 = loop.feed_candles(Timeframe.M5, [c0, c1], has_forming_candle=True)
    assert len(events1) == 1
    assert events1[0].candle == c0

    events2 = loop.feed_candles(Timeframe.M5, [c0, c1], has_forming_candle=True)
    assert len(events2) == 0

    t2 = t1 + timedelta(minutes=5)
    c2 = _make_candle(t2)
    events3 = loop.feed_candles(Timeframe.M5, [c0, c1, c2], has_forming_candle=True)
    assert len(events3) == 1
    assert events3[0].candle == c1


@pytest.mark.asyncio
async def test_candle_loop_handler_dispatch() -> None:
    loop = CandleLoop()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)
    t1 = t0 + timedelta(minutes=5)
    c1 = _make_candle(t1)

    received_sync: list[CandleCloseEvent] = []
    received_async: list[CandleCloseEvent] = []

    def sync_handler(evt: CandleCloseEvent) -> None:
        received_sync.append(evt)

    async def async_handler(evt: CandleCloseEvent) -> None:
        received_async.append(evt)

    loop.register_handler(Timeframe.M5, sync_handler)
    loop.register_handler(Timeframe.M5, async_handler)

    events = await loop.on_candles_async(Timeframe.M5, [c0, c1], has_forming_candle=True)

    assert len(events) == 1
    assert len(received_sync) == 1
    assert len(received_async) == 1
    assert received_sync[0].candle == c0
    assert received_async[0].candle == c0
    assert received_sync[0].closed_at == t0 + timedelta(minutes=5)


def test_watchdog_monitors_broker_connection() -> None:
    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)

    now = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    watchdog.update_tick(now)
    watchdog.beat_loop(now=now)

    health_ok = watchdog.check_health(now=now)
    assert health_ok.is_healthy is True
    assert health_ok.broker_connected is True
    assert health_ok.status_code == 200
    assert health_ok.status == "ok"

    broker.set_connected(False)
    health_degraded = watchdog.check_health(now=now)
    assert health_degraded.is_healthy is False
    assert health_degraded.broker_connected is False
    assert DegradationReason.BROKER_DISCONNECTED in health_degraded.reasons
    assert health_degraded.status_code == 503
    assert health_degraded.status == "degraded"


def test_watchdog_calculates_tick_age() -> None:
    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)

    t_tick = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t_tick)

    t_check = t_tick + timedelta(seconds=12.5)
    health = watchdog.check_health(now=t_check)

    assert abs(health.last_tick_age_s - 12.5) < 0.001


def test_watchdog_tick_staleness_market_open() -> None:
    broker = MockBroker(connected=True)
    config = WatchdogConfig(max_tick_age_s=30.0, ignore_stale_when_market_closed=True)
    session_filter = SessionFilter()
    watchdog = Watchdog(broker=broker, config=config, session_filter=session_filter)

    t_tick = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t_tick)
    watchdog.beat_loop(now=t_tick)

    t_check = t_tick + timedelta(seconds=45)
    health = watchdog.check_health(now=t_check)

    assert health.is_healthy is False
    assert DegradationReason.TICK_STALE in health.reasons
    assert health.status_code == 503


def test_watchdog_tick_staleness_ignored_when_market_closed() -> None:
    broker = MockBroker(connected=True)
    config = WatchdogConfig(max_tick_age_s=30.0, ignore_stale_when_market_closed=True)
    session_filter = SessionFilter()
    watchdog = Watchdog(broker=broker, config=config, session_filter=session_filter)

    t_tick = datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t_tick)
    watchdog.beat_loop(now=t_tick)

    t_check = t_tick + timedelta(hours=1)
    health = watchdog.check_health(now=t_check)

    assert DegradationReason.TICK_STALE not in health.reasons
    assert health.is_healthy is True


def test_watchdog_loop_heartbeat_hang_detection() -> None:
    broker = MockBroker(connected=True)
    config = WatchdogConfig(max_loop_silence_s=60.0)
    watchdog = Watchdog(broker=broker, config=config)

    t0 = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t0)
    watchdog.beat_loop(now=t0)

    t_check = t0 + timedelta(seconds=70)
    watchdog.update_tick(t_check)

    health = watchdog.check_health(now=t_check)
    assert health.is_healthy is False
    assert DegradationReason.LOOP_STALE in health.reasons
    assert health.status_code == 503


def test_watchdog_syncs_server_state() -> None:
    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)
    server_state = HealthStatusState()

    t0 = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t0)
    watchdog.beat_loop(now=t0)

    t1 = t0 + timedelta(seconds=5)
    health = watchdog.sync_server_state(server_state, now=t1)

    assert server_state.broker_connected is True
    assert abs(server_state.last_tick_age_s - 5.0) < 0.001
    assert health.is_healthy is True

    server_state.trading_halted = True
    health_halted = watchdog.sync_server_state(server_state, now=t1)
    assert health_halted.is_healthy is False
    assert DegradationReason.TRADING_HALTED in health_halted.reasons


def test_server_health_endpoint_semantics_with_watchdog() -> None:
    state = HealthStatusState()
    health_healthy = WatchdogHealth(
        is_healthy=True,
        broker_connected=True,
        last_tick_age_s=2.5,
        trading_halted=False,
        reasons=(),
        status_code=200,
        status="ok",
        last_loop_beat_age_s=1.0,
        timestamp=datetime.now(UTC),
    )
    state.apply_watchdog_health(health_healthy)

    code, body = state.get_health()
    assert code == HTTPStatus.OK
    assert body["status"] == "ok"
    assert body["broker_connected"] is True
    assert body["last_tick_age_s"] == 2.5
    assert body["trading_halted"] is False

    health_degraded = WatchdogHealth(
        is_healthy=False,
        broker_connected=False,
        last_tick_age_s=50.0,
        trading_halted=False,
        reasons=(DegradationReason.BROKER_DISCONNECTED, DegradationReason.TICK_STALE),
        status_code=503,
        status="degraded",
        last_loop_beat_age_s=10.0,
        timestamp=datetime.now(UTC),
    )
    state.apply_watchdog_health(health_degraded)

    code, body = state.get_health()
    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "degraded"
    assert body["broker_connected"] is False
    assert body["last_tick_age_s"] == 50.0


def test_watchdog_config_validation() -> None:
    with pytest.raises(ValueError, match="max_tick_age_s must be positive"):
        WatchdogConfig(max_tick_age_s=0.0)

    with pytest.raises(ValueError, match="max_loop_silence_s must be positive"):
        WatchdogConfig(max_loop_silence_s=-5.0)

    with pytest.raises(ValueError, match="check_interval_s must be positive"):
        WatchdogConfig(check_interval_s=0.0)

    with pytest.raises(ValueError, match="reconnect_backoff_base_s must be positive"):
        WatchdogConfig(reconnect_backoff_base_s=-1.0)

    with pytest.raises(
        ValueError, match="reconnect_backoff_max_s must be >= reconnect_backoff_base_s"
    ):
        WatchdogConfig(reconnect_backoff_base_s=10.0, reconnect_backoff_max_s=5.0)


def test_watchdog_update_tick_naive_raises() -> None:
    watchdog = Watchdog()
    naive_dt = datetime(2026, 3, 10, 10, 0)
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        watchdog.update_tick(naive_dt)


def test_watchdog_no_broker_defaults_to_disconnected() -> None:
    watchdog = Watchdog(broker=None)
    health = watchdog.check_health()
    assert health.broker_connected is False
    assert health.is_healthy is False
    assert DegradationReason.BROKER_DISCONNECTED in health.reasons


def test_candle_loop_detect_series_gaps() -> None:
    loop = CandleLoop()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    t3 = t0 + timedelta(minutes=20)

    c0 = _make_candle(t0)
    c1 = _make_candle(t1)
    c3 = _make_candle(t3)

    assert loop.detect_series_gaps(Timeframe.M5) == []

    loop.feed_candles(Timeframe.M5, [c0, c1, c3], has_forming_candle=False)
    gaps = loop.detect_series_gaps(Timeframe.M5)

    assert len(gaps) == 1
    assert gaps[0].missing_bars >= 2


def test_candle_loop_empty_feed_and_unknown_timeframe() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5], max_history_bars=2)
    assert loop.feed_candles(Timeframe.M5, []) == []

    received: list[CandleCloseEvent] = []
    loop.register_handler(Timeframe.M15, lambda e: received.append(e))

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0, tf=Timeframe.M15)
    t1 = t0 + timedelta(minutes=15)
    c1 = _make_candle(t1, tf=Timeframe.M15)

    events = loop.feed_candles(Timeframe.M15, [c0, c1], has_forming_candle=True)
    assert len(events) == 1
    assert len(received) == 1


def test_candle_loop_history_truncation_and_processed_queue_purge() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5], max_history_bars=2)

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)
    t1 = t0 + timedelta(minutes=5)
    c1 = _make_candle(t1)
    t2 = t1 + timedelta(minutes=5)
    c2 = _make_candle(t2)

    loop.feed_candles(Timeframe.M5, [c0, c1], has_forming_candle=False)
    assert len(loop.get_closed_series(Timeframe.M5)) == 2

    loop.feed_candles(Timeframe.M5, [c0, c1, c2], has_forming_candle=False)
    series = loop.get_closed_series(Timeframe.M5)
    assert len(series) == 2
    assert series[0] == c1
    assert series[1] == c2


def test_repro_bug_t17_01_weekend_tick_staleness_and_loop_stale_in_http() -> None:
    broker = MockBroker(connected=True)
    session_filter = SessionFilter()
    watchdog = Watchdog(broker=broker, session_filter=session_filter)
    server_state = HealthStatusState()

    t_tick = datetime(2026, 3, 14, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t_tick)
    watchdog.beat_loop(now=t_tick)

    t_check = t_tick + timedelta(hours=1)
    watchdog.sync_server_state(server_state, now=t_check)

    status_code, body = server_state.get_health()
    assert status_code == HTTPStatus.OK
    assert body["status"] == "ok"

    t_tuesday = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    watchdog.update_tick(t_tuesday)
    watchdog.beat_loop(now=t_tuesday)

    t_hang = t_tuesday + timedelta(seconds=130)
    watchdog.update_tick(t_hang)
    watchdog.sync_server_state(server_state, now=t_hang)

    code_hang, body_hang = server_state.get_health()
    assert code_hang == HTTPStatus.SERVICE_UNAVAILABLE
    assert body_hang["status"] == "degraded"
    assert "LOOP_STALE" in body_hang["reasons"]


def test_repro_bug_t17_03_async_handler_scheduled_in_sync_feed() -> None:
    loop = CandleLoop()
    executed = False

    async def async_handler(evt: CandleCloseEvent) -> None:
        nonlocal executed
        executed = True

    loop.register_handler(Timeframe.M5, async_handler)

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)

    async def _runner() -> None:
        loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=False)
        await asyncio.sleep(0.01)

    asyncio.run(_runner())
    assert executed is True


def test_repro_bug_t17_04_push_candle_and_on_new_candles_multi_tf() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5, Timeframe.M15])

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c_m5 = _make_candle(t0, tf=Timeframe.M5)
    c_m15 = _make_candle(t0, tf=Timeframe.M15)

    events_pushed = loop.push_candle(Timeframe.M5, c_m5, has_forming_candle=False)
    assert len(events_pushed) == 1
    assert events_pushed[0].candle == c_m5

    t1 = t0 + timedelta(minutes=5)
    c_m5_1 = _make_candle(t1, tf=Timeframe.M5)
    c_m15_1 = _make_candle(t0 + timedelta(minutes=15), tf=Timeframe.M15)

    async def _run_multi() -> list[CandleCloseEvent]:
        return await loop.on_new_candles(
            {
                Timeframe.M15: [c_m15, c_m15_1],
                Timeframe.M5: [c_m5, c_m5_1],
            },
            has_forming_candle=False,
        )

    multi_events = asyncio.run(_run_multi())
    assert len(multi_events) >= 2
    for i in range(1, len(multi_events)):
        assert multi_events[i].closed_at >= multi_events[i - 1].closed_at


def test_repro_finding_f11_backfill_ordering_in_history() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5])
    t1 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

    c1 = _make_candle(t1)
    c0 = _make_candle(t0)

    loop.feed_candles(Timeframe.M5, [c1], has_forming_candle=False)
    loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=False)

    history = loop.get_closed_series(Timeframe.M5)
    assert len(history) == 2
    assert history[0].time == t0
    assert history[1].time == t1


def test_repro_finding_f10_boot_in_open_market_without_ticks_is_stale() -> None:
    broker = MockBroker(connected=True)
    session_filter = SessionFilter()
    watchdog = Watchdog(broker=broker, session_filter=session_filter)

    now_tuesday = datetime(2026, 3, 10, 12, 0, 0, tzinfo=UTC)
    health = watchdog.check_health(now=now_tuesday)

    assert health.is_healthy is False
    assert DegradationReason.TICK_STALE in health.reasons


def test_repro_bug_t17_02_server_watchdog_runner_and_real_http_503() -> None:
    broker = MockBroker(connected=False)
    watchdog = Watchdog(broker=broker)
    state = HealthStatusState()

    runner = WatchdogRunner(watchdog=watchdog, target=state, interval_s=0.1)
    runner.start()

    server = create_server(host="127.0.0.1", port=0, state=state)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        urllib.request.urlopen(f"{base_url}/health")
        pytest.fail("Expected HTTP 503 Service Unavailable")
    except urllib.error.HTTPError as err:
        assert err.code == HTTPStatus.SERVICE_UNAVAILABLE
        data = json.loads(err.read().decode("utf-8"))
        assert data["status"] == "degraded"
        assert data["broker_connected"] is False
        assert "BROKER_DISCONNECTED" in data["reasons"]
    finally:
        runner.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


def test_watchdog_beat_loop_per_timeframe() -> None:
    watchdog = Watchdog()
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)

    watchdog.beat_loop(timeframe=Timeframe.M5, now=t0)
    watchdog.beat_loop(timeframe=Timeframe.H1, now=t1)

    assert watchdog.get_last_loop_beat(Timeframe.M5) == t0
    assert watchdog.get_last_loop_beat(Timeframe.H1) == t1
    assert watchdog.get_last_loop_beat() == t1
    assert watchdog.get_last_loop_beat(Timeframe.D1) is None


def test_runner_double_start_and_error_handling(caplog: pytest.LogCaptureFixture) -> None:
    class FailingBroker(MockBroker):
        async def connect(self) -> bool:
            raise RuntimeError("Broker connection timeout")

    class ExplodingWatchdog(Watchdog):
        def sync_server_state(self, target: Any, now: Any = None) -> Any:
            raise RuntimeError("Sync cycle crash")

    broker = FailingBroker(connected=False)
    watchdog = Watchdog(broker=broker, config=WatchdogConfig(reconnect_backoff_base_s=0.01))
    state = HealthStatusState()

    runner = WatchdogRunner(watchdog=watchdog, target=state, interval_s=0.05)
    runner.start()
    assert runner._thread is not None
    assert runner._thread.is_alive()

    runner.start()
    time.sleep(0.12)
    runner.stop()
    assert not runner._thread.is_alive()

    exploding_dog = ExplodingWatchdog(broker=broker)
    exploding_runner = WatchdogRunner(watchdog=exploding_dog, target=state, interval_s=0.05)
    with caplog.at_level(logging.ERROR):
        exploding_runner.start()
        time.sleep(0.08)
        exploding_runner.stop()
    assert any(
        "Error in watchdog sync cycle" in rec.message
        or "Initial watchdog sync failed" in rec.message
        for rec in caplog.records
    )

    failing_reconnect_dog = Watchdog(
        broker=FailingBroker(connected=False),
        config=WatchdogConfig(reconnect_backoff_base_s=0.01),
    )
    failing_runner = WatchdogRunner(watchdog=failing_reconnect_dog, target=state, interval_s=0.03)
    with caplog.at_level(logging.WARNING):
        failing_runner.start()
        time.sleep(0.08)
        failing_runner.stop()
    assert any(
        "Broker reconnection failed" in rec.message
        or "Reconnection attempt failed in runner" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_watchdog_reconnect_flow() -> None:
    broker = MockBroker(connected=False)
    watchdog = Watchdog(broker=broker, config=WatchdogConfig(reconnect_backoff_base_s=0.1))
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

    reconnected = await watchdog.maybe_reconnect(now=t0)
    assert reconnected is True
    assert broker.is_connected() is True

    assert await watchdog.maybe_reconnect(now=t0) is False

    broker.set_connected(False)
    assert await watchdog.maybe_reconnect(now=t0 + timedelta(milliseconds=10)) is False
    assert await watchdog.maybe_reconnect(now=t0 + timedelta(milliseconds=200)) is True
    assert broker.is_connected() is True

    class ErrorBroker(MockBroker):
        async def connect(self) -> bool:
            raise ConnectionRefusedError("Offline")

    err_dog = Watchdog(broker=ErrorBroker(connected=False))
    assert await err_dog.maybe_reconnect() is False


def test_candle_loop_sync_dispatch_no_running_loop_raises() -> None:
    loop = CandleLoop()

    async def async_handler(evt: CandleCloseEvent) -> None:
        pass

    loop.register_handler(Timeframe.M5, async_handler)
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)

    with pytest.raises(RuntimeError, match="Async handler cannot be dispatched synchronously"):
        loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=False, dispatch=True)

    assert loop.get_last_closed_candle(Timeframe.M5) is None


@pytest.mark.asyncio
async def test_repro_r2_01_batch_lookahead_and_multi_tf_temporal_isolation() -> None:
    loop = CandleLoop()

    t_1000 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t_1005 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)
    t_1010 = datetime(2026, 3, 10, 10, 10, tzinfo=UTC)
    t_1015 = datetime(2026, 3, 10, 10, 15, tzinfo=UTC)

    c_1000 = _make_candle(t_1000, Timeframe.M5)
    c_1005 = _make_candle(t_1005, Timeframe.M5)
    c_1010 = _make_candle(t_1010, Timeframe.M5)
    c_1015 = _make_candle(t_1015, Timeframe.M5)

    events = loop.feed_candles(
        Timeframe.M5,
        [c_1000, c_1005, c_1010, c_1015],
        has_forming_candle=True,
        dispatch=False,
    )
    assert len(events) == 3
    assert events[0].forming_candle is None
    assert events[1].forming_candle is None
    assert events[2].forming_candle is not None
    assert events[2].forming_candle.time == t_1015

    multi_loop = CandleLoop()
    h1_closed_seen_at_m5_1005: list[Candle] = []

    async def m5_handler(evt: CandleCloseEvent) -> None:
        if evt.candle.time == t_1000:
            h1_series = multi_loop.get_closed_series(Timeframe.H1)
            h1_closed_seen_at_m5_1005.extend(h1_series)

    multi_loop.register_handler(Timeframe.M5, m5_handler)

    h1_1000 = _make_candle(t_1000, Timeframe.H1)
    h1_1100 = _make_candle(datetime(2026, 3, 10, 11, 0, tzinfo=UTC), Timeframe.H1)

    candles_by_tf = {
        Timeframe.M5: [c_1000, c_1005, c_1010, c_1015],
        Timeframe.H1: [h1_1000, h1_1100],
    }
    await multi_loop.on_new_candles(candles_by_tf, has_forming_candle=True)
    assert len(h1_closed_seen_at_m5_1005) == 0


@pytest.mark.asyncio
async def test_repro_r2_02_on_new_candles_handler_dispatch_and_routing() -> None:
    loop = CandleLoop()
    called_events: list[tuple[str, CandleCloseEvent]] = []

    def sync_m5_handler(evt: CandleCloseEvent) -> None:
        called_events.append(("sync_m5", evt))

    async def async_h1_handler(evt: CandleCloseEvent) -> None:
        called_events.append(("async_h1", evt))

    loop.register_handler(Timeframe.M5, sync_m5_handler)
    loop.register_handler(Timeframe.H1, async_h1_handler)

    t_1000 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t_1005 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)

    c_m5 = _make_candle(t_1000, Timeframe.M5)
    c_m5_forming = _make_candle(t_1005, Timeframe.M5)
    c_h1 = _make_candle(t_1000, Timeframe.H1)
    c_h1_forming = _make_candle(datetime(2026, 3, 10, 11, 0, tzinfo=UTC), Timeframe.H1)

    candles_by_tf = {
        Timeframe.M5: [c_m5, c_m5_forming],
        Timeframe.H1: [c_h1, c_h1_forming],
    }

    events = await loop.on_new_candles(candles_by_tf, has_forming_candle=True)
    assert len(events) == 2
    assert len(called_events) == 2
    assert called_events[0][0] == "sync_m5"
    assert called_events[0][1].candle.time == t_1000
    assert called_events[0][1].closed_at == t_1005
    assert called_events[1][0] == "async_h1"
    assert called_events[1][1].candle.time == t_1000
    assert called_events[1][1].closed_at == datetime(2026, 3, 10, 11, 0, tzinfo=UTC)


def test_watchdog_no_tick_market_closed_with_ignore_false() -> None:
    session_filter = SessionFilter()
    sunday = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
    config = WatchdogConfig(ignore_stale_when_market_closed=False)
    dog = Watchdog(session_filter=session_filter, config=config)
    health = dog.check_health(now=sunday)
    assert health.is_healthy is False
    assert DegradationReason.TICK_STALE in health.reasons


def test_runtime_status_and_watchdog_config_property() -> None:
    assert RuntimeStatus.OK.value == "ok"
    assert RuntimeStatus.DEGRADED.value == "degraded"
    assert RuntimeStatus.HALTED.value == "halted"

    dog = Watchdog(config=WatchdogConfig(check_interval_s=2.5))
    assert dog.config.check_interval_s == 2.5


def test_health_status_state_thread_safety_and_halt_isolation() -> None:
    state = HealthStatusState()
    state.trading_halted = True

    health = WatchdogHealth(
        is_healthy=True,
        broker_connected=True,
        last_tick_age_s=1.0,
        trading_halted=False,
        reasons=(),
        status_code=200,
        status="ok",
        last_loop_beat_age_s=0.5,
        timestamp=datetime.now(UTC),
    )
    state.apply_watchdog_health(health)

    assert state.trading_halted is True
    code, payload = state.get_health()
    assert code == HTTPStatus.OK
    assert payload["status"] == "halted"
    assert "TRADING_HALTED" in payload["reasons"]


def test_run_server_with_watchdog_clean_lifecycle() -> None:
    broker = MockBroker(connected=True)
    dog = Watchdog(broker=broker, config=WatchdogConfig(check_interval_s=0.05))
    now = datetime.now(UTC)
    dog.update_tick(now)
    dog.beat_loop(now=now)

    state = HealthStatusState()
    runner = WatchdogRunner(watchdog=dog, target=state, interval_s=0.05)
    runner.start()

    server = create_server(host="127.0.0.1", port=0, state=state)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            assert resp.status == HTTPStatus.OK
    finally:
        runner.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
    assert runner._thread is not None
    assert not runner._thread.is_alive()


def test_watchdog_runner_with_guardian_bridge(tmp_path: Path) -> None:
    cfg = GuardianConfig(common_dir=tmp_path, heartbeat_interval_s=0.05)
    bridge = GuardianBridge(config=cfg)

    dog = Watchdog()
    state = HealthStatusState()
    runner = WatchdogRunner(watchdog=dog, target=state, interval_s=0.02, guardian_bridge=bridge)
    runner.start()

    hb_file = tmp_path / cfg.heartbeat_filename
    for _ in range(50):
        if hb_file.exists():
            break
        time.sleep(0.01)

    runner.stop()
    assert hb_file.exists()
    content = hb_file.read_text(encoding="utf-8")
    assert "ok" in content or "degraded" in content


def test_watchdog_runner_with_exploding_guardian_bridge() -> None:
    dog = Watchdog()
    state = HealthStatusState()
    mock_bridge = MagicMock()
    mock_bridge.pulse.side_effect = RuntimeError("Bridge explosion")

    runner = WatchdogRunner(
        watchdog=dog, target=state, interval_s=0.02, guardian_bridge=mock_bridge
    )
    runner.start()
    time.sleep(0.06)
    runner.stop()
    assert runner._thread is not None
    assert not runner._thread.is_alive()


@pytest.mark.asyncio
async def test_repro_r3_01_async_and_sync_batch_forming_candle_isolation() -> None:
    t_1000 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t_1005 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)
    t_1010 = datetime(2026, 3, 10, 10, 10, tzinfo=UTC)
    t_1015 = datetime(2026, 3, 10, 10, 15, tzinfo=UTC)

    c0 = _make_candle(t_1000)
    c1 = _make_candle(t_1005)
    c2 = _make_candle(t_1010)
    c3 = _make_candle(t_1015)

    async_loop = CandleLoop()
    async_forming_seen_at_1000: list[Candle | None] = []
    async_closed_len_seen_at_1000: list[int] = []

    async def async_handler(evt: CandleCloseEvent) -> None:
        if evt.candle.time == t_1000:
            async_forming_seen_at_1000.append(async_loop.get_forming_candle(Timeframe.M5))
            async_closed_len_seen_at_1000.append(len(async_loop.get_closed_series(Timeframe.M5)))

    async_loop.register_handler(Timeframe.M5, async_handler)
    await async_loop.on_candles_async(Timeframe.M5, [c0, c1, c2, c3], has_forming_candle=True)

    assert len(async_forming_seen_at_1000) == 1
    assert async_forming_seen_at_1000[0] is None
    assert async_closed_len_seen_at_1000[0] == 1

    sync_loop = CandleLoop()
    sync_forming_seen_at_1000: list[Candle | None] = []
    sync_closed_len_seen_at_1000: list[int] = []

    def sync_handler(evt: CandleCloseEvent) -> None:
        if evt.candle.time == t_1000:
            sync_forming_seen_at_1000.append(sync_loop.get_forming_candle(Timeframe.M5))
            sync_closed_len_seen_at_1000.append(len(sync_loop.get_closed_series(Timeframe.M5)))

    sync_loop.register_handler(Timeframe.M5, sync_handler)
    sync_loop.feed_candles(Timeframe.M5, [c0, c1, c2, c3], has_forming_candle=True, dispatch=True)

    assert len(sync_forming_seen_at_1000) == 1
    assert sync_forming_seen_at_1000[0] is None
    assert sync_closed_len_seen_at_1000[0] == 1


def test_repro_r3_02_backfill_candle_closed_series_causality() -> None:
    loop = CandleLoop()
    t1 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)

    c1 = _make_candle(t1)
    c0 = _make_candle(t0)

    loop.feed_candles(Timeframe.M5, [c1], has_forming_candle=False)
    events = loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=False)

    assert len(events) == 1
    assert len(events[0].closed_series) == 1
    assert events[0].closed_series[0].time == t0


def test_repro_r3_06_boot_hang_without_beat_detects_loop_stale() -> None:
    session_filter = SessionFilter()
    config = WatchdogConfig(max_loop_silence_s=30.0)
    boot_time = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    dog = Watchdog(session_filter=session_filter, config=config, boot_time=boot_time)

    health_hang = dog.check_health(now=boot_time + timedelta(seconds=45))
    assert health_hang.is_healthy is False
    assert DegradationReason.LOOP_STALE in health_hang.reasons

    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        Watchdog(boot_time=datetime(2026, 3, 10, 12, 0))


def test_repro_r3_08_watchdog_log_deduplication(caplog: pytest.LogCaptureFixture) -> None:
    broker = MockBroker(connected=False)
    dog = Watchdog(broker=broker)

    now = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    with caplog.at_level(logging.WARNING):
        dog.check_health(now=now)
        dog.check_health(now=now + timedelta(seconds=1))
        dog.check_health(now=now + timedelta(seconds=2))

    degraded_records = [r for r in caplog.records if "Health check degraded" in r.message]
    assert len(degraded_records) == 1


def test_repro_r3_09_beat_loop_naive_raises() -> None:
    dog = Watchdog()
    naive = datetime(2026, 3, 10, 10, 0)
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        dog.beat_loop(now=naive)


@pytest.mark.asyncio
async def test_repro_r4_01_temporal_idempotency_large_batch_beyond_max_history() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5], max_history_bars=2)

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)
    t2 = datetime(2026, 3, 10, 10, 10, tzinfo=UTC)

    c0 = _make_candle(t0)
    c1 = _make_candle(t1)
    c2 = _make_candle(t2)

    batch = [c0, c1, c2]

    events1 = loop.feed_candles(Timeframe.M5, batch, has_forming_candle=False)
    assert len(events1) == 3

    events2 = loop.feed_candles(Timeframe.M5, batch, has_forming_candle=False)
    assert len(events2) == 0

    events3 = await loop.on_new_candles({Timeframe.M5: batch}, has_forming_candle=False)
    assert len(events3) == 0


def test_candle_loop_invalid_max_history_bars_raises() -> None:
    with pytest.raises(ValueError, match="max_history_bars must be >= 1"):
        CandleLoop(max_history_bars=0)
    with pytest.raises(ValueError, match="max_history_bars must be >= 1"):
        CandleLoop(max_history_bars=-5)


def test_server_infinite_tick_age_strict_json_null() -> None:
    state = HealthStatusState()
    health = WatchdogHealth(
        is_healthy=False,
        broker_connected=True,
        last_tick_age_s=float("inf"),
        trading_halted=False,
        reasons=(DegradationReason.TICK_STALE,),
        status_code=503,
        status="degraded",
        last_loop_beat_age_s=0.0,
        timestamp=datetime.now(UTC),
    )
    state.apply_watchdog_health(health)
    code, payload = state.get_health()

    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert payload["last_tick_age_s"] is None

    server = create_server(host="127.0.0.1", port=0, state=state)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
            pytest.fail("Expected 503")
        except urllib.error.HTTPError as err:
            assert err.code == HTTPStatus.SERVICE_UNAVAILABLE
            raw_body = err.read().decode("utf-8")
            assert "Infinity" not in raw_body
            data = json.loads(raw_body)
            assert data["last_tick_age_s"] is None
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_candle_loop_handler_exception_isolation_sync_and_async(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5])
    calls: list[str] = []

    def failing_sync_handler(evt: CandleCloseEvent) -> None:
        calls.append("failing_sync")
        raise RuntimeError("Handler bug in sync")

    def good_sync_handler(evt: CandleCloseEvent) -> None:
        calls.append("good_sync")

    loop.register_handler(Timeframe.M5, failing_sync_handler)
    loop.register_handler(Timeframe.M5, good_sync_handler)

    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    c0 = _make_candle(t0)

    with caplog.at_level(logging.ERROR):
        events = loop.feed_candles(Timeframe.M5, [c0], has_forming_candle=False, dispatch=True)

    assert len(events) == 1
    assert "failing_sync" in calls
    assert "good_sync" in calls
    assert any(
        "Unhandled exception in sync candle handler" in rec.message for rec in caplog.records
    )

    async_loop = CandleLoop(timeframes=[Timeframe.M5])
    async_calls: list[str] = []

    async def failing_async_handler(evt: CandleCloseEvent) -> None:
        async_calls.append("failing_async")
        raise ValueError("Handler bug in async")

    async def good_async_handler(evt: CandleCloseEvent) -> None:
        async_calls.append("good_async")

    async_loop.register_handler(Timeframe.M5, failing_async_handler)
    async_loop.register_handler(Timeframe.M5, good_async_handler)

    with caplog.at_level(logging.ERROR):
        a_events = await async_loop.on_new_candles({Timeframe.M5: [c0]}, has_forming_candle=False)

    assert len(a_events) == 1
    assert "failing_async" in async_calls
    assert "good_async" in async_calls
    assert any(
        "Unhandled exception in async candle handler" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_candle_loop_forming_candle_updates_on_refeed_without_new_closed() -> None:
    loop = CandleLoop(timeframes=[Timeframe.M5])
    t0 = datetime(2026, 3, 10, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 10, 10, 5, tzinfo=UTC)

    c0 = _make_candle(t0, Timeframe.M5)
    forming_1 = _make_candle(t1, Timeframe.M5, close=2000.0)
    forming_2 = _make_candle(t1, Timeframe.M5, close=2005.0)

    loop.feed_candles(Timeframe.M5, [c0, forming_1], has_forming_candle=True)
    forming = loop.get_forming_candle(Timeframe.M5)
    assert forming is not None
    assert forming.close == 2000.0

    loop.feed_candles(Timeframe.M5, [c0, forming_2], has_forming_candle=True)
    forming = loop.get_forming_candle(Timeframe.M5)
    assert forming is not None
    assert forming.close == 2005.0

    forming_3 = _make_candle(t1, Timeframe.M5, close=2010.0)
    await loop.on_new_candles({Timeframe.M5: [c0, forming_3]}, has_forming_candle=True)
    forming = loop.get_forming_candle(Timeframe.M5)
    assert forming is not None
    assert forming.close == 2010.0


@pytest.mark.asyncio
async def test_watchdog_naive_datetime_in_check_health_and_reconnect_raises() -> None:
    dog = Watchdog()
    naive = datetime(2026, 3, 10, 10, 0)

    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        dog.check_health(now=naive)

    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        await dog.maybe_reconnect(now=naive)
