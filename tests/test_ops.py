import io
import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any

import pytest

from test_runtime import MockBroker
from xau_bot.runtime.models import DegradationReason, WatchdogHealth
from xau_bot.runtime.ops.alerts import (
    ALERT_TITLE_DEGRADED,
    ALERT_TITLE_HALTED,
    ALERT_TITLE_RECOVERED,
    ALERT_TITLE_RESUMED,
    AlertManager,
    ConsoleAlertChannel,
    TelegramAlertChannel,
)
from xau_bot.runtime.ops.logging import JsonLogFormatter, SensitiveDataFilter
from xau_bot.runtime.runner import WatchdogRunner
from xau_bot.runtime.watchdog import Watchdog
from xau_bot.server import HealthStatusState


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_sensitive_data_filter_masks_secrets() -> None:
    sanitizer = SensitiveDataFilter(
        sensitive_patterns=[
            r"password\s*[:=]\s*([^\s,]+)",
            r"token\s*[:=]\s*([^\s,]+)",
            r"secret\s*[:=]\s*([^\s,]+)",
        ]
    )
    logger = logging.getLogger("test.sanitizer")
    record = logger.makeRecord(
        name="test.sanitizer",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Connecting with password=SuperSecretPass123 and token: my_secret_token_abc!",
        args=(),
        exc_info=None,
    )
    assert sanitizer.filter(record) is True
    assert "SuperSecretPass123" not in record.msg
    assert "my_secret_token_abc" not in record.msg
    assert "***REDACTED***" in record.msg


def test_json_log_formatter_produces_rfc8259_json() -> None:
    formatter = JsonLogFormatter()
    logger = logging.getLogger("test.json")
    record = logger.makeRecord(
        name="test.json",
        level=logging.WARNING,
        fn="test.py",
        lno=20,
        msg="Tick staleness detected",
        args=(),
        exc_info=None,
        extra={"event": "tick_stale", "tick_age_s": 45.2},
    )
    output = formatter.format(record)
    data = json.loads(output)
    assert data["level"] == "WARNING"
    assert data["logger"] == "test.json"
    assert data["message"] == "Tick staleness detected"
    assert data["event"] == "tick_stale"
    assert data["tick_age_s"] == 45.2
    assert "timestamp" in data


def test_refined_health_status_state_halted_returns_200() -> None:
    state = HealthStatusState(broker_connected=True, trading_halted=False)
    code, payload = state.get_health()
    assert code == HTTPStatus.OK
    assert payload["status"] == "ok"
    assert payload["trading_halted"] is False

    state.trading_halted = True
    code_halted, payload_halted = state.get_health()
    assert code_halted == HTTPStatus.OK
    assert payload_halted["status"] == "halted"
    assert payload_halted["trading_halted"] is True
    assert "TRADING_HALTED" in payload_halted["reasons"]

    degraded_health = WatchdogHealth(
        is_healthy=False,
        broker_connected=False,
        last_tick_age_s=0.0,
        trading_halted=False,
        reasons=(DegradationReason.BROKER_DISCONNECTED,),
        status_code=503,
        status="degraded",
        last_loop_beat_age_s=0.0,
        timestamp=datetime.now(UTC),
    )
    state.apply_watchdog_health(degraded_health)
    code_degraded, payload_degraded = state.get_health()
    assert code_degraded == HTTPStatus.SERVICE_UNAVAILABLE
    assert payload_degraded["status"] == "degraded"
    assert "BROKER_DISCONNECTED" in payload_degraded["reasons"]


@pytest.mark.asyncio
async def test_alert_manager_dispatches_and_deduplicates(caplog: pytest.LogCaptureFixture) -> None:
    channel = ConsoleAlertChannel()
    manager = AlertManager(channels=[channel], deduplicate_s=10.0)

    with caplog.at_level(logging.WARNING):
        sent_count = await manager.notify(
            level="WARNING",
            title=ALERT_TITLE_DEGRADED,
            message="Broker disconnected",
            metadata={"reasons": ["BROKER_DISCONNECTED"]},
        )
    assert sent_count == 1
    assert any(ALERT_TITLE_DEGRADED in rec.message for rec in caplog.records)

    sent_dup = await manager.notify(
        level="WARNING",
        title=ALERT_TITLE_DEGRADED,
        message="Broker disconnected again",
        metadata={"reasons": ["BROKER_DISCONNECTED"]},
    )
    assert sent_dup == 0


@pytest.mark.asyncio
async def test_telegram_alert_channel_resilience_on_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("Telegram server unreachable")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)

    channel = TelegramAlertChannel(token="dummy_token", chat_id="123456", timeout_s=0.5)
    with caplog.at_level(logging.WARNING):
        success = await channel.send_alert(
            level="ERROR",
            title="Critical Crash",
            message="Database connection lost",
        )
    assert success is False
    assert any(
        "telegram_alert_failed" in str(rec.__dict__)
        or "Failed to send Telegram alert" in rec.message
        for rec in caplog.records
    )


def test_runner_dispatches_alerts_on_health_transition() -> None:
    class SpyAlertChannel:
        def __init__(self) -> None:
            self.alerts: list[tuple[str, str, str]] = []

        async def send_alert(
            self,
            level: str,
            title: str,
            message: str,
            metadata: Any = None,
        ) -> bool:
            self.alerts.append((level, title, message))
            return True

    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)
    now = datetime.now(UTC)
    watchdog.update_tick(now)
    watchdog.beat_loop(now=now)
    state = HealthStatusState()
    spy = SpyAlertChannel()
    manager = AlertManager(channels=[spy], deduplicate_s=0.0)

    runner = WatchdogRunner(watchdog=watchdog, target=state, interval_s=0.02, alert_manager=manager)
    runner.start()

    broker.set_connected(False)
    assert _wait_for(lambda: any(ALERT_TITLE_DEGRADED in title for _, title, _ in spy.alerts))

    now2 = datetime.now(UTC)
    watchdog.update_tick(now2)
    watchdog.beat_loop(now=now2)
    broker.set_connected(True)
    assert _wait_for(
        lambda: ("INFO", ALERT_TITLE_RECOVERED) in [(lvl, t) for lvl, t, _ in spy.alerts]
    )

    runner.stop()


def test_sensitive_data_filter_args_tuple_and_dict() -> None:
    sanitizer = SensitiveDataFilter()
    logger = logging.getLogger("test.sanitizer.args")

    rec_tuple = logger.makeRecord(
        name="test.sanitizer.args",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Log com args: %s %d",
        args=("password: secret_pass", 42),
        exc_info=None,
    )
    sanitizer.filter(rec_tuple)
    assert "***REDACTED***" in rec_tuple.msg
    assert "secret_pass" not in rec_tuple.msg

    rec_dict = logger.makeRecord(
        name="test.sanitizer.args",
        level=logging.INFO,
        fn="test.py",
        lno=10,
        msg="Log com dict args: %(secret)s %(num)d",
        args={"secret": "token: topsecret", "num": 99},
        exc_info=None,
    )
    sanitizer.filter(rec_dict)
    assert "***REDACTED***" in rec_dict.msg
    assert "topsecret" not in rec_dict.msg


def test_sensitive_data_filter_pattern_without_capture_group() -> None:
    sanitizer = SensitiveDataFilter(sensitive_patterns=[r"raw_secret_[0-9]+"])
    logger = logging.getLogger("test.sanitizer.raw")
    rec = logger.makeRecord(
        name="test.sanitizer.raw",
        level=logging.INFO,
        fn="test.py",
        lno=1,
        msg="Chave detectada: raw_secret_12345",
        args=(),
        exc_info=None,
    )
    sanitizer.filter(rec)
    assert "raw_secret_12345" not in rec.msg
    assert "***REDACTED***" in rec.msg


def test_json_log_formatter_non_serializable_and_exception() -> None:
    formatter = JsonLogFormatter()
    logger = logging.getLogger("test.json.edge")

    class Unserializable:
        def __str__(self) -> str:
            return "CustomUnserializableObject"

    try:
        raise ValueError("Simulated error for traceback")
    except ValueError:
        exc_info = sys.exc_info()

    rec = logger.makeRecord(
        name="test.json.edge",
        level=logging.ERROR,
        fn="test.py",
        lno=20,
        msg="Falha no subsistema",
        args=(),
        exc_info=exc_info,
    )
    rec.custom_obj = Unserializable()
    rec.nan_val = float("nan")

    output = formatter.format(rec)
    data = json.loads(output)
    assert data["level"] == "ERROR"
    assert "Simulated error for traceback" in data["exception"]
    assert data["custom_obj"] == "CustomUnserializableObject"
    assert str(data["nan_val"]) == "nan"


@pytest.mark.asyncio
async def test_telegram_channel_empty_credentials_returns_false() -> None:
    channel = TelegramAlertChannel(token="", chat_id="")
    success = await channel.send_alert(level="INFO", title="Title", message="Msg")
    assert success is False


@pytest.mark.asyncio
async def test_alert_manager_flapping_and_failed_channel_retry() -> None:
    class FailingChannel:
        def __init__(self) -> None:
            self.call_count = 0

        async def send_alert(
            self, level: str, title: str, message: str, metadata: Any = None
        ) -> bool:
            self.call_count += 1
            return False

    failing_ch = FailingChannel()
    manager = AlertManager(channels=[failing_ch], deduplicate_s=300.0)
    sent_count_1 = await manager.notify(
        level="WARNING", title=ALERT_TITLE_DEGRADED, message="Fail 1"
    )
    assert sent_count_1 == 0
    assert failing_ch.call_count == 1

    sent_count_2 = await manager.notify(
        level="WARNING", title=ALERT_TITLE_DEGRADED, message="Fail 2"
    )
    assert sent_count_2 == 0
    assert failing_ch.call_count == 2

    class SuccessChannel:
        def __init__(self) -> None:
            self.delivered: list[tuple[str, str]] = []

        async def send_alert(
            self, level: str, title: str, message: str, metadata: Any = None
        ) -> bool:
            self.delivered.append((level, title))
            return True

    success_ch = SuccessChannel()
    mgr2 = AlertManager(channels=[success_ch], deduplicate_s=300.0)

    c1 = await mgr2.notify(level="WARNING", title=ALERT_TITLE_DEGRADED, message="Broker offline")
    assert c1 == 1

    c2 = await mgr2.notify(level="WARNING", title=ALERT_TITLE_DEGRADED, message="Broker offline")
    assert c2 == 0

    c3 = await mgr2.notify(level="INFO", title=ALERT_TITLE_RECOVERED, message="Broker online")
    assert c3 == 1

    c4 = await mgr2.notify(
        level="WARNING", title=ALERT_TITLE_DEGRADED, message="Broker offline again"
    )
    assert c4 == 1
    assert len(success_ch.delivered) == 3


def test_sensitive_data_filter_comprehensive_extras_and_json() -> None:
    sanitizer = SensitiveDataFilter()
    logger = logging.getLogger("test.sanitizer.comprehensive")

    rec_extra = logger.makeRecord(
        name="test.sanitizer.comprehensive",
        level=logging.INFO,
        fn="test.py",
        lno=1,
        msg="Log com extra sensível",
        args=(),
        exc_info=None,
    )
    rec_extra.__dict__["password"] = "SuperSecretPass1"
    rec_extra.__dict__["telegram_token"] = "SecretToken123"
    rec_extra.__dict__["balance"] = 10000.0
    rec_extra.__dict__["equity"] = 9850.50

    sanitizer.filter(rec_extra)
    assert rec_extra.__dict__["password"] == "***REDACTED***"
    assert rec_extra.__dict__["telegram_token"] == "***REDACTED***"
    assert rec_extra.__dict__["balance"] == "***REDACTED***"
    assert rec_extra.__dict__["equity"] == "***REDACTED***"

    rec_json = logger.makeRecord(
        name="test.sanitizer.comprehensive",
        level=logging.INFO,
        fn="test.py",
        lno=2,
        msg=(
            'Resposta da API: {"token": "my_api_token_99", "api_key": "AK_SECRET_123", "balance":'
            " 10234.55}"
        ),
        args=(),
        exc_info=None,
    )
    sanitizer.filter(rec_json)
    assert "my_api_token_99" not in rec_json.msg
    assert "AK_SECRET_123" not in rec_json.msg
    assert "10234.55" not in rec_json.msg
    assert "***REDACTED***" in rec_json.msg


def test_runner_kill_switch_alert_and_no_false_degraded() -> None:
    class SpyChannel:
        def __init__(self) -> None:
            self.alerts: list[tuple[str, str, str]] = []

        async def send_alert(
            self, level: str, title: str, message: str, metadata: Any = None
        ) -> bool:
            self.alerts.append((level, title, message))
            return True

    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)
    now = datetime.now(UTC)
    watchdog.update_tick(now)
    watchdog.beat_loop(now=now)

    state = HealthStatusState(broker_connected=True, trading_halted=False)
    spy = SpyChannel()
    mgr = AlertManager(channels=[spy], deduplicate_s=0.0)

    runner = WatchdogRunner(watchdog=watchdog, target=state, interval_s=0.02, alert_manager=mgr)
    runner.start()

    state.trading_halted = True
    assert _wait_for(lambda: any(a[1] == ALERT_TITLE_HALTED for a in spy.alerts))

    assert not any(ALERT_TITLE_DEGRADED in title for _, title, _ in spy.alerts)
    kill_alerts = [a for a in spy.alerts if a[1] == ALERT_TITLE_HALTED]
    assert len(kill_alerts) >= 1
    assert kill_alerts[0][0] == "ERROR"

    state.trading_halted = False
    assert _wait_for(
        lambda: ("INFO", ALERT_TITLE_RESUMED) in [(lvl, t) for lvl, t, _ in spy.alerts]
    )

    runner.stop()


def test_runner_initial_startup_halted_alert() -> None:
    class SpyChannel:
        def __init__(self) -> None:
            self.alerts: list[tuple[str, str, str]] = []

        async def send_alert(
            self, level: str, title: str, message: str, metadata: Any = None
        ) -> bool:
            self.alerts.append((level, title, message))
            return True

    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)
    now = datetime.now(UTC)
    watchdog.update_tick(now)
    watchdog.beat_loop(now=now)

    state = HealthStatusState(broker_connected=True, trading_halted=True)
    spy = SpyChannel()
    mgr = AlertManager(channels=[spy], deduplicate_s=0.0)

    runner = WatchdogRunner(watchdog=watchdog, target=state, interval_s=0.02, alert_manager=mgr)
    runner.start()

    assert _wait_for(
        lambda: ("ERROR", ALERT_TITLE_HALTED) in [(lvl, t) for lvl, t, _ in spy.alerts]
    )
    runner.stop()


@pytest.mark.asyncio
async def test_telegram_alert_channel_escapes_markdown_special_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, Any] = {}

    class DummyResponse:
        status = 200

        def __enter__(self) -> "DummyResponse":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    def mock_urlopen(req: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return DummyResponse()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    channel = TelegramAlertChannel(token="dummy_token", chat_id="123456")
    success = await channel.send_alert(
        level="WARNING",
        title=ALERT_TITLE_DEGRADED,
        message="Service degraded. Reasons: BROKER_DISCONNECTED, TICK_STALE, LOOP_STALE",
    )
    assert success is True
    sent_text = captured_payload.get("text", "")
    assert "BROKER\\_DISCONNECTED" in sent_text
    assert "TICK\\_STALE" in sent_text
    assert "LOOP\\_STALE" in sent_text


@pytest.mark.asyncio
async def test_alert_manager_and_runner_kill_switch_cycle_dedup() -> None:
    class RecordingChannel:
        def __init__(self) -> None:
            self.delivered: list[tuple[str, str]] = []

        async def send_alert(
            self, level: str, title: str, message: str, metadata: Any = None
        ) -> bool:
            self.delivered.append((level, title))
            return True

    channel = RecordingChannel()
    mgr = AlertManager(channels=[channel], deduplicate_s=300.0)

    c1 = await mgr.notify(level="ERROR", title=ALERT_TITLE_HALTED, message="Kill switch 1")
    assert c1 == 1

    c2 = await mgr.notify(
        level="ERROR", title=ALERT_TITLE_HALTED, message="Kill switch 1 duplicate"
    )
    assert c2 == 0

    c3 = await mgr.notify(level="INFO", title=ALERT_TITLE_RESUMED, message="Resume 1")
    assert c3 == 1

    c4 = await mgr.notify(level="ERROR", title=ALERT_TITLE_HALTED, message="Kill switch 2")
    assert c4 == 1

    c5 = await mgr.notify(level="INFO", title=ALERT_TITLE_RESUMED, message="Resume 2")
    assert c5 == 1

    assert channel.delivered == [
        ("ERROR", ALERT_TITLE_HALTED),
        ("INFO", ALERT_TITLE_RESUMED),
        ("ERROR", ALERT_TITLE_HALTED),
        ("INFO", ALERT_TITLE_RESUMED),
    ]


def test_json_log_formatter_sanitizes_exception_traceback() -> None:
    formatter = JsonLogFormatter()
    logger = logging.getLogger("test.json.exc_sanitize")

    try:
        raise ValueError(
            "MT5 connection failed with password=SuperSecretPass123 and token: my_secret_token_abc"
        )
    except ValueError:
        exc_info = sys.exc_info()

    rec = logger.makeRecord(
        name="test.json.exc_sanitize",
        level=logging.ERROR,
        fn="test.py",
        lno=10,
        msg="Erro de conexão com broker",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(rec)
    data = json.loads(output)
    assert "SuperSecretPass123" not in data["exception"]
    assert "my_secret_token_abc" not in data["exception"]
    assert "***REDACTED***" in data["exception"]


def test_watchdog_logs_info_on_trading_halted_alone(caplog: pytest.LogCaptureFixture) -> None:
    broker = MockBroker(connected=True)
    watchdog = Watchdog(broker=broker)
    now = datetime.now(UTC)
    watchdog.update_tick(now)
    watchdog.beat_loop(now=now)

    with caplog.at_level(logging.INFO):
        health = watchdog.check_health(now=now, trading_halted=True)

    assert health.status == "degraded"
    assert "TRADING_HALTED" in [r.value for r in health.reasons]
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    info_records = [
        rec for rec in caplog.records if getattr(rec, "event", None) == "trading_halted"
    ]
    assert len(info_records) == 1


def test_sensitive_data_filter_masks_exception_traceback_in_text_mode() -> None:
    from xau_bot.cli.run import configure_logging

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    buf = io.StringIO()
    try:
        configure_logging(log_json=False, log_level="DEBUG", stream=buf)
        logger = logging.getLogger("test.traceback.leak")

        try:
            raise ValueError(
                "Connection failed with password=SuperSecretPass123 and token: my_secret_token_abc"
            )
        except ValueError:
            logger.exception("Falha de conexão com broker")

        output = buf.getvalue()
        assert "SuperSecretPass123" not in output
        assert "my_secret_token_abc" not in output
        assert "***REDACTED***" in output
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_sensitive_data_filter_args_formatting_fallback() -> None:
    sanitizer = SensitiveDataFilter()
    logger = logging.getLogger("test.sanitizer.fallback")

    rec = logger.makeRecord(
        name="test.sanitizer.fallback",
        level=logging.INFO,
        fn="test.py",
        lno=1,
        msg="bad format %d",
        args=("password=secret_val",),
        exc_info=None,
    )
    assert sanitizer.filter(rec) is True
    assert rec.args == ()
    assert "secret_val" not in rec.msg
    assert "***REDACTED***" in rec.msg
