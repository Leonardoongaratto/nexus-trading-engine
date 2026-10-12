import argparse
import asyncio
import io
import json
import logging
import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from xau_bot.advisor.modes import ExecutionMode
from xau_bot.broker.mt5_adapter import Mt5AdapterConfig, Mt5BrokerAdapter
from xau_bot.broker.sim_adapter import SimBrokerAdapter
from xau_bot.cli.main import build_parser, main
from xau_bot.cli.run import (
    configure_logging,
    create_alert_manager,
    create_broker_adapter,
    run_bot_cli,
)
from xau_bot.config import Settings
from xau_bot.domain.models import Timeframe
from xau_bot.filters.sessions import SessionFilter
from xau_bot.runtime.ops.alerts import ConsoleAlertChannel, TelegramAlertChannel
from xau_bot.runtime.runner import WatchdogRunner
from xau_bot.runtime.watchdog import Watchdog
from xau_bot.server import HealthStatusState, create_server, run_server


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_cli_parser_run_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"
    assert args.broker == "sim"
    assert args.host is None
    assert args.port is None
    assert args.log_json is False
    assert args.log_level == "INFO"


def test_cli_parser_run_custom_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--broker",
            "mt5",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--log-json",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.broker == "mt5"
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.log_json is True
    assert args.log_level == "DEBUG"


def test_create_broker_adapter_sim() -> None:
    settings = Settings()
    broker = create_broker_adapter("sim", settings)
    assert isinstance(broker, SimBrokerAdapter)


def test_create_broker_adapter_mt5() -> None:
    settings = Settings(
        mt5_login=SecretStr("123456"),
        mt5_password=SecretStr("pass"),
        mt5_server="DemoServer",
    )
    broker = create_broker_adapter("mt5", settings)
    assert isinstance(broker, Mt5BrokerAdapter)


def test_create_broker_adapter_invalid() -> None:
    settings = Settings()
    with pytest.raises(ValueError, match="Adaptador de broker não suportado"):
        create_broker_adapter("unknown", settings)


def test_create_alert_manager_console_only() -> None:
    settings = Settings()
    stdout = io.StringIO()
    manager = create_alert_manager(settings, stdout)
    assert len(manager.channels) == 1
    assert isinstance(manager.channels[0], ConsoleAlertChannel)


def test_create_alert_manager_with_telegram() -> None:
    settings = Settings(
        telegram_token=SecretStr("token123"),
        telegram_chat_id=SecretStr("chat456"),
    )
    stdout = io.StringIO()
    manager = create_alert_manager(settings, stdout)
    assert len(manager.channels) == 2
    assert isinstance(manager.channels[0], ConsoleAlertChannel)
    assert isinstance(manager.channels[1], TelegramAlertChannel)


def test_configure_logging_standard_and_json() -> None:
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        stream = io.StringIO()
        configure_logging(log_json=False, log_level="INFO", stream=stream)
        logger = logging.getLogger("test.cli.standard")
        logger.info("Teste log padrão com password=secret123")
        output = stream.getvalue()
        assert "secret123" not in output
        assert "***REDACTED***" in output

        stream_json = io.StringIO()
        configure_logging(log_json=True, log_level="DEBUG", stream=stream_json)
        logger.debug("Teste log JSON")
        output_json = stream_json.getvalue()
        assert '"level": "DEBUG"' in output_json
        assert '"message": "Teste log JSON"' in output_json
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def test_run_bot_cli_invokes_run_server() -> None:
    args = argparse.Namespace(
        broker="sim",
        host="127.0.0.1",
        port=8080,
        log_json=False,
        log_level="INFO",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with patch("xau_bot.cli.run.run_server") as mock_run:
        exit_code = run_bot_cli(args, stdout=stdout, stderr=stderr)
        assert exit_code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["settings"].server_host == "127.0.0.1"
        assert kwargs["settings"].server_port == 8080
        assert kwargs["watchdog"] is not None
        assert kwargs["alert_manager"] is not None


def test_run_bot_cli_mt5_branch_invokes_run_server() -> None:
    args = argparse.Namespace(
        broker="mt5",
        host="127.0.0.1",
        port=8085,
        log_json=False,
        log_level="INFO",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with patch("xau_bot.cli.run.run_server") as mock_run:
        exit_code = run_bot_cli(args, stdout=stdout, stderr=stderr)
        assert exit_code == 0
        mock_run.assert_called_once()


def test_main_cli_dispatch_run() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    with patch("xau_bot.cli.run.run_server"):
        code = main(["run", "--broker", "sim", "--port", "8888"], stdout=stdout, stderr=stderr)
        assert code == 0


def test_run_bot_cli_keyboard_interrupt_handled() -> None:
    args = argparse.Namespace(
        broker="sim",
        host=None,
        port=None,
        log_json=False,
        log_level="INFO",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt):
        code = run_bot_cli(args, stdout=stdout, stderr=stderr)
        assert code == 0


def test_create_broker_adapter_mt5_invalid_login_sanitized() -> None:
    settings = Settings(mt5_login=SecretStr("not-a-number-secret-login"))
    with pytest.raises(
        ValueError, match="MT5_LOGIN deve ser um identificador numérico válido"
    ) as exc_info:
        create_broker_adapter("mt5", settings)
    assert "not-a-number-secret-login" not in str(exc_info.value)


def test_run_bot_cli_non_loopback_host_warning() -> None:
    args = argparse.Namespace(
        broker="sim",
        host="0.0.0.0",
        port=8080,
        log_json=False,
        log_level="INFO",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    with patch("xau_bot.cli.run.run_server"):
        run_bot_cli(args, stdout=stdout, stderr=stderr)
    log_output = stderr.getvalue()
    assert "não-loopback" in log_output


class _TestSimulatedFeed:
    def __init__(self, watchdog: Watchdog, broker: Any, stop_event: threading.Event) -> None:
        self.watchdog = watchdog
        self.broker = broker
        self.stop_event = stop_event
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        asyncio.run(self.broker.connect())
        now = datetime.now(UTC)
        self.watchdog.update_tick(now)
        self.watchdog.beat_loop(now=now)

        def _loop() -> None:
            while not self.stop_event.is_set():
                t = datetime.now(UTC)
                self.watchdog.update_tick(t)
                self.watchdog.beat_loop(now=t)
                self.stop_event.wait(timeout=0.05)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def test_real_cli_run_sim_integration_health_200() -> None:
    settings = Settings()
    broker = SimBrokerAdapter()
    session_filter = SessionFilter(server_tz=settings.broker_server_tz)
    watchdog = Watchdog(broker=broker, session_filter=session_filter)
    state = HealthStatusState()
    stdout = io.StringIO()
    alert_mgr = create_alert_manager(settings, stdout=stdout)

    runner = WatchdogRunner(
        watchdog=watchdog, target=state, interval_s=0.05, alert_manager=alert_mgr
    )
    feed_stop = threading.Event()
    feed = _TestSimulatedFeed(watchdog=watchdog, broker=broker, stop_event=feed_stop)

    feed.start()
    runner.start()

    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{actual_port}"
    try:
        assert _wait_for(lambda: state.broker_connected is True)
        with urllib.request.urlopen(f"{base_url}/health") as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["broker_connected"] is True
            assert data["trading_halted"] is False
    finally:
        feed.stop()
        runner.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def test_real_cli_run_mt5_disconnected_resilient_ac1809(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    cfg = Mt5AdapterConfig(login=None, password=None, server="NonExistentServer", timeout_ms=500)
    broker = Mt5BrokerAdapter(config=cfg)
    session_filter = SessionFilter(server_tz=settings.broker_server_tz)
    watchdog = Watchdog(broker=broker, session_filter=session_filter)
    state = HealthStatusState()
    stdout = io.StringIO()
    alert_mgr = create_alert_manager(settings, stdout=stdout)

    reconnect_attempts = 0
    orig_maybe_reconnect = watchdog.maybe_reconnect

    async def spy_reconnect(*args: Any, **kwargs: Any) -> bool:
        nonlocal reconnect_attempts
        reconnect_attempts += 1
        return await orig_maybe_reconnect(*args, **kwargs)

    monkeypatch.setattr(watchdog, "maybe_reconnect", spy_reconnect)

    runner = WatchdogRunner(
        watchdog=watchdog, target=state, interval_s=0.05, alert_manager=alert_mgr
    )
    runner.start()

    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://127.0.0.1:{actual_port}"
    try:
        assert _wait_for(lambda: reconnect_attempts >= 1)
        try:
            urllib.request.urlopen(f"{base_url}/health")
            pytest.fail("Expected 503 degraded when MT5 is disconnected")
        except urllib.error.HTTPError as err:
            assert err.code == 503
            data = json.loads(err.read().decode("utf-8"))
            assert data["status"] == "degraded"
            assert data["broker_connected"] is False
            assert "BROKER_DISCONNECTED" in data["reasons"]
    finally:
        runner.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)


def test_run_server_with_watchdog_lifecycle() -> None:
    broker = SimBrokerAdapter()
    watchdog = Watchdog(broker=broker)
    settings = Settings(server_host="127.0.0.1", server_port=0)

    mock_server = MagicMock()
    mock_server.server_address = ("127.0.0.1", 0)

    shutdown_handler: Any = None

    def capture_signal(sig: int, handler: Any) -> Any:
        nonlocal shutdown_handler
        if sig == signal.SIGINT:
            shutdown_handler = handler

    with (
        patch("xau_bot.server.create_server", return_value=mock_server),
        patch("signal.signal", side_effect=capture_signal),
    ):
        mock_server.serve_forever.return_value = None
        run_server(settings=settings, watchdog=watchdog)

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()
        assert shutdown_handler is not None
        shutdown_handler(signal.SIGINT, None)
        mock_server.shutdown.assert_called()


def test_run_bot_cli_wires_and_starts_live_loop_runner() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    live_runner_started = False
    live_runner_stopped = False
    captured_engine_kwargs: dict[str, Any] = {}

    class MockLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_engine_kwargs
            captured_engine_kwargs = kwargs

    class MockLiveRunner:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            nonlocal live_runner_started
            live_runner_started = True

        def stop(self, timeout_s: float = 3.0) -> None:
            nonlocal live_runner_stopped
            live_runner_stopped = True

    with (
        patch("xau_bot.cli.run.LiveTradingEngine", MockLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner", MockLiveRunner),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert live_runner_started
        assert live_runner_stopped

        comp_filter = captured_engine_kwargs.get("composite_filter")
        assert comp_filter is not None
        assert comp_filter.session_filter is not None
        assert comp_filter.news_filter is not None
        assert comp_filter.spread_filter is not None


def test_create_broker_adapter_mt5_reads_server_tz_from_settings() -> None:
    settings = Settings(broker_server_tz="Europe/Athens")
    adapter = create_broker_adapter("mt5", settings)
    assert isinstance(adapter, Mt5BrokerAdapter)
    assert adapter._config.server_tz == "Europe/Athens"


def test_cli_run_wires_news_filter_with_max_age_and_calendar(tmp_path: Path) -> None:
    cal_file = tmp_path / "calendar.json"
    cal_file.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(guardian_common_dir=str(tmp_path), guardian_calendar_max_age_h=12.0)
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_filter: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_filter
            captured_filter = kwargs.get("composite_filter")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_filter is not None
        assert captured_filter.news_filter is not None
        assert captured_filter.news_filter.config.max_age_hours == 12.0


def test_cli_run_auto_mode_without_calendar_is_fail_closed() -> None:
    settings = Settings(
        execution_mode=ExecutionMode.AUTO,
        guardian_common_dir=None,
        guardian_calendar_max_age_h=24.0,
    )
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_filter: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_filter
            captured_filter = kwargs.get("composite_filter")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_filter is not None
        assert captured_filter.news_filter is not None
        assert captured_filter.news_filter.config.max_age_hours == 24.0
        now = datetime.now(UTC)
        result = captured_filter.news_filter.check(now)
        assert result.can_trade is False


def test_cli_run_wires_journal_repository_to_live_trading_engine(tmp_path: Path) -> None:
    db_file = tmp_path / "test_journal.db"
    settings = Settings(journal_db=str(db_file))
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_journal_repo: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_journal_repo
            captured_journal_repo = kwargs.get("journal_repo")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_journal_repo is not None
        assert captured_journal_repo.db_path == str(db_file)


def test_cli_run_shadow_params_invalid_json_aborts_with_exit_2() -> None:
    settings = Settings(shadow_params_json="{invalid_json_syntax")
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    stderr = io.StringIO()
    with patch("xau_bot.cli.run.Settings", return_value=settings):
        code = run_bot_cli(args, stderr=stderr)
        assert code == 2
        assert "JSON inválido em shadow_params_json" in stderr.getvalue()


def test_cli_run_shadow_params_invalid_spec_aborts_with_exit_2() -> None:
    settings_unknown = Settings(shadow_params_json='{"unknown_param": 10.0}')
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    stderr = io.StringIO()
    with patch("xau_bot.cli.run.Settings", return_value=settings_unknown):
        code = run_bot_cli(args, stderr=stderr)
        assert code == 2
        assert "Parâmetro desconhecido" in stderr.getvalue()

    settings_out_of_bounds = Settings(shadow_params_json='{"max_spread_atr_ratio": 99.0}')
    stderr_oob = io.StringIO()
    with patch("xau_bot.cli.run.Settings", return_value=settings_out_of_bounds):
        code = run_bot_cli(args, stderr=stderr_oob)
        assert code == 2
        assert "fora da faixa permitida" in stderr_oob.getvalue()

    settings_unsupported_key = Settings(shadow_params_json='{"risk_per_trade_pct": 1.0}')
    stderr_unsupported = io.StringIO()
    with patch("xau_bot.cli.run.Settings", return_value=settings_unsupported_key):
        code = run_bot_cli(args, stderr=stderr_unsupported)
        assert code == 2
        assert "Chaves não suportadas para execução em sombra" in stderr_unsupported.getvalue()


def test_cli_run_shadow_params_valid_wires_shadow_engine() -> None:
    valid_json = '{"max_spread_atr_ratio": 0.08}'
    settings = Settings(shadow_params_json=valid_json)
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_shadow_engine: Any = None
    captured_shadow_params: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_shadow_engine, captured_shadow_params
            captured_shadow_engine = kwargs.get("shadow_engine")
            captured_shadow_params = kwargs.get("shadow_params")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_shadow_engine is not None
        assert captured_shadow_params == {"max_spread_atr_ratio": 0.08}


def test_cli_parser_run_setup_stats_argument(tmp_path: Path) -> None:
    stats_file = tmp_path / "stats.json"
    parser = build_parser()
    args = parser.parse_args(["run", "--setup-stats", str(stats_file)])
    assert args.setup_stats == str(stats_file)


def test_cli_run_wires_stats_store(tmp_path: Path) -> None:
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "GLOBAL": {
                    "setup_type": "GLOBAL",
                    "sample_size": 250,
                    "win_count": 150,
                    "loss_count": 100,
                    "win_rate": 0.6,
                    "expectancy_r": 0.45,
                    "avg_win_r": 1.5,
                    "avg_loss_r": -1.0,
                    "profit_factor": 2.25,
                    "gate_1_validated": True,
                    "validation_warning": None,
                }
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(setup_stats_path=str(stats_file))
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_proposal_manager: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_proposal_manager
            captured_proposal_manager = kwargs.get("proposal_manager")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_proposal_manager is not None
        assert captured_proposal_manager.stats_store is not None
        stats = captured_proposal_manager.stats_store.get("GLOBAL")
        assert stats is not None
        assert stats.sample_size == 250


def test_cli_run_wires_stats_store_missing_file_fallback(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.json"
    settings = Settings(setup_stats_path=str(missing_file))
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_proposal_manager: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_proposal_manager
            captured_proposal_manager = kwargs.get("proposal_manager")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_proposal_manager is not None
        assert captured_proposal_manager.stats_store is not None
        assert captured_proposal_manager.stats_store.count() == 0


def test_cli_parser_run_llm_veto_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--llm-veto", "--llm-veto-active"])
    assert args.llm_veto is True
    assert args.llm_veto_active is True


def test_cli_run_wires_llm_veto_service() -> None:
    settings = Settings(
        llm_veto_enabled=True,
        llm_api_key=SecretStr("sk-dummy-test-key"),
        llm_veto_shadow_mode=True,
    )
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--host", "127.0.0.1", "--port", "0"])

    captured_proposal_manager: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_proposal_manager
            captured_proposal_manager = kwargs.get("proposal_manager")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_proposal_manager is not None
        assert captured_proposal_manager.veto_service is not None
        assert captured_proposal_manager.veto_service.shadow_mode is True
        assert captured_proposal_manager.veto_service.model == "gpt-4o-mini-2024-07-18"


def test_cli_run_llm_veto_active_without_api_key_aborts_fail_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        llm_api_key=None,
    )
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--llm-veto-active"])

    with patch("xau_bot.cli.run.Settings", return_value=settings):
        code = run_bot_cli(args)

    assert code == 2
    captured = capsys.readouterr()
    assert "LLM_API_KEY" in captured.err


def test_cli_run_llm_veto_shadow_without_api_key_warns_and_runs() -> None:
    settings = Settings(
        llm_api_key=None,
    )
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--llm-veto"])

    captured_proposal_manager: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_proposal_manager
            captured_proposal_manager = kwargs.get("proposal_manager")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_proposal_manager is not None
        assert captured_proposal_manager.veto_service is None


def test_cli_parser_run_symbol_and_crypto_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--symbol", "BTCUSD", "--crypto-24-7"])
    assert args.symbol == "BTCUSD"
    assert args.crypto_24_7 is True


def test_cli_run_passes_symbol_and_enables_24_7_for_crypto() -> None:
    settings = Settings()
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--symbol", "BTCUSD"])

    captured_symbol: str | None = None
    captured_session_filter: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_symbol
            captured_symbol = kwargs.get("symbol")

    def spy_session_filter(*args: Any, **kwargs: Any) -> SessionFilter:
        nonlocal captured_session_filter
        sf = SessionFilter(*args, **kwargs)
        captured_session_filter = sf
        return sf

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.SessionFilter", side_effect=spy_session_filter),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_symbol == "BTCUSD"
        assert captured_session_filter is not None
        assert captured_session_filter.enable_24_7 is True


def test_cli_run_explicit_crypto_24_7_flag() -> None:
    settings = Settings(symbol="XAUUSD")
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--crypto-24-7"])

    captured_session_filter: Any = None

    def spy_session_filter(*args: Any, **kwargs: Any) -> SessionFilter:
        nonlocal captured_session_filter
        sf = SessionFilter(*args, **kwargs)
        captured_session_filter = sf
        return sf

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.SessionFilter", side_effect=spy_session_filter),
        patch("xau_bot.cli.run.LiveTradingEngine"),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_session_filter is not None
        assert captured_session_filter.enable_24_7 is True


def test_cli_parser_run_timeframe_entry_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--timeframe-entry", "M1,M5"])
    assert args.timeframe_entry == "M1,M5"


def test_cli_run_passes_custom_timeframe_entry() -> None:
    settings = Settings()
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--timeframe-entry", "M1,M5"])

    captured_trigger_timeframe: Any = None

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_trigger_timeframe
            captured_trigger_timeframe = kwargs.get("trigger_timeframe")

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner"),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert captured_trigger_timeframe == (Timeframe.M1, Timeframe.M5)


def test_cli_run_multi_symbol_instantiates_engines_for_both() -> None:
    settings = Settings()
    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim", "--symbol", "BTCUSDm,XAUUSDm"])

    captured_engines: list[dict[str, Any]] = []
    captured_runner_kwargs: dict[str, Any] = {}

    class SpyLiveEngine:
        def __init__(self, **kwargs: Any) -> None:
            captured_engines.append(kwargs)

    class SpyLiveRunner:
        def __init__(self, **kwargs: Any) -> None:
            nonlocal captured_runner_kwargs
            captured_runner_kwargs = kwargs

        def start(self) -> None:
            pass

        def stop(self, timeout_s: float = 3.0) -> None:
            pass

    with (
        patch("xau_bot.cli.run.Settings", return_value=settings),
        patch("xau_bot.cli.run.LiveTradingEngine", SpyLiveEngine),
        patch("xau_bot.cli.run.LiveTradingLoopRunner", SpyLiveRunner),
        patch("xau_bot.cli.run.run_server", side_effect=KeyboardInterrupt),
    ):
        code = run_bot_cli(args)
        assert code == 0
        assert len(captured_engines) == 2
        assert captured_engines[0]["symbol"] == "BTCUSDm"
        assert captured_engines[0]["composite_filter"].session_filter.enable_24_7 is True
        assert captured_engines[0]["magic"] == 123450
        assert captured_engines[1]["symbol"] == "XAUUSDm"
        assert captured_engines[1]["composite_filter"].session_filter.enable_24_7 is False
        assert captured_engines[1]["magic"] == 123550
        assert len(captured_runner_kwargs["engines"]) == 2
