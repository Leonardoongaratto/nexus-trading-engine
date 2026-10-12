import json
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from http import HTTPStatus
from typing import Any

import pytest

import xau_bot.server
from xau_bot.config import Settings
from xau_bot.server import HealthStatusState, create_server


def test_health_status_state_defaults() -> None:
    state = HealthStatusState()
    code, health = state.get_health()
    assert code == HTTPStatus.OK
    assert health["status"] == "ok"
    assert health["broker_connected"] is False
    assert health["trading_halted"] is False
    assert health["last_tick_age_s"] == 0.0

    status_code, status_data = state.get_status()
    assert status_code == HTTPStatus.OK
    assert status_data["positions"] == []
    assert status_data["daily_pnl"] == 0.0
    assert status_data["kill_switches_active"] is False
    assert status_data["last_setup"] is None


def test_health_status_state_degraded() -> None:
    state = HealthStatusState(broker_connected=False, watchdog_reasons=("BROKER_DISCONNECTED",))
    code, health = state.get_health()
    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert health["status"] == "degraded"
    assert "BROKER_DISCONNECTED" in health["reasons"]


def test_health_status_state_halted() -> None:
    state = HealthStatusState(broker_connected=True, trading_halted=True)
    code, health = state.get_health()
    assert code == HTTPStatus.OK
    assert health["status"] == "halted"
    assert health["trading_halted"] is True
    assert health["reasons"] == ["TRADING_HALTED"]


@pytest.fixture
def running_server() -> Generator[tuple[str, HealthStatusState], None, None]:
    state = HealthStatusState()
    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{actual_port}"
    try:
        yield base_url, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_server_get_health(running_server: tuple[str, HealthStatusState]) -> None:
    base_url, _ = running_server
    with urllib.request.urlopen(f"{base_url}/health") as response:
        assert response.status == HTTPStatus.OK
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["broker_connected"] is False
        assert data["trading_halted"] is False


def test_server_get_health_with_query_params(
    running_server: tuple[str, HealthStatusState],
) -> None:
    base_url, _ = running_server
    with urllib.request.urlopen(f"{base_url}/health?foo=bar&baz=1") as response:
        assert response.status == HTTPStatus.OK
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "ok"


def test_server_get_health_degraded(running_server: tuple[str, HealthStatusState]) -> None:
    base_url, state = running_server
    state.watchdog_reasons = ("BROKER_DISCONNECTED",)

    try:
        urllib.request.urlopen(f"{base_url}/health")
        pytest.fail("Expected HTTPError 503")
    except urllib.error.HTTPError as err:
        assert err.code == HTTPStatus.SERVICE_UNAVAILABLE
        data = json.loads(err.read().decode("utf-8"))
        assert data["status"] == "degraded"
        assert "BROKER_DISCONNECTED" in data["reasons"]


def test_server_get_health_halted(running_server: tuple[str, HealthStatusState]) -> None:
    base_url, state = running_server
    state.watchdog_reasons = ()
    state.trading_halted = True

    with urllib.request.urlopen(f"{base_url}/health") as response:
        assert response.status == HTTPStatus.OK
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "halted"
        assert data["trading_halted"] is True
        assert data["reasons"] == ["TRADING_HALTED"]


def test_server_get_status(running_server: tuple[str, HealthStatusState]) -> None:
    base_url, _ = running_server
    with urllib.request.urlopen(f"{base_url}/status") as response:
        assert response.status == HTTPStatus.OK
        data = json.loads(response.read().decode("utf-8"))
        assert data["positions"] == []
        assert data["daily_pnl"] == 0.0
        assert data["kill_switches_active"] is False


def test_server_not_found(running_server: tuple[str, HealthStatusState]) -> None:
    base_url, _ = running_server
    try:
        urllib.request.urlopen(f"{base_url}/nonexistent")
        pytest.fail("Expected HTTPError 404")
    except urllib.error.HTTPError as err:
        assert err.code == HTTPStatus.NOT_FOUND
        data = json.loads(err.read().decode("utf-8"))
        assert data["error"] == "not found"


def test_run_server_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("signal.signal", lambda sig, handler: None)

    settings = Settings(server_host="127.0.0.1", server_port=0)
    closed = False

    class MockServer:
        server_address = ("127.0.0.1", 8000)

        def serve_forever(self) -> None:
            pass

        def server_close(self) -> None:
            nonlocal closed
            closed = True

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(
        xau_bot.server,
        "create_server",
        lambda host, port, state: MockServer(),
    )
    xau_bot.server.run_server(settings)
    assert closed is True


def test_run_server_shutdown_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(server_host="127.0.0.1", server_port=0)
    captured_handler: Any = None

    def fake_signal(sig: int, handler: Any) -> None:
        nonlocal captured_handler
        captured_handler = handler

    monkeypatch.setattr("signal.signal", fake_signal)

    class MockServer:
        server_address = ("127.0.0.1", 8000)

        def __init__(self) -> None:
            self.shutdown_called = False
            self.close_called = False
            self.shutdown_event = threading.Event()

        def serve_forever(self) -> None:
            pass

        def server_close(self) -> None:
            self.close_called = True

        def shutdown(self) -> None:
            self.shutdown_called = True
            self.shutdown_event.set()

    mock_server = MockServer()
    monkeypatch.setattr(
        xau_bot.server,
        "create_server",
        lambda host, port, state: mock_server,
    )

    xau_bot.server.run_server(settings)

    assert captured_handler is not None
    captured_handler(15, None)
    assert mock_server.shutdown_event.wait(timeout=2.0) is True
    assert mock_server.shutdown_called is True
    assert mock_server.close_called is True


def test_build_guardian_bridge(tmp_path: Any) -> None:
    from xau_bot.server import build_guardian_bridge

    # Sem diretorio comum configurado
    s_none = Settings(guardian_common_dir=None)
    assert build_guardian_bridge(s_none) is None

    # Com diretorio comum configurado
    s_valid = Settings(
        guardian_common_dir=str(tmp_path),
        guardian_heartbeat_interval_s=2.0,
        guardian_heartbeat_timeout_s=15.0,
        guardian_magic=123456,
    )
    bridge = build_guardian_bridge(s_valid)
    assert bridge is not None
    assert bridge.config.common_dir == tmp_path
    assert bridge.config.heartbeat_interval_s == 2.0
    assert bridge.config.heartbeat_timeout_s == 15.0
    assert bridge.config.magic_number == 123456


def test_run_server_guardian_bridge_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import time

    from xau_bot.broker.sim_adapter import SimBrokerAdapter
    from xau_bot.runtime.watchdog import Watchdog

    dir1 = tmp_path / "case1"
    dir1.mkdir(parents=True, exist_ok=True)
    settings1 = Settings(
        server_host="127.0.0.1",
        server_port=0,
        guardian_common_dir=str(dir1),
        guardian_heartbeat_interval_s=0.5,
    )

    class FastMockServer1:
        server_address = ("127.0.0.1", 0)

        def serve_forever(self) -> None:
            hb = dir1 / "xau_heartbeat.json"
            for _ in range(50):
                if hb.is_file():
                    break
                time.sleep(0.02)

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(
        xau_bot.server,
        "create_server",
        lambda host, port, state: FastMockServer1(),
    )
    xau_bot.server.run_server(settings1, watchdog=None)
    assert (dir1 / "xau_heartbeat.json").is_file()

    dir2 = tmp_path / "case2"
    dir2.mkdir(parents=True, exist_ok=True)
    settings2 = Settings(
        server_host="127.0.0.1",
        server_port=0,
        guardian_common_dir=str(dir2),
        guardian_heartbeat_interval_s=0.5,
    )

    class FastMockServer2:
        server_address = ("127.0.0.1", 0)

        def serve_forever(self) -> None:
            hb = dir2 / "xau_heartbeat.json"
            for _ in range(50):
                if hb.is_file():
                    break
                time.sleep(0.02)

        def server_close(self) -> None:
            pass

    monkeypatch.setattr(
        xau_bot.server,
        "create_server",
        lambda host, port, state: FastMockServer2(),
    )
    broker = SimBrokerAdapter()
    watchdog = Watchdog(broker=broker)
    xau_bot.server.run_server(settings2, watchdog=watchdog)
    assert (dir2 / "xau_heartbeat.json").is_file()
