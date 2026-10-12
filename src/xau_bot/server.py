import asyncio
import json
import logging
import math
import signal
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from xau_bot.config import Settings

if TYPE_CHECKING:
    from xau_bot.runtime.models import WatchdogHealth
    from xau_bot.runtime.ops.alerts import AlertManager
    from xau_bot.runtime.watchdog import Watchdog

logger = logging.getLogger("xau_bot.server")


class HealthStatusState:
    def __init__(
        self,
        broker_connected: bool = False,
        trading_halted: bool = False,
        watchdog_reasons: tuple[str, ...] = (),
        execution_mode: str = "semiauto",
        proposal_manager: Any | None = None,
        order_executor: Any | None = None,
        runtime_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.broker_connected: bool = broker_connected
        self.trading_halted: bool = trading_halted
        self.last_tick_age_s: float = 0.0
        self.positions: list[dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.kill_switches_active: bool = False
        self.last_setup: dict[str, Any] | None = None
        self.watchdog_synced: bool = False
        self.watchdog_healthy: bool | None = None
        self.watchdog_reasons: tuple[str, ...] = watchdog_reasons
        self.execution_mode: str = execution_mode
        self.proposal_manager: Any | None = proposal_manager
        self.order_executor: Any | None = order_executor
        self.runtime_loop: asyncio.AbstractEventLoop | None = runtime_loop

    def apply_watchdog_health(self, health: "WatchdogHealth") -> None:
        with self._lock:
            self.watchdog_synced = True
            self.watchdog_healthy = bool(health.is_healthy)
            self.broker_connected = bool(health.broker_connected)
            self.last_tick_age_s = float(health.last_tick_age_s)
            self.watchdog_reasons = tuple(r.value for r in health.reasons)

    def set_trading_halted(self, halted: bool) -> None:
        with self._lock:
            self.trading_halted = bool(halted)

    def get_health(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            infra_reasons = [r for r in self.watchdog_reasons if r != "TRADING_HALTED"]
            infra_healthy = len(infra_reasons) == 0

            reasons = list(infra_reasons)
            if self.trading_halted:
                reasons.append("TRADING_HALTED")

            if not infra_healthy:
                status_code = HTTPStatus.SERVICE_UNAVAILABLE
                status_str = "degraded"
            elif self.trading_halted:
                status_code = HTTPStatus.OK
                status_str = "halted"
            else:
                status_code = HTTPStatus.OK
                status_str = "ok"

            last_tick_age = None if math.isinf(self.last_tick_age_s) else self.last_tick_age_s
            payload: dict[str, Any] = {
                "status": status_str,
                "broker_connected": self.broker_connected,
                "last_tick_age_s": last_tick_age,
                "trading_halted": self.trading_halted,
            }
            if reasons:
                payload["reasons"] = reasons
            return int(status_code), payload

    def get_status(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            if self.proposal_manager is not None:
                self.proposal_manager.check_expirations()
            proposals_stats = (
                self.proposal_manager.get_stats()
                if self.proposal_manager is not None
                else {
                    "total": 0,
                    "advisory": 0,
                    "pending": 0,
                    "approved": 0,
                    "rejected": 0,
                    "expired": 0,
                    "dispatched": 0,
                    "dispatch_failed": 0,
                    "cancelled": 0,
                }
            )
            payload = {
                "broker_connected": self.broker_connected,
                "trading_halted": self.trading_halted,
                "positions": list(self.positions),
                "daily_pnl": self.daily_pnl,
                "kill_switches_active": self.kill_switches_active,
                "last_setup": self.last_setup,
                "execution_mode": self.execution_mode,
                "proposals": proposals_stats,
            }
            return int(HTTPStatus.OK), payload


class HealthCheckHandler(BaseHTTPRequestHandler):
    state: HealthStatusState = HealthStatusState()

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/health":
            code, payload = self.state.get_health()
            self._send_json(code, payload)
        elif path == "/status":
            code, payload = self.state.get_status()
            self._send_json(code, payload)
        elif path == "/api/proposals":
            if self.state.proposal_manager is not None:
                self.state.proposal_manager.check_expirations()
                query_params = urllib.parse.parse_qs(parsed_url.query)
                status_filter = query_params.get("status", [None])[0]
                status_enum = None
                if status_filter:
                    try:
                        from xau_bot.advisor.modes import ProposalStatus

                        status_enum = ProposalStatus(status_filter.upper())
                    except ValueError:
                        pass
                raw_props = self.state.proposal_manager.get_proposals(status=status_enum)
                raw_props.sort(key=lambda p: p.created_at, reverse=True)
                props = [p.to_dict() for p in raw_props]
            else:
                props = []
            self._send_json(HTTPStatus.OK, {"proposals": props})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        parts = [p for p in path.strip("/").split("/") if p]

        if len(parts) == 4 and parts[0] == "api" and parts[1] == "proposals":
            prop_id = parts[2]
            action = parts[3]

            if action == "approve":
                if self.state.proposal_manager is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "proposal not found"})
                    return

                ok, reason, prop = self.state.proposal_manager.approve_proposal(prop_id)
                if not ok:
                    if reason == "invalid_mode":
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "analyst proposals cannot be approved or dispatched"},
                        )
                    elif reason == "not_found":
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "proposal not found"})
                    elif reason in ("expired", "invalid_state"):
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "proposal already expired or rejected"},
                        )
                    else:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": reason})
                    return

                if (
                    self.state.order_executor is not None
                    and prop is not None
                    and prop.intent is not None
                ):
                    loop = self.state.runtime_loop
                    if loop is None:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None

                    try:
                        if loop and loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                self.state.order_executor.execute_intent(prop.intent), loop
                            )
                            res = future.result(timeout=10.0)
                        else:
                            res = asyncio.run(self.state.order_executor.execute_intent(prop.intent))

                        if getattr(res, "success", False):
                            ticket = getattr(res, "ticket", None)
                            self.state.proposal_manager.mark_dispatched(
                                prop_id, order_ticket=ticket
                            )
                            self._send_json(
                                HTTPStatus.OK,
                                {
                                    "status": "approved",
                                    "proposal_id": prop_id,
                                    "order_dispatched": True,
                                },
                            )
                        else:
                            err_msg = (
                                getattr(res, "error_message", None)
                                or f"Retcode {getattr(res, 'retcode', 'unknown')}"
                            )
                            self.state.proposal_manager.mark_dispatch_failed(
                                prop_id, reason=str(err_msg)
                            )
                            self._send_json(
                                HTTPStatus.BAD_GATEWAY,
                                {
                                    "status": "dispatch_failed",
                                    "proposal_id": prop_id,
                                    "error": str(err_msg),
                                },
                            )
                    except Exception as exc:
                        err_str = str(exc)
                        logger.warning(
                            "Falha ao despachar ordem para proposta aprovada %s: %s",
                            prop_id,
                            err_str,
                        )
                        self.state.proposal_manager.mark_dispatch_failed(prop_id, reason=err_str)
                        self._send_json(
                            HTTPStatus.BAD_GATEWAY,
                            {
                                "status": "dispatch_failed",
                                "proposal_id": prop_id,
                                "error": err_str,
                            },
                        )
                    return

                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "approved",
                        "proposal_id": prop_id,
                        "order_dispatched": False,
                    },
                )
                return

            if action == "reject":
                if self.state.proposal_manager is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "proposal not found"})
                    return

                raw_cl = self.headers.get("Content-Length", "0")
                try:
                    content_len = int(raw_cl)
                    if content_len < 0 or content_len > 1_000_000:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                        return
                except ValueError:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST, {"error": "malformed content length header"}
                    )
                    return

                reason = "rejected_by_operator"
                if content_len > 0:
                    body = self.rfile.read(content_len)
                    try:
                        data = json.loads(body.decode("utf-8"))
                        if isinstance(data, dict) and "reason" in data:
                            reason = str(data["reason"])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json body"})
                        return

                ok, r_status, prop = self.state.proposal_manager.reject_proposal(
                    prop_id, reason=reason
                )
                if not ok:
                    if r_status == "not_found":
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "proposal not found"})
                    elif r_status in ("expired", "invalid_state"):
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "proposal already expired or resolved"},
                        )
                    else:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": r_status})
                    return

                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "rejected",
                        "proposal_id": prop_id,
                    },
                )
                return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    state: HealthStatusState | None = None,
) -> ThreadingHTTPServer:
    active_state = state or HealthStatusState()

    class BoundRequestHandler(HealthCheckHandler):
        pass

    BoundRequestHandler.state = active_state
    return ThreadingHTTPServer((host, port), BoundRequestHandler)


def build_guardian_bridge(
    settings: Settings,
    risk_guard: Any | None = None,
    watchdog: Any | None = None,
) -> Any | None:
    if not settings.guardian_common_dir:
        return None

    from pathlib import Path

    from xau_bot.guardian.bridge import GuardianBridge
    from xau_bot.guardian.models import GuardianConfig
    from xau_bot.risk.guards import RiskGuard
    from xau_bot.risk.models import DailyRiskLimits

    active_risk_guard = risk_guard or RiskGuard(
        limits=DailyRiskLimits(
            max_daily_loss_percent=settings.max_daily_loss_pct / 100.0,
            max_daily_drawdown_percent=settings.max_drawdown_pct / 100.0,
            max_consecutive_losses=settings.max_consecutive_losses,
        )
    )
    guardian_cfg = GuardianConfig(
        common_dir=Path(settings.guardian_common_dir),
        heartbeat_interval_s=settings.guardian_heartbeat_interval_s,
        heartbeat_timeout_s=settings.guardian_heartbeat_timeout_s,
        calendar_max_age_h=settings.guardian_calendar_max_age_h,
        magic_number=settings.guardian_magic,
    )
    return GuardianBridge(
        config=guardian_cfg,
        risk_guard=active_risk_guard,
        watchdog=watchdog,
    )


def run_server(
    settings: Settings | None = None,
    watchdog: "Watchdog | None" = None,
    alert_manager: "AlertManager | None" = None,
    risk_guard: Any | None = None,
    guardian_bridge: Any | None = None,
    proposal_manager: Any | None = None,
    order_executor: Any | None = None,
    state: HealthStatusState | None = None,
) -> None:
    current_settings = settings or Settings()
    mode_str = getattr(current_settings, "execution_mode", "semiauto")
    if hasattr(mode_str, "value"):
        mode_str = mode_str.value

    active_state = state or HealthStatusState(
        execution_mode=str(mode_str),
        proposal_manager=proposal_manager,
        order_executor=order_executor,
    )
    state = active_state

    active_bridge = guardian_bridge
    if active_bridge is None:
        active_bridge = build_guardian_bridge(
            current_settings,
            risk_guard=risk_guard,
            watchdog=watchdog,
        )

    runner = None
    if watchdog is not None:
        from xau_bot.runtime.runner import WatchdogRunner

        runner = WatchdogRunner(
            watchdog=watchdog,
            target=state,
            interval_s=watchdog.config.check_interval_s,
            alert_manager=alert_manager,
            guardian_bridge=active_bridge,
        )
        runner.start()
    elif active_bridge is not None:
        active_bridge.start()

    server = create_server(current_settings.server_host, current_settings.server_port, state)

    def _handle_shutdown(signum: int, frame: Any) -> None:
        logger.info("Signal %d received, shutting down server...", signum)
        if runner is not None:
            runner.stop()
        if active_bridge is not None and runner is None:
            active_bridge.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

    host, port = str(server.server_address[0]), int(server.server_address[1])
    logger.info("Serving at http://%s:%d", host, port)
    try:
        server.serve_forever()
    finally:
        if runner is not None:
            runner.stop()
        if active_bridge is not None and runner is None:
            active_bridge.stop()
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_server()
