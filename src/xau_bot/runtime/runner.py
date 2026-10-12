from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

from xau_bot.runtime.models import HealthSyncTarget
from xau_bot.runtime.ops.alerts import (
    ALERT_TITLE_DEGRADED,
    ALERT_TITLE_HALTED,
    ALERT_TITLE_RECOVERED,
    ALERT_TITLE_RESUMED,
)
from xau_bot.runtime.watchdog import Watchdog

if TYPE_CHECKING:
    from xau_bot.guardian.bridge import GuardianBridge
    from xau_bot.runtime.ops.alerts import AlertManager

logger = logging.getLogger("xau_bot.runtime.runner")


class WatchdogRunner:
    def __init__(
        self,
        watchdog: Watchdog,
        target: HealthSyncTarget,
        interval_s: float = 1.0,
        alert_manager: AlertManager | None = None,
        guardian_bridge: GuardianBridge | None = None,
    ) -> None:
        self.watchdog: Watchdog = watchdog
        self.target: HealthSyncTarget = target
        self.interval_s: float = max(0.1, interval_s)
        self.alert_manager: AlertManager | None = alert_manager
        self.guardian_bridge: GuardianBridge | None = guardian_bridge
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_infra_healthy: bool | None = None
        self._last_trading_halted: bool | None = None
        self._alert_queue: queue.Queue[tuple[str, str, str, dict[str, Any] | None, str | None]] = (
            queue.Queue()
        )
        self._alert_worker_thread: threading.Thread | None = None

    def _dispatch_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        dedup_group: str | None = None,
    ) -> None:
        if self.alert_manager is None:
            return
        self._alert_queue.put((level, title, message, metadata, dedup_group))

    def _process_alerts(self) -> None:
        while not self._stop_event.is_set() or not self._alert_queue.empty():
            try:
                item = self._alert_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            level, title, message, metadata, dedup_group = item
            mgr = self.alert_manager
            if mgr is not None:
                try:
                    asyncio.run(
                        mgr.notify(
                            level=level,
                            title=title,
                            message=message,
                            metadata=metadata,
                            dedup_group=dedup_group,
                        )
                    )
                except Exception as alert_err:
                    logger.warning("Failed to dispatch alert %s: %s", title, alert_err)
            self._alert_queue.task_done()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        if self._alert_worker_thread is None or not self._alert_worker_thread.is_alive():
            self._alert_worker_thread = threading.Thread(
                target=self._process_alerts, daemon=True, name="AlertWorker"
            )
            self._alert_worker_thread.start()

        try:
            initial_health = self.watchdog.sync_server_state(self.target)
            infra_reasons = [r.value for r in initial_health.reasons if r.value != "TRADING_HALTED"]
            self._last_infra_healthy = len(infra_reasons) == 0
            self._last_trading_halted = bool(self.target.trading_halted)

            if not self._last_infra_healthy:
                reasons_str = ", ".join(infra_reasons)
                self._dispatch_alert(
                    level="WARNING",
                    title=ALERT_TITLE_DEGRADED,
                    message=f"Service degraded at startup. Reasons: {reasons_str}",
                    metadata={"reasons": infra_reasons},
                    dedup_group="infra",
                )
            if self._last_trading_halted:
                self._dispatch_alert(
                    level="ERROR",
                    title=ALERT_TITLE_HALTED,
                    message="Trading halted at startup by kill switch.",
                    dedup_group="trading",
                )
        except Exception as err:
            logger.error("Initial watchdog sync failed: %s", err)

        self._thread = threading.Thread(target=self._run, daemon=True, name="WatchdogRunner")
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        if self._alert_worker_thread is not None and self._alert_worker_thread.is_alive():
            self._alert_worker_thread.join(timeout=timeout_s)

    def _run(self) -> None:
        logger.info("WatchdogRunner started (interval=%.2fs)", self.interval_s)
        while not self._stop_event.is_set():
            try:
                health = self.watchdog.sync_server_state(self.target)
                infra_reasons = [r.value for r in health.reasons if r.value != "TRADING_HALTED"]
                infra_healthy = len(infra_reasons) == 0
                trading_halted = bool(self.target.trading_halted)

                if (
                    self._last_infra_healthy is not None
                    and self._last_infra_healthy != infra_healthy
                ):
                    if not infra_healthy:
                        reasons_str = ", ".join(infra_reasons)
                        self._dispatch_alert(
                            level="WARNING",
                            title=ALERT_TITLE_DEGRADED,
                            message=f"Service degraded. Reasons: {reasons_str}",
                            metadata={"reasons": infra_reasons},
                            dedup_group="infra",
                        )
                    else:
                        self._dispatch_alert(
                            level="INFO",
                            title=ALERT_TITLE_RECOVERED,
                            message="Service health restored to OK",
                            dedup_group="infra",
                        )
                self._last_infra_healthy = infra_healthy

                if (
                    self._last_trading_halted is not None
                    and self._last_trading_halted != trading_halted
                ):
                    if trading_halted:
                        self._dispatch_alert(
                            level="ERROR",
                            title=ALERT_TITLE_HALTED,
                            message="Trading halted by daily kill switch or risk limit.",
                            dedup_group="trading",
                        )
                    else:
                        self._dispatch_alert(
                            level="INFO",
                            title=ALERT_TITLE_RESUMED,
                            message="Trading pause cleared, operations resumed.",
                            dedup_group="trading",
                        )
                self._last_trading_halted = trading_halted

                if not health.broker_connected:
                    try:
                        asyncio.run(self.watchdog.maybe_reconnect())
                    except Exception as err:
                        logger.warning(
                            "Reconnection attempt failed in runner: %s",
                            err,
                            extra={"event": "runner_reconnect_failed"},
                        )

                if self.guardian_bridge is not None:
                    try:
                        self.guardian_bridge.pulse(
                            status="ok" if infra_healthy else "degraded",
                            trading_halted=trading_halted,
                        )
                        sig = self.guardian_bridge.check_guardian_signal()
                        if sig is not None:
                            if hasattr(self.target, "trading_halted"):
                                self.target.trading_halted = True
                            if hasattr(self.target, "kill_switches_active"):
                                self.target.kill_switches_active = True
                        elif not self.guardian_bridge.killswitch_file.is_file():
                            self.guardian_bridge.reset_processed_signal()
                    except Exception as g_err:
                        logger.warning(
                            "Guardian sync failed in runner: %s",
                            g_err,
                            extra={"event": "runner_guardian_sync_failed"},
                        )

            except Exception as err:
                logger.error(
                    "Error in watchdog sync cycle: %s",
                    err,
                    extra={"event": "watchdog_sync_cycle_error"},
                )

            self._stop_event.wait(timeout=self.interval_s)
        logger.info("WatchdogRunner stopped cleanly")
