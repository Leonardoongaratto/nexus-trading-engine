import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from xau_bot.guardian.heartbeat import HeartbeatWriter
from xau_bot.guardian.models import GuardianConfig, GuardianSignal, HeartbeatPayload
from xau_bot.risk.guards import RiskGuard
from xau_bot.runtime.watchdog import Watchdog

logger = logging.getLogger("xau_bot.guardian.bridge")


class GuardianBridge:
    def __init__(
        self,
        config: GuardianConfig,
        risk_guard: RiskGuard | None = None,
        watchdog: Watchdog | None = None,
    ) -> None:
        self.config: GuardianConfig = config
        self.risk_guard: RiskGuard | None = risk_guard
        self.watchdog: Watchdog | None = watchdog
        self.writer: HeartbeatWriter = HeartbeatWriter(
            common_dir=self.config.common_dir,
            filename=self.config.heartbeat_filename,
        )
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()
        self._last_processed_signal_ts: datetime | None = None

    @property
    def killswitch_file(self) -> Path:
        return self.config.common_dir / self.config.killswitch_filename

    def pulse(
        self,
        status: str = "ok",
        trading_halted: bool = False,
        now: datetime | None = None,
    ) -> HeartbeatPayload:
        return self.writer.write(status=status, trading_halted=trading_halted, now=now)

    def check_guardian_signal(self) -> GuardianSignal | None:
        if not self.killswitch_file.is_file():
            return None

        try:
            content = self.killswitch_file.read_text(encoding="utf-8-sig")
            signal = GuardianSignal.from_json(content)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, OSError) as e:
            logger.error(
                "Failed to parse guardian kill switch file: %s",
                e,
                extra={"event": "GUARDIAN_SIGNAL_PARSE_ERROR"},
            )
            return None

        if self._last_processed_signal_ts != signal.timestamp:
            self._last_processed_signal_ts = signal.timestamp
            logger.warning(
                "Guardian kill switch signal detected: %s (loss: %s%%)",
                signal.reason,
                signal.current_loss_pct,
                extra={
                    "event": "GUARDIAN_SIGNAL_DETECTED",
                    "reason": str(signal.reason),
                    "current_loss_pct": signal.current_loss_pct,
                },
            )

        if self.risk_guard is not None:
            if not self.risk_guard.has_state:
                from xau_bot.risk.models import AccountState

                now = signal.timestamp
                self.risk_guard.initialize_day(
                    now,
                    AccountState(
                        balance=0.0,
                        equity=0.0,
                        margin=0.0,
                        free_margin=0.0,
                        leverage=100.0,
                    ),
                )
            if not self.risk_guard.is_halted:
                halt_reason = f"GUARDIAN_{signal.reason}"
                self.risk_guard.trigger_manual_halt(reason=halt_reason, halt_time=signal.timestamp)

        return signal

    def reset_processed_signal(self) -> None:
        self._last_processed_signal_ts = None

    def clear_guardian_signal(self) -> bool:

        if not self.killswitch_file.exists():
            return False

        try:
            self.killswitch_file.unlink(missing_ok=True)
            logger.info("Cleared guardian kill switch signal file")
            return True
        except OSError as e:
            logger.warning("Failed to remove guardian signal file: %s", e)
            return False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="GuardianBridgeWorker",
                daemon=True,
            )
            self._thread.start()
            logger.info("Guardian bridge background worker started")

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._thread is None:
                return
            self._stop_event.set()
            thread = self._thread
            self._thread = None

        thread.join(timeout=timeout)
        logger.info("Guardian bridge background worker stopped")

    def _run_loop(self) -> None:
        interval = self.config.heartbeat_interval_s
        while not self._stop_event.is_set():
            try:
                trading_halted = False
                if self.risk_guard is not None and self.risk_guard.has_state:
                    trading_halted = self.risk_guard.is_halted

                self.pulse(status="ok", trading_halted=trading_halted)
                self.check_guardian_signal()
            except Exception as e:
                logger.error(
                    "Error in guardian bridge loop: %s",
                    e,
                    extra={"event": "GUARDIAN_BRIDGE_ERROR"},
                )

            self._stop_event.wait(timeout=interval)
