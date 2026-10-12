import logging
import threading
from datetime import UTC, datetime

from xau_bot.broker.ports import BrokerPort
from xau_bot.domain.models import Timeframe
from xau_bot.filters.sessions import SessionFilter
from xau_bot.runtime.models import (
    DegradationReason,
    HealthSyncTarget,
    WatchdogConfig,
    WatchdogHealth,
)

logger = logging.getLogger("xau_bot.runtime.watchdog")


class Watchdog:
    def __init__(
        self,
        broker: BrokerPort | None = None,
        config: WatchdogConfig | None = None,
        session_filter: SessionFilter | None = None,
        boot_time: datetime | None = None,
    ) -> None:
        if boot_time is not None and boot_time.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        self._broker: BrokerPort | None = broker
        self._config: WatchdogConfig = config or WatchdogConfig()
        self._session_filter: SessionFilter | None = session_filter
        self._lock: threading.Lock = threading.Lock()
        self._created_at: datetime = (boot_time or datetime.now(UTC)).astimezone(UTC)
        self._last_tick_time: datetime | None = None
        self._last_loop_beat: datetime | None = None
        self._last_loop_beat_by_tf: dict[Timeframe, datetime] = {}
        self._reconnect_attempts: int = 0
        self._last_reconnect_time: datetime | None = None
        self._last_logged_reasons: tuple[DegradationReason, ...] | None = None

    def update_tick(self, timestamp: datetime) -> None:
        if timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        dt_utc = timestamp.astimezone(UTC)
        with self._lock:
            self._last_tick_time = dt_utc

    def beat_loop(self, timeframe: Timeframe | None = None, now: datetime | None = None) -> None:
        if now is not None and now.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        dt = (now or datetime.now(UTC)).astimezone(UTC)
        with self._lock:
            self._last_loop_beat = dt
            if timeframe is not None:
                self._last_loop_beat_by_tf[timeframe] = dt

    def get_last_loop_beat(self, timeframe: Timeframe | None = None) -> datetime | None:
        with self._lock:
            if timeframe is not None:
                return self._last_loop_beat_by_tf.get(timeframe)
            return self._last_loop_beat

    def _is_market_open(self, dt: datetime) -> bool:
        if self._session_filter is not None:
            return self._session_filter.is_market_open(dt)
        return True

    @property
    def config(self) -> WatchdogConfig:
        return self._config

    def _check_broker_connected(self) -> bool:
        if self._broker is None:
            return False
        return self._broker.is_connected()

    def _evaluate_tick_age(self, now_utc: datetime) -> tuple[float, bool]:
        with self._lock:
            last_tick = self._last_tick_time

        if last_tick is None:
            market_open = self._is_market_open(now_utc)
            stale = market_open or not self._config.ignore_stale_when_market_closed
            if stale:
                return float("inf"), True
            return 0.0, False

        tick_age_s = max(0.0, (now_utc - last_tick).total_seconds())
        if tick_age_s > self._config.max_tick_age_s:
            market_open = self._is_market_open(now_utc)
            if market_open or not self._config.ignore_stale_when_market_closed:
                return tick_age_s, True
        return tick_age_s, False

    def _evaluate_loop_age(self, now_utc: datetime) -> tuple[float, bool]:
        with self._lock:
            last_beat = self._last_loop_beat

        if last_beat is None:
            since_boot_s = max(0.0, (now_utc - self._created_at).total_seconds())
            if since_boot_s > self._config.max_loop_silence_s:
                market_open = self._is_market_open(now_utc)
                if market_open or not self._config.ignore_stale_when_market_closed:
                    return since_boot_s, True
            return 0.0, False

        loop_age_s = max(0.0, (now_utc - last_beat).total_seconds())
        if loop_age_s > self._config.max_loop_silence_s:
            market_open = self._is_market_open(now_utc)
            if market_open or not self._config.ignore_stale_when_market_closed:
                return loop_age_s, True
        return loop_age_s, False

    def check_health(
        self,
        now: datetime | None = None,
        trading_halted: bool = False,
    ) -> WatchdogHealth:
        if now is not None and now.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        now_utc = (now or datetime.now(UTC)).astimezone(UTC)
        reasons: list[DegradationReason] = []

        broker_connected = self._check_broker_connected()
        if not broker_connected:
            reasons.append(DegradationReason.BROKER_DISCONNECTED)

        tick_age_s, is_tick_stale = self._evaluate_tick_age(now_utc)
        if is_tick_stale:
            reasons.append(DegradationReason.TICK_STALE)

        loop_age_s, is_loop_stale = self._evaluate_loop_age(now_utc)
        if is_loop_stale:
            reasons.append(DegradationReason.LOOP_STALE)

        if trading_halted:
            reasons.append(DegradationReason.TRADING_HALTED)

        is_healthy = len(reasons) == 0
        status_code = 200 if is_healthy else 503
        status_str = "ok" if is_healthy else "degraded"
        reasons_tuple = tuple(reasons)

        health = WatchdogHealth(
            is_healthy=is_healthy,
            broker_connected=broker_connected,
            last_tick_age_s=tick_age_s,
            trading_halted=trading_halted,
            reasons=reasons_tuple,
            status_code=status_code,
            status=status_str,
            last_loop_beat_age_s=loop_age_s,
            timestamp=now_utc,
        )

        with self._lock:
            if not is_healthy:
                if reasons_tuple != self._last_logged_reasons:
                    self._last_logged_reasons = reasons_tuple
                    infra_reasons = [r.value for r in reasons if r.value != "TRADING_HALTED"]
                    if not infra_reasons:
                        logger.info(
                            "Trading halted by kill switch (infrastructure healthy)",
                            extra={"event": "trading_halted", "reasons": ["TRADING_HALTED"]},
                        )
                    else:
                        logger.warning(
                            "Health check degraded: reasons=%s",
                            [r.value for r in reasons],
                            extra={
                                "event": "watchdog_degraded",
                                "reasons": [r.value for r in reasons],
                            },
                        )
            else:
                if self._last_logged_reasons is not None:
                    self._last_logged_reasons = None
                    logger.info("Health restored to OK", extra={"event": "watchdog_recovered"})

        return health

    async def maybe_reconnect(self, now: datetime | None = None) -> bool:
        if now is not None and now.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")

        if not self._config.auto_reconnect or self._broker is None or self._broker.is_connected():
            return False

        now_utc = (now or datetime.now(UTC)).astimezone(UTC)
        delay = min(
            self._config.reconnect_backoff_max_s,
            self._config.reconnect_backoff_base_s * (2**self._reconnect_attempts),
        )
        if self._last_reconnect_time is not None:
            elapsed = (now_utc - self._last_reconnect_time).total_seconds()
            if elapsed < delay:
                return False

        self._last_reconnect_time = now_utc
        self._reconnect_attempts += 1
        try:
            connected = await self._broker.connect()
            if connected:
                self._reconnect_attempts = 0
            return connected
        except Exception as exc:
            logger.warning(
                "Broker reconnection failed: %s",
                exc,
                extra={"event": "broker_reconnect_failed"},
            )
            return False

    def sync_server_state(
        self,
        target: HealthSyncTarget,
        now: datetime | None = None,
    ) -> WatchdogHealth:
        health = self.check_health(now=now, trading_halted=target.trading_halted)
        target.apply_watchdog_health(health)
        return health
