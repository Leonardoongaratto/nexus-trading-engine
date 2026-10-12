import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from xau_bot.domain.models import Candle, Timeframe
from xau_bot.domain.series import CandleSeries, DataGap
from xau_bot.runtime.models import CandleCloseEvent

logger = logging.getLogger("xau_bot.runtime.candle_loop")

CandleHandler = Callable[[CandleCloseEvent], Coroutine[Any, Any, None] | None]


class CandleLoop:
    def __init__(
        self,
        timeframes: Sequence[Timeframe] = (
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        ),
        max_history_bars: int = 1000,
    ) -> None:
        if max_history_bars < 1:
            raise ValueError("max_history_bars must be >= 1")
        self.timeframes: tuple[Timeframe, ...] = tuple(timeframes)
        self.max_history_bars: int = max_history_bars
        self._handlers: dict[Timeframe, list[CandleHandler]] = {tf: [] for tf in self.timeframes}
        self._closed_history: dict[Timeframe, list[Candle]] = {tf: [] for tf in self.timeframes}
        self._forming_candles: dict[Timeframe, Candle | None] = dict.fromkeys(self.timeframes)
        self._processed_times: dict[Timeframe, set[datetime]] = {
            tf: set() for tf in self.timeframes
        }
        self._processed_queue: dict[Timeframe, deque[datetime]] = {
            tf: deque(maxlen=max_history_bars) for tf in self.timeframes
        }
        self._discarded_watermark: dict[Timeframe, datetime | None] = dict.fromkeys(
            self.timeframes, None
        )
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _ensure_timeframe(self, timeframe: Timeframe) -> None:
        if timeframe not in self._forming_candles:
            self._forming_candles[timeframe] = None
            self._closed_history[timeframe] = []
            self._processed_times[timeframe] = set()
            self._processed_queue[timeframe] = deque(maxlen=self.max_history_bars)
            self._discarded_watermark[timeframe] = None
        if timeframe not in self._handlers:
            self._handlers[timeframe] = []

    def register_handler(self, timeframe: Timeframe, handler: CandleHandler) -> None:
        self._ensure_timeframe(timeframe)
        self._handlers[timeframe].append(handler)

    def get_last_closed_candle(self, timeframe: Timeframe) -> Candle | None:
        history = self._closed_history.get(timeframe)
        return history[-1] if history else None

    def get_closed_series(self, timeframe: Timeframe) -> tuple[Candle, ...]:
        return tuple(self._closed_history.get(timeframe, []))

    def get_forming_candle(self, timeframe: Timeframe) -> Candle | None:
        return self._forming_candles.get(timeframe)

    def _is_already_processed(self, timeframe: Timeframe, dt: datetime) -> bool:
        watermark = self._discarded_watermark.get(timeframe)
        if watermark is not None and dt <= watermark:
            return True
        return dt in self._processed_times.get(timeframe, set())

    def _mark_processed(self, timeframe: Timeframe, dt: datetime) -> None:
        times_set = self._processed_times[timeframe]
        queue = self._processed_queue[timeframe]
        if len(queue) == queue.maxlen and queue:
            oldest = queue.popleft()
            times_set.discard(oldest)
            current_wm = self._discarded_watermark[timeframe]
            self._discarded_watermark[timeframe] = (
                max(current_wm, oldest) if current_wm is not None else oldest
            )
        queue.append(dt)
        times_set.add(dt)

    def _record_closed_candle(
        self,
        timeframe: Timeframe,
        candle: Candle,
        is_last: bool,
        forming: Candle | None,
        closed_at: datetime,
    ) -> CandleCloseEvent:
        self._mark_processed(timeframe, candle.time)
        history = self._closed_history[timeframe]
        history.append(candle)
        history.sort(key=lambda c: c.time)
        if len(history) > self.max_history_bars:
            del history[: len(history) - self.max_history_bars]

        event_forming = forming if is_last else None
        if is_last:
            self._forming_candles[timeframe] = forming

        event_series = tuple(c for c in history if c.time <= candle.time)
        return CandleCloseEvent(
            timeframe=timeframe,
            candle=candle,
            closed_series=event_series,
            forming_candle=event_forming,
            closed_at=closed_at,
        )

    def feed_candles(
        self,
        timeframe: Timeframe,
        candles: Sequence[Candle],
        has_forming_candle: bool = True,
        dispatch: bool = True,
    ) -> list[CandleCloseEvent]:
        if not candles:
            return []

        if dispatch:
            handlers = self._handlers.get(timeframe, [])
            if any(inspect.iscoroutinefunction(h) for h in handlers):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    raise RuntimeError(
                        "Async handler cannot be dispatched synchronously without an active "
                        "asyncio event loop. Use on_candles_async or on_new_candles."
                    ) from None

        series = CandleSeries(candles, timeframe=timeframe, has_forming_candle=has_forming_candle)
        closed_candles = series.closed_candles
        forming_candle = series.forming_candle

        self._ensure_timeframe(timeframe)

        new_events: list[CandleCloseEvent] = []
        total_closed = len(closed_candles)
        if total_closed == 0 or all(
            self._is_already_processed(timeframe, c.time) for c in closed_candles
        ):
            self._forming_candles[timeframe] = forming_candle

        for idx, candle in enumerate(closed_candles):
            if self._is_already_processed(timeframe, candle.time):
                continue

            closed_at = candle.time + timedelta(seconds=timeframe.duration_seconds)
            is_last = idx == total_closed - 1
            event = self._record_closed_candle(
                timeframe=timeframe,
                candle=candle,
                is_last=is_last,
                forming=forming_candle,
                closed_at=closed_at,
            )
            new_events.append(event)
            if dispatch:
                self._dispatch_sync(timeframe, event)

        return new_events

    def push_candle(
        self,
        timeframe: Timeframe,
        candle: Candle,
        has_forming_candle: bool = False,
        dispatch: bool = True,
    ) -> list[CandleCloseEvent]:
        return self.feed_candles(
            timeframe,
            [candle],
            has_forming_candle=has_forming_candle,
            dispatch=dispatch,
        )

    async def on_candles_async(
        self,
        timeframe: Timeframe,
        candles: Sequence[Candle],
        has_forming_candle: bool = True,
    ) -> list[CandleCloseEvent]:
        return await self.on_new_candles(
            {timeframe: candles},
            has_forming_candle=has_forming_candle,
        )

    async def on_new_candles(
        self,
        candles_by_tf: Mapping[Timeframe, Sequence[Candle]],
        has_forming_candle: bool = True,
    ) -> list[CandleCloseEvent]:
        parsed_items: list[tuple[datetime, Timeframe, Candle, bool, Candle | None]] = []

        for tf, candles in candles_by_tf.items():
            if not candles:
                continue
            self._ensure_timeframe(tf)

            series = CandleSeries(candles, timeframe=tf, has_forming_candle=has_forming_candle)
            closed = series.closed_candles
            forming = series.forming_candle
            total_closed = len(closed)
            if total_closed == 0 or all(self._is_already_processed(tf, c.time) for c in closed):
                self._forming_candles[tf] = forming

            for idx, candle in enumerate(closed):
                closed_at = candle.time + timedelta(seconds=tf.duration_seconds)
                is_last = idx == total_closed - 1
                parsed_items.append((closed_at, tf, candle, is_last, forming))

        parsed_items.sort(key=lambda item: (item[0], item[1].duration_seconds, item[1].value))

        all_events: list[CandleCloseEvent] = []
        for closed_at, tf, candle, is_last, forming in parsed_items:
            if self._is_already_processed(tf, candle.time):
                continue

            event = self._record_closed_candle(
                timeframe=tf,
                candle=candle,
                is_last=is_last,
                forming=forming,
                closed_at=closed_at,
            )
            all_events.append(event)

            handlers = self._handlers.get(tf, [])
            for handler in handlers:
                try:
                    res = handler(event)
                    if inspect.isawaitable(res):
                        await res
                except Exception as exc:
                    logger.error(
                        "Unhandled exception in async candle handler: %s",
                        exc,
                        exc_info=True,
                        extra={"event": "candle_handler_failed", "timeframe": tf.value},
                    )

        return all_events

    def _dispatch_sync(self, timeframe: Timeframe, event: CandleCloseEvent) -> None:
        handlers = self._handlers.get(timeframe, [])
        for handler in handlers:
            try:
                res = handler(event)
                if inspect.isawaitable(res):
                    try:
                        running_loop = asyncio.get_running_loop()
                        task = running_loop.create_task(res)
                        self._background_tasks.add(task)
                        task.add_done_callback(self._on_task_done)
                    except RuntimeError:
                        res.close()
                        raise RuntimeError(
                            "Async handler cannot be dispatched synchronously without an active "
                            "asyncio event loop. Use on_candles_async or on_new_candles."
                        ) from None
            except RuntimeError as err:
                if "Async handler cannot be dispatched synchronously" in str(err):
                    raise
                logger.error(
                    "Unhandled exception in sync candle handler: %s",
                    err,
                    exc_info=True,
                    extra={"event": "candle_handler_failed", "timeframe": timeframe.value},
                )
            except Exception as exc:
                logger.error(
                    "Unhandled exception in sync candle handler: %s",
                    exc,
                    exc_info=True,
                    extra={"event": "candle_handler_failed", "timeframe": timeframe.value},
                )

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Unhandled exception in background candle handler: %s",
                    exc,
                    extra={"event": "async_candle_handler_failed"},
                )

    def detect_series_gaps(
        self,
        timeframe: Timeframe,
        max_gap_multiplier: float = 1.5,
        is_market_open: Callable[[datetime], bool] | None = None,
    ) -> list[DataGap]:
        history = self.get_closed_series(timeframe)
        if len(history) < 2:
            return []
        series = CandleSeries(history, timeframe=timeframe, has_forming_candle=False)
        return series.detect_gaps(
            max_gap_multiplier=max_gap_multiplier,
            is_market_open=is_market_open,
        )
