from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from xau_bot.domain.models import Candle, Timeframe


class DomainSeriesError(ValueError):
    pass


class UnsortedSeriesError(DomainSeriesError):
    pass


class DuplicateCandleError(DomainSeriesError):
    pass


class TimeframeMismatchError(DomainSeriesError):
    pass


@dataclass(frozen=True, slots=True)
class DataGap:
    start_time: datetime
    end_time: datetime
    missing_bars: int


class CandleSeries:
    def __init__(
        self,
        candles: Sequence[Candle],
        timeframe: Timeframe = Timeframe.M5,
        has_forming_candle: bool = False,
    ) -> None:
        self.timeframe = timeframe
        self.has_forming_candle = has_forming_candle

        candle_tuple = tuple(candles)
        self._validate_sequence(candle_tuple)
        self._candles = candle_tuple

    def _validate_sequence(self, candles: tuple[Candle, ...]) -> None:
        for i in range(len(candles)):
            if candles[i].timeframe != self.timeframe:
                raise TimeframeMismatchError(
                    f"Candle timeframe mismatch: expected {self.timeframe}, "
                    f"got {candles[i].timeframe}"
                )
            if i > 0:
                prev_time = candles[i - 1].time
                curr_time = candles[i].time
                if curr_time == prev_time:
                    raise DuplicateCandleError(f"Duplicate candle timestamp found: {curr_time}")
                if curr_time < prev_time:
                    raise UnsortedSeriesError(
                        f"Candles not sorted chronologically: {prev_time} > {curr_time}"
                    )

    @property
    def closed_candles(self) -> tuple[Candle, ...]:
        if self.has_forming_candle and self._candles:
            return self._candles[:-1]
        return self._candles

    @property
    def forming_candle(self) -> Candle | None:
        if self.has_forming_candle and self._candles:
            return self._candles[-1]
        return None

    def detect_gaps(
        self,
        max_gap_multiplier: float = 1.5,
        is_market_open: Callable[[datetime], bool] | None = None,
    ) -> list[DataGap]:
        gaps: list[DataGap] = []
        candles = self.closed_candles
        expected_step = timedelta(seconds=self.timeframe.duration_seconds)
        threshold = expected_step * max_gap_multiplier

        for i in range(len(candles) - 1):
            diff = candles[i + 1].time - candles[i].time
            if diff > threshold:
                if is_market_open is not None:
                    missing_market_bars = 0
                    curr = candles[i].time + expected_step
                    end = candles[i + 1].time
                    while curr < end:
                        if is_market_open(curr):
                            missing_market_bars += 1
                        curr += expected_step
                    if missing_market_bars > 0:
                        gaps.append(
                            DataGap(
                                start_time=candles[i].time,
                                end_time=candles[i + 1].time,
                                missing_bars=missing_market_bars,
                            )
                        )
                else:
                    missing = int(diff.total_seconds() // self.timeframe.duration_seconds) - 1
                    gaps.append(
                        DataGap(
                            start_time=candles[i].time,
                            end_time=candles[i + 1].time,
                            missing_bars=max(1, missing),
                        )
                    )
        return gaps

    def __len__(self) -> int:
        return len(self._candles)

    def __getitem__(self, index: int) -> Candle:
        return self._candles[index]

    def __iter__(self) -> Iterator[Candle]:
        return iter(self._candles)
