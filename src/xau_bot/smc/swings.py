from collections.abc import Sequence
from datetime import datetime, timedelta
from enum import StrEnum
from typing import cast

from xau_bot.domain.models import Candle, Swing, SwingType


class ATRMode(StrEnum):
    WILDER = "WILDER"
    SMA = "SMA"


def calculate_true_range(candles: Sequence[Candle]) -> list[float]:
    if not candles:
        return []

    tr_list: list[float] = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        curr = candles[i]
        prev_close = candles[i - 1].close
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev_close),
            abs(curr.low - prev_close),
        )
        tr_list.append(tr)
    return tr_list


def calculate_atr(
    candles: Sequence[Candle],
    period: int = 14,
    mode: ATRMode = ATRMode.WILDER,
) -> list[float | None]:
    if period < 1:
        raise ValueError(f"Period must be at least 1, got {period}")

    tr_list = calculate_true_range(candles)
    n = len(tr_list)
    atr_values: list[float | None] = [None] * n

    if n < period:
        return atr_values

    initial_sum = sum(tr_list[:period])
    initial_atr = initial_sum / period
    atr_values[period - 1] = initial_atr

    if mode == ATRMode.WILDER:
        curr_atr = initial_atr
        for i in range(period, n):
            curr_atr = (curr_atr * (period - 1) + tr_list[i]) / period
            atr_values[i] = curr_atr
    else:
        for i in range(period, n):
            window_sum = sum(tr_list[i - period + 1 : i + 1])
            atr_values[i] = window_sum / period

    return atr_values


def current_atr(
    candles: Sequence[Candle],
    period: int = 14,
    mode: ATRMode = ATRMode.WILDER,
) -> float:
    if len(candles) < period:
        raise ValueError(
            f"Insufficient candles for ATR calculation: required {period}, got {len(candles)}"
        )
    atr_series = calculate_atr(candles, period=period, mode=mode)
    return cast(float, atr_series[-1])


def find_swings(
    candles: Sequence[Candle],
    left_bars: int = 2,
    right_bars: int = 2,
) -> list[Swing]:
    if left_bars < 1 or right_bars < 1:
        raise ValueError(
            f"left_bars and right_bars must be at least 1, "
            f"got left_bars={left_bars}, right_bars={right_bars}"
        )

    swings: list[Swing] = []
    n = len(candles)
    if n < left_bars + right_bars + 1:
        return swings

    for i in range(left_bars, n - right_bars):
        curr = candles[i]
        curr_high = curr.high
        curr_low = curr.low

        is_high = True
        for j in range(i - left_bars, i):
            if curr_high <= candles[j].high:
                is_high = False
                break
        if is_high:
            for j in range(i + 1, i + 1 + right_bars):
                if curr_high <= candles[j].high:
                    is_high = False
                    break

        if is_high:
            confirmation_bar = candles[i + right_bars]
            confirmed_time = confirmation_bar.time + timedelta(
                seconds=confirmation_bar.timeframe.duration_seconds
            )
            swings.append(
                Swing(
                    time=curr.time,
                    price=curr_high,
                    type=SwingType.HIGH,
                    confirmed_time=confirmed_time,
                    index=i,
                )
            )

        is_low = True
        for j in range(i - left_bars, i):
            if curr_low >= candles[j].low:
                is_low = False
                break
        if is_low:
            for j in range(i + 1, i + 1 + right_bars):
                if curr_low >= candles[j].low:
                    is_low = False
                    break

        if is_low:
            confirmation_bar = candles[i + right_bars]
            confirmed_time = confirmation_bar.time + timedelta(
                seconds=confirmation_bar.timeframe.duration_seconds
            )
            swings.append(
                Swing(
                    time=curr.time,
                    price=curr_low,
                    type=SwingType.LOW,
                    confirmed_time=confirmed_time,
                    index=i,
                )
            )

    swings.sort(key=lambda s: (s.confirmed_time, s.index))
    return swings


def get_confirmed_swings(
    swings: Sequence[Swing],
    as_of_time: datetime,
) -> list[Swing]:
    if as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")
    return [s for s in swings if s.confirmed_time <= as_of_time]
