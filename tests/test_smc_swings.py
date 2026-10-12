from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.domain.models import Candle, Swing, SwingType, Timeframe
from xau_bot.smc.swings import (
    ATRMode,
    calculate_atr,
    calculate_true_range,
    current_atr,
    find_swings,
    get_confirmed_swings,
)


def make_candle(
    index: int,
    high: float,
    low: float,
    open_: float | None = None,
    close: float | None = None,
    timeframe: Timeframe = Timeframe.M5,
    base_time: datetime | None = None,
) -> Candle:
    op = open_ if open_ is not None else (high + low) / 2.0
    cl = close if close is not None else (high + low) / 2.0
    start = base_time or datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t = start + timedelta(seconds=index * timeframe.duration_seconds)
    return Candle(
        time=t,
        open=op,
        high=high,
        low=low,
        close=cl,
        volume=100.0,
        timeframe=timeframe,
    )


def test_true_range_calculation() -> None:
    c0 = make_candle(0, high=2655.0, low=2648.0, open_=2650.0, close=2652.0)
    c1 = make_candle(1, high=2660.0, low=2651.0, open_=2653.0, close=2658.0)
    c2 = make_candle(2, high=2656.0, low=2640.0, open_=2655.0, close=2642.0)

    tr_list = calculate_true_range([c0, c1, c2])
    assert len(tr_list) == 3
    assert tr_list[0] == pytest.approx(7.0)
    assert tr_list[1] == pytest.approx(
        max(2660.0 - 2651.0, abs(2660.0 - 2652.0), abs(2651.0 - 2652.0))
    )
    assert tr_list[1] == pytest.approx(9.0)
    assert tr_list[2] == pytest.approx(
        max(2656.0 - 2640.0, abs(2656.0 - 2658.0), abs(2640.0 - 2658.0))
    )
    assert tr_list[2] == pytest.approx(18.0)


def test_true_range_empty() -> None:
    assert calculate_true_range([]) == []


def test_atr_wilder_and_sma_exact_values() -> None:
    candles = [
        make_candle(0, high=2602.0, low=2600.0, open_=2601.0, close=2601.0),
        make_candle(1, high=2605.0, low=2601.0, open_=2602.0, close=2603.0),
        make_candle(2, high=2609.0, low=2603.0, open_=2604.0, close=2606.0),
        make_candle(3, high=2614.0, low=2606.0, open_=2607.0, close=2610.0),
        make_candle(4, high=2620.0, low=2610.0, open_=2611.0, close=2615.0),
    ]
    tr_list = calculate_true_range(candles)
    assert tr_list == [2.0, 4.0, 6.0, 8.0, 10.0]

    period = 3
    atr_wilder = calculate_atr(candles, period=period, mode=ATRMode.WILDER)
    assert len(atr_wilder) == 5
    assert atr_wilder[0] is None
    assert atr_wilder[1] is None
    assert atr_wilder[2] == pytest.approx(4.0)
    assert atr_wilder[3] == pytest.approx(16.0 / 3.0)
    assert atr_wilder[4] == pytest.approx(62.0 / 9.0)

    atr_sma = calculate_atr(candles, period=period, mode=ATRMode.SMA)
    assert len(atr_sma) == 5
    assert atr_sma[0] is None
    assert atr_sma[1] is None
    assert atr_sma[2] == pytest.approx(4.0)
    assert atr_sma[3] == pytest.approx(6.0)
    assert atr_sma[4] == pytest.approx(8.0)

    assert atr_wilder[3] != pytest.approx(atr_sma[3])
    assert atr_wilder[4] != pytest.approx(atr_sma[4])

    assert current_atr(candles, period=period, mode=ATRMode.WILDER) == pytest.approx(62.0 / 9.0)
    assert current_atr(candles, period=period, mode=ATRMode.SMA) == pytest.approx(8.0)


def test_current_atr() -> None:
    candles = [
        make_candle(i, high=2655.0, low=2645.0, open_=2650.0, close=2650.0) for i in range(14)
    ]
    val = current_atr(candles, period=14)
    assert val == pytest.approx(10.0)


def test_current_atr_insufficient_bars_raises() -> None:
    candles = [
        make_candle(i, high=2655.0, low=2645.0, open_=2650.0, close=2650.0) for i in range(5)
    ]
    with pytest.raises(ValueError, match="Insufficient candles"):
        current_atr(candles, period=14)


def test_atr_invalid_period_raises() -> None:
    candles = [make_candle(0, high=2655.0, low=2645.0, open_=2650.0, close=2650.0)]
    with pytest.raises(ValueError, match="Period must be at least 1"):
        calculate_atr(candles, period=0)


def test_find_swings_high_and_low() -> None:
    highs = [2650.0, 2652.0, 2660.0, 2654.0, 2651.0, 2645.0, 2648.0, 2652.0]
    lows = [2645.0, 2648.0, 2655.0, 2649.0, 2642.0, 2638.0, 2642.0, 2646.0]
    candles = [make_candle(i, high=highs[i], low=lows[i]) for i in range(len(highs))]

    swings = find_swings(candles, left_bars=2, right_bars=2)
    assert len(swings) == 2

    sh = swings[0]
    assert sh.type == SwingType.HIGH
    assert sh.index == 2
    assert sh.price == 2660.0
    assert sh.time == candles[2].time
    assert sh.confirmed_time == candles[4].time + timedelta(
        seconds=candles[4].timeframe.duration_seconds
    )

    sl = swings[1]
    assert sl.type == SwingType.LOW
    assert sl.index == 5
    assert sl.price == 2638.0
    assert sl.time == candles[5].time
    assert sl.confirmed_time == candles[7].time + timedelta(
        seconds=candles[7].timeframe.duration_seconds
    )


FindSwingsCallable = Callable[[Sequence[Candle], int, int], list[Swing]]


def _assert_causal_equivalence(
    find_fn: FindSwingsCallable,
    candles: list[Candle],
    left_bars: int = 2,
    right_bars: int = 2,
) -> None:
    all_swings = find_fn(candles, left_bars, right_bars)
    duration = timedelta(seconds=Timeframe.M5.duration_seconds)
    for k in range(len(candles)):
        prefix_candles = candles[: k + 1]
        incremental_swings = find_fn(prefix_candles, left_bars, right_bars)
        as_of_time = candles[k].time + duration
        confirmed_from_full = get_confirmed_swings(all_swings, as_of_time=as_of_time)
        assert incremental_swings == confirmed_from_full


def test_find_swings_anti_lookahead_simulation() -> None:
    highs = [
        2650.0,
        2652.0,
        2660.0,
        2655.0,
        2653.0,
        2675.0,
        2650.0,
        2645.0,
        2648.0,
        2652.0,
        2640.0,
        2650.0,
    ]
    lows = [
        2640.0,
        2642.0,
        2650.0,
        2645.0,
        2643.0,
        2648.0,
        2635.0,
        2628.0,
        2636.0,
        2640.0,
        2615.0,
        2630.0,
    ]
    candles = [make_candle(i, high=highs[i], low=lows[i]) for i in range(len(highs))]

    _assert_causal_equivalence(find_swings, candles, left_bars=2, right_bars=2)

    for k in range(len(candles)):
        prefix_candles = candles[: k + 1]
        incremental_swings = find_swings(prefix_candles, left_bars=2, right_bars=2)

        if k < 4:
            assert len(incremental_swings) == 0
        elif 4 <= k < 7:
            assert len(incremental_swings) == 1
            assert incremental_swings[0].index == 2
            assert incremental_swings[0].price == 2660.0
            assert incremental_swings[0].type == SwingType.HIGH
        elif 7 <= k < 9:
            assert len(incremental_swings) == 2
            assert incremental_swings[0].index == 2
            assert incremental_swings[1].index == 5
            assert incremental_swings[1].price == 2675.0
            assert incremental_swings[1].type == SwingType.HIGH
        elif 9 <= k < 11:
            assert len(incremental_swings) == 3
            assert incremental_swings[0].index == 2
            assert incremental_swings[1].index == 5
            assert incremental_swings[2].index == 7
            assert incremental_swings[2].price == 2628.0
            assert incremental_swings[2].type == SwingType.LOW
        else:
            assert len(incremental_swings) == 4
            assert incremental_swings[0].index == 2
            assert incremental_swings[1].index == 5
            assert incremental_swings[2].index == 7
            assert incremental_swings[3].index == 9
            assert incremental_swings[3].price == 2652.0
            assert incremental_swings[3].type == SwingType.HIGH


def test_anti_lookahead_detects_future_leakage_mutant() -> None:
    def lookahead_mutant_find_swings(
        candles_seq: Sequence[Candle], left_bars: int = 2, right_bars: int = 2
    ) -> list[Swing]:
        swings: list[Swing] = []
        n = len(candles_seq)
        for i in range(left_bars, n - right_bars):
            curr_high = candles_seq[i].high
            is_high = all(curr_high > candles_seq[j].high for j in range(i - left_bars, i))
            if is_high:
                is_high = all(
                    curr_high > candles_seq[j].high for j in range(i + 1, i + 1 + right_bars)
                )
            if (
                is_high
                and (i + right_bars + 1 < n)
                and (curr_high <= candles_seq[i + right_bars + 1].high)
            ):
                is_high = False
            if is_high:
                conf_bar = candles_seq[i + right_bars]
                c_time = conf_bar.time + timedelta(seconds=conf_bar.timeframe.duration_seconds)
                swings.append(
                    Swing(
                        time=candles_seq[i].time,
                        price=curr_high,
                        type=SwingType.HIGH,
                        confirmed_time=c_time,
                        index=i,
                    )
                )
        return swings

    highs = [2650.0, 2652.0, 2660.0, 2655.0, 2653.0, 2675.0]
    lows = [2640.0, 2642.0, 2650.0, 2645.0, 2643.0, 2648.0]
    candles = [make_candle(i, high=highs[i], low=lows[i]) for i in range(len(highs))]

    with pytest.raises(AssertionError):
        _assert_causal_equivalence(lookahead_mutant_find_swings, candles, left_bars=2, right_bars=2)


def test_find_swings_different_lookback() -> None:
    highs = [2640.0, 2650.0, 2645.0]
    lows = [2635.0, 2642.0, 2638.0]
    candles = [make_candle(i, high=highs[i], low=lows[i]) for i in range(len(highs))]

    swings_n1 = find_swings(candles, left_bars=1, right_bars=1)
    assert len(swings_n1) == 1
    assert swings_n1[0].index == 1
    assert swings_n1[0].price == 2650.0

    swings_n2 = find_swings(candles, left_bars=2, right_bars=2)
    assert len(swings_n2) == 0


def test_find_swings_rejects_flat_tops() -> None:
    highs = [2640.0, 2650.0, 2650.0, 2645.0]
    lows = [2635.0, 2642.0, 2642.0, 2638.0]
    candles = [make_candle(i, high=highs[i], low=lows[i]) for i in range(len(highs))]

    swings = find_swings(candles, left_bars=1, right_bars=1)
    assert len(swings) == 0


def test_calculate_atr_fewer_candles_than_period() -> None:
    candles = [make_candle(i, high=2655.0, low=2645.0) for i in range(3)]
    res = calculate_atr(candles, period=5)
    assert res == [None, None, None]


def test_get_confirmed_swings_rejects_naive_datetime() -> None:
    naive_dt = datetime(2026, 9, 24, 12, 0)
    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        get_confirmed_swings([], as_of_time=naive_dt)


def test_find_swings_fewer_candles_than_window() -> None:
    candles = [make_candle(i, high=2655.0, low=2645.0) for i in range(4)]
    assert find_swings(candles, left_bars=2, right_bars=2) == []


def test_find_swings_invalid_parameters() -> None:
    candles = [make_candle(0, high=2655.0, low=2645.0, open_=2650.0, close=2650.0)]
    with pytest.raises(ValueError, match="left_bars and right_bars must be at least 1"):
        find_swings(candles, left_bars=0, right_bars=2)

    with pytest.raises(ValueError, match="left_bars and right_bars must be at least 1"):
        find_swings(candles, left_bars=2, right_bars=0)
