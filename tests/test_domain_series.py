from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.domain.models import Candle, Timeframe
from xau_bot.domain.series import (
    CandleSeries,
    DataGap,
    DuplicateCandleError,
    TimeframeMismatchError,
    UnsortedSeriesError,
)
from xau_bot.filters.sessions import SessionFilter


def make_candle(minutes: int, timeframe: Timeframe = Timeframe.M5) -> Candle:
    base = datetime(2026, 9, 24, 12, 0, tzinfo=UTC) + timedelta(minutes=minutes)
    return Candle(
        time=base,
        open=2650.0,
        high=2655.0,
        low=2648.0,
        close=2652.0,
        volume=100.0,
        timeframe=timeframe,
    )


def test_series_chronological_ordering() -> None:
    c1 = make_candle(0)
    c2 = make_candle(5)
    c3 = make_candle(10)

    series = CandleSeries(candles=[c1, c2, c3], timeframe=Timeframe.M5)
    assert len(series) == 3
    assert series[0] == c1
    assert series[2] == c3


def test_series_rejects_unsorted() -> None:
    c1 = make_candle(0)
    c2 = make_candle(5)
    with pytest.raises(UnsortedSeriesError):
        CandleSeries(candles=[c2, c1], timeframe=Timeframe.M5)


def test_series_rejects_duplicates() -> None:
    c1 = make_candle(0)
    c2 = make_candle(0)
    with pytest.raises(DuplicateCandleError):
        CandleSeries(candles=[c1, c2], timeframe=Timeframe.M5)


def test_series_different_timeframe_rejected() -> None:
    c1 = make_candle(0, Timeframe.M5)
    c2 = make_candle(5, Timeframe.M15)
    with pytest.raises(TimeframeMismatchError, match="Candle timeframe mismatch"):
        CandleSeries(candles=[c1, c2], timeframe=Timeframe.M5)


def test_forming_candle_isolation() -> None:
    candles = [make_candle(i * 5) for i in range(4)]
    series = CandleSeries(
        candles=candles,
        timeframe=Timeframe.M5,
        has_forming_candle=True,
    )
    assert series.has_forming_candle is True
    assert series.forming_candle == candles[-1]
    closed = series.closed_candles
    assert len(closed) == 3
    assert closed == tuple(candles[:-1])


def test_series_without_forming_candle() -> None:
    candles = [make_candle(i * 5) for i in range(3)]
    series = CandleSeries(
        candles=candles,
        timeframe=Timeframe.M5,
        has_forming_candle=False,
    )
    assert series.forming_candle is None
    assert series.closed_candles == tuple(candles)


def test_series_gap_detection() -> None:
    c1 = make_candle(0)
    c2 = make_candle(5)
    c3 = make_candle(20)

    series = CandleSeries(candles=[c1, c2, c3], timeframe=Timeframe.M5)
    gaps = series.detect_gaps()
    assert len(gaps) == 1
    gap = gaps[0]
    assert isinstance(gap, DataGap)
    assert gap.start_time == c2.time
    assert gap.end_time == c3.time
    assert gap.missing_bars == 2


def test_series_iteration() -> None:
    candles = [make_candle(i * 5) for i in range(3)]
    series = CandleSeries(candles=candles, timeframe=Timeframe.M5)
    iterated = list(series)
    assert iterated == candles


def test_series_gap_detection_with_market_pauses() -> None:
    t_friday = datetime(2026, 9, 25, 20, 55, tzinfo=UTC)
    t_sunday = datetime(2026, 9, 27, 22, 0, tzinfo=UTC)

    c_friday = Candle(
        time=t_friday,
        open=2650.0,
        high=2655.0,
        low=2648.0,
        close=2652.0,
        timeframe=Timeframe.M5,
    )
    c_sunday = Candle(
        time=t_sunday,
        open=2652.0,
        high=2654.0,
        low=2650.0,
        close=2653.0,
        timeframe=Timeframe.M5,
    )

    series = CandleSeries(candles=[c_friday, c_sunday], timeframe=Timeframe.M5)
    sf = SessionFilter()

    raw_gaps = series.detect_gaps()
    assert len(raw_gaps) == 1

    market_gaps = series.detect_gaps(is_market_open=sf.is_market_open)
    assert len(market_gaps) == 0


def test_series_gap_detection_during_market_hours_detected() -> None:
    t1 = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 23, 14, 20, tzinfo=UTC)

    c1 = Candle(time=t1, open=2650.0, high=2655.0, low=2648.0, close=2652.0, timeframe=Timeframe.M5)
    c2 = Candle(time=t2, open=2652.0, high=2654.0, low=2650.0, close=2653.0, timeframe=Timeframe.M5)

    series = CandleSeries(candles=[c1, c2], timeframe=Timeframe.M5)
    sf = SessionFilter()
    market_gaps = series.detect_gaps(is_market_open=sf.is_market_open)
    assert len(market_gaps) == 1
    assert market_gaps[0].missing_bars == 3


def test_resample_to_h1() -> None:
    from xau_bot.domain.series import resample_to_h1

    assert resample_to_h1([]) == []

    t0 = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)
    m5_candles = [
        Candle(
            time=t0 + timedelta(minutes=5 * i),
            open=2650.0 + i,
            high=2655.0 + i,
            low=2645.0 + i,
            close=2652.0 + i,
            volume=10.0,
            timeframe=Timeframe.M5,
        )
        for i in range(12)
    ]
    m5_candles.append(
        Candle(
            time=t0 + timedelta(minutes=60),
            open=2670.0,
            high=2675.0,
            low=2668.0,
            close=2672.0,
            volume=20.0,
            timeframe=Timeframe.M5,
        )
    )

    h1_candles = resample_to_h1(m5_candles)
    assert len(h1_candles) == 2
    assert h1_candles[0].time == t0
    assert h1_candles[0].timeframe == Timeframe.H1
    assert h1_candles[0].open == 2650.0
    assert h1_candles[0].high == 2666.0
    assert h1_candles[0].low == 2645.0
    assert h1_candles[0].close == 2663.0
    assert h1_candles[0].volume == 120.0
    assert h1_candles[1].time == t0 + timedelta(minutes=60)
    assert h1_candles[1].open == 2670.0
    assert h1_candles[1].close == 2672.0
