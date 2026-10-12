from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from xau_bot.domain.models import (
    BOS,
    Candle,
    ChoCH,
    DealingRange,
    Direction,
    LiquidityPool,
    LiquiditySweep,
    LiquidityType,
    Swing,
    SwingType,
    Timeframe,
)


def _make_candle(
    time: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    timeframe: Timeframe = Timeframe.M5,
) -> Candle:
    return Candle(
        time=time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        timeframe=timeframe,
    )


def _make_swing(
    time: datetime,
    price: float,
    swing_type: SwingType,
    confirmed_time: datetime,
    index: int = 0,
) -> Swing:
    return Swing(
        time=time,
        price=price,
        type=swing_type,
        confirmed_time=confirmed_time,
        index=index,
    )


def test_bos_model_valid_bullish() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf, index=2)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    breaking_candle = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)
    confirmed_time = t_break + timedelta(seconds=Timeframe.M5.duration_seconds)

    bos = BOS(
        time=t_break,
        confirmed_time=confirmed_time,
        direction=Direction.BULLISH,
        broken_swing=swing_high,
        breaking_candle=breaking_candle,
        price=2660.0,
    )

    assert bos.direction == Direction.BULLISH
    assert bos.price == 2660.0
    assert bos.broken_swing == swing_high
    assert bos.breaking_candle == breaking_candle
    assert bos.time == t_break
    assert bos.confirmed_time == confirmed_time


def test_bos_model_valid_bearish() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_low = _make_swing(t_swing, 2640.0, SwingType.LOW, t_conf, index=2)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    breaking_candle = _make_candle(t_break, open_=2642.0, high=2643.0, low=2635.0, close=2637.0)
    confirmed_time = t_break + timedelta(seconds=Timeframe.M5.duration_seconds)

    bos = BOS(
        time=t_break,
        confirmed_time=confirmed_time,
        direction=Direction.BEARISH,
        broken_swing=swing_low,
        breaking_candle=breaking_candle,
        price=2640.0,
    )

    assert bos.direction == Direction.BEARISH
    assert bos.price == 2640.0


def test_bos_rejects_naive_datetime() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    breaking_candle = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)

    naive_dt = datetime(2026, 9, 24, 12, 30)
    valid_dt = datetime(2026, 9, 24, 12, 35, tzinfo=UTC)

    with pytest.raises(ValueError, match="BOS time must be timezone-aware"):
        BOS(
            time=naive_dt,
            confirmed_time=valid_dt,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="BOS confirmed_time must be timezone-aware"):
        BOS(
            time=t_break,
            confirmed_time=naive_dt,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )


def test_bos_rejects_confirmed_time_not_after_time() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    breaking_candle = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)

    with pytest.raises(ValueError, match="confirmed_time must be strictly after"):
        BOS(
            time=t_break,
            confirmed_time=t_break,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )


def test_bos_rejects_invalid_price() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    breaking_candle = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)
    conf = t_break + timedelta(seconds=300)

    with pytest.raises(ValueError, match="price must be positive"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=0.0,
        )

    with pytest.raises(ValueError, match="price must be finite"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=float("nan"),
        )


def test_bos_rejects_direction_and_candle_mismatch() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)
    swing_low = _make_swing(t_swing, 2640.0, SwingType.LOW, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    conf = t_break + timedelta(seconds=300)

    candle_bull = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)
    with pytest.raises(ValueError, match="Bullish BOS requires broken_swing of type HIGH"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            broken_swing=swing_low,
            breaking_candle=candle_bull,
            price=2640.0,
        )

    candle_wick_only = _make_candle(t_break, open_=2655.0, high=2665.0, low=2654.0, close=2659.0)
    with pytest.raises(ValueError, match="Bullish BOS requires candle close > broken_swing price"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=candle_wick_only,
            price=2660.0,
        )

    candle_bear = _make_candle(t_break, open_=2642.0, high=2643.0, low=2635.0, close=2637.0)
    with pytest.raises(ValueError, match="Bearish BOS requires broken_swing of type LOW"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            broken_swing=swing_high,
            breaking_candle=candle_bear,
            price=2660.0,
        )

    candle_wick_bear = _make_candle(t_break, open_=2645.0, high=2646.0, low=2635.0, close=2641.0)
    with pytest.raises(ValueError, match="Bearish BOS requires candle close < broken_swing price"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            broken_swing=swing_low,
            breaking_candle=candle_wick_bear,
            price=2640.0,
        )

    with pytest.raises(ValueError, match="Invalid direction"):
        BOS(
            time=t_break,
            confirmed_time=conf,
            direction=cast(Any, "INVALID"),
            broken_swing=swing_high,
            breaking_candle=candle_bull,
            price=2660.0,
        )


def test_choch_model_valid_and_invariants() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    swing_high = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)
    swing_low = _make_swing(t_swing, 2640.0, SwingType.LOW, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    conf = t_break + timedelta(seconds=300)
    breaking_candle = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)

    choch = ChoCH(
        time=t_break,
        confirmed_time=conf,
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=swing_high,
        breaking_candle=breaking_candle,
        price=2660.0,
    )
    assert choch.direction == Direction.BULLISH
    assert choch.previous_trend == Direction.BEARISH
    assert choch.price == 2660.0

    with pytest.raises(ValueError, match="ChoCH requires reversal"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="Invalid direction"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=cast(Any, "INVALID"),
            previous_trend=Direction.BEARISH,
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="Invalid previous_trend"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=cast(Any, "INVALID"),
            broken_swing=swing_high,
            breaking_candle=breaking_candle,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="Bullish ChoCH requires broken_swing of type HIGH"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=swing_low,
            breaking_candle=breaking_candle,
            price=2640.0,
        )

    candle_wick_high = _make_candle(t_break, open_=2655.0, high=2665.0, low=2654.0, close=2659.0)
    with pytest.raises(
        ValueError, match="Bullish ChoCH requires candle close > broken_swing price"
    ):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=swing_high,
            breaking_candle=candle_wick_high,
            price=2660.0,
        )

    candle_bear = _make_candle(t_break, open_=2645.0, high=2646.0, low=2635.0, close=2637.0)
    with pytest.raises(ValueError, match="Bearish ChoCH requires broken_swing of type LOW"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            previous_trend=Direction.BULLISH,
            broken_swing=swing_high,
            breaking_candle=candle_bear,
            price=2660.0,
        )

    candle_wick_low = _make_candle(t_break, open_=2645.0, high=2646.0, low=2635.0, close=2641.0)
    with pytest.raises(
        ValueError, match="Bearish ChoCH requires candle close < broken_swing price"
    ):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            previous_trend=Direction.BULLISH,
            broken_swing=swing_low,
            breaking_candle=candle_wick_low,
            price=2640.0,
        )


def test_liquidity_pool_model() -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    s1 = _make_swing(t1, 2660.0, SwingType.HIGH, t1 + timedelta(seconds=600), index=2)
    s2 = _make_swing(t2, 2660.5, SwingType.HIGH, t2 + timedelta(seconds=600), index=8)

    pool = LiquidityPool(
        type=LiquidityType.BSL,
        price=2660.25,
        swings=(s1, s2),
        time=t2,
        confirmed_time=s2.confirmed_time,
    )
    assert pool.type == LiquidityType.BSL
    assert pool.price == 2660.25
    assert len(pool.swings) == 2
    assert pool.swept is False

    with pytest.raises(ValueError, match="LiquidityPool swings cannot be empty"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(),
            time=t1,
            confirmed_time=t2,
        )

    with pytest.raises(ValueError, match="BSL pool requires swings of type HIGH"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(_make_swing(t1, 2640.0, SwingType.LOW, t2),),
            time=t1,
            confirmed_time=t2,
        )

    with pytest.raises(ValueError, match="confirmed_time must be >= time"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s1,),
            time=t2,
            confirmed_time=t1,
        )

    c_candle = _make_candle(t2, open_=2655.0, high=2665.0, low=2654.0, close=2658.0)
    with pytest.raises(ValueError, match="swept=True requires swept_at to be set"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s1,),
            time=t1,
            confirmed_time=t2,
            swept=True,
            swept_at=None,
            swept_by_candle=c_candle,
        )

    with pytest.raises(ValueError, match="swept=True requires swept_by_candle to be set"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s1,),
            time=t1,
            confirmed_time=t2,
            swept=True,
            swept_at=t2,
            swept_by_candle=None,
        )

    with pytest.raises(ValueError, match="swept=False requires swept_at to be None"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s1,),
            time=t1,
            confirmed_time=t2,
            swept=False,
            swept_at=t2,
        )

    with pytest.raises(ValueError, match="swept=False requires swept_by_candle to be None"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s1,),
            time=t1,
            confirmed_time=t2,
            swept=False,
            swept_by_candle=c_candle,
        )


def test_liquidity_sweep_model() -> None:
    t = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s = _make_swing(t, 2660.0, SwingType.HIGH, t + timedelta(seconds=600))
    pool = LiquidityPool(
        type=LiquidityType.BSL,
        price=2660.0,
        swings=(s,),
        time=t,
        confirmed_time=s.confirmed_time,
    )

    t_sweep = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    conf = t_sweep + timedelta(seconds=300)
    candle = _make_candle(t_sweep, open_=2655.0, high=2664.0, low=2654.0, close=2658.0)

    sweep = LiquiditySweep(
        time=t_sweep,
        confirmed_time=conf,
        type=LiquidityType.BSL,
        level=2660.0,
        candle=candle,
        pool=pool,
    )
    assert sweep.level == 2660.0
    assert sweep.type == LiquidityType.BSL

    candle_body_break = _make_candle(t_sweep, open_=2655.0, high=2664.0, low=2654.0, close=2662.0)
    with pytest.raises(ValueError, match="BSL sweep requires candle close <= level"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=2660.0,
            candle=candle_body_break,
            pool=pool,
        )


def test_dealing_range_model() -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    high_swing = _make_swing(t1, 2670.0, SwingType.HIGH, t1 + timedelta(seconds=600))
    low_swing = _make_swing(t2, 2630.0, SwingType.LOW, t2 + timedelta(seconds=600))

    dr = DealingRange(
        high=2670.0,
        low=2630.0,
        high_swing=high_swing,
        low_swing=low_swing,
    )
    assert dr.range == pytest.approx(40.0)
    assert dr.equilibrium == pytest.approx(2650.0)
    assert dr.is_premium(2655.0) is True
    assert dr.is_premium(2645.0) is False
    assert dr.is_discount(2645.0) is True
    assert dr.is_discount(2655.0) is False

    ote_buy = dr.ote_zone(Direction.BULLISH)
    assert ote_buy[0] == pytest.approx(2638.4)
    assert ote_buy[1] == pytest.approx(2645.2)

    ote_sell = dr.ote_zone(Direction.BEARISH)
    assert ote_sell[0] == pytest.approx(2654.8)
    assert ote_sell[1] == pytest.approx(2661.6)

    with pytest.raises(ValueError, match="high must be strictly greater than low"):
        DealingRange(
            high=2630.0,
            low=2670.0,
            high_swing=high_swing,
            low_swing=low_swing,
        )

    with pytest.raises(ValueError, match="high_swing must be of type HIGH"):
        bad_swing = cast(Any, low_swing)
        DealingRange(
            high=2670.0,
            low=2630.0,
            high_swing=bad_swing,
            low_swing=low_swing,
        )

    with pytest.raises(ValueError, match="low_swing must be of type LOW"):
        bad_low = cast(Any, high_swing)
        DealingRange(
            high=2670.0,
            low=2630.0,
            high_swing=high_swing,
            low_swing=bad_low,
        )

    with pytest.raises(ValueError, match="DealingRange low must be positive"):
        DealingRange(
            high=2670.0,
            low=-10.0,
            high_swing=high_swing,
            low_swing=low_swing,
        )

    with pytest.raises(ValueError, match="DealingRange high must be finite"):
        DealingRange(
            high=float("nan"),
            low=2630.0,
            high_swing=high_swing,
            low_swing=low_swing,
        )


def test_choch_invariants_extended() -> None:
    t_swing = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t_conf = datetime(2026, 9, 24, 12, 10, tzinfo=UTC)
    sh = _make_swing(t_swing, 2660.0, SwingType.HIGH, t_conf)
    sl = _make_swing(t_swing, 2640.0, SwingType.LOW, t_conf)

    t_break = datetime(2026, 9, 24, 12, 30, tzinfo=UTC)
    conf = t_break + timedelta(seconds=300)
    candle_bull = _make_candle(t_break, open_=2658.0, high=2665.0, low=2657.0, close=2663.0)
    candle_bear = _make_candle(t_break, open_=2642.0, high=2643.0, low=2635.0, close=2637.0)

    with pytest.raises(ValueError, match="ChoCH time must be timezone-aware"):
        ChoCH(
            time=datetime(2026, 9, 24, 12, 30),
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="ChoCH confirmed_time must be timezone-aware"):
        ChoCH(
            time=t_break,
            confirmed_time=datetime(2026, 9, 24, 12, 35),
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="confirmed_time must be strictly after"):
        ChoCH(
            time=t_break,
            confirmed_time=t_break,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="price must be positive"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=0.0,
        )

    with pytest.raises(ValueError, match="price must be finite"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=float("inf"),
        )

    with pytest.raises(ValueError, match="Invalid direction"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=cast(Any, "UP"),
            previous_trend=Direction.BEARISH,
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="Invalid previous_trend"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BULLISH,
            previous_trend=cast(Any, "DOWN"),
            broken_swing=sh,
            breaking_candle=candle_bull,
            price=2660.0,
        )

    with pytest.raises(ValueError, match="Bearish ChoCH requires broken_swing of type LOW"):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            previous_trend=Direction.BULLISH,
            broken_swing=sh,
            breaking_candle=candle_bear,
            price=2660.0,
        )

    candle_not_below = _make_candle(t_break, open_=2642.0, high=2643.0, low=2638.0, close=2641.0)
    with pytest.raises(
        ValueError, match="Bearish ChoCH requires candle close < broken_swing price"
    ):
        ChoCH(
            time=t_break,
            confirmed_time=conf,
            direction=Direction.BEARISH,
            previous_trend=Direction.BULLISH,
            broken_swing=sl,
            breaking_candle=candle_not_below,
            price=2640.0,
        )


def test_liquidity_models_extended() -> None:
    t = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_low = _make_swing(t, 2630.0, SwingType.LOW, t + timedelta(seconds=600))
    c_candle = _make_candle(
        t + timedelta(seconds=600), open_=2635.0, high=2636.0, low=2625.0, close=2631.0
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2630.0,
        swings=(s_low,),
        time=t,
        confirmed_time=s_low.confirmed_time,
    )

    with pytest.raises(ValueError, match="swept_at must be timezone-aware"):
        LiquidityPool(
            type=LiquidityType.SSL,
            price=2630.0,
            swings=(s_low,),
            time=t,
            confirmed_time=s_low.confirmed_time,
            swept=True,
            swept_at=datetime(2026, 9, 24, 13, 0),
            swept_by_candle=c_candle,
        )

    with pytest.raises(ValueError, match="price must be finite"):
        LiquidityPool(
            type=LiquidityType.SSL,
            price=float("nan"),
            swings=(s_low,),
            time=t,
            confirmed_time=s_low.confirmed_time,
        )

    with pytest.raises(ValueError, match="price must be positive"):
        LiquidityPool(
            type=LiquidityType.SSL,
            price=-10.0,
            swings=(s_low,),
            time=t,
            confirmed_time=s_low.confirmed_time,
        )

    with pytest.raises(ValueError, match="Invalid liquidity type"):
        LiquidityPool(
            type=cast(Any, "UNKNOWN"),
            price=2630.0,
            swings=(s_low,),
            time=t,
            confirmed_time=s_low.confirmed_time,
        )

    s_high = _make_swing(t, 2660.0, SwingType.HIGH, t + timedelta(seconds=600))
    with pytest.raises(ValueError, match="SSL pool requires swings of type LOW"):
        LiquidityPool(
            type=LiquidityType.SSL,
            price=2630.0,
            swings=(s_high,),
            time=t,
            confirmed_time=s_high.confirmed_time,
        )

    t_sweep = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    conf = t_sweep + timedelta(seconds=300)
    c_ssl_sweep = _make_candle(t_sweep, open_=2635.0, high=2636.0, low=2625.0, close=2631.0)
    sweep_ssl = LiquiditySweep(
        time=t_sweep,
        confirmed_time=conf,
        type=LiquidityType.SSL,
        level=2630.0,
        candle=c_ssl_sweep,
        pool=pool_ssl,
    )
    assert sweep_ssl.type == LiquidityType.SSL

    c_no_sweep = _make_candle(t_sweep, open_=2635.0, high=2636.0, low=2631.0, close=2632.0)
    with pytest.raises(ValueError, match="SSL sweep requires candle low < level"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.SSL,
            level=2630.0,
            candle=c_no_sweep,
            pool=pool_ssl,
        )

    c_body_break = _make_candle(t_sweep, open_=2635.0, high=2636.0, low=2625.0, close=2628.0)
    with pytest.raises(ValueError, match="SSL sweep requires candle close >= level"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.SSL,
            level=2630.0,
            candle=c_body_break,
            pool=pool_ssl,
        )


def test_liquidity_sweep_and_pool_validation_invariants() -> None:
    t = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_high = _make_swing(t, 2660.0, SwingType.HIGH, t + timedelta(seconds=600))
    s_low = _make_swing(t, 2630.0, SwingType.LOW, t + timedelta(seconds=600))
    pool_bsl = LiquidityPool(
        type=LiquidityType.BSL,
        price=2660.0,
        swings=(s_high,),
        time=t,
        confirmed_time=s_high.confirmed_time,
    )

    with pytest.raises(ValueError, match="LiquidityPool time must be timezone-aware"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s_high,),
            time=datetime(2026, 9, 24, 12, 0),
            confirmed_time=s_high.confirmed_time,
        )

    with pytest.raises(ValueError, match="LiquidityPool confirmed_time must be timezone-aware"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s_high,),
            time=t,
            confirmed_time=datetime(2026, 9, 24, 12, 10),
        )

    with pytest.raises(ValueError, match="LiquidityPool swings cannot be empty"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(),
            time=t,
            confirmed_time=s_high.confirmed_time,
        )

    with pytest.raises(ValueError, match="BSL pool requires swings of type HIGH"):
        LiquidityPool(
            type=LiquidityType.BSL,
            price=2660.0,
            swings=(s_low,),
            time=t,
            confirmed_time=s_low.confirmed_time,
        )

    t_sweep = datetime(2026, 9, 24, 13, 0, tzinfo=UTC)
    conf = t_sweep + timedelta(seconds=300)
    c_bsl_sweep = _make_candle(t_sweep, open_=2655.0, high=2665.0, low=2654.0, close=2658.0)

    sweep = LiquiditySweep(
        time=t_sweep,
        confirmed_time=conf,
        type=LiquidityType.BSL,
        level=2660.0,
        candle=c_bsl_sweep,
        pool=pool_bsl,
    )
    assert sweep.level == 2660.0

    with pytest.raises(ValueError, match="LiquiditySweep time must be timezone-aware"):
        LiquiditySweep(
            time=datetime(2026, 9, 24, 13, 0),
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=2660.0,
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    with pytest.raises(ValueError, match="LiquiditySweep confirmed_time must be timezone-aware"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=datetime(2026, 9, 24, 13, 5),
            type=LiquidityType.BSL,
            level=2660.0,
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    with pytest.raises(ValueError, match="confirmed_time must be strictly after sweep time"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=t_sweep,
            type=LiquidityType.BSL,
            level=2660.0,
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    with pytest.raises(ValueError, match="Sweep level must be finite"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=float("nan"),
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    with pytest.raises(ValueError, match="Sweep level must be positive"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=-5.0,
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    with pytest.raises(ValueError, match="Invalid liquidity type"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=cast(Any, "UNKNOWN"),
            level=2660.0,
            candle=c_bsl_sweep,
            pool=pool_bsl,
        )

    c_low_high = _make_candle(t_sweep, open_=2655.0, high=2659.0, low=2654.0, close=2658.0)
    with pytest.raises(ValueError, match="BSL sweep requires candle high > level"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=2660.0,
            candle=c_low_high,
            pool=pool_bsl,
        )

    c_close_above = _make_candle(t_sweep, open_=2655.0, high=2665.0, low=2654.0, close=2662.0)
    with pytest.raises(ValueError, match="BSL sweep requires candle close <= level"):
        LiquiditySweep(
            time=t_sweep,
            confirmed_time=conf,
            type=LiquidityType.BSL,
            level=2660.0,
            candle=c_close_above,
            pool=pool_bsl,
        )
