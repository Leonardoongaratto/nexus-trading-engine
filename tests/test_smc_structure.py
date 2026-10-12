from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.domain.models import (
    BOS,
    Candle,
    ChoCH,
    Direction,
    LiquidityType,
    Swing,
    SwingType,
    Timeframe,
)
from xau_bot.smc.structure import (
    calculate_dealing_range,
    detect_liquidity_pools,
    detect_liquidity_sweeps,
    detect_structure,
    get_current_dealing_range,
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


def make_swing(
    index: int,
    price: float,
    swing_type: SwingType,
    candles: Sequence[Candle],
    right_bars: int = 2,
) -> Swing:
    candle = candles[index]
    conf_candle = candles[index + right_bars]
    duration = timedelta(seconds=conf_candle.timeframe.duration_seconds)
    return Swing(
        time=candle.time,
        price=price,
        type=swing_type,
        confirmed_time=conf_candle.time + duration,
        index=index,
    )


def test_bos_bullish_requires_body_close() -> None:
    highs = [2650.0, 2652.0, 2660.0, 2655.0, 2653.0, 2665.0, 2666.0]
    lows = [2640.0, 2642.0, 2650.0, 2645.0, 2643.0, 2652.0, 2657.0]
    opens = [2645.0, 2648.0, 2655.0, 2650.0, 2648.0, 2655.0, 2658.0]
    closes = [2648.0, 2650.0, 2658.0, 2648.0, 2645.0, 2658.0, 2664.0]
    candles = [
        make_candle(i, high=highs[i], low=lows[i], open_=opens[i], close=closes[i])
        for i in range(len(highs))
    ]

    sh = make_swing(index=2, price=2660.0, swing_type=SwingType.HIGH, candles=candles, right_bars=2)
    breaks = detect_structure(candles, [sh])

    assert len(breaks) == 1
    bos = breaks[0]
    assert isinstance(bos, BOS)
    assert bos.direction == Direction.BULLISH
    assert bos.price == 2660.0
    assert bos.breaking_candle == candles[6]
    assert bos.broken_swing == sh


def test_bos_bearish_requires_body_close() -> None:
    highs = [2655.0, 2650.0, 2646.0, 2648.0, 2647.0, 2645.0, 2642.0]
    lows = [2648.0, 2642.0, 2640.0, 2643.0, 2644.0, 2638.0, 2633.0]
    opens = [2652.0, 2648.0, 2644.0, 2645.0, 2646.0, 2644.0, 2640.0]
    closes = [2650.0, 2644.0, 2642.0, 2646.0, 2645.0, 2642.0, 2635.0]
    candles = [
        make_candle(i, high=highs[i], low=lows[i], open_=opens[i], close=closes[i])
        for i in range(len(highs))
    ]

    sl = make_swing(index=2, price=2640.0, swing_type=SwingType.LOW, candles=candles, right_bars=2)
    breaks = detect_structure(candles, [sl])

    assert len(breaks) == 1
    bos = breaks[0]
    assert isinstance(bos, BOS)
    assert bos.direction == Direction.BEARISH
    assert bos.price == 2640.0
    assert bos.breaking_candle == candles[6]


def test_choch_reversal_bearish_to_bullish() -> None:
    highs = [2665.0, 2662.0, 2660.0, 2650.0, 2642.0, 2635.0, 2632.0, 2645.0, 2658.0, 2665.0]
    lows = [2658.0, 2655.0, 2652.0, 2638.0, 2632.0, 2628.0, 2625.0, 2630.0, 2642.0, 2655.0]
    opens = [2660.0, 2658.0, 2654.0, 2648.0, 2638.0, 2632.0, 2628.0, 2632.0, 2645.0, 2656.0]
    closes = [2662.0, 2656.0, 2652.0, 2640.0, 2634.0, 2630.0, 2627.0, 2642.0, 2655.0, 2663.0]
    candles = [
        make_candle(i, high=highs[i], low=lows[i], open_=opens[i], close=closes[i])
        for i in range(len(highs))
    ]

    sh = make_swing(index=2, price=2660.0, swing_type=SwingType.HIGH, candles=candles, right_bars=2)
    sl = make_swing(index=6, price=2625.0, swing_type=SwingType.LOW, candles=candles, right_bars=2)

    breaks = detect_structure(candles, [sh, sl], initial_trend=Direction.BEARISH)
    assert len(breaks) == 1
    event = breaks[0]
    assert isinstance(event, ChoCH)
    assert event.direction == Direction.BULLISH
    assert event.previous_trend == Direction.BEARISH
    assert event.price == 2660.0
    assert event.breaking_candle == candles[9]


def test_choch_reversal_bullish_to_bearish() -> None:
    highs = [2635.0, 2638.0, 2645.0, 2652.0, 2660.0, 2668.0, 2670.0, 2660.0, 2650.0, 2642.0]
    lows = [2628.0, 2630.0, 2640.0, 2646.0, 2652.0, 2660.0, 2662.0, 2648.0, 2638.0, 2632.0]
    opens = [2630.0, 2632.0, 2641.0, 2648.0, 2654.0, 2662.0, 2665.0, 2658.0, 2648.0, 2639.0]
    closes = [2634.0, 2636.0, 2644.0, 2650.0, 2658.0, 2666.0, 2668.0, 2650.0, 2640.0, 2635.0]
    candles = [
        make_candle(i, high=highs[i], low=lows[i], open_=opens[i], close=closes[i])
        for i in range(len(highs))
    ]

    sl = make_swing(index=2, price=2640.0, swing_type=SwingType.LOW, candles=candles, right_bars=2)
    sh = make_swing(index=6, price=2670.0, swing_type=SwingType.HIGH, candles=candles, right_bars=2)

    breaks = detect_structure(candles, [sl, sh], initial_trend=Direction.BULLISH)
    assert len(breaks) == 1
    event = breaks[0]
    assert isinstance(event, ChoCH)
    assert event.direction == Direction.BEARISH
    assert event.previous_trend == Direction.BULLISH
    assert event.price == 2640.0
    assert event.breaking_candle == candles[9]


def test_structure_anti_lookahead_causal_equivalence() -> None:
    candles = [
        make_candle(0, open_=2636.0, close=2638.0, high=2640.0, low=2635.0),
        make_candle(1, open_=2638.0, close=2644.0, high=2645.0, low=2636.0),
        make_candle(2, open_=2644.0, close=2658.0, high=2660.0, low=2642.0),
        make_candle(3, open_=2658.0, close=2648.0, high=2659.0, low=2646.0),
        make_candle(4, open_=2648.0, close=2644.0, high=2650.0, low=2642.0),
        make_candle(5, open_=2644.0, close=2642.0, high=2648.0, low=2640.0),
        make_candle(6, open_=2642.0, close=2664.0, high=2665.0, low=2641.0),
        make_candle(7, open_=2664.0, close=2668.0, high=2670.0, low=2662.0),
        make_candle(8, open_=2668.0, close=2654.0, high=2669.0, low=2650.0),
        make_candle(9, open_=2654.0, close=2642.0, high=2655.0, low=2638.0),
        make_candle(10, open_=2642.0, close=2630.0, high=2645.0, low=2628.0),
    ]

    sh1 = make_swing(
        index=2, price=2660.0, swing_type=SwingType.HIGH, candles=candles, right_bars=2
    )
    sl1 = make_swing(index=5, price=2640.0, swing_type=SwingType.LOW, candles=candles, right_bars=2)
    swings = [sh1, sl1]

    all_breaks = detect_structure(candles, swings)
    duration = timedelta(seconds=Timeframe.M5.duration_seconds)

    for k in range(len(candles)):
        prefix = candles[: k + 1]
        incremental = detect_structure(prefix, swings)
        as_of_time = candles[k].time + duration
        expected = [b for b in all_breaks if b.confirmed_time <= as_of_time]
        assert incremental == expected


def test_liquidity_pools_detection() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_h1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    s_h2 = Swing(
        time=t0 + timedelta(minutes=30),
        price=2660.4,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=40),
        index=8,
    )

    s_l1 = Swing(
        time=t0 + timedelta(minutes=15),
        price=2630.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )
    s_l2 = Swing(
        time=t0 + timedelta(minutes=45),
        price=2630.5,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=55),
        index=11,
    )

    pools = detect_liquidity_pools([s_h1, s_h2, s_l1, s_l2], tolerance=1.0)
    assert len(pools) == 2

    bsl = [p for p in pools if p.type == LiquidityType.BSL][0]
    assert len(bsl.swings) == 2
    assert bsl.price == pytest.approx(2660.4)

    ssl = [p for p in pools if p.type == LiquidityType.SSL][0]
    assert len(ssl.swings) == 2
    assert ssl.price == pytest.approx(2630.0)


def test_liquidity_sweeps_detection() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_h1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )

    c_sweep = Candle(
        time=t0 + timedelta(minutes=15),
        open=2655.0,
        high=2664.0,
        low=2654.0,
        close=2658.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    c_break = Candle(
        time=t0 + timedelta(minutes=20),
        open=2658.0,
        high=2666.0,
        low=2657.0,
        close=2664.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )

    sweeps = detect_liquidity_sweeps([c_sweep, c_break], [s_h1])
    assert len(sweeps) == 1
    assert sweeps[0].candle == c_sweep
    assert sweeps[0].level == 2660.0
    assert sweeps[0].type == LiquidityType.BSL


def test_dealing_range_and_ote() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    sh = Swing(
        time=t0,
        price=2700.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    sl = Swing(
        time=t0 + timedelta(minutes=15),
        price=2600.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )

    dr = calculate_dealing_range(sh, sl)
    assert dr.high == 2700.0
    assert dr.low == 2600.0
    assert dr.range == pytest.approx(100.0)
    assert dr.equilibrium == pytest.approx(2650.0)

    ote_buy = dr.ote_zone(Direction.BULLISH)
    assert ote_buy[0] == pytest.approx(2621.0)
    assert ote_buy[1] == pytest.approx(2638.0)

    ote_sell = dr.ote_zone(Direction.BEARISH)
    assert ote_sell[0] == pytest.approx(2662.0)
    assert ote_sell[1] == pytest.approx(2679.0)


def test_get_current_dealing_range_from_swings() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    sh = Swing(
        time=t0,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    sl = Swing(
        time=t0 + timedelta(minutes=15),
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )

    dr = get_current_dealing_range([sh, sl])
    assert dr is not None
    assert dr.high == 2680.0
    assert dr.low == 2640.0
    assert dr.equilibrium == pytest.approx(2660.0)

    assert get_current_dealing_range([]) is None
    assert get_current_dealing_range([sh]) is None


def test_structure_empty_inputs_and_trend_continuation() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c1 = make_candle(0, high=2650.0, low=2640.0, close=2645.0, base_time=t0)
    sh = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )

    assert detect_structure([], [sh]) == []
    assert detect_structure([c1], []) == []

    c_break1 = make_candle(1, high=2666.0, low=2655.0, close=2665.0, base_time=t0)
    breaks_bullish = detect_structure([c_break1], [sh], initial_trend=Direction.BULLISH)
    assert len(breaks_bullish) == 1
    assert breaks_bullish[0].direction == Direction.BULLISH
    assert isinstance(breaks_bullish[0], BOS)

    sl = Swing(
        time=t0, price=2640.0, type=SwingType.LOW, confirmed_time=t0 + timedelta(minutes=5), index=0
    )
    c_break2 = make_candle(1, high=2645.0, low=2630.0, close=2635.0, base_time=t0)
    breaks_bearish = detect_structure([c_break2], [sl], initial_trend=Direction.BEARISH)
    assert len(breaks_bearish) == 1
    assert breaks_bearish[0].direction == Direction.BEARISH
    assert isinstance(breaks_bearish[0], BOS)


def test_liquidity_pools_edge_cases() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s = Swing(
        time=t0,
        price=2650.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )

    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        detect_liquidity_pools([s], tolerance=-0.1)

    with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
        detect_liquidity_pools([s], tolerance=float("nan"))

    assert detect_liquidity_pools([]) == []
    assert detect_liquidity_pools([s], tolerance=1.0, include_single_swings=False) == []
    assert len(detect_liquidity_pools([s], tolerance=1.0, include_single_swings=True)) == 1

    s0 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )
    s1 = Swing(
        time=t0 + timedelta(minutes=10),
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=2,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=20),
        price=2660.5,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=25),
        index=4,
    )
    pools_all = detect_liquidity_pools([s0, s1, s2], tolerance=1.0)
    assert len(pools_all) == 2
    eqh = [p for p in pools_all if len(p.swings) == 2][0]
    assert eqh.price == pytest.approx(2660.5)
    single = [p for p in pools_all if len(p.swings) == 1][0]
    assert single.price == pytest.approx(2680.0)

    pools_eqh_only = detect_liquidity_pools(
        [s0, s1, s2], tolerance=1.0, include_single_swings=False
    )
    assert len(pools_eqh_only) == 1
    assert pools_eqh_only[0].price == pytest.approx(2660.5)

    chain_a = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=10,
    )
    chain_b = Swing(
        time=t0 + timedelta(minutes=10),
        price=2661.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=11,
    )
    chain_c = Swing(
        time=t0 + timedelta(minutes=20),
        price=2662.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=25),
        index=12,
    )
    pools_chain = detect_liquidity_pools([chain_a, chain_b, chain_c], tolerance=1.0)
    assert len(pools_chain) == 2
    cluster_ab = [p for p in pools_chain if len(p.swings) == 2][0]
    cluster_c = [p for p in pools_chain if len(p.swings) == 1][0]
    assert cluster_ab.price == pytest.approx(2661.0)
    assert cluster_c.price == pytest.approx(2662.0)


def test_liquidity_sweeps_ssl_and_future_pool() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_l1 = Swing(
        time=t0,
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )

    assert detect_liquidity_sweeps([], [s_l1]) == []
    c_test = make_candle(0, high=2650.0, low=2645.0, close=2648.0, base_time=t0)
    assert detect_liquidity_sweeps([c_test], []) == []

    s_future = Swing(
        time=t0,
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(hours=5),
        index=0,
    )
    assert detect_liquidity_sweeps([c_test], [s_future]) == []

    c_untouched = Candle(
        time=t0 + timedelta(minutes=10),
        open=2650.0,
        high=2655.0,
        low=2645.0,
        close=2648.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    c_sweep_ssl = Candle(
        time=t0 + timedelta(minutes=15),
        open=2645.0,
        high=2648.0,
        low=2636.0,
        close=2642.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    c_break_ssl = Candle(
        time=t0 + timedelta(minutes=20),
        open=2642.0,
        high=2644.0,
        low=2632.0,
        close=2635.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    sweeps = detect_liquidity_sweeps([c_untouched, c_sweep_ssl, c_break_ssl], [s_l1])
    assert len(sweeps) == 1
    assert sweeps[0].type == LiquidityType.SSL
    assert sweeps[0].level == 2640.0
    assert sweeps[0].candle == c_sweep_ssl


def test_get_current_dealing_range_edge_cases() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    sh_old = Swing(
        time=t0,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    sl_old = Swing(
        time=t0 + timedelta(minutes=15),
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )
    sh_inv = Swing(
        time=t0 + timedelta(minutes=30),
        price=2630.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=40),
        index=8,
    )
    sl_inv = Swing(
        time=t0 + timedelta(minutes=45),
        price=2650.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=55),
        index=11,
    )

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        get_current_dealing_range([sh_old, sl_old], as_of_time=datetime(2026, 9, 24, 12, 0))

    dr_filtered = get_current_dealing_range([sh_old, sl_old], as_of_time=t0 + timedelta(minutes=20))
    assert dr_filtered is None

    dr_fallback = get_current_dealing_range([sh_old, sl_old, sh_inv, sl_inv])
    assert dr_fallback is None

    sh_low = Swing(
        time=t0,
        price=2620.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    sl_high = Swing(
        time=t0 + timedelta(minutes=15),
        price=2660.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )
    assert get_current_dealing_range([sh_low, sl_high]) is None


def test_repro_bug_t05_01_bos_does_not_break_obsolete_surpassed_swings() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    ha = Swing(
        time=t0 - timedelta(minutes=30),
        price=2650.0,
        type=SwingType.HIGH,
        confirmed_time=t0,
        index=0,
    )
    hb = Swing(
        time=t0 - timedelta(minutes=15),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=1,
    )

    candles = [
        make_candle(2, high=2666.0, low=2658.0, close=2665.0, base_time=t0),
        make_candle(3, high=2665.0, low=2658.0, close=2663.0, base_time=t0),
    ]

    breaks = detect_structure(candles, [ha, hb], initial_trend=Direction.BULLISH)
    assert len(breaks) == 1
    assert breaks[0].broken_swing == hb
    assert breaks[0].price == 2660.0


def test_repro_r1_choch_preserves_protected_origin_swing() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    sl0 = Swing(
        time=t0,
        price=2630.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )
    sh1 = Swing(
        time=t0 + timedelta(minutes=5),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=1,
    )
    c2 = make_candle(2, high=2666.0, low=2655.0, close=2664.0, base_time=t0)
    c3 = make_candle(3, high=2635.0, low=2618.0, close=2620.0, base_time=t0)

    events = detect_structure([c2, c3], [sl0, sh1], initial_trend=Direction.BEARISH)
    assert len(events) == 2
    assert isinstance(events[0], ChoCH)
    assert events[0].direction == Direction.BULLISH
    assert events[0].price == 2660.0
    assert events[0].broken_swing == sh1

    assert isinstance(events[1], ChoCH)
    assert events[1].direction == Direction.BEARISH
    assert events[1].price == 2630.0
    assert events[1].broken_swing == sl0


def test_repro_r2_bos_preserves_higher_prior_swing() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    h0 = Swing(
        time=t0,
        price=2700.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )
    h1 = Swing(
        time=t0 + timedelta(minutes=5),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=1,
    )
    c2 = make_candle(2, high=2668.0, low=2655.0, close=2665.0, base_time=t0)
    c3 = make_candle(3, high=2710.0, low=2670.0, close=2705.0, base_time=t0)

    events = detect_structure([c2, c3], [h0, h1], initial_trend=Direction.BULLISH)
    assert len(events) == 2
    assert isinstance(events[0], BOS)
    assert events[0].direction == Direction.BULLISH
    assert events[0].price == 2660.0
    assert events[0].broken_swing == h1

    assert isinstance(events[1], BOS)
    assert events[1].direction == Direction.BULLISH
    assert events[1].price == 2700.0
    assert events[1].broken_swing == h0


def test_repro_r3_liquidity_sweeps_future_swing_does_not_alter_past_level() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_a = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=2,
    )
    s_b = Swing(
        time=t0 + timedelta(minutes=5),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=3,
    )
    c_sweep = make_candle(5, high=2660.5, low=2655.0, close=2658.0, base_time=t0)

    sweeps_ab = detect_liquidity_sweeps([c_sweep], [s_a, s_b], tolerance=1.0)
    assert len(sweeps_ab) == 1
    assert sweeps_ab[0].level == 2660.0

    s_c = Swing(
        time=t0 + timedelta(minutes=50),
        price=2660.8,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=60),
        index=10,
    )
    sweeps_abc = detect_liquidity_sweeps([c_sweep], [s_a, s_b, s_c], tolerance=1.0)
    assert len(sweeps_abc) == 1
    assert sweeps_abc[0].level == 2660.0
    assert sweeps_ab[0].level == sweeps_abc[0].level


def test_repro_r4_liquidity_sweeps_single_pool_not_delayed_by_future_cluster() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_a = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=2,
    )
    s_b = Swing(
        time=t0 + timedelta(minutes=50),
        price=2660.3,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=60),
        index=10,
    )
    c_sweep = make_candle(4, high=2662.0, low=2655.0, close=2658.0, base_time=t0)

    sweeps_a = detect_liquidity_sweeps([c_sweep], [s_a], tolerance=1.0)
    assert len(sweeps_a) == 1
    assert sweeps_a[0].level == 2660.0

    sweeps_ab = detect_liquidity_sweeps([c_sweep], [s_a, s_b], tolerance=1.0)
    assert len(sweeps_ab) == 1
    assert sweeps_ab[0].level == 2660.0
    assert sweeps_a[0].level == sweeps_ab[0].level


@pytest.mark.parametrize(
    ("include_single_swings", "atr", "atr_multiplier", "tolerance"),
    [
        (True, None, None, 1.0),
        (False, None, None, 1.0),
        (True, 15.0, 0.1, 1.0),
        (False, 15.0, 0.1, 1.0),
    ],
)
def test_liquidity_sweeps_anti_lookahead_causal_equivalence(
    include_single_swings: bool,
    atr: float | None,
    atr_multiplier: float | None,
    tolerance: float,
) -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0 + timedelta(minutes=5),
        price=2661.3,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=1,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=15),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=25),
        index=3,
    )
    s3 = Swing(
        time=t0 + timedelta(minutes=45),
        price=2665.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=55),
        index=9,
    )
    swings = [s1, s2, s3]

    candles = [
        make_candle(0, high=2655.0, low=2650.0, close=2652.0, base_time=t0),
        make_candle(1, high=2661.3, low=2652.0, close=2655.0, base_time=t0),
        make_candle(2, high=2656.0, low=2650.0, close=2653.0, base_time=t0),
        make_candle(3, high=2660.0, low=2654.0, close=2657.0, base_time=t0),
        make_candle(4, high=2658.0, low=2652.0, close=2655.0, base_time=t0),
        make_candle(5, high=2662.0, low=2655.0, close=2658.0, base_time=t0),
        make_candle(6, high=2668.0, low=2657.0, close=2665.0, base_time=t0),
        make_candle(7, high=2662.0, low=2655.0, close=2658.0, base_time=t0),
        make_candle(8, high=2660.0, low=2652.0, close=2655.0, base_time=t0),
        make_candle(9, high=2665.0, low=2654.0, close=2657.0, base_time=t0),
        make_candle(10, high=2660.0, low=2652.0, close=2655.0, base_time=t0),
        make_candle(11, high=2666.0, low=2658.0, close=2662.0, base_time=t0),
        make_candle(12, high=2660.0, low=2653.0, close=2656.0, base_time=t0),
    ]

    all_sweeps = detect_liquidity_sweeps(
        candles,
        swings,
        tolerance=tolerance,
        atr=atr,
        atr_multiplier=atr_multiplier,
        include_single_swings=include_single_swings,
    )

    if atr is not None:
        eqh_sweeps = [s for s in all_sweeps if set(s.pool.swings) == {s1, s2}]
        assert len(eqh_sweeps) == 1
        assert eqh_sweeps[0].level == 2661.3
        if not include_single_swings:
            assert len(all_sweeps) == 1
    else:
        assert not any(len(s.pool.swings) >= 2 for s in all_sweeps)
        if not include_single_swings:
            assert len(all_sweeps) == 0

    sweeps_full = detect_liquidity_sweeps(
        candles,
        swings,
        tolerance=tolerance,
        atr=atr,
        atr_multiplier=atr_multiplier,
        include_single_swings=True,
    )
    sweeps_clusters = detect_liquidity_sweeps(
        candles,
        swings,
        tolerance=tolerance,
        atr=atr,
        atr_multiplier=atr_multiplier,
        include_single_swings=False,
    )
    assert [s for s in sweeps_full if len(s.pool.swings) >= 2] == sweeps_clusters

    duration = timedelta(seconds=Timeframe.M5.duration_seconds)

    for k in range(len(candles)):
        prefix_candles = candles[: k + 1]
        as_of_time = prefix_candles[-1].time + duration
        prefix_sweeps = detect_liquidity_sweeps(
            prefix_candles,
            swings,
            tolerance=tolerance,
            atr=atr,
            atr_multiplier=atr_multiplier,
            include_single_swings=include_single_swings,
        )
        expected_sweeps = [s for s in all_sweeps if s.confirmed_time <= as_of_time]
        assert prefix_sweeps == expected_sweeps


def test_repro_achado_1_bos_does_not_break_older_high_already_surpassed_by_prior_candle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    h0 = Swing(
        time=t0,
        price=2700.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=5),
        index=0,
    )
    h2 = Swing(
        time=t0 + timedelta(minutes=10),
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=2,
    )
    c1 = make_candle(3, high=2710.0, low=2675.0, close=2705.0, base_time=t0)
    c2 = make_candle(4, high=2708.0, low=2695.0, close=2703.0, base_time=t0)

    breaks = detect_structure([c1, c2], [h0, h2], initial_trend=Direction.BULLISH)
    assert len(breaks) == 1
    assert breaks[0].price == 2680.0
    assert breaks[0].broken_swing == h2


def test_repro_achado_3_swing_formed_after_mitigation_is_not_lost_in_consumed_pool() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_a = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=20),
        index=0,
    )
    c_break = make_candle(4, high=2668.0, low=2655.0, close=2665.0, base_time=t0)

    s_c = Swing(
        time=t0 + timedelta(minutes=30),
        price=2660.5,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=50),
        index=6,
    )
    c_sweep = make_candle(11, high=2663.0, low=2655.0, close=2659.0, base_time=t0)

    sweeps_ac = detect_liquidity_sweeps([c_break, c_sweep], swings=[s_a, s_c])
    assert len(sweeps_ac) == 1
    assert sweeps_ac[0].level == 2660.5

    sweeps_c = detect_liquidity_sweeps([c_break, c_sweep], swings=[s_c])
    assert len(sweeps_c) == 1
    assert sweeps_c[0].level == 2660.5


def test_repro_bug_t05_03_liquidity_pool_sweep_vs_breakout_state_and_consumption() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_h = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )

    c_break = make_candle(3, high=2666.0, low=2658.0, close=2665.0, base_time=t0)
    c_wick = make_candle(4, high=2670.0, low=2655.0, close=2658.0, base_time=t0)
    sweeps_breakout = detect_liquidity_sweeps([c_break, c_wick], [s_h])
    assert len(sweeps_breakout) == 0

    c_sweep1 = make_candle(3, high=2664.0, low=2655.0, close=2658.0, base_time=t0)
    c_sweep2 = make_candle(4, high=2663.0, low=2656.0, close=2657.0, base_time=t0)
    sweeps = detect_liquidity_sweeps([c_sweep1, c_sweep2], [s_h])
    assert len(sweeps) == 1
    assert sweeps[0].pool.swept is True
    assert sweeps[0].pool.swept_at == c_sweep1.time
    assert sweeps[0].pool.swept_by_candle == c_sweep1


def test_repro_bug_t05_04_liquidity_pools_isolated_swings_and_atr_tolerance() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=20),
        price=2660.8,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=30),
        index=6,
    )

    pools_single = detect_liquidity_pools([s1], include_single_swings=True)
    assert len(pools_single) == 1
    assert len(pools_single[0].swings) == 1
    assert pools_single[0].price == 2660.0

    pools_tight = detect_liquidity_pools([s1, s2], tolerance=0.5)
    assert len(pools_tight) == 2

    pools_atr = detect_liquidity_pools([s1, s2], tolerance=0.5, atr=15.0, atr_multiplier=0.1)
    assert len(pools_atr) == 1
    assert len(pools_atr[0].swings) == 2
    assert pools_atr[0].price == pytest.approx(2660.8)

    with pytest.raises(ValueError, match="Both atr and atr_multiplier must be provided together"):
        detect_liquidity_pools([s1], atr=10.0)

    with pytest.raises(ValueError, match="Both atr and atr_multiplier must be provided together"):
        detect_liquidity_pools([s1], atr_multiplier=0.1)

    with pytest.raises(ValueError, match="effective tolerance must be finite and non-negative"):
        detect_liquidity_pools([s1], atr=float("nan"), atr_multiplier=0.1)

    with pytest.raises(ValueError, match="effective tolerance must be finite and non-negative"):
        detect_liquidity_pools([s1], atr=-10.0, atr_multiplier=0.1)


def test_liquidity_pools_as_of_time_and_ssl_breakout() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=15),
        price=2640.5,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=25),
        index=5,
    )

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        detect_liquidity_pools([s1, s2], as_of_time=datetime(2026, 9, 24, 12, 0))

    pools_filtered = detect_liquidity_pools([s1, s2], as_of_time=t0 + timedelta(minutes=15))
    assert len(pools_filtered) == 1
    assert len(pools_filtered[0].swings) == 1

    pool = pools_filtered[0]
    c_break = make_candle(3, high=2642.0, low=2630.0, close=2635.0, base_time=t0)
    c_wick = make_candle(4, high=2644.0, low=2636.0, close=2642.0, base_time=t0)
    sweeps = detect_liquidity_sweeps([c_break, c_wick], pool.swings)
    assert len(sweeps) == 0


def test_detect_liquidity_sweeps_with_swings_parameter() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    c_sweep = make_candle(3, high=2664.0, low=2655.0, close=2658.0, base_time=t0)
    sweeps = detect_liquidity_sweeps([c_sweep], [s1])
    assert len(sweeps) == 1
    assert sweeps[0].level == 2660.0
    assert sweeps[0].type == LiquidityType.BSL


def test_detect_liquidity_sweeps_deduplicates_swings() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=2,
    )
    c_sweep = make_candle(3, high=2664.0, low=2655.0, close=2658.0, base_time=t0)
    sweeps = detect_liquidity_sweeps([c_sweep], [s1, s1])
    assert len(sweeps) == 1
    assert sweeps[0].level == 2660.0


def test_repro_r4_achado_1_detect_liquidity_sweeps_with_swings_honors_atr_tolerance() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=0,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=5),
        price=2661.3,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=1,
    )
    candle = make_candle(4, high=2660.5, low=2655.0, close=2659.0, base_time=t0)
    sweeps = detect_liquidity_sweeps([candle], [s1, s2], atr=15.0, atr_multiplier=0.1)
    assert sweeps == []


def test_repro_r4_achado_2_broken_swing_cannot_form_pool_later_when_single_swings_disabled() -> (
    None
):
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_a = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=20),
        index=0,
    )
    c_break = make_candle(4, high=2668.0, low=2655.0, close=2665.0, base_time=t0)
    s_c = Swing(
        time=t0 + timedelta(minutes=30),
        price=2660.5,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=50),
        index=6,
    )
    c_sweep = make_candle(11, high=2663.0, low=2655.0, close=2659.0, base_time=t0)

    sweeps = detect_liquidity_sweeps(
        [c_break, c_sweep],
        swings=[s_a, s_c],
        include_single_swings=False,
    )
    assert sweeps == []


def test_repro_r5_achado_1_outside_bar_preserves_both_high_and_low_swings() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_high = Swing(
        time=t0 + timedelta(minutes=10),
        price=2670.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=20),
        index=2,
    )
    s_low = Swing(
        time=t0 + timedelta(minutes=10),
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=20),
        index=2,
    )
    c5 = make_candle(5, high=2650.0, low=2638.0, close=2645.0, base_time=t0)

    sweeps_a = detect_liquidity_sweeps([c5], [s_high, s_low])
    assert len(sweeps_a) == 1
    assert sweeps_a[0].type == LiquidityType.SSL
    assert sweeps_a[0].level == 2640.0

    c_break_high = make_candle(4, high=2675.0, low=2660.0, close=2672.0, base_time=t0)
    sweeps_b = detect_liquidity_sweeps([c_break_high, c5], [s_high, s_low])
    assert len(sweeps_b) == 1
    assert sweeps_b[0].type == LiquidityType.SSL
    assert sweeps_b[0].level == 2640.0


def test_repro_r5_achado_2_wick_swept_swing_cannot_inflate_eqh_when_single_swings_disabled() -> (
    None
):
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0 + timedelta(minutes=10),
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=20),
        index=2,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=25),
        price=2660.4,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=35),
        index=5,
    )
    c_sweep_s1 = make_candle(5, high=2661.5, low=2655.0, close=2658.0, base_time=t0)
    c_after = make_candle(8, high=2661.0, low=2655.0, close=2658.0, base_time=t0)

    sweeps = detect_liquidity_sweeps(
        [c_sweep_s1, c_after],
        [s1, s2],
        include_single_swings=False,
    )
    assert sweeps == []


def test_repro_r5_achado_4_detect_structure_outside_bar_preserves_both_swings() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_high = Swing(
        time=t0 + timedelta(minutes=10),
        price=2670.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=20),
        index=2,
    )
    s_low = Swing(
        time=t0 + timedelta(minutes=10),
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=t0 + timedelta(minutes=20),
        index=2,
    )
    c1 = make_candle(4, high=2675.0, low=2660.0, close=2672.0, base_time=t0)
    c2 = make_candle(5, high=2650.0, low=2635.0, close=2638.0, base_time=t0)

    events = detect_structure([c1, c2], [s_high, s_low])
    assert len(events) == 2
    assert events[0].broken_swing == s_high
    assert events[1].broken_swing == s_low


def test_partial_wick_within_pool_consumes_lower_swing_without_sweep() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s1 = Swing(
        time=t0,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=10),
        index=0,
    )
    s2 = Swing(
        time=t0 + timedelta(minutes=5),
        price=2660.8,
        type=SwingType.HIGH,
        confirmed_time=t0 + timedelta(minutes=15),
        index=1,
    )
    c_partial = make_candle(4, high=2660.3, low=2655.0, close=2658.0, base_time=t0)
    c_higher = make_candle(5, high=2661.5, low=2655.0, close=2659.0, base_time=t0)

    sweeps_false = detect_liquidity_sweeps(
        [c_partial, c_higher],
        [s1, s2],
        tolerance=1.0,
        include_single_swings=False,
    )
    assert sweeps_false == []

    sweeps_true = detect_liquidity_sweeps(
        [c_partial, c_higher],
        [s1, s2],
        tolerance=1.0,
        include_single_swings=True,
    )
    assert len(sweeps_true) == 1
    assert sweeps_true[0].level == 2660.8
    assert sweeps_true[0].pool.swings == (s2,)
