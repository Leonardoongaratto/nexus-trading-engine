from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from xau_bot.domain.models import (
    BOS,
    Candle,
    Direction,
    FVGType,
    OrderBlockType,
    Swing,
    SwingType,
    Timeframe,
)
from xau_bot.smc.poi import (
    _resolve_atr,
    detect_fvg,
    detect_order_blocks,
    get_active_fvgs,
    get_active_order_blocks,
)


def make_candle(
    idx: int,
    open: float,
    high: float,
    low: float,
    close: float,
    base_time: datetime | None = None,
    timeframe: Timeframe = Timeframe.M5,
) -> Candle:
    t0 = base_time or datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t = t0 + timedelta(seconds=idx * timeframe.duration_seconds)
    return Candle(
        time=t,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        timeframe=timeframe,
    )


def test_detect_fvg_bullish_and_bearish() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0),
        make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0, base_time=t0),
        make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0),
        make_candle(3, open=2662.0, high=2662.0, low=2650.0, close=2651.0, base_time=t0),
        make_candle(4, open=2651.0, high=2653.0, low=2645.0, close=2647.0, base_time=t0),
    ]

    fvgs = detect_fvg(candles)
    assert len(fvgs) == 2

    bull_fvg = fvgs[0]
    assert bull_fvg.type == FVGType.BULLISH
    assert bull_fvg.top == 2655.0
    assert bull_fvg.bottom == 2652.0
    assert bull_fvg.size == pytest.approx(3.0)
    assert bull_fvg.consequent_encroachment == pytest.approx(2653.5)
    assert bull_fvg.time == candles[1].time
    assert bull_fvg.confirmed_time == candles[2].time + timedelta(minutes=5)

    bear_fvg = fvgs[1]
    assert bear_fvg.type == FVGType.BEARISH
    assert bear_fvg.top == 2655.0
    assert bear_fvg.bottom == 2653.0
    assert bear_fvg.size == pytest.approx(2.0)
    assert bear_fvg.consequent_encroachment == pytest.approx(2654.0)


def test_detect_fvg_atr_threshold_filtering() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0),
        make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0, base_time=t0),
        make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0),
    ]

    fvgs_all = detect_fvg(candles, min_size=1.0)
    assert len(fvgs_all) == 1

    fvgs_filtered = detect_fvg(candles, min_size=5.0)
    assert len(fvgs_filtered) == 0

    fvgs_atr_pass = detect_fvg(candles, atr=10.0, atr_multiplier=0.2)
    assert len(fvgs_atr_pass) == 1

    fvgs_atr_reject = detect_fvg(candles, atr=10.0, atr_multiplier=0.4)
    assert len(fvgs_atr_reject) == 0


def test_detect_fvg_mitigation_and_ce_test() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0),
        make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0, base_time=t0),
        make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0),
        make_candle(3, open=2662.0, high=2662.0, low=2654.5, close=2656.0, base_time=t0),
        make_candle(4, open=2656.0, high=2657.0, low=2653.0, close=2654.0, base_time=t0),
        make_candle(5, open=2654.0, high=2655.0, low=2651.0, close=2651.5, base_time=t0),
    ]

    fvgs_at_2 = detect_fvg(candles[:3])
    assert len(fvgs_at_2) == 1
    assert not fvgs_at_2[0].mitigated
    assert not fvgs_at_2[0].ce_tested
    assert not fvgs_at_2[0].invalidated

    fvgs_at_3 = detect_fvg(candles[:4])
    assert fvgs_at_3[0].mitigated
    assert fvgs_at_3[0].mitigated_at == candles[3].time + timedelta(minutes=5)
    assert not fvgs_at_3[0].ce_tested
    assert not fvgs_at_3[0].invalidated

    fvgs_at_4 = detect_fvg(candles[:5])
    assert fvgs_at_4[0].mitigated
    assert fvgs_at_4[0].ce_tested
    assert fvgs_at_4[0].ce_tested_at == candles[4].time + timedelta(minutes=5)
    assert not fvgs_at_4[0].invalidated

    fvgs_at_5 = detect_fvg(candles[:6], invalidation_by_close=True)
    assert fvgs_at_5[0].invalidated
    assert fvgs_at_5[0].invalidated_at == candles[5].time + timedelta(minutes=5)

    active = get_active_fvgs(fvgs_at_5)
    assert len(active) == 0


def test_detect_fvg_invalidation_by_close_vs_wick() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0),
        make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0, base_time=t0),
        make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0),
        make_candle(3, open=2662.0, high=2662.0, low=2651.0, close=2653.0, base_time=t0),
    ]

    fvgs_close = detect_fvg(candles, invalidation_by_close=True)
    assert not fvgs_close[0].invalidated
    assert fvgs_close[0].mitigated
    assert fvgs_close[0].ce_tested

    fvgs_wick = detect_fvg(candles, invalidation_by_close=False)
    assert fvgs_wick[0].invalidated


def test_detect_fvg_anti_lookahead_causal_equivalence() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2648.0, close=2651.0, base_time=t0),
        make_candle(1, open=2651.0, high=2662.0, low=2651.0, close=2660.0, base_time=t0),
        make_candle(2, open=2660.0, high=2665.0, low=2656.0, close=2663.0, base_time=t0),
        make_candle(3, open=2663.0, high=2664.0, low=2655.0, close=2657.0, base_time=t0),
        make_candle(4, open=2657.0, high=2658.0, low=2653.0, close=2655.0, base_time=t0),
        make_candle(5, open=2655.0, high=2656.0, low=2646.0, close=2647.0, base_time=t0),
        make_candle(6, open=2647.0, high=2648.0, low=2638.0, close=2640.0, base_time=t0),
        make_candle(7, open=2640.0, high=2643.0, low=2639.0, close=2641.0, base_time=t0),
    ]

    duration = timedelta(seconds=Timeframe.M5.duration_seconds)
    full_fvgs = detect_fvg(candles)

    for k in range(2, len(candles)):
        prefix = candles[: k + 1]
        as_of_time = prefix[-1].time + duration
        prefix_fvgs = detect_fvg(prefix)
        act_prefix = get_active_fvgs(prefix_fvgs, as_of_time=as_of_time, include_mitigated=True)
        act_full = get_active_fvgs(full_fvgs, as_of_time=as_of_time, include_mitigated=True)
        assert act_prefix == act_full


def test_detect_order_blocks_bullish_from_bos() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2655.0, high=2662.0, low=2654.0, close=2661.0, base_time=t0),
        make_candle(1, open=2661.0, high=2661.5, low=2652.0, close=2653.0, base_time=t0),
        make_candle(2, open=2653.0, high=2665.0, low=2653.0, close=2664.0, base_time=t0),
    ]

    h0 = Swing(
        time=candles[0].time,
        price=2662.0,
        type=SwingType.HIGH,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=h0,
        breaking_candle=candles[2],
        price=2662.0,
    )

    obs = detect_order_blocks(candles, structure_events=[bos])
    assert len(obs) == 1
    ob = obs[0]
    assert ob.type == OrderBlockType.BULLISH
    assert ob.candle == candles[1]
    assert ob.high == 2661.5
    assert ob.low == 2652.0
    assert ob.mean_threshold == pytest.approx((2661.5 + 2652.0) / 2.0)
    assert ob.confirmed_time == bos.confirmed_time
    assert ob.breaking_event == bos


def test_detect_order_blocks_bearish_from_bos() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2665.0, high=2666.0, low=2658.0, close=2659.0, base_time=t0),
        make_candle(1, open=2659.0, high=2667.0, low=2659.0, close=2666.5, base_time=t0),
        make_candle(2, open=2666.5, high=2667.0, low=2655.0, close=2656.0, base_time=t0),
    ]

    l0 = Swing(
        time=candles[0].time,
        price=2658.0,
        type=SwingType.LOW,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=l0,
        breaking_candle=candles[2],
        price=2658.0,
    )

    obs = detect_order_blocks(candles, structure_events=[bos])
    assert len(obs) == 1
    ob = obs[0]
    assert ob.type == OrderBlockType.BEARISH
    assert ob.candle == candles[1]
    assert ob.high == 2667.0
    assert ob.low == 2659.0
    assert ob.mean_threshold == pytest.approx((2667.0 + 2659.0) / 2.0)
    assert ob.confirmed_time == bos.confirmed_time


def test_detect_order_blocks_from_fvg() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2655.0, high=2656.0, low=2648.0, close=2649.0, base_time=t0),
        make_candle(1, open=2649.0, high=2662.0, low=2649.0, close=2660.0, base_time=t0),
        make_candle(2, open=2660.0, high=2666.0, low=2658.0, close=2665.0, base_time=t0),
    ]

    fvgs = detect_fvg(candles)
    assert len(fvgs) == 1

    obs = detect_order_blocks(candles, fvgs=fvgs)
    assert len(obs) == 1
    assert obs[0].type == OrderBlockType.BULLISH
    assert obs[0].candle == candles[0]
    assert obs[0].fvg == fvgs[0]


def test_detect_order_blocks_displacement_filtering() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2655.0, high=2660.0, low=2654.0, close=2659.0, base_time=t0),
        make_candle(1, open=2659.0, high=2659.5, low=2655.0, close=2656.0, base_time=t0),
        make_candle(2, open=2656.0, high=2662.0, low=2656.0, close=2661.0, base_time=t0),
    ]

    h0 = Swing(
        time=candles[0].time,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=h0,
        breaking_candle=candles[2],
        price=2660.0,
    )

    obs_pass = detect_order_blocks(
        candles, structure_events=[bos], atr=10.0, min_displacement_multiplier=0.5
    )
    assert len(obs_pass) == 1

    obs_reject = detect_order_blocks(
        candles, structure_events=[bos], atr=10.0, min_displacement_multiplier=1.5
    )
    assert len(obs_reject) == 0


def test_detect_order_blocks_mitigation_and_invalidation() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2655.0, high=2662.0, low=2654.0, close=2661.0, base_time=t0),
        make_candle(1, open=2661.0, high=2661.5, low=2652.0, close=2653.0, base_time=t0),
        make_candle(2, open=2653.0, high=2665.0, low=2653.0, close=2664.0, base_time=t0),
        make_candle(3, open=2664.0, high=2665.0, low=2662.0, close=2663.0, base_time=t0),
        make_candle(4, open=2662.0, high=2663.0, low=2651.0, close=2654.0, base_time=t0),
        make_candle(5, open=2654.0, high=2655.0, low=2648.0, close=2650.0, base_time=t0),
    ]

    h0 = Swing(
        time=candles[0].time,
        price=2662.0,
        type=SwingType.HIGH,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=h0,
        breaking_candle=candles[2],
        price=2662.0,
    )

    obs_at_3 = detect_order_blocks(candles[:4], structure_events=[bos])
    assert len(obs_at_3) == 1
    assert not obs_at_3[0].mitigated
    assert not obs_at_3[0].invalidated

    obs_at_4 = detect_order_blocks(candles[:5], structure_events=[bos])
    assert obs_at_4[0].mitigated
    assert obs_at_4[0].mitigated_at == candles[4].time + timedelta(minutes=5)
    assert not obs_at_4[0].invalidated

    obs_at_5 = detect_order_blocks(candles[:6], structure_events=[bos])
    assert obs_at_5[0].invalidated
    assert obs_at_5[0].invalidated_at == candles[5].time + timedelta(minutes=5)

    active = get_active_order_blocks(obs_at_5)
    assert len(active) == 0


def test_detect_order_blocks_anti_lookahead_causal_equivalence() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2655.0, high=2662.0, low=2654.0, close=2661.0, base_time=t0),
        make_candle(1, open=2661.0, high=2661.5, low=2652.0, close=2653.0, base_time=t0),
        make_candle(2, open=2663.0, high=2668.0, low=2663.0, close=2666.0, base_time=t0),
        make_candle(3, open=2666.0, high=2667.0, low=2660.0, close=2662.0, base_time=t0),
        make_candle(4, open=2662.0, high=2663.0, low=2655.0, close=2658.0, base_time=t0),
    ]

    h0 = Swing(
        time=candles[0].time,
        price=2662.0,
        type=SwingType.HIGH,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=h0,
        breaking_candle=candles[2],
        price=2662.0,
    )

    duration = timedelta(seconds=Timeframe.M5.duration_seconds)
    full_fvgs = detect_fvg(candles)
    full_obs = detect_order_blocks(candles, structure_events=[bos], fvgs=full_fvgs)

    for k in range(len(candles)):
        prefix = candles[: k + 1]
        as_of_time = prefix[-1].time + duration
        prefix_events = [bos] if prefix[-1].time >= bos.breaking_candle.time else []
        prefix_fvgs = detect_fvg(prefix)
        prefix_obs = detect_order_blocks(prefix, structure_events=prefix_events, fvgs=prefix_fvgs)
        act_prefix = get_active_order_blocks(
            prefix_obs, as_of_time=as_of_time, include_mitigated=True
        )
        act_full = get_active_order_blocks(full_obs, as_of_time=as_of_time, include_mitigated=True)
        assert act_prefix == act_full


def test_poi_empty_inputs_and_edge_cases() -> None:
    assert detect_fvg([]) == []
    assert detect_fvg([make_candle(0, 2650.0, 2652.0, 2649.0, 2651.0)]) == []
    assert detect_order_blocks([]) == []
    assert detect_order_blocks([make_candle(0, 2650.0, 2652.0, 2649.0, 2651.0)]) == []
    assert get_active_fvgs([]) == []
    assert get_active_order_blocks([]) == []


def test_detect_fvg_bearish_mitigation_and_invalidation() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2665.0, high=2666.0, low=2662.0, close=2663.0, base_time=t0),
        make_candle(1, open=2663.0, high=2663.0, low=2650.0, close=2651.0, base_time=t0),
        make_candle(2, open=2651.0, high=2654.0, low=2648.0, close=2649.0, base_time=t0),
        make_candle(3, open=2649.0, high=2656.0, low=2648.0, close=2652.0, base_time=t0),
        make_candle(4, open=2652.0, high=2659.0, low=2651.0, close=2655.0, base_time=t0),
        make_candle(5, open=2655.0, high=2664.0, low=2654.0, close=2663.0, base_time=t0),
    ]

    fvgs_at_2 = detect_fvg(candles[:3])
    assert len(fvgs_at_2) == 1
    assert fvgs_at_2[0].type == FVGType.BEARISH
    assert fvgs_at_2[0].top == 2662.0
    assert fvgs_at_2[0].bottom == 2654.0
    assert fvgs_at_2[0].consequent_encroachment == pytest.approx(2658.0)
    assert not fvgs_at_2[0].mitigated

    fvgs_at_3 = detect_fvg(candles[:4])
    assert fvgs_at_3[0].mitigated
    assert not fvgs_at_3[0].ce_tested
    assert not fvgs_at_3[0].invalidated

    fvgs_at_4 = detect_fvg(candles[:5])
    assert fvgs_at_4[0].mitigated
    assert fvgs_at_4[0].ce_tested
    assert not fvgs_at_4[0].invalidated

    fvgs_at_5_close = detect_fvg(candles[:6], invalidation_by_close=True)
    assert fvgs_at_5_close[0].invalidated
    assert fvgs_at_5_close[0].invalidated_at == candles[5].time + timedelta(minutes=5)

    c6 = make_candle(6, open=2668.0, high=2670.0, low=2667.0, close=2669.0, base_time=t0)
    fvgs_at_6 = detect_fvg(candles[:6] + [c6], invalidation_by_close=True)
    assert fvgs_at_6[0].invalidated

    c5_wick = make_candle(5, open=2655.0, high=2663.0, low=2654.0, close=2660.0, base_time=t0)
    fvgs_at_5_wick = detect_fvg(candles[:5] + [c5_wick], invalidation_by_close=False)
    assert fvgs_at_5_wick[0].invalidated


def test_detect_order_blocks_bearish_mitigation_and_invalidation() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2660.0, high=2662.0, low=2658.0, close=2659.0, base_time=t0),
        make_candle(1, open=2659.0, high=2668.0, low=2659.0, close=2667.0, base_time=t0),
        make_candle(2, open=2667.0, high=2667.0, low=2650.0, close=2652.0, base_time=t0),
        make_candle(3, open=2652.0, high=2655.0, low=2650.0, close=2653.0, base_time=t0),
        make_candle(4, open=2653.0, high=2661.0, low=2652.0, close=2658.0, base_time=t0),
        make_candle(5, open=2658.0, high=2670.0, low=2657.0, close=2669.0, base_time=t0),
    ]

    l0 = Swing(
        time=candles[0].time,
        price=2658.0,
        type=SwingType.LOW,
        confirmed_time=candles[1].time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=l0,
        breaking_candle=candles[2],
        price=2658.0,
    )

    obs_at_3 = detect_order_blocks(candles[:4], structure_events=[bos])
    assert len(obs_at_3) == 1
    assert obs_at_3[0].type == OrderBlockType.BEARISH
    assert not obs_at_3[0].mitigated

    obs_at_4 = detect_order_blocks(candles[:5], structure_events=[bos])
    assert obs_at_4[0].mitigated
    assert not obs_at_4[0].invalidated

    obs_at_5 = detect_order_blocks(candles[:6], structure_events=[bos])
    assert obs_at_5[0].invalidated
    assert obs_at_5[0].invalidated_at == candles[5].time + timedelta(minutes=5)


def test_detect_order_blocks_from_bearish_fvg() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2660.0, high=2668.0, low=2659.0, close=2667.0, base_time=t0),
        make_candle(1, open=2667.0, high=2667.0, low=2654.0, close=2655.0, base_time=t0),
        make_candle(2, open=2655.0, high=2658.0, low=2650.0, close=2651.0, base_time=t0),
    ]

    fvgs = detect_fvg(candles)
    assert len(fvgs) == 1
    assert fvgs[0].type == FVGType.BEARISH

    obs = detect_order_blocks(candles, fvgs=fvgs)
    assert len(obs) == 1
    assert obs[0].type == OrderBlockType.BEARISH
    assert obs[0].candle == candles[0]
    assert obs[0].fvg == fvgs[0]


def test_detect_order_blocks_unmatched_context_and_no_opposing_candle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2650.0, high=2655.0, low=2649.0, close=2654.0, base_time=t0),
        make_candle(1, open=2654.0, high=2660.0, low=2653.0, close=2659.0, base_time=t0),
        make_candle(2, open=2659.0, high=2665.0, low=2658.0, close=2664.0, base_time=t0),
    ]

    outside_candle = make_candle(
        99, open=2700.0, high=2705.0, low=2695.0, close=2704.0, base_time=t0
    )
    swing_h = Swing(
        time=t0 - timedelta(minutes=10),
        price=2650.0,
        type=SwingType.HIGH,
        confirmed_time=t0,
        index=0,
    )
    unmatched_bos = BOS(
        time=outside_candle.time,
        confirmed_time=outside_candle.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=swing_h,
        breaking_candle=outside_candle,
        price=2650.0,
    )
    assert detect_order_blocks(candles, structure_events=[unmatched_bos]) == []

    matched_bos_all_green = BOS(
        time=candles[2].time,
        confirmed_time=candles[2].time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=swing_h,
        breaking_candle=candles[2],
        price=2650.0,
    )
    assert detect_order_blocks(candles, structure_events=[matched_bos_all_green]) == []


def test_active_helpers_filters_and_validations() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2649.0, close=2650.0, base_time=t0)
    c1 = make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0, base_time=t0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    c3 = make_candle(3, open=2662.0, high=2662.0, low=2653.0, close=2658.0, base_time=t0)
    c4 = make_candle(4, open=2658.0, high=2658.0, low=2651.0, close=2654.0, base_time=t0)

    fvgs = detect_fvg([c0, c1, c2, c3, c4])
    assert len(fvgs) == 1

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        get_active_fvgs(fvgs, as_of_time=datetime(2026, 9, 24, 12, 10))

    active_before = get_active_fvgs(fvgs, as_of_time=c0.time)
    assert len(active_before) == 0

    active_no_mitigated = get_active_fvgs(fvgs, include_mitigated=False)
    assert len(active_no_mitigated) == 0

    active_with_mitigated = get_active_fvgs(fvgs, include_mitigated=True)
    assert len(active_with_mitigated) == 1

    obs = detect_order_blocks([c0, c1, c2, c3, c4], fvgs=fvgs)
    assert len(obs) == 1

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        get_active_order_blocks(obs, as_of_time=datetime(2026, 9, 24, 12, 10))

    ob_active_before = get_active_order_blocks(obs, as_of_time=c0.time)
    assert len(ob_active_before) == 0

    ob_active_at_c3 = get_active_order_blocks(obs, as_of_time=c3.time, include_mitigated=False)
    assert len(ob_active_at_c3) == 1

    ob_active_no_mit = get_active_order_blocks(obs, include_mitigated=False)
    assert len(ob_active_no_mit) == 0

    ob_active_with_mit = get_active_order_blocks(obs, include_mitigated=True)
    assert len(ob_active_with_mit) == 1


def test_detect_fvg_bearish_size_filter_and_wick_invalidation() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2661.0, low=2655.0, close=2656.0, base_time=t0)
    c1 = make_candle(1, open=2656.0, high=2656.0, low=2650.0, close=2651.0, base_time=t0)
    c2 = make_candle(2, open=2651.0, high=2654.6, low=2648.0, close=2649.0, base_time=t0)
    assert detect_fvg([c0, c1, c2], min_size=1.0) == []

    c3_safe = make_candle(3, open=2649.0, high=2653.0, low=2647.0, close=2648.0, base_time=t0)
    fvgs_safe = detect_fvg([c0, c1, c2, c3_safe], min_size=0.1, invalidation_by_close=False)
    assert len(fvgs_safe) == 1
    assert not fvgs_safe[0].invalidated

    c3_inv = make_candle(3, open=2649.0, high=2656.0, low=2648.0, close=2650.0, base_time=t0)
    fvgs_inv = detect_fvg([c0, c1, c2, c3_inv], min_size=0.1, invalidation_by_close=False)
    assert len(fvgs_inv) == 1
    assert fvgs_inv[0].invalidated
    assert fvgs_inv[0].invalidated_at == c3_inv.time + timedelta(minutes=5)

    cb0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0)
    cb1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0, base_time=t0)
    cb2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    cb3_safe = make_candle(3, open=2662.0, high=2662.0, low=2653.0, close=2658.0, base_time=t0)
    bull_fvgs = detect_fvg([cb0, cb1, cb2, cb3_safe], invalidation_by_close=False)
    assert len(bull_fvgs) == 1
    assert not bull_fvgs[0].invalidated


def test_detect_order_blocks_bearish_bos_displacement_and_no_bullish_candle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2661.0, low=2658.0, close=2659.0, base_time=t0)
    c1 = make_candle(1, open=2659.0, high=2662.0, low=2658.0, close=2661.0, base_time=t0)
    c2 = make_candle(2, open=2661.0, high=2661.0, low=2650.0, close=2651.0, base_time=t0)
    candles = [c0, c1, c2]

    l0 = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    bearish_event = BOS(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=l0,
        breaking_candle=c2,
        price=2655.0,
    )

    assert (
        detect_order_blocks(
            candles,
            structure_events=[bearish_event],
            atr=50.0,
            min_displacement_multiplier=10.0,
        )
        == []
    )

    c0_red = make_candle(0, open=2660.0, high=2661.0, low=2656.0, close=2657.0, base_time=t0)
    c1_red = make_candle(1, open=2657.0, high=2658.0, low=2653.0, close=2654.0, base_time=t0)
    c2_red = make_candle(2, open=2654.0, high=2654.0, low=2645.0, close=2646.0, base_time=t0)
    l0_red = Swing(
        time=c0_red.time,
        price=2655.0,
        type=SwingType.LOW,
        confirmed_time=c1_red.time + timedelta(minutes=5),
        index=0,
    )
    red_event = BOS(
        time=c2_red.time,
        confirmed_time=c2_red.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=l0_red,
        breaking_candle=c2_red,
        price=2655.0,
    )
    assert detect_order_blocks([c0_red, c1_red, c2_red], structure_events=[red_event]) == []


def test_detect_order_blocks_fvg_unmatched_and_displacement_filters() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2650.0, base_time=t0)
    c1 = make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0, base_time=t0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    fvgs = detect_fvg([c0, c1, c2])
    assert len(fvgs) == 1

    external_candle = make_candle(
        99, open=2600.0, high=2601.0, low=2599.0, close=2600.0, base_time=t0
    )
    assert detect_order_blocks([external_candle], fvgs=fvgs) == []

    c0_green = make_candle(0, open=2649.0, high=2652.0, low=2649.0, close=2651.0, base_time=t0)
    fvg_all_green = detect_fvg([c0_green, c1, c2])
    assert detect_order_blocks([c0_green, c1, c2], fvgs=fvg_all_green) == []

    assert (
        detect_order_blocks(
            [c0, c1, c2],
            fvgs=fvgs,
            atr=50.0,
            min_displacement_multiplier=10.0,
        )
        == []
    )

    cb0 = make_candle(0, open=2655.0, high=2658.0, low=2654.0, close=2657.0, base_time=t0)
    cb1 = make_candle(1, open=2657.0, high=2657.0, low=2645.0, close=2646.0, base_time=t0)
    cb2 = make_candle(2, open=2646.0, high=2647.0, low=2639.0, close=2640.0, base_time=t0)
    bearish_fvgs = detect_fvg([cb0, cb1, cb2])
    assert len(bearish_fvgs) == 1

    cb0_red = make_candle(0, open=2658.0, high=2658.0, low=2654.0, close=2655.0, base_time=t0)
    fvg_all_red = detect_fvg([cb0_red, cb1, cb2])
    assert detect_order_blocks([cb0_red, cb1, cb2], fvgs=fvg_all_red) == []

    assert (
        detect_order_blocks(
            [cb0, cb1, cb2],
            fvgs=bearish_fvgs,
            atr=50.0,
            min_displacement_multiplier=10.0,
        )
        == []
    )


def test_active_helpers_historical_as_of_time_branches() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2649.0, close=2650.0, base_time=t0)
    c1 = make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0, base_time=t0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    c3 = make_candle(3, open=2662.0, high=2665.0, low=2658.0, close=2663.0, base_time=t0)
    c4 = make_candle(4, open=2663.0, high=2663.0, low=2653.0, close=2658.0, base_time=t0)
    c5 = make_candle(5, open=2658.0, high=2658.0, low=2648.0, close=2648.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5]

    fvgs = detect_fvg(candles)
    assert len(fvgs) == 1
    assert fvgs[0].mitigated
    assert fvgs[0].mitigated_at == c4.time + timedelta(minutes=5)
    assert fvgs[0].invalidated
    assert fvgs[0].invalidated_at == c5.time + timedelta(minutes=5)

    active_at_c3 = get_active_fvgs(
        fvgs, as_of_time=c3.time + timedelta(minutes=5), include_mitigated=False
    )
    assert len(active_at_c3) == 1

    active_at_c4_no_mit = get_active_fvgs(
        fvgs, as_of_time=c4.time + timedelta(minutes=5), include_mitigated=False
    )
    assert len(active_at_c4_no_mit) == 0

    active_at_c4_with_mit = get_active_fvgs(
        fvgs, as_of_time=c4.time + timedelta(minutes=5), include_mitigated=True
    )
    assert len(active_at_c4_with_mit) == 1

    active_at_c5 = get_active_fvgs(
        fvgs, as_of_time=c5.time + timedelta(minutes=5), include_mitigated=True
    )
    assert len(active_at_c5) == 0

    obs = detect_order_blocks(candles, fvgs=fvgs)
    assert len(obs) == 1
    assert obs[0].mitigated
    assert obs[0].mitigated_at == c5.time + timedelta(minutes=5)
    assert obs[0].invalidated
    assert obs[0].invalidated_at == c5.time + timedelta(minutes=5)

    ob_active_at_c4 = get_active_order_blocks(
        obs, as_of_time=c4.time + timedelta(minutes=5), include_mitigated=False
    )
    assert len(ob_active_at_c4) == 1

    ob_active_at_c5 = get_active_order_blocks(
        obs, as_of_time=c5.time + timedelta(minutes=5), include_mitigated=True
    )
    assert len(ob_active_at_c5) == 0


def test_repro_bug_t06_01_and_02_lookahead_and_prefix_equivalence() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2649.0, close=2650.0, base_time=t0)
    c1 = make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0, base_time=t0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    c3 = make_candle(3, open=2662.0, high=2665.0, low=2658.0, close=2663.0, base_time=t0)
    c4 = make_candle(4, open=2663.0, high=2663.0, low=2653.0, close=2658.0, base_time=t0)
    c5 = make_candle(5, open=2658.0, high=2658.0, low=2648.0, close=2648.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5]

    full_fvgs = detect_fvg(candles)
    prefix_c4_fvgs = detect_fvg(candles[:5])

    active_full_at_c5_open = get_active_fvgs(full_fvgs, as_of_time=c5.time, include_mitigated=True)
    active_prefix_at_c5_open = get_active_fvgs(
        prefix_c4_fvgs, as_of_time=c5.time, include_mitigated=True
    )
    assert len(active_full_at_c5_open) == len(active_prefix_at_c5_open)

    for k in range(2, len(candles)):
        close_k = candles[k].time + timedelta(seconds=candles[k].timeframe.duration_seconds)
        prefix_fvgs = detect_fvg(candles[: k + 1])
        act_prefix = get_active_fvgs(prefix_fvgs, as_of_time=close_k, include_mitigated=True)
        act_full = get_active_fvgs(full_fvgs, as_of_time=close_k, include_mitigated=True)
        assert act_prefix == act_full


def test_repro_bug_t06_03_atr_series_causal_filtering() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2650.0, base_time=t0)
    c1 = make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0, base_time=t0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0, base_time=t0)
    candles = [c0, c1, c2]

    atr_series: list[float | None] = [None, None, 2.0]
    fvgs = detect_fvg(candles, atr=atr_series, atr_multiplier=1.0)
    assert len(fvgs) == 1

    atr_series_too_high: list[float | None] = [None, None, 10.0]
    fvgs_filtered = detect_fvg(candles, atr=atr_series_too_high, atr_multiplier=1.0)
    assert len(fvgs_filtered) == 0


def test_repro_bug_t06_04_ob_deduplication_causal_leakage() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2648.0, close=2649.0, base_time=t0)
    c1 = make_candle(1, open=2649.0, high=2655.0, low=2649.0, close=2654.0, base_time=t0)
    c2 = make_candle(2, open=2654.0, high=2662.0, low=2654.0, close=2660.0, base_time=t0)
    c3 = make_candle(3, open=2660.0, high=2662.0, low=2654.0, close=2661.0, base_time=t0)
    c4 = make_candle(4, open=2661.0, high=2670.0, low=2660.0, close=2668.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4]

    fvgs = detect_fvg(candles)
    assert len(fvgs) == 1
    assert fvgs[0].confirmed_time == c2.time + timedelta(minutes=5)

    swing_high = Swing(
        time=c0.time,
        price=2665.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=c4.time,
        confirmed_time=c4.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        breaking_candle=c4,
        broken_swing=swing_high,
        price=swing_high.price,
    )

    full_obs = detect_order_blocks(candles, structure_events=[bos], fvgs=fvgs)
    prefix_obs = detect_order_blocks(candles[:4], structure_events=[], fvgs=detect_fvg(candles[:4]))

    as_of_c3_close = c3.time + timedelta(minutes=5)
    act_full = get_active_order_blocks(full_obs, as_of_time=as_of_c3_close, include_mitigated=True)
    act_prefix = get_active_order_blocks(
        prefix_obs, as_of_time=as_of_c3_close, include_mitigated=True
    )

    assert len(act_prefix) == 1
    assert len(act_full) == 1
    assert act_prefix == act_full


def test_repro_bug_t06_05_flat_candle_crash_in_order_block() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c_bear = make_candle(0, open=2651.0, high=2652.0, low=2647.0, close=2648.0, base_time=t0)
    c_flat = make_candle(1, open=2650.0, high=2650.0, low=2650.0, close=2650.0, base_time=t0)
    c_bull1 = make_candle(2, open=2648.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0)
    c_bull2 = make_candle(3, open=2654.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c_bear, c_flat, c_bull1, c_bull2]

    swing_high = Swing(
        time=c_bear.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c_bull1.time + timedelta(minutes=5),
        index=0,
    )
    bos_bull = BOS(
        time=c_bull2.time,
        confirmed_time=c_bull2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        breaking_candle=c_bull2,
        broken_swing=swing_high,
        price=swing_high.price,
    )

    obs_bull = detect_order_blocks(candles, structure_events=[bos_bull])
    assert len(obs_bull) == 1
    assert obs_bull[0].candle == c_bear

    c_bull = make_candle(0, open=2648.0, high=2653.0, low=2648.0, close=2652.0, base_time=t0)
    c_bear1 = make_candle(2, open=2652.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c_bear2 = make_candle(3, open=2646.0, high=2646.0, low=2635.0, close=2636.0, base_time=t0)
    candles_bear = [c_bull, c_flat, c_bear1, c_bear2]
    swing_low = Swing(
        time=c_bull.time,
        price=2648.0,
        type=SwingType.LOW,
        confirmed_time=c_bear1.time + timedelta(minutes=5),
        index=0,
    )
    bos_bear = BOS(
        time=c_bear2.time,
        confirmed_time=c_bear2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        breaking_candle=c_bear2,
        broken_swing=swing_low,
        price=swing_low.price,
    )
    obs_bear = detect_order_blocks(candles_bear, structure_events=[bos_bear])
    assert len(obs_bear) == 1
    assert obs_bear[0].candle == c_bull

    fvgs_bull = detect_fvg(candles)
    obs_fvg_bull = detect_order_blocks(candles, fvgs=fvgs_bull)
    assert len(obs_fvg_bull) == 1
    assert obs_fvg_bull[0].candle == c_bear

    fvgs_bear = detect_fvg(candles_bear)
    obs_fvg_bear = detect_order_blocks(candles_bear, fvgs=fvgs_bear)
    assert len(obs_fvg_bear) == 1
    assert obs_fvg_bear[0].candle == c_bull


def test_repro_bug_t06_06_r2_criteria_and_edge_cases() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2648.0, close=2649.0, base_time=t0)
    candles = [c0]
    for idx in range(1, 25):
        candles.append(
            make_candle(idx, open=2650.0, high=2651.0, low=2649.5, close=2650.5, base_time=t0)
        )
    c_break = make_candle(25, open=2650.5, high=2665.0, low=2650.0, close=2664.0, base_time=t0)
    candles.append(c_break)

    swing_high = Swing(
        time=candles[1].time,
        price=2651.0,
        type=SwingType.HIGH,
        confirmed_time=candles[5].time + timedelta(minutes=5),
        index=1,
    )
    bos = BOS(
        time=c_break.time,
        confirmed_time=c_break.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        breaking_candle=c_break,
        broken_swing=swing_high,
        price=swing_high.price,
    )

    obs_lookback_default = detect_order_blocks(candles, structure_events=[bos], max_lookback=20)
    assert len(obs_lookback_default) == 0

    obs_lookback_boundary_out = detect_order_blocks(
        candles, structure_events=[bos], max_lookback=24
    )
    assert len(obs_lookback_boundary_out) == 0

    obs_lookback_boundary_in = detect_order_blocks(candles, structure_events=[bos], max_lookback=25)
    assert len(obs_lookback_boundary_in) == 1
    assert obs_lookback_boundary_in[0].candle == c0

    obs_lookback_large = detect_order_blocks(candles, structure_events=[bos], max_lookback=30)
    assert len(obs_lookback_large) == 1
    assert obs_lookback_large[0].candle == c0

    with pytest.raises(ValueError, match="max_lookback must be positive"):
        detect_order_blocks(candles, max_lookback=0)

    atr_series_ob: list[float | None] = [None] * len(candles)
    atr_series_ob[25] = 2.0
    obs_atr = detect_order_blocks(
        candles,
        structure_events=[bos],
        atr=atr_series_ob,
        min_displacement_multiplier=1.0,
        max_lookback=30,
    )
    assert len(obs_atr) == 1

    atr_series_ob_high: list[float | None] = [None] * len(candles)
    atr_series_ob_high[25] = 20.0
    obs_atr_high = detect_order_blocks(
        candles,
        structure_events=[bos],
        atr=atr_series_ob_high,
        min_displacement_multiplier=1.0,
        max_lookback=30,
    )
    assert len(obs_atr_high) == 0

    atr_series_none: list[float | None] = [None] * len(candles)
    obs_none = detect_order_blocks(
        candles,
        structure_events=[bos],
        atr=atr_series_none,
        min_displacement_multiplier=1.0,
        max_lookback=30,
    )
    assert len(obs_none) == 0

    with pytest.raises(ValueError, match="atr sequence length"):
        detect_order_blocks(candles, atr=[1.0])

    with pytest.raises(ValueError, match="atr sequence length"):
        detect_fvg(candles, atr=[1.0])

    fvg_candles = [
        make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2650.0),
        make_candle(1, open=2650.0, high=2660.0, low=2650.0, close=2659.0),
        make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0),
    ]
    assert len(detect_fvg(fvg_candles)) == 1
    assert detect_fvg(fvg_candles, atr=[None, None, None], atr_multiplier=1.0) == []


def test_repro_bug_t06_07_nested_fvg_in_order_block_causal_leakage() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2652.0, high=2652.0, low=2648.0, close=2649.0, base_time=t0)
    c1 = make_candle(1, open=2649.0, high=2655.0, low=2649.0, close=2654.0, base_time=t0)
    c2 = make_candle(2, open=2654.0, high=2662.0, low=2654.0, close=2660.0, base_time=t0)
    c3 = make_candle(3, open=2660.0, high=2662.0, low=2655.0, close=2661.0, base_time=t0)
    c4 = make_candle(4, open=2661.0, high=2662.0, low=2653.0, close=2656.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4]

    duration = timedelta(seconds=Timeframe.M5.duration_seconds)
    full_fvgs = detect_fvg(candles)
    full_obs = detect_order_blocks(candles, fvgs=full_fvgs)
    assert len(full_obs) == 1
    assert full_obs[0].fvg is not None
    assert full_obs[0].fvg.mitigated is True

    for k in range(2, len(candles)):
        prefix = candles[: k + 1]
        as_of_time = prefix[-1].time + duration
        prefix_fvgs = detect_fvg(prefix)
        prefix_obs = detect_order_blocks(prefix, fvgs=prefix_fvgs)
        act_prefix = get_active_order_blocks(
            prefix_obs, as_of_time=as_of_time, include_mitigated=True
        )
        act_full = get_active_order_blocks(full_obs, as_of_time=as_of_time, include_mitigated=True)
        assert act_prefix == act_full


def test_poi_parameter_validations() -> None:
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0)
    candles = [c0, c1, c2]
    fvg = detect_fvg(candles)[0]
    assert detect_order_blocks([c0, c1], fvgs=[fvg]) == []
    assert _resolve_atr([1.0, 2.0], 5) is None
    assert _resolve_atr([1.0, 2.0], -1) is None

    with pytest.raises(ValueError, match="min_size must be non-negative"):
        detect_fvg(candles, min_size=-1.0)

    with pytest.raises(ValueError, match="atr_multiplier must be non-negative"):
        detect_fvg(candles, atr=2.0, atr_multiplier=-0.5)

    with pytest.raises(ValueError, match="atr must be positive"):
        detect_fvg(candles, atr=0.0, atr_multiplier=1.0)

    with pytest.raises(ValueError, match="atr must be positive"):
        detect_fvg(candles, atr=-2.0, atr_multiplier=1.0)

    with pytest.raises(TypeError, match="Unsupported atr type"):
        detect_fvg(candles, atr=cast(Any, {"invalid": "type"}))

    with pytest.raises(ValueError, match="min_displacement_multiplier must be non-negative"):
        detect_order_blocks(candles, min_displacement_multiplier=-1.0)

    with pytest.raises(ValueError, match="max_lookback must be positive"):
        detect_order_blocks(candles, max_lookback=-5)

    with pytest.raises(ValueError, match="atr must be positive"):
        detect_order_blocks(candles, atr=-1.0)
