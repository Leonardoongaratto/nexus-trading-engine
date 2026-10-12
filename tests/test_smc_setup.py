from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.domain.models import (
    BOS,
    FVG,
    Candle,
    ChoCH,
    Direction,
    FVGType,
    LiquidityPool,
    LiquiditySweep,
    LiquidityType,
    OrderBlock,
    OrderBlockType,
    SetupStatus,
    SetupTriggerType,
    Swing,
    SwingType,
    Timeframe,
    TradeSetup,
)
from xau_bot.smc.setup import (
    detect_setups,
    determine_htf_bias,
    evaluate_setup_status,
    get_active_setups,
)
from xau_bot.smc.structure import detect_structure
from xau_bot.smc.swings import find_swings


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


def test_detect_setups_empty_or_insufficient_candles() -> None:
    assert detect_setups([]) == []
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2655.0, low=2650.0, close=2654.0)
    assert detect_setups([c0, c1]) == []


def test_detect_setups_bullish_complete_sequence() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2662.0, low=2650.0, close=2652.0, base_time=t0)
    c1 = make_candle(1, open=2652.0, high=2658.0, low=2651.0, close=2656.0, base_time=t0)
    c2 = make_candle(2, open=2656.0, high=2657.0, low=2648.0, close=2649.0, base_time=t0)
    c3 = make_candle(3, open=2649.0, high=2652.0, low=2647.0, close=2651.0, base_time=t0)
    c4 = make_candle(4, open=2651.0, high=2665.0, low=2650.0, close=2663.0, base_time=t0)
    c5 = make_candle(5, open=2663.0, high=2675.0, low=2660.0, close=2674.0, base_time=t0)
    c6 = make_candle(6, open=2674.0, high=2680.0, low=2670.0, close=2678.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5, c6]

    s_high_0 = Swing(
        time=c1.time,
        price=2658.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2650.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2650.0,
        swings=(s_low_0,),
        time=c0.time,
        confirmed_time=c1.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2650.0,
        candle=c3,
        pool=pool_ssl,
    )
    choch_bull = ChoCH(
        time=c4.time,
        confirmed_time=c4.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c4,
        price=2658.0,
    )
    fvg_bull = FVG(
        time=c4.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2660.0,
        bottom=2652.0,
        candle_0=c3,
        candle_1=c4,
        candle_2=c5,
    )

    target_high = Swing(
        time=c6.time,
        price=2685.0,
        type=SwingType.HIGH,
        confirmed_time=c6.time + timedelta(minutes=5),
        index=6,
    )

    setups = detect_setups(
        candles,
        swings=[s_low_0, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        fvgs=[fvg_bull],
        trigger_type=SetupTriggerType.FVG_CE,
        min_rr=2.0,
    )

    assert len(setups) == 1
    s = setups[0]
    assert s.direction == Direction.BULLISH
    assert s.trigger_type == SetupTriggerType.FVG_CE
    assert s.entry_price == pytest.approx(fvg_bull.consequent_encroachment)
    assert s.stop_loss < s.entry_price
    assert s.stop_loss <= sweep_ssl.candle.low
    assert s.take_profit > s.entry_price
    assert s.risk_reward_ratio >= 2.0
    assert s.sweep == sweep_ssl
    assert s.choch == choch_bull
    assert s.poi == fvg_bull
    assert "SWEEP_CONFIRMED" in s.confluences
    assert "CHOCH_CONFIRMED" in s.confluences
    assert "FVG_PRESENT" in s.confluences


def test_detect_setups_bearish_complete_sequence_with_order_block() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2653.0, base_time=t0)
    c1 = make_candle(1, open=2653.0, high=2665.0, low=2652.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2664.0, high=2667.0, low=2660.0, close=2662.0, base_time=t0)
    c3 = make_candle(3, open=2662.0, high=2663.0, low=2640.0, close=2642.0, base_time=t0)
    candles = [c0, c1, c2, c3]

    s_high_1 = Swing(
        time=c1.time,
        price=2665.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_bsl = LiquidityPool(
        type=LiquidityType.BSL,
        price=2665.0,
        swings=(s_high_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_bsl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2665.0,
        candle=c2,
        pool=pool_bsl,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    choch_bear = ChoCH(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=s_low_0,
        breaking_candle=c3,
        price=2645.0,
    )
    ob_bear = OrderBlock(
        time=c2.time,
        confirmed_time=choch_bear.confirmed_time,
        type=OrderBlockType.BEARISH,
        high=c2.high,
        low=c2.low,
        open=c2.open,
        close=c2.close,
        candle=c2,
        displacement_atr_ratio=2.0,
        breaking_event=choch_bear,
    )

    target_low = Swing(
        time=c0.time,
        price=2620.0,
        type=SwingType.LOW,
        confirmed_time=c0.time + timedelta(minutes=5),
        index=0,
    )

    setups = detect_setups(
        candles,
        swings=[s_high_1, s_low_0, target_low],
        structure_events=[choch_bear],
        sweeps=[sweep_bsl],
        order_blocks=[ob_bear],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        min_rr=2.0,
    )

    assert len(setups) == 1
    s = setups[0]
    assert s.direction == Direction.BEARISH
    assert s.trigger_type == SetupTriggerType.OB_MEAN_THRESHOLD
    assert s.entry_price == pytest.approx(ob_bear.mean_threshold)
    assert s.stop_loss > s.entry_price
    assert s.stop_loss >= sweep_bsl.candle.high
    assert s.take_profit < s.entry_price
    assert s.risk_reward_ratio >= 2.0
    assert s.poi == ob_bear


def test_detect_setups_rejects_choch_without_prior_sweep() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2649.0, close=2652.0, base_time=t0)
    c1 = make_candle(1, open=2652.0, high=2665.0, low=2651.0, close=2664.0, base_time=t0)
    candles = [c0, c1]

    s_high_0 = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    choch_bull = ChoCH(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c1,
        price=2655.0,
    )

    assert (
        detect_setups(
            candles,
            structure_events=[choch_bull],
            sweeps=[],
            require_sweep=True,
        )
        == []
    )


def test_detect_setups_rejects_sweep_occurring_after_choch() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2649.0, close=2652.0, base_time=t0)
    c1 = make_candle(1, open=2652.0, high=2665.0, low=2651.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2664.0, high=2666.0, low=2640.0, close=2648.0, base_time=t0)
    candles = [c0, c1, c2]

    s_high_0 = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    choch_bull = ChoCH(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c1,
        price=2655.0,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low_0,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep_tardio = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c2,
        pool=pool_ssl,
    )

    assert (
        detect_setups(
            candles,
            structure_events=[choch_bull],
            sweeps=[sweep_tardio],
            require_sweep=True,
        )
        == []
    )


def test_detect_setups_rejects_when_poi_required_but_missing() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2648.0, base_time=t0)
    c1 = make_candle(1, open=2648.0, high=2652.0, low=2640.0, close=2649.0, base_time=t0)
    c2 = make_candle(2, open=2649.0, high=2662.0, low=2648.0, close=2661.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2655.0,
    )

    assert (
        detect_setups(
            candles,
            structure_events=[choch],
            sweeps=[sweep],
            fvgs=[],
            order_blocks=[],
            require_poi=True,
        )
        == []
    )


def test_detect_setups_risk_reward_filtering() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2646.0, high=2648.0, low=2644.0, close=2647.0, base_time=t0)
    c1 = make_candle(1, open=2647.0, high=2654.0, low=2638.0, close=2652.0, base_time=t0)
    c2 = make_candle(2, open=2652.0, high=2662.0, low=2650.0, close=2661.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2640.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2640.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2640.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2648.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2648.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2650.0,
        bottom=2648.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target_low_rr = Swing(
        time=c2.time,
        price=2651.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    assert (
        detect_setups(
            candles,
            swings=[target_low_rr],
            structure_events=[choch],
            sweeps=[sweep],
            fvgs=[fvg],
            min_rr=3.0,
        )
        == []
    )


def test_detect_setups_trigger_types() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    ob = OrderBlock(
        time=c0.time,
        confirmed_time=choch.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=2.0,
        breaking_event=choch,
    )

    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    s_proximal = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        order_blocks=[ob],
        trigger_type=SetupTriggerType.FVG_PROXIMAL,
    )[0]
    assert s_proximal.entry_price == fvg.top

    s_ob_ext = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        order_blocks=[ob],
        trigger_type=SetupTriggerType.OB_EXTREME,
    )[0]
    assert s_ob_ext.entry_price == ob.high

    s_ote = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        order_blocks=[ob],
        trigger_type=SetupTriggerType.OTE_RETRACEMENT,
    )[0]
    assert s_ote.entry_price > s_ote.stop_loss
    assert s_ote.entry_price < c2.high


def test_detect_setups_htf_alignment() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    htf_bull_0 = make_candle(
        0,
        open=2630.0,
        high=2650.0,
        low=2625.0,
        close=2648.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )
    htf_bull_1 = make_candle(
        1,
        open=2648.0,
        high=2670.0,
        low=2645.0,
        close=2668.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )
    htf_candles_bull = [htf_bull_0, htf_bull_1]

    s_aligned = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles_bull,
        htf_alignment_required=True,
    )
    assert len(s_aligned) == 1
    assert "HTF_ALIGNED" in s_aligned[0].confluences

    htf_bear_0 = make_candle(
        0,
        open=2680.0,
        high=2682.0,
        low=2660.0,
        close=2662.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )
    htf_bear_1 = make_candle(
        1,
        open=2662.0,
        high=2665.0,
        low=2640.0,
        close=2645.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
    )
    htf_candles_bear = [htf_bear_0, htf_bear_1]

    s_opposed = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles_bear,
        htf_alignment_required=True,
    )
    assert len(s_opposed) == 0


def test_evaluate_setup_status_lifecycle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    setup = TradeSetup(
        id="s1",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BULLISH,
        entry_price=2653.0,
        stop_loss=2643.0,
        take_profit=2673.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        valid_until=choch.confirmed_time + timedelta(minutes=30),
    )

    c_trigger = make_candle(3, open=2664.0, high=2665.0, low=2652.0, close=2658.0, base_time=t0)
    st_trig = evaluate_setup_status(setup, [c_trigger])
    assert st_trig.status == SetupStatus.TRIGGERED
    assert st_trig.status_updated_at == c_trigger.time

    c_inval = make_candle(3, open=2642.0, high=2642.5, low=2640.0, close=2641.0, base_time=t0)
    st_inv = evaluate_setup_status(setup, [c_inval])
    assert st_inv.status == SetupStatus.INVALIDATED
    assert st_inv.status_updated_at == c_inval.time

    c_expired = make_candle(10, open=2664.0, high=2666.0, low=2660.0, close=2662.0, base_time=t0)
    st_exp = evaluate_setup_status(setup, [c_expired])
    assert st_exp.status == SetupStatus.EXPIRED


def test_get_active_setups_filtering() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    s1 = TradeSetup(
        id="s1",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BULLISH,
        entry_price=2653.0,
        stop_loss=2643.0,
        take_profit=2673.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
    )
    s2 = TradeSetup(
        id="s2",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BULLISH,
        entry_price=2653.0,
        stop_loss=2643.0,
        take_profit=2673.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        status=SetupStatus.TRIGGERED,
        status_updated_at=choch.confirmed_time + timedelta(minutes=10),
    )

    assert len(get_active_setups([s1, s2], include_triggered=False)) == 1
    assert len(get_active_setups([s1, s2], include_triggered=True)) == 2
    assert len(get_active_setups([s1], as_of_time=choch.confirmed_time - timedelta(minutes=1))) == 0


def test_determine_htf_bias() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2630.0, high=2650.0, low=2620.0, close=2645.0, base_time=t0)
    c1 = make_candle(1, open=2645.0, high=2670.0, low=2640.0, close=2665.0, base_time=t0)
    c2 = make_candle(2, open=2665.0, high=2685.0, low=2660.0, close=2680.0, base_time=t0)
    assert determine_htf_bias([c0, c1, c2]) == Direction.BULLISH

    c0_d = make_candle(0, open=2680.0, high=2685.0, low=2660.0, close=2665.0, base_time=t0)
    c1_d = make_candle(1, open=2665.0, high=2670.0, low=2640.0, close=2645.0, base_time=t0)
    c2_d = make_candle(2, open=2645.0, high=2650.0, low=2620.0, close=2625.0, base_time=t0)
    assert determine_htf_bias([c0_d, c1_d, c2_d]) == Direction.BEARISH

    assert determine_htf_bias([]) is None


def test_setups_anti_lookahead_causal_equivalence() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candles = [
        make_candle(0, open=2660.0, high=2662.0, low=2650.0, close=2652.0, base_time=t0),
        make_candle(1, open=2652.0, high=2658.0, low=2651.0, close=2656.0, base_time=t0),
        make_candle(2, open=2656.0, high=2657.0, low=2648.0, close=2649.0, base_time=t0),
        make_candle(3, open=2649.0, high=2652.0, low=2647.0, close=2651.0, base_time=t0),
        make_candle(4, open=2651.0, high=2665.0, low=2650.0, close=2663.0, base_time=t0),
        make_candle(5, open=2663.0, high=2675.0, low=2660.0, close=2674.0, base_time=t0),
        make_candle(6, open=2674.0, high=2680.0, low=2670.0, close=2678.0, base_time=t0),
    ]
    c0, c1, c2, c3, c4, c5, c6 = candles
    s_high_0 = Swing(
        time=c1.time,
        price=2658.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2650.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    target_high_prior = Swing(
        time=c0.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2650.0,
        swings=(s_low_0,),
        time=c0.time,
        confirmed_time=c1.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2650.0,
        candle=c3,
        pool=pool_ssl,
    )
    choch_bull = ChoCH(
        time=c4.time,
        confirmed_time=c4.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c4,
        price=2658.0,
    )
    fvg_bull = FVG(
        time=c4.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2660.0,
        bottom=2652.0,
        candle_0=c3,
        candle_1=c4,
        candle_2=c5,
    )
    swings = [s_low_0, s_high_0, target_high_prior]
    events = [choch_bull]
    sweeps = [sweep_ssl]
    fvgs = [fvg_bull]

    duration = timedelta(seconds=Timeframe.M5.duration_seconds)
    full_setups = detect_setups(
        candles,
        swings=swings,
        structure_events=events,
        sweeps=sweeps,
        fvgs=fvgs,
        order_blocks=[],
        trigger_type=SetupTriggerType.FVG_CE,
    )
    assert len(full_setups) >= 1
    s_full = full_setups[0]

    for k in range(len(candles)):
        prefix = candles[: k + 1]
        as_of_time = prefix[-1].time + duration
        prefix_setups = detect_setups(
            prefix,
            swings=swings,
            structure_events=events,
            sweeps=sweeps,
            fvgs=fvgs,
            order_blocks=[],
            trigger_type=SetupTriggerType.FVG_CE,
            as_of_time=as_of_time,
        )
        act_prefix = get_active_setups(prefix_setups, as_of_time=as_of_time)
        act_full = get_active_setups(full_setups, as_of_time=as_of_time)
        assert act_prefix == act_full
        if act_prefix:
            assert act_prefix[0].entry_price == s_full.entry_price
            assert act_prefix[0].stop_loss == s_full.stop_loss
            assert act_prefix[0].take_profit == s_full.take_profit
            assert act_prefix[0].risk_reward_ratio == s_full.risk_reward_ratio


def test_setup_parameter_validations() -> None:
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2655.0, low=2650.0, close=2654.0)
    c2 = make_candle(2, open=2654.0, high=2660.0, low=2653.0, close=2659.0)
    candles = [c0, c1, c2]

    with pytest.raises(ValueError, match="min_rr must be positive"):
        detect_setups(candles, min_rr=0.0)

    with pytest.raises(ValueError, match="validity_bars must be positive"):
        detect_setups(candles, validity_bars=-1)

    with pytest.raises(ValueError, match="max_lookback_bars must be positive"):
        detect_setups(candles, max_lookback_bars=0)

    with pytest.raises(ValueError, match="min_ob_displacement_atr must be non-negative"):
        detect_setups(candles, min_ob_displacement_atr=-0.1)

    with pytest.raises(ValueError, match="min_body_range_ratio must be non-negative"):
        detect_setups(candles, min_body_range_ratio=-0.1)

    with pytest.raises(ValueError, match="ote_level must be between 0.62 and 0.79"):
        detect_setups(candles, ote_level=0.5)

    with pytest.raises(ValueError, match="ote_level must be between 0.62 and 0.79"):
        detect_setups(candles, ote_level=0.85)

    with pytest.raises(ValueError, match="htf_lookback_bars must be positive"):
        detect_setups(candles, htf_lookback_bars=0)

    with pytest.raises(ValueError, match="lookback_bars must be positive"):
        determine_htf_bias(candles, lookback_bars=-1)

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        detect_setups(candles, as_of_time=datetime(2026, 9, 24, 12, 0))

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        get_active_setups([], as_of_time=datetime(2026, 9, 24, 12, 0))

    with pytest.raises(ValueError, match="as_of_time must be timezone-aware"):
        evaluate_setup_status(
            TradeSetup(
                id="s_err",
                time=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
                confirmed_time=datetime(2026, 9, 24, 12, 5, tzinfo=UTC),
                direction=Direction.BULLISH,
                entry_price=2650.0,
                stop_loss=2645.0,
                take_profit=2660.0,
                risk_reward_ratio=2.0,
                timeframe=Timeframe.M5,
                trigger_type=SetupTriggerType.FVG_CE,
            ),
            [],
            as_of_time=datetime(2026, 9, 24, 12, 0),
        )


def test_determine_htf_bias_single_candle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c_bull = make_candle(0, open=2650.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0)
    assert determine_htf_bias([c_bull]) == Direction.BULLISH

    c_bear = make_candle(0, open=2654.0, high=2655.0, low=2648.0, close=2649.0, base_time=t0)
    assert determine_htf_bias([c_bear]) == Direction.BEARISH


def test_evaluate_setup_status_bearish_lifecycle() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2665.0, low=2658.0, close=2664.0, base_time=t0)
    c1 = make_candle(1, open=2664.0, high=2670.0, low=2655.0, close=2657.0, base_time=t0)
    c2 = make_candle(2, open=2657.0, high=2658.0, low=2648.0, close=2650.0, base_time=t0)

    s_high = Swing(
        time=c0.time,
        price=2668.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.BSL,
        price=2668.0,
        swings=(s_high,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2668.0,
        candle=c1,
        pool=pool,
    )
    s_low_broken = Swing(
        time=c0.time,
        price=2658.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=s_low_broken,
        breaking_candle=c2,
        price=2658.0,
    )
    setup = TradeSetup(
        id="s_bear",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BEARISH,
        entry_price=2660.0,
        stop_loss=2670.0,
        take_profit=2640.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        sweep=sweep,
        choch=choch,
        valid_until=choch.confirmed_time + timedelta(minutes=30),
    )

    c_before = make_candle(0, open=2660.0, high=2661.0, low=2659.0, close=2660.0, base_time=t0)
    st_unchanged = evaluate_setup_status(setup, [c_before])
    assert st_unchanged.status == SetupStatus.PENDING

    c_trigger = make_candle(3, open=2650.0, high=2661.0, low=2649.0, close=2658.0, base_time=t0)
    st_trig = evaluate_setup_status(setup, [c_trigger])
    assert st_trig.status == SetupStatus.TRIGGERED
    assert st_trig.status_updated_at == c_trigger.time

    c_inval = make_candle(3, open=2672.0, high=2675.0, low=2671.0, close=2674.0, base_time=t0)
    st_inv = evaluate_setup_status(setup, [c_inval])
    assert st_inv.status == SetupStatus.INVALIDATED
    assert st_inv.status_updated_at == c_inval.time

    c_expired = make_candle(10, open=2650.0, high=2655.0, low=2649.0, close=2651.0, base_time=t0)
    st_exp = evaluate_setup_status(setup, [c_expired])
    assert st_exp.status == SetupStatus.EXPIRED

    c_filtered = make_candle(4, open=2650.0, high=2662.0, low=2649.0, close=2658.0, base_time=t0)
    st_as_of = evaluate_setup_status(
        setup, [c_filtered], as_of_time=c_filtered.time - timedelta(minutes=1)
    )
    assert st_as_of.status == SetupStatus.PENDING


def test_detect_setups_fallback_triggers_and_bearish_ote() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    ob = OrderBlock(
        time=c0.time,
        confirmed_time=choch.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=2.0,
        breaking_event=choch,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    s_fvg_ce_fallback = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[],
        order_blocks=[ob],
        trigger_type=SetupTriggerType.FVG_CE,
    )[0]
    assert s_fvg_ce_fallback.entry_price == ob.mean_threshold

    s_fvg_prox_fallback = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[],
        order_blocks=[ob],
        trigger_type=SetupTriggerType.FVG_PROXIMAL,
    )[0]
    assert s_fvg_prox_fallback.entry_price == ob.high

    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )

    s_ob_ext_fallback = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        order_blocks=[],
        trigger_type=SetupTriggerType.OB_EXTREME,
    )[0]
    assert s_ob_ext_fallback.entry_price == fvg.top

    s_ob_mt_fallback = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        order_blocks=[],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
    )[0]
    assert s_ob_mt_fallback.entry_price == fvg.consequent_encroachment

    bos = BOS(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    s_bos = detect_setups(
        candles,
        swings=[target],
        structure_events=[bos],
        sweeps=[sweep],
        fvgs=[fvg],
        require_choch=False,
    )[0]
    assert "BOS_CONFIRMED" in s_bos.confluences


def test_repro_bug_t07_01_take_profit_lookahead() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2662.0, low=2650.0, close=2652.0, base_time=t0)
    c1 = make_candle(1, open=2652.0, high=2658.0, low=2651.0, close=2656.0, base_time=t0)
    c2 = make_candle(2, open=2656.0, high=2657.0, low=2648.0, close=2649.0, base_time=t0)
    c3 = make_candle(3, open=2649.0, high=2652.0, low=2647.0, close=2651.0, base_time=t0)
    c4 = make_candle(4, open=2651.0, high=2665.0, low=2650.0, close=2663.0, base_time=t0)
    c5 = make_candle(5, open=2663.0, high=2675.0, low=2660.0, close=2674.0, base_time=t0)
    c6 = make_candle(6, open=2674.0, high=2680.0, low=2670.0, close=2678.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5, c6]

    s_high_0 = Swing(
        time=c1.time,
        price=2658.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2650.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2650.0,
        swings=(s_low_0,),
        time=c0.time,
        confirmed_time=c1.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2650.0,
        candle=c3,
        pool=pool_ssl,
    )
    choch_bull = ChoCH(
        time=c4.time,
        confirmed_time=c4.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c4,
        price=2658.0,
    )
    fvg_bull = FVG(
        time=c4.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2660.0,
        bottom=2652.0,
        candle_0=c3,
        candle_1=c4,
        candle_2=c5,
    )

    future_target_high = Swing(
        time=c6.time,
        price=2685.0,
        type=SwingType.HIGH,
        confirmed_time=c6.time + timedelta(minutes=5),
        index=6,
    )

    setups_full = detect_setups(
        candles,
        swings=[s_low_0, s_high_0, future_target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        fvgs=[fvg_bull],
        trigger_type=SetupTriggerType.FVG_CE,
        min_rr=2.0,
    )
    assert len(setups_full) == 1
    s_full = setups_full[0]

    setups_causal = detect_setups(
        candles,
        swings=[s_low_0, s_high_0, future_target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        fvgs=[fvg_bull],
        trigger_type=SetupTriggerType.FVG_CE,
        min_rr=2.0,
        as_of_time=s_full.confirmed_time,
    )
    assert len(setups_causal) == 1
    assert s_full.take_profit == setups_causal[0].take_profit
    assert s_full.take_profit != future_target_high.price


def test_repro_bug_t07_02_htf_bias_lookahead() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    htf_past_bullish = make_candle(
        0,
        open=2630.0,
        high=2650.0,
        low=2625.0,
        close=2648.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 11, 0, tzinfo=UTC),
    )
    htf_future_bearish = make_candle(
        0,
        open=2680.0,
        high=2682.0,
        low=2615.0,
        close=2620.0,
        timeframe=Timeframe.H1,
        base_time=datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
    )
    htf_candles = [htf_past_bullish, htf_future_bearish]

    setups = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles,
        htf_alignment_required=True,
        as_of_time=choch.confirmed_time,
    )
    assert len(setups) == 1
    assert setups[0].direction == Direction.BULLISH


def test_repro_bug_t07_04_candle_touching_entry_and_stop_triggers_first() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    setup = TradeSetup(
        id="s_bull",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BULLISH,
        entry_price=2656.0,
        stop_loss=2646.5,
        take_profit=2675.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        valid_until=choch.confirmed_time + timedelta(minutes=30),
    )

    c_cross = make_candle(3, open=2660.0, high=2661.0, low=2645.0, close=2650.0, base_time=t0)
    res = evaluate_setup_status(setup, [c_cross])
    assert res.status == SetupStatus.TRIGGERED


def test_repro_bug_t07_05_candle_in_formation_ignored_by_as_of_time() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    setup = TradeSetup(
        id="s_bull",
        time=choch.time,
        confirmed_time=choch.confirmed_time,
        direction=Direction.BULLISH,
        entry_price=2656.0,
        stop_loss=2646.5,
        take_profit=2675.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        valid_until=choch.confirmed_time + timedelta(minutes=30),
    )

    c_in_formation = make_candle(
        3, open=2660.0, high=2661.0, low=2652.0, close=2658.0, base_time=t0
    )
    as_of = c_in_formation.time + timedelta(minutes=2)
    res = evaluate_setup_status(setup, [c_in_formation], as_of_time=as_of)
    assert res.status == SetupStatus.PENDING


def test_repro_bug_t07_06_mitigated_poi_rejected() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2652.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg_mitigated = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
        mitigated=True,
        mitigated_at=c2.time + timedelta(minutes=5),
        mitigated_by_candle=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    setups = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg_mitigated],
        require_poi=True,
    )
    assert len(setups) == 0


def test_repro_bug_t07_07_absence_of_displacement_rejects_setup() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2651.0, low=2648.0, close=2649.0, base_time=t0)
    c1 = make_candle(1, open=2649.0, high=2650.0, low=2644.0, close=2649.0, base_time=t0)
    c2 = make_candle(2, open=2649.0, high=2652.0, low=2648.5, close=2651.5, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(
        time=c0.time,
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    pool = LiquidityPool(
        type=LiquidityType.SSL,
        price=2645.0,
        swings=(s_low,),
        time=c0.time,
        confirmed_time=c1.time,
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time,
        price=2651.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2651.0,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    setups = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        require_poi=False,
        require_displacement=True,
    )
    assert len(setups) == 0


def test_repro_bug_t07_08_gap_into_zone_triggers_order() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    s_bull = TradeSetup(
        id="s_bull",
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        direction=Direction.BULLISH,
        entry_price=2656.0,
        stop_loss=2646.5,
        take_profit=2675.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
    )
    c_gap_bull = make_candle(2, open=2650.0, high=2652.0, low=2646.0, close=2648.0, base_time=t0)
    res_bull = evaluate_setup_status(s_bull, [c_gap_bull])
    assert res_bull.status == SetupStatus.TRIGGERED

    s_bear = TradeSetup(
        id="s_bear",
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        direction=Direction.BEARISH,
        entry_price=2660.0,
        stop_loss=2670.0,
        take_profit=2640.0,
        risk_reward_ratio=2.0,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
    )
    c_gap_bear = make_candle(2, open=2665.0, high=2671.0, low=2663.0, close=2668.0, base_time=t0)
    res_bear = evaluate_setup_status(s_bear, [c_gap_bear])
    assert res_bear.status == SetupStatus.TRIGGERED


def test_repro_bug_t07_09_htf_bias_invariant_to_older_history() -> None:
    t0 = datetime(2026, 9, 24, 6, 0, tzinfo=UTC)
    c_old_bull = make_candle(
        0, open=2550.0, high=2650.0, low=2545.0, close=2645.0, timeframe=Timeframe.H1, base_time=t0
    )
    c_bear_1 = make_candle(
        1, open=2645.0, high=2646.0, low=2630.0, close=2632.0, timeframe=Timeframe.H1, base_time=t0
    )
    c_bear_2 = make_candle(
        2, open=2632.0, high=2633.0, low=2620.0, close=2622.0, timeframe=Timeframe.H1, base_time=t0
    )
    c_bear_3 = make_candle(
        3, open=2622.0, high=2623.0, low=2610.0, close=2612.0, timeframe=Timeframe.H1, base_time=t0
    )
    c_bear_4 = make_candle(
        4, open=2612.0, high=2613.0, low=2600.0, close=2602.0, timeframe=Timeframe.H1, base_time=t0
    )
    c_bear_5 = make_candle(
        5, open=2602.0, high=2603.0, low=2585.0, close=2588.0, timeframe=Timeframe.H1, base_time=t0
    )
    htf_series_recent = [c_bear_1, c_bear_2, c_bear_3, c_bear_4, c_bear_5]
    htf_series_with_old = [c_old_bull, c_bear_1, c_bear_2, c_bear_3, c_bear_4, c_bear_5]
    bias_recent = determine_htf_bias(htf_series_recent)
    bias_with_old = determine_htf_bias(htf_series_with_old)
    assert bias_recent == Direction.BEARISH
    assert bias_with_old == Direction.BEARISH


def test_repro_bug_t07_10_ob_mitigated_poi_rejected() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    ob_mitigated = OrderBlock(
        time=c0.time,
        confirmed_time=choch.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=2.0,
        breaking_event=choch,
        mitigated=True,
        mitigated_at=choch.confirmed_time,
        mitigated_by_candle=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    setups = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[],
        order_blocks=[ob_mitigated],
        require_poi=True,
    )
    assert len(setups) == 0


def test_detect_setups_sweep_filters_and_synthetic_tp() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )

    sweep_future_conf = LiquiditySweep(
        time=c1.time,
        confirmed_time=choch.confirmed_time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    assert (
        detect_setups(
            candles,
            structure_events=[choch],
            sweeps=[sweep_future_conf],
            fvgs=[fvg],
            require_sweep=True,
        )
        == []
    )

    t_start = t0 - timedelta(minutes=55)
    c_old = make_candle(0, open=2650.0, high=2652.0, low=2643.0, close=2646.0, base_time=t_start)
    intermediate = [
        make_candle(
            i + 1,
            open=2646.0,
            high=2650.0,
            low=2645.0,
            close=2648.0,
            base_time=t_start,
        )
        for i in range(8)
    ]
    candles_lookback = [c_old, *intermediate, c0, c1, c2]
    assert all(
        candles_lookback[k].time < candles_lookback[k + 1].time
        for k in range(len(candles_lookback) - 1)
    )
    sweep_outside_lookback = LiquiditySweep(
        time=c_old.time,
        confirmed_time=c_old.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c_old,
        pool=pool,
    )
    assert (
        detect_setups(
            candles_lookback,
            structure_events=[choch],
            sweeps=[sweep_outside_lookback],
            fvgs=[fvg],
            max_lookback_bars=5,
            require_sweep=True,
        )
        == []
    )
    assert (
        len(
            detect_setups(
                candles_lookback,
                structure_events=[choch],
                sweeps=[sweep_outside_lookback],
                fvgs=[fvg],
                max_lookback_bars=20,
                require_sweep=True,
            )
        )
        == 1
    )

    sweep_valid = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    assert (
        detect_setups(
            candles,
            swings=[],
            structure_events=[choch],
            sweeps=[sweep_valid],
            fvgs=[fvg],
            allow_synthetic_tp=False,
        )
        == []
    )


def test_repro_finding_1_htf_structural_bias_disagrees_with_window_and_executes() -> None:
    t0 = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)
    tf_h1 = Timeframe.H1
    c0 = make_candle(
        0, open=2610.0, high=2620.0, low=2608.0, close=2618.0, base_time=t0, timeframe=tf_h1
    )
    c1 = make_candle(
        1, open=2618.0, high=2635.0, low=2615.0, close=2630.0, base_time=t0, timeframe=tf_h1
    )
    c2 = make_candle(
        2, open=2630.0, high=2650.0, low=2625.0, close=2645.0, base_time=t0, timeframe=tf_h1
    )
    c3 = make_candle(
        3, open=2645.0, high=2670.0, low=2640.0, close=2665.0, base_time=t0, timeframe=tf_h1
    )
    c4 = make_candle(
        4, open=2658.0, high=2660.0, low=2635.0, close=2640.0, base_time=t0, timeframe=tf_h1
    )
    c5 = make_candle(
        5, open=2640.0, high=2650.0, low=2620.0, close=2625.0, base_time=t0, timeframe=tf_h1
    )
    c6 = make_candle(
        6, open=2625.0, high=2630.0, low=2615.0, close=2620.0, base_time=t0, timeframe=tf_h1
    )
    c7 = make_candle(
        7, open=2620.0, high=2640.0, low=2620.0, close=2635.0, base_time=t0, timeframe=tf_h1
    )
    c8 = make_candle(
        8, open=2635.0, high=2655.0, low=2630.0, close=2650.0, base_time=t0, timeframe=tf_h1
    )
    c9 = make_candle(
        9, open=2650.0, high=2678.0, low=2648.0, close=2675.0, base_time=t0, timeframe=tf_h1
    )
    c10 = make_candle(
        10, open=2675.0, high=2676.0, low=2655.0, close=2658.0, base_time=t0, timeframe=tf_h1
    )
    c11 = make_candle(
        11, open=2658.0, high=2660.0, low=2640.0, close=2642.0, base_time=t0, timeframe=tf_h1
    )
    c12 = make_candle(
        12, open=2642.0, high=2645.0, low=2630.0, close=2632.0, base_time=t0, timeframe=tf_h1
    )
    c13 = make_candle(
        13, open=2632.0, high=2635.0, low=2620.0, close=2622.0, base_time=t0, timeframe=tf_h1
    )
    htf_candles = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13]

    recent_4 = htf_candles[-4:]
    assert recent_4[-1].close < recent_4[0].open

    bias = determine_htf_bias(htf_candles, lookback_bars=4)
    assert bias == Direction.BULLISH

    older_prefix = [
        make_candle(
            i,
            open=2700.0 - i * 5,
            high=2705.0 - i * 5,
            low=2695.0 - i * 5,
            close=2698.0 - i * 5,
            base_time=t0 - timedelta(hours=20),
            timeframe=tf_h1,
        )
        for i in range(10)
    ]
    extended_candles = [*older_prefix, *htf_candles]
    bias_extended = determine_htf_bias(extended_candles, lookback_bars=4)
    assert bias_extended == Direction.BULLISH

    bos_event = BOS(
        time=c9.time,
        confirmed_time=c9.time + timedelta(hours=1),
        direction=Direction.BULLISH,
        broken_swing=Swing(
            time=c3.time,
            price=2670.0,
            type=SwingType.HIGH,
            confirmed_time=c5.time + timedelta(hours=1),
            index=3,
        ),
        breaking_candle=c9,
        price=2670.0,
    )
    assert determine_htf_bias(htf_candles, structure_events=[bos_event]) == Direction.BULLISH


def test_repro_finding_2_dealing_range_consistent_with_impulse_leg_and_ote() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    setups_ote = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        trigger_type=SetupTriggerType.OTE_RETRACEMENT,
    )
    assert len(setups_ote) == 1
    s_ote = setups_ote[0]
    assert s_ote.dealing_range is not None
    assert s_ote.dealing_range.low == pytest.approx(c1.low)
    assert s_ote.dealing_range.high == pytest.approx(c2.high)
    assert s_ote.dealing_range.is_discount(s_ote.entry_price) is True
    ote_zone = s_ote.dealing_range.ote_zone(Direction.BULLISH)
    assert ote_zone[0] <= s_ote.entry_price <= ote_zone[1]

    t0_b = datetime(2026, 9, 24, 14, 0, tzinfo=UTC)
    cb0 = make_candle(0, open=2650.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0_b)
    cb1 = make_candle(1, open=2654.0, high=2658.0, low=2644.0, close=2645.0, base_time=t0_b)
    cb2 = make_candle(2, open=2645.0, high=2646.0, low=2635.0, close=2636.0, base_time=t0_b)
    candles_b = [cb0, cb1, cb2]

    sb_high = Swing(
        time=cb0.time, price=2655.0, type=SwingType.HIGH, confirmed_time=cb1.time, index=0
    )
    pool_b = LiquidityPool(
        type=LiquidityType.BSL,
        price=2655.0,
        swings=(sb_high,),
        time=cb0.time,
        confirmed_time=cb1.time,
    )
    sweep_b = LiquiditySweep(
        time=cb1.time,
        confirmed_time=cb1.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2655.0,
        candle=cb1,
        pool=pool_b,
    )
    sb_low_broken = Swing(
        time=cb0.time, price=2648.0, type=SwingType.LOW, confirmed_time=cb1.time, index=0
    )
    choch_b = ChoCH(
        time=cb2.time,
        confirmed_time=cb2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=sb_low_broken,
        breaking_candle=cb2,
        price=2648.0,
    )
    fvg_b = FVG(
        time=cb1.time,
        confirmed_time=cb2.time + timedelta(minutes=5),
        type=FVGType.BEARISH,
        top=2648.0,
        bottom=2646.0,
        candle_0=cb0,
        candle_1=cb1,
        candle_2=cb2,
    )
    target_b = Swing(
        time=cb2.time,
        price=2620.0,
        type=SwingType.LOW,
        confirmed_time=cb2.time + timedelta(minutes=5),
        index=2,
    )

    setups_ote_b = detect_setups(
        candles_b,
        swings=[target_b],
        structure_events=[choch_b],
        sweeps=[sweep_b],
        fvgs=[fvg_b],
        trigger_type=SetupTriggerType.OTE_RETRACEMENT,
    )
    assert len(setups_ote_b) == 1
    s_ote_b = setups_ote_b[0]
    assert s_ote_b.dealing_range is not None
    assert s_ote_b.dealing_range.high == pytest.approx(cb1.high)
    assert s_ote_b.dealing_range.low == pytest.approx(cb2.low)
    assert s_ote_b.dealing_range.is_premium(s_ote_b.entry_price) is True
    ote_zone_b = s_ote_b.dealing_range.ote_zone(Direction.BEARISH)
    assert ote_zone_b[0] <= s_ote_b.entry_price <= ote_zone_b[1]


def test_detect_setups_bearish_synthetic_tp_and_no_target_rejection() -> None:
    t0_b = datetime(2026, 9, 24, 14, 0, tzinfo=UTC)
    cb0 = make_candle(0, open=2650.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0_b)
    cb1 = make_candle(1, open=2654.0, high=2658.0, low=2644.0, close=2645.0, base_time=t0_b)
    cb2 = make_candle(2, open=2645.0, high=2646.0, low=2635.0, close=2636.0, base_time=t0_b)
    candles_b = [cb0, cb1, cb2]

    sb_high = Swing(
        time=cb0.time, price=2655.0, type=SwingType.HIGH, confirmed_time=cb1.time, index=0
    )
    pool_b = LiquidityPool(
        type=LiquidityType.BSL,
        price=2655.0,
        swings=(sb_high,),
        time=cb0.time,
        confirmed_time=cb1.time,
    )
    sweep_b = LiquiditySweep(
        time=cb1.time,
        confirmed_time=cb1.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2655.0,
        candle=cb1,
        pool=pool_b,
    )
    sb_low_broken = Swing(
        time=cb0.time, price=2648.0, type=SwingType.LOW, confirmed_time=cb1.time, index=0
    )
    choch_b = ChoCH(
        time=cb2.time,
        confirmed_time=cb2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=sb_low_broken,
        breaking_candle=cb2,
        price=2648.0,
    )
    fvg_b = FVG(
        time=cb1.time,
        confirmed_time=cb2.time + timedelta(minutes=5),
        type=FVGType.BEARISH,
        top=2648.0,
        bottom=2646.0,
        candle_0=cb0,
        candle_1=cb1,
        candle_2=cb2,
    )

    setups_syn = detect_setups(
        candles_b,
        swings=[],
        structure_events=[choch_b],
        sweeps=[sweep_b],
        fvgs=[fvg_b],
        allow_synthetic_tp=True,
        min_rr=2.0,
    )
    assert len(setups_syn) == 1
    s_syn = setups_syn[0]
    expected_tp = s_syn.entry_price - (s_syn.stop_loss - s_syn.entry_price) * 2.0
    assert s_syn.take_profit == pytest.approx(expected_tp)

    setups_no_syn = detect_setups(
        candles_b,
        swings=[],
        structure_events=[choch_b],
        sweeps=[sweep_b],
        fvgs=[fvg_b],
        allow_synthetic_tp=False,
    )
    assert setups_no_syn == []


def test_detect_setups_ob_displacement_threshold() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t0)
    c1 = make_candle(1, open=2646.0, high=2653.0, low=2644.0, close=2651.0, base_time=t0)
    c2 = make_candle(2, open=2651.0, high=2654.0, low=2650.0, close=2653.0, base_time=t0)
    candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    ob_low_disp = OrderBlock(
        time=c0.time,
        confirmed_time=choch.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=0.8,
        breaking_event=choch,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    res_rejected = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        order_blocks=[ob_low_disp],
        require_poi=True,
        require_displacement=True,
        min_ob_displacement_atr=1.0,
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
    )
    assert res_rejected == []

    res_accepted = detect_setups(
        candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        order_blocks=[ob_low_disp],
        require_poi=True,
        require_displacement=True,
        min_ob_displacement_atr=0.5,
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
    )
    assert len(res_accepted) == 1


def test_repro_finding_1_htf_precomputed_causal_filter_kills_m1() -> None:
    t0 = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)
    tf_h1 = Timeframe.H1

    htf_candles = [
        Candle(
            time=t0 + timedelta(hours=0),
            open=2610.0,
            high=2620.0,
            low=2608.0,
            close=2618.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=1),
            open=2618.0,
            high=2635.0,
            low=2615.0,
            close=2630.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=2),
            open=2630.0,
            high=2650.0,
            low=2625.0,
            close=2645.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=3),
            open=2645.0,
            high=2670.0,
            low=2640.0,
            close=2665.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=4),
            open=2658.0,
            high=2660.0,
            low=2635.0,
            close=2640.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=5),
            open=2640.0,
            high=2650.0,
            low=2620.0,
            close=2625.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=6),
            open=2625.0,
            high=2630.0,
            low=2615.0,
            close=2620.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=7),
            open=2620.0,
            high=2640.0,
            low=2620.0,
            close=2635.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=8),
            open=2635.0,
            high=2655.0,
            low=2630.0,
            close=2650.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=9),
            open=2650.0,
            high=2678.0,
            low=2648.0,
            close=2675.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=10),
            open=2675.0,
            high=2676.0,
            low=2655.0,
            close=2658.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=11),
            open=2658.0,
            high=2660.0,
            low=2640.0,
            close=2642.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=12),
            open=2642.0,
            high=2645.0,
            low=2630.0,
            close=2632.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=13),
            open=2632.0,
            high=2635.0,
            low=2620.0,
            close=2622.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=14),
            open=2622.0,
            high=2624.0,
            low=2618.0,
            close=2620.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=15),
            open=2620.0,
            high=2622.0,
            low=2616.0,
            close=2618.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
        Candle(
            time=t0 + timedelta(hours=16),
            open=2618.0,
            high=2619.0,
            low=2605.0,
            close=2610.0,
            volume=100.0,
            timeframe=tf_h1,
        ),
    ]

    htf_swings = find_swings(htf_candles)
    htf_events = detect_structure(htf_candles, swings=htf_swings)
    assert len(htf_events) == 2
    assert htf_events[0].direction == Direction.BULLISH
    assert htf_events[0].confirmed_time == t0 + timedelta(hours=10)
    assert htf_events[1].direction == Direction.BEARISH
    assert htf_events[1].confirmed_time == t0 + timedelta(hours=17)

    t_ltf = t0 + timedelta(hours=10)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t_ltf)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t_ltf)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t_ltf)
    ltf_candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    setups = detect_setups(
        ltf_candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles,
        htf_alignment_required=True,
    )
    assert len(setups) == 1
    assert "HTF_ALIGNED" in setups[0].confluences
    assert setups[0].direction == Direction.BULLISH


def test_repro_finding_1_structure_events_last_event_rule_kills_m2() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    tf_h1 = Timeframe.H1
    c0 = make_candle(
        0, open=2650.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0, timeframe=tf_h1
    )
    s_dummy = Swing(
        time=t0, price=2650.0, type=SwingType.HIGH, confirmed_time=t0 + timedelta(hours=1), index=0
    )

    s_low_dummy = Swing(
        time=t0, price=2660.0, type=SwingType.LOW, confirmed_time=t0 + timedelta(hours=1), index=0
    )
    c_break = make_candle(
        1, open=2650.0, high=2652.0, low=2635.0, close=2638.0, base_time=t0, timeframe=tf_h1
    )

    e1_bull = BOS(
        time=t0 + timedelta(hours=1),
        confirmed_time=t0 + timedelta(hours=2),
        direction=Direction.BULLISH,
        broken_swing=s_dummy,
        breaking_candle=c0,
        price=2650.0,
    )
    e2_bear = ChoCH(
        time=t0 + timedelta(hours=3),
        confirmed_time=t0 + timedelta(hours=4),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=s_low_dummy,
        breaking_candle=c_break,
        price=2660.0,
    )

    bias = determine_htf_bias([c0], structure_events=[e1_bull, e2_bear])
    assert bias == Direction.BEARISH


def test_repro_finding_1_empty_structure_events_falls_back_to_window_kills_m8() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    tf_h1 = Timeframe.H1
    c_bull = make_candle(
        0, open=2650.0, high=2655.0, low=2648.0, close=2654.0, base_time=t0, timeframe=tf_h1
    )

    bias = determine_htf_bias([c_bull], structure_events=[])
    assert bias == Direction.BULLISH


def test_repro_finding_1_htf_lookback_bars_parameterized_kills_m9_m10() -> None:
    t0 = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)
    tf_h1 = Timeframe.H1

    h0 = make_candle(
        0, open=2670.0, high=2672.0, low=2664.0, close=2665.0, base_time=t0, timeframe=tf_h1
    )
    h1 = make_candle(
        1, open=2665.0, high=2666.0, low=2659.0, close=2660.0, base_time=t0, timeframe=tf_h1
    )
    h2 = make_candle(
        2, open=2660.0, high=2661.0, low=2639.0, close=2640.0, base_time=t0, timeframe=tf_h1
    )
    h3 = make_candle(
        3, open=2640.0, high=2646.0, low=2639.0, close=2645.0, base_time=t0, timeframe=tf_h1
    )
    h4 = make_candle(
        4, open=2645.0, high=2656.0, low=2644.0, close=2655.0, base_time=t0, timeframe=tf_h1
    )
    h5 = make_candle(
        5, open=2655.0, high=2660.0, low=2654.0, close=2658.0, base_time=t0, timeframe=tf_h1
    )
    htf_candles = [h0, h1, h2, h3, h4, h5]

    bias_3 = determine_htf_bias(htf_candles, lookback_bars=3)
    bias_5 = determine_htf_bias(htf_candles, lookback_bars=5)
    assert bias_3 == Direction.BULLISH
    assert bias_5 == Direction.BEARISH

    t_ltf = t0 + timedelta(hours=6)
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2645.0, close=2646.0, base_time=t_ltf)
    c1 = make_candle(1, open=2646.0, high=2656.0, low=2644.0, close=2655.0, base_time=t_ltf)
    c2 = make_candle(2, open=2655.0, high=2665.0, low=2654.0, close=2664.0, base_time=t_ltf)
    ltf_candles = [c0, c1, c2]

    s_low = Swing(time=c0.time, price=2645.0, type=SwingType.LOW, confirmed_time=c1.time, index=0)
    pool = LiquidityPool(
        type=LiquidityType.SSL, price=2645.0, swings=(s_low,), time=c0.time, confirmed_time=c1.time
    )
    sweep = LiquiditySweep(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2645.0,
        candle=c1,
        pool=pool,
    )
    s_high_broken = Swing(
        time=c0.time, price=2652.0, type=SwingType.HIGH, confirmed_time=c1.time, index=0
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_broken,
        breaking_candle=c2,
        price=2652.0,
    )
    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2654.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    target = Swing(
        time=c2.time,
        price=2680.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=2,
    )

    res_3 = detect_setups(
        ltf_candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles,
        htf_alignment_required=True,
        htf_lookback_bars=3,
    )
    assert len(res_3) == 1
    assert "HTF_ALIGNED" in res_3[0].confluences

    res_5 = detect_setups(
        ltf_candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles,
        htf_alignment_required=True,
        htf_lookback_bars=5,
    )
    assert len(res_5) == 0

    res_short = detect_setups(
        ltf_candles,
        swings=[target],
        structure_events=[choch],
        sweeps=[sweep],
        fvgs=[fvg],
        htf_candles=htf_candles[2:],
        htf_alignment_required=True,
        htf_lookback_bars=3,
    )
    assert len(res_short) == 1


def test_detect_setups_emits_htf_counter() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2662.0, low=2650.0, close=2652.0, base_time=t0)
    c1 = make_candle(1, open=2652.0, high=2658.0, low=2651.0, close=2656.0, base_time=t0)
    c2 = make_candle(2, open=2656.0, high=2657.0, low=2648.0, close=2649.0, base_time=t0)
    c3 = make_candle(3, open=2649.0, high=2652.0, low=2647.0, close=2651.0, base_time=t0)
    c4 = make_candle(4, open=2651.0, high=2665.0, low=2650.0, close=2663.0, base_time=t0)
    c5 = make_candle(5, open=2663.0, high=2675.0, low=2660.0, close=2674.0, base_time=t0)
    c6 = make_candle(6, open=2674.0, high=2680.0, low=2670.0, close=2678.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5, c6]

    s_high_0 = Swing(
        time=c1.time,
        price=2658.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    s_low_0 = Swing(
        time=c0.time,
        price=2650.0,
        type=SwingType.LOW,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2650.0,
        swings=(s_low_0,),
        time=c0.time,
        confirmed_time=c1.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2650.0,
        candle=c3,
        pool=pool_ssl,
    )
    choch_bull = ChoCH(
        time=c4.time,
        confirmed_time=c4.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c4,
        price=2658.0,
    )
    fvg_bull = FVG(
        time=c4.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2660.0,
        bottom=2652.0,
        candle_0=c3,
        candle_1=c4,
        candle_2=c5,
    )
    target_high = Swing(
        time=c6.time,
        price=2685.0,
        type=SwingType.HIGH,
        confirmed_time=c6.time + timedelta(minutes=5),
        index=6,
    )

    htf_candles = [
        make_candle(
            0,
            open=2700.0,
            high=2705.0,
            low=2670.0,
            close=2675.0,
            timeframe=Timeframe.H1,
            base_time=t0 - timedelta(hours=2),
        ),
        make_candle(
            1,
            open=2675.0,
            high=2680.0,
            low=2640.0,
            close=2645.0,
            timeframe=Timeframe.H1,
            base_time=t0 - timedelta(hours=2),
        ),
    ]

    setups = detect_setups(
        candles,
        swings=[s_low_0, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        fvgs=[fvg_bull],
        htf_candles=htf_candles,
        htf_alignment_required=False,
    )
    assert len(setups) == 1
    assert "HTF_COUNTER" in setups[0].confluences
