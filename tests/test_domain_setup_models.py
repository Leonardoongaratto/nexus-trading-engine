import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from xau_bot.domain.models import (
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
    Setup,
    SetupStatus,
    SetupTriggerType,
    Swing,
    SwingType,
    Timeframe,
    TradeSetup,
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


def make_bullish_setup_fixtures() -> tuple[LiquiditySweep, ChoCH, FVG]:
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
    s_high = Swing(
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
        broken_swing=s_high,
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
    return sweep, choch, fvg


def test_setup_trigger_type_enum() -> None:
    assert SetupTriggerType.FVG_CE.value == "FVG_CE"
    assert SetupTriggerType.FVG_PROXIMAL.value == "FVG_PROXIMAL"
    assert SetupTriggerType.OB_EXTREME.value == "OB_EXTREME"
    assert SetupTriggerType.OB_MEAN_THRESHOLD.value == "OB_MEAN_THRESHOLD"
    assert SetupTriggerType.OTE_RETRACEMENT.value == "OTE_RETRACEMENT"


def test_setup_status_enum() -> None:
    assert SetupStatus.PENDING.value == "PENDING"
    assert SetupStatus.TRIGGERED.value == "TRIGGERED"
    assert SetupStatus.INVALIDATED.value == "INVALIDATED"
    assert SetupStatus.EXPIRED.value == "EXPIRED"
    assert len(SetupStatus) == 4


def test_trade_setup_bullish_valid_and_properties() -> None:
    sweep, choch, fvg = make_bullish_setup_fixtures()
    t_setup = choch.confirmed_time
    entry = 2653.0
    sl = 2643.0
    tp = 2673.0
    expected_rr = (tp - entry) / (entry - sl)
    assert expected_rr == pytest.approx(2.0)

    setup = TradeSetup(
        id="setup_b_1",
        time=choch.time,
        confirmed_time=t_setup,
        direction=Direction.BULLISH,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward_ratio=expected_rr,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        poi=fvg,
        confluences=("SWEEP_CONFIRMED", "CHOCH_CONFIRMED", "FVG_PRESENT"),
        valid_until=t_setup + timedelta(hours=2),
    )

    assert setup.id == "setup_b_1"
    assert setup.direction == Direction.BULLISH
    assert setup.status == SetupStatus.PENDING
    assert setup.poi == fvg
    assert setup.risk_reward_ratio == pytest.approx(2.0)
    assert len(setup.confluences) == 3
    assert setup == TradeSetup(
        id="setup_b_1",
        time=choch.time,
        confirmed_time=t_setup,
        direction=Direction.BULLISH,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward_ratio=expected_rr,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.FVG_CE,
        sweep=sweep,
        choch=choch,
        poi=fvg,
        confluences=("SWEEP_CONFIRMED", "CHOCH_CONFIRMED", "FVG_PRESENT"),
        valid_until=t_setup + timedelta(hours=2),
    )


def test_trade_setup_bearish_valid_and_order_block_poi() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2660.0, high=2668.0, low=2659.0, close=2667.0, base_time=t0)
    c1 = make_candle(1, open=2667.0, high=2669.0, low=2658.0, close=2658.5, base_time=t0)
    c2 = make_candle(2, open=2658.5, high=2659.0, low=2648.0, close=2649.0, base_time=t0)

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
    s_low = Swing(
        time=c0.time,
        price=2659.0,
        type=SwingType.LOW,
        confirmed_time=c1.time,
        index=0,
    )
    choch = ChoCH(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=s_low,
        breaking_candle=c2,
        price=2659.0,
    )
    ob = OrderBlock(
        time=c0.time,
        confirmed_time=choch.confirmed_time,
        type=OrderBlockType.BEARISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=2.5,
        breaking_event=choch,
    )

    t_setup = choch.confirmed_time
    entry = 2663.5
    sl = 2670.0
    tp = 2644.0
    expected_rr = (entry - tp) / (sl - entry)
    assert expected_rr == pytest.approx(3.0)

    setup = TradeSetup(
        id="setup_s_1",
        time=choch.time,
        confirmed_time=t_setup,
        direction=Direction.BEARISH,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward_ratio=expected_rr,
        timeframe=Timeframe.M5,
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        sweep=sweep,
        choch=choch,
        poi=ob,
    )

    assert setup.direction == Direction.BEARISH
    assert setup.poi == ob
    assert setup.risk_reward_ratio == pytest.approx(3.0)


def test_trade_setup_immutability() -> None:
    sweep, choch, fvg = make_bullish_setup_fixtures()
    setup = TradeSetup(
        id="setup_imm",
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        attr_to_mutate = "entry_price"
        setattr(setup, attr_to_mutate, 2655.0)


def test_trade_setup_validations_and_invariants() -> None:
    sweep, choch, fvg = make_bullish_setup_fixtures()
    t_setup = choch.confirmed_time

    with pytest.raises(ValueError, match="setup id must not be empty"):
        TradeSetup(
            id="",
            time=choch.time,
            confirmed_time=t_setup,
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

    with pytest.raises(ValueError, match="time must be timezone-aware"):
        TradeSetup(
            id="s1",
            time=datetime(2026, 9, 24, 12, 0),
            confirmed_time=t_setup,
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

    with pytest.raises(ValueError, match="confirmed_time must be timezone-aware"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=datetime(2026, 9, 24, 12, 10),
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

    with pytest.raises(ValueError, match="confirmed_time must be >= time"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=choch.time - timedelta(minutes=5),
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

    with pytest.raises(ValueError, match="valid_until must be timezone-aware"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
            valid_until=datetime(2026, 9, 24, 15, 0),
        )

    with pytest.raises(ValueError, match="valid_until must be >= confirmed_time"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
            valid_until=t_setup - timedelta(minutes=1),
        )

    with pytest.raises(ValueError, match="entry_price must be positive"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=0.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(ValueError, match="stop_loss must be positive"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=-10.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(ValueError, match="take_profit must be positive"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=-1.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(ValueError, match="stop_loss must be finite"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=float("nan"),
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(
        ValueError, match="bullish setup requires stop_loss < entry_price < take_profit"
    ):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2655.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(
        ValueError, match="bullish setup requires stop_loss < entry_price < take_profit"
    ):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2650.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(
        ValueError, match="bearish setup requires stop_loss > entry_price > take_profit"
    ):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BEARISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2633.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(ValueError, match="bullish setup requires SSL liquidity sweep"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=LiquiditySweep(
                time=choch.time,
                confirmed_time=t_setup,
                type=LiquidityType.BSL,
                level=2660.0,
                candle=make_candle(
                    9,
                    open=2650.0,
                    high=2665.0,
                    low=2648.0,
                    close=2655.0,
                    base_time=choch.time,
                ),
                pool=LiquidityPool(
                    type=LiquidityType.BSL,
                    price=2660.0,
                    swings=(
                        Swing(
                            time=choch.time,
                            price=2660.0,
                            type=SwingType.HIGH,
                            confirmed_time=t_setup,
                            index=0,
                        ),
                    ),
                    time=choch.time,
                    confirmed_time=t_setup,
                ),
            ),
            choch=choch,
        )

    with pytest.raises(ValueError, match="choch direction must match setup direction"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=ChoCH(
                time=choch.time,
                confirmed_time=t_setup,
                direction=Direction.BEARISH,
                previous_trend=Direction.BULLISH,
                broken_swing=sweep.pool.swings[0],
                breaking_candle=make_candle(
                    10,
                    open=2650.0,
                    high=2651.0,
                    low=2640.0,
                    close=2642.0,
                    base_time=choch.time,
                ),
                price=2645.0,
            ),
        )

    with pytest.raises(ValueError, match="POI direction must match setup direction"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
            poi=FVG(
                time=choch.time - timedelta(minutes=10),
                confirmed_time=t_setup,
                type=FVGType.BEARISH,
                top=2665.0,
                bottom=2658.0,
                candle_0=make_candle(
                    0,
                    open=2670.0,
                    high=2672.0,
                    low=2665.0,
                    close=2668.0,
                    base_time=choch.time - timedelta(minutes=15),
                ),
                candle_1=make_candle(
                    1,
                    open=2668.0,
                    high=2669.0,
                    low=2655.0,
                    close=2656.0,
                    base_time=choch.time - timedelta(minutes=15),
                ),
                candle_2=make_candle(
                    2,
                    open=2656.0,
                    high=2658.0,
                    low=2648.0,
                    close=2649.0,
                    base_time=choch.time - timedelta(minutes=15),
                ),
            ),
        )

    with pytest.raises(ValueError, match="risk_reward_ratio mismatch"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=5.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    with pytest.raises(ValueError, match="status_updated_at must be timezone-aware"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
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
            status_updated_at=datetime(2026, 9, 24, 12, 30),
        )

    with pytest.raises(ValueError, match="status_updated_at must be >= confirmed_time"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
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
            status_updated_at=t_setup - timedelta(minutes=5),
        )

    with pytest.raises(ValueError, match="status must be a SetupStatus enum"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=Direction.BULLISH,
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
            status=cast(Any, "INVALID_STATUS"),
        )

    with pytest.raises(ValueError, match="Invalid direction"):
        TradeSetup(
            id="s1",
            time=choch.time,
            confirmed_time=t_setup,
            direction=cast(Any, "SIDEWAYS"),
            entry_price=2653.0,
            stop_loss=2643.0,
            take_profit=2673.0,
            risk_reward_ratio=2.0,
            timeframe=Timeframe.M5,
            trigger_type=SetupTriggerType.FVG_CE,
            sweep=sweep,
            choch=choch,
        )

    for invalid_rr in (-1.0, 0.0):
        with pytest.raises(ValueError, match="risk_reward_ratio must be positive"):
            TradeSetup(
                id="s1",
                time=choch.time,
                confirmed_time=t_setup,
                direction=Direction.BULLISH,
                entry_price=2653.0,
                stop_loss=2643.0,
                take_profit=2673.0,
                risk_reward_ratio=invalid_rr,
                timeframe=Timeframe.M5,
                trigger_type=SetupTriggerType.FVG_CE,
                sweep=sweep,
                choch=choch,
            )

    for non_finite_rr in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="risk_reward_ratio must be finite"):
            TradeSetup(
                id="s1",
                time=choch.time,
                confirmed_time=t_setup,
                direction=Direction.BULLISH,
                entry_price=2653.0,
                stop_loss=2643.0,
                take_profit=2673.0,
                risk_reward_ratio=non_finite_rr,
                timeframe=Timeframe.M5,
                trigger_type=SetupTriggerType.FVG_CE,
                sweep=sweep,
                choch=choch,
            )


def test_setup_alias_and_backward_compatibility() -> None:
    assert Setup is TradeSetup
