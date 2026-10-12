import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.domain.models import (
    BOS,
    FVG,
    Candle,
    Direction,
    FVGMitigationState,
    FVGType,
    OrderBlock,
    OrderBlockType,
    Swing,
    SwingType,
    Timeframe,
)

UTC = UTC


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


def test_fvg_type_and_mitigation_state_enum_values() -> None:
    assert FVGType.BULLISH == "BULLISH"
    assert FVGType.BEARISH == "BEARISH"
    assert FVGMitigationState.UNMITIGATED == "UNMITIGATED"
    assert FVGMitigationState.PARTIALLY_MITIGATED == "PARTIALLY_MITIGATED"
    assert FVGMitigationState.CE_TESTED == "CE_TESTED"
    assert FVGMitigationState.INVALIDATED == "INVALIDATED"


def test_bullish_fvg_creation_and_properties() -> None:
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0)

    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2655.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )

    assert fvg.type == FVGType.BULLISH
    assert fvg.top == 2655.0
    assert fvg.bottom == 2652.0
    assert fvg.size == pytest.approx(3.0)
    assert fvg.consequent_encroachment == pytest.approx(2653.5)
    assert not fvg.mitigated
    assert not fvg.ce_tested
    assert not fvg.invalidated
    assert fvg.mitigated_at is None
    assert fvg.invalidated_at is None


def test_bearish_fvg_creation_and_properties() -> None:
    c0 = make_candle(0, open=2665.0, high=2666.0, low=2663.0, close=2664.0)
    c1 = make_candle(1, open=2664.0, high=2664.0, low=2654.0, close=2655.0)
    c2 = make_candle(2, open=2655.0, high=2658.0, low=2652.0, close=2653.0)

    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BEARISH,
        top=2663.0,
        bottom=2658.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )

    assert fvg.type == FVGType.BEARISH
    assert fvg.top == 2663.0
    assert fvg.bottom == 2658.0
    assert fvg.size == pytest.approx(5.0)
    assert fvg.consequent_encroachment == pytest.approx(2660.5)


def test_fvg_validation_invariants() -> None:
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0)

    with pytest.raises(ValueError, match="timezone-aware"):
        FVG(
            time=datetime(2026, 9, 24, 12, 5),
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        FVG(
            time=c1.time,
            confirmed_time=datetime(2026, 9, 24, 12, 15),
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="confirmed_time must be strictly after"):
        FVG(
            time=c1.time,
            confirmed_time=c1.time,
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="top must be strictly greater than bottom"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=2652.0,
            bottom=2655.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="positive"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=-10.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="finite"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=float("nan"),
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="Bullish FVG requires candle_2.low > candle_0.high"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=2652.0,
            candle_0=c2,
            candle_1=c1,
            candle_2=c0,
        )

    with pytest.raises(ValueError, match="Bearish FVG requires candle_2.high < candle_0.low"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BEARISH,
            top=2663.0,
            bottom=2658.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )


def test_order_block_type_enum() -> None:
    assert OrderBlockType.BULLISH == "BULLISH"
    assert OrderBlockType.BEARISH == "BEARISH"


def test_bullish_order_block_creation_and_properties() -> None:
    c_ob = make_candle(0, open=2655.0, high=2656.0, low=2648.0, close=2649.0)
    c_break = make_candle(2, open=2658.0, high=2670.0, low=2657.0, close=2668.0)
    swing_high = Swing(
        time=c_ob.time - timedelta(minutes=15),
        price=2665.0,
        type=SwingType.HIGH,
        confirmed_time=c_ob.time - timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=c_break.time,
        confirmed_time=c_break.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        broken_swing=swing_high,
        breaking_candle=c_break,
        price=2665.0,
    )

    ob = OrderBlock(
        time=c_ob.time,
        confirmed_time=bos.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c_ob.high,
        low=c_ob.low,
        open=c_ob.open,
        close=c_ob.close,
        candle=c_ob,
        breaking_event=bos,
        displacement_atr_ratio=1.8,
    )

    assert ob.type == OrderBlockType.BULLISH
    assert ob.high == 2656.0
    assert ob.low == 2648.0
    assert ob.open == 2655.0
    assert ob.close == 2649.0
    assert ob.mean_threshold == pytest.approx(2652.0)
    assert ob.displacement_atr_ratio == 1.8
    assert ob.breaking_event == bos
    assert not ob.mitigated
    assert not ob.invalidated


def test_bearish_order_block_creation_and_properties() -> None:
    c_ob = make_candle(0, open=2660.0, high=2668.0, low=2659.0, close=2667.0)
    c_break = make_candle(2, open=2655.0, high=2656.0, low=2640.0, close=2642.0)
    swing_low = Swing(
        time=c_ob.time - timedelta(minutes=15),
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c_ob.time - timedelta(minutes=5),
        index=0,
    )
    bos = BOS(
        time=c_break.time,
        confirmed_time=c_break.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=swing_low,
        breaking_candle=c_break,
        price=2645.0,
    )

    ob = OrderBlock(
        time=c_ob.time,
        confirmed_time=bos.confirmed_time,
        type=OrderBlockType.BEARISH,
        high=c_ob.high,
        low=c_ob.low,
        open=c_ob.open,
        close=c_ob.close,
        candle=c_ob,
        breaking_event=bos,
        displacement_atr_ratio=2.2,
    )

    assert ob.type == OrderBlockType.BEARISH
    assert ob.high == 2668.0
    assert ob.low == 2659.0
    assert ob.mean_threshold == pytest.approx(2663.5)


def test_order_block_validation_invariants() -> None:
    c_ob = make_candle(0, open=2655.0, high=2656.0, low=2648.0, close=2649.0)

    with pytest.raises(ValueError, match="timezone-aware"):
        OrderBlock(
            time=datetime(2026, 9, 24, 12, 0),
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=c_ob.high,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="confirmed_time must be at or after"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time,
            type=OrderBlockType.BULLISH,
            high=c_ob.high,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="high must be strictly greater than low"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=2648.0,
            low=2656.0,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="positive"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=2656.0,
            low=-5.0,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="finite"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=float("inf"),
            low=2648.0,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="OrderBlock prices must match candle"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=2699.0,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    swing_low = Swing(
        time=c_ob.time - timedelta(minutes=15),
        price=2645.0,
        type=SwingType.LOW,
        confirmed_time=c_ob.time - timedelta(minutes=5),
        index=0,
    )
    c_break_bearish = make_candle(1, open=2648.0, high=2648.0, low=2640.0, close=2642.0)
    bearish_bos = BOS(
        time=c_break_bearish.time,
        confirmed_time=c_break_bearish.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        broken_swing=swing_low,
        breaking_candle=c_break_bearish,
        price=2645.0,
    )
    with pytest.raises(ValueError, match="direction mismatch with breaking_event"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=OrderBlockType.BULLISH,
            high=c_ob.high,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
            breaking_event=bearish_bos,
        )


def test_fvg_additional_validations() -> None:
    from typing import Any, cast

    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0)

    with pytest.raises(ValueError, match="Invalid FVG type"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=cast(Any, "INVALID"),
            top=2655.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="Bullish FVG levels must match candles"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time + timedelta(minutes=5),
            type=FVGType.BULLISH,
            top=2656.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    c_bear0 = make_candle(0, open=2665.0, high=2666.0, low=2663.0, close=2664.0)
    c_bear1 = make_candle(1, open=2664.0, high=2664.0, low=2654.0, close=2655.0)
    c_bear2 = make_candle(2, open=2655.0, high=2658.0, low=2652.0, close=2653.0)

    with pytest.raises(ValueError, match="Bearish FVG levels must match candles"):
        FVG(
            time=c_bear1.time,
            confirmed_time=c_bear2.time + timedelta(minutes=5),
            type=FVGType.BEARISH,
            top=2664.0,
            bottom=2658.0,
            candle_0=c_bear0,
            candle_1=c_bear1,
            candle_2=c_bear2,
        )


def test_order_block_additional_validations() -> None:
    from typing import Any, cast

    c_ob = make_candle(0, open=2655.0, high=2656.0, low=2648.0, close=2649.0)

    with pytest.raises(ValueError, match="confirmed_time must be timezone-aware"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=datetime(2026, 9, 24, 12, 15),
            type=OrderBlockType.BULLISH,
            high=c_ob.high,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )

    with pytest.raises(ValueError, match="Invalid OrderBlock type"):
        OrderBlock(
            time=c_ob.time,
            confirmed_time=c_ob.time + timedelta(minutes=15),
            type=cast(Any, "INVALID"),
            high=c_ob.high,
            low=c_ob.low,
            open=c_ob.open,
            close=c_ob.close,
            candle=c_ob,
        )


def test_fvg_state_property_and_state_consistency_validations() -> None:
    c0 = make_candle(0, open=2650.0, high=2652.0, low=2649.0, close=2651.0)
    c1 = make_candle(1, open=2651.0, high=2660.0, low=2651.0, close=2659.0)
    c2 = make_candle(2, open=2659.0, high=2663.0, low=2655.0, close=2662.0)
    c3 = make_candle(3, open=2662.0, high=2662.0, low=2653.0, close=2658.0)

    fvg = FVG(
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=FVGType.BULLISH,
        top=2655.0,
        bottom=2652.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    assert fvg.state == FVGMitigationState.UNMITIGATED

    fvg_mit = dataclasses.replace(
        fvg,
        mitigated=True,
        mitigated_at=c3.time + timedelta(minutes=5),
        mitigated_by_candle=c3,
    )
    assert fvg_mit.state == FVGMitigationState.PARTIALLY_MITIGATED

    fvg_ce = dataclasses.replace(
        fvg_mit,
        ce_tested=True,
        ce_tested_at=c3.time + timedelta(minutes=5),
    )
    assert fvg_ce.state == FVGMitigationState.CE_TESTED

    fvg_inv = dataclasses.replace(
        fvg_ce,
        invalidated=True,
        invalidated_at=c3.time + timedelta(minutes=5),
        invalidated_by_candle=c3,
    )
    assert fvg_inv.state == FVGMitigationState.INVALIDATED

    with pytest.raises(ValueError, match="confirmed_time must be at or after candle_2 close"):
        FVG(
            time=c1.time,
            confirmed_time=c2.time,
            type=FVGType.BULLISH,
            top=2655.0,
            bottom=2652.0,
            candle_0=c0,
            candle_1=c1,
            candle_2=c2,
        )

    with pytest.raises(ValueError, match="mitigated=True requires mitigated_at to be set"):
        dataclasses.replace(fvg, mitigated=True, mitigated_by_candle=c3)

    with pytest.raises(ValueError, match="mitigated=True requires mitigated_by_candle to be set"):
        dataclasses.replace(fvg, mitigated=True, mitigated_at=c3.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="mitigated_at must be timezone-aware"):
        dataclasses.replace(
            fvg,
            mitigated=True,
            mitigated_at=datetime(2026, 9, 24, 12, 20),
            mitigated_by_candle=c3,
        )

    with pytest.raises(ValueError, match="mitigated_at must be >= confirmed_time"):
        dataclasses.replace(
            fvg,
            mitigated=True,
            mitigated_at=c1.time,
            mitigated_by_candle=c3,
        )

    with pytest.raises(ValueError, match="mitigated=False requires mitigated_at to be None"):
        dataclasses.replace(fvg, mitigated=False, mitigated_at=c3.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="ce_tested=True requires ce_tested_at to be set"):
        dataclasses.replace(fvg, ce_tested=True)

    with pytest.raises(ValueError, match="ce_tested_at must be >= confirmed_time"):
        dataclasses.replace(fvg, ce_tested=True, ce_tested_at=c1.time)

    with pytest.raises(ValueError, match="ce_tested=False requires ce_tested_at to be None"):
        dataclasses.replace(fvg, ce_tested=False, ce_tested_at=c3.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="invalidated=True requires invalidated_at to be set"):
        dataclasses.replace(fvg, invalidated=True, invalidated_by_candle=c3)

    with pytest.raises(
        ValueError, match="invalidated=True requires invalidated_by_candle to be set"
    ):
        dataclasses.replace(fvg, invalidated=True, invalidated_at=c3.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="invalidated_at must be >= confirmed_time"):
        dataclasses.replace(
            fvg,
            invalidated=True,
            invalidated_at=c1.time,
            invalidated_by_candle=c3,
        )

    with pytest.raises(ValueError, match="invalidated=False requires invalidated_at to be None"):
        dataclasses.replace(fvg, invalidated=False, invalidated_at=c3.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="mitigated=False requires mitigated_by_candle to be None"):
        dataclasses.replace(fvg, mitigated=False, mitigated_by_candle=c3)

    with pytest.raises(ValueError, match="ce_tested_at must be timezone-aware"):
        dataclasses.replace(fvg, ce_tested=True, ce_tested_at=datetime(2026, 9, 24, 12, 20))

    with pytest.raises(ValueError, match="invalidated_at must be timezone-aware"):
        dataclasses.replace(
            fvg,
            invalidated=True,
            invalidated_at=datetime(2026, 9, 24, 12, 20),
            invalidated_by_candle=c3,
        )

    with pytest.raises(
        ValueError, match="invalidated=False requires invalidated_by_candle to be None"
    ):
        dataclasses.replace(fvg, invalidated=False, invalidated_by_candle=c3)


def test_order_block_state_consistency_validations() -> None:
    c_ob = make_candle(0, open=2655.0, high=2656.0, low=2648.0, close=2649.0)
    c_other = make_candle(3, open=2658.0, high=2660.0, low=2650.0, close=2651.0)
    ob = OrderBlock(
        time=c_ob.time,
        confirmed_time=c_ob.time + timedelta(minutes=10),
        type=OrderBlockType.BULLISH,
        high=c_ob.high,
        low=c_ob.low,
        open=c_ob.open,
        close=c_ob.close,
        candle=c_ob,
    )

    with pytest.raises(ValueError, match="mitigated=True requires mitigated_at to be set"):
        dataclasses.replace(ob, mitigated=True, mitigated_by_candle=c_other)

    with pytest.raises(ValueError, match="mitigated=True requires mitigated_by_candle to be set"):
        dataclasses.replace(ob, mitigated=True, mitigated_at=c_other.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="mitigated_at must be timezone-aware"):
        dataclasses.replace(
            ob,
            mitigated=True,
            mitigated_at=datetime(2026, 9, 24, 12, 20),
            mitigated_by_candle=c_other,
        )

    with pytest.raises(ValueError, match="mitigated_at must be >= confirmed_time"):
        dataclasses.replace(
            ob,
            mitigated=True,
            mitigated_at=c_ob.time,
            mitigated_by_candle=c_other,
        )

    with pytest.raises(ValueError, match="mitigated=False requires mitigated_at to be None"):
        dataclasses.replace(ob, mitigated=False, mitigated_at=c_other.time + timedelta(minutes=5))

    with pytest.raises(ValueError, match="invalidated=True requires invalidated_at to be set"):
        dataclasses.replace(ob, invalidated=True, invalidated_by_candle=c_other)

    with pytest.raises(
        ValueError, match="invalidated=True requires invalidated_by_candle to be set"
    ):
        dataclasses.replace(
            ob, invalidated=True, invalidated_at=c_other.time + timedelta(minutes=5)
        )

    with pytest.raises(ValueError, match="invalidated_at must be >= confirmed_time"):
        dataclasses.replace(
            ob,
            invalidated=True,
            invalidated_at=c_ob.time,
            invalidated_by_candle=c_other,
        )

    with pytest.raises(ValueError, match="invalidated=False requires invalidated_at to be None"):
        dataclasses.replace(
            ob, invalidated=False, invalidated_at=c_other.time + timedelta(minutes=5)
        )

    with pytest.raises(ValueError, match="mitigated=False requires mitigated_by_candle to be None"):
        dataclasses.replace(ob, mitigated=False, mitigated_by_candle=c_other)

    with pytest.raises(ValueError, match="invalidated_at must be timezone-aware"):
        dataclasses.replace(
            ob,
            invalidated=True,
            invalidated_at=datetime(2026, 9, 24, 12, 20),
            invalidated_by_candle=c_other,
        )

    with pytest.raises(
        ValueError, match="invalidated=False requires invalidated_by_candle to be None"
    ):
        dataclasses.replace(ob, invalidated=False, invalidated_by_candle=c_other)
