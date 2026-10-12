from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from xau_bot.domain.models import (
    FVG,
    Candle,
    Direction,
    FVGType,
    OrderBlock,
    OrderBlockType,
    OrderIntent,
    Setup,
    Swing,
    SwingType,
    Tick,
    Timeframe,
)


def test_candle_valid_invariants() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candle = Candle(
        time=now,
        open=2650.0,
        high=2655.0,
        low=2648.0,
        close=2653.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    assert candle.time == now
    assert candle.open == 2650.0
    assert candle.high == 2655.0
    assert candle.low == 2648.0
    assert candle.close == 2653.0
    assert candle.volume == 100.0
    assert candle.timeframe == Timeframe.M5
    assert candle.is_bullish is True
    assert candle.is_bearish is False
    assert candle.range == 7.0
    assert candle.body_range == 3.0


def test_candle_bearish() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candle = Candle(
        time=now,
        open=2655.0,
        high=2658.0,
        low=2645.0,
        close=2648.0,
        volume=50.0,
        timeframe=Timeframe.M5,
    )
    assert candle.is_bullish is False
    assert candle.is_bearish is True
    assert candle.range == 13.0
    assert candle.body_range == 7.0


def test_candle_invalid_ohlc() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="High cannot be less than"):
        Candle(
            time=now,
            open=2650.0,
            high=2645.0,
            low=2640.0,
            close=2642.0,
            volume=10.0,
            timeframe=Timeframe.M5,
        )

    with pytest.raises(ValueError, match="Low cannot be greater than"):
        Candle(
            time=now,
            open=2650.0,
            high=2655.0,
            low=2652.0,
            close=2653.0,
            volume=10.0,
            timeframe=Timeframe.M5,
        )

    with pytest.raises(ValueError, match="Volume cannot be negative"):
        Candle(
            time=now,
            open=2650.0,
            high=2655.0,
            low=2648.0,
            close=2653.0,
            volume=-1.0,
            timeframe=Timeframe.M5,
        )


def test_candle_naive_datetime_rejected() -> None:
    naive_dt = datetime(2026, 9, 24, 12, 0)
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        Candle(
            time=naive_dt,
            open=2650.0,
            high=2655.0,
            low=2648.0,
            close=2653.0,
            volume=10.0,
            timeframe=Timeframe.M5,
        )


def test_tick_valid_invariants() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    tick = Tick(
        time=now,
        bid=2650.10,
        ask=2650.35,
        last=2650.20,
        volume=5.0,
    )
    assert tick.time == now
    assert tick.bid == 2650.10
    assert tick.ask == 2650.35
    assert pytest.approx(tick.spread) == 0.25


def test_tick_invalid_invariants() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="Ask cannot be less than bid"):
        Tick(time=now, bid=2650.50, ask=2650.10)

    with pytest.raises(ValueError, match="Prices must be positive"):
        Tick(time=now, bid=-1.0, ask=2650.0)

    with pytest.raises(ValueError, match="Volume cannot be negative"):
        Tick(time=now, bid=2650.0, ask=2650.50, volume=-2.0)

    naive_dt = datetime(2026, 9, 24, 12, 0)
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        Tick(time=naive_dt, bid=2650.0, ask=2650.50)


@pytest.mark.parametrize(
    "invalid_val",
    [float("nan"), float("inf"), float("-inf")],
)
def test_candle_rejects_non_finite_values(invalid_val: float) -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="must be finite"):
        Candle(time=now, open=invalid_val, high=2655.0, low=2648.0, close=2652.0)
    with pytest.raises(ValueError, match="must be finite"):
        Candle(time=now, open=2650.0, high=invalid_val, low=2648.0, close=2652.0)
    with pytest.raises(ValueError, match="must be finite"):
        Candle(time=now, open=2650.0, high=2655.0, low=invalid_val, close=2652.0)
    with pytest.raises(ValueError, match="must be finite"):
        Candle(time=now, open=2650.0, high=2655.0, low=2648.0, close=invalid_val)
    with pytest.raises(ValueError, match="must be finite"):
        Candle(time=now, open=2650.0, high=2655.0, low=2648.0, close=2652.0, volume=invalid_val)


@pytest.mark.parametrize(
    ("open_price", "high_price", "low_price", "close_price"),
    [
        (0.0, 2655.0, 2648.0, 2652.0),
        (-5.0, 2655.0, 2648.0, 2652.0),
        (2650.0, 0.0, 2648.0, 2652.0),
        (2650.0, 2655.0, 0.0, 2652.0),
        (2650.0, 2655.0, 2648.0, 0.0),
        (2650.0, 2655.0, 2648.0, -10.0),
    ],
)
def test_candle_rejects_non_positive_prices(
    open_price: float, high_price: float, low_price: float, close_price: float
) -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="Prices must be positive"):
        Candle(time=now, open=open_price, high=high_price, low=low_price, close=close_price)


@pytest.mark.parametrize(
    "invalid_val",
    [float("nan"), float("inf"), float("-inf")],
)
def test_tick_rejects_non_finite_values(invalid_val: float) -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="must be finite"):
        Tick(time=now, bid=invalid_val, ask=2650.50)
    with pytest.raises(ValueError, match="must be finite"):
        Tick(time=now, bid=2650.0, ask=invalid_val)
    with pytest.raises(ValueError, match="must be finite"):
        Tick(time=now, bid=2650.0, ask=2650.50, last=invalid_val)
    with pytest.raises(ValueError, match="must be finite"):
        Tick(time=now, bid=2650.0, ask=2650.50, volume=invalid_val)


def test_tick_rejects_negative_last() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="Last price cannot be negative"):
        Tick(time=now, bid=2650.0, ask=2650.50, last=-1.0)


def test_timeframe_durations() -> None:
    assert Timeframe.M1.duration_seconds == 60
    assert Timeframe.M5.duration_seconds == 300
    assert Timeframe.M15.duration_seconds == 900
    assert Timeframe.H1.duration_seconds == 3600
    assert Timeframe.H4.duration_seconds == 14400
    assert Timeframe.D1.duration_seconds == 86400


def test_candle_wicks() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    candle = Candle(
        time=now,
        open=2650.0,
        high=2655.0,
        low=2645.0,
        close=2652.0,
        volume=10.0,
    )
    assert candle.upper_wick == 3.0
    assert candle.lower_wick == 5.0


def test_swing_model() -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    swing = Swing(
        time=t1,
        price=2660.0,
        type=SwingType.HIGH,
        confirmed_time=t2,
        index=10,
    )
    assert swing.time == t1
    assert swing.price == 2660.0
    assert swing.type == SwingType.HIGH
    assert swing.confirmed_time == t2
    assert swing.index == 10


def test_swing_rejects_naive_datetimes() -> None:
    aware = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    naive = datetime(2026, 9, 24, 12, 0)
    with pytest.raises(ValueError, match="Swing timestamp must be timezone-aware"):
        Swing(time=naive, price=2660.0, type=SwingType.HIGH, confirmed_time=aware, index=0)

    with pytest.raises(ValueError, match="confirmed_time must be timezone-aware"):
        Swing(time=aware, price=2660.0, type=SwingType.HIGH, confirmed_time=naive, index=0)


def test_swing_rejects_confirmed_time_not_after_time() -> None:
    t1 = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="confirmed_time must be strictly after swing time"):
        Swing(time=t1, price=2660.0, type=SwingType.HIGH, confirmed_time=t2, index=0)

    with pytest.raises(ValueError, match="confirmed_time must be strictly after swing time"):
        Swing(time=t1, price=2660.0, type=SwingType.HIGH, confirmed_time=t1, index=0)


@pytest.mark.parametrize("invalid_price", [0.0, -10.0, float("nan"), float("inf"), float("-inf")])
def test_swing_rejects_invalid_price(invalid_price: float) -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    with pytest.raises(ValueError):
        Swing(time=t1, price=invalid_price, type=SwingType.HIGH, confirmed_time=t2, index=0)


def test_swing_rejects_negative_index() -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    with pytest.raises(ValueError, match="Index cannot be negative"):
        Swing(time=t1, price=2660.0, type=SwingType.HIGH, confirmed_time=t2, index=-1)


def test_swing_rejects_invalid_type() -> None:
    t1 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    t2 = datetime(2026, 9, 24, 12, 15, tzinfo=UTC)
    invalid_type = cast(Any, "INVALID")
    with pytest.raises(ValueError, match="Invalid swing type"):
        Swing(time=t1, price=2660.0, type=invalid_type, confirmed_time=t2, index=0)


def test_fvg_and_order_block_models() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = Candle(
        time=now,
        open=2650.0,
        high=2652.0,
        low=2649.0,
        close=2651.0,
        timeframe=Timeframe.M5,
    )
    c1 = Candle(
        time=now + timedelta(minutes=5),
        open=2651.0,
        high=2660.0,
        low=2651.0,
        close=2659.0,
        timeframe=Timeframe.M5,
    )
    c2 = Candle(
        time=now + timedelta(minutes=10),
        open=2659.0,
        high=2663.0,
        low=2655.0,
        close=2662.0,
        timeframe=Timeframe.M5,
    )

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
    assert fvg.top == 2655.0
    assert fvg.bottom == 2652.0
    assert fvg.mitigated is False

    ob = OrderBlock(
        time=c0.time,
        confirmed_time=c0.time + timedelta(minutes=5),
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
    )
    assert ob.type == OrderBlockType.BULLISH
    assert ob.mitigated is False


def test_order_intent_and_setup() -> None:
    now = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.10,
        price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        magic=1001,
        comment="xau_smc_setup_1",
    )
    assert intent.volume == 0.10
    assert intent.magic == 1001

    setup = Setup(
        id="setup_001",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=3.0,
        confluences=("SWEEP", "DISPLACEMENT", "FVG", "CHOCH"),
        valid_until=datetime(2026, 9, 24, 14, 0, tzinfo=UTC),
    )
    assert setup.risk_reward == 3.0
    assert setup.risk_reward_ratio == 3.0
    assert len(setup.confluences) == 4
