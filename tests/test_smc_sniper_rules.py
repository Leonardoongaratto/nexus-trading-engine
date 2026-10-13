from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from xau_bot.domain.models import (
    Candle,
    ChoCH,
    Direction,
    LiquidityPool,
    LiquiditySweep,
    LiquidityType,
    OrderBlock,
    OrderBlockType,
    SetupTriggerType,
    Swing,
    SwingType,
    Timeframe,
)
from xau_bot.smc.setup import detect_setups


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


def test_premium_discount_filter_blocks_sell_in_discount() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    # Range total: high em c2 (2678.0), low em c5 (2620.0) -> Equilibrium = 2649.0
    # Ob em c4 com low=2642, high=2650 -> mean_threshold = 2646.0 (< 2649.0, Discount)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2653.0, base_time=t0)
    c1 = make_candle(1, open=2653.0, high=2675.0, low=2652.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2664.0, high=2678.0, low=2660.0, close=2670.0, base_time=t0)
    c3 = make_candle(3, open=2670.0, high=2671.0, low=2648.0, close=2649.0, base_time=t0)
    c4 = make_candle(4, open=2649.0, high=2650.0, low=2642.0, close=2644.0, base_time=t0)
    c5 = make_candle(5, open=2644.0, high=2645.0, low=2620.0, close=2622.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5]

    s_high_1 = Swing(
        time=c1.time,
        price=2675.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_bsl = LiquidityPool(
        type=LiquidityType.BSL,
        price=2675.0,
        swings=(s_high_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_bsl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2675.0,
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
        time=c5.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        direction=Direction.BEARISH,
        previous_trend=Direction.BULLISH,
        broken_swing=s_low_0,
        breaking_candle=c5,
        price=2645.0,
    )
    # OB na faixa 2642 - 2650 (mean threshold 2646.0, que está em Discount < 2649.0)
    ob_discount = OrderBlock(
        time=c4.time,
        confirmed_time=choch_bear.confirmed_time,
        type=OrderBlockType.BEARISH,
        high=c4.high,
        low=c4.low,
        open=c4.open,
        close=c4.close,
        candle=c4,
        displacement_atr_ratio=2.0,
        breaking_event=choch_bear,
    )
    target_low = Swing(
        time=c0.time,
        price=2550.0,
        type=SwingType.LOW,
        confirmed_time=c0.time + timedelta(minutes=5),
        index=0,
    )

    setups_blocked = detect_setups(
        candles,
        swings=[s_high_1, s_low_0, target_low],
        structure_events=[choch_bear],
        sweeps=[sweep_bsl],
        order_blocks=[ob_discount],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=True,
    )
    assert len(setups_blocked) == 0

    setups_allowed = detect_setups(
        candles,
        swings=[s_high_1, s_low_0, target_low],
        structure_events=[choch_bear],
        sweeps=[sweep_bsl],
        order_blocks=[ob_discount],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
    )
    assert len(setups_allowed) == 1
    assert setups_allowed[0].direction == Direction.BEARISH


def test_premium_discount_filter_blocks_buy_in_premium() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    # Range total: low em c2 (2620.0), high em c5 (2680.0) -> Equilibrium = 2650.0
    # Ob em c4 com low=2650, high=2658 -> mean_threshold = 2654.0 (> 2650.0, Premium)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2647.0, base_time=t0)
    c1 = make_candle(1, open=2647.0, high=2648.0, low=2625.0, close=2636.0, base_time=t0)
    c2 = make_candle(2, open=2636.0, high=2640.0, low=2620.0, close=2626.0, base_time=t0)
    c3 = make_candle(3, open=2626.0, high=2652.0, low=2625.0, close=2651.0, base_time=t0)
    c4 = make_candle(4, open=2651.0, high=2658.0, low=2650.0, close=2656.0, base_time=t0)
    c5 = make_candle(5, open=2656.0, high=2680.0, low=2655.0, close=2678.0, base_time=t0)
    candles = [c0, c1, c2, c3, c4, c5]

    s_low_1 = Swing(
        time=c1.time,
        price=2625.0,
        type=SwingType.LOW,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2625.0,
        swings=(s_low_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2625.0,
        candle=c2,
        pool=pool_ssl,
    )
    s_high_0 = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    choch_bull = ChoCH(
        time=c5.time,
        confirmed_time=c5.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c5,
        price=2655.0,
    )
    # OB na faixa 2650 - 2658 (mean threshold 2654.0, que está em Premium > 2650.0)
    ob_premium = OrderBlock(
        time=c4.time,
        confirmed_time=choch_bull.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c4.high,
        low=c4.low,
        open=c4.open,
        close=c4.close,
        candle=c4,
        displacement_atr_ratio=2.0,
        breaking_event=choch_bull,
    )
    target_high = Swing(
        time=c0.time,
        price=2750.0,
        type=SwingType.HIGH,
        confirmed_time=c0.time + timedelta(minutes=5),
        index=0,
    )

    setups_blocked = detect_setups(
        candles,
        swings=[s_low_1, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        order_blocks=[ob_premium],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=True,
    )
    assert len(setups_blocked) == 0

    setups_allowed = detect_setups(
        candles,
        swings=[s_low_1, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        order_blocks=[ob_premium],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
    )
    assert len(setups_allowed) == 1
    assert setups_allowed[0].direction == Direction.BULLISH


def test_sweep_guard_blocks_sell_on_absorption_wick() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    # Candle c2 (sweep): open=2663, high=2668, low=2645, close=2664
    # Range = 23.0, Pavio inferior = min(2663, 2664) - 2645 = 18.0 (78% do range!)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2653.0, base_time=t0)
    c1 = make_candle(1, open=2653.0, high=2665.0, low=2652.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2663.0, high=2668.0, low=2645.0, close=2664.0, base_time=t0)
    c3 = make_candle(3, open=2664.0, high=2665.0, low=2640.0, close=2642.0, base_time=t0)
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

    setups_blocked = detect_setups(
        candles,
        swings=[s_high_1, s_low_0, target_low],
        structure_events=[choch_bear],
        sweeps=[sweep_bsl],
        order_blocks=[ob_bear],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
        require_sweep_guard=True,
    )
    assert len(setups_blocked) == 0

    setups_allowed = detect_setups(
        candles,
        swings=[s_high_1, s_low_0, target_low],
        structure_events=[choch_bear],
        sweeps=[sweep_bsl],
        order_blocks=[ob_bear],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
        require_sweep_guard=False,
    )
    assert len(setups_allowed) == 1


def test_sweep_guard_blocks_buy_on_upper_rejection_wick() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    # Candle c2 (sweep SSL): open=2636, high=2655, low=2630, close=2637
    # Range = 25.0, Pavio superior = 2655 - max(2636, 2637) = 18.0 (72% do range!)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2647.0, base_time=t0)
    c1 = make_candle(1, open=2647.0, high=2648.0, low=2635.0, close=2636.0, base_time=t0)
    c2 = make_candle(2, open=2636.0, high=2655.0, low=2630.0, close=2637.0, base_time=t0)
    c3 = make_candle(3, open=2637.0, high=2670.0, low=2636.0, close=2668.0, base_time=t0)
    candles = [c0, c1, c2, c3]

    s_low_1 = Swing(
        time=c1.time,
        price=2635.0,
        type=SwingType.LOW,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_ssl = LiquidityPool(
        type=LiquidityType.SSL,
        price=2635.0,
        swings=(s_low_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_ssl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.SSL,
        level=2635.0,
        candle=c2,
        pool=pool_ssl,
    )
    s_high_0 = Swing(
        time=c0.time,
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=c1.time + timedelta(minutes=5),
        index=0,
    )
    choch_bull = ChoCH(
        time=c3.time,
        confirmed_time=c3.time + timedelta(minutes=5),
        direction=Direction.BULLISH,
        previous_trend=Direction.BEARISH,
        broken_swing=s_high_0,
        breaking_candle=c3,
        price=2655.0,
    )
    ob_bull = OrderBlock(
        time=c2.time,
        confirmed_time=choch_bull.confirmed_time,
        type=OrderBlockType.BULLISH,
        high=c2.high,
        low=c2.low,
        open=c2.open,
        close=c2.close,
        candle=c2,
        displacement_atr_ratio=2.0,
        breaking_event=choch_bull,
    )
    target_high = Swing(
        time=c0.time,
        price=2700.0,
        type=SwingType.HIGH,
        confirmed_time=c0.time + timedelta(minutes=5),
        index=0,
    )

    setups_blocked = detect_setups(
        candles,
        swings=[s_low_1, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        order_blocks=[ob_bull],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
        require_sweep_guard=True,
    )
    assert len(setups_blocked) == 0

    setups_allowed = detect_setups(
        candles,
        swings=[s_low_1, s_high_0, target_high],
        structure_events=[choch_bull],
        sweeps=[sweep_ssl],
        order_blocks=[ob_bull],
        trigger_type=SetupTriggerType.OB_MEAN_THRESHOLD,
        require_premium_discount=False,
        require_sweep_guard=False,
    )
    assert len(setups_allowed) == 1
    assert setups_allowed[0].direction == Direction.BULLISH


def test_confluences_include_institutional_tags() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    # Cenário de venda em Premium com sweep limpo (sem pavio de absorção)
    # Range: High=2680 (sweep em c2), Low=2640 (c3) -> Equilibrium = 2660.0
    # Entry OB em c2 (2668.0 > 2660.0 -> Premium!)
    # c2: open=2664, high=2680, low=2662, close=2672
    # Pavio inferior = min(2664, 2672) - 2662 = 2.0 / 18.0 = 11% (bem abaixo de 40%)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2653.0, base_time=t0)
    c1 = make_candle(1, open=2653.0, high=2675.0, low=2652.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2664.0, high=2680.0, low=2662.0, close=2672.0, base_time=t0)
    c3 = make_candle(3, open=2672.0, high=2673.0, low=2640.0, close=2642.0, base_time=t0)
    candles = [c0, c1, c2, c3]

    s_high_1 = Swing(
        time=c1.time,
        price=2675.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_bsl = LiquidityPool(
        type=LiquidityType.BSL,
        price=2675.0,
        swings=(s_high_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_bsl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2675.0,
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
        price=2600.0,
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
        require_premium_discount=True,
        require_sweep_guard=True,
    )
    assert len(setups) == 1
    s = setups[0]
    assert "PREMIUM_ALIGNED" in s.confluences
    assert "SWEEP_GUARD_CONFIRMED" in s.confluences


def test_sweep_guard_zero_range_candle_not_tagged() -> None:
    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    c0 = make_candle(0, open=2650.0, high=2655.0, low=2645.0, close=2653.0, base_time=t0)
    c1 = make_candle(1, open=2653.0, high=2675.0, low=2652.0, close=2664.0, base_time=t0)
    c2 = make_candle(2, open=2664.0, high=2680.0, low=2662.0, close=2672.0, base_time=t0)
    c3 = make_candle(3, open=2672.0, high=2673.0, low=2640.0, close=2642.0, base_time=t0)
    candles = [c0, c1, c2, c3]

    s_high_1 = Swing(
        time=c1.time,
        price=2675.0,
        type=SwingType.HIGH,
        confirmed_time=c2.time + timedelta(minutes=5),
        index=1,
    )
    pool_bsl = LiquidityPool(
        type=LiquidityType.BSL,
        price=2675.0,
        swings=(s_high_1,),
        time=c1.time,
        confirmed_time=c2.time + timedelta(minutes=5),
    )
    sweep_bsl = LiquiditySweep(
        time=c2.time,
        confirmed_time=c2.time + timedelta(minutes=5),
        type=LiquidityType.BSL,
        level=2675.0,
        candle=c2,
        pool=pool_bsl,
    )
    # Simula candle de sweep com amplitude zero
    zero_range_candle = make_candle(
        2, open=2670.0, high=2670.0, low=2670.0, close=2670.0, base_time=t0
    )
    object.__setattr__(sweep_bsl, "candle", zero_range_candle)

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
        price=2600.0,
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
        require_premium_discount=False,
        require_sweep_guard=True,
    )
    assert len(setups) == 1
    # Quando range é zero, o sweep guard não valida pavio e NÃO emite SWEEP_GUARD_CONFIRMED
    assert "SWEEP_GUARD_CONFIRMED" not in setups[0].confluences


@pytest.mark.asyncio
async def test_live_engine_htf_hierarchy_and_settings_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from xau_bot.advisor.modes import AdvisorDispatcher, ProposalManager
    from xau_bot.broker.models import SymbolInfo
    from xau_bot.config import Settings
    from xau_bot.runtime.candle_loop import CandleLoop
    from xau_bot.runtime.live_loop import LiveTradingEngine
    from xau_bot.runtime.models import CandleCloseEvent

    settings = Settings(
        smc_require_premium_discount=True,
        smc_require_sweep_guard=True,
        smc_sweep_guard_max_rejection_ratio=0.40,
        smc_htf_alignment_required=True,
    )
    broker = AsyncMock()
    broker.get_symbol_info.return_value = SymbolInfo(
        name="XAUUSD",
        bid=2650.0,
        ask=2650.25,
        spread_points=0.25,
        point_size=0.01,
        contract_size=100.0,
        min_volume=0.01,
        max_volume=100.0,
        volume_step=0.01,
        stops_level_points=1.0,
        freeze_level_points=0.5,
        tick_size=0.01,
        tick_value=1.0,
    )
    proposal_mgr = ProposalManager()
    advisor_disp = AdvisorDispatcher(proposal_manager=proposal_mgr)
    candle_loop = CandleLoop(timeframes=(Timeframe.M5, Timeframe.H1, Timeframe.H4, Timeframe.D1))

    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_mgr,
        advisor_dispatcher=advisor_disp,
        candle_loop=candle_loop,
        symbol="XAUUSD",
        trigger_timeframe=Timeframe.M5,
        require_premium_discount=settings.smc_require_premium_discount,
        require_sweep_guard=settings.smc_require_sweep_guard,
        sweep_guard_max_rejection_ratio=settings.smc_sweep_guard_max_rejection_ratio,
        htf_alignment_required=settings.smc_htf_alignment_required,
    )

    t0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
    m5_candles = [
        make_candle(
            i,
            open=2640.0 + i * 0.1,
            high=2642.0 + i * 0.1,
            low=2639.0 + i * 0.1,
            close=2641.0 + i * 0.1,
            base_time=t0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    c_h1 = make_candle(
        0, open=2640.0, high=2660.0, low=2638.0, close=2655.0, base_time=t0, timeframe=Timeframe.H1
    )
    c_h4 = make_candle(
        0, open=2630.0, high=2670.0, low=2625.0, close=2665.0, base_time=t0, timeframe=Timeframe.H4
    )
    c_d1 = make_candle(
        0, open=2600.0, high=2700.0, low=2590.0, close=2680.0, base_time=t0, timeframe=Timeframe.D1
    )

    # 1. Caso A: H1 disponível -> H1 deve ser selecionado como htf_candles
    candle_loop._closed_history[Timeframe.M5] = list(m5_candles)
    candle_loop._closed_history[Timeframe.H1] = [c_h1]
    candle_loop._closed_history[Timeframe.H4] = [c_h4]
    candle_loop._closed_history[Timeframe.D1] = [c_d1]

    captured_kwargs: dict[str, Any] = {}
    import xau_bot.runtime.live_loop as live_loop_mod

    def mock_detect(*args: Any, **kwargs: Any) -> list[Any]:
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(live_loop_mod, "detect_setups", mock_detect)

    last_m5 = m5_candles[-1]
    evt = CandleCloseEvent(
        timeframe=Timeframe.M5,
        candle=last_m5,
        closed_series=tuple(m5_candles),
        forming_candle=None,
        closed_at=last_m5.time + timedelta(minutes=5),
    )
    await engine.on_candle_close(evt)

    # Verifica se os flags de Settings foram repassados a detect_setups
    assert captured_kwargs["require_premium_discount"] is True
    assert captured_kwargs["require_sweep_guard"] is True
    assert captured_kwargs["sweep_guard_max_rejection_ratio"] == 0.40
    assert captured_kwargs["htf_alignment_required"] is True
    # E que htf_candles recebeu a série de H1 (duração 3600s)
    htf_used = captured_kwargs.get("htf_candles")
    assert htf_used is not None
    assert htf_used[0].timeframe == Timeframe.H1

    # 2. Caso B: Sem H1, mas com H4 -> H4 deve ser selecionado
    candle_loop_no_h1 = CandleLoop(timeframes=(Timeframe.M5, Timeframe.H4, Timeframe.D1))
    candle_loop_no_h1._closed_history[Timeframe.M5] = list(m5_candles)
    candle_loop_no_h1._closed_history[Timeframe.H4] = [c_h4]
    candle_loop_no_h1._closed_history[Timeframe.D1] = [c_d1]
    engine.candle_loop = candle_loop_no_h1

    captured_kwargs.clear()
    await engine.on_candle_close(evt)
    htf_used_b = captured_kwargs.get("htf_candles")
    assert htf_used_b is not None
    assert htf_used_b[0].timeframe == Timeframe.H4

    # 3. Caso C: Sem H1 e sem H4, com D1 -> D1 deve ser selecionado
    candle_loop_d1_only = CandleLoop(timeframes=(Timeframe.M5, Timeframe.D1))
    candle_loop_d1_only._closed_history[Timeframe.M5] = list(m5_candles)
    candle_loop_d1_only._closed_history[Timeframe.D1] = [c_d1]
    engine.candle_loop = candle_loop_d1_only

    captured_kwargs.clear()
    await engine.on_candle_close(evt)
    htf_used_c = captured_kwargs.get("htf_candles")
    assert htf_used_c is not None
    assert htf_used_c[0].timeframe == Timeframe.D1
