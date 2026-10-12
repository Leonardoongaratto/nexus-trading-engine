import math
from datetime import UTC, datetime

import pytest

from xau_bot.domain.models import Candle, Direction, Swing, SwingType
from xau_bot.risk.models import (
    OpenPosition,
    PositionRiskConfig,
    RiskRejectReason,
    SymbolRiskSpec,
)
from xau_bot.risk.protection import (
    check_breakeven,
    check_partial_close,
    check_smart_exit,
    check_trailing_stop,
)


@pytest.fixture
def spec() -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=10.0,  # 0.10 price units
        freeze_level_points=5.0,  # 0.05 price units
        commission_per_lot=5.0,  # $5 / 100oz = 0.05 price units
        spread_points=20.0,  # 0.20 price units
    )


@pytest.fixture
def buy_position() -> OpenPosition:
    return OpenPosition(
        ticket=1001,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2690.00,
        current_take_profit=2730.00,
        initial_stop_loss=2690.00,  # 10.00 distance = 1R
        initial_take_profit=2730.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def sell_position() -> OpenPosition:
    return OpenPosition(
        ticket=1002,
        symbol="XAUUSD",
        direction=Direction.BEARISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2710.00,
        current_take_profit=2670.00,
        initial_stop_loss=2710.00,  # 10.00 distance = 1R
        initial_take_profit=2670.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )


def test_breakeven_activation_at_1_5r_buy(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(
        be_trigger_r=1.5,
        be_offset_points=5.0,  # 0.05 extra points
    )

    res_under = check_breakeven(
        position=buy_position,
        current_price=2714.00,
        spec=spec,
        config=config,
    )
    assert res_under.should_modify is False
    assert res_under.current_r_multiple == 1.4

    res_hit = check_breakeven(
        position=buy_position,
        current_price=2715.00,
        spec=spec,
        config=config,
    )
    assert res_hit.should_modify is True
    assert res_hit.new_stop_loss == 2700.30
    assert res_hit.current_r_multiple == 1.5


def test_breakeven_activation_at_1_5r_sell(
    sell_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(
        be_trigger_r=1.5,
        be_offset_points=5.0,  # 0.05 extra points
    )

    res = check_breakeven(
        position=sell_position,
        current_price=2685.00,
        spec=spec,
        config=config,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2699.70
    assert res.current_r_multiple == 1.5


def test_breakeven_stops_level_violation(buy_position: OpenPosition) -> None:
    spec_large_stops = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=2000.0,  # $20.00 min distance
        freeze_level_points=5.0,
        commission_per_lot=5.0,
        spread_points=20.0,
    )
    config = PositionRiskConfig(be_trigger_r=1.5)
    res = check_breakeven(
        position=buy_position,
        current_price=2715.00,
        spec=spec_large_stops,
        config=config,
    )
    assert res.should_modify is False
    assert res.reject_reason == RiskRejectReason.STOPS_LEVEL_VIOLATION


def test_breakeven_freeze_level_violation(buy_position: OpenPosition) -> None:
    spec_large_freeze = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=10.0,
        freeze_level_points=3000.0,  # $30.00 freeze distance
        commission_per_lot=5.0,
        spread_points=20.0,
    )
    config = PositionRiskConfig(be_trigger_r=1.5)
    res = check_breakeven(
        position=buy_position,
        current_price=2715.00,
        spec=spec_large_freeze,
        config=config,
    )
    assert res.should_modify is False
    assert res.reject_reason == RiskRejectReason.FREEZE_LEVEL_VIOLATION


def test_breakeven_already_better_than_candidate(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1003,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2705.00,
        current_take_profit=2730.00,
        initial_stop_loss=2690.00,
        initial_take_profit=2730.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    config = PositionRiskConfig(be_trigger_r=1.5)
    res = check_breakeven(
        position=pos,
        current_price=2720.00,
        spec=spec,
        config=config,
    )
    assert res.should_modify is False
    assert "better than breakeven" in res.reason.lower()


def test_trailing_stop_atr_mode_buy(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=20.0,
    )
    res = check_trailing_stop(
        position=buy_position,
        current_price=2710.00,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2704.00
    assert res.delta_sl == 14.00


def test_trailing_stop_min_step_filtering(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1004,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2704.00,
        current_take_profit=2730.00,
        initial_stop_loss=2690.00,
        initial_take_profit=2730.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=20.0,  # 0.20 min step
    )
    res = check_trailing_stop(
        position=pos,
        current_price=2710.10,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res.should_modify is False
    assert math.isclose(res.delta_sl, 0.10, rel_tol=1e-5)
    assert "below minimum trailing step" in res.reason.lower()


def test_trailing_stop_structural_swing_mode(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1005,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2695.00,
        current_take_profit=2740.00,
        initial_stop_loss=2690.00,
        initial_take_profit=2740.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_low = Swing(
        time=t_swing,
        price=2708.50,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=10,
    )
    config = PositionRiskConfig(
        trailing_swing_buffer_points=5.0,
        min_trailing_step_points=10.0,
    )
    res = check_trailing_stop(
        position=pos,
        current_price=2715.00,
        spec=spec,
        config=config,
        structural_swing=swing_low,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2708.45
    assert math.isclose(res.delta_sl, 2708.45 - 2695.00, rel_tol=1e-5)


def test_breakeven_sell_stops_and_freeze_levels(sell_position: OpenPosition) -> None:
    spec_large_stops = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=2000.0,
        freeze_level_points=5.0,
        commission_per_lot=5.0,
        spread_points=20.0,
    )
    config = PositionRiskConfig(be_trigger_r=1.5)
    res_stop = check_breakeven(
        position=sell_position,
        current_price=2685.00,
        spec=spec_large_stops,
        config=config,
    )
    assert res_stop.should_modify is False
    assert res_stop.reject_reason == RiskRejectReason.STOPS_LEVEL_VIOLATION

    spec_large_freeze = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=10.0,
        freeze_level_points=3000.0,
        commission_per_lot=5.0,
        spread_points=20.0,
    )
    res_freeze = check_breakeven(
        position=sell_position,
        current_price=2685.00,
        spec=spec_large_freeze,
        config=config,
    )
    assert res_freeze.should_modify is False
    assert res_freeze.reject_reason == RiskRejectReason.FREEZE_LEVEL_VIOLATION


def test_breakeven_sell_already_better_than_candidate(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1006,
        symbol="XAUUSD",
        direction=Direction.BEARISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2695.00,
        current_take_profit=2670.00,
        initial_stop_loss=2710.00,
        initial_take_profit=2670.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    config = PositionRiskConfig(be_trigger_r=1.5)
    res = check_breakeven(
        position=pos,
        current_price=2680.00,
        spec=spec,
        config=config,
    )
    assert res.should_modify is False
    assert "better than breakeven" in res.reason.lower()


def test_check_breakeven_invalid_inputs(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(be_trigger_r=1.5)

    with pytest.raises(ValueError, match="current_price must be positive"):
        check_breakeven(buy_position, 0.0, spec, config)

    with pytest.raises(ValueError, match="spread cannot be negative"):
        check_breakeven(buy_position, 2715.0, spec, config, spread=-1.0)

    zero_risk_pos = OpenPosition(
        ticket=1007,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2700.00,
        current_take_profit=2730.00,
        initial_stop_loss=2700.00,
        initial_take_profit=2730.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="initial_risk must be strictly positive"):
        check_breakeven(zero_risk_pos, 2715.0, spec, config)


def test_trailing_stop_bearish_atr_and_swing(
    sell_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=20.0,
    )
    res_atr = check_trailing_stop(
        position=sell_position,
        current_price=2690.00,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res_atr.should_modify is True
    assert res_atr.new_stop_loss == 2696.00
    assert res_atr.delta_sl == 14.00

    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_high = Swing(
        time=t_swing,
        price=2694.00,
        type=SwingType.HIGH,
        confirmed_time=t_conf,
        index=10,
    )
    config_offset = PositionRiskConfig(
        be_offset_points=5.0,
        min_trailing_step_points=10.0,
    )
    res_swing = check_trailing_stop(
        position=sell_position,
        current_price=2685.00,
        spec=spec,
        config=config_offset,
        structural_swing=swing_high,
    )
    assert res_swing.should_modify is True
    assert res_swing.new_stop_loss == 2694.25


def test_trailing_stop_bearish_both_modes(
    sell_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_high = Swing(
        time=t_swing,
        price=2695.00,
        type=SwingType.HIGH,
        confirmed_time=t_conf,
        index=10,
    )
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        be_offset_points=5.0,
        min_trailing_step_points=10.0,
    )
    res = check_trailing_stop(
        position=sell_position,
        current_price=2685.00,
        spec=spec,
        config=config,
        atr=3.00,
        structural_swing=swing_high,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2691.00


def test_trailing_stop_bearish_min_step_and_ratchet(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1008,
        symbol="XAUUSD",
        direction=Direction.BEARISH,
        volume=0.10,
        entry_price=2700.00,
        current_stop_loss=2696.00,
        current_take_profit=2670.00,
        initial_stop_loss=2710.00,
        initial_take_profit=2670.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=20.0,  # 0.20
    )
    res_step = check_trailing_stop(
        position=pos,
        current_price=2689.95,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res_step.should_modify is False
    assert "below minimum trailing step" in res_step.reason.lower()

    res_ratchet = check_trailing_stop(
        position=pos,
        current_price=2692.00,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res_ratchet.should_modify is False
    assert "does not advance" in res_ratchet.reason.lower()


def test_trailing_stop_sell_stops_and_freeze_levels(sell_position: OpenPosition) -> None:
    spec_large_stops = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=2000.0,  # $20.00
        freeze_level_points=5.0,
    )
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=10.0,
    )
    res_stop = check_trailing_stop(
        position=sell_position,
        current_price=2685.00,
        spec=spec_large_stops,
        config=config,
        atr=3.00,
    )
    assert res_stop.should_modify is False
    assert res_stop.reject_reason == RiskRejectReason.STOPS_LEVEL_VIOLATION

    spec_large_freeze = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=10.0,
        freeze_level_points=3000.0,  # $30.00
    )
    res_freeze = check_trailing_stop(
        position=sell_position,
        current_price=2685.00,
        spec=spec_large_freeze,
        config=config,
        atr=3.00,
    )
    assert res_freeze.should_modify is False
    assert res_freeze.reject_reason == RiskRejectReason.FREEZE_LEVEL_VIOLATION


def test_check_trailing_stop_invalid_inputs(
    buy_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig()

    with pytest.raises(ValueError, match="current_price must be positive"):
        check_trailing_stop(buy_position, 0.0, spec, config, atr=3.0)

    with pytest.raises(ValueError, match="Either atr or structural_swing must be provided"):
        check_trailing_stop(buy_position, 2710.0, spec, config)

    with pytest.raises(ValueError, match="spread cannot be negative"):
        check_trailing_stop(buy_position, 2710.0, spec, config, atr=3.0, spread=-1.0)

    with pytest.raises(ValueError, match="atr must be positive"):
        check_trailing_stop(buy_position, 2710.0, spec, config, atr=0.0)

    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_high = Swing(
        time=t_swing,
        price=2708.50,
        type=SwingType.HIGH,
        confirmed_time=t_conf,
        index=10,
    )
    with pytest.raises(ValueError, match="Bullish trailing requires a Swing LOW"):
        check_trailing_stop(buy_position, 2710.0, spec, config, structural_swing=swing_high)

    swing_low = Swing(
        time=t_swing,
        price=2690.00,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=10,
    )
    with pytest.raises(ValueError, match="Bearish trailing requires a Swing HIGH"):
        check_trailing_stop(
            OpenPosition(
                ticket=1009,
                symbol="XAUUSD",
                direction=Direction.BEARISH,
                volume=0.10,
                entry_price=2700.0,
                current_stop_loss=2710.0,
                current_take_profit=2670.0,
                initial_stop_loss=2710.0,
                initial_take_profit=2670.0,
                open_time=t_swing,
            ),
            2685.0,
            spec,
            config,
            structural_swing=swing_low,
        )


def test_trailing_stop_bullish_ratchet_and_both_modes(
    buy_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        min_trailing_step_points=10.0,
    )
    res_ratchet = check_trailing_stop(
        position=buy_position,
        current_price=2695.00,
        spec=spec,
        config=config,
        atr=3.00,
    )
    assert res_ratchet.should_modify is False
    assert "does not advance" in res_ratchet.reason.lower()

    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_low = Swing(
        time=t_swing,
        price=2704.00,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=10,
    )
    config_both = PositionRiskConfig(
        trailing_atr_multiplier=2.0,
        be_offset_points=5.0,
        min_trailing_step_points=10.0,
    )
    res_both = check_trailing_stop(
        position=buy_position,
        current_price=2715.00,
        spec=spec,
        config=config_both,
        atr=3.00,
        structural_swing=swing_low,
        spread=0.15,
    )
    assert res_both.should_modify is True
    assert res_both.new_stop_loss == 2709.00


def test_breakeven_explicit_spread(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(be_trigger_r=1.5, be_offset_points=0.0)
    res = check_breakeven(
        position=buy_position,
        current_price=2715.00,
        spec=spec,
        config=config,
        spread=0.30,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2700.35


def test_breakeven_directional_rounding_covers_fractional_costs(
    buy_position: OpenPosition, sell_position: OpenPosition
) -> None:
    # Commission $0.40/lot on 100oz contract = $0.004 price units.
    # Ceil rounds Buy SL to 2700.01; Floor rounds Sell SL to 2699.99 to fully cover costs.
    spec_fractional = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=0.40,
        spread_points=0.0,
    )
    config = PositionRiskConfig(be_trigger_r=1.5, be_offset_points=0.0)

    res_buy = check_breakeven(
        position=buy_position,
        current_price=2715.00,
        spec=spec_fractional,
        config=config,
        spread=0.0,
    )
    assert res_buy.should_modify is True
    assert res_buy.new_stop_loss == 2700.01

    res_sell = check_breakeven(
        position=sell_position,
        current_price=2685.00,
        spec=spec_fractional,
        config=config,
        spread=0.0,
    )
    assert res_sell.should_modify is True
    assert res_sell.new_stop_loss == 2699.99


def test_trailing_stop_buy_stops_level_violation(
    buy_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    spec_stops = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=50.0,
        freeze_level_points=5.0,
    )
    config = PositionRiskConfig(trailing_atr_multiplier=0.1, min_trailing_step_points=1.0)
    res = check_trailing_stop(
        position=buy_position,
        current_price=2715.00,
        spec=spec_stops,
        config=config,
        atr=2.00,
    )
    assert res.should_modify is False
    assert res.reject_reason == RiskRejectReason.STOPS_LEVEL_VIOLATION


def test_trailing_stop_buy_freeze_level_violation(
    buy_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    spec_freeze = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=0.0,
        freeze_level_points=5.0,
    )
    config = PositionRiskConfig(trailing_atr_multiplier=1.0, min_trailing_step_points=0.5)
    res = check_trailing_stop(
        position=buy_position,
        current_price=2690.04,
        spec=spec_freeze,
        config=config,
        atr=0.01,
    )
    assert res.should_modify is False
    assert res.reject_reason == RiskRejectReason.FREEZE_LEVEL_VIOLATION


def test_trailing_stop_decoupled_swing_buffer(
    buy_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 45, tzinfo=UTC)
    swing_low = Swing(
        time=t_swing,
        price=2705.00,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=10,
    )
    config = PositionRiskConfig(
        be_offset_points=20.0,
        trailing_swing_buffer_points=3.0,
        min_trailing_step_points=1.0,
    )
    res = check_trailing_stop(
        position=buy_position,
        current_price=2715.00,
        spec=spec,
        config=config,
        structural_swing=swing_low,
    )
    assert res.should_modify is True
    assert res.new_stop_loss == 2704.97


def test_partial_close_buy_triggers_at_1r(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(partial_close_trigger_r=1.0, partial_close_ratio=0.5)
    # Entry=2700, SL=2690 -> 1R = 10.0 pts -> Price=2710.0
    res = check_partial_close(
        position=buy_position,
        current_price=2710.00,
        spec=spec,
        config=config,
    )
    assert res.should_close is True
    assert res.volume_to_close == 0.05
    assert res.remaining_volume == 0.05
    assert res.current_r_multiple == 1.0


def test_partial_close_sell_triggers_at_1r(
    sell_position: OpenPosition, spec: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(partial_close_trigger_r=1.0, partial_close_ratio=0.5)
    # Entry=2700, SL=2710 -> 1R = 10.0 pts -> Price=2690.0
    res = check_partial_close(
        position=sell_position,
        current_price=2690.00,
        spec=spec,
        config=config,
    )
    assert res.should_close is True
    assert res.volume_to_close == 0.05
    assert res.remaining_volume == 0.05
    assert res.current_r_multiple == 1.0


def test_partial_close_under_trigger_r(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(partial_close_trigger_r=1.0, partial_close_ratio=0.5)
    res = check_partial_close(
        position=buy_position,
        current_price=2708.00,  # 0.8R
        spec=spec,
        config=config,
    )
    assert res.should_close is False
    assert res.current_r_multiple == 0.8


def test_partial_close_already_closed(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(partial_close_trigger_r=1.0, partial_close_ratio=0.5)
    res = check_partial_close(
        position=buy_position,
        current_price=2715.00,
        spec=spec,
        config=config,
        already_partially_closed=True,
    )
    assert res.should_close is False
    assert "already partially closed" in res.reason


def test_partial_close_disabled(buy_position: OpenPosition, spec: SymbolRiskSpec) -> None:
    config = PositionRiskConfig(partial_close_enabled=False)
    res = check_partial_close(
        position=buy_position,
        current_price=2720.00,
        spec=spec,
        config=config,
    )
    assert res.should_close is False
    assert "disabled" in res.reason


def test_partial_close_below_min_volume(spec: SymbolRiskSpec) -> None:
    pos = OpenPosition(
        ticket=1003,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.01,  # minimum lot
        entry_price=2700.00,
        current_stop_loss=2690.00,
        current_take_profit=2730.00,
        initial_stop_loss=2690.00,
        initial_take_profit=2730.00,
        open_time=datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
    )
    config = PositionRiskConfig(partial_close_trigger_r=1.0, partial_close_ratio=0.5)
    res = check_partial_close(
        position=pos,
        current_price=2715.00,
        spec=spec,
        config=config,
    )
    assert res.should_close is False
    assert res.reject_reason == RiskRejectReason.LOT_BELOW_MINIMUM


def test_smart_exit_buy_opposite_choch(buy_position: OpenPosition) -> None:
    config = PositionRiskConfig(smart_exit_enabled=True, smart_exit_min_r=0.5)
    t_swing = datetime(2026, 9, 25, 12, 20, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 35, tzinfo=UTC)
    # A confirmed swing low formed after buy position was opened
    swing_low = Swing(
        time=t_swing,
        price=2706.00,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=12,
    )
    # Candle that closes below the swing low
    c_rev = Candle(
        time=datetime(2026, 9, 25, 12, 40, tzinfo=UTC),
        open=2707.00,
        high=2707.50,
        low=2704.00,
        close=2704.50,  # below 2706.00
        volume=100.0,
    )
    res = check_smart_exit(
        position=buy_position,
        current_price=2705.00,  # still in profit (entry 2700, gain 5.0 = 0.5R)
        recent_candles=[c_rev],
        swings=[swing_low],
        config=config,
    )
    assert res.should_exit is True
    assert res.reversal_type == "ChoCH_BEARISH"
    assert "Opposite ChoCH" in res.reason


def test_smart_exit_sell_opposite_choch(sell_position: OpenPosition) -> None:
    config = PositionRiskConfig(smart_exit_enabled=True, smart_exit_min_r=0.5)
    t_swing = datetime(2026, 9, 25, 12, 20, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 35, tzinfo=UTC)
    # Confirmed swing high after sell position
    swing_high = Swing(
        time=t_swing,
        price=2694.00,
        type=SwingType.HIGH,
        confirmed_time=t_conf,
        index=12,
    )
    # Candle closes above the swing high
    c_rev = Candle(
        time=datetime(2026, 9, 25, 12, 40, tzinfo=UTC),
        open=2693.00,
        high=2696.00,
        low=2692.50,
        close=2695.50,  # above 2694.00
        volume=100.0,
    )
    res = check_smart_exit(
        position=sell_position,
        current_price=2695.00,  # entry 2700, gain 5.0 = 0.5R
        recent_candles=[c_rev],
        swings=[swing_high],
        config=config,
    )
    assert res.should_exit is True
    assert res.reversal_type == "ChoCH_BULLISH"


def test_smart_exit_no_reversal(buy_position: OpenPosition) -> None:
    config = PositionRiskConfig(smart_exit_enabled=True, smart_exit_min_r=0.5)
    t_swing = datetime(2026, 9, 25, 12, 20, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 35, tzinfo=UTC)
    swing_low = Swing(
        time=t_swing,
        price=2705.00,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=12,
    )
    # Candle stays above swing low
    c_bull = Candle(
        time=datetime(2026, 9, 25, 12, 40, tzinfo=UTC),
        open=2708.00,
        high=2712.00,
        low=2707.00,
        close=2711.00,
        volume=100.0,
    )
    res = check_smart_exit(
        position=buy_position,
        current_price=2711.00,
        recent_candles=[c_bull],
        swings=[swing_low],
        config=config,
    )
    assert res.should_exit is False
    assert "intact" in res.reason


def test_smart_exit_min_r_not_reached(buy_position: OpenPosition) -> None:
    config = PositionRiskConfig(smart_exit_enabled=True, smart_exit_min_r=0.5)
    # Price is at 2702.00 (gain 2.0 = 0.2R < 0.5R)
    res = check_smart_exit(
        position=buy_position,
        current_price=2702.00,
        recent_candles=[],
        swings=[],
        config=config,
    )
    assert res.should_exit is False
    assert "below smart_exit_min_r" in res.reason


def test_bitcoin_scale_partial_close_and_smart_exit() -> None:
    btc_spec = SymbolRiskSpec(
        symbol="BTCUSD",
        contract_size=1.0,
        tick_size=0.01,
        tick_value=0.01,
        lot_min=0.01,
        lot_max=50.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=100.0,
        freeze_level_points=50.0,
        commission_per_lot=0.0,
        spread_points=150.0,  # $1.50 spread
    )
    t_open = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    btc_pos = OpenPosition(
        ticket=5001,
        symbol="BTCUSD",
        direction=Direction.BULLISH,
        volume=0.20,
        entry_price=65000.0,
        current_stop_loss=64500.0,
        current_take_profit=66500.0,
        initial_stop_loss=64500.0,  # 500.0 distance = 1R
        initial_take_profit=66500.0,
        open_time=t_open,
    )
    config = PositionRiskConfig(
        partial_close_trigger_r=1.0,
        partial_close_ratio=0.5,
        smart_exit_enabled=True,
        smart_exit_min_r=0.5,
    )

    # 1. 1R reached at 65500.0 -> 50% partial (0.10 lot)
    res_pc = check_partial_close(btc_pos, current_price=65500.0, spec=btc_spec, config=config)
    assert res_pc.should_close is True
    assert res_pc.volume_to_close == 0.10
    assert res_pc.remaining_volume == 0.10
    assert res_pc.current_r_multiple == 1.0

    # 2. Smart exit on bearish ChoCH after partial (above smart_exit_min_r=0.5R)
    t_swing = datetime(2026, 9, 25, 12, 30, tzinfo=UTC)
    t_conf = datetime(2026, 9, 25, 12, 35, tzinfo=UTC)
    swing_low = Swing(
        time=t_swing,
        price=65300.0,
        type=SwingType.LOW,
        confirmed_time=t_conf,
        index=10,
    )
    c_bear = Candle(
        time=datetime(2026, 9, 25, 12, 40, tzinfo=UTC),
        open=65350.0,
        high=65380.0,
        low=65240.0,
        close=65260.0,  # Closes below swing low, gain = 260 pts = 0.52R > 0.5R
        volume=50.0,
    )
    res_se = check_smart_exit(
        position=btc_pos,
        current_price=65260.0,
        recent_candles=[c_bear],
        swings=[swing_low],
        config=config,
    )
    assert res_se.should_exit is True
    assert res_se.reversal_type == "ChoCH_BEARISH"
