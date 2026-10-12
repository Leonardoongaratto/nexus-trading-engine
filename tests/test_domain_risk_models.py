from datetime import UTC, datetime
from typing import Any, cast

import pytest

from xau_bot.domain.models import Direction
from xau_bot.risk.models import (
    AccountState,
    BreakevenResult,
    LotCalculationResult,
    MarginValidationResult,
    OpenPosition,
    PositionRiskConfig,
    SymbolRiskSpec,
    TrailingStopResult,
)


def test_symbol_risk_spec_valid() -> None:
    spec = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        stops_level_points=10.0,
        freeze_level_points=5.0,
        commission_per_lot=5.0,
        spread_points=15.0,
    )
    assert spec.symbol == "XAUUSD"
    assert spec.contract_size == 100.0
    assert spec.tick_size == 0.01
    assert spec.tick_value == 1.0
    assert spec.lot_min == 0.01
    assert spec.lot_max == 100.0
    assert spec.lot_step == 0.01


@pytest.mark.parametrize(
    "field,invalid_val",
    [
        ("contract_size", 0.0),
        ("contract_size", -100.0),
        ("contract_size", float("nan")),
        ("tick_size", 0.0),
        ("tick_size", float("inf")),
        ("tick_value", 0.0),
        ("tick_value", -1.0),
        ("lot_min", 0.0),
        ("lot_max", 0.0),
        ("lot_step", 0.0),
        ("point", 0.0),
        ("stops_level_points", -1.0),
        ("freeze_level_points", -1.0),
        ("commission_per_lot", -5.0),
        ("spread_points", -1.0),
    ],
)
def test_symbol_risk_spec_invariants(field: str, invalid_val: float) -> None:
    valid_args: dict[str, Any] = {
        "symbol": "XAUUSD",
        "contract_size": 100.0,
        "tick_size": 0.01,
        "tick_value": 1.0,
        "lot_min": 0.01,
        "lot_max": 100.0,
        "lot_step": 0.01,
        "point": 0.01,
        "stops_level_points": 10.0,
        "freeze_level_points": 5.0,
        "commission_per_lot": 5.0,
        "spread_points": 15.0,
    }
    valid_args[field] = invalid_val
    with pytest.raises(ValueError):
        SymbolRiskSpec(**valid_args)


def test_symbol_risk_spec_lot_step_invariants() -> None:
    with pytest.raises(ValueError, match="lot_min cannot exceed lot_max"):
        SymbolRiskSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            lot_min=10.0,
            lot_max=5.0,
            lot_step=0.01,
            point=0.01,
        )

    with pytest.raises(ValueError, match="lot_step cannot exceed lot_min"):
        SymbolRiskSpec(
            symbol="XAUUSD",
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            lot_min=0.01,
            lot_max=100.0,
            lot_step=0.1,
            point=0.01,
        )


def test_account_state_valid_and_invariants() -> None:
    acc = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )
    assert acc.balance == 10000.0
    assert acc.equity == 10000.0
    assert acc.margin == 0.0
    assert acc.free_margin == 10000.0
    assert acc.leverage == 100.0

    with pytest.raises(ValueError, match="leverage must be at least 1.0"):
        AccountState(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            leverage=0.5,
        )

    with pytest.raises(ValueError, match="finite"):
        AccountState(
            balance=float("nan"),
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            leverage=100.0,
        )


def test_position_risk_config_valid_and_invariants() -> None:
    cfg = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        max_risk_per_trade_usd=1000.0,
        fixed_risk_usd=None,
        be_trigger_r=1.5,
        be_offset_points=2.0,
        min_trailing_step_points=10.0,
        trailing_atr_multiplier=2.0,
    )
    assert cfg.risk_per_trade_percent == 0.01
    assert cfg.be_trigger_r == 1.5

    with pytest.raises(ValueError, match="risk_per_trade_percent must be between 0 and 1"):
        PositionRiskConfig(risk_per_trade_percent=0.0)

    with pytest.raises(ValueError, match="risk_per_trade_percent must be between 0 and 1"):
        PositionRiskConfig(risk_per_trade_percent=1.5)

    with pytest.raises(ValueError, match="be_trigger_r must be positive"):
        PositionRiskConfig(be_trigger_r=0.0)

    with pytest.raises(ValueError, match="min_trailing_step_points cannot be negative"):
        PositionRiskConfig(min_trailing_step_points=-1.0)


def test_open_position_valid_and_invariants() -> None:
    t0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
    pos = OpenPosition(
        ticket=123456,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.10,
        entry_price=2700.0,
        current_stop_loss=2690.0,
        current_take_profit=2720.0,
        initial_stop_loss=2690.0,
        initial_take_profit=2720.0,
        open_time=t0,
    )
    assert pos.ticket == 123456
    assert pos.volume == 0.10

    with pytest.raises(ValueError, match="open_time must be timezone-aware"):
        OpenPosition(
            ticket=123456,
            symbol="XAUUSD",
            direction=Direction.BULLISH,
            volume=0.10,
            entry_price=2700.0,
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=datetime(2026, 9, 25, 12, 0),
        )

    with pytest.raises(ValueError, match="positive"):
        OpenPosition(
            ticket=123456,
            symbol="XAUUSD",
            direction=Direction.BULLISH,
            volume=-0.10,
            entry_price=2700.0,
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=t0,
        )


def test_lot_calculation_result_immutability() -> None:
    res = LotCalculationResult(
        success=True,
        lot=0.15,
        raw_lot=0.1542,
        risk_amount_usd=100.0,
        loss_at_sl_usd=97.5,
        commission_usd=0.75,
        spread_cost_usd=2.25,
        price_distance=6.3,
        reject_reason=None,
        details="Calculated successfully",
    )
    assert res.success is True
    assert res.lot == 0.15
    field_to_modify = "lot"
    with pytest.raises(AttributeError):
        setattr(res, field_to_modify, 0.2)


def test_margin_validation_result_immutability() -> None:
    res = MarginValidationResult(
        is_valid=True,
        required_margin=270.0,
        free_margin=10000.0,
        margin_level_after_trade=3703.7,
        reject_reason=None,
    )
    assert res.is_valid is True
    field_to_modify = "is_valid"
    with pytest.raises(AttributeError):
        setattr(res, field_to_modify, False)


def test_breakeven_and_trailing_results_immutability() -> None:
    be = BreakevenResult(
        should_modify=True,
        new_stop_loss=2700.5,
        current_r_multiple=1.65,
        reason="Target 1.5R reached",
        reject_reason=None,
    )
    assert be.should_modify is True
    assert be.new_stop_loss == 2700.5

    ts = TrailingStopResult(
        should_modify=True,
        new_stop_loss=2705.0,
        delta_sl=5.0,
        reason="Trailing stop advance by 5.0",
        reject_reason=None,
    )
    assert ts.should_modify is True
    assert ts.delta_sl == 5.0


def test_symbol_risk_spec_empty_symbol() -> None:
    with pytest.raises(ValueError, match="symbol must not be empty"):
        SymbolRiskSpec(
            symbol="",
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            lot_min=0.01,
            lot_max=100.0,
            lot_step=0.01,
            point=0.01,
        )


def test_account_state_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin cannot be negative"):
        AccountState(
            balance=10000.0,
            equity=10000.0,
            margin=-10.0,
            free_margin=10000.0,
            leverage=100.0,
        )


def test_position_risk_config_optional_bounds() -> None:
    with pytest.raises(ValueError, match="max_risk_per_trade_usd must be positive"):
        PositionRiskConfig(max_risk_per_trade_usd=0.0)

    with pytest.raises(ValueError, match="fixed_risk_usd must be finite"):
        PositionRiskConfig(fixed_risk_usd=float("nan"))

    with pytest.raises(ValueError, match="expected_slippage_points cannot be negative"):
        PositionRiskConfig(expected_slippage_points=-1.0)

    with pytest.raises(ValueError, match="expected_slippage_points must be finite"):
        PositionRiskConfig(expected_slippage_points=float("nan"))

    with pytest.raises(ValueError, match="trailing_swing_buffer_points cannot be negative"):
        PositionRiskConfig(trailing_swing_buffer_points=-1.0)

    with pytest.raises(ValueError, match="trailing_atr_multiplier must be positive"):
        PositionRiskConfig(trailing_atr_multiplier=0.0)

    with pytest.raises(ValueError, match="be_offset_points cannot be negative"):
        PositionRiskConfig(be_offset_points=-1.0)


def test_lot_calculation_result_nan() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        LotCalculationResult(
            success=False,
            lot=float("nan"),
            raw_lot=0.0,
            risk_amount_usd=100.0,
            loss_at_sl_usd=0.0,
            commission_usd=0.0,
            spread_cost_usd=0.0,
            price_distance=0.0,
            slippage_cost_usd=0.0,
        )

    with pytest.raises(ValueError, match="slippage_cost_usd must be finite"):
        LotCalculationResult(
            success=False,
            lot=0.01,
            raw_lot=0.01,
            risk_amount_usd=100.0,
            loss_at_sl_usd=0.0,
            commission_usd=0.0,
            spread_cost_usd=0.0,
            price_distance=0.0,
            slippage_cost_usd=float("nan"),
        )


def test_margin_validation_result_non_finite() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        MarginValidationResult(
            is_valid=False,
            required_margin=float("inf"),
            free_margin=1000.0,
        )

    with pytest.raises(ValueError, match="margin_level_after_trade must be finite"):
        MarginValidationResult(
            is_valid=False,
            required_margin=100.0,
            free_margin=1000.0,
            margin_level_after_trade=float("nan"),
        )


def test_breakeven_result_validation() -> None:
    with pytest.raises(ValueError, match="current_r_multiple must be finite"):
        BreakevenResult(
            should_modify=False,
            current_r_multiple=float("nan"),
        )

    with pytest.raises(ValueError, match="new_stop_loss must be finite"):
        BreakevenResult(
            should_modify=True,
            new_stop_loss=float("inf"),
        )

    with pytest.raises(ValueError, match="new_stop_loss must be positive"):
        BreakevenResult(
            should_modify=True,
            new_stop_loss=0.0,
        )


def test_trailing_stop_result_validation() -> None:
    with pytest.raises(ValueError, match="delta_sl must be finite"):
        TrailingStopResult(
            should_modify=False,
            delta_sl=float("nan"),
        )

    with pytest.raises(ValueError, match="new_stop_loss must be finite"):
        TrailingStopResult(
            should_modify=True,
            new_stop_loss=float("nan"),
        )

    with pytest.raises(ValueError, match="new_stop_loss must be positive"):
        TrailingStopResult(
            should_modify=True,
            new_stop_loss=-10.0,
        )


def test_open_position_invariants() -> None:
    t0 = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="ticket must be positive"):
        OpenPosition(
            ticket=0,
            symbol="XAUUSD",
            direction=Direction.BULLISH,
            volume=0.1,
            entry_price=2700.0,
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=t0,
        )

    with pytest.raises(ValueError, match="symbol must not be empty"):
        OpenPosition(
            ticket=1,
            symbol="",
            direction=Direction.BULLISH,
            volume=0.1,
            entry_price=2700.0,
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=t0,
        )

    with pytest.raises(ValueError, match="direction must be a Direction enum"):
        OpenPosition(
            ticket=1,
            symbol="XAUUSD",
            direction=cast(Any, "BUY"),
            volume=0.1,
            entry_price=2700.0,
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=t0,
        )

    with pytest.raises(ValueError, match="entry_price must be finite"):
        OpenPosition(
            ticket=1,
            symbol="XAUUSD",
            direction=Direction.BULLISH,
            volume=0.1,
            entry_price=float("nan"),
            current_stop_loss=2690.0,
            current_take_profit=2720.0,
            initial_stop_loss=2690.0,
            initial_take_profit=2720.0,
            open_time=t0,
        )
