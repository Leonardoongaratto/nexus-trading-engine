import math
from typing import Any, cast

import pytest

from xau_bot.domain.models import Direction
from xau_bot.risk.models import (
    AccountState,
    PositionRiskConfig,
    RiskRejectReason,
    SymbolRiskSpec,
)
from xau_bot.risk.position import (
    adjust_stop_loss_for_spread,
    calculate_lot_size,
)


@pytest.fixture
def base_account() -> AccountState:
    return AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )


@pytest.fixture
def spec_100oz() -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=5.0,
        spread_points=20.0,
    )


@pytest.fixture
def spec_10oz() -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol="XAUUSD_MINI",
        contract_size=10.0,
        tick_size=0.01,
        tick_value=0.10,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=2.0,
        spread_points=20.0,
    )


@pytest.fixture
def spec_1oz() -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol="XAUUSD_MICRO",
        contract_size=1.0,
        tick_size=0.01,
        tick_value=0.01,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=0.5,
        spread_points=20.0,
    )


def test_adjust_stop_loss_for_spread() -> None:
    sell_base_sl = 2710.00
    spread = 0.25
    adjusted_sell_sl = adjust_stop_loss_for_spread(
        direction=Direction.BEARISH,
        base_sl=sell_base_sl,
        spread=spread,
        buffer_points=5.0,
        point=0.01,
    )
    assert math.isclose(adjusted_sell_sl, 2710.30, rel_tol=1e-6)

    buy_base_sl = 2690.00
    adjusted_buy_sl = adjust_stop_loss_for_spread(
        direction=Direction.BULLISH,
        base_sl=buy_base_sl,
        spread=spread,
        buffer_points=5.0,
        point=0.01,
    )
    assert math.isclose(adjusted_buy_sl, 2689.95, rel_tol=1e-6)


@pytest.mark.parametrize(
    "spec_fixture,expected_lot,expected_loss",
    [
        ("spec_100oz", 0.19, 99.75),
        ("spec_10oz", 1.85, 99.90),
        ("spec_1oz", 17.54, 99.98),
    ],
)
def test_calculate_lot_size_contract_sizes_ac_100(
    request: pytest.FixtureRequest,
    base_account: AccountState,
    spec_fixture: str,
    expected_lot: float,
    expected_loss: float,
) -> None:
    spec: SymbolRiskSpec = request.getfixturevalue(spec_fixture)
    config = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        include_commission_in_risk=True,
        include_spread_in_risk=True,
    )

    entry_price = 2700.00
    stop_loss = 2695.00

    result = calculate_lot_size(
        account=base_account,
        spec=spec,
        config=config,
        direction=Direction.BULLISH,
        entry_price=entry_price,
        stop_loss=stop_loss,
    )

    assert result.success is True
    assert result.lot == expected_lot
    assert math.isclose(result.loss_at_sl_usd, expected_loss, abs_tol=0.02)
    assert result.lot >= spec.lot_min
    assert result.lot <= spec.lot_max
    steps = round(result.lot / spec.lot_step, 6)
    assert math.isclose(steps, round(steps), abs_tol=1e-5)
    assert result.loss_at_sl_usd <= 100.00 + 1e-6
    assert result.risk_amount_usd == 100.00


def test_calculate_lot_size_sell_with_spread_buffer(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(risk_per_trade_percent=0.01)
    entry_price = 2700.00
    base_sl = 2705.00

    result = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BEARISH,
        entry_price=entry_price,
        stop_loss=base_sl,
        adjust_sl_for_spread=True,
    )

    assert result.success is True
    assert result.lot == 0.19
    assert math.isclose(result.loss_at_sl_usd, 99.75, rel_tol=1e-4)
    assert result.spread_cost_usd == 0.0
    assert result.loss_at_sl_usd <= 100.00 + 1e-6


def test_calculate_lot_size_lot_below_min_ac_101(
    spec_100oz: SymbolRiskSpec,
) -> None:
    small_account = AccountState(
        balance=200.0,
        equity=200.0,
        margin=0.0,
        free_margin=200.0,
        leverage=100.0,
    )
    config = PositionRiskConfig(risk_per_trade_percent=0.01)

    result = calculate_lot_size(
        account=small_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )

    assert result.success is False
    assert result.lot == 0.0
    assert result.reject_reason == RiskRejectReason.LOT_BELOW_MINIMUM
    assert "below minimum" in result.details.lower()


def test_calculate_lot_size_invalid_prices(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(risk_per_trade_percent=0.01)

    res_bull = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2705.00,
    )
    assert res_bull.success is False
    assert res_bull.reject_reason == RiskRejectReason.INVALID_PRICES

    res_bear = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BEARISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )
    assert res_bear.success is False
    assert res_bear.reject_reason == RiskRejectReason.INVALID_PRICES


def test_calculate_lot_size_fixed_risk_and_hard_cap(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config_fixed = PositionRiskConfig(fixed_risk_usd=50.0)
    res_fixed = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config_fixed,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )
    assert res_fixed.success is True
    assert res_fixed.risk_amount_usd == 50.0
    assert res_fixed.loss_at_sl_usd <= 50.0 + 1e-6

    config_capped = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        max_risk_per_trade_usd=30.0,
    )
    res_capped = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config_capped,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )
    assert res_capped.success is True
    assert res_capped.risk_amount_usd == 30.0
    assert res_capped.loss_at_sl_usd <= 30.0 + 1e-6


def test_adjust_stop_loss_for_spread_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="base_sl must be positive"):
        adjust_stop_loss_for_spread(Direction.BEARISH, 0.0, 0.2, point=0.01)

    with pytest.raises(ValueError, match="spread cannot be negative"):
        adjust_stop_loss_for_spread(Direction.BEARISH, 2700.0, -0.2, point=0.01)

    with pytest.raises(ValueError, match="buffer_points cannot be negative"):
        adjust_stop_loss_for_spread(Direction.BEARISH, 2700.0, 0.2, point=0.01, buffer_points=-1.0)

    with pytest.raises(ValueError, match="point must be positive"):
        adjust_stop_loss_for_spread(Direction.BEARISH, 2700.0, 0.2, point=0.0)


def test_calculate_lot_size_invalid_inputs(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(risk_per_trade_percent=0.01)

    with pytest.raises(ValueError, match="entry_price must be positive"):
        calculate_lot_size(base_account, spec_100oz, config, Direction.BULLISH, 0.0, 2690.0)

    with pytest.raises(ValueError, match="stop_loss must be positive"):
        calculate_lot_size(base_account, spec_100oz, config, Direction.BULLISH, 2700.0, -10.0)

    with pytest.raises(ValueError, match="spread cannot be negative"):
        calculate_lot_size(
            base_account, spec_100oz, config, Direction.BULLISH, 2700.0, 2690.0, spread=-0.5
        )

    with pytest.raises(ValueError, match="Invalid direction"):
        calculate_lot_size(base_account, spec_100oz, config, cast(Any, "INVALID"), 2700.0, 2690.0)


def test_calculate_lot_size_zero_or_negative_risk(spec_100oz: SymbolRiskSpec) -> None:
    broke_account = AccountState(
        balance=0.0,
        equity=0.0,
        margin=0.0,
        free_margin=0.0,
        leverage=100.0,
    )
    config = PositionRiskConfig(risk_per_trade_percent=0.01)
    res = calculate_lot_size(broke_account, spec_100oz, config, Direction.BULLISH, 2700.0, 2690.0)
    assert res.success is False
    assert res.reject_reason == RiskRejectReason.ZERO_OR_NEGATIVE_RISK


def test_calculate_lot_size_bearish_no_spread_adjustment(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(risk_per_trade_percent=0.01, include_spread_in_risk=False)
    res = calculate_lot_size(
        base_account,
        spec_100oz,
        config,
        Direction.BEARISH,
        2700.0,
        2710.0,
        adjust_sl_for_spread=False,
    )
    assert res.success is True
    assert res.price_distance == 10.0
    assert res.spread_cost_usd == 0.0


def test_calculate_lot_size_clamping_to_lot_max(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    spec_low_max = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=0.05,
        lot_step=0.01,
        point=0.01,
    )
    config = PositionRiskConfig(risk_per_trade_percent=0.05)
    res = calculate_lot_size(
        base_account,
        spec_low_max,
        config,
        Direction.BULLISH,
        2700.0,
        2699.0,
    )
    assert res.success is True
    assert res.lot == 0.05


def test_lot_sizing_respects_custom_tick_value(
    base_account: AccountState,
) -> None:
    spec_custom_tick = SymbolRiskSpec(
        symbol="XAUUSD_EUR",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=5.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=5.0,
        spread_points=20.0,
    )
    config = PositionRiskConfig(risk_per_trade_percent=0.01)

    # 500 ticks * 5.0 = $2500 price loss per lot, spread = $100, commission = $5. Total = $2605/lot.
    # Raw lot = 100 / 2605 = 0.03838... -> stepped lot = 0.03. Loss at SL = 0.03 * 2605 = $78.15.
    res = calculate_lot_size(
        account=base_account,
        spec=spec_custom_tick,
        config=config,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )
    assert res.success is True
    assert res.lot == 0.03
    assert math.isclose(res.loss_at_sl_usd, 78.15, rel_tol=1e-4)


def test_sell_adjust_sl_for_spread_and_no_include_spread(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        include_spread_in_risk=False,
    )
    res = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BEARISH,
        entry_price=2700.00,
        stop_loss=2705.00,
        adjust_sl_for_spread=True,
    )
    assert res.success is True
    assert res.lot == 0.19
    assert math.isclose(res.loss_at_sl_usd, 99.75, rel_tol=1e-4)
    assert res.spread_cost_usd == 0.0


def test_lot_sizing_with_expected_slippage_points(
    base_account: AccountState, spec_100oz: SymbolRiskSpec
) -> None:
    config = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        expected_slippage_points=10.0,
    )
    # Total loss per lot = 500 + 20 + 5 + 10 = $535.
    # Raw lot = 100 / 535 = 0.1869... -> stepped lot = 0.18. Loss = 0.18 * 535 = $96.30.
    res = calculate_lot_size(
        account=base_account,
        spec=spec_100oz,
        config=config,
        direction=Direction.BULLISH,
        entry_price=2700.00,
        stop_loss=2695.00,
    )
    assert res.success is True
    assert res.lot == 0.18
    assert math.isclose(res.loss_at_sl_usd, 96.30, rel_tol=1e-4)
    assert math.isclose(res.slippage_cost_usd, 1.80, rel_tol=1e-4)


def test_independent_manual_reference_values_ac_100(
    base_account: AccountState,
    spec_100oz: SymbolRiskSpec,
    spec_10oz: SymbolRiskSpec,
    spec_1oz: SymbolRiskSpec,
) -> None:
    config = PositionRiskConfig(
        risk_per_trade_percent=0.01,
        include_commission_in_risk=True,
        include_spread_in_risk=True,
    )

    res_100 = calculate_lot_size(
        base_account, spec_100oz, config, Direction.BULLISH, 2700.00, 2695.00
    )
    assert res_100.success is True
    assert res_100.lot == 0.19
    assert math.isclose(res_100.loss_at_sl_usd, 99.75, rel_tol=1e-4)

    res_10 = calculate_lot_size(
        base_account, spec_10oz, config, Direction.BULLISH, 2700.00, 2695.00
    )
    assert res_10.success is True
    assert res_10.lot == 1.85
    assert math.isclose(res_10.loss_at_sl_usd, 99.90, rel_tol=1e-4)

    res_1 = calculate_lot_size(base_account, spec_1oz, config, Direction.BULLISH, 2700.00, 2695.00)
    assert res_1.success is True
    assert res_1.lot == 17.54
    assert math.isclose(res_1.loss_at_sl_usd, 99.98, abs_tol=0.01)
