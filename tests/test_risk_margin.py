import math

import pytest

from xau_bot.risk.margin import calculate_margin, validate_margin
from xau_bot.risk.models import (
    AccountState,
    RiskRejectReason,
    SymbolRiskSpec,
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
    )


def test_validate_margin_sufficient_margin(spec: SymbolRiskSpec) -> None:
    account = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )

    result = validate_margin(
        account=account,
        spec=spec,
        lot=1.0,
        entry_price=2700.0,
    )

    assert result.is_valid is True
    assert math.isclose(result.required_margin, 2700.0, rel_tol=1e-6)
    assert result.free_margin == 10000.0
    assert result.margin_level_after_trade is not None
    assert math.isclose(result.margin_level_after_trade, 370.37037, rel_tol=1e-4)
    assert result.reject_reason is None


def test_validate_margin_insufficient_margin(spec: SymbolRiskSpec) -> None:
    account = AccountState(
        balance=2000.0,
        equity=2000.0,
        margin=500.0,
        free_margin=1500.0,
        leverage=100.0,
    )

    result = validate_margin(
        account=account,
        spec=spec,
        lot=1.0,
        entry_price=2700.0,
    )

    assert result.is_valid is False
    assert math.isclose(result.required_margin, 2700.0, rel_tol=1e-6)
    assert result.reject_reason == RiskRejectReason.INSUFFICIENT_MARGIN


def test_validate_margin_min_margin_level_threshold(spec: SymbolRiskSpec) -> None:
    account = AccountState(
        balance=3000.0,
        equity=3000.0,
        margin=0.0,
        free_margin=3000.0,
        leverage=100.0,
    )

    result = validate_margin(
        account=account,
        spec=spec,
        lot=1.0,
        entry_price=2700.0,
        min_margin_level=150.0,
    )

    assert result.is_valid is False
    assert result.reject_reason == RiskRejectReason.INSUFFICIENT_MARGIN


def test_validate_margin_invalid_inputs(spec: SymbolRiskSpec) -> None:
    account = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )

    with pytest.raises(ValueError, match="lots must be positive"):
        validate_margin(account, spec, lot=0.0, entry_price=2700.0)

    with pytest.raises(ValueError, match="price must be positive"):
        validate_margin(account, spec, lot=1.0, entry_price=-100.0)

    with pytest.raises(ValueError, match="min_margin_level must be positive"):
        validate_margin(account, spec, lot=1.0, entry_price=2700.0, min_margin_level=0.0)


def test_calculate_margin_validations() -> None:
    with pytest.raises(ValueError, match="lots must be positive and finite"):
        calculate_margin(lots=0.0, price=2600.0, contract_size=100.0, leverage=100.0)

    with pytest.raises(ValueError, match="price must be positive and finite"):
        calculate_margin(lots=1.0, price=0.0, contract_size=100.0, leverage=100.0)

    with pytest.raises(ValueError, match="contract_size must be positive and finite"):
        calculate_margin(lots=1.0, price=2600.0, contract_size=0.0, leverage=100.0)

    with pytest.raises(ValueError, match="contract_size must be positive and finite"):
        calculate_margin(lots=1.0, price=2600.0, contract_size=float("inf"), leverage=100.0)

    with pytest.raises(ValueError, match="leverage must be >= 1.0"):
        calculate_margin(lots=1.0, price=2600.0, contract_size=100.0, leverage=0.5)

    with pytest.raises(ValueError, match="leverage must be >= 1.0"):
        calculate_margin(lots=1.0, price=2600.0, contract_size=100.0, leverage=float("nan"))
