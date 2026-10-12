import math

from xau_bot.risk.models import (
    AccountState,
    MarginValidationResult,
    RiskRejectReason,
    SymbolRiskSpec,
)


def calculate_margin(
    lots: float,
    price: float,
    contract_size: float,
    leverage: float,
) -> float:
    if not math.isfinite(lots) or lots <= 0:
        raise ValueError(f"lots must be positive and finite, got {lots}")
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"price must be positive and finite, got {price}")
    if not math.isfinite(contract_size) or contract_size <= 0:
        raise ValueError(f"contract_size must be positive and finite, got {contract_size}")
    if not math.isfinite(leverage) or leverage < 1.0:
        raise ValueError(f"leverage must be >= 1.0, got {leverage}")

    return (lots * contract_size * price) / leverage


def validate_margin(
    account: AccountState,
    spec: SymbolRiskSpec,
    lot: float,
    entry_price: float,
    min_margin_level: float | None = None,
) -> MarginValidationResult:
    """Validate margin requirement for a prospective order.

    Calculates required margin assuming account currency equals symbol quote
    currency (USD). Dynamic cross-currency conversion via MT5 order_calc_margin
    will be integrated in T-15.
    """
    if min_margin_level is not None and (
        not math.isfinite(min_margin_level) or min_margin_level <= 0
    ):
        raise ValueError(f"min_margin_level must be positive and finite, got {min_margin_level}")

    required_margin = calculate_margin(
        lots=lot,
        price=entry_price,
        contract_size=spec.contract_size,
        leverage=account.leverage,
    )

    total_projected_margin = account.margin + required_margin
    projected_margin_level = (account.equity / total_projected_margin) * 100.0

    if required_margin > account.free_margin:
        return MarginValidationResult(
            is_valid=False,
            required_margin=required_margin,
            free_margin=account.free_margin,
            margin_level_after_trade=projected_margin_level,
            reject_reason=RiskRejectReason.INSUFFICIENT_MARGIN,
        )

    if min_margin_level is not None and projected_margin_level < min_margin_level:
        return MarginValidationResult(
            is_valid=False,
            required_margin=required_margin,
            free_margin=account.free_margin,
            margin_level_after_trade=projected_margin_level,
            reject_reason=RiskRejectReason.INSUFFICIENT_MARGIN,
        )

    return MarginValidationResult(
        is_valid=True,
        required_margin=required_margin,
        free_margin=account.free_margin,
        margin_level_after_trade=projected_margin_level,
        reject_reason=None,
    )
