import math

from xau_bot.domain.models import Direction
from xau_bot.risk.models import (
    AccountState,
    LotCalculationResult,
    PositionRiskConfig,
    RiskRejectReason,
    SymbolRiskSpec,
)


def adjust_stop_loss_for_spread(
    direction: Direction,
    base_sl: float,
    spread: float,
    point: float,
    buffer_points: float = 0.0,
) -> float:
    if not math.isfinite(base_sl) or base_sl <= 0:
        raise ValueError(f"base_sl must be positive and finite, got {base_sl}")
    if not math.isfinite(spread) or spread < 0:
        raise ValueError(f"spread cannot be negative and must be finite, got {spread}")
    if not math.isfinite(buffer_points) or buffer_points < 0:
        raise ValueError(f"buffer_points cannot be negative, got {buffer_points}")
    if not math.isfinite(point) or point <= 0:
        raise ValueError(f"point must be positive, got {point}")

    if direction == Direction.BEARISH:
        return base_sl + spread + (buffer_points * point)
    return base_sl - (buffer_points * point)


def calculate_lot_size(
    account: AccountState,
    spec: SymbolRiskSpec,
    config: PositionRiskConfig,
    direction: Direction,
    entry_price: float,
    stop_loss: float,
    spread: float | None = None,
    adjust_sl_for_spread: bool = True,
) -> LotCalculationResult:
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(f"entry_price must be positive and finite, got {entry_price}")
    if not math.isfinite(stop_loss) or stop_loss <= 0:
        raise ValueError(f"stop_loss must be positive and finite, got {stop_loss}")

    if spread is None:
        spread = spec.spread_points * spec.point
    elif not math.isfinite(spread) or spread < 0:
        raise ValueError(f"spread cannot be negative and must be finite, got {spread}")

    if direction == Direction.BULLISH:
        if stop_loss >= entry_price:
            return LotCalculationResult(
                success=False,
                lot=0.0,
                raw_lot=0.0,
                risk_amount_usd=0.0,
                loss_at_sl_usd=0.0,
                commission_usd=0.0,
                spread_cost_usd=0.0,
                price_distance=0.0,
                slippage_cost_usd=0.0,
                reject_reason=RiskRejectReason.INVALID_PRICES,
                details="Bullish trade requires stop_loss < entry_price",
            )
        effective_sl = stop_loss
        price_distance = entry_price - effective_sl
    elif direction == Direction.BEARISH:
        if stop_loss <= entry_price:
            return LotCalculationResult(
                success=False,
                lot=0.0,
                raw_lot=0.0,
                risk_amount_usd=0.0,
                loss_at_sl_usd=0.0,
                commission_usd=0.0,
                spread_cost_usd=0.0,
                price_distance=0.0,
                slippage_cost_usd=0.0,
                reject_reason=RiskRejectReason.INVALID_PRICES,
                details="Bearish trade requires stop_loss > entry_price",
            )
        if adjust_sl_for_spread:
            effective_sl = adjust_stop_loss_for_spread(
                direction=direction,
                base_sl=stop_loss,
                spread=spread,
                point=spec.point,
                buffer_points=0.0,
            )
        else:
            effective_sl = stop_loss
        price_distance = effective_sl - entry_price
    else:
        raise ValueError(f"Invalid direction: {direction!r}")

    if config.fixed_risk_usd is not None:
        target_risk = config.fixed_risk_usd
    else:
        target_risk = account.equity * config.risk_per_trade_percent

    if config.max_risk_per_trade_usd is not None:
        target_risk = min(target_risk, config.max_risk_per_trade_usd)

    if target_risk <= 0:
        return LotCalculationResult(
            success=False,
            lot=0.0,
            raw_lot=0.0,
            risk_amount_usd=target_risk,
            loss_at_sl_usd=0.0,
            commission_usd=0.0,
            spread_cost_usd=0.0,
            price_distance=price_distance,
            slippage_cost_usd=0.0,
            reject_reason=RiskRejectReason.ZERO_OR_NEGATIVE_RISK,
            details="Target risk amount must be positive",
        )

    price_loss_per_lot = (price_distance / spec.tick_size) * spec.tick_value

    if direction == Direction.BULLISH:
        spread_cost_per_lot = (
            (spread / spec.tick_size) * spec.tick_value if config.include_spread_in_risk else 0.0
        )
    else:
        if adjust_sl_for_spread:
            spread_cost_per_lot = 0.0
        else:
            spread_cost_per_lot = (
                (spread / spec.tick_size) * spec.tick_value
                if config.include_spread_in_risk
                else 0.0
            )

    commission_per_lot = spec.commission_per_lot if config.include_commission_in_risk else 0.0

    slippage_distance = config.expected_slippage_points * spec.point
    slippage_cost_per_lot = (slippage_distance / spec.tick_size) * spec.tick_value

    total_loss_per_lot = (
        price_loss_per_lot + spread_cost_per_lot + commission_per_lot + slippage_cost_per_lot
    )

    if total_loss_per_lot <= 0:
        raise ValueError("Total loss per lot must be strictly positive")

    raw_lot = target_risk / total_loss_per_lot
    steps = math.floor((raw_lot / spec.lot_step) + 1e-9)
    step_decimals = max(0, -int(math.floor(math.log10(spec.lot_step))))
    stepped_lot = round(steps * spec.lot_step, step_decimals)

    if stepped_lot < spec.lot_min:
        return LotCalculationResult(
            success=False,
            lot=0.0,
            raw_lot=raw_lot,
            risk_amount_usd=target_risk,
            loss_at_sl_usd=0.0,
            commission_usd=0.0,
            spread_cost_usd=0.0,
            price_distance=price_distance,
            slippage_cost_usd=0.0,
            reject_reason=RiskRejectReason.LOT_BELOW_MINIMUM,
            details=f"Calculated lot {stepped_lot:.4f} is below minimum {spec.lot_min:.4f}",
        )

    final_lot = min(stepped_lot, spec.lot_max)
    actual_loss = round(final_lot * total_loss_per_lot, 4)
    actual_commission = round(final_lot * commission_per_lot, 4)
    actual_spread_cost = round(final_lot * spread_cost_per_lot, 4)
    actual_slippage_cost = round(final_lot * slippage_cost_per_lot, 4)

    return LotCalculationResult(
        success=True,
        lot=final_lot,
        raw_lot=raw_lot,
        risk_amount_usd=target_risk,
        loss_at_sl_usd=actual_loss,
        commission_usd=actual_commission,
        spread_cost_usd=actual_spread_cost,
        price_distance=price_distance,
        slippage_cost_usd=actual_slippage_cost,
        reject_reason=None,
        details="Calculated successfully",
    )
