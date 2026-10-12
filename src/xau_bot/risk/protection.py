import math
from collections.abc import Sequence

from xau_bot.domain.models import Candle, Direction, Swing, SwingType
from xau_bot.risk.models import (
    BreakevenResult,
    OpenPosition,
    PartialCloseResult,
    PositionRiskConfig,
    RiskRejectReason,
    SmartExitDecision,
    SymbolRiskSpec,
    TrailingStopResult,
)


def check_breakeven(
    position: OpenPosition,
    current_price: float,
    spec: SymbolRiskSpec,
    config: PositionRiskConfig,
    spread: float | None = None,
) -> BreakevenResult:
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"current_price must be positive and finite, got {current_price}")

    if spread is None:
        spread = spec.spread_points * spec.point
    elif not math.isfinite(spread) or spread < 0:
        raise ValueError(f"spread cannot be negative and must be finite, got {spread}")

    initial_risk = abs(position.entry_price - position.initial_stop_loss)
    if initial_risk <= 0:
        raise ValueError("initial_risk must be strictly positive")

    if position.direction == Direction.BULLISH:
        gain = current_price - position.entry_price
    elif position.direction == Direction.BEARISH:
        gain = position.entry_price - current_price
    else:
        raise ValueError(f"Invalid direction: {position.direction!r}")

    current_r = gain / initial_risk

    if current_r < config.be_trigger_r:
        return BreakevenResult(
            should_modify=False,
            current_r_multiple=round(current_r, 4),
            reason="Trigger R not reached",
            reject_reason=None,
        )

    commission_dist = (spec.commission_per_lot / spec.tick_value) * spec.tick_size
    offset_dist = config.be_offset_points * spec.point
    total_cost_buffer = spread + commission_dist + offset_dist

    price_decimals = max(0, -int(math.floor(math.log10(spec.point))))

    if position.direction == Direction.BULLISH:
        candidate_sl = position.entry_price + total_cost_buffer
        steps = math.ceil((candidate_sl / spec.point) - 1e-9)
        candidate_sl = round(steps * spec.point, price_decimals)
    else:
        candidate_sl = position.entry_price - total_cost_buffer
        steps = math.floor((candidate_sl / spec.point) + 1e-9)
        candidate_sl = round(steps * spec.point, price_decimals)

    if position.direction == Direction.BULLISH:
        if position.current_stop_loss >= candidate_sl:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reason="Current SL already at or better than breakeven",
                reject_reason=None,
            )
    else:
        if position.current_stop_loss <= candidate_sl:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reason="Current SL already at or better than breakeven",
                reject_reason=None,
            )

    stops_level_dist = spec.stops_level_points * spec.point
    freeze_level_dist = spec.freeze_level_points * spec.point

    if position.direction == Direction.BULLISH:
        if (current_price - candidate_sl) < stops_level_dist - 1e-9:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reject_reason=RiskRejectReason.STOPS_LEVEL_VIOLATION,
                reason="Candidate SL violates broker stops level",
            )
        if (current_price - position.current_stop_loss) <= freeze_level_dist + 1e-9:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reject_reason=RiskRejectReason.FREEZE_LEVEL_VIOLATION,
                reason="Order is in freeze zone",
            )
    else:
        if (candidate_sl - current_price) < stops_level_dist - 1e-9:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reject_reason=RiskRejectReason.STOPS_LEVEL_VIOLATION,
                reason="Candidate SL violates broker stops level",
            )
        if (position.current_stop_loss - current_price) <= freeze_level_dist + 1e-9:
            return BreakevenResult(
                should_modify=False,
                current_r_multiple=round(current_r, 4),
                reject_reason=RiskRejectReason.FREEZE_LEVEL_VIOLATION,
                reason="Order is in freeze zone",
            )

    return BreakevenResult(
        should_modify=True,
        new_stop_loss=candidate_sl,
        current_r_multiple=round(current_r, 4),
        reason="Breakeven condition met with costs covered",
        reject_reason=None,
    )


def check_trailing_stop(
    position: OpenPosition,
    current_price: float,
    spec: SymbolRiskSpec,
    config: PositionRiskConfig,
    atr: float | None = None,
    structural_swing: Swing | None = None,
    spread: float | None = None,
) -> TrailingStopResult:
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"current_price must be positive and finite, got {current_price}")

    if atr is None and structural_swing is None:
        raise ValueError("Either atr or structural_swing must be provided")

    if spread is None:
        spread = spec.spread_points * spec.point
    elif not math.isfinite(spread) or spread < 0:
        raise ValueError(f"spread cannot be negative and must be finite, got {spread}")

    price_decimals = max(0, -int(math.floor(math.log10(spec.point))))

    swing_sl: float | None = None
    if structural_swing is not None:
        if position.direction == Direction.BULLISH:
            if structural_swing.type != SwingType.LOW:
                raise ValueError("Bullish trailing requires a Swing LOW")
            swing_sl = structural_swing.price - (config.trailing_swing_buffer_points * spec.point)
        elif position.direction == Direction.BEARISH:
            if structural_swing.type != SwingType.HIGH:
                raise ValueError("Bearish trailing requires a Swing HIGH")
            swing_sl = (
                structural_swing.price + spread + (config.trailing_swing_buffer_points * spec.point)
            )
        else:
            raise ValueError(f"Invalid direction: {position.direction!r}")

    atr_sl: float | None = None
    if atr is not None:
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError(f"atr must be positive and finite, got {atr}")
        if position.direction == Direction.BULLISH:
            atr_sl = current_price - (config.trailing_atr_multiplier * atr)
        elif position.direction == Direction.BEARISH:
            atr_sl = current_price + (config.trailing_atr_multiplier * atr)
        else:
            raise ValueError(f"Invalid direction: {position.direction!r}")

    if swing_sl is not None and atr_sl is not None:
        if position.direction == Direction.BULLISH:
            selected_sl = max(swing_sl, atr_sl)
        else:
            selected_sl = min(swing_sl, atr_sl)
    elif swing_sl is not None:
        selected_sl = swing_sl
    elif atr_sl is not None:
        selected_sl = atr_sl
    else:
        raise ValueError("No valid stop loss calculated")

    candidate_sl = round(selected_sl, price_decimals)

    if position.direction == Direction.BULLISH:
        delta_sl = candidate_sl - position.current_stop_loss
        if delta_sl <= 0:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reason="Candidate SL does not advance beyond current SL",
                reject_reason=None,
            )
    else:
        delta_sl = position.current_stop_loss - candidate_sl
        if delta_sl <= 0:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reason="Candidate SL does not advance beyond current SL",
                reject_reason=None,
            )

    min_step_dist = config.min_trailing_step_points * spec.point
    if delta_sl < min_step_dist - 1e-9:
        return TrailingStopResult(
            should_modify=False,
            delta_sl=round(delta_sl, price_decimals),
            reason="Advance below minimum trailing step",
            reject_reason=None,
        )

    stops_level_dist = spec.stops_level_points * spec.point
    freeze_level_dist = spec.freeze_level_points * spec.point

    if position.direction == Direction.BULLISH:
        if (current_price - candidate_sl) < stops_level_dist - 1e-9:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reject_reason=RiskRejectReason.STOPS_LEVEL_VIOLATION,
                reason="Candidate SL violates broker stops level",
            )
        if (current_price - position.current_stop_loss) <= freeze_level_dist + 1e-9:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reject_reason=RiskRejectReason.FREEZE_LEVEL_VIOLATION,
                reason="Order is in freeze zone",
            )
    else:
        if (candidate_sl - current_price) < stops_level_dist - 1e-9:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reject_reason=RiskRejectReason.STOPS_LEVEL_VIOLATION,
                reason="Candidate SL violates broker stops level",
            )
        if (position.current_stop_loss - current_price) <= freeze_level_dist + 1e-9:
            return TrailingStopResult(
                should_modify=False,
                delta_sl=round(delta_sl, price_decimals),
                reject_reason=RiskRejectReason.FREEZE_LEVEL_VIOLATION,
                reason="Order is in freeze zone",
            )

    return TrailingStopResult(
        should_modify=True,
        new_stop_loss=candidate_sl,
        delta_sl=round(delta_sl, price_decimals),
        reason=f"Trailing stop advance by {delta_sl:.2f}",
        reject_reason=None,
    )


def check_partial_close(
    position: OpenPosition,
    current_price: float,
    spec: SymbolRiskSpec,
    config: PositionRiskConfig,
    already_partially_closed: bool = False,
) -> PartialCloseResult:
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"current_price must be positive and finite, got {current_price}")

    if already_partially_closed:
        return PartialCloseResult(
            should_close=False,
            reason="Position already partially closed",
        )

    if not config.partial_close_enabled:
        return PartialCloseResult(
            should_close=False,
            reason="Partial close disabled in configuration",
        )

    initial_risk = abs(position.entry_price - position.initial_stop_loss)
    if initial_risk <= 0:
        raise ValueError("initial_risk must be strictly positive")

    if position.direction == Direction.BULLISH:
        gain = current_price - position.entry_price
    elif position.direction == Direction.BEARISH:
        gain = position.entry_price - current_price
    else:
        raise ValueError(f"Invalid direction: {position.direction!r}")

    current_r = gain / initial_risk

    if current_r < config.partial_close_trigger_r:
        return PartialCloseResult(
            should_close=False,
            current_r_multiple=round(current_r, 4),
            reason=f"Trigger R ({config.partial_close_trigger_r}) not reached",
        )

    target_vol_close = position.volume * config.partial_close_ratio
    steps = round((target_vol_close - spec.lot_min) / spec.lot_step)
    vol_to_close = round(spec.lot_min + steps * spec.lot_step, 4)

    lot_decimals = max(0, -int(math.floor(math.log10(spec.lot_step)))) if spec.lot_step > 0 else 2
    vol_to_close = round(vol_to_close, lot_decimals)
    remaining_vol = round(position.volume - vol_to_close, lot_decimals)

    if vol_to_close < spec.lot_min:
        return PartialCloseResult(
            should_close=False,
            current_r_multiple=round(current_r, 4),
            reason=f"Volume to close ({vol_to_close}) below lot_min ({spec.lot_min})",
            reject_reason=RiskRejectReason.LOT_BELOW_MINIMUM,
        )

    if remaining_vol < spec.lot_min:
        return PartialCloseResult(
            should_close=False,
            current_r_multiple=round(current_r, 4),
            reason=f"Remaining volume ({remaining_vol}) below lot_min ({spec.lot_min})",
            reject_reason=RiskRejectReason.LOT_BELOW_MINIMUM,
        )

    return PartialCloseResult(
        should_close=True,
        volume_to_close=vol_to_close,
        remaining_volume=remaining_vol,
        current_r_multiple=round(current_r, 4),
        reason=f"Partial close trigger reached at {round(current_r, 2)}R",
    )


def check_smart_exit(
    position: OpenPosition,
    current_price: float,
    recent_candles: Sequence[Candle],
    swings: Sequence[Swing],
    config: PositionRiskConfig,
) -> SmartExitDecision:
    if not math.isfinite(current_price) or current_price <= 0:
        raise ValueError(f"current_price must be positive and finite, got {current_price}")

    if not config.smart_exit_enabled:
        return SmartExitDecision(should_exit=False, reason="Smart exit disabled")

    initial_risk = abs(position.entry_price - position.initial_stop_loss)
    if initial_risk <= 0:
        raise ValueError("initial_risk must be strictly positive")

    if position.direction == Direction.BULLISH:
        gain = current_price - position.entry_price
    elif position.direction == Direction.BEARISH:
        gain = position.entry_price - current_price
    else:
        raise ValueError(f"Invalid direction: {position.direction!r}")

    current_r = gain / initial_risk
    if current_r < config.smart_exit_min_r:
        return SmartExitDecision(
            should_exit=False,
            reason=(
                f"Current R ({round(current_r, 2)}) below "
                f"smart_exit_min_r ({config.smart_exit_min_r})"
            ),
        )

    relevant_swings = [s for s in swings if s.confirmed_time > position.open_time]
    if not relevant_swings or not recent_candles:
        return SmartExitDecision(should_exit=False, reason="No relevant swings after entry")

    latest_closed_candle = recent_candles[-1]

    if position.direction == Direction.BULLISH:
        swing_lows = [s for s in relevant_swings if s.type == SwingType.LOW]
        if swing_lows:
            last_swing_low = swing_lows[-1]
            if latest_closed_candle.close < last_swing_low.price:
                return SmartExitDecision(
                    should_exit=True,
                    reason=(
                        f"Opposite ChoCH detected: candle close {latest_closed_candle.close} "
                        f"below swing low {last_swing_low.price}"
                    ),
                    reversal_type="ChoCH_BEARISH",
                )
    elif position.direction == Direction.BEARISH:
        swing_highs = [s for s in relevant_swings if s.type == SwingType.HIGH]
        if swing_highs:
            last_swing_high = swing_highs[-1]
            if latest_closed_candle.close > last_swing_high.price:
                return SmartExitDecision(
                    should_exit=True,
                    reason=(
                        f"Opposite ChoCH detected: candle close {latest_closed_candle.close} "
                        f"above swing high {last_swing_high.price}"
                    ),
                    reversal_type="ChoCH_BULLISH",
                )

    return SmartExitDecision(should_exit=False, reason="Market structure intact")
