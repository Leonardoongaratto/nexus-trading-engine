import dataclasses
from collections.abc import Sequence
from datetime import datetime, timedelta

from xau_bot.domain.models import (
    BOS,
    FVG,
    OTE_LOWER,
    OTE_UPPER,
    Candle,
    ChoCH,
    DealingRange,
    Direction,
    FVGType,
    LiquiditySweep,
    LiquidityType,
    OrderBlock,
    OrderBlockType,
    SetupStatus,
    SetupTriggerType,
    Swing,
    SwingType,
    TradeSetup,
)
from xau_bot.smc.poi import detect_fvg, detect_order_blocks
from xau_bot.smc.structure import (
    detect_liquidity_sweeps,
    detect_structure,
)
from xau_bot.smc.swings import find_swings


def determine_htf_bias(
    htf_candles: Sequence[Candle],
    lookback_bars: int = 5,
    structure_events: Sequence[BOS | ChoCH] | None = None,
) -> Direction | None:
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if not htf_candles:
        return None
    if structure_events is not None:
        if structure_events:
            return structure_events[-1].direction
    elif len(htf_candles) >= lookback_bars:
        htf_swings = find_swings(htf_candles)
        htf_events = detect_structure(htf_candles, swings=htf_swings)
        if htf_events:
            return htf_events[-1].direction
    recent = htf_candles[-min(len(htf_candles), lookback_bars) :]
    if recent[-1].close >= recent[0].open:
        return Direction.BULLISH
    return Direction.BEARISH


def evaluate_setup_status(
    setup: TradeSetup,
    subsequent_candles: Sequence[Candle],
    as_of_time: datetime | None = None,
) -> TradeSetup:
    if as_of_time is not None and as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    duration = timedelta(seconds=setup.timeframe.duration_seconds)
    for c in subsequent_candles:
        if c.time < setup.confirmed_time:
            continue
        if as_of_time is not None and (c.time + duration > as_of_time):
            continue

        if setup.valid_until is not None and c.time > setup.valid_until:
            return dataclasses.replace(setup, status=SetupStatus.EXPIRED, status_updated_at=c.time)

        if setup.direction == Direction.BULLISH:
            if c.open <= setup.stop_loss:
                return dataclasses.replace(
                    setup, status=SetupStatus.INVALIDATED, status_updated_at=c.time
                )
            if c.open <= setup.entry_price or c.low <= setup.entry_price:
                return dataclasses.replace(
                    setup, status=SetupStatus.TRIGGERED, status_updated_at=c.time
                )
        else:
            if c.open >= setup.stop_loss:
                return dataclasses.replace(
                    setup, status=SetupStatus.INVALIDATED, status_updated_at=c.time
                )
            if c.open >= setup.entry_price or c.high >= setup.entry_price:
                return dataclasses.replace(
                    setup, status=SetupStatus.TRIGGERED, status_updated_at=c.time
                )

    return setup


def get_active_setups(
    setups: Sequence[TradeSetup],
    as_of_time: datetime | None = None,
    include_triggered: bool = False,
) -> list[TradeSetup]:
    if as_of_time is not None and as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    active: list[TradeSetup] = []
    for s in setups:
        if as_of_time is not None and s.confirmed_time > as_of_time:
            continue
        if s.status in (SetupStatus.INVALIDATED, SetupStatus.EXPIRED):
            continue
        if not include_triggered and s.status == SetupStatus.TRIGGERED:
            continue
        active.append(s)
    return active


def detect_setups(
    candles: Sequence[Candle],
    swings: Sequence[Swing] | None = None,
    structure_events: Sequence[BOS | ChoCH] | None = None,
    sweeps: Sequence[LiquiditySweep] | None = None,
    fvgs: Sequence[FVG] | None = None,
    order_blocks: Sequence[OrderBlock] | None = None,
    htf_candles: Sequence[Candle] | None = None,
    min_rr: float = 2.0,
    max_lookback_bars: int = 50,
    trigger_type: SetupTriggerType = SetupTriggerType.FVG_CE,
    require_sweep: bool = True,
    require_choch: bool = True,
    require_poi: bool = True,
    require_displacement: bool = True,
    min_ob_displacement_atr: float = 1.0,
    min_body_range_ratio: float = 1.2,
    ote_level: float = 0.62,
    allow_synthetic_tp: bool = True,
    htf_alignment_required: bool = False,
    htf_lookback_bars: int = 5,
    buffer_pips: float = 0.5,
    validity_bars: int = 20,
    as_of_time: datetime | None = None,
    require_premium_discount: bool = False,
    require_sweep_guard: bool = False,
    sweep_guard_max_rejection_ratio: float = 0.40,
) -> list[TradeSetup]:
    """Detect SMC entry setups across confirmed structural events.

    Parameters
    ----------
    allow_synthetic_tp:
        When True, if no opposing structural swing target was confirmed prior
        to the setup confirmation time, generates a synthetic take profit price
        at exactly `min_rr` risk:reward ratio. When False, setups without prior
        structural targets are discarded. Defaults to True.
    """
    if min_rr <= 0:
        raise ValueError("min_rr must be positive")
    if validity_bars <= 0:
        raise ValueError("validity_bars must be positive")
    if max_lookback_bars <= 0:
        raise ValueError("max_lookback_bars must be positive")
    if min_ob_displacement_atr < 0:
        raise ValueError("min_ob_displacement_atr must be non-negative")
    if min_body_range_ratio < 0:
        raise ValueError("min_body_range_ratio must be non-negative")
    if htf_lookback_bars <= 0:
        raise ValueError("htf_lookback_bars must be positive")
    if not (OTE_LOWER <= ote_level <= OTE_UPPER):
        raise ValueError(f"ote_level must be between {OTE_LOWER} and {OTE_UPPER}")
    if as_of_time is not None and as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    if len(candles) < 3:
        return []

    tf = candles[0].timeframe
    duration = timedelta(seconds=tf.duration_seconds)

    eval_candles = [c for c in candles if as_of_time is None or (c.time + duration <= as_of_time)]
    if len(eval_candles) < 3:
        return []

    if swings is None:
        eval_swings = find_swings(eval_candles)
    else:
        eval_swings = [s for s in swings if as_of_time is None or s.confirmed_time <= as_of_time]

    if structure_events is None:
        eval_events = detect_structure(eval_candles, swings=eval_swings)
    else:
        eval_events = [
            e for e in structure_events if as_of_time is None or e.confirmed_time <= as_of_time
        ]

    if sweeps is None:
        eval_sweeps = detect_liquidity_sweeps(eval_candles, swings=eval_swings)
    else:
        eval_sweeps = [s for s in sweeps if as_of_time is None or s.confirmed_time <= as_of_time]

    if fvgs is None:
        eval_fvgs = detect_fvg(eval_candles)
    else:
        eval_fvgs = [f for f in fvgs if as_of_time is None or f.confirmed_time <= as_of_time]

    if order_blocks is None:
        eval_obs = detect_order_blocks(eval_candles, structure_events=eval_events, fvgs=eval_fvgs)
    else:
        eval_obs = [o for o in order_blocks if as_of_time is None or o.confirmed_time <= as_of_time]

    htf_all_events: list[BOS | ChoCH] | None = None
    if htf_candles is not None and len(htf_candles) >= htf_lookback_bars:
        htf_swings = find_swings(htf_candles)
        htf_all_events = detect_structure(htf_candles, swings=htf_swings)

    candle_indices = {c.time: idx for idx, c in enumerate(eval_candles)}
    setups: list[TradeSetup] = []
    seen_setups: set[tuple[Direction, float, float, float, datetime]] = set()

    for event in eval_events:
        if require_choch and not isinstance(event, ChoCH):
            continue

        direction = event.direction

        event_htf_bias: Direction | None = None
        if htf_candles is not None:
            closed_htf = [
                c
                for c in htf_candles
                if c.time + timedelta(seconds=c.timeframe.duration_seconds) <= event.confirmed_time
            ]
            closed_events = (
                [e for e in htf_all_events if e.confirmed_time <= event.confirmed_time]
                if htf_all_events is not None
                else None
            )
            event_htf_bias = determine_htf_bias(
                closed_htf, lookback_bars=htf_lookback_bars, structure_events=closed_events
            )
            if htf_alignment_required and event_htf_bias != direction:
                continue

        choch_time = event.breaking_candle.time
        if choch_time not in candle_indices:
            continue
        choch_idx = candle_indices[choch_time]
        start_idx = max(0, choch_idx - max_lookback_bars)

        qualifying_sweeps: list[LiquiditySweep] = []
        expected_sweep_type = (
            LiquidityType.SSL if direction == Direction.BULLISH else LiquidityType.BSL
        )
        for sw in eval_sweeps:
            if sw.type != expected_sweep_type:
                continue
            if sw.candle.time > choch_time:
                continue
            if sw.confirmed_time > event.confirmed_time:
                continue
            sw_idx = candle_indices.get(sw.candle.time, -1)
            if sw_idx < start_idx:
                continue
            qualifying_sweeps.append(sw)

        if require_sweep and not qualifying_sweeps:
            continue

        sweep = qualifying_sweeps[-1] if qualifying_sweeps else None
        sweep_guard_passed = False

        if require_sweep_guard and sweep is not None:
            sw_c = sweep.candle
            sw_range = sw_c.high - sw_c.low
            if sw_range > 0:
                lower_wick = min(sw_c.open, sw_c.close) - sw_c.low
                upper_wick = sw_c.high - max(sw_c.open, sw_c.close)
                if (
                    direction == Direction.BEARISH
                    and (lower_wick / sw_range) >= sweep_guard_max_rejection_ratio
                ):
                    continue
                if (
                    direction == Direction.BULLISH
                    and (upper_wick / sw_range) >= sweep_guard_max_rejection_ratio
                ):
                    continue
                sweep_guard_passed = True

        cand_fvgs: list[FVG] = []
        expected_fvg_type = FVGType.BULLISH if direction == Direction.BULLISH else FVGType.BEARISH
        for f in eval_fvgs:
            if f.type != expected_fvg_type:
                continue
            if sweep is not None and f.time < sweep.candle.time:
                continue
            if f.time > event.breaking_candle.time + duration * 3:
                continue
            if (
                f.mitigated
                and f.mitigated_at is not None
                and f.mitigated_at <= event.confirmed_time
            ):
                continue
            cand_fvgs.append(f)

        cand_obs: list[OrderBlock] = []
        expected_ob_type = (
            OrderBlockType.BULLISH if direction == Direction.BULLISH else OrderBlockType.BEARISH
        )
        for ob in eval_obs:
            if ob.type != expected_ob_type:
                continue
            if sweep is not None and ob.time < sweep.candle.time - duration:
                continue
            if ob.time > event.breaking_candle.time + duration * 3:
                continue
            if (
                ob.mitigated
                and ob.mitigated_at is not None
                and ob.mitigated_at <= event.confirmed_time
            ):
                continue
            cand_obs.append(ob)

        if require_poi and not cand_fvgs and not cand_obs:
            continue

        if require_displacement:
            has_fvg_disp = len(cand_fvgs) > 0
            has_ob_disp = any(o.displacement_atr_ratio >= min_ob_displacement_atr for o in cand_obs)
            avg_range = sum(c.high - c.low for c in eval_candles[start_idx : choch_idx + 1]) / max(
                1, choch_idx - start_idx + 1
            )
            impulse_start = sweep.candle.time if sweep is not None else eval_candles[start_idx].time
            impulse_candles = [
                c for c in eval_candles if impulse_start <= c.time <= event.breaking_candle.time
            ]
            has_candle_disp = any(
                (
                    (c.close - c.open) >= min_body_range_ratio * avg_range
                    if direction == Direction.BULLISH
                    else (c.open - c.close) >= min_body_range_ratio * avg_range
                )
                for c in impulse_candles
            )
            if not (has_fvg_disp or has_ob_disp or has_candle_disp):
                continue

        setup_dealing_range: DealingRange | None = None
        if direction == Direction.BULLISH:
            leg_low = (
                sweep.candle.low
                if sweep is not None
                else min(c.low for c in eval_candles[start_idx : choch_idx + 1])
            )
            leg_high = event.breaking_candle.high
            if leg_high > leg_low:
                setup_dealing_range = DealingRange(high=leg_high, low=leg_low)
        else:
            leg_high = (
                sweep.candle.high
                if sweep is not None
                else max(c.high for c in eval_candles[start_idx : choch_idx + 1])
            )
            leg_low = event.breaking_candle.low
            if leg_high > leg_low:
                setup_dealing_range = DealingRange(high=leg_high, low=leg_low)

        selected_poi: FVG | OrderBlock | None = None
        entry_price: float = 0.0

        if trigger_type == SetupTriggerType.FVG_CE:
            if cand_fvgs:
                selected_poi = cand_fvgs[-1]
                entry_price = cand_fvgs[-1].consequent_encroachment
            elif cand_obs:
                selected_poi = cand_obs[-1]
                entry_price = cand_obs[-1].mean_threshold
        elif trigger_type == SetupTriggerType.FVG_PROXIMAL:
            if cand_fvgs:
                selected_poi = cand_fvgs[-1]
                entry_price = (
                    cand_fvgs[-1].top if direction == Direction.BULLISH else cand_fvgs[-1].bottom
                )
            elif cand_obs:
                selected_poi = cand_obs[-1]
                entry_price = (
                    cand_obs[-1].high if direction == Direction.BULLISH else cand_obs[-1].low
                )
        elif trigger_type == SetupTriggerType.OB_EXTREME:
            if cand_obs:
                selected_poi = cand_obs[-1]
                entry_price = (
                    cand_obs[-1].high if direction == Direction.BULLISH else cand_obs[-1].low
                )
            elif cand_fvgs:
                selected_poi = cand_fvgs[-1]
                entry_price = (
                    cand_fvgs[-1].top if direction == Direction.BULLISH else cand_fvgs[-1].bottom
                )
        elif trigger_type == SetupTriggerType.OB_MEAN_THRESHOLD:
            if cand_obs:
                selected_poi = cand_obs[-1]
                entry_price = cand_obs[-1].mean_threshold
            elif cand_fvgs:
                selected_poi = cand_fvgs[-1]
                entry_price = cand_fvgs[-1].consequent_encroachment
        elif trigger_type == SetupTriggerType.OTE_RETRACEMENT:
            if setup_dealing_range is not None:
                if direction == Direction.BULLISH:
                    entry_price = setup_dealing_range.high - setup_dealing_range.range * ote_level
                else:
                    entry_price = setup_dealing_range.low + setup_dealing_range.range * ote_level
            selected_poi = cand_fvgs[-1] if cand_fvgs else (cand_obs[-1] if cand_obs else None)

        if entry_price <= 0:
            continue

        if require_premium_discount and setup_dealing_range is not None:
            if direction == Direction.BULLISH and setup_dealing_range.is_premium(entry_price):
                continue
            if direction == Direction.BEARISH and setup_dealing_range.is_discount(entry_price):
                continue

        confirmed_time = max(
            event.confirmed_time,
            selected_poi.confirmed_time if selected_poi else event.confirmed_time,
        )
        if as_of_time is not None and confirmed_time > as_of_time:
            continue

        broken_swing = getattr(event, "broken_swing", None)
        selected_target_swing: Swing | None = None

        if direction == Direction.BULLISH:
            low_anchors: list[float] = []
            if sweep is not None:
                low_anchors.append(sweep.candle.low)
            if isinstance(selected_poi, OrderBlock):
                low_anchors.append(selected_poi.low)
            elif isinstance(selected_poi, FVG):
                low_anchors.append(selected_poi.bottom)
            if not low_anchors:
                low_anchors.append(eval_candles[choch_idx].low)
            structural_low = min(low_anchors)
            stop_loss = structural_low - buffer_pips
            if stop_loss >= entry_price:
                continue

            target_swings = [
                s
                for s in eval_swings
                if s.type == SwingType.HIGH
                and s.price > entry_price
                and s.confirmed_time <= confirmed_time
                and s != broken_swing
            ]
            if target_swings:
                selected_target_swing = min(target_swings, key=lambda s: s.price)
                take_profit = selected_target_swing.price
            elif allow_synthetic_tp:
                take_profit = entry_price + (entry_price - stop_loss) * min_rr
            else:
                continue

            rr = (take_profit - entry_price) / (entry_price - stop_loss)
        else:
            high_anchors: list[float] = []
            if sweep is not None:
                high_anchors.append(sweep.candle.high)
            if isinstance(selected_poi, OrderBlock):
                high_anchors.append(selected_poi.high)
            elif isinstance(selected_poi, FVG):
                high_anchors.append(selected_poi.top)
            if not high_anchors:
                high_anchors.append(eval_candles[choch_idx].high)
            structural_high = max(high_anchors)
            stop_loss = structural_high + buffer_pips
            if stop_loss <= entry_price:
                continue

            target_swings = [
                s
                for s in eval_swings
                if s.type == SwingType.LOW
                and s.price < entry_price
                and s.confirmed_time <= confirmed_time
                and s != broken_swing
            ]
            if target_swings:
                selected_target_swing = max(target_swings, key=lambda s: s.price)
                take_profit = selected_target_swing.price
            elif allow_synthetic_tp:
                take_profit = entry_price - (stop_loss - entry_price) * min_rr
            else:
                continue

            rr = (entry_price - take_profit) / (stop_loss - entry_price)

        if rr < min_rr - 1e-9:
            continue

        valid_until = confirmed_time + duration * validity_bars

        dedup_key = (
            direction,
            round(entry_price, 3),
            round(stop_loss, 3),
            round(take_profit, 3),
            confirmed_time,
        )
        if dedup_key in seen_setups:
            continue
        seen_setups.add(dedup_key)

        confluences_list: list[str] = []
        if sweep is not None:
            confluences_list.append("SWEEP_CONFIRMED")
        if isinstance(event, ChoCH):
            confluences_list.append("CHOCH_CONFIRMED")
        elif isinstance(event, BOS):
            confluences_list.append("BOS_CONFIRMED")
        if cand_fvgs:
            confluences_list.append("FVG_PRESENT")
        if cand_obs:
            confluences_list.append("OB_PRESENT")
        if trigger_type == SetupTriggerType.OTE_RETRACEMENT:
            confluences_list.append("OTE_ALIGNED")
        if event_htf_bias is not None:
            if event_htf_bias == direction:
                confluences_list.append("HTF_ALIGNED")
            else:
                confluences_list.append("HTF_COUNTER")
        if setup_dealing_range is not None:
            if direction == Direction.BULLISH and setup_dealing_range.is_discount(entry_price):
                confluences_list.append("DISCOUNT_ALIGNED")
            elif direction == Direction.BEARISH and setup_dealing_range.is_premium(entry_price):
                confluences_list.append("PREMIUM_ALIGNED")
        if sweep_guard_passed:
            confluences_list.append("SWEEP_GUARD_CONFIRMED")

        setup_id = (
            f"setup_{direction.value.lower()}_"
            f"{int(event.time.timestamp())}_{int(confirmed_time.timestamp())}"
        )

        choch_instance = event if isinstance(event, ChoCH) else None

        setup = TradeSetup(
            id=setup_id,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr,
            confluences=tuple(confluences_list),
            valid_until=valid_until,
            status=SetupStatus.PENDING,
            trigger_type=trigger_type,
            time=event.time,
            confirmed_time=confirmed_time,
            timeframe=tf,
            sweep=sweep,
            choch=choch_instance,
            poi=selected_poi,
            target_swing=selected_target_swing,
            dealing_range=setup_dealing_range,
        )
        setups.append(setup)

    return setups
