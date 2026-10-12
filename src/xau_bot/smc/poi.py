import dataclasses
from collections.abc import Sequence
from datetime import datetime, timedelta

from xau_bot.domain.models import (
    BOS,
    FVG,
    Candle,
    ChoCH,
    Direction,
    FVGType,
    OrderBlock,
    OrderBlockType,
)


def _resolve_atr(atr: float | Sequence[float | None] | None, idx: int) -> float | None:
    if atr is None:
        return None
    if isinstance(atr, (int, float)):
        if atr <= 0:
            raise ValueError(f"atr must be positive, got {atr}")
        return float(atr)
    if isinstance(atr, Sequence) and not isinstance(atr, (str, bytes)):
        if 0 <= idx < len(atr):
            val = atr[idx]
            if val is None or val <= 0:
                return None
            return float(val)
        return None
    raise TypeError(f"Unsupported atr type: {type(atr)}")


def detect_fvg(
    candles: Sequence[Candle],
    min_size: float = 0.0,
    atr: float | Sequence[float | None] | None = None,
    atr_multiplier: float | None = None,
    invalidation_by_close: bool = True,
) -> list[FVG]:
    if len(candles) < 3:
        return []
    if min_size < 0:
        raise ValueError("min_size must be non-negative")
    if atr_multiplier is not None and atr_multiplier < 0:
        raise ValueError("atr_multiplier must be non-negative")
    if (
        atr is not None
        and isinstance(atr, Sequence)
        and not isinstance(atr, (str, bytes))
        and len(atr) != len(candles)
    ):
        raise ValueError(
            f"atr sequence length ({len(atr)}) must match candles length ({len(candles)})"
        )

    duration = timedelta(seconds=candles[0].timeframe.duration_seconds)
    fvgs: list[FVG] = []

    for i in range(2, len(candles)):
        c0 = candles[i - 2]
        c1 = candles[i - 1]
        c2 = candles[i]

        bar_atr = _resolve_atr(atr, i)
        if atr is not None and atr_multiplier is not None and atr_multiplier > 0:
            if bar_atr is None:
                continue
            required_size = max(min_size, bar_atr * atr_multiplier)
        else:
            required_size = min_size

        if c2.low > c0.high:
            size = c2.low - c0.high
            if size >= required_size:
                fvg = FVG(
                    time=c1.time,
                    confirmed_time=c2.time + duration,
                    type=FVGType.BULLISH,
                    top=c2.low,
                    bottom=c0.high,
                    candle_0=c0,
                    candle_1=c1,
                    candle_2=c2,
                )
                fvgs.append(fvg)
        elif c2.high < c0.low:
            size = c0.low - c2.high
            if size >= required_size:
                fvg = FVG(
                    time=c1.time,
                    confirmed_time=c2.time + duration,
                    type=FVGType.BEARISH,
                    top=c0.low,
                    bottom=c2.high,
                    candle_0=c0,
                    candle_1=c1,
                    candle_2=c2,
                )
                fvgs.append(fvg)

    updated_fvgs: list[FVG] = []
    for fvg in fvgs:
        mitigated = False
        mitigated_at = None
        mitigated_by_candle = None
        ce_tested = False
        ce_tested_at = None
        invalidated = False
        invalidated_at = None
        invalidated_by_candle = None

        for candle in candles:
            if candle.time < fvg.confirmed_time:
                continue

            c_duration = timedelta(seconds=candle.timeframe.duration_seconds)
            event_time = candle.time + c_duration

            if fvg.type == FVGType.BULLISH:
                if not mitigated and candle.low < fvg.top:
                    mitigated = True
                    mitigated_at = event_time
                    mitigated_by_candle = candle
                if not ce_tested and candle.low <= fvg.consequent_encroachment:
                    ce_tested = True
                    ce_tested_at = event_time
                if not invalidated:
                    if invalidation_by_close:
                        if candle.close < fvg.bottom:
                            invalidated = True
                            invalidated_at = event_time
                            invalidated_by_candle = candle
                    else:
                        if candle.low <= fvg.bottom:
                            invalidated = True
                            invalidated_at = event_time
                            invalidated_by_candle = candle
            else:
                if not mitigated and candle.high > fvg.bottom:
                    mitigated = True
                    mitigated_at = event_time
                    mitigated_by_candle = candle
                if not ce_tested and candle.high >= fvg.consequent_encroachment:
                    ce_tested = True
                    ce_tested_at = event_time
                if not invalidated:
                    if invalidation_by_close:
                        if candle.close > fvg.top:
                            invalidated = True
                            invalidated_at = event_time
                            invalidated_by_candle = candle
                    else:
                        if candle.high >= fvg.top:
                            invalidated = True
                            invalidated_at = event_time
                            invalidated_by_candle = candle

        updated_fvg = dataclasses.replace(
            fvg,
            mitigated=mitigated,
            mitigated_at=mitigated_at,
            mitigated_by_candle=mitigated_by_candle,
            ce_tested=ce_tested,
            ce_tested_at=ce_tested_at,
            invalidated=invalidated,
            invalidated_at=invalidated_at,
            invalidated_by_candle=invalidated_by_candle,
        )
        updated_fvgs.append(updated_fvg)

    return updated_fvgs


def detect_order_blocks(
    candles: Sequence[Candle],
    structure_events: Sequence[BOS | ChoCH] | None = None,
    fvgs: Sequence[FVG] | None = None,
    atr: float | Sequence[float | None] | None = None,
    min_displacement_multiplier: float = 1.0,
    max_lookback: int = 20,
) -> list[OrderBlock]:
    if not candles:
        return []
    if max_lookback <= 0:
        raise ValueError("max_lookback must be positive")
    if min_displacement_multiplier < 0:
        raise ValueError("min_displacement_multiplier must be non-negative")
    if isinstance(atr, (int, float)) and atr <= 0:
        raise ValueError(f"atr must be positive, got {atr}")
    if (
        atr is not None
        and isinstance(atr, Sequence)
        and not isinstance(atr, (str, bytes))
        and len(atr) != len(candles)
    ):
        raise ValueError(
            f"atr sequence length ({len(atr)}) must match candles length ({len(candles)})"
        )

    candle_by_time = {c.time: idx for idx, c in enumerate(candles)}
    candidates: list[OrderBlock] = []

    if structure_events:
        for event in structure_events:
            break_time = event.breaking_candle.time
            if break_time not in candle_by_time:
                continue
            break_idx = candle_by_time[break_time]
            bar_atr = _resolve_atr(atr, break_idx)
            start_idx = max(-1, break_idx - 1 - max_lookback)

            if event.direction == Direction.BULLISH:
                ob_candle: Candle | None = None
                for idx in range(break_idx - 1, start_idx, -1):
                    c = candles[idx]
                    if c.high <= c.low:
                        continue
                    if c.close <= c.open:
                        ob_candle = c
                        break
                if ob_candle is not None:
                    displacement = event.breaking_candle.close - ob_candle.low
                    ratio = displacement / bar_atr if bar_atr and bar_atr > 0 else 0.0
                    if (
                        atr is not None
                        and min_displacement_multiplier > 0
                        and (bar_atr is None or ratio < min_displacement_multiplier)
                    ):
                        continue
                    ob = OrderBlock(
                        time=ob_candle.time,
                        confirmed_time=event.confirmed_time,
                        type=OrderBlockType.BULLISH,
                        high=ob_candle.high,
                        low=ob_candle.low,
                        open=ob_candle.open,
                        close=ob_candle.close,
                        candle=ob_candle,
                        displacement_atr_ratio=ratio,
                        breaking_event=event,
                    )
                    candidates.append(ob)
            else:
                ob_candle = None
                for idx in range(break_idx - 1, start_idx, -1):
                    c = candles[idx]
                    if c.high <= c.low:
                        continue
                    if c.close >= c.open:
                        ob_candle = c
                        break
                if ob_candle is not None:
                    displacement = ob_candle.high - event.breaking_candle.close
                    ratio = displacement / bar_atr if bar_atr and bar_atr > 0 else 0.0
                    if (
                        atr is not None
                        and min_displacement_multiplier > 0
                        and (bar_atr is None or ratio < min_displacement_multiplier)
                    ):
                        continue
                    ob = OrderBlock(
                        time=ob_candle.time,
                        confirmed_time=event.confirmed_time,
                        type=OrderBlockType.BEARISH,
                        high=ob_candle.high,
                        low=ob_candle.low,
                        open=ob_candle.open,
                        close=ob_candle.close,
                        candle=ob_candle,
                        displacement_atr_ratio=ratio,
                        breaking_event=event,
                    )
                    candidates.append(ob)

    if fvgs:
        for fvg in fvgs:
            c0_time = fvg.candle_0.time
            if c0_time not in candle_by_time:
                continue
            c0_idx = candle_by_time[c0_time]
            c2_time = fvg.candle_2.time
            if c2_time not in candle_by_time:
                continue
            fvg_idx = candle_by_time[c2_time]
            bar_atr = _resolve_atr(atr, fvg_idx)
            start_idx = max(-1, c0_idx - max_lookback)

            if fvg.type == FVGType.BULLISH:
                ob_candle = None
                for idx in range(c0_idx, start_idx, -1):
                    c = candles[idx]
                    if c.high <= c.low:
                        continue
                    if c.close <= c.open:
                        ob_candle = c
                        break
                if ob_candle is not None:
                    displacement = fvg.candle_2.close - ob_candle.low
                    ratio = displacement / bar_atr if bar_atr and bar_atr > 0 else 0.0
                    if (
                        atr is not None
                        and min_displacement_multiplier > 0
                        and (bar_atr is None or ratio < min_displacement_multiplier)
                    ):
                        continue
                    ob = OrderBlock(
                        time=ob_candle.time,
                        confirmed_time=fvg.confirmed_time,
                        type=OrderBlockType.BULLISH,
                        high=ob_candle.high,
                        low=ob_candle.low,
                        open=ob_candle.open,
                        close=ob_candle.close,
                        candle=ob_candle,
                        displacement_atr_ratio=ratio,
                        fvg=fvg,
                    )
                    candidates.append(ob)
            else:
                ob_candle = None
                for idx in range(c0_idx, start_idx, -1):
                    c = candles[idx]
                    if c.high <= c.low:
                        continue
                    if c.close >= c.open:
                        ob_candle = c
                        break
                if ob_candle is not None:
                    displacement = ob_candle.high - fvg.candle_2.close
                    ratio = displacement / bar_atr if bar_atr and bar_atr > 0 else 0.0
                    if (
                        atr is not None
                        and min_displacement_multiplier > 0
                        and (bar_atr is None or ratio < min_displacement_multiplier)
                    ):
                        continue
                    ob = OrderBlock(
                        time=ob_candle.time,
                        confirmed_time=fvg.confirmed_time,
                        type=OrderBlockType.BEARISH,
                        high=ob_candle.high,
                        low=ob_candle.low,
                        open=ob_candle.open,
                        close=ob_candle.close,
                        candle=ob_candle,
                        displacement_atr_ratio=ratio,
                        fvg=fvg,
                    )
                    candidates.append(ob)

    candidates.sort(
        key=lambda x: (
            x.confirmed_time,
            0 if x.breaking_event is not None else 1,
            x.time,
        )
    )
    order_blocks: list[OrderBlock] = []
    seen_ob_candles: set[datetime] = set()
    for ob in candidates:
        if ob.time not in seen_ob_candles:
            order_blocks.append(ob)
            seen_ob_candles.add(ob.time)

    updated_obs: list[OrderBlock] = []
    for ob in order_blocks:
        mitigated = False
        mitigated_at = None
        mitigated_by_candle = None
        invalidated = False
        invalidated_at = None
        invalidated_by_candle = None

        for candle in candles:
            if candle.time < ob.confirmed_time:
                continue

            c_duration = timedelta(seconds=candle.timeframe.duration_seconds)
            event_time = candle.time + c_duration

            if ob.type == OrderBlockType.BULLISH:
                if not mitigated and candle.low <= ob.high:
                    mitigated = True
                    mitigated_at = event_time
                    mitigated_by_candle = candle
                if not invalidated and candle.close < ob.low:
                    invalidated = True
                    invalidated_at = event_time
                    invalidated_by_candle = candle
            else:
                if not mitigated and candle.high >= ob.low:
                    mitigated = True
                    mitigated_at = event_time
                    mitigated_by_candle = candle
                if not invalidated and candle.close > ob.high:
                    invalidated = True
                    invalidated_at = event_time
                    invalidated_by_candle = candle

        updated_ob = dataclasses.replace(
            ob,
            mitigated=mitigated,
            mitigated_at=mitigated_at,
            mitigated_by_candle=mitigated_by_candle,
            invalidated=invalidated,
            invalidated_at=invalidated_at,
            invalidated_by_candle=invalidated_by_candle,
        )
        updated_obs.append(updated_ob)

    return sorted(updated_obs, key=lambda o: (o.confirmed_time, o.time))


def _project_fvg(f: FVG, as_of_time: datetime) -> FVG:
    is_invalidated = f.invalidated and (
        f.invalidated_at is not None and f.invalidated_at <= as_of_time
    )
    is_mitigated = f.mitigated and (f.mitigated_at is not None and f.mitigated_at <= as_of_time)
    is_ce_tested = f.ce_tested and (f.ce_tested_at is not None and f.ce_tested_at <= as_of_time)
    return dataclasses.replace(
        f,
        mitigated=is_mitigated,
        mitigated_at=f.mitigated_at if is_mitigated else None,
        mitigated_by_candle=f.mitigated_by_candle if is_mitigated else None,
        ce_tested=is_ce_tested,
        ce_tested_at=f.ce_tested_at if is_ce_tested else None,
        invalidated=is_invalidated,
        invalidated_at=f.invalidated_at if is_invalidated else None,
        invalidated_by_candle=f.invalidated_by_candle if is_invalidated else None,
    )


def get_active_fvgs(
    fvgs: Sequence[FVG],
    as_of_time: datetime | None = None,
    include_mitigated: bool = False,
) -> list[FVG]:
    if as_of_time is not None and as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    active: list[FVG] = []
    for f in fvgs:
        if as_of_time is not None and f.confirmed_time > as_of_time:
            continue
        is_invalidated = f.invalidated and (
            as_of_time is None or (f.invalidated_at is not None and f.invalidated_at <= as_of_time)
        )
        if is_invalidated:
            continue

        is_mitigated = f.mitigated and (
            as_of_time is None or (f.mitigated_at is not None and f.mitigated_at <= as_of_time)
        )
        if not include_mitigated and is_mitigated:
            continue

        if as_of_time is not None:
            active.append(_project_fvg(f, as_of_time))
        else:
            active.append(f)
    return active


def get_active_order_blocks(
    obs: Sequence[OrderBlock],
    as_of_time: datetime | None = None,
    include_mitigated: bool = False,
) -> list[OrderBlock]:
    if as_of_time is not None and as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    active: list[OrderBlock] = []
    for ob in obs:
        if as_of_time is not None and ob.confirmed_time > as_of_time:
            continue
        is_invalidated = ob.invalidated and (
            as_of_time is None
            or (ob.invalidated_at is not None and ob.invalidated_at <= as_of_time)
        )
        if is_invalidated:
            continue

        is_mitigated = ob.mitigated and (
            as_of_time is None or (ob.mitigated_at is not None and ob.mitigated_at <= as_of_time)
        )
        if not include_mitigated and is_mitigated:
            continue

        if as_of_time is not None:
            ob_snapshot = dataclasses.replace(
                ob,
                mitigated=is_mitigated,
                mitigated_at=ob.mitigated_at if is_mitigated else None,
                mitigated_by_candle=ob.mitigated_by_candle if is_mitigated else None,
                invalidated=False,
                invalidated_at=None,
                invalidated_by_candle=None,
                fvg=_project_fvg(ob.fvg, as_of_time) if ob.fvg is not None else None,
            )
            active.append(ob_snapshot)
        else:
            active.append(ob)
    return active
