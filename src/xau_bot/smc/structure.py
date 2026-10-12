import dataclasses
import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from xau_bot.domain.models import (
    BOS,
    Candle,
    ChoCH,
    DealingRange,
    Direction,
    LiquidityPool,
    LiquiditySweep,
    LiquidityType,
    Swing,
    SwingType,
)


def _handle_bullish_break(
    candle: Candle,
    as_of_time: datetime,
    target_high: Swing,
    current_trend: Direction | None,
    available_swings: list[Swing],
    broken_swings: set[Swing],
    events: list[BOS | ChoCH],
) -> Direction:
    is_choch = current_trend == Direction.BEARISH
    if is_choch:
        event: BOS | ChoCH = ChoCH(
            time=candle.time,
            confirmed_time=as_of_time,
            direction=Direction.BULLISH,
            previous_trend=Direction.BEARISH,
            broken_swing=target_high,
            breaking_candle=candle,
            price=target_high.price,
        )
    else:
        event = BOS(
            time=candle.time,
            confirmed_time=as_of_time,
            direction=Direction.BULLISH,
            broken_swing=target_high,
            breaking_candle=candle,
            price=target_high.price,
        )
    events.append(event)
    broken_swings.add(target_high)
    for s in available_swings:
        if s.type == SwingType.HIGH and s.price < candle.close:
            broken_swings.add(s)
    return Direction.BULLISH


def _handle_bearish_break(
    candle: Candle,
    as_of_time: datetime,
    target_low: Swing,
    current_trend: Direction | None,
    available_swings: list[Swing],
    broken_swings: set[Swing],
    events: list[BOS | ChoCH],
) -> Direction:
    is_choch = current_trend == Direction.BULLISH
    if is_choch:
        event: BOS | ChoCH = ChoCH(
            time=candle.time,
            confirmed_time=as_of_time,
            direction=Direction.BEARISH,
            previous_trend=Direction.BULLISH,
            broken_swing=target_low,
            breaking_candle=candle,
            price=target_low.price,
        )
    else:
        event = BOS(
            time=candle.time,
            confirmed_time=as_of_time,
            direction=Direction.BEARISH,
            broken_swing=target_low,
            breaking_candle=candle,
            price=target_low.price,
        )
    events.append(event)
    broken_swings.add(target_low)
    for s in available_swings:
        if s.type == SwingType.LOW and s.price > candle.close:
            broken_swings.add(s)
    return Direction.BEARISH


def detect_structure(
    candles: Sequence[Candle],
    swings: Sequence[Swing],
    initial_trend: Direction | None = None,
) -> list[BOS | ChoCH]:
    events: list[BOS | ChoCH] = []
    if not candles or not swings:
        return events

    sorted_swings = sorted(swings, key=lambda s: (s.confirmed_time, s.index))
    broken_swings: set[Swing] = set()
    current_trend: Direction | None = initial_trend

    for candle in candles:
        candle_duration = timedelta(seconds=candle.timeframe.duration_seconds)
        as_of_time = candle.time + candle_duration

        available_swings = [
            s for s in sorted_swings if s.confirmed_time <= as_of_time and s not in broken_swings
        ]
        if not available_swings:
            continue

        active_highs = [s for s in available_swings if s.type == SwingType.HIGH]
        active_lows = [s for s in available_swings if s.type == SwingType.LOW]

        bull_first = current_trend != Direction.BEARISH
        order = ("bullish", "bearish") if bull_first else ("bearish", "bullish")

        for side in order:
            if side == "bullish":
                if active_highs and candle.close > active_highs[-1].price:
                    current_trend = _handle_bullish_break(
                        candle,
                        as_of_time,
                        active_highs[-1],
                        current_trend,
                        available_swings,
                        broken_swings,
                        events,
                    )
                    break
            else:
                if active_lows and candle.close < active_lows[-1].price:
                    current_trend = _handle_bearish_break(
                        candle,
                        as_of_time,
                        active_lows[-1],
                        current_trend,
                        available_swings,
                        broken_swings,
                        events,
                    )
                    break

    return events


def detect_liquidity_pools(
    swings: Sequence[Swing],
    tolerance: float = 1.0,
    atr: float | None = None,
    atr_multiplier: float | None = None,
    include_single_swings: bool = True,
    as_of_time: datetime | None = None,
) -> list[LiquidityPool]:
    if (atr is not None and atr_multiplier is None) or (atr is None and atr_multiplier is not None):
        raise ValueError("Both atr and atr_multiplier must be provided together")

    if atr is not None and atr_multiplier is not None:
        effective_tolerance = atr * atr_multiplier
    else:
        effective_tolerance = tolerance

    if not math.isfinite(effective_tolerance) or effective_tolerance < 0:
        raise ValueError(
            f"effective tolerance must be finite and non-negative, got {effective_tolerance}"
        )

    if as_of_time is not None:
        if as_of_time.tzinfo is None:
            raise ValueError("as_of_time must be timezone-aware")
        filtered_swings = [s for s in swings if s.confirmed_time <= as_of_time]
    else:
        filtered_swings = list(swings)

    if not filtered_swings:
        return []

    sorted_swings = sorted(filtered_swings, key=lambda s: (s.confirmed_time, s.index))
    high_swings = [s for s in sorted_swings if s.type == SwingType.HIGH]
    low_swings = [s for s in sorted_swings if s.type == SwingType.LOW]

    def _cluster_swings(items: list[Swing], pool_type: LiquidityType) -> list[LiquidityPool]:
        result_pools: list[LiquidityPool] = []
        visited: set[int] = set()
        for i, s1 in enumerate(items):
            if i in visited:
                continue
            cluster: list[Swing] = [s1]
            visited.add(i)
            for j in range(i + 1, len(items)):
                if j in visited:
                    continue
                s2 = items[j]
                new_max = max(max(s.price for s in cluster), s2.price)
                new_min = min(min(s.price for s in cluster), s2.price)
                if new_max - new_min <= effective_tolerance:
                    cluster.append(s2)
                    visited.add(j)

            if len(cluster) >= 2:
                level = (
                    max(s.price for s in cluster)
                    if pool_type == LiquidityType.BSL
                    else min(s.price for s in cluster)
                )
                latest_time = max(s.time for s in cluster)
                conf_time = max(s.confirmed_time for s in cluster)
                result_pools.append(
                    LiquidityPool(
                        type=pool_type,
                        price=level,
                        swings=tuple(cluster),
                        time=latest_time,
                        confirmed_time=conf_time,
                    )
                )
            elif include_single_swings:
                result_pools.append(
                    LiquidityPool(
                        type=pool_type,
                        price=s1.price,
                        swings=(s1,),
                        time=s1.time,
                        confirmed_time=s1.confirmed_time,
                    )
                )
        return result_pools

    pools: list[LiquidityPool] = []
    pools.extend(_cluster_swings(high_swings, LiquidityType.BSL))
    pools.extend(_cluster_swings(low_swings, LiquidityType.SSL))
    pools.sort(key=lambda p: (p.confirmed_time, p.time))
    return pools


def detect_liquidity_sweeps(
    candles: Sequence[Candle],
    swings: Sequence[Swing],
    tolerance: float = 1.0,
    atr: float | None = None,
    atr_multiplier: float | None = None,
    include_single_swings: bool = True,
) -> list[LiquiditySweep]:
    sweeps: list[LiquiditySweep] = []
    if not candles or not swings:
        return sweeps

    all_swings = list(dict.fromkeys(swings))
    all_swings.sort(key=lambda s: (s.confirmed_time, s.index))

    mitigated_swings: set[Swing] = set()

    for candle in candles:
        candle_duration = timedelta(seconds=candle.timeframe.duration_seconds)
        as_of_time = candle.time + candle_duration

        available_swings = [
            s for s in all_swings if s.confirmed_time <= as_of_time and s not in mitigated_swings
        ]
        if not available_swings:
            continue

        current_pools = detect_liquidity_pools(
            available_swings,
            tolerance=tolerance,
            atr=atr,
            atr_multiplier=atr_multiplier,
            include_single_swings=include_single_swings,
            as_of_time=as_of_time,
        )

        for pool in current_pools:
            if pool.type == LiquidityType.BSL:
                if candle.high > pool.price and candle.close <= pool.price:
                    swept_pool = dataclasses.replace(
                        pool,
                        swept=True,
                        swept_at=candle.time,
                        swept_by_candle=candle,
                    )
                    sweeps.append(
                        LiquiditySweep(
                            time=candle.time,
                            confirmed_time=as_of_time,
                            type=LiquidityType.BSL,
                            level=pool.price,
                            candle=candle,
                            pool=swept_pool,
                        )
                    )
            else:
                if candle.low < pool.price and candle.close >= pool.price:
                    swept_pool = dataclasses.replace(
                        pool,
                        swept=True,
                        swept_at=candle.time,
                        swept_by_candle=candle,
                    )
                    sweeps.append(
                        LiquiditySweep(
                            time=candle.time,
                            confirmed_time=as_of_time,
                            type=LiquidityType.SSL,
                            level=pool.price,
                            candle=candle,
                            pool=swept_pool,
                        )
                    )

        for s in available_swings:
            if (s.type == SwingType.HIGH and candle.high > s.price) or (
                s.type == SwingType.LOW and candle.low < s.price
            ):
                mitigated_swings.add(s)

    return sweeps


def calculate_dealing_range(high_swing: Swing, low_swing: Swing) -> DealingRange:
    return DealingRange(
        high=high_swing.price,
        low=low_swing.price,
        high_swing=high_swing,
        low_swing=low_swing,
    )


def get_current_dealing_range(
    swings: Sequence[Swing],
    as_of_time: datetime | None = None,
) -> DealingRange | None:
    if not swings:
        return None

    if as_of_time is not None:
        if as_of_time.tzinfo is None:
            raise ValueError("as_of_time must be timezone-aware")
        filtered = [s for s in swings if s.confirmed_time <= as_of_time]
    else:
        filtered = list(swings)

    sorted_swings = sorted(filtered, key=lambda s: (s.confirmed_time, s.index))
    highs = [s for s in sorted_swings if s.type == SwingType.HIGH]
    lows = [s for s in sorted_swings if s.type == SwingType.LOW]

    if not highs or not lows:
        return None

    latest_high = highs[-1]
    latest_low = lows[-1]

    if latest_high.price > latest_low.price:
        return calculate_dealing_range(latest_high, latest_low)

    return None
