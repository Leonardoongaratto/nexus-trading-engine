import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum


class Direction(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class Timeframe(StrEnum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def duration_seconds(self) -> int:
        match self:
            case Timeframe.M1:
                return 60
            case Timeframe.M5:
                return 300
            case Timeframe.M15:
                return 900
            case Timeframe.H1:
                return 3600
            case Timeframe.H4:
                return 14400
            case Timeframe.D1:
                return 86400


class SwingType(StrEnum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timeframe: Timeframe = Timeframe.M5

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        for name, val in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ):
            if not math.isfinite(val):
                raise ValueError(f"Candle field '{name}' must be finite, got {val}")
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("Prices must be positive")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"High cannot be less than open or close "
                f"(open={self.open}, close={self.close}, high={self.high})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"Low cannot be greater than open or close "
                f"(open={self.open}, close={self.close}, low={self.low})"
            )
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_range(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass(frozen=True, slots=True)
class Tick:
    time: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    flags: int = 0

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        for name, val in (
            ("bid", self.bid),
            ("ask", self.ask),
            ("last", self.last),
            ("volume", self.volume),
        ):
            if not math.isfinite(val):
                raise ValueError(f"Tick field '{name}' must be finite, got {val}")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("Prices must be positive")
        if self.last < 0:
            raise ValueError("Last price cannot be negative")
        if self.ask < self.bid:
            raise ValueError(f"Ask cannot be less than bid (bid={self.bid}, ask={self.ask})")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Swing:
    time: datetime
    price: float
    type: SwingType
    confirmed_time: datetime
    index: int

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("Swing timestamp must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("confirmed_time must be timezone-aware")
        if self.confirmed_time <= self.time:
            raise ValueError(
                f"confirmed_time must be strictly after swing time: "
                f"time={self.time}, confirmed_time={self.confirmed_time}"
            )
        if not math.isfinite(self.price):
            raise ValueError(f"Swing price must be finite, got {self.price}")
        if self.price <= 0:
            raise ValueError(f"Swing price must be positive, got {self.price}")
        if self.index < 0:
            raise ValueError(f"Index cannot be negative, got {self.index}")
        if not isinstance(self.type, SwingType):
            raise ValueError(f"Invalid swing type: {self.type!r}")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    direction: Direction
    order_type: str
    volume: float
    price: float
    stop_loss: float
    take_profit: float
    magic: int
    comment: str
    expiration: datetime | None = None


class LiquidityType(StrEnum):
    BSL = "BSL"
    SSL = "SSL"


@dataclass(frozen=True, slots=True)
class BOS:
    time: datetime
    confirmed_time: datetime
    direction: Direction
    broken_swing: Swing
    breaking_candle: Candle
    price: float

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("BOS time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("BOS confirmed_time must be timezone-aware")
        if self.confirmed_time <= self.time:
            raise ValueError(
                f"confirmed_time must be strictly after BOS time: "
                f"time={self.time}, confirmed_time={self.confirmed_time}"
            )
        if not math.isfinite(self.price):
            raise ValueError(f"BOS price must be finite, got {self.price}")
        if self.price <= 0:
            raise ValueError(f"BOS price must be positive, got {self.price}")
        if not isinstance(self.direction, Direction):
            raise ValueError(f"Invalid direction: {self.direction!r}")
        if self.direction == Direction.BULLISH:
            if self.broken_swing.type != SwingType.HIGH:
                raise ValueError(
                    f"Bullish BOS requires broken_swing of type HIGH, got {self.broken_swing.type}"
                )
            if self.breaking_candle.close <= self.broken_swing.price:
                raise ValueError(
                    f"Bullish BOS requires candle close > broken_swing price "
                    f"(close={self.breaking_candle.close}, price={self.broken_swing.price})"
                )
        else:
            if self.broken_swing.type != SwingType.LOW:
                raise ValueError(
                    f"Bearish BOS requires broken_swing of type LOW, got {self.broken_swing.type}"
                )
            if self.breaking_candle.close >= self.broken_swing.price:
                raise ValueError(
                    f"Bearish BOS requires candle close < broken_swing price "
                    f"(close={self.breaking_candle.close}, price={self.broken_swing.price})"
                )


@dataclass(frozen=True, slots=True)
class ChoCH:
    time: datetime
    confirmed_time: datetime
    direction: Direction
    previous_trend: Direction
    broken_swing: Swing
    breaking_candle: Candle
    price: float

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("ChoCH time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("ChoCH confirmed_time must be timezone-aware")
        if self.confirmed_time <= self.time:
            raise ValueError(
                f"confirmed_time must be strictly after ChoCH time: "
                f"time={self.time}, confirmed_time={self.confirmed_time}"
            )
        if not math.isfinite(self.price):
            raise ValueError(f"ChoCH price must be finite, got {self.price}")
        if self.price <= 0:
            raise ValueError(f"ChoCH price must be positive, got {self.price}")
        if not isinstance(self.direction, Direction):
            raise ValueError(f"Invalid direction: {self.direction!r}")
        if not isinstance(self.previous_trend, Direction):
            raise ValueError(f"Invalid previous_trend: {self.previous_trend!r}")
        if self.direction == self.previous_trend:
            raise ValueError(
                f"ChoCH requires reversal "
                f"(direction={self.direction} != previous_trend={self.previous_trend})"
            )
        if self.direction == Direction.BULLISH:
            if self.broken_swing.type != SwingType.HIGH:
                raise ValueError(
                    f"Bullish ChoCH requires broken_swing of type HIGH, "
                    f"got {self.broken_swing.type}"
                )
            if self.breaking_candle.close <= self.broken_swing.price:
                raise ValueError(
                    f"Bullish ChoCH requires candle close > broken_swing price "
                    f"(close={self.breaking_candle.close}, price={self.broken_swing.price})"
                )
        else:
            if self.broken_swing.type != SwingType.LOW:
                raise ValueError(
                    f"Bearish ChoCH requires broken_swing of type LOW, got {self.broken_swing.type}"
                )
            if self.breaking_candle.close >= self.broken_swing.price:
                raise ValueError(
                    f"Bearish ChoCH requires candle close < broken_swing price "
                    f"(close={self.breaking_candle.close}, price={self.broken_swing.price})"
                )


@dataclass(frozen=True, slots=True)
class LiquidityPool:
    type: LiquidityType
    price: float
    swings: tuple[Swing, ...]
    time: datetime
    confirmed_time: datetime
    swept: bool = False
    swept_at: datetime | None = None
    swept_by_candle: Candle | None = None

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("LiquidityPool time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("LiquidityPool confirmed_time must be timezone-aware")
        if self.confirmed_time < self.time:
            raise ValueError(
                f"confirmed_time must be >= time: "
                f"time={self.time}, confirmed_time={self.confirmed_time}"
            )
        if self.swept_at is not None and self.swept_at.tzinfo is None:
            raise ValueError("swept_at must be timezone-aware")
        if self.swept:
            if self.swept_at is None:
                raise ValueError("swept=True requires swept_at to be set")
            if self.swept_by_candle is None:
                raise ValueError("swept=True requires swept_by_candle to be set")
        else:
            if self.swept_at is not None:
                raise ValueError("swept=False requires swept_at to be None")
            if self.swept_by_candle is not None:
                raise ValueError("swept=False requires swept_by_candle to be None")
        if not math.isfinite(self.price):
            raise ValueError(f"LiquidityPool price must be finite, got {self.price}")
        if self.price <= 0:
            raise ValueError(f"LiquidityPool price must be positive, got {self.price}")
        if not isinstance(self.type, LiquidityType):
            raise ValueError(f"Invalid liquidity type: {self.type!r}")
        if not self.swings:
            raise ValueError("LiquidityPool swings cannot be empty")
        expected_type = SwingType.HIGH if self.type == LiquidityType.BSL else SwingType.LOW
        for s in self.swings:
            if s.type != expected_type:
                raise ValueError(
                    f"{self.type} pool requires swings of type {expected_type.name}, "
                    f"got swing of type {s.type.name}"
                )


@dataclass(frozen=True, slots=True)
class LiquiditySweep:
    time: datetime
    confirmed_time: datetime
    type: LiquidityType
    level: float
    candle: Candle
    pool: LiquidityPool

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("LiquiditySweep time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("LiquiditySweep confirmed_time must be timezone-aware")
        if self.confirmed_time <= self.time:
            raise ValueError("confirmed_time must be strictly after sweep time")
        if not math.isfinite(self.level):
            raise ValueError(f"Sweep level must be finite, got {self.level}")
        if self.level <= 0:
            raise ValueError(f"Sweep level must be positive, got {self.level}")
        if not isinstance(self.type, LiquidityType):
            raise ValueError(f"Invalid liquidity type: {self.type!r}")
        if self.type == LiquidityType.BSL:
            if self.candle.high <= self.level:
                raise ValueError(
                    f"BSL sweep requires candle high > level ({self.candle.high} <= {self.level})"
                )
            if self.candle.close > self.level:
                raise ValueError(
                    f"BSL sweep requires candle close <= level ({self.candle.close} > {self.level})"
                )
        else:
            if self.candle.low >= self.level:
                raise ValueError(
                    f"SSL sweep requires candle low < level ({self.candle.low} >= {self.level})"
                )
            if self.candle.close < self.level:
                raise ValueError(
                    f"SSL sweep requires candle close >= level ({self.candle.close} < {self.level})"
                )


OTE_LOWER: float = 0.62
OTE_UPPER: float = 0.79


@dataclass(frozen=True, slots=True)
class DealingRange:
    high: float
    low: float
    high_swing: Swing | None = None
    low_swing: Swing | None = None

    def __post_init__(self) -> None:
        for name, val in (("high", self.high), ("low", self.low)):
            if not math.isfinite(val):
                raise ValueError(f"DealingRange {name} must be finite, got {val}")
        if self.low <= 0:
            raise ValueError(f"DealingRange low must be positive, got {self.low}")
        if self.high <= self.low:
            raise ValueError(
                f"high must be strictly greater than low (high={self.high}, low={self.low})"
            )
        if self.high_swing is not None and self.high_swing.type != SwingType.HIGH:
            raise ValueError(f"high_swing must be of type HIGH, got {self.high_swing.type}")
        if self.low_swing is not None and self.low_swing.type != SwingType.LOW:
            raise ValueError(f"low_swing must be of type LOW, got {self.low_swing.type}")

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    def is_premium(self, price: float) -> bool:
        return price > self.equilibrium

    def is_discount(self, price: float) -> bool:
        return price < self.equilibrium

    def ote_zone(
        self,
        direction: Direction,
        lower_ratio: float = OTE_LOWER,
        upper_ratio: float = OTE_UPPER,
    ) -> tuple[float, float]:
        rng = self.range
        if direction == Direction.BULLISH:
            level_lower = self.high - lower_ratio * rng
            level_upper = self.high - upper_ratio * rng
            return (level_upper, level_lower)
        level_lower = self.low + lower_ratio * rng
        level_upper = self.low + upper_ratio * rng
        return (level_lower, level_upper)


class FVGType(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class FVGMitigationState(StrEnum):
    UNMITIGATED = "UNMITIGATED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
    CE_TESTED = "CE_TESTED"
    INVALIDATED = "INVALIDATED"


@dataclass(frozen=True, slots=True)
class FVG:
    time: datetime
    confirmed_time: datetime
    type: FVGType
    top: float
    bottom: float
    candle_0: Candle
    candle_1: Candle
    candle_2: Candle
    mitigated: bool = False
    mitigated_at: datetime | None = None
    mitigated_by_candle: Candle | None = None
    ce_tested: bool = False
    ce_tested_at: datetime | None = None
    invalidated: bool = False
    invalidated_at: datetime | None = None
    invalidated_by_candle: Candle | None = None

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("FVG time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("FVG confirmed_time must be timezone-aware")
        if self.confirmed_time <= self.time:
            raise ValueError("confirmed_time must be strictly after FVG time")
        duration = timedelta(seconds=self.candle_2.timeframe.duration_seconds)
        if self.confirmed_time < self.candle_2.time + duration:
            raise ValueError("confirmed_time must be at or after candle_2 close")
        if self.mitigated_at is not None and self.mitigated_at.tzinfo is None:
            raise ValueError("mitigated_at must be timezone-aware")
        if self.mitigated:
            if self.mitigated_at is None:
                raise ValueError("mitigated=True requires mitigated_at to be set")
            if self.mitigated_by_candle is None:
                raise ValueError("mitigated=True requires mitigated_by_candle to be set")
            if self.mitigated_at < self.confirmed_time:
                raise ValueError("mitigated_at must be >= confirmed_time")
        else:
            if self.mitigated_at is not None:
                raise ValueError("mitigated=False requires mitigated_at to be None")
            if self.mitigated_by_candle is not None:
                raise ValueError("mitigated=False requires mitigated_by_candle to be None")
        if self.ce_tested_at is not None and self.ce_tested_at.tzinfo is None:
            raise ValueError("ce_tested_at must be timezone-aware")
        if self.ce_tested:
            if self.ce_tested_at is None:
                raise ValueError("ce_tested=True requires ce_tested_at to be set")
            if self.ce_tested_at < self.confirmed_time:
                raise ValueError("ce_tested_at must be >= confirmed_time")
        else:
            if self.ce_tested_at is not None:
                raise ValueError("ce_tested=False requires ce_tested_at to be None")
        if self.invalidated_at is not None and self.invalidated_at.tzinfo is None:
            raise ValueError("invalidated_at must be timezone-aware")
        if self.invalidated:
            if self.invalidated_at is None:
                raise ValueError("invalidated=True requires invalidated_at to be set")
            if self.invalidated_by_candle is None:
                raise ValueError("invalidated=True requires invalidated_by_candle to be set")
            if self.invalidated_at < self.confirmed_time:
                raise ValueError("invalidated_at must be >= confirmed_time")
        else:
            if self.invalidated_at is not None:
                raise ValueError("invalidated=False requires invalidated_at to be None")
            if self.invalidated_by_candle is not None:
                raise ValueError("invalidated=False requires invalidated_by_candle to be None")
        for name, val in (("top", self.top), ("bottom", self.bottom)):
            if not math.isfinite(val):
                raise ValueError(f"FVG {name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"FVG {name} must be positive, got {val}")
        if self.top <= self.bottom:
            raise ValueError(
                f"top must be strictly greater than bottom (top={self.top}, bottom={self.bottom})"
            )
        if not isinstance(self.type, FVGType):
            raise ValueError(f"Invalid FVG type: {self.type!r}")
        if self.type == FVGType.BULLISH:
            if self.candle_2.low <= self.candle_0.high:
                raise ValueError(
                    f"Bullish FVG requires candle_2.low > candle_0.high "
                    f"({self.candle_2.low} <= {self.candle_0.high})"
                )
            if self.top != self.candle_2.low or self.bottom != self.candle_0.high:
                raise ValueError(
                    f"Bullish FVG levels must match candles: "
                    f"top={self.top} vs {self.candle_2.low}, "
                    f"bottom={self.bottom} vs {self.candle_0.high}"
                )
        else:
            if self.candle_2.high >= self.candle_0.low:
                raise ValueError(
                    f"Bearish FVG requires candle_2.high < candle_0.low "
                    f"({self.candle_2.high} >= {self.candle_0.low})"
                )
            if self.top != self.candle_0.low or self.bottom != self.candle_2.high:
                raise ValueError(
                    f"Bearish FVG levels must match candles: "
                    f"top={self.top} vs {self.candle_0.low}, "
                    f"bottom={self.bottom} vs {self.candle_2.high}"
                )

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def consequent_encroachment(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def state(self) -> FVGMitigationState:
        if self.invalidated:
            return FVGMitigationState.INVALIDATED
        if self.ce_tested:
            return FVGMitigationState.CE_TESTED
        if self.mitigated:
            return FVGMitigationState.PARTIALLY_MITIGATED
        return FVGMitigationState.UNMITIGATED


class OrderBlockType(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass(frozen=True, slots=True)
class OrderBlock:
    time: datetime
    confirmed_time: datetime
    type: OrderBlockType
    high: float
    low: float
    open: float
    close: float
    candle: Candle
    displacement_atr_ratio: float = 0.0
    breaking_event: BOS | ChoCH | None = None
    fvg: FVG | None = None
    mitigated: bool = False
    mitigated_at: datetime | None = None
    mitigated_by_candle: Candle | None = None
    invalidated: bool = False
    invalidated_at: datetime | None = None
    invalidated_by_candle: Candle | None = None

    def __post_init__(self) -> None:
        if self.time.tzinfo is None:
            raise ValueError("OrderBlock time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("OrderBlock confirmed_time must be timezone-aware")
        duration = timedelta(seconds=self.candle.timeframe.duration_seconds)
        if self.confirmed_time < self.time + duration:
            raise ValueError(
                f"confirmed_time must be at or after candle close "
                f"({self.confirmed_time} < {self.time + duration})"
            )
        if self.mitigated_at is not None and self.mitigated_at.tzinfo is None:
            raise ValueError("mitigated_at must be timezone-aware")
        if self.mitigated:
            if self.mitigated_at is None:
                raise ValueError("mitigated=True requires mitigated_at to be set")
            if self.mitigated_by_candle is None:
                raise ValueError("mitigated=True requires mitigated_by_candle to be set")
            if self.mitigated_at < self.confirmed_time:
                raise ValueError("mitigated_at must be >= confirmed_time")
        else:
            if self.mitigated_at is not None:
                raise ValueError("mitigated=False requires mitigated_at to be None")
            if self.mitigated_by_candle is not None:
                raise ValueError("mitigated=False requires mitigated_by_candle to be None")
        if self.invalidated_at is not None and self.invalidated_at.tzinfo is None:
            raise ValueError("invalidated_at must be timezone-aware")
        if self.invalidated:
            if self.invalidated_at is None:
                raise ValueError("invalidated=True requires invalidated_at to be set")
            if self.invalidated_by_candle is None:
                raise ValueError("invalidated=True requires invalidated_by_candle to be set")
            if self.invalidated_at < self.confirmed_time:
                raise ValueError("invalidated_at must be >= confirmed_time")
        else:
            if self.invalidated_at is not None:
                raise ValueError("invalidated=False requires invalidated_at to be None")
            if self.invalidated_by_candle is not None:
                raise ValueError("invalidated=False requires invalidated_by_candle to be None")
        for name, val in (
            ("high", self.high),
            ("low", self.low),
            ("open", self.open),
            ("close", self.close),
        ):
            if not math.isfinite(val):
                raise ValueError(f"OrderBlock {name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"OrderBlock {name} must be positive, got {val}")
        if self.high <= self.low:
            raise ValueError(
                f"high must be strictly greater than low (high={self.high}, low={self.low})"
            )
        if (
            self.high != self.candle.high
            or self.low != self.candle.low
            or self.open != self.candle.open
            or self.close != self.candle.close
        ):
            raise ValueError("OrderBlock prices must match candle prices")
        if not isinstance(self.type, OrderBlockType):
            raise ValueError(f"Invalid OrderBlock type: {self.type!r}")
        if self.breaking_event is not None:
            expected_dir = (
                Direction.BULLISH if self.type == OrderBlockType.BULLISH else Direction.BEARISH
            )
            if self.breaking_event.direction != expected_dir:
                raise ValueError(
                    f"OrderBlock direction mismatch with breaking_event: "
                    f"{self.type} vs {self.breaking_event.direction}"
                )

    @property
    def mean_threshold(self) -> float:
        return (self.high + self.low) / 2.0


class SetupTriggerType(StrEnum):
    FVG_CE = "FVG_CE"
    FVG_PROXIMAL = "FVG_PROXIMAL"
    OB_EXTREME = "OB_EXTREME"
    OB_MEAN_THRESHOLD = "OB_MEAN_THRESHOLD"
    OTE_RETRACEMENT = "OTE_RETRACEMENT"


class SetupStatus(StrEnum):
    PENDING = "PENDING"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class TradeSetup:
    id: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    time: datetime
    confirmed_time: datetime
    risk_reward_ratio: float
    confluences: tuple[str, ...] = ()
    valid_until: datetime | None = None
    status: SetupStatus = SetupStatus.PENDING
    trigger_type: SetupTriggerType = SetupTriggerType.FVG_CE
    timeframe: Timeframe = Timeframe.M5
    sweep: LiquiditySweep | None = None
    choch: ChoCH | None = None
    poi: FVG | OrderBlock | None = None
    status_updated_at: datetime | None = None
    target_swing: Swing | None = None
    dealing_range: DealingRange | None = None

    @property
    def risk_reward(self) -> float:
        return self.risk_reward_ratio

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("setup id must not be empty")
        if self.time.tzinfo is None:
            raise ValueError("time must be timezone-aware")
        if self.confirmed_time.tzinfo is None:
            raise ValueError("confirmed_time must be timezone-aware")
        if self.confirmed_time < self.time:
            raise ValueError(
                f"confirmed_time must be >= time: "
                f"time={self.time}, confirmed_time={self.confirmed_time}"
            )
        if self.valid_until is not None:
            if self.valid_until.tzinfo is None:
                raise ValueError("valid_until must be timezone-aware")
            if self.valid_until < self.confirmed_time:
                raise ValueError("valid_until must be >= confirmed_time")
        if self.status_updated_at is not None:
            if self.status_updated_at.tzinfo is None:
                raise ValueError("status_updated_at must be timezone-aware")
            if self.status_updated_at < self.confirmed_time:
                raise ValueError("status_updated_at must be >= confirmed_time")

        for name, val in (
            ("entry_price", self.entry_price),
            ("stop_loss", self.stop_loss),
            ("take_profit", self.take_profit),
            ("risk_reward_ratio", self.risk_reward_ratio),
        ):
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

        if self.direction == Direction.BULLISH:
            if not (self.stop_loss < self.entry_price < self.take_profit):
                raise ValueError(
                    f"bullish setup requires stop_loss < entry_price < take_profit: "
                    f"sl={self.stop_loss}, entry={self.entry_price}, tp={self.take_profit}"
                )
            expected_rr = (self.take_profit - self.entry_price) / (
                self.entry_price - self.stop_loss
            )
        elif self.direction == Direction.BEARISH:
            if not (self.stop_loss > self.entry_price > self.take_profit):
                raise ValueError(
                    f"bearish setup requires stop_loss > entry_price > take_profit: "
                    f"sl={self.stop_loss}, entry={self.entry_price}, tp={self.take_profit}"
                )
            expected_rr = (self.entry_price - self.take_profit) / (
                self.stop_loss - self.entry_price
            )
        else:
            raise ValueError(f"Invalid direction: {self.direction!r}")

        if abs(self.risk_reward_ratio - expected_rr) > 1e-3:
            raise ValueError(
                f"risk_reward_ratio mismatch: "
                f"expected {expected_rr:.4f}, got {self.risk_reward_ratio:.4f}"
            )

        if not isinstance(self.status, SetupStatus):
            raise ValueError(f"status must be a SetupStatus enum, got {self.status!r}")

        if self.sweep is not None:
            expected_pool = (
                LiquidityType.SSL if self.direction == Direction.BULLISH else LiquidityType.BSL
            )
            if self.sweep.type != expected_pool:
                dir_name = self.direction.value.lower()
                raise ValueError(f"{dir_name} setup requires {expected_pool.value} liquidity sweep")
        if self.choch is not None and self.choch.direction != self.direction:
            raise ValueError(
                f"choch direction must match setup direction: "
                f"{self.choch.direction} vs {self.direction}"
            )
        if self.poi is not None:
            poi_dir = self.poi.type.name
            if poi_dir != self.direction.name:
                raise ValueError(
                    f"POI direction must match setup direction: {poi_dir} vs {self.direction.name}"
                )


Setup = TradeSetup
