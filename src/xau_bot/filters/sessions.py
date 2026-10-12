from datetime import UTC, datetime, tzinfo
from enum import StrEnum
from zoneinfo import ZoneInfo


class SessionName(StrEnum):
    ASIA = "ASIA"
    LONDON_OPEN = "LONDON_OPEN"
    NEW_YORK_OPEN = "NEW_YORK_OPEN"
    LONDON_CLOSE = "LONDON_CLOSE"


class TradeBlockReason(StrEnum):
    WEEKEND = "WEEKEND"
    ROLLOVER = "ROLLOVER"
    OUTSIDE_KILL_ZONE = "OUTSIDE_KILL_ZONE"
    OPENING_BLACKOUT = "OPENING_BLACKOUT"
    SPREAD_ABOVE_THRESHOLD = "SPREAD_ABOVE_THRESHOLD"
    NEWS_BLACKOUT = "NEWS_BLACKOUT"
    NEWS_CALENDAR_STALE = "NEWS_CALENDAR_STALE"
    MARKET_DATA_UNAVAILABLE = "MARKET_DATA_UNAVAILABLE"


def _ensure_timezone_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")
    return dt


def to_ny_time(dt: datetime) -> datetime:
    return _ensure_timezone_aware(dt).astimezone(ZoneInfo("America/New_York"))


def to_london_time(dt: datetime) -> datetime:
    return _ensure_timezone_aware(dt).astimezone(ZoneInfo("Europe/London"))


def to_server_time(dt: datetime, server_tz: tzinfo | None = None) -> datetime:
    tz = server_tz or UTC
    return _ensure_timezone_aware(dt).astimezone(tz)


def to_utc(dt: datetime) -> datetime:
    return _ensure_timezone_aware(dt).astimezone(UTC)


class SessionFilter:
    def __init__(
        self,
        ny_tz: str | tzinfo = "America/New_York",
        london_tz: str | tzinfo = "Europe/London",
        server_tz: str | tzinfo = "UTC",
        enable_24_7: bool = False,
    ) -> None:
        self.ny_tz = ZoneInfo(ny_tz) if isinstance(ny_tz, str) else ny_tz
        self.london_tz = ZoneInfo(london_tz) if isinstance(london_tz, str) else london_tz
        self.server_tz = ZoneInfo(server_tz) if isinstance(server_tz, str) else server_tz
        self.enable_24_7 = bool(enable_24_7)

    def to_server_time(self, dt: datetime) -> datetime:
        return to_server_time(dt, server_tz=self.server_tz)

    def is_weekend(self, dt: datetime) -> bool:
        if self.enable_24_7:
            return False
        ny = _ensure_timezone_aware(dt).astimezone(self.ny_tz)
        weekday = ny.weekday()
        time_minutes = ny.hour * 60 + ny.minute

        if weekday == 4:
            return time_minutes >= 17 * 60
        if weekday == 5:
            return True
        if weekday == 6:
            return time_minutes < 18 * 60
        return False

    def is_in_rollover(self, dt: datetime) -> bool:
        if self.enable_24_7 or self.is_weekend(dt):
            return False
        ny = _ensure_timezone_aware(dt).astimezone(self.ny_tz)
        return 17 <= ny.hour < 18

    def is_market_open(self, dt: datetime) -> bool:
        if self.enable_24_7:
            return True
        return not self.is_weekend(dt) and not self.is_in_rollover(dt)

    def is_in_kill_zone(self, dt: datetime) -> tuple[bool, SessionName | None]:
        aware = _ensure_timezone_aware(dt)
        london = aware.astimezone(self.london_tz)
        ny = aware.astimezone(self.ny_tz)
        utc = aware.astimezone(UTC)

        london_minutes = london.hour * 60 + london.minute
        if 7 * 60 <= london_minutes < 10 * 60:
            return True, SessionName.LONDON_OPEN

        ny_minutes = ny.hour * 60 + ny.minute
        if 7 * 60 <= ny_minutes < 10 * 60:
            return True, SessionName.NEW_YORK_OPEN

        if 15 * 60 <= london_minutes < 17 * 60:
            return True, SessionName.LONDON_CLOSE

        utc_minutes = utc.hour * 60 + utc.minute
        if 0 <= utc_minutes < 9 * 60:
            return True, SessionName.ASIA

        return False, None

    def is_in_opening_blackout(self, dt: datetime) -> bool:
        aware = _ensure_timezone_aware(dt)
        london = aware.astimezone(self.london_tz)
        ny = aware.astimezone(self.ny_tz)

        london_minutes = london.hour * 60 + london.minute
        if 7 * 60 <= london_minutes < 7 * 60 + 15:
            return True

        ny_minutes = ny.hour * 60 + ny.minute
        return 7 * 60 <= ny_minutes < 7 * 60 + 15

    def can_trade(self, dt: datetime) -> tuple[bool, TradeBlockReason | None]:
        _ensure_timezone_aware(dt)
        if self.enable_24_7:
            return True, None
        if self.is_weekend(dt):
            return False, TradeBlockReason.WEEKEND
        if self.is_in_rollover(dt):
            return False, TradeBlockReason.ROLLOVER
        in_kz, _ = self.is_in_kill_zone(dt)
        if not in_kz:
            return False, TradeBlockReason.OUTSIDE_KILL_ZONE
        if self.is_in_opening_blackout(dt):
            return False, TradeBlockReason.OPENING_BLACKOUT
        return True, None
