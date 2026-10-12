from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from xau_bot.filters.sessions import (
    SessionFilter,
    SessionName,
    TradeBlockReason,
    to_london_time,
    to_ny_time,
    to_server_time,
    to_utc,
)


def test_timezone_conversion_dst_shift() -> None:
    us_dst_uk_std = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

    ny_time = to_ny_time(us_dst_uk_std)
    assert ny_time.hour == 8
    assert ny_time.tzname() == "EDT"

    london_time = to_london_time(us_dst_uk_std)
    assert london_time.hour == 12
    assert london_time.tzname() == "GMT"

    both_dst = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
    ny_time_summer = to_ny_time(both_dst)
    assert ny_time_summer.hour == 8
    assert ny_time_summer.tzname() == "EDT"

    london_time_summer = to_london_time(both_dst)
    assert london_time_summer.hour == 13
    assert london_time_summer.tzname() == "BST"


def test_server_timezone_conversion() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    utc_time = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    server_time = to_server_time(utc_time, server_tz=broker_tz)
    assert server_time.hour == 14

    back_to_utc = to_utc(server_time)
    assert back_to_utc == utc_time


def test_session_filter_to_server_time() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    sf = SessionFilter(server_tz=broker_tz)
    utc_time = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    server_time = sf.to_server_time(utc_time)
    assert server_time.hour == 14
    assert server_time.tzinfo == broker_tz


def test_session_filter_kill_zones() -> None:
    sf = SessionFilter()

    london_kz_time = datetime(2026, 9, 24, 7, 30, tzinfo=ZoneInfo("Europe/London"))
    in_kz, name = sf.is_in_kill_zone(london_kz_time)
    assert in_kz is True
    assert name == SessionName.LONDON_OPEN

    ny_kz_time = datetime(2026, 9, 24, 8, 30, tzinfo=ZoneInfo("America/New_York"))
    in_kz, name = sf.is_in_kill_zone(ny_kz_time)
    assert in_kz is True
    assert name == SessionName.NEW_YORK_OPEN

    london_close_time = datetime(2026, 9, 24, 15, 30, tzinfo=ZoneInfo("Europe/London"))
    in_kz, name = sf.is_in_kill_zone(london_close_time)
    assert in_kz is True
    assert name == SessionName.LONDON_CLOSE

    asia_time = datetime(2026, 9, 24, 2, 0, tzinfo=UTC)
    in_kz, name = sf.is_in_kill_zone(asia_time)
    assert in_kz is True
    assert name == SessionName.ASIA

    dead_zone_time = datetime(2026, 9, 24, 13, 0, tzinfo=ZoneInfo("America/New_York"))
    in_kz, name = sf.is_in_kill_zone(dead_zone_time)
    assert in_kz is False
    assert name is None


def test_opening_blackout() -> None:
    sf = SessionFilter()

    london_opening = datetime(2026, 9, 24, 7, 5, tzinfo=ZoneInfo("Europe/London"))
    assert sf.is_in_opening_blackout(london_opening) is True

    london_post_blackout = datetime(2026, 9, 24, 7, 20, tzinfo=ZoneInfo("Europe/London"))
    assert sf.is_in_opening_blackout(london_post_blackout) is False

    ny_opening = datetime(2026, 9, 24, 7, 10, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_in_opening_blackout(ny_opening) is True

    ny_post_blackout = datetime(2026, 9, 24, 7, 25, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_in_opening_blackout(ny_post_blackout) is False


def test_rollover_filter() -> None:
    sf = SessionFilter()

    rollover_time = datetime(2026, 9, 24, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_in_rollover(rollover_time) is True

    normal_trading_time = datetime(2026, 9, 24, 16, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_in_rollover(normal_trading_time) is False


def test_weekend_detection() -> None:
    sf = SessionFilter()

    friday_after_close = datetime(2026, 9, 25, 17, 15, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_weekend(friday_after_close) is True

    saturday_midday = datetime(2026, 9, 26, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_weekend(saturday_midday) is True

    sunday_before_open = datetime(2026, 9, 27, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_weekend(sunday_before_open) is True

    sunday_after_open = datetime(2026, 9, 27, 18, 5, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_weekend(sunday_after_open) is False

    wednesday_trading = datetime(2026, 9, 23, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_weekend(wednesday_trading) is False


def test_can_trade_evaluations() -> None:
    sf = SessionFilter()

    valid_ny_trade = datetime(2026, 9, 24, 8, 30, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = sf.can_trade(valid_ny_trade)
    assert allowed is True
    assert reason is None

    blackout_trade = datetime(2026, 9, 24, 7, 5, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = sf.can_trade(blackout_trade)
    assert allowed is False
    assert reason == TradeBlockReason.OPENING_BLACKOUT

    rollover_trade = datetime(2026, 9, 24, 17, 15, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = sf.can_trade(rollover_trade)
    assert allowed is False
    assert reason == TradeBlockReason.ROLLOVER

    weekend_trade = datetime(2026, 9, 26, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = sf.can_trade(weekend_trade)
    assert allowed is False
    assert reason == TradeBlockReason.WEEKEND

    dead_zone_trade = datetime(2026, 9, 24, 13, 0, tzinfo=ZoneInfo("America/New_York"))
    allowed, reason = sf.can_trade(dead_zone_trade)
    assert allowed is False
    assert reason == TradeBlockReason.OUTSIDE_KILL_ZONE


def test_session_filter_naive_datetime_rejected() -> None:
    sf = SessionFilter()
    naive_dt = datetime(2026, 9, 24, 12, 0)
    with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
        sf.can_trade(naive_dt)


def test_rollover_on_weekend_returns_false() -> None:
    sf = SessionFilter()
    saturday_1730 = datetime(2026, 9, 26, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_in_rollover(saturday_1730) is False


def test_is_market_open() -> None:
    sf = SessionFilter()
    wednesday_open = datetime(2026, 9, 23, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_market_open(wednesday_open) is True

    wednesday_rollover = datetime(2026, 9, 23, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_market_open(wednesday_rollover) is False

    saturday_weekend = datetime(2026, 9, 26, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert sf.is_market_open(saturday_weekend) is False


@pytest.mark.parametrize(
    (
        "dt_utc",
        "expected_in_kz",
        "expected_session",
        "expected_blackout",
        "expected_can_trade",
        "expected_reason",
    ),
    [
        (
            datetime(2026, 3, 16, 11, 5, tzinfo=UTC),
            True,
            SessionName.NEW_YORK_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 3, 16, 11, 20, tzinfo=UTC),
            True,
            SessionName.NEW_YORK_OPEN,
            False,
            True,
            None,
        ),
        (
            datetime(2026, 3, 16, 10, 55, tzinfo=UTC),
            False,
            None,
            False,
            False,
            TradeBlockReason.OUTSIDE_KILL_ZONE,
        ),
        (
            datetime(2026, 3, 16, 7, 5, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 3, 16, 7, 25, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            False,
            True,
            None,
        ),
        (
            datetime(2026, 10, 28, 11, 5, tzinfo=UTC),
            True,
            SessionName.NEW_YORK_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 10, 28, 11, 30, tzinfo=UTC),
            True,
            SessionName.NEW_YORK_OPEN,
            False,
            True,
            None,
        ),
        (
            datetime(2026, 10, 28, 7, 10, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 10, 28, 7, 30, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            False,
            True,
            None,
        ),
        (
            datetime(2026, 1, 14, 11, 5, tzinfo=UTC),
            False,
            None,
            False,
            False,
            TradeBlockReason.OUTSIDE_KILL_ZONE,
        ),
        (
            datetime(2026, 1, 14, 12, 5, tzinfo=UTC),
            True,
            SessionName.NEW_YORK_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 6, 17, 6, 5, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            True,
            False,
            TradeBlockReason.OPENING_BLACKOUT,
        ),
        (
            datetime(2026, 6, 17, 7, 5, tzinfo=UTC),
            True,
            SessionName.LONDON_OPEN,
            False,
            True,
            None,
        ),
    ],
)
def test_kill_zones_and_blackout_during_dst_shift_weeks(
    dt_utc: datetime,
    expected_in_kz: bool,
    expected_session: SessionName | None,
    expected_blackout: bool,
    expected_can_trade: bool,
    expected_reason: TradeBlockReason | None,
) -> None:
    sf = SessionFilter()
    in_kz, session = sf.is_in_kill_zone(dt_utc)
    assert in_kz is expected_in_kz
    assert session == expected_session

    blackout = sf.is_in_opening_blackout(dt_utc)
    assert blackout is expected_blackout

    can_trade, reason = sf.can_trade(dt_utc)
    assert can_trade is expected_can_trade
    assert reason == expected_reason


def test_session_filter_24_7_mode() -> None:
    sf_247 = SessionFilter(enable_24_7=True)

    # 1. Saturday afternoon (weekend for forex)
    saturday_time = datetime(2026, 9, 26, 15, 0, tzinfo=UTC)
    assert sf_247.is_weekend(saturday_time) is False
    assert sf_247.is_market_open(saturday_time) is True
    can_trade, reason = sf_247.can_trade(saturday_time)
    assert can_trade is True
    assert reason is None

    # 2. Rollover hour in NY
    rollover_time = datetime(2026, 9, 24, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    assert sf_247.is_in_rollover(rollover_time) is False
    assert sf_247.is_market_open(rollover_time) is True
    can_trade_ro, reason_ro = sf_247.can_trade(rollover_time)
    assert can_trade_ro is True
    assert reason_ro is None

    # 3. Dead zone outside kill zones
    dead_zone_time = datetime(2026, 9, 24, 13, 0, tzinfo=ZoneInfo("America/New_York"))
    can_trade_dz, reason_dz = sf_247.can_trade(dead_zone_time)
    assert can_trade_dz is True
    assert reason_dz is None
