import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import (
    FilterCheckResult,
    NewsEvent,
    NewsFilterConfig,
    NewsImpact,
    SpreadFilterConfig,
)
from xau_bot.filters.news import (
    NewsFilter,
    load_calendar_from_file,
    parse_calendar_json,
)
from xau_bot.filters.sessions import SessionFilter, TradeBlockReason
from xau_bot.filters.spread import SpreadFilter


def test_spread_filter_config_validation() -> None:
    cfg = SpreadFilterConfig(max_spread_atr_ratio=0.05, max_spread_points=50.0)
    assert cfg.max_spread_atr_ratio == 0.05
    assert cfg.max_spread_points == 50.0

    with pytest.raises(ValueError, match="max_spread_atr_ratio must be positive"):
        SpreadFilterConfig(max_spread_atr_ratio=0.0)

    with pytest.raises(ValueError, match="max_spread_atr_ratio must be finite"):
        SpreadFilterConfig(max_spread_atr_ratio=float("nan"))

    with pytest.raises(ValueError, match="max_spread_points must be positive"):
        SpreadFilterConfig(max_spread_points=-1.0)

    with pytest.raises(ValueError, match="point_size must be positive"):
        SpreadFilterConfig(point_size=0.0)

    with pytest.raises(ValueError, match="point_size must be finite"):
        SpreadFilterConfig(point_size=float("nan"))


def test_spread_filter_blocks_when_ratio_exceeded() -> None:
    config = SpreadFilterConfig(max_spread_atr_ratio=0.05)
    filter_ = SpreadFilter(config)

    result = filter_.check(spread=0.30, atr=5.0)
    assert result.can_trade is False
    assert result.reason == TradeBlockReason.SPREAD_ABOVE_THRESHOLD
    assert "ratio 0.0600 > 0.0500" in result.details


def test_spread_filter_passes_when_ratio_within_limit() -> None:
    config = SpreadFilterConfig(max_spread_atr_ratio=0.05)
    filter_ = SpreadFilter(config)

    result = filter_.check(spread=0.20, atr=5.0)
    assert result.can_trade is True
    assert result.reason is None
    assert "within threshold" in result.details


def test_spread_filter_max_spread_points_limit() -> None:
    config = SpreadFilterConfig(max_spread_atr_ratio=0.10, max_spread_points=30.0, point_size=0.01)
    filter_ = SpreadFilter(config)

    result = filter_.check(spread=0.35, atr=10.0)
    assert result.can_trade is False
    assert result.reason == TradeBlockReason.SPREAD_ABOVE_THRESHOLD
    assert "exceeds maximum points 30.0" in result.details

    result_pass = filter_.check(spread=0.20, atr=10.0)
    assert result_pass.can_trade is True
    assert result_pass.reason is None


def test_spread_filter_invalid_inputs_fail_closed() -> None:
    filter_ = SpreadFilter()

    with pytest.raises(ValueError, match="spread cannot be negative"):
        filter_.check(spread=-1.0, atr=5.0)

    with pytest.raises(ValueError, match="atr must be positive"):
        filter_.check(spread=0.20, atr=0.0)

    with pytest.raises(ValueError, match="spread must be finite"):
        filter_.check(spread=float("inf"), atr=5.0)

    with pytest.raises(ValueError, match="atr must be finite"):
        filter_.check(spread=0.20, atr=float("nan"))


def test_news_event_model_invariants() -> None:
    valid_time = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    event = NewsEvent(
        title="US Non-Farm Payrolls",
        currency="USD",
        impact=NewsImpact.HIGH,
        time=valid_time,
        actual=250.0,
        forecast=200.0,
        previous=180.0,
        source_id="nfp-2026-09",
    )
    assert event.title == "US Non-Farm Payrolls"
    assert event.currency == "USD"
    assert event.impact == NewsImpact.HIGH

    with pytest.raises(ValueError, match="title must not be empty"):
        NewsEvent(title="", currency="USD", impact=NewsImpact.HIGH, time=valid_time)

    with pytest.raises(ValueError, match="currency must not be empty"):
        NewsEvent(title="CPI", currency="", impact=NewsImpact.HIGH, time=valid_time)

    with pytest.raises(ValueError, match="time must be timezone-aware"):
        NewsEvent(
            title="CPI",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=datetime(2026, 9, 26, 12, 30),
        )


def test_parse_calendar_json_formats() -> None:
    raw_list = [
        {
            "title": "Fed Interest Rate Decision",
            "currency": "USD",
            "impact": "HIGH",
            "time": "2026-09-26T18:00:00Z",
        },
        {
            "title": "ECB Press Conference",
            "currency": "EUR",
            "impact": "MEDIUM",
            "time": "2026-09-26T12:30:00+00:00",
        },
    ]

    events = parse_calendar_json(json.dumps(raw_list))
    assert len(events) == 2
    assert events[0].title == "ECB Press Conference"
    assert events[0].currency == "EUR"
    assert events[0].impact == NewsImpact.MEDIUM
    assert events[0].time == datetime(2026, 9, 26, 12, 30, tzinfo=UTC)

    assert events[1].title == "Fed Interest Rate Decision"
    assert events[1].currency == "USD"
    assert events[1].impact == NewsImpact.HIGH
    assert events[1].time == datetime(2026, 9, 26, 18, 0, tzinfo=UTC)

    raw_dict = {"events": raw_list}
    events_from_dict = parse_calendar_json(raw_dict)
    assert len(events_from_dict) == 2
    assert events_from_dict[0].impact == NewsImpact.MEDIUM
    assert events_from_dict[1].impact == NewsImpact.HIGH


def test_parse_calendar_json_corrupt_data() -> None:
    with pytest.raises(ValueError, match="Invalid JSON content"):
        parse_calendar_json("invalid json {")

    with pytest.raises(ValueError, match="Expected JSON array or object with 'events'"):
        parse_calendar_json(json.dumps({"wrong_key": 123}))

    with pytest.raises(ValueError, match="missing required field"):
        parse_calendar_json(json.dumps([{"title": "CPI"}]))

    with pytest.raises(ValueError, match="event must be an object"):
        parse_calendar_json([123])

    with pytest.raises(ValueError, match="event must be an object"):
        parse_calendar_json([None])

    with pytest.raises(ValueError, match="actual must be numeric"):
        parse_calendar_json(
            [
                {
                    "title": "CPI",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": "2026-09-26T12:00:00Z",
                    "actual": [],
                }
            ]
        )

    with pytest.raises(ValueError, match="forecast must be numeric"):
        parse_calendar_json(
            [
                {
                    "title": "CPI",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": "2026-09-26T12:00:00Z",
                    "forecast": "invalid",
                }
            ]
        )


def test_news_filter_blackout_window_exact_boundaries() -> None:
    event_time = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    event = NewsEvent(
        title="US CPI m/m",
        currency="USD",
        impact=NewsImpact.HIGH,
        time=event_time,
    )

    news_filter = NewsFilter(
        events=[event],
        config=NewsFilterConfig(
            blackout_minutes_before=30.0,
            blackout_minutes_after=30.0,
            currencies=("USD",),
        ),
    )

    res_before_outside = news_filter.check(event_time - timedelta(minutes=31))
    assert res_before_outside.can_trade is True

    res_start = news_filter.check(event_time - timedelta(minutes=30))
    assert res_start.can_trade is False
    assert res_start.reason == TradeBlockReason.NEWS_BLACKOUT
    assert "US CPI m/m" in res_start.details

    res_exact = news_filter.check(event_time)
    assert res_exact.can_trade is False
    assert res_exact.reason == TradeBlockReason.NEWS_BLACKOUT

    res_end = news_filter.check(event_time + timedelta(minutes=30))
    assert res_end.can_trade is False
    assert res_end.reason == TradeBlockReason.NEWS_BLACKOUT

    res_after_outside = news_filter.check(event_time + timedelta(minutes=31))
    assert res_after_outside.can_trade is True


def test_news_filter_ignores_untracked_currency_and_low_impact() -> None:
    event_time = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)
    events = [
        NewsEvent(
            title="JPY Trade Balance",
            currency="JPY",
            impact=NewsImpact.HIGH,
            time=event_time,
        ),
        NewsEvent(
            title="USD Crude Oil Inventories",
            currency="USD",
            impact=NewsImpact.LOW,
            time=event_time,
        ),
        NewsEvent(
            title="USD Existing Home Sales",
            currency="USD",
            impact=NewsImpact.MEDIUM,
            time=event_time,
        ),
    ]

    news_filter = NewsFilter(
        events=events,
        config=NewsFilterConfig(
            currencies=("USD",),
            min_impact=NewsImpact.HIGH,
        ),
    )

    res = news_filter.check(event_time)
    assert res.can_trade is True
    assert res.reason is None


def test_news_filter_load_from_file(tmp_path: Path) -> None:
    calendar_file = tmp_path / "calendar.json"
    data = [
        {
            "title": "US Non-Farm Payrolls",
            "currency": "USD",
            "impact": "HIGH",
            "time": "2026-10-02T12:30:00Z",
        }
    ]
    calendar_file.write_text(json.dumps(data), encoding="utf-8")

    news_filter = NewsFilter()
    news_filter.load_events_from_file(calendar_file)

    check_time = datetime(2026, 10, 2, 12, 15, tzinfo=UTC)
    res = news_filter.check(check_time)
    assert res.can_trade is False
    assert res.reason == TradeBlockReason.NEWS_BLACKOUT
    assert "US Non-Farm Payrolls" in res.details


def test_load_calendar_missing_file_raises(tmp_path: Path) -> None:
    missing_file = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        load_calendar_from_file(missing_file)


def test_news_filter_upcoming_events() -> None:
    now = datetime(2026, 9, 26, 10, 0, tzinfo=UTC)
    events = [
        NewsEvent(
            title="Old News",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=now - timedelta(hours=2),
        ),
        NewsEvent(
            title="Upcoming Soon",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=now + timedelta(hours=1),
        ),
        NewsEvent(
            title="Upcoming Tomorrow",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=now + timedelta(hours=10),
        ),
        NewsEvent(
            title="Far Away",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=now + timedelta(days=5),
        ),
    ]

    news_filter = NewsFilter(events=events)
    upcoming = news_filter.get_upcoming_events(now, horizon_hours=12.0)
    assert len(upcoming) == 2
    assert upcoming[0].title == "Upcoming Soon"
    assert upcoming[1].title == "Upcoming Tomorrow"


def test_composite_trade_filter_full_pipeline() -> None:
    ny_tz = ZoneInfo("America/New_York")
    session_filter = SessionFilter()

    news_time = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)
    news_filter = NewsFilter(
        events=[
            NewsEvent(
                title="US FOMC Rate Decision",
                currency="USD",
                impact=NewsImpact.HIGH,
                time=news_time,
            )
        ]
    )
    spread_filter = SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=0.05))

    composite = CompositeTradeFilter(
        session_filter=session_filter,
        news_filter=news_filter,
        spread_filter=spread_filter,
    )

    weekend_time = datetime(2026, 9, 26, 12, 0, tzinfo=ny_tz)
    res_weekend = composite.check(weekend_time, spread=0.2, atr=5.0)
    assert res_weekend.can_trade is False
    assert res_weekend.reason == TradeBlockReason.WEEKEND

    rollover_time = datetime(2026, 9, 23, 17, 15, tzinfo=ny_tz)
    res_rollover = composite.check(rollover_time, spread=0.2, atr=5.0)
    assert res_rollover.can_trade is False
    assert res_rollover.reason == TradeBlockReason.ROLLOVER

    res_news = composite.check(news_time, spread=0.2, atr=5.0)
    assert res_news.can_trade is False
    assert res_news.reason == TradeBlockReason.NEWS_BLACKOUT

    normal_trade_time = datetime(2026, 9, 23, 10, 45, tzinfo=ny_tz)
    res_spread = composite.check(normal_trade_time, spread=0.40, atr=5.0)
    assert res_spread.can_trade is False
    assert res_spread.reason == TradeBlockReason.SPREAD_ABOVE_THRESHOLD

    res_clear = composite.check(normal_trade_time, spread=0.20, atr=5.0)
    assert res_clear.can_trade is True
    assert res_clear.reason is None


def test_models_additional_invariants() -> None:
    valid_time = datetime(2026, 9, 26, 12, 30, tzinfo=UTC)

    with pytest.raises(ValueError, match="actual must be finite"):
        NewsEvent(
            title="CPI",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=valid_time,
            actual=float("nan"),
        )
    with pytest.raises(ValueError, match="forecast must be finite"):
        NewsEvent(
            title="CPI",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=valid_time,
            forecast=float("inf"),
        )
    with pytest.raises(ValueError, match="previous must be finite"):
        NewsEvent(
            title="CPI",
            currency="USD",
            impact=NewsImpact.HIGH,
            time=valid_time,
            previous=float("-inf"),
        )

    with pytest.raises(ValueError, match="max_spread_points must be finite"):
        SpreadFilterConfig(max_spread_points=float("nan"))

    with pytest.raises(ValueError, match="blackout_minutes_before must be finite"):
        NewsFilterConfig(blackout_minutes_before=float("inf"))
    with pytest.raises(ValueError, match="blackout_minutes_before cannot be negative"):
        NewsFilterConfig(blackout_minutes_before=-5.0)
    with pytest.raises(ValueError, match="blackout_minutes_after must be finite"):
        NewsFilterConfig(blackout_minutes_after=float("nan"))
    with pytest.raises(ValueError, match="blackout_minutes_after cannot be negative"):
        NewsFilterConfig(blackout_minutes_after=-1.0)

    with pytest.raises(ValueError, match="currencies cannot be empty"):
        NewsFilterConfig(currencies=())
    with pytest.raises(ValueError, match="currency must be a non-empty string"):
        NewsFilterConfig(currencies=("",))

    norm_cfg = NewsFilterConfig(currencies=("usd", "eur"))
    assert norm_cfg.currencies == ("USD", "EUR")

    with pytest.raises(ValueError, match="Incoherent FilterCheckResult"):
        FilterCheckResult(can_trade=True, reason=TradeBlockReason.WEEKEND)
    with pytest.raises(ValueError, match="Incoherent FilterCheckResult"):
        FilterCheckResult(can_trade=False, reason=None)


def test_spread_filter_timestamp_handling() -> None:
    filter_ = SpreadFilter()
    now_utc = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)

    res = filter_.check(spread=0.20, atr=5.0, timestamp=now_utc)
    assert res.can_trade is True
    assert res.timestamp == now_utc

    naive_time = datetime(2026, 9, 26, 12, 0)
    with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
        filter_.check(spread=0.20, atr=5.0, timestamp=naive_time)


def test_news_parser_additional_branches() -> None:
    valid_time = datetime(2026, 9, 26, 14, 0, tzinfo=UTC)

    events_dt = parse_calendar_json(
        [
            {
                "title": "Rate Decision",
                "currency": "USD",
                "impact": "HIGH",
                "time": valid_time,
            }
        ]
    )
    assert len(events_dt) == 1
    assert events_dt[0].time == valid_time

    with pytest.raises(ValueError, match="unsupported time format"):
        parse_calendar_json(
            [
                {
                    "title": "Rate Decision",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": 123456,
                }
            ]
        )

    with pytest.raises(ValueError, match="time must be timezone-aware"):
        parse_calendar_json(
            [
                {
                    "title": "Rate Decision",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": "2026-09-26T12:00:00",
                }
            ]
        )

    with pytest.raises(ValueError, match="Expected JSON array or object with 'events'"):
        parse_calendar_json(cast(Any, 123))


def test_news_filter_methods_and_timezone_checks() -> None:
    now_utc = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
    event = NewsEvent(
        title="Fed Speech",
        currency="USD",
        impact=NewsImpact.HIGH,
        time=now_utc,
    )
    news_filter = NewsFilter([event])

    events_list = news_filter.get_events()
    assert len(events_list) == 1
    assert events_list[0].title == "Fed Speech"

    naive_time = datetime(2026, 9, 26, 12, 0)
    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        news_filter.is_blackout(naive_time)

    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        news_filter.get_upcoming_events(naive_time)


def test_composite_trade_filter_optional_filters_and_naive_check() -> None:
    ny_tz = ZoneInfo("America/New_York")
    normal_trade_time = datetime(2026, 9, 23, 10, 30, tzinfo=ny_tz)

    naive_time = datetime(2026, 9, 23, 10, 30)
    composite = CompositeTradeFilter()
    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        composite.check(naive_time)

    res_session_only = composite.check(normal_trade_time)
    assert res_session_only.can_trade is True


def test_repro_bug_t10_01_composite_blocks_when_spread_or_atr_missing() -> None:
    ny_tz = ZoneInfo("America/New_York")
    normal_trade_time = datetime(2026, 9, 23, 10, 30, tzinfo=ny_tz)
    spread_filter = SpreadFilter()
    composite = CompositeTradeFilter(spread_filter=spread_filter)

    res_none_spread = composite.check(normal_trade_time, spread=None, atr=5.0)
    assert res_none_spread.can_trade is False
    assert res_none_spread.reason == TradeBlockReason.MARKET_DATA_UNAVAILABLE

    res_none_atr = composite.check(normal_trade_time, spread=0.20, atr=None)
    assert res_none_atr.can_trade is False
    assert res_none_atr.reason == TradeBlockReason.MARKET_DATA_UNAVAILABLE

    res_both_none = composite.check(normal_trade_time, spread=None, atr=None)
    assert res_both_none.can_trade is False
    assert res_both_none.reason == TradeBlockReason.MARKET_DATA_UNAVAILABLE


def test_repro_bug_t10_02_spread_points_unit_coherence() -> None:
    config = SpreadFilterConfig(
        max_spread_atr_ratio=0.10,
        max_spread_points=7.0,
        point_size=0.01,
    )
    filter_ = SpreadFilter(config)

    res_exact_bound = filter_.check(spread=0.07, atr=10.0)
    assert res_exact_bound.can_trade is True
    assert res_exact_bound.reason is None

    res_above = filter_.check(spread=0.08, atr=10.0)
    assert res_above.can_trade is False
    assert res_above.reason == TradeBlockReason.SPREAD_ABOVE_THRESHOLD
    assert "exceeds maximum points 7.0" in res_above.details


def test_news_filter_get_minutes_to_next_news() -> None:
    t0 = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    ev_high_usd = NewsEvent(
        title="CPI",
        currency="USD",
        impact=NewsImpact.HIGH,
        time=t0 + timedelta(minutes=25),
    )
    ev_low_usd = NewsEvent(
        title="Low News",
        currency="USD",
        impact=NewsImpact.LOW,
        time=t0 + timedelta(minutes=10),
    )
    ev_high_eur = NewsEvent(
        title="ECB",
        currency="EUR",
        impact=NewsImpact.HIGH,
        time=t0 + timedelta(minutes=15),
    )

    filter_ = NewsFilter(events=[ev_high_usd, ev_low_usd, ev_high_eur])

    mins = filter_.get_minutes_to_next_news(t0)
    assert mins == pytest.approx(25.0)

    empty_filter = NewsFilter(events=[])
    assert empty_filter.get_minutes_to_next_news(t0) is None

    with pytest.raises(ValueError, match="timezone-aware"):
        filter_.get_minutes_to_next_news(datetime(2026, 9, 23, 12, 0))
