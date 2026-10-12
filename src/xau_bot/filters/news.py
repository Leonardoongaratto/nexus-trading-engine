import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from xau_bot.filters.models import FilterCheckResult, NewsEvent, NewsFilterConfig, NewsImpact
from xau_bot.filters.sessions import TradeBlockReason


def _parse_float_field(val: Any, field_name: str) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must be numeric, got {val}") from e


def _parse_event_dict(item: Any) -> NewsEvent:
    if not isinstance(item, dict):
        raise ValueError(f"event must be an object, got {type(item).__name__}")

    for req in ("title", "currency", "impact", "time"):
        if req not in item:
            raise ValueError(f"missing required field '{req}' in event: {item}")

    raw_time = item["time"]
    if isinstance(raw_time, datetime):
        event_time = raw_time
    elif isinstance(raw_time, str):
        event_time = datetime.fromisoformat(raw_time)
    else:
        raise ValueError(f"unsupported time format: {raw_time}")

    if event_time.tzinfo is None:
        raise ValueError(f"time must be timezone-aware: {raw_time}")

    event_time_utc = event_time.astimezone(UTC)
    raw_impact = str(item["impact"]).upper()
    impact = NewsImpact(raw_impact)

    actual = _parse_float_field(item.get("actual"), "actual")
    forecast = _parse_float_field(item.get("forecast"), "forecast")
    previous = _parse_float_field(item.get("previous"), "previous")

    return NewsEvent(
        title=str(item["title"]),
        currency=str(item["currency"]).upper(),
        impact=impact,
        time=event_time_utc,
        actual=actual,
        forecast=forecast,
        previous=previous,
        source_id=str(item["source_id"]) if item.get("source_id") is not None else None,
    )


def parse_calendar_payload(
    raw: str | bytes | dict[str, Any] | list[Any],
) -> tuple[list[NewsEvent], datetime | None]:
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON content: {e}") from e
    else:
        data = raw

    generated_at: datetime | None = None
    if isinstance(data, dict):
        if "generated_at" in data and data["generated_at"] is not None:
            raw_gen = data["generated_at"]
            if isinstance(raw_gen, datetime):
                dt_gen = raw_gen
            elif isinstance(raw_gen, str):
                dt_gen = datetime.fromisoformat(raw_gen)
            else:
                raise ValueError(f"unsupported generated_at format: {raw_gen}")
            if dt_gen.tzinfo is None:
                raise ValueError(f"generated_at must be timezone-aware: {raw_gen}")
            generated_at = dt_gen.astimezone(UTC)

        if "events" not in data or not isinstance(data["events"], list):
            raise ValueError("Expected JSON array or object with 'events' list")
        event_list = data["events"]
    elif isinstance(data, list):
        event_list = data
    else:
        raise ValueError("Expected JSON array or object with 'events'")

    parsed = [_parse_event_dict(item) for item in event_list]
    parsed.sort(key=lambda ev: ev.time)
    return parsed, generated_at


def parse_calendar_json(raw: str | bytes | dict[str, Any] | list[Any]) -> list[NewsEvent]:
    events, _ = parse_calendar_payload(raw)
    return events


def load_calendar_from_file(file_path: Path | str) -> list[NewsEvent]:
    events, _ = load_calendar_payload_from_file(file_path)
    return events


def load_calendar_payload_from_file(
    file_path: Path | str,
) -> tuple[list[NewsEvent], datetime | None]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Economic calendar file not found: {path}")
    content = path.read_text(encoding="utf-8-sig")
    return parse_calendar_payload(content)


class NewsFilter:
    def __init__(
        self,
        events: Sequence[NewsEvent] | None = None,
        config: NewsFilterConfig | None = None,
        generated_at: datetime | None = None,
    ) -> None:
        self.config: NewsFilterConfig = config if config is not None else NewsFilterConfig()
        self._events: list[NewsEvent] = []
        self._generated_at: datetime | None = (
            generated_at.astimezone(UTC) if generated_at is not None else None
        )
        if events is not None:
            self.set_events(events, generated_at=generated_at)

    def set_events(
        self,
        events: Sequence[NewsEvent],
        generated_at: datetime | None = None,
    ) -> None:
        self._events = sorted(events, key=lambda ev: ev.time)
        if generated_at is not None:
            if generated_at.tzinfo is None:
                raise ValueError("generated_at must be timezone-aware")
            self._generated_at = generated_at.astimezone(UTC)

    def load_events_from_file(self, file_path: Path | str) -> None:
        events, gen_at = load_calendar_payload_from_file(file_path)
        self.set_events(events, generated_at=gen_at)

    def get_events(self) -> list[NewsEvent]:
        return list(self._events)

    def is_blackout(self, current_time: datetime) -> tuple[bool, NewsEvent | None]:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        dt_utc = current_time.astimezone(UTC)

        impact_order = {
            NewsImpact.LOW: 1,
            NewsImpact.MEDIUM: 2,
            NewsImpact.HIGH: 3,
            NewsImpact.HOLIDAY: 0,
        }
        min_rank = impact_order.get(self.config.min_impact, 3)

        for event in self._events:
            if event.currency not in self.config.currencies:
                continue

            event_rank = impact_order.get(event.impact, 0)
            if event_rank < min_rank:
                continue

            start_window = event.time - timedelta(minutes=self.config.blackout_minutes_before)
            end_window = event.time + timedelta(minutes=self.config.blackout_minutes_after)

            if start_window <= dt_utc <= end_window:
                return True, event

        return False, None

    def check(self, current_time: datetime) -> FilterCheckResult:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        dt_utc = current_time.astimezone(UTC)

        if self.config.max_age_hours is not None:
            if self._generated_at is None:
                return FilterCheckResult(
                    can_trade=False,
                    reason=TradeBlockReason.NEWS_CALENDAR_STALE,
                    details="News calendar missing generation timestamp (fail-closed)",
                    timestamp=current_time,
                )
            age_s = (dt_utc - self._generated_at).total_seconds()
            max_age_s = self.config.max_age_hours * 3600.0
            if age_s > max_age_s:
                return FilterCheckResult(
                    can_trade=False,
                    reason=TradeBlockReason.NEWS_CALENDAR_STALE,
                    details=(
                        f"News calendar is stale: age {age_s / 3600.0:.1f}h "
                        f"> {self.config.max_age_hours:.1f}h"
                    ),
                    timestamp=current_time,
                )

        in_blackout, event = self.is_blackout(current_time)
        if in_blackout and event is not None:
            return FilterCheckResult(
                can_trade=False,
                reason=TradeBlockReason.NEWS_BLACKOUT,
                details=(
                    f"News blackout: {event.title} ({event.currency}) at {event.time.isoformat()}"
                ),
                timestamp=current_time,
            )

        return FilterCheckResult(
            can_trade=True,
            reason=None,
            details=f"No {self.config.min_impact}+ news in blackout window",
            timestamp=current_time,
        )

    def get_upcoming_events(
        self,
        current_time: datetime,
        horizon_hours: float = 24.0,
    ) -> list[NewsEvent]:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        dt_utc = current_time.astimezone(UTC)
        end_time = dt_utc + timedelta(hours=horizon_hours)

        return [ev for ev in self._events if dt_utc <= ev.time <= end_time]

    def get_minutes_to_next_news(
        self,
        current_time: datetime,
    ) -> float | None:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        dt_utc = current_time.astimezone(UTC)
        impact_order = {
            NewsImpact.LOW: 1,
            NewsImpact.MEDIUM: 2,
            NewsImpact.HIGH: 3,
            NewsImpact.HOLIDAY: 0,
        }
        min_rank = impact_order.get(self.config.min_impact, 3)
        future_events = [
            ev
            for ev in self._events
            if ev.currency in self.config.currencies
            and impact_order.get(ev.impact, 0) >= min_rank
            and ev.time >= dt_utc
        ]
        if not future_events:
            return None
        closest = min(future_events, key=lambda ev: ev.time)
        return (closest.time - dt_utc).total_seconds() / 60.0
