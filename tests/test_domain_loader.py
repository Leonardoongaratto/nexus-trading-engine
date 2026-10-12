import io
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from xau_bot.domain.loader import (
    _parse_datetime,
    load_candles_from_csv,
    load_candles_from_dicts,
    load_candles_from_file,
    load_candles_from_json,
    load_ticks_from_csv,
    load_ticks_from_dicts,
    load_ticks_from_file,
    load_ticks_from_json,
)
from xau_bot.domain.models import Timeframe


def test_load_candles_from_dicts_valid() -> None:
    records = [
        {
            "time": 1727179200,
            "open": 2650.0,
            "high": 2655.0,
            "low": 2648.0,
            "close": 2653.0,
            "volume": 120.0,
        },
        {
            "time": 1727179500,
            "open": 2653.0,
            "high": 2658.0,
            "low": 2651.0,
            "close": 2656.0,
            "volume": 140.0,
        },
    ]

    series = load_candles_from_dicts(records, timeframe=Timeframe.M5)
    assert len(series) == 2
    assert series[0].open == 2650.0
    assert series[1].close == 2656.0
    assert series[0].time.tzinfo == UTC


def test_load_candles_with_datetime_objects() -> None:
    records = [
        {
            "time": datetime(2026, 9, 24, 12, 0, tzinfo=UTC),
            "open": 2650.0,
            "high": 2655.0,
            "low": 2648.0,
            "close": 2653.0,
            "volume": 100.0,
        }
    ]
    series = load_candles_from_dicts(records, timeframe=Timeframe.M5)
    assert len(series) == 1
    assert series[0].open == 2650.0


def test_load_candles_with_server_tz() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    records = [
        {
            "time": datetime(2026, 9, 24, 14, 0),
            "open": 2650.0,
            "high": 2655.0,
            "low": 2648.0,
            "close": 2653.0,
        }
    ]
    series = load_candles_from_dicts(records, timeframe=Timeframe.M5, server_tz=broker_tz)
    assert len(series) == 1
    assert series[0].time == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_load_candles_missing_field_raises() -> None:
    records = [
        {
            "time": 1727179200,
            "open": 2650.0,
        }
    ]
    with pytest.raises(ValueError, match="Missing required field"):
        load_candles_from_dicts(records, timeframe=Timeframe.M5)


def test_load_ticks_from_dicts_valid() -> None:
    records = [
        {
            "time": 1727179200.123,
            "bid": 2650.10,
            "ask": 2650.35,
            "last": 2650.20,
            "volume": 5.0,
            "flags": 6,
        },
        {
            "time": 1727179200.456,
            "bid": 2650.15,
            "ask": 2650.40,
            "last": 2650.25,
            "volume": 10.0,
            "flags": 6,
        },
    ]
    ticks = load_ticks_from_dicts(records)
    assert len(ticks) == 2
    assert ticks[0].bid == 2650.10
    assert ticks[1].ask == 2650.40
    assert ticks[0].time.tzinfo == UTC


def test_load_ticks_with_server_tz() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    records = [
        {
            "time": datetime(2026, 9, 24, 14, 0),
            "bid": 2650.10,
            "ask": 2650.35,
        }
    ]
    ticks = load_ticks_from_dicts(records, server_tz=broker_tz)
    assert len(ticks) == 1
    assert ticks[0].time == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_load_ticks_invalid_raises() -> None:
    records = [
        {
            "time": 1727179200.0,
            "bid": 2650.50,
            "ask": 2650.10,
        }
    ]
    with pytest.raises(ValueError):
        load_ticks_from_dicts(records)


def test_load_ticks_missing_field_raises() -> None:
    records = [{"time": 1727179200.0, "bid": 2650.0}]
    with pytest.raises(ValueError, match="Missing required field 'ask'"):
        load_ticks_from_dicts(records)


def test_parse_datetime_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse datetime"):
        _parse_datetime("invalid-string-date")


def test_parse_datetime_naive() -> None:
    naive = datetime(2026, 9, 24, 12, 0)
    aware = _parse_datetime(naive)
    assert aware.tzinfo == UTC


@pytest.mark.parametrize("bool_val", [True, False])
def test_parse_datetime_rejects_bool(bool_val: bool) -> None:
    with pytest.raises(ValueError, match="Cannot parse datetime from boolean"):
        _parse_datetime(bool_val)


@pytest.mark.parametrize("invalid_val", [float("nan"), float("inf"), float("-inf")])
def test_parse_datetime_rejects_non_finite(invalid_val: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _parse_datetime(invalid_val)


def test_parse_datetime_overflow_raises() -> None:
    with pytest.raises(ValueError, match="Invalid timestamp out of range"):
        _parse_datetime(1e25)


def test_parse_datetime_epoch_with_server_tz() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    raw_epoch = 1790258400
    parsed = _parse_datetime(raw_epoch, server_tz=broker_tz)
    assert parsed == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_load_candles_epoch_with_server_tz() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    records = [
        {
            "time": 1790258400,
            "open": 2650.0,
            "high": 2655.0,
            "low": 2648.0,
            "close": 2652.0,
        }
    ]
    series = load_candles_from_dicts(records, server_tz=broker_tz)
    assert len(series) == 1
    assert series[0].time == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_load_ticks_epoch_with_server_tz() -> None:
    broker_tz = ZoneInfo("Etc/GMT-2")
    records = [
        {
            "time": 1790258400.0,
            "bid": 2650.10,
            "ask": 2650.35,
        }
    ]
    ticks = load_ticks_from_dicts(records, server_tz=broker_tz)
    assert len(ticks) == 1
    assert ticks[0].time == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_parse_datetime_string_epoch() -> None:
    parsed = _parse_datetime("1727179200")
    assert parsed == datetime(2024, 9, 24, 12, 0, tzinfo=UTC)


def test_parse_datetime_string_iso_utc() -> None:
    parsed = _parse_datetime("2026-09-24T12:00:00Z")
    assert parsed == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_parse_datetime_string_naive_uses_server_tz() -> None:
    tz = ZoneInfo("Etc/GMT-2")
    parsed = _parse_datetime("2026-09-24 14:00:00", server_tz=tz)
    assert parsed == datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_parse_datetime_string_empty_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse datetime from empty string"):
        _parse_datetime("   ")


def test_parse_datetime_string_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse datetime from string"):
        _parse_datetime("not-a-valid-date")


def test_load_candles_from_csv_stream() -> None:
    csv_data = """time,open,high,low,close,volume
2026-09-24T12:00:00Z,2650.0,2655.0,2648.0,2653.0,100.0
2026-09-24T12:05:00Z,2653.0,2658.0,2651.0,2656.0,120.0
"""
    series = load_candles_from_csv(io.StringIO(csv_data), timeframe=Timeframe.M5)
    assert len(series) == 2
    assert series[0].open == 2650.0
    assert series[1].close == 2656.0


def test_load_ticks_from_csv_stream() -> None:
    csv_data = """time,bid,ask,last,volume,flags
2026-09-24T12:00:00Z,2650.10,2650.30,2650.20,5.0,6
2026-09-24T12:00:01Z,2650.15,2650.35,2650.25,10.0,6
"""
    ticks = load_ticks_from_csv(io.StringIO(csv_data))
    assert len(ticks) == 2
    assert ticks[0].bid == 2650.10
    assert ticks[1].ask == 2650.35

    json_data = (
        "[\n"
        '  {"time": "2026-09-24T12:00:00Z", "open": 2650.0, "high": 2655.0, '
        '"low": 2648.0, "close": 2653.0, "volume": 50.0},\n'
        '  {"time": "2026-09-24T12:05:00Z", "open": 2653.0, "high": 2658.0, '
        '"low": 2651.0, "close": 2656.0, "volume": 60.0}\n'
        "]"
    )
    series = load_candles_from_json(io.StringIO(json_data), timeframe=Timeframe.M5)
    assert len(series) == 2


def test_load_ticks_from_json_stream() -> None:
    json_data = """[
        {"time": "2026-09-24T12:00:00Z", "bid": 2650.10, "ask": 2650.30},
        {"time": "2026-09-24T12:00:01Z", "bid": 2650.15, "ask": 2650.35}
    ]"""
    ticks = load_ticks_from_json(io.StringIO(json_data))
    assert len(ticks) == 2


def test_load_candles_and_ticks_from_file(tmp_path: Path) -> None:
    csv_file = tmp_path / "candles.csv"
    csv_file.write_text(
        "Time,Open,High,Low,Close,Volume\n2026-09-24T12:00:00Z,2650.0,2655.0,2648.0,2653.0,10.0\n"
    )
    series = load_candles_from_file(csv_file)
    assert len(series) == 1

    json_file = tmp_path / "ticks.json"
    json_file.write_text('[{"time": "2026-09-24T12:00:00Z", "bid": 2650.10, "ask": 2650.30}]')
    ticks = load_ticks_from_file(json_file)
    assert len(ticks) == 1


def test_load_file_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_candles_from_file("nonexistent_path_12345.csv")
    with pytest.raises(FileNotFoundError):
        load_ticks_from_file("nonexistent_path_12345.json")


def test_load_csv_missing_fields_raises() -> None:
    bad_csv = "time,open,high\n2026-09-24T12:00:00Z,2650.0,2655.0\n"
    with pytest.raises(ValueError, match="Missing required field"):
        load_candles_from_csv(io.StringIO(bad_csv))


def test_parse_datetime_string_overflow_raises() -> None:
    with pytest.raises(ValueError, match="Invalid timestamp out of range"):
        _parse_datetime("1e25")


def test_load_candles_jsonlines_and_empty() -> None:
    assert len(load_candles_from_json(io.StringIO(""))) == 0

    jsonl = (
        '{"time": "2026-09-24T12:00:00Z", "open": 2650.0, "high": 2655.0, '
        '"low": 2648.0, "close": 2653.0}\n'
        '{"time": "2026-09-24T12:05:00Z", "open": 2653.0, "high": 2658.0, '
        '"low": 2651.0, "close": 2656.0}\n'
    )
    series = load_candles_from_json(io.StringIO(jsonl))
    assert len(series) == 2


def test_load_json_invalid_structure_raises() -> None:
    with pytest.raises(ValueError, match="Expected JSON list of records"):
        load_ticks_from_json(io.StringIO('{"not": "a list"}'))


def test_load_candles_and_ticks_inverted_formats(tmp_path: Path) -> None:
    json_candles = tmp_path / "candles.json"
    json_candles.write_text(
        '[{"time": "2026-09-24T12:00:00Z", "open": 2650.0, "high": 2655.0, '
        '"low": 2648.0, "close": 2653.0}]'
    )
    assert len(load_candles_from_file(json_candles)) == 1

    csv_ticks = tmp_path / "ticks.csv"
    csv_ticks.write_text("time,bid,ask\n2026-09-24T12:00:00Z,2650.10,2650.30\n")
    assert len(load_ticks_from_file(csv_ticks)) == 1


def test_load_json_dict_wrappers() -> None:
    wrapped_ticks = '{"ticks": [{"time": "2026-09-24T12:00:00Z", "bid": 2650.10, "ask": 2650.30}]}'
    ticks = load_ticks_from_json(io.StringIO(wrapped_ticks))
    assert len(ticks) == 1

    wrapped_candles = (
        '{"candles": [{"time": "2026-09-24T12:00:00Z", "open": 2650.0, '
        '"high": 2655.0, "low": 2648.0, "close": 2653.0}]}'
    )
    candles = load_candles_from_json(io.StringIO(wrapped_candles))
    assert len(candles) == 1


def test_load_json_non_dict_record_in_lines_raises() -> None:
    with pytest.raises(ValueError, match="Expected JSON records to be dictionaries"):
        load_ticks_from_json(io.StringIO("12345\n"))


def test_normalize_dict_keys_with_none() -> None:
    from xau_bot.domain.loader import _normalize_dict_keys

    assert _normalize_dict_keys({None: "foo", "TIME": "bar"}) == {"time": "bar"}


def test_parse_datetime_milliseconds() -> None:
    from xau_bot.domain.loader import parse_datetime

    # 1727179200 is 2024-09-24 12:00:00 UTC
    dt_sec = parse_datetime(1727179200)
    dt_ms_int = parse_datetime(1727179200000)
    dt_ms_str = parse_datetime("1727179200000")

    assert dt_ms_int == dt_sec
    assert dt_ms_str == dt_sec
    assert dt_ms_int.tzinfo == UTC


def test_load_setups_from_json_file(tmp_path: Path) -> None:
    from xau_bot.domain.loader import load_setups_from_json
    from xau_bot.domain.models import Direction

    setups_file = tmp_path / "setups.json"
    setups_file.write_text(
        '[{"id": "setup-1", "direction": "BUY", "entry_price": 2650.0, '
        '"stop_loss": 2640.0, "take_profit": 2670.0, "time": "2026-09-24T12:00:00Z", '
        '"confirmed_time": "2026-09-24T12:05:00Z", "risk_reward_ratio": 2.0}]',
        encoding="utf-8",
    )

    setups = load_setups_from_json(setups_file)
    assert len(setups) == 1
    assert setups[0].id == "setup-1"
    assert setups[0].direction == Direction.BULLISH
    assert setups[0].entry_price == 2650.0

    # Non-existent file
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_setups_from_json(tmp_path / "missing.json")

    # Invalid structure (not list)
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{"id": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="deve conter uma lista"):
        load_setups_from_json(invalid_file)
