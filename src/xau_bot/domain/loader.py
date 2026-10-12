import csv
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any, TextIO

from xau_bot.domain.models import (
    Candle,
    Direction,
    OrderBlock,
    OrderBlockType,
    Tick,
    Timeframe,
    TradeSetup,
)
from xau_bot.domain.series import CandleSeries


def _parse_datetime(raw: Any, server_tz: tzinfo | None = None) -> datetime:
    if isinstance(raw, bool):
        raise ValueError(f"Cannot parse datetime from boolean: {raw!r}")
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            tz = server_tz or UTC
            return raw.replace(tzinfo=tz).astimezone(UTC)
        return raw.astimezone(UTC)
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            raise ValueError(f"Timestamp must be finite, got {raw}")
        tz = server_tz or UTC
        val = float(raw)
        epoch_sec = val / 1000.0 if val > 1e11 else val
        try:
            return datetime.fromtimestamp(epoch_sec, tz=UTC).replace(tzinfo=tz).astimezone(UTC)
        except (OverflowError, OSError, ValueError) as err:
            raise ValueError(f"Invalid timestamp out of range: {raw!r}") from err
    if isinstance(raw, str):
        trimmed = raw.strip()
        if not trimmed:
            raise ValueError("Cannot parse datetime from empty string")

        is_num = False
        val = 0.0
        try:
            val = float(trimmed)
            is_num = math.isfinite(val)
        except ValueError:
            is_num = False

        if is_num:
            tz = server_tz or UTC
            epoch_sec = val / 1000.0 if val > 1e11 else val
            try:
                return datetime.fromtimestamp(epoch_sec, tz=UTC).replace(tzinfo=tz).astimezone(UTC)
            except (OverflowError, OSError, ValueError) as err:
                raise ValueError(f"Invalid timestamp out of range: {raw!r}") from err

        try:
            dt = datetime.fromisoformat(trimmed)
            if dt.tzinfo is None:
                tz = server_tz or UTC
                return dt.replace(tzinfo=tz).astimezone(UTC)
            return dt.astimezone(UTC)
        except ValueError as err:
            raise ValueError(f"Cannot parse datetime from string: {raw!r}") from err

    raise ValueError(f"Cannot parse datetime from: {raw!r}")


parse_datetime = _parse_datetime


def _normalize_dict_keys(rec: dict[Any, Any]) -> dict[str, Any]:
    norm: dict[str, Any] = {}
    for k, v in rec.items():
        if k is None or not isinstance(k, str):
            continue
        cleaned = k.strip().lower()
        if cleaned == "timestamp":
            cleaned = "time"
        norm[cleaned] = v
    return norm


def load_candles_from_dicts(
    records: Sequence[dict[str, Any]],
    timeframe: Timeframe = Timeframe.M5,
    has_forming_candle: bool = False,
    server_tz: tzinfo | None = None,
) -> CandleSeries:
    candles: list[Candle] = []
    required_fields = ("time", "open", "high", "low", "close")

    for idx, raw_rec in enumerate(records):
        rec = _normalize_dict_keys(raw_rec)
        for field in required_fields:
            if field not in rec:
                raise ValueError(f"Missing required field '{field}' at record index {idx}")

        dt = _parse_datetime(rec["time"], server_tz=server_tz)
        candle = Candle(
            time=dt,
            open=float(rec["open"]),
            high=float(rec["high"]),
            low=float(rec["low"]),
            close=float(rec["close"]),
            volume=float(rec.get("volume", rec.get("tick_volume", 0.0))),
            timeframe=timeframe,
        )
        candles.append(candle)

    return CandleSeries(
        candles=candles,
        timeframe=timeframe,
        has_forming_candle=has_forming_candle,
    )


def load_ticks_from_dicts(
    records: Sequence[dict[str, Any]],
    server_tz: tzinfo | None = None,
) -> list[Tick]:
    ticks: list[Tick] = []
    required_fields = ("time", "bid", "ask")

    for idx, raw_rec in enumerate(records):
        rec = _normalize_dict_keys(raw_rec)
        for field in required_fields:
            if field not in rec:
                raise ValueError(f"Missing required field '{field}' at record index {idx}")

        dt = _parse_datetime(rec["time"], server_tz=server_tz)
        tick = Tick(
            time=dt,
            bid=float(rec["bid"]),
            ask=float(rec["ask"]),
            last=float(rec.get("last", 0.0)),
            volume=float(rec.get("volume", 0.0)),
            flags=int(rec.get("flags", 0)),
        )
        ticks.append(tick)

    return ticks


def load_candles_from_csv(
    file_or_stream: str | Path | TextIO,
    timeframe: Timeframe = Timeframe.M5,
    has_forming_candle: bool = False,
    server_tz: tzinfo | None = None,
) -> CandleSeries:
    if isinstance(file_or_stream, (str, Path)):
        with open(file_or_stream, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        reader = csv.DictReader(file_or_stream)
        records = list(reader)

    return load_candles_from_dicts(
        records,
        timeframe=timeframe,
        has_forming_candle=has_forming_candle,
        server_tz=server_tz,
    )


def load_ticks_from_csv(
    file_or_stream: str | Path | TextIO,
    server_tz: tzinfo | None = None,
) -> list[Tick]:
    if isinstance(file_or_stream, (str, Path)):
        with open(file_or_stream, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    else:
        reader = csv.DictReader(file_or_stream)
        records = list(reader)

    return load_ticks_from_dicts(records, server_tz=server_tz)


def _load_json_records(file_or_stream: str | Path | TextIO) -> list[dict[str, Any]]:
    if isinstance(file_or_stream, (str, Path)):
        with open(file_or_stream, encoding="utf-8") as f:
            content = f.read().strip()
    else:
        content = file_or_stream.read().strip()

    if not content:
        return []

    if content.startswith("["):
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records_list: list[dict[str, Any]] = [r for r in parsed if isinstance(r, dict)]
            return records_list
        raise ValueError(f"Expected JSON list of records, got {type(parsed).__name__}")

    if content.startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for key in ("ticks", "candles", "data", "records"):
                    if key in parsed and isinstance(parsed[key], list):
                        wrapped_list: list[dict[str, Any]] = [
                            r for r in parsed[key] if isinstance(r, dict)
                        ]
                        return wrapped_list
            raise ValueError("Expected JSON list of records")
        except json.JSONDecodeError:
            pass

    records: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
            else:
                raise ValueError("Expected JSON records to be dictionaries")
    return records


def load_candles_from_json(
    file_or_stream: str | Path | TextIO,
    timeframe: Timeframe = Timeframe.M5,
    has_forming_candle: bool = False,
    server_tz: tzinfo | None = None,
) -> CandleSeries:
    records = _load_json_records(file_or_stream)
    return load_candles_from_dicts(
        records,
        timeframe=timeframe,
        has_forming_candle=has_forming_candle,
        server_tz=server_tz,
    )


def load_ticks_from_json(
    file_or_stream: str | Path | TextIO,
    server_tz: tzinfo | None = None,
) -> list[Tick]:
    records = _load_json_records(file_or_stream)
    return load_ticks_from_dicts(records, server_tz=server_tz)


def load_candles_from_file(
    file_path: str | Path,
    timeframe: Timeframe = Timeframe.M5,
    has_forming_candle: bool = False,
    server_tz: tzinfo | None = None,
) -> CandleSeries:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in (".json", ".jsonl"):
        return load_candles_from_json(
            path,
            timeframe=timeframe,
            has_forming_candle=has_forming_candle,
            server_tz=server_tz,
        )
    return load_candles_from_csv(
        path,
        timeframe=timeframe,
        has_forming_candle=has_forming_candle,
        server_tz=server_tz,
    )


def load_ticks_from_file(
    file_path: str | Path,
    server_tz: tzinfo | None = None,
) -> list[Tick]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() in (".json", ".jsonl"):
        return load_ticks_from_json(path, server_tz=server_tz)
    return load_ticks_from_csv(path, server_tz=server_tz)


parse_datetime = _parse_datetime


def load_setups_from_json(
    file_path: str | Path,
    server_tz: tzinfo | None = None,
) -> list[TradeSetup]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de setups não encontrado: {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("JSON de setups deve conter uma lista de objetos de setup.")
    setups: list[TradeSetup] = []
    for r in data:
        raw_dir = str(r["direction"]).strip().upper()
        if raw_dir in ("BUY", "LONG", "BULLISH"):
            direction = Direction.BULLISH
        elif raw_dir in ("SELL", "SHORT", "BEARISH"):
            direction = Direction.BEARISH
        else:
            direction = Direction(raw_dir)
        t = _parse_datetime(r["time"], server_tz=server_tz)
        ct = _parse_datetime(r["confirmed_time"], server_tz=server_tz)
        vu = (
            _parse_datetime(r["valid_until"], server_tz=server_tz) if r.get("valid_until") else None
        )
        tf = Timeframe(r["timeframe"]) if "timeframe" in r else Timeframe.M5
        ep = float(r["entry_price"])
        sl = float(r["stop_loss"])
        tp = float(r["take_profit"])

        disp_val = r.get("displacement_atr_ratio")
        if disp_val is None and isinstance(r.get("poi"), dict):
            disp_val = r.get("poi", {}).get("displacement_atr_ratio")

        poi = None
        if disp_val is not None:
            dummy_candle = Candle(
                time=t,
                open=ep,
                high=max(ep, sl, tp),
                low=min(ep, sl, tp),
                close=ep,
                volume=100.0,
                timeframe=tf,
            )
            poi = OrderBlock(
                time=t,
                confirmed_time=ct,
                type=(
                    OrderBlockType.BULLISH
                    if direction == Direction.BULLISH
                    else OrderBlockType.BEARISH
                ),
                high=dummy_candle.high,
                low=dummy_candle.low,
                open=dummy_candle.open,
                close=dummy_candle.close,
                candle=dummy_candle,
                displacement_atr_ratio=float(disp_val),
            )

        s = TradeSetup(
            id=str(r["id"]),
            direction=direction,
            entry_price=ep,
            stop_loss=sl,
            take_profit=tp,
            time=t,
            confirmed_time=ct,
            risk_reward_ratio=float(r.get("risk_reward_ratio", 2.0)),
            valid_until=vu,
            timeframe=tf,
            poi=poi,
        )
        setups.append(s)
    return setups


def load_setups_from_file(
    file_path: str | Path,
    server_tz: tzinfo | None = None,
) -> list[TradeSetup]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de setups não encontrado: {path}")
    return load_setups_from_json(path, server_tz=server_tz)
