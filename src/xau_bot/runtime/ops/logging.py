import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

STANDARD_LOG_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "id",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

DEFAULT_SENSITIVE_PATTERNS = (
    r"""(?i)["']?(?:password|passwd|pwd|token|secret|(?:api[_-]?)?key|login|user_login|auth)["']?\s*[:=]\s*["']?([^\s,"'}{]+)""",
    r"""(?i)["']?(?:balance|equity|saldo|credit)["']?\s*[:=]\s*["']?([^\s,"'}{]+)""",
    r"""(?i)\b(?:bearer\s+)([A-Za-z0-9_\-\.]+)""",
)

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)password|passwd|pwd|token|secret|api[_-]?key|key|login|balance|equity|saldo"
)


class SensitiveDataFilter(logging.Filter):
    def __init__(
        self,
        name: str = "",
        sensitive_patterns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(name)
        patterns = (
            sensitive_patterns if sensitive_patterns is not None else DEFAULT_SENSITIVE_PATTERNS
        )
        self._compiled_regexes = [re.compile(p, re.IGNORECASE) for p in patterns]

    def _redact_text(self, text: str) -> str:
        res = text
        for regex in self._compiled_regexes:

            def repl(m: re.Match[str]) -> str:
                full = m.group(0)
                if m.lastindex and m.lastindex >= 1:
                    start_rel = m.start(1) - m.start(0)
                    end_rel = m.end(1) - m.start(0)
                    return full[:start_rel] + "***REDACTED***" + full[end_rel:]
                return "***REDACTED***"

            res = regex.sub(repl, res)
        return res

    def _sanitize_value(self, val: Any) -> Any:
        if isinstance(val, str):
            return self._redact_text(val)
        if isinstance(val, dict):
            return {
                k: (
                    "***REDACTED***"
                    if isinstance(k, str) and _SENSITIVE_KEY_PATTERN.search(k)
                    else self._sanitize_value(v)
                )
                for k, v in val.items()
            }
        if isinstance(val, (list, tuple)):
            sanitized_items = [self._sanitize_value(item) for item in val]
            return tuple(sanitized_items) if isinstance(val, tuple) else sanitized_items
        return val

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            try:
                record.msg = record.getMessage()
                record.args = ()
            except Exception:
                record.msg = f"{record.msg} {record.args!r}"
                record.args = ()

        if isinstance(record.msg, str):
            record.msg = self._redact_text(record.msg)

        if record.exc_info and not record.exc_text:
            formatted_exc = logging.Formatter().formatException(record.exc_info)
            record.exc_text = self._redact_text(formatted_exc)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self._redact_text(record.exc_text)

        if record.stack_info:
            record.stack_info = self._redact_text(record.stack_info)

        for key, val in list(record.__dict__.items()):
            if key in STANDARD_LOG_RECORD_ATTRS:
                continue
            if _SENSITIVE_KEY_PATTERN.search(key):
                record.__dict__[key] = "***REDACTED***"
            else:
                record.__dict__[key] = self._sanitize_value(val)

        return True


class JsonLogFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Any = "%",
        validate: bool = True,
        sensitive_filter: SensitiveDataFilter | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, validate=validate)
        self._filter = sensitive_filter or SensitiveDataFilter()

    def format(self, record: logging.LogRecord) -> str:
        record_dict = record.__dict__
        created_dt = datetime.fromtimestamp(record.created, UTC)
        now_iso = created_dt.isoformat()

        payload: dict[str, Any] = {
            "timestamp": now_iso,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            payload["event"] = record.event

        for key, val in record_dict.items():
            if key not in STANDARD_LOG_RECORD_ATTRS and key not in payload:
                try:
                    json.dumps(val, allow_nan=False)
                    payload[key] = val
                except (TypeError, ValueError):
                    payload[key] = str(val)

        if record.exc_text:
            payload["exception"] = self._filter._redact_text(record.exc_text)
        elif record.exc_info:
            formatted_exc = self.formatException(record.exc_info)
            payload["exception"] = self._filter._redact_text(formatted_exc)

        return json.dumps(payload, allow_nan=False)
