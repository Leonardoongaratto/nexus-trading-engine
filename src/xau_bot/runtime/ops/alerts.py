import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import SecretStr

logger = logging.getLogger("xau_bot.runtime.ops.alerts")

ALERT_TITLE_DEGRADED = "Watchdog Health Degraded"
ALERT_TITLE_RECOVERED = "Watchdog Health Recovered"
ALERT_TITLE_HALTED = "Trading Halted"
ALERT_TITLE_RESUMED = "Trading Resumed"


def _escape_telegram_markdown(text: str) -> str:
    escaped = text
    for char in ("_", "*", "`", "["):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped


class AlertChannel(Protocol):
    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool: ...


class ConsoleAlertChannel:
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream

    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        level_map = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        lvl = level_map.get(level.upper(), logging.INFO)
        extra_data = dict(metadata or {})
        if self.stream is not None and hasattr(self.stream, "write"):
            self.stream.write(f"[{level.upper()}] {title}: {message}\n")
            if hasattr(self.stream, "flush"):
                self.stream.flush()

        logger.log(
            lvl,
            "[%s] %s: %s",
            level.upper(),
            title,
            message,
            extra={
                "event": "console_alert",
                "alert_title": title,
                "alert_level": level,
                "alert_metadata": extra_data,
            },
        )
        return True


class TelegramAlertChannel:
    def __init__(
        self,
        token: SecretStr | str,
        chat_id: SecretStr | str,
        timeout_s: float = 5.0,
    ) -> None:
        self._token: str = token.get_secret_value() if isinstance(token, SecretStr) else str(token)
        self._chat_id: str = (
            chat_id.get_secret_value() if isinstance(chat_id, SecretStr) else str(chat_id)
        )
        self._timeout_s: float = timeout_s

    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        if not self._token or not self._chat_id:
            return False

        meta_str = ""
        if metadata:
            try:
                meta_str = "\n`" + json.dumps(dict(metadata), default=str, allow_nan=False) + "`"
            except Exception:
                meta_str = f"\n`{dict(metadata)!r}`"

        escaped_title = _escape_telegram_markdown(title)
        escaped_message = _escape_telegram_markdown(message)
        text = f"*{escaped_title}* \\[{level.upper()}]\n{escaped_message}{meta_str}"
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        def _do_post() -> bool:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    return bool(resp.status == 200)
            except Exception as exc:
                logger.warning(
                    "Failed to send Telegram alert: %s",
                    exc,
                    extra={"event": "telegram_alert_failed", "title": title},
                )
                return False

        return await asyncio.to_thread(_do_post)


class AlertManager:
    def __init__(
        self,
        channels: Sequence[AlertChannel] = (),
        deduplicate_s: float = 300.0,
    ) -> None:
        self.channels: tuple[AlertChannel, ...] = tuple(channels)
        self.deduplicate_s: float = deduplicate_s
        self._last_sent: dict[tuple[int, str, str, str], float] = {}
        self._active_group_state: dict[tuple[int, str], str] = {}

    def _resolve_dedup_group(self, title: str, dedup_group: str | None) -> str:
        if dedup_group is not None:
            return dedup_group
        if (
            title in (ALERT_TITLE_DEGRADED, ALERT_TITLE_RECOVERED)
            or "Degraded" in title
            or "Recovered" in title
        ):
            return "infra"
        if (
            title in (ALERT_TITLE_HALTED, ALERT_TITLE_RESUMED)
            or "Halted" in title
            or "Resumed" in title
            or "Kill Switch" in title
            or "Retomado" in title
        ):
            return "trading"
        return title

    def _should_deduplicate_channel(
        self, channel_idx: int, level: str, title: str, group: str
    ) -> bool:
        active = self._active_group_state.get((channel_idx, group))
        if active is not None and active != title:
            return False

        key = (channel_idx, group, level.upper(), title)
        now = time.monotonic()
        last = self._last_sent.get(key)
        return bool(last is not None and (now - last) < self.deduplicate_s)

    def _record_sent_channel(self, channel_idx: int, level: str, title: str, group: str) -> None:
        key = (channel_idx, group, level.upper(), title)
        now = time.monotonic()
        self._last_sent[key] = now

        prev_active = self._active_group_state.get((channel_idx, group))
        self._active_group_state[(channel_idx, group)] = title
        if prev_active != title:
            self._last_sent = {
                k: v
                for k, v in self._last_sent.items()
                if not (k[0] == channel_idx and k[1] == group and k[3] != title)
            }

    def clear_cache(self) -> None:
        self._last_sent.clear()
        self._active_group_state.clear()

    async def notify(
        self,
        level: str,
        title: str,
        message: str,
        metadata: Mapping[str, Any] | None = None,
        dedup_group: str | None = None,
    ) -> int:
        group = self._resolve_dedup_group(title, dedup_group)

        channels_to_send: list[tuple[int, AlertChannel]] = []
        for idx, channel in enumerate(self.channels):
            if not self._should_deduplicate_channel(idx, level, title, group):
                channels_to_send.append((idx, channel))

        if not channels_to_send:
            return 0

        tasks = [
            channel.send_alert(level=level, title=title, message=message, metadata=metadata)
            for _, channel in channels_to_send
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        delivered = 0
        for (idx, _), res in zip(channels_to_send, results, strict=True):
            if res is True:
                delivered += 1
                self._record_sent_channel(idx, level, title, group)

        return delivered
