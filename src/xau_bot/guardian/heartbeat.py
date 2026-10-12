import contextlib
import json
import logging
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from xau_bot.guardian.models import HeartbeatPayload

logger = logging.getLogger("xau_bot.guardian.heartbeat")


class HeartbeatWriter:
    def __init__(
        self,
        common_dir: Path | str,
        filename: str = "xau_heartbeat.json",
        pid: int | None = None,
    ) -> None:
        self.common_dir: Path = Path(common_dir)
        self.filename: str = filename
        self._pid: int = pid if pid is not None else os.getpid()
        self._sequence: int = 0
        self._lock: threading.Lock = threading.Lock()

    @property
    def target_file(self) -> Path:
        return self.common_dir / self.filename

    def write(
        self,
        status: str = "ok",
        trading_halted: bool = False,
        now: datetime | None = None,
    ) -> HeartbeatPayload:
        if now is not None and now.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        dt_utc = (now or datetime.now(UTC)).astimezone(UTC)

        with self._lock:
            self._sequence += 1
            payload = HeartbeatPayload(
                timestamp=dt_utc,
                status=status,
                pid=self._pid,
                sequence=self._sequence,
                trading_halted=trading_halted,
            )

        self.common_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = self.common_dir / f"{self.filename}.tmp.{uuid.uuid4().hex}"
        content = payload.to_json()

        try:
            tmp_file.write_text(content, encoding="utf-8")

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_file, self.target_file)
                    break
                except (PermissionError, OSError) as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            "Failed to write atomic heartbeat after retries: %s",
                            e,
                            extra={"event": "HEARTBEAT_WRITE_FAILED"},
                        )
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            if tmp_file.exists():
                with contextlib.suppress(OSError):
                    tmp_file.unlink(missing_ok=True)

        return payload


class HeartbeatReader:
    def __init__(
        self,
        common_dir: Path | str,
        filename: str = "xau_heartbeat.json",
        timeout_s: float = 30.0,
    ) -> None:
        self.common_dir: Path = Path(common_dir)
        self.filename: str = filename
        self.timeout_s: float = timeout_s

    @property
    def target_file(self) -> Path:
        return self.common_dir / self.filename

    def read(self) -> HeartbeatPayload | None:
        if not self.target_file.is_file():
            return None

        try:
            content = self.target_file.read_text(encoding="utf-8-sig")
            return HeartbeatPayload.from_json(content)

        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, OSError) as e:
            logger.warning(
                "Failed to read or parse heartbeat file: %s",
                e,
                extra={"event": "HEARTBEAT_READ_ERROR"},
            )
            return None

    def is_alive(self, now: datetime | None = None) -> tuple[bool, str]:
        if now is not None and now.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        dt_utc = (now or datetime.now(UTC)).astimezone(UTC)

        if not self.target_file.is_file():
            return False, "HEARTBEAT_MISSING"

        payload = self.read()
        if payload is None:
            return False, "HEARTBEAT_CORRUPT"

        age_s = (dt_utc - payload.timestamp).total_seconds()
        if age_s > self.timeout_s:
            return False, f"HEARTBEAT_TIMEOUT: age={age_s:.1f}s > {self.timeout_s}s"

        return True, "OK"
