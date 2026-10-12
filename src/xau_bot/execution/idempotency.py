import asyncio
import hashlib
import re
import time

from xau_bot.domain.models import Direction, OrderIntent
from xau_bot.execution.models import ExecutionResult


def generate_order_comment(intent: OrderIntent, prefix: str = "XAU") -> str:
    clean_prefix = re.sub(r"[^A-Za-z0-9]", "", prefix)[:6] or "XAU"
    dir_char = "B" if intent.direction == Direction.BULLISH else "S"
    exp_str = intent.expiration.isoformat() if intent.expiration is not None else "none"
    payload = (
        f"{intent.symbol}:{intent.direction.value}:{intent.order_type}:"
        f"{intent.volume:.4f}:{intent.price:.4f}:{intent.stop_loss:.4f}:"
        f"{intent.take_profit:.4f}:{intent.magic}:{exp_str}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    comment = f"{clean_prefix}_{dir_char}_{digest}"
    return comment[:31]


class InFlightTracker:
    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._in_flight: set[str] = set()
        self._completed: dict[str, tuple[float, ExecutionResult]] = {}

    async def acquire(self, key: str) -> bool:
        async with self._lock:
            if key in self._in_flight:
                return False
            self._in_flight.add(key)
            return True

    async def release(self, key: str) -> None:
        async with self._lock:
            self._in_flight.discard(key)

    async def is_in_flight(self, key: str) -> bool:
        async with self._lock:
            return key in self._in_flight

    async def mark_completed(self, key: str, result: ExecutionResult) -> None:
        async with self._lock:
            self._completed[key] = (time.monotonic(), result)

    async def get_completed(self, key: str) -> ExecutionResult | None:
        async with self._lock:
            entry = self._completed.get(key)
            if entry is None:
                return None
            stored_time, result = entry
            if time.monotonic() - stored_time > self._ttl_seconds:
                del self._completed[key]
                return None
            return result
