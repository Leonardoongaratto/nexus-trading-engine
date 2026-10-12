from dataclasses import dataclass
from enum import StrEnum

from xau_bot.broker.models import BrokerOrder, BrokerPosition


class ExecutionStatus(StrEnum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    RECONCILED = "RECONCILED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class RetcodeAction(StrEnum):
    SUCCESS = "SUCCESS"
    FATAL_ABORT = "FATAL_ABORT"
    RETRY_WITH_RECONCILIATION = "RETRY_WITH_RECONCILIATION"


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    found: bool
    is_position: bool = False
    is_order: bool = False
    ticket: int | None = None
    position: BrokerPosition | None = None
    order: BrokerOrder | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool
    status: ExecutionStatus
    ticket: int | None = None
    price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    error_message: str = ""
    reconciled: bool = False
    retries_attempted: int = 0
    comment: str = ""
    magic: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    magic: int = 123456
    max_retries: int = 3
    retry_delay_seconds: float = 0.1
    max_slippage_points: float = 5.0
    comment_prefix: str = "XAU"
