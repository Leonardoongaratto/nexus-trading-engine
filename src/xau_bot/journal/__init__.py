"""Trade Journal module for logging, evaluating and auditing trade setups."""

from __future__ import annotations

from xau_bot.journal.evaluator import OutcomeEvaluator, create_snapshot_from_proposal
from xau_bot.journal.models import (
    JournalEntry,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)
from xau_bot.journal.reporter import JournalReporter
from xau_bot.journal.repository import JournalRepository

__all__ = [
    "JournalEntry",
    "JournalReporter",
    "JournalRepository",
    "LossCategory",
    "LossRootCause",
    "OutcomeData",
    "OutcomeEvaluator",
    "SnapshotData",
    "TradeOutcome",
    "create_snapshot_from_proposal",
]
