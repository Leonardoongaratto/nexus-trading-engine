"""SQLite repository for persisting trade journal entries."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.domain.models import Direction
from xau_bot.journal.models import (
    JournalEntry,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)


class JournalRepository:
    """Thread-safe SQLite repository for trade journal entries."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            path_obj = Path(self.db_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT UNIQUE NOT NULL,
                    ticket INTEGER,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    loss_cause TEXT,
                    loss_category TEXT,
                    snapshot_json TEXT NOT NULL,
                    outcome_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_journal_prop ON journal_entries(proposal_id);
                CREATE INDEX IF NOT EXISTS idx_journal_ticket ON journal_entries(ticket);
                CREATE INDEX IF NOT EXISTS idx_journal_outcome ON journal_entries(outcome);

                CREATE TABLE IF NOT EXISTS shadow_evaluations (
                    id TEXT PRIMARY KEY,
                    setup_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    active_allowed INTEGER NOT NULL,
                    shadow_allowed INTEGER NOT NULL,
                    divergence TEXT NOT NULL,
                    active_reason TEXT,
                    shadow_reason TEXT,
                    is_divergent INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_shadow_setup ON shadow_evaluations(setup_id);
                """
            )

    def save_entry(self, entry: JournalEntry) -> None:
        with self._lock, self._conn:
            snapshot_json = json.dumps(entry.snapshot.to_dict())
            outcome_json = json.dumps(entry.outcome_data.to_dict()) if entry.outcome_data else None
            now_iso = entry.updated_at.astimezone(UTC).isoformat()
            created_iso = entry.created_at.astimezone(UTC).isoformat()

            self._conn.execute(
                """
                INSERT INTO journal_entries (
                    id, proposal_id, ticket, symbol, direction, status,
                    outcome, loss_cause, loss_category, snapshot_json,
                    outcome_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    ticket=excluded.ticket,
                    status=excluded.status,
                    outcome=excluded.outcome,
                    loss_cause=excluded.loss_cause,
                    loss_category=excluded.loss_category,
                    snapshot_json=excluded.snapshot_json,
                    outcome_json=excluded.outcome_json,
                    updated_at=excluded.updated_at
                """,
                (
                    entry.id,
                    entry.proposal_id,
                    entry.ticket,
                    entry.symbol,
                    entry.direction.value,
                    entry.status.value,
                    entry.outcome.value,
                    entry.loss_cause.value if entry.loss_cause else None,
                    entry.loss_category.value if entry.loss_category else None,
                    snapshot_json,
                    outcome_json,
                    created_iso,
                    now_iso,
                ),
            )

    def update_outcome(
        self,
        proposal_id: str,
        outcome_data: OutcomeData,
        outcome: TradeOutcome,
        loss_cause: LossRootCause | None = None,
        loss_category: LossCategory | None = None,
        status: ProposalStatus | None = None,
    ) -> bool:
        with self._lock:
            outcome_json = json.dumps(outcome_data.to_dict())
            now_iso = datetime.now(UTC).isoformat()
            set_clauses = [
                "outcome_json = ?",
                "outcome = ?",
                "loss_cause = ?",
                "loss_category = ?",
                "updated_at = ?",
            ]
            params: list[Any] = [
                outcome_json,
                outcome.value,
                loss_cause.value if loss_cause else None,
                loss_category.value if loss_category else None,
                now_iso,
            ]
            if status is not None:
                set_clauses.append("status = ?")
                params.append(status.value)

            params.append(proposal_id)
            with self._conn:
                cursor = self._conn.execute(
                    f"UPDATE journal_entries SET {', '.join(set_clauses)} WHERE proposal_id = ?",
                    params,
                )
                return cursor.rowcount > 0

    def get_entry(self, entry_id_or_proposal_id: str) -> JournalEntry | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM journal_entries WHERE id = ? OR proposal_id = ?",
                (entry_id_or_proposal_id, entry_id_or_proposal_id),
            )
            row = cursor.fetchone()
            return self._row_to_entry(row) if row else None

    def get_by_proposal_id(self, proposal_id: str) -> JournalEntry | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM journal_entries WHERE proposal_id = ?",
                (proposal_id,),
            )
            row = cursor.fetchone()
            return self._row_to_entry(row) if row else None

    def get_by_ticket(self, ticket: int) -> JournalEntry | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM journal_entries WHERE ticket = ?",
                (ticket,),
            )
            row = cursor.fetchone()
            return self._row_to_entry(row) if row else None

    def list_entries(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        outcome: TradeOutcome | None = None,
        limit: int | None = 100,
    ) -> list[JournalEntry]:
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if start_time is not None:
                clauses.append("created_at >= ?")
                params.append(start_time.astimezone(UTC).isoformat())
            if end_time is not None:
                clauses.append("created_at <= ?")
                params.append(end_time.astimezone(UTC).isoformat())
            if outcome is not None:
                clauses.append("outcome = ?")
                params.append(outcome.value)

            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            if limit is not None:
                query = (
                    f"SELECT * FROM journal_entries {where_str} ORDER BY created_at DESC LIMIT ?"
                )
                params.append(limit)
            else:
                query = f"SELECT * FROM journal_entries {where_str} ORDER BY created_at DESC"
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_entry(r) for r in rows]

    def count_entries(self, outcome: TradeOutcome | None = None) -> int:
        with self._lock:
            if outcome is not None:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM journal_entries WHERE outcome = ?",
                    (outcome.value,),
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM journal_entries")
            val = cursor.fetchone()[0]
            return int(val)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        snapshot_dict = json.loads(row["snapshot_json"])
        snapshot = SnapshotData.from_dict(snapshot_dict)

        outcome_data: OutcomeData | None = None
        if row["outcome_json"]:
            outcome_dict = json.loads(row["outcome_json"])
            outcome_data = OutcomeData.from_dict(outcome_dict)

        created_at = datetime.fromisoformat(row["created_at"])
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        updated_at = datetime.fromisoformat(row["updated_at"])
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)

        loss_cause = LossRootCause(row["loss_cause"]) if row["loss_cause"] else None
        loss_category = LossCategory(row["loss_category"]) if row["loss_category"] else None

        return JournalEntry(
            id=row["id"],
            proposal_id=row["proposal_id"],
            ticket=row["ticket"],
            symbol=row["symbol"],
            direction=Direction(row["direction"]),
            status=ProposalStatus(row["status"]),
            outcome=TradeOutcome(row["outcome"]),
            loss_cause=loss_cause,
            loss_category=loss_category,
            snapshot=snapshot,
            outcome_data=outcome_data,
            created_at=created_at,
            updated_at=updated_at,
        )

    def save_shadow_evaluation(self, evaluation: Any) -> None:
        eval_dict = evaluation.to_dict() if hasattr(evaluation, "to_dict") else dict(evaluation)
        eval_id = f"shd_{eval_dict['setup_id']}_{eval_dict['timestamp']}"
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO shadow_evaluations (
                    id, setup_id, direction, timestamp, active_allowed,
                    shadow_allowed, divergence, active_reason, shadow_reason,
                    is_divergent, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    str(eval_dict["setup_id"]),
                    str(eval_dict["direction"]),
                    str(eval_dict["timestamp"]),
                    1 if eval_dict["active_allowed"] else 0,
                    1 if eval_dict["shadow_allowed"] else 0,
                    str(eval_dict["divergence"]),
                    str(eval_dict.get("active_reason", "")),
                    str(eval_dict.get("shadow_reason", "")),
                    1 if eval_dict.get("is_divergent", False) else 0,
                    json.dumps(eval_dict),
                ),
            )

    def list_shadow_evaluations(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock, self._conn:
            query = "SELECT payload_json FROM shadow_evaluations ORDER BY timestamp DESC"
            params: list[Any] = []
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            cursor = self._conn.execute(query, params)
            return [json.loads(row[0]) for row in cursor.fetchall()]
