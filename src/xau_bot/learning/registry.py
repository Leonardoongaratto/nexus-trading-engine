"""Audit-trailed parameter registry with atomic versioning and rollbacks in SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xau_bot.learning.models import ParameterVersion


class ParameterRegistry:
    """Manages versioned configuration snapshots and rollback history in SQLite."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def __enter__(self) -> ParameterRegistry:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _initialize_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parameter_versions (
                    version_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    source_hypothesis_id TEXT,
                    applied_by TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    rollback_target_id TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_param_versions_created
                ON parameter_versions(created_at);
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_param_versions_active
                ON parameter_versions(is_active);
                """
            )

    def create_version(
        self,
        parameters: dict[str, Any],
        source_hypothesis_id: str | None = None,
        applied_by: str = "operator",
        activate: bool = True,
        created_at: datetime | None = None,
        rollback_target_id: str | None = None,
    ) -> ParameterVersion:
        version_id = f"pv-{uuid.uuid4().hex[:8]}"
        now = created_at or datetime.now(UTC)
        params_json = json.dumps(parameters, sort_keys=True)
        is_active_int = 1 if activate else 0

        with self._conn:
            if activate:
                self._conn.execute("UPDATE parameter_versions SET is_active = 0;")

            self._conn.execute(
                """
                INSERT INTO parameter_versions (
                    version_id, created_at, parameters_json,
                    source_hypothesis_id, applied_by, is_active, rollback_target_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    version_id,
                    now.isoformat(),
                    params_json,
                    source_hypothesis_id,
                    applied_by,
                    is_active_int,
                    rollback_target_id,
                ),
            )

        return ParameterVersion(
            version_id=version_id,
            created_at=now,
            parameters=parameters,
            source_hypothesis_id=source_hypothesis_id,
            applied_by=applied_by,
            is_active=activate,
            rollback_target_id=rollback_target_id,
        )

    def get_version(self, version_id: str) -> ParameterVersion | None:
        cursor = self._conn.execute(
            """
            SELECT version_id, created_at, parameters_json,
                   source_hypothesis_id, applied_by, is_active, rollback_target_id
            FROM parameter_versions
            WHERE version_id = ?;
            """,
            (version_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def get_active_version(self, raise_if_none: bool = False) -> ParameterVersion | None:
        cursor = self._conn.execute(
            """
            SELECT version_id, created_at, parameters_json,
                   source_hypothesis_id, applied_by, is_active, rollback_target_id
            FROM parameter_versions
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1;
            """
        )
        row = cursor.fetchone()
        if row is None:
            if raise_if_none:
                raise RuntimeError("No active parameter version found in registry.")
            return None
        return self._row_to_version(row)

    def list_versions(self) -> list[ParameterVersion]:
        cursor = self._conn.execute(
            """
            SELECT version_id, created_at, parameters_json,
                   source_hypothesis_id, applied_by, is_active, rollback_target_id
            FROM parameter_versions
            ORDER BY created_at DESC;
            """
        )
        return [self._row_to_version(row) for row in cursor.fetchall()]

    def rollback(self, target_version_id: str, applied_by: str = "operator") -> ParameterVersion:
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE;")
            target = self.get_version(target_version_id)
            if target is None:
                raise ValueError(f"Target version '{target_version_id}' not found.")

            active = self.get_active_version()
            if active is not None and active.version_id == target_version_id:
                raise ValueError(
                    f"Target version '{target_version_id}' is already the active version."
                )

            return self.create_version(
                parameters=target.parameters,
                source_hypothesis_id=target.source_hypothesis_id,
                applied_by=applied_by,
                activate=True,
                rollback_target_id=target_version_id,
            )

    def _row_to_version(self, row: sqlite3.Row) -> ParameterVersion:
        raw_ts = row["created_at"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        return ParameterVersion(
            version_id=str(row["version_id"]),
            created_at=ts,
            parameters=json.loads(row["parameters_json"]),
            source_hypothesis_id=row["source_hypothesis_id"],
            applied_by=str(row["applied_by"]),
            is_active=bool(row["is_active"]),
            rollback_target_id=row["rollback_target_id"],
        )
