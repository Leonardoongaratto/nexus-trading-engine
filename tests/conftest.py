"""Pytest fixtures and configuration for xau_bot tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_journal_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    db_file = tmp_path / "test_session_journal.db"
    monkeypatch.setenv("XAU_JOURNAL_DB", str(db_file))
    yield
