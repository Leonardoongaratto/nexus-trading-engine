"""Unit tests for the CLI journal subcommand."""

from __future__ import annotations

import io
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from xau_bot.cli.main import build_parser, main
from xau_bot.domain.models import Direction
from xau_bot.journal import (
    JournalEntry,
    JournalRepository,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)


def _seed_db(db_path: Path) -> str:
    repo = JournalRepository(db_path)
    now = datetime(2026, 10, 8, 11, 0, tzinfo=UTC)

    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        user_decision="APPROVED",
    )
    out_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2640.0,
        mfe_r=2.2,
        mae_r=0.3,
        realized_r=2.0,
        pnl_usd=400.0,
        duration_seconds=1200.0,
        exit_reason="Take Profit Hit",
    )
    entry = JournalEntry(
        id="jrn_cli_01",
        proposal_id="prop_cli_01",
        snapshot=snap,
        ticket=1234,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        outcome=TradeOutcome.WIN,
        outcome_data=out_data,
        created_at=now,
    )
    repo.save_entry(entry)
    repo.close()
    return "prop_cli_01"


def test_cli_journal_parser_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(["journal", "--list"])
    assert args.command == "journal"
    assert args.list is True


def test_cli_journal_list_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "empty.db"
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=["journal", "--db", str(db_path), "--list"],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 0
        output = stdout.getvalue()
        assert "Nenhuma entrada encontrada" in output


def test_cli_journal_list_with_data() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        prop_id = _seed_db(db_path)

        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=["journal", "--db", str(db_path), "--list"],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 0
        output = stdout.getvalue()
        assert prop_id in output
        assert "WIN" in output
        assert "+2.00R" in output
        assert "BULLISH" in output
        assert "PENDING_APPROVAL" in output
        assert "2026-10-08" in output


def test_cli_journal_report_success() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        prop_id = _seed_db(db_path)
        out_dir = Path(tmp_dir) / "reports"

        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=[
                "journal",
                "--db",
                str(db_path),
                "--report",
                prop_id,
                "--output-dir",
                str(out_dir),
            ],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 0
        output = stdout.getvalue()
        assert "# Trade Post-Mortem Report" in output
        assert prop_id in output
        assert "Relatório exportado com sucesso" in output
        assert out_dir.exists()


def test_cli_journal_report_not_found() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        _seed_db(db_path)

        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=["journal", "--db", str(db_path), "--report", "prop_unknown"],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 1
        err_output = stderr.getvalue()
        assert "não encontrada" in err_output


def test_cli_journal_evaluate_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "empty.db"
        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=["journal", "--db", str(db_path), "--evaluate"],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 0
        output = stdout.getvalue()
        assert "Nenhuma entrada" in output


def test_cli_journal_evaluate_with_unexecuted_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "unexec.db"
        repo = JournalRepository(db_path)
        now = datetime(2026, 10, 8, 11, 0, tzinfo=UTC)

        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=1.5,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
            user_decision="PENDING",
        )
        out_data = OutcomeData(
            executed=False,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            exit_reason="Proposal expired by timeout",
        )
        entry = JournalEntry(
            id="jrn_unexec",
            proposal_id="prop_unexec_01",
            snapshot=snap,
            direction=Direction.BULLISH,
            outcome=TradeOutcome.EXPIRED,
            outcome_data=out_data,
            created_at=now,
        )
        repo.save_entry(entry)
        repo.close()

        stdout = io.StringIO()
        stderr = io.StringIO()

        code = main(
            argv=["journal", "--db", str(db_path), "--evaluate"],
            stdout=stdout,
            stderr=stderr,
        )

        assert code == 0
        output = stdout.getvalue()
        assert "prop_unexec_01" in output
        assert "EXPIRED" in output
