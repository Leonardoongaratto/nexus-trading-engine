"""Unit tests for the CLI learn subcommand (T-A6)."""

from __future__ import annotations

import io
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from xau_bot.cli.main import build_parser, main
from xau_bot.domain.models import Direction
from xau_bot.journal import (
    JournalEntry,
    JournalRepository,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)


def _seed_db(db_path: Path) -> None:
    repo = JournalRepository(db_path)
    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

    # Seed 35 trades with low displacement (< 1.0 ATR) and high losses
    for i in range(25):
        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=0.7,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
        )
        out = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            realized_r=-1.0,
        )
        entry = JournalEntry(
            id=f"jrn_loss_{i}",
            proposal_id=f"prop_loss_{i}",
            snapshot=snap,
            direction=Direction.BULLISH,
            outcome=TradeOutcome.LOSS,
            loss_cause=LossRootCause.INSUFFICIENT_DISPLACEMENT,
            loss_category=LossCategory.BAD_DECISION,
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

    for i in range(10):
        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=0.7,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
        )
        out = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            realized_r=2.0,
        )
        entry = JournalEntry(
            id=f"jrn_win_{i}",
            proposal_id=f"prop_win_{i}",
            snapshot=snap,
            direction=Direction.BULLISH,
            outcome=TradeOutcome.WIN,
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

    # 65 normal trades (displacement = 1.8 ATR): 60 wins, 5 losses
    for i in range(5):
        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=1.8,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
        )
        out = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            realized_r=-1.0,
        )
        entry = JournalEntry(
            id=f"jrn_norm_loss_{i}",
            proposal_id=f"prop_norm_loss_{i}",
            snapshot=snap,
            direction=Direction.BULLISH,
            outcome=TradeOutcome.LOSS,
            loss_cause=LossRootCause.NORMAL_VARIANCE,
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

    for i in range(60):
        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=1.8,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
        )
        out = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            realized_r=2.0,
        )
        entry = JournalEntry(
            id=f"jrn_norm_win_{i}",
            proposal_id=f"prop_norm_win_{i}",
            snapshot=snap,
            direction=Direction.BULLISH,
            outcome=TradeOutcome.WIN,
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

    repo.close()


def test_cli_learn_parser_help() -> None:
    parser = build_parser()
    subparsers_actions = [action for action in parser._actions if action.dest == "command"]
    assert len(subparsers_actions) == 1
    choices = subparsers_actions[0].choices
    assert choices is not None
    assert "learn" in choices


def test_cli_learn_empty_db() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    repo = JournalRepository(db_path)
    repo.close()

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["learn", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out_val = stdout.getvalue()
    assert "# Relatório de Aprendizado e Revisão Periódica" in out_val
    assert "Total de Trades Analisados | 0" in out_val

    db_path.unlink(missing_ok=True)


def test_cli_learn_with_data_markdown() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    _seed_db(db_path)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["learn", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out_val = stdout.getvalue()
    assert "Total de Trades Analisados | 100" in out_val
    assert "INSUFFICIENT_DISPLACEMENT" in out_val
    assert "min_displacement_multiplier" in out_val

    db_path.unlink(missing_ok=True)


def test_cli_learn_with_data_json() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    _seed_db(db_path)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["learn", "--db", str(db_path), "--json"], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out_val = stdout.getvalue()
    parsed = json.loads(out_val)
    assert parsed["total_trades_analyzed"] == 100
    assert parsed["total_losses"] == 30
    assert parsed["loss_cause_distribution"]["INSUFFICIENT_DISPLACEMENT"] == 25

    db_path.unlink(missing_ok=True)


def test_cli_learn_output_dir(tmp_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    _seed_db(db_path)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["learn", "--db", str(db_path), "--output-dir", str(tmp_path)],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    generated_files = list(tmp_path.glob("learning_review_*.md"))
    assert len(generated_files) == 1

    db_path.unlink(missing_ok=True)


def test_cli_learn_output_dir_json(tmp_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    _seed_db(db_path)

    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["learn", "--db", str(db_path), "--output-dir", str(tmp_path), "--json"],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    generated_files = list(tmp_path.glob("learning_review_*.json"))
    assert len(generated_files) == 1

    db_path.unlink(missing_ok=True)


def test_cli_learn_invalid_db_path() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(["learn", "--db", "nonexistent_file_xyz.db"], stdout=stdout, stderr=stderr)
    assert exit_code == 1
    assert "não encontrado" in stderr.getvalue()


def test_cli_learn_invalid_param_value() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        ["learn", "--db", ":memory:", "--min-samples", "0"], stdout=stdout, stderr=stderr
    )
    assert exit_code == 2
    assert "Erro de parâmetro" in stderr.getvalue()
