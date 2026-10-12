"""Tests for CLI subcommands xau-bot params and xau-bot validate."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xau_bot.cli.main import main
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


def test_cli_params_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["params", "--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "xau-bot params" in captured.out or "usage:" in captured.out


def test_cli_validate_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "--help"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "xau-bot validate" in captured.out or "usage:" in captured.out


def test_cli_params_list_empty(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_path = tmp_path / "params.db"
    ParameterRegistry(db_path).close()
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["params", "list", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out = stdout.getvalue()
    assert "Nenhuma versão de parâmetros registrada" in out or "[]" in out


def test_cli_params_create_and_list(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "params",
            "create",
            "--db",
            str(db_path),
            "--param",
            "min_displacement_multiplier=1.2",
            "--param",
            "max_spread_atr_ratio=0.04",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    assert "pv-" in stdout.getvalue()

    stdout_list = io.StringIO()
    stderr_list = io.StringIO()
    exit_code = main(
        ["params", "list", "--db", str(db_path)], stdout=stdout_list, stderr=stderr_list
    )
    assert exit_code == 0
    out_list = stdout_list.getvalue()
    assert "min_displacement_multiplier" in out_list
    assert "1.2" in out_list


def test_cli_params_rollback(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"

    main(
        ["params", "create", "--db", str(db_path), "--param", "max_spread_atr_ratio=0.20"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    main(
        ["params", "create", "--db", str(db_path), "--param", "max_spread_atr_ratio=0.35"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    stdout_json = io.StringIO()
    main(
        ["params", "list", "--db", str(db_path), "--json"],
        stdout=stdout_json,
        stderr=io.StringIO(),
    )
    versions = json.loads(stdout_json.getvalue())
    assert len(versions) == 2
    v1_id = versions[1]["version_id"]

    stdout_rb = io.StringIO()
    stderr_rb = io.StringIO()
    exit_code = main(
        ["params", "rollback", "--db", str(db_path), "--version", v1_id],
        stdout=stdout_rb,
        stderr=stderr_rb,
    )
    assert exit_code == 0
    assert "Rollback executado com sucesso" in stdout_rb.getvalue()


def test_cli_params_rollback_non_existent(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_path = tmp_path / "params.db"
    ParameterRegistry(db_path).close()
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["params", "rollback", "--db", str(db_path), "--version", "invalid_id"],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 1
    assert "não encontrada" in stderr.getvalue() or "not found" in stderr.getvalue()


def test_cli_validate_missing_db() -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["validate", "--db", "missing_db.db"], stdout=stdout, stderr=stderr)
    assert exit_code == 1
    assert "não encontrado" in stderr.getvalue()


def test_cli_validate_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    from xau_bot.journal.repository import JournalRepository

    repo = JournalRepository(db_path)
    repo.close()

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["validate", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    assert "Nenhuma hipótese" in stdout.getvalue()


def test_cli_params_show_and_json(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_path = tmp_path / "params.db"
    ParameterRegistry(db_path).close()
    stdout_empty = io.StringIO()
    main(["params", "show", "--db", str(db_path)], stdout=stdout_empty)
    assert "Nenhuma versão ativa" in stdout_empty.getvalue()

    main(
        [
            "params",
            "create",
            "--db",
            str(db_path),
            "--param",
            "min_displacement_multiplier=1.2",
            "--param",
            "max_daily_trades=5",
        ],
        stdout=io.StringIO(),
    )

    stdout_show = io.StringIO()
    exit_code = main(["params", "show", "--db", str(db_path)], stdout=stdout_show)
    assert exit_code == 0
    assert "min_displacement_multiplier" in stdout_show.getvalue()

    stdout_json = io.StringIO()
    exit_code = main(["params", "show", "--db", str(db_path), "--json"], stdout=stdout_json)
    assert exit_code == 0
    parsed = json.loads(stdout_json.getvalue())
    assert parsed["parameters"]["min_displacement_multiplier"] == 1.2
    assert parsed["parameters"]["max_daily_trades"] == 5


def test_cli_params_create_invalid_args(tmp_path: Path) -> None:
    db_path = tmp_path / "params.db"
    stderr = io.StringIO()
    code1 = main(["params", "create", "--db", str(db_path)], stderr=stderr)
    assert code1 == 2

    stderr2 = io.StringIO()
    code2 = main(
        ["params", "create", "--db", str(db_path), "--param", "invalid_param"], stderr=stderr2
    )
    assert code2 == 2


def test_cli_validate_invalid_params() -> None:
    stderr = io.StringIO()
    code = main(["validate", "--db", ":memory:", "--fdr-q", "2.0"], stderr=stderr)
    assert code == 2


def test_cli_validate_with_hypotheses_and_output_dir(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    repo = JournalRepository(db_path)
    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

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
            loss_category=LossCategory.BAD_DECISION,
            loss_cause=LossRootCause.INSUFFICIENT_DISPLACEMENT,
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

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
            realized_r=1.5,
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
            displacement_atr=1.5,
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
            outcome_data=out,
            created_at=now,
        )
        repo.save_entry(entry)

    for i in range(75):
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

    out_dir = tmp_path / "out_val"
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        [
            "validate",
            "--db",
            str(db_path),
            "--output-dir",
            str(out_dir),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    out_val = stdout.getvalue()
    assert "min_displacement_multiplier" in out_val
    assert "VALIDATED" in out_val

    reports = list(out_dir.glob("validation_report_*.md"))
    assert len(reports) == 1

    stdout_rej = io.StringIO()
    stderr_rej = io.StringIO()
    code_rej = main(
        [
            "validate",
            "--db",
            str(db_path),
            "--min-retention",
            "0.85",
        ],
        stdout=stdout_rej,
        stderr=stderr_rej,
    )
    assert code_rej == 0
    assert "REJECTED" in stdout_rej.getvalue()


def test_cli_params_missing_db_exit_1(tmp_path: Path) -> None:
    non_existent = str(tmp_path / "missing.db")
    stderr = io.StringIO()
    stdout = io.StringIO()

    assert main(["params", "list", "--db", non_existent], stdout=stdout, stderr=stderr) == 1
    assert not Path(non_existent).exists()

    assert main(["params", "show", "--db", non_existent], stdout=stdout, stderr=stderr) == 1
    assert not Path(non_existent).exists()

    assert (
        main(["params", "rollback", "v1", "--db", non_existent], stdout=stdout, stderr=stderr) == 1
    )
    assert not Path(non_existent).exists()


def test_cli_params_create_strict_validation(tmp_path: Path) -> None:
    stderr = io.StringIO()
    stdout = io.StringIO()

    code_unknown = main(
        ["params", "create", "--db", ":memory:", "--param", "foo_bar=100"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code_unknown == 2
    assert "desconhecido" in stderr.getvalue().lower() or "inválido" in stderr.getvalue().lower()

    stderr2 = io.StringIO()
    code_invalid = main(
        ["params", "create", "--db", ":memory:", "--param", "min_displacement_multiplier=nan_str"],
        stdout=stdout,
        stderr=stderr2,
    )
    assert code_invalid == 2

    stderr_nan = io.StringIO()
    code_nan = main(
        ["params", "create", "--db", ":memory:", "--param", "min_displacement_multiplier=nan"],
        stdout=stdout,
        stderr=stderr_nan,
    )
    assert code_nan == 2
    assert "nan" in stderr_nan.getvalue().lower()

    stderr_range = io.StringIO()
    code_range = main(
        ["params", "create", "--db", ":memory:", "--param", "risk_per_trade_pct=50"],
        stdout=stdout,
        stderr=stderr_range,
    )
    assert code_range == 2
    assert "faixa" in stderr_range.getvalue().lower()


def test_cli_params_promote_subcommand(tmp_path: Path) -> None:
    import sqlite3

    journal_path = tmp_path / "journal.db"
    params_path = tmp_path / "params.db"

    conn = sqlite3.connect(str(journal_path))
    try:
        conn.execute(
            """
            CREATE TABLE validation_results (
                hypothesis_id TEXT PRIMARY KEY,
                parameter_name TEXT NOT NULL,
                current_value TEXT NOT NULL,
                proposed_value TEXT NOT NULL,
                verdict TEXT NOT NULL,
                expectancy_delta_r REAL NOT NULL,
                drawdown_delta_pct REAL NOT NULL,
                trade_retention_rate REAL NOT NULL,
                rejection_reason TEXT,
                validated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO validation_results VALUES (
                'HYP-REJ', 'risk_per_trade_pct', '1.0', '2.0', 'REJECTED',
                -0.10, 5.0, 0.65, 'Drawdown excessivo', '2026-10-08T12:00:00Z', '{}'
            );
            """
        )
        conn.execute(
            """
            INSERT INTO validation_results VALUES (
                'HYP-OK', 'min_displacement_multiplier', '1.0', '1.5', 'VALIDATED',
                0.25, -0.5, 0.85, '', '2026-10-08T12:00:00Z', '{}'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    stdout_missing = io.StringIO()
    stderr_missing = io.StringIO()
    code_missing = main(
        [
            "params",
            "promote",
            "--hypothesis-id",
            "HYP-NAO-EXISTE",
            "--journal-db",
            str(journal_path),
            "--db",
            str(params_path),
        ],
        stdout=stdout_missing,
        stderr=stderr_missing,
    )
    assert code_missing == 1
    assert "não encontrada" in stderr_missing.getvalue().lower()

    stderr_no_jrn = io.StringIO()
    code_no_jrn = main(
        [
            "params",
            "promote",
            "--hypothesis-id",
            "HYP-OK",
            "--journal-db",
            str(tmp_path / "non_existent_journal.db"),
            "--db",
            str(params_path),
        ],
        stderr=stderr_no_jrn,
    )
    assert code_no_jrn == 1
    assert "não encontrado" in stderr_no_jrn.getvalue().lower()

    stdout_rej = io.StringIO()
    stderr_rej = io.StringIO()
    code_rej = main(
        [
            "params",
            "promote",
            "--hypothesis-id",
            "HYP-REJ",
            "--journal-db",
            str(journal_path),
            "--db",
            str(params_path),
        ],
        stdout=stdout_rej,
        stderr=stderr_rej,
    )
    assert code_rej == 1
    assert "não pode ser promovida" in stderr_rej.getvalue().lower()

    stdout_ok = io.StringIO()
    stderr_ok = io.StringIO()
    code_ok = main(
        [
            "params",
            "promote",
            "--hypothesis-id",
            "HYP-OK",
            "--journal-db",
            str(journal_path),
            "--db",
            str(params_path),
        ],
        stdout=stdout_ok,
        stderr=stderr_ok,
    )
    assert code_ok == 0
    assert "HYP-OK" in stdout_ok.getvalue()
    assert "promovida com sucesso" in stdout_ok.getvalue()


def test_cli_validate_fdr_rejection_scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer
    from xau_bot.learning.models import Hypothesis, HypothesisStatus, LearningReviewReport

    db_path = tmp_path / "journal.db"
    JournalRepository(db_path).close()

    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="HYP-FDR-FAIL",
        created_at=now,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.50,
        baseline_loss_rate=0.48,
        p_value=0.85,
        z_score=0.2,
        observed_expectancy_r=0.05,
        status=HypothesisStatus.PROPOSED,
        description="FDR failure test",
        rationale="FDR failure test",
    )
    mock_report = LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=100,
        total_wins=60,
        total_losses=40,
        total_breakevens=0,
        win_rate=0.60,
        loss_rate=0.40,
        global_expectancy_r=0.50,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={},
        feature_stats=(),
        hypotheses=(hyp,),
        accumulating_observations=(),
    )
    monkeypatch.setattr(LearningAnalyzer, "run_review", lambda self: mock_report)

    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        ["validate", "--db", str(db_path), "--fdr-q", "0.05"],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    out_val = stdout.getvalue()
    assert "HYP-FDR-FAIL" in out_val
    assert "REJECTED" in out_val
    assert "FDR" in out_val
