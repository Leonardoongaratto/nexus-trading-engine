"""CLI subcommand for empirical hypothesis validation and backtesting."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from xau_bot.domain.loader import (
    load_candles_from_file,
    load_setups_from_file,
    load_ticks_from_file,
)
from xau_bot.journal.models import JournalEntry, TradeOutcome
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.analyzer import LearningAnalyzer
from xau_bot.learning.models import HypothesisStatus, ValidationMetrics, ValidationResult
from xau_bot.learning.stats import calculate_expectancy, fdr_benjamini_hochberg
from xau_bot.learning.validator import HypothesisValidator


def build_validate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        default="journal.db",
        help="Caminho para o banco de dados SQLite do journal (padrão: journal.db).",
    )
    parser.add_argument(
        "--candles",
        default=None,
        help="Caminho para arquivo de candles para validação via backtest causal.",
    )
    parser.add_argument(
        "--ticks",
        default=None,
        help="Caminho para arquivo de ticks para validação via backtest causal.",
    )
    parser.add_argument(
        "--setups",
        default=None,
        help="Caminho para arquivo JSON de setups pré-computados para validação.",
    )
    parser.add_argument(
        "--min-retention",
        type=float,
        default=0.70,
        help="Taxa mínima de retenção amostral de trades (padrão: 0.70).",
    )
    parser.add_argument(
        "--max-dd-ratio",
        type=float,
        default=1.10,
        help="Razão máxima tolerada de aumento de drawdown (padrão: 1.10).",
    )
    parser.add_argument(
        "--min-delta-r",
        type=float,
        default=0.0,
        help="Melhora mínima exigida de expectativa em R (padrão: 0.0).",
    )
    parser.add_argument(
        "--fdr-q",
        type=float,
        default=0.05,
        help="Limiar de FDR Benjamini-Hochberg para testes múltiplos (padrão: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretório onde salvar o relatório gerado.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emite o relatório em formato JSON estruturado no stdout.",
    )


def _compute_metrics_from_entries(entries: list[JournalEntry]) -> ValidationMetrics:
    valid_entries = [e for e in entries if e.outcome_data is not None]
    total = len(valid_entries)
    if total == 0:
        return ValidationMetrics(
            total_trades=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            expectancy_r=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
        )

    wins = sum(1 for e in valid_entries if e.outcome == TradeOutcome.WIN)
    losses = sum(1 for e in valid_entries if e.outcome == TradeOutcome.LOSS)
    r_multiples: list[float] = []

    gross_profit = 0.0
    gross_loss = 0.0

    peak = 10000.0
    balance = 10000.0
    max_dd_pct = 0.0

    for entry in valid_entries:
        outcome = entry.outcome_data
        assert outcome is not None
        r = outcome.realized_r
        pnl = outcome.pnl_usd

        r_multiples.append(r)
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += abs(pnl)

        balance += pnl
        if balance > peak:
            peak = balance
        dd_pct = ((peak - balance) / peak) * 100.0 if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    win_rate = round(wins / total, 4) if total > 0 else 0.0
    expectancy_r = calculate_expectancy(r_multiples) if r_multiples else 0.0
    profit_factor = (
        round(gross_profit / gross_loss, 4)
        if gross_loss > 0
        else (round(gross_profit, 4) if gross_profit > 0 else 0.0)
    )

    return ValidationMetrics(
        total_trades=total,
        win_count=wins,
        loss_count=losses,
        win_rate=win_rate,
        expectancy_r=expectancy_r,
        max_drawdown_pct=round(max_dd_pct, 4),
        profit_factor=profit_factor,
    )


def _filter_entries_for_hypothesis(
    entries: list[JournalEntry],
    parameter_name: str,
    proposed_value: str,
) -> list[JournalEntry]:
    if parameter_name == "min_displacement_multiplier":
        try:
            min_disp = float(proposed_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Valor numérico inválido para min_displacement_multiplier: '{proposed_value}'"
            ) from exc
        return [e for e in entries if e.snapshot.displacement_atr >= min_disp]

    if parameter_name == "max_spread_atr_ratio":
        try:
            max_ratio = float(proposed_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Valor numérico inválido para max_spread_atr_ratio: '{proposed_value}'"
            ) from exc
        filtered: list[JournalEntry] = []
        for e in entries:
            atr = e.snapshot.atr_m5
            if atr > 0:
                ratio = e.snapshot.spread_price / atr
                if ratio <= max_ratio:
                    filtered.append(e)
            else:
                filtered.append(e)
        return filtered

    return entries


def _save_validation_results_to_db(db_path: str, results: list[ValidationResult]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
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
        for r in results:
            conn.execute(
                """
                INSERT OR REPLACE INTO validation_results (
                    hypothesis_id, parameter_name, current_value, proposed_value,
                    verdict, expectancy_delta_r, drawdown_delta_pct, trade_retention_rate,
                    rejection_reason, validated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    r.hypothesis_id,
                    r.parameter_name,
                    r.current_value,
                    r.proposed_value,
                    r.verdict.value,
                    r.expectancy_delta_r,
                    r.drawdown_delta_pct,
                    r.trade_retention_rate,
                    r.rejection_reason,
                    (r.validated_at or datetime.now(UTC)).isoformat(),
                    json.dumps(r.to_dict()),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _render_markdown_report(results: list[ValidationResult]) -> str:
    lines = ["# Relatório de Validação de Hipóteses (T-A7)\n\n"]
    lines.append(
        "| ID Hipótese | Parâmetro | Proposto | Delta R | Retenção | Veredito | Justificativa |\n"
    )
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
    for r in results:
        lines.append(
            f"| `{r.hypothesis_id}` | `{r.parameter_name}` | `{r.proposed_value}` | "
            f"{r.expectancy_delta_r:+.2f}R | {r.trade_retention_rate * 100:.1f}% | "
            f"**{r.verdict.value}** | {r.rejection_reason or 'Aprovada'} |\n"
        )
    return "".join(lines)


def run_validate_cli(
    args: argparse.Namespace,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    try:
        validator = HypothesisValidator(
            min_trade_retention=float(args.min_retention),
            max_drawdown_tolerance_ratio=float(args.max_dd_ratio),
            min_expectancy_delta_r=float(args.min_delta_r),
            alpha=0.05,
        )
        q_threshold = float(args.fdr_q)
        if not (0.0 < q_threshold < 1.0):
            raise ValueError(f"fdr-q must be between 0.0 and 1.0, got {q_threshold}")
    except ValueError as exc:
        stderr.write(f"Erro de parâmetro: {exc}\n")
        return 2

    candles_arg = getattr(args, "candles", None)
    if candles_arg is not None and not Path(candles_arg).exists():
        stderr.write(f"Arquivo de candles não encontrado: '{candles_arg}'\n")
        return 1

    ticks_arg = getattr(args, "ticks", None)
    if ticks_arg is not None and not Path(ticks_arg).exists():
        stderr.write(f"Arquivo de ticks não encontrado: '{ticks_arg}'\n")
        return 1

    setups_arg = getattr(args, "setups", None)
    if setups_arg is not None and not Path(setups_arg).exists():
        stderr.write(f"Arquivo de setups não encontrado: '{setups_arg}'\n")
        return 1

    db_str = str(getattr(args, "db", "journal.db"))
    if db_str != ":memory:" and not Path(db_str).exists():
        stderr.write(f"Erro: Arquivo de banco de dados '{db_str}' não encontrado.\n")
        return 1

    try:
        repo = JournalRepository(db_str)
    except (sqlite3.Error, OSError) as exc:
        stderr.write(f"Erro ao inicializar repositório: {exc}\n")
        return 1

    try:
        analyzer = LearningAnalyzer(repo)
        report = analyzer.run_review()
        proposed_hypotheses = list(report.hypotheses)
        all_entries = repo.list_entries(limit=None)
    finally:
        repo.close()

    if not proposed_hypotheses:
        stdout.write("Nenhuma hipótese proposta (PROPOSED) encontrada para validação no banco.\n")
        return 0

    p_values = [h.p_value for h in proposed_hypotheses]
    fdr_results = fdr_benjamini_hochberg(p_values, q_threshold=q_threshold)

    results: list[ValidationResult] = []
    now = datetime.now(UTC)

    has_backtest_data = candles_arg is not None and ticks_arg is not None
    loaded_candles = (
        list(load_candles_from_file(Path(candles_arg)).closed_candles)
        if (candles_arg is not None and ticks_arg is not None)
        else None
    )
    loaded_ticks = (
        load_ticks_from_file(Path(ticks_arg))
        if (candles_arg is not None and ticks_arg is not None)
        else None
    )
    loaded_setups = load_setups_from_file(Path(setups_arg)) if setups_arg is not None else None

    journal_baseline_metrics = _compute_metrics_from_entries(all_entries)

    for hyp, passes_fdr in zip(proposed_hypotheses, fdr_results, strict=True):
        if not passes_fdr:
            val_res = ValidationResult(
                hypothesis_id=hyp.id,
                parameter_name=hyp.parameter_name,
                current_value=hyp.current_value,
                proposed_value=hyp.proposed_value,
                baseline_metrics=journal_baseline_metrics,
                candidate_metrics=journal_baseline_metrics,
                expectancy_delta_r=0.0,
                drawdown_delta_pct=0.0,
                trade_retention_rate=1.0,
                is_statistically_superior=False,
                p_value_improvement=hyp.p_value,
                verdict=HypothesisStatus.REJECTED,
                rejection_reason=(
                    f"Rejeitada pelo controle FDR Benjamini-Hochberg (q={q_threshold})."
                ),
                validated_at=now,
            )
            results.append(val_res)
            continue

        if has_backtest_data and loaded_ticks is not None:
            try:
                val_res = validator.validate(
                    hypothesis=hyp,
                    candles=loaded_candles,
                    ticks=loaded_ticks,
                    setups=loaded_setups,
                    oos_split_ratio=0.30,
                    validated_at=now,
                )
            except ValueError as exc:
                val_res = ValidationResult(
                    hypothesis_id=hyp.id,
                    parameter_name=hyp.parameter_name,
                    current_value=hyp.current_value,
                    proposed_value=hyp.proposed_value,
                    baseline_metrics=journal_baseline_metrics,
                    candidate_metrics=journal_baseline_metrics,
                    expectancy_delta_r=0.0,
                    drawdown_delta_pct=0.0,
                    trade_retention_rate=1.0,
                    is_statistically_superior=False,
                    p_value_improvement=hyp.p_value,
                    verdict=HypothesisStatus.REJECTED,
                    rejection_reason=f"Erro na validação em backtest: {exc}",
                    validated_at=now,
                )
            results.append(val_res)
            continue

        try:
            filtered_entries = _filter_entries_for_hypothesis(
                all_entries,
                hyp.parameter_name,
                hyp.proposed_value,
            )
            candidate_metrics = _compute_metrics_from_entries(filtered_entries)

            val_res = validator.evaluate_metrics(
                hypothesis=hyp,
                baseline=journal_baseline_metrics,
                candidate=candidate_metrics,
                oos_baseline=None,
                oos_candidate=None,
                validated_at=now,
            )
        except ValueError as exc:
            val_res = ValidationResult(
                hypothesis_id=hyp.id,
                parameter_name=hyp.parameter_name,
                current_value=hyp.current_value,
                proposed_value=hyp.proposed_value,
                baseline_metrics=journal_baseline_metrics,
                candidate_metrics=journal_baseline_metrics,
                expectancy_delta_r=0.0,
                drawdown_delta_pct=0.0,
                trade_retention_rate=1.0,
                is_statistically_superior=False,
                p_value_improvement=hyp.p_value,
                verdict=HypothesisStatus.REJECTED,
                rejection_reason=f"Valor proposto inválido: {exc}",
                validated_at=now,
            )

        results.append(val_res)

    if db_str != ":memory:":
        _save_validation_results_to_db(db_str, results)

    as_json = getattr(args, "json", False)
    if as_json:
        payload = [r.to_dict() for r in results]
        stdout.write(json.dumps(payload, indent=2))
        stdout.write("\n")
    else:
        stdout.write(_render_markdown_report(results))

    out_dir_str = getattr(args, "output_dir", None)
    if out_dir_str:
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"validation_report_{now.strftime('%Y%m%d_%H%M%S')}.{'json' if as_json else 'md'}"
        target_path = out_dir / fname
        if as_json:
            target_path.write_text(
                json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
            )
        else:
            target_path.write_text(_render_markdown_report(results), encoding="utf-8")

        msg = f"Relatório de validação salvo em: {target_path.resolve()}\n"
        if as_json:
            stderr.write(msg)
        else:
            stdout.write(f"\n{msg}")

    return 0
