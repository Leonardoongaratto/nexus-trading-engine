"""CLI subcommands for learning review and hypothesis generation."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TextIO

from xau_bot.config import Settings
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.analyzer import LearningAnalyzer
from xau_bot.learning.human import HumanDiscretionAnalyzer
from xau_bot.learning.reporter import LearningReporter
from xau_bot.llm.models import NarrativeReport
from xau_bot.llm.narrator import LLMNarrator


class SerializableReport(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def build_learn_parser(parser: argparse.ArgumentParser) -> None:
    """Configures CLI arguments for the learn subcommand."""
    default_db = Settings().journal_db
    parser.add_argument(
        "--db",
        type=str,
        default=default_db,
        help=f"Caminho para o banco de dados SQLite do journal (padrão: {default_db}).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Limiar mínimo amostral N para formulação de hipótese (padrão: 30).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Nível de significância estatística p-value bilateral (padrão: 0.05).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório onde salvar o arquivo de relatório gerado.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["review", "human"],
        default=None,
        help="Ação de aprendizado a executar ('review' ou 'human'). Padrão: review.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Executa a análise de discricionariedade humana (propostas aprovadas vs ignoradas).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emite o relatório em formato JSON estruturado em vez de Markdown.",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Gera parecer qualitativo do Narrador LLM para o relatório.",
    )


def _emit_report[TReport: SerializableReport](
    report: TReport,
    is_json: bool,
    output_dir: str | None,
    prefix: str,
    ts_str: str,
    markdown_generator: Callable[[TReport, NarrativeReport | None], str],
    out: TextIO,
    err: TextIO,
    narrative: NarrativeReport | None = None,
) -> None:
    if is_json:
        payload = report.to_dict()
        if narrative is not None:
            payload["narrative"] = narrative.to_dict()
        json_text = json.dumps(payload, indent=2, ensure_ascii=False)
        out.write(json_text + "\n")
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / f"{prefix}_{ts_str}.json"
            file_path.write_text(json_text, encoding="utf-8")
            err.write(f"Relatório JSON salvo em: {file_path}\n")
    else:
        md_text = markdown_generator(report, narrative)
        out.write(md_text + "\n")
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / f"{prefix}_{ts_str}.md"
            file_path.write_text(md_text, encoding="utf-8")
            out.write(f"\nRelatório Markdown salvo em: {file_path}\n")


def run_learn_cli(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Executes the learn CLI command."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    if getattr(args, "action", None) == "review" and getattr(args, "human", False):
        err.write("Erro: Ação 'review' e flag '--human' são mutuamente exclusivas.\n")
        return 2

    if args.db != ":memory:" and not Path(args.db).exists():
        err.write(f"Erro: Arquivo de banco de dados '{args.db}' não encontrado.\n")
        return 1

    try:
        repo = JournalRepository(db_path=args.db)
    except Exception as exc:
        err.write(f"Erro ao acessar banco de dados do journal: {exc}\n")
        return 1

    try:
        is_human = (getattr(args, "action", None) == "human") or getattr(args, "human", False)
        if is_human:
            try:
                human_analyzer = HumanDiscretionAnalyzer(
                    repository=repo,
                    min_samples=args.min_samples,
                    alpha=args.alpha,
                )
            except ValueError as val_err:
                err.write(f"Erro de parâmetro: {val_err}\n")
                return 2

            try:
                report_h = human_analyzer.analyze()
            except sqlite3.Error as db_err:
                err.write(f"Erro no banco de dados durante análise: {db_err}\n")
                return 1

            should_narrate = getattr(args, "narrate", False)
            narrator = LLMNarrator() if should_narrate else None
            narrative_h = narrator.narrate_human_discretion(report_h) if narrator else None

            ts_str = report_h.analyzed_at.strftime("%Y%m%d_%H%M%S")
            _emit_report(
                report=report_h,
                is_json=getattr(args, "json", False),
                output_dir=args.output_dir,
                prefix="human_discretion",
                ts_str=ts_str,
                markdown_generator=LearningReporter.generate_human_discretion_markdown,
                out=out,
                err=err,
                narrative=narrative_h,
            )
            return 0

        try:
            analyzer = LearningAnalyzer(
                repository=repo,
                min_samples=args.min_samples,
                alpha=args.alpha,
            )
        except ValueError as val_err:
            err.write(f"Erro de parâmetro: {val_err}\n")
            return 2

        try:
            report = analyzer.run_review()
        except sqlite3.Error as db_err:
            err.write(f"Erro no banco de dados durante revisão: {db_err}\n")
            return 1

        should_narrate = getattr(args, "narrate", False)
        narrator = LLMNarrator() if should_narrate else None
        narrative = narrator.narrate_learning_review(report) if narrator else None

        ts_str = report.reviewed_at.strftime("%Y%m%d_%H%M%S")
        _emit_report(
            report=report,
            is_json=getattr(args, "json", False),
            output_dir=args.output_dir,
            prefix="learning_review",
            ts_str=ts_str,
            markdown_generator=LearningReporter.generate_markdown_report,
            out=out,
            err=err,
            narrative=narrative,
        )
        return 0
    finally:
        repo.close()
