"""CLI subcommands for Trade Journal inspection and reporting."""

from __future__ import annotations

import argparse
import sys
from typing import TextIO

from xau_bot.config import Settings
from xau_bot.journal.models import TradeOutcome
from xau_bot.journal.reporter import JournalReporter
from xau_bot.journal.repository import JournalRepository
from xau_bot.llm.narrator import LLMNarrator


def build_journal_parser(parser: argparse.ArgumentParser) -> None:
    """Configures CLI arguments for the journal subcommand."""
    default_db = Settings().journal_db
    parser.add_argument(
        "--db",
        type=str,
        default=default_db,
        help=f"Caminho para o banco de dados SQLite do journal (padrão: {default_db}).",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["post-mortem", "report", "list", "pending"],
        help="Ação posicional opcional (post-mortem, report, list, pending).",
    )
    parser.add_argument(
        "entry_id",
        nargs="?",
        help="ID da proposta para post-mortem ou report.",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--list",
        action="store_true",
        help="Lista as entradas recentes do diário de trading.",
    )
    group.add_argument(
        "--report",
        type=str,
        metavar="PROPOSAL_ID",
        help="Gera relatório detalhado em Markdown para o ID da proposta.",
    )
    group.add_argument(
        "--post-mortem",
        dest="post_mortem",
        type=str,
        metavar="PROPOSAL_ID",
        help="Gera relatório de post-mortem com parecer do narrador para o ID da proposta.",
    )
    group.add_argument(
        "--pending",
        "--evaluate",
        dest="pending",
        action="store_true",
        help="Lista propostas não executadas ou expiradas pendentes de avaliação.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Quantidade máxima de entradas a listar (padrão: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretório onde salvar o arquivo de relatório gerado.",
    )
    parser.add_argument(
        "--narrate",
        action="store_true",
        help="Gera parecer qualitativo do Narrador LLM para o trade.",
    )


def run_journal_cli(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Executes the journal CLI command."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    action = getattr(args, "action", None)
    if action == "post-mortem":
        if not getattr(args, "entry_id", None):
            err.write("Erro: ID da proposta é obrigatório para post-mortem.\n")
            return 2
        args.report = args.entry_id
        args.narrate = True
    elif action == "report":
        if not getattr(args, "entry_id", None):
            err.write("Erro: ID da proposta é obrigatório para report.\n")
            return 2
        args.report = args.entry_id
    elif action == "list":
        args.list = True
    elif action == "pending":
        args.pending = True
    elif getattr(args, "post_mortem", None):
        args.report = args.post_mortem
        args.narrate = True

    has_command = (
        getattr(args, "list", False)
        or getattr(args, "report", None)
        or getattr(args, "pending", False)
    )
    if not has_command:
        err.write(
            "Erro: É necessário especificar uma ação "
            "(--list, --report, --pending ou post-mortem <id>).\n"
        )
        return 2

    repo = JournalRepository(db_path=args.db)

    try:
        if args.list:
            entries = repo.list_entries(limit=args.limit)
            if not entries:
                out.write("Nenhuma entrada encontrada no diário de trading.\n")
                return 0

            out.write(
                f"{'DATA (UTC)':<20} | {'PROPOSTA ID':<16} | {'DIRECAO':<7} | {'STATUS':<16} | "
                f"{'DESFECHO':<11} | {'RESULTADO R':<11}\n"
            )
            out.write("-" * 92 + "\n")
            for e in entries:
                r_val = (
                    f"{e.outcome_data.realized_r:+.2f}R"
                    if e.outcome_data and e.outcome_data.executed
                    else (f"({e.outcome_data.realized_r:+.2f}R hip)" if e.outcome_data else "N/A")
                )
                ts_str = e.created_at.strftime("%Y-%m-%d %H:%M")
                out.write(
                    f"{ts_str:<20} | {e.proposal_id:<16} | {e.direction.value:<7} | "
                    f"{e.status.value:<16} | {e.outcome.value:<11} | {r_val:<11}\n"
                )
            return 0

        if args.report:
            entry = repo.get_by_proposal_id(args.report)
            if not entry:
                err.write(f"Erro: Proposta '{args.report}' não encontrada no banco {args.db}.\n")
                return 1

            narrative = None
            if getattr(args, "narrate", False):
                narrator = LLMNarrator()
                narrative = narrator.narrate_trade_post_mortem(entry)

            markdown_report = JournalReporter.generate_markdown_report(entry, narrative=narrative)
            out.write(markdown_report + "\n")

            if args.output_dir:
                saved_path = JournalReporter.export_report_file(
                    entry, args.output_dir, narrative=narrative
                )
                out.write(f"\n[INFO] Relatório exportado com sucesso em: {saved_path}\n")

            return 0

        if args.pending:
            entries = repo.list_entries(limit=args.limit)
            unexec = [
                e for e in entries if e.outcome in (TradeOutcome.UNEXECUTED, TradeOutcome.EXPIRED)
            ]
            if not unexec:
                out.write("Nenhuma entrada não executada ou expirada para avaliar.\n")
                return 0

            out.write(f"Encontradas {len(unexec)} entrada(s) pendente(s) de avaliação:\n")
            for e in unexec:
                reason = e.outcome_data.exit_reason if e.outcome_data else "Pendente"
                out.write(f"- {e.proposal_id}: {e.status.value} [{e.outcome.value}] ({reason})\n")
            return 0

        return 0
    finally:
        repo.close()
