"""Markdown and SVG report generator for Trade Journal entries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from xau_bot.journal.models import JournalEntry, LossCategory, TradeOutcome

if TYPE_CHECKING:
    from xau_bot.llm.models import NarrativeReport


class JournalReporter:
    """Generates Markdown post-trade evaluation reports and visual price level diagrams."""

    @staticmethod
    def generate_svg_chart(entry: JournalEntry) -> str:
        """Generates an embedded lightweight SVG diagram depicting price levels and outcome."""
        out = entry.outcome_data
        entry_p = out.entry_price if out else entry.snapshot.poi_level
        sl_p = out.stop_loss if out else 0.0
        tp_p = out.take_profit if out else 0.0
        exit_p = out.exit_price if out and out.exit_price else entry_p

        prices = [p for p in (entry_p, sl_p, tp_p, exit_p) if p > 0]
        if not prices:
            return ""

        min_p = min(prices) - 2.0
        max_p = max(prices) + 2.0
        span = max(1.0, max_p - min_p)

        height = 200
        width = 500
        pad_top = 25
        usable_h = height - 50

        def y_coord(val: float) -> float:
            ratio = (val - min_p) / span
            return pad_top + usable_h * (1.0 - ratio)

        y_entry = y_coord(entry_p)
        y_sl = y_coord(sl_p) if sl_p > 0 else 0
        y_tp = y_coord(tp_p) if tp_p > 0 else 0
        y_exit = y_coord(exit_p)

        color_outcome = (
            "#22c55e"
            if entry.outcome == TradeOutcome.WIN
            else ("#ef4444" if entry.outcome == TradeOutcome.LOSS else "#3b82f6")
        )

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'style="background-color: #1e1e24; border-radius: 8px; font-family: monospace;">',
            (
                f'<text x="20" y="20" fill="#94a3b8" font-size="12">'
                f"Preço XAU/USD ({entry.direction.value})</text>"
            ),
            (
                f'<line x1="80" y1="{y_entry}" x2="460" y2="{y_entry}" '
                f'stroke="#38bdf8" stroke-width="2" stroke-dasharray="4"/>'
            ),
            (
                f'<text x="85" y="{y_entry - 5}" fill="#38bdf8" font-size="11">'
                f"Entrada: {entry_p:.2f}</text>"
            ),
        ]

        if tp_p > 0:
            svg_parts.extend(
                [
                    (
                        f'<line x1="80" y1="{y_tp}" x2="460" y2="{y_tp}" '
                        f'stroke="#22c55e" stroke-width="2"/>'
                    ),
                    (
                        f'<text x="85" y="{y_tp - 5}" fill="#22c55e" font-size="11">'
                        f"Take Profit: {tp_p:.2f}</text>"
                    ),
                ]
            )

        if sl_p > 0:
            svg_parts.extend(
                [
                    (
                        f'<line x1="80" y1="{y_sl}" x2="460" y2="{y_sl}" '
                        f'stroke="#ef4444" stroke-width="2"/>'
                    ),
                    (
                        f'<text x="85" y="{y_sl - 5}" fill="#ef4444" font-size="11">'
                        f"Stop Loss: {sl_p:.2f}</text>"
                    ),
                ]
            )

        if out and out.exit_price:
            svg_parts.extend(
                [
                    (
                        f'<circle cx="350" cy="{y_exit}" r="6" fill="{color_outcome}" '
                        f'stroke="#ffffff" stroke-width="2"/>'
                    ),
                    (
                        f'<text x="365" y="{y_exit + 4}" fill="{color_outcome}" '
                        f'font-size="11" font-weight="bold">Saída: {exit_p:.2f}</text>'
                    ),
                ]
            )

        svg_parts.append("</svg>")
        return "".join(svg_parts)

    @classmethod
    def generate_markdown_report(
        cls,
        entry: JournalEntry,
        narrative: NarrativeReport | None = None,
    ) -> str:
        """Generates a complete post-mortem markdown audit report for a journal entry."""
        out = entry.outcome_data
        snap = entry.snapshot

        svg = cls.generate_svg_chart(entry)

        lines = [
            f"# Trade Post-Mortem Report — {entry.proposal_id}",
            "",
            f"- **Data/Hora**: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Ativo**: {entry.symbol} | **Direção**: {entry.direction.value}",
            f"- **Status Proposta**: {entry.status.value}",
            f"- **Desfecho**: **{entry.outcome.value}**",
        ]

        if entry.ticket:
            lines.append(f"- **Ticket Broker**: #{entry.ticket}")

        if out and out.executed:
            lines.extend(
                [
                    f"- **Resultado em R**: **{out.realized_r:+.2f}R**",
                    f"- **PnL (USD)**: ${out.pnl_usd:+.2f}",
                    f"- **MFE (Máx Favorável)**: +{out.mfe_r:.2f}R",
                    f"- **MAE (Máx Adverso)**: -{out.mae_r:.2f}R",
                    f"- **Duração**: {out.duration_seconds:.1f}s",
                    f"- **Motivo de Saída**: {out.exit_reason or 'N/A'}",
                ]
            )
            if out.post_exit_behavior:
                lines.append(f"- **Comportamento Pós-Saída**: {out.post_exit_behavior}")
        elif out and not out.executed:
            lines.extend(
                [
                    "- **Executado**: Não (Sugestão Ignorada/Expirada)",
                    f"- **Resultado Hipotético**: {out.realized_r:+.2f}R",
                    f"- **MFE Hipotético**: +{out.mfe_r:.2f}R",
                    f"- **MAE Hipotético**: -{out.mae_r:.2f}R",
                    f"- **Motivo**: {out.exit_reason}",
                ]
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 1. Diagnóstico de Causa Raiz",
                "",
            ]
        )

        if entry.outcome == TradeOutcome.LOSS:
            cat_desc = (
                "Variância Normal (Estatística Esperada)"
                if entry.loss_category == LossCategory.NORMAL_VARIANCE
                else "Decisão Ruim / Desvio Operacional"
            )
            cause_desc = entry.loss_cause.value if entry.loss_cause else "Desconhecido"
            lines.extend(
                [
                    f"- **Categoria da Perda**: {cat_desc}",
                    f"- **Causa Raiz Diagnosticada**: `{cause_desc}`",
                    "",
                    "> [!NOTE]",
                    "> Diagnóstico baseado em viés HTF, deslocamento ATR e macroeconomia.",
                ]
            )
        elif entry.outcome == TradeOutcome.WIN:
            lines.append("Operação vencedora em conformidade com as confluências SMC.")
        elif entry.outcome == TradeOutcome.BREAKEVEN:
            lines.append(
                "Operação encerrada no ponto de equilíbrio (+ custos) via proteção dinâmica."
            )
        else:
            lines.append(f"Sugestão com desfecho {entry.outcome.value}.")

        news_str = (
            f"{snap.news_distance_min:.1f} min"
            if snap.news_distance_min is not None
            else "Sem notícias"
        )
        conf_str = ", ".join(snap.confluences) if snap.confluences else "Padrão SMC"

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. Snapshot Técnico no Momento da Entrada",
                "",
                "| Parâmetro | Valor Registrado |",
                "| :--- | :--- |",
                f"| **Sessão** | {snap.session} |",
                f"| **Viés HTF** | {snap.htf_bias} |",
                f"| **POI** | {snap.poi_type} @ {snap.poi_level:.2f} |",
                f"| **Deslocamento** | {snap.displacement_atr:.2f} × ATR |",
                f"| **Tamanho FVG** | {snap.fvg_size_atr:.2f} × ATR |",
                f"| **ChoCH Nível** | {snap.choch_level:.2f} |",
                f"| **Spread no Momento** | ${snap.spread_price:.2f} |",
                f"| **ATR M5** | ${snap.atr_m5:.2f} |",
                f"| **Distância Notícia** | {news_str} |",
                f"| **Confluências** | {conf_str} |",
                f"| **Versão Motor** | {snap.engine_version} |",
                "",
                "---",
                "",
                "## 3. Diagrama Visual dos Níveis de Preço",
                "",
                svg,
                "",
            ]
        )

        if narrative:
            status_desc = "Fallback Operacional" if narrative.is_fallback else "Análise Concluída"
            lines.extend(
                [
                    "---",
                    "",
                    "## 4. Parecer Qualitativo do Narrador LLM",
                    "",
                    f"- **Modelo**: `{narrative.model}` ({narrative.elapsed_seconds:.2f}s)",
                    f"- **Status**: {status_desc}",
                    "",
                    "### Resumo Executivo",
                    narrative.summary,
                    "",
                ]
            )
            if narrative.key_insights:
                lines.extend(
                    [
                        "### Insights Chave",
                        *(f"- {insight}" for insight in narrative.key_insights),
                        "",
                    ]
                )
            if narrative.suggested_action:
                lines.extend(
                    [
                        "### Ação Sugerida",
                        narrative.suggested_action,
                        "",
                    ]
                )

        return "\n".join(lines)

    @classmethod
    def export_report_file(
        cls,
        entry: JournalEntry,
        output_dir: Path | str,
        narrative: NarrativeReport | None = None,
    ) -> Path:
        """Saves markdown report to file and returns created Path."""
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = f"{entry.created_at.strftime('%Y%m%d_%H%M%S')}_{entry.proposal_id}.md"
        report_path = dir_path / filename
        content = cls.generate_markdown_report(entry, narrative=narrative)
        report_path.write_text(content, encoding="utf-8")
        return report_path
