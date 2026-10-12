"""Markdown report generator for the Learning and Periodic Review module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from xau_bot.learning.models import HumanDiscretionReport, LearningReviewReport

if TYPE_CHECKING:
    from xau_bot.llm.models import NarrativeReport


class LearningReporter:
    """Generates structured Markdown reports for periodic learning reviews."""

    @staticmethod
    def generate_markdown_report(
        report: LearningReviewReport,
        narrative: NarrativeReport | None = None,
    ) -> str:
        lines: list[str] = [
            "# Relatório de Aprendizado e Revisão Periódica (Learning Review)",
            "",
            "## 1. Resumo Executivo da População de Trades",
            "",
            "| Métrica | Valor |",
            "| :--- | :--- |",
            f"| Data da Revisão (UTC) | {report.reviewed_at.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Total de Trades Analisados | {report.total_trades_analyzed} |",
            f"| Vitórias (WIN) | {report.total_wins} ({report.win_rate * 100:.1f}%) |",
            f"| Derrotas (LOSS) | {report.total_losses} ({report.loss_rate * 100:.1f}%) |",
            f"| Empates (BREAKEVEN) | {report.total_breakevens} |",
            f"| Expectativa Matemática Global em R | {report.global_expectancy_r:+.2f}R |",
            f"| Limiar Mínimo Amostral (Gating) | N >= {report.min_samples} |",
            f"| Nível de Significância | alpha = {report.alpha:.2f} (p < {report.alpha:.2f}) |",
            "",
            "## 2. Taxonomia e Ranking de Causas Raiz de Perda",
            "",
        ]

        if report.loss_cause_distribution:
            lines.extend(
                [
                    "| Causa Raiz de Perda | Ocorrências | % do Total de Perdas |",
                    "| :--- | :---: | :---: |",
                ]
            )
            total_losses = report.total_losses if report.total_losses > 0 else 1
            for cause, count in report.loss_cause_distribution.items():
                pct = (count / total_losses) * 100
                lines.append(f"| `{cause}` | {count} | {pct:.1f}% |")
            lines.append("")
        else:
            lines.extend(
                [
                    "_Nenhuma perda registrada ou detalhada na amostra selecionada._",
                    "",
                ]
            )

        lines.extend(
            [
                "## 3. Hipóteses Estatisticamente Significativas para Ajuste de Parâmetros",
                "",
            ]
        )
        if report.hypotheses:
            lines.extend(
                [
                    "| ID | Parâmetro | Atual | Proposto | N | Loss Rate | p-val | R | Racional |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
                ]
            )
            for h in report.hypotheses:
                lr_str = f"{h.condition_loss_rate * 100:.1f}% vs {h.baseline_loss_rate * 100:.1f}%"
                lines.append(
                    f"| `{h.id}` | `{h.parameter_name}` | `{h.current_value}` | "
                    f"`{h.proposed_value}` | {h.sample_size} | {lr_str} | {h.p_value:.4f} | "
                    f"{h.observed_expectancy_r:+.2f}R | {h.rationale} |"
                )
            lines.append("")
        else:
            lines.extend(
                [
                    "_Nenhuma hipótese formal proposta. Nenhuma condição adversa atingiu "
                    f"significância estatística (N >= {report.min_samples}, "
                    f"p < {report.alpha:.2f})._",
                    "",
                ]
            )

        lines.extend(
            [
                f"## 4. Observações em Acumulação Amostral (Aguardando N >= {report.min_samples})",
                "",
            ]
        )
        if report.accumulating_observations:
            lines.extend(
                [
                    "| Feature | Condição | N | Loss Rate | p-value | R Médio | Status |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
                ]
            )
            for obs in report.accumulating_observations:
                lines.append(
                    f"| `{obs.feature_name}` | `{obs.condition_key}` | {obs.sample_size} | "
                    f"{obs.loss_rate * 100:.1f}% | {obs.p_value:.4f} | "
                    f"{obs.mean_realized_r:+.2f}R | ACCUMULATING |"
                )
            lines.append("")
        else:
            lines.extend(
                [
                    "_Nenhuma condição em acumulação amostral detectada com desvio relevante._",
                    "",
                ]
            )

        lines.extend(
            [
                "## 5. Breakdown Estatístico de Features Técnicas",
                "",
            ]
        )
        if report.feature_stats:
            lines.extend(
                [
                    "| Feature | Condição | N | W | L | Loss Rate | p-val | R Médio | Sig? |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for fs in report.feature_stats:
                sig_str = "**SIM**" if fs.is_statistically_significant else "Não"
                lines.append(
                    f"| `{fs.feature_name}` | `{fs.condition_key}` | {fs.sample_size} | "
                    f"{fs.win_count} | {fs.loss_count} | {fs.loss_rate * 100:.1f}% | "
                    f"{fs.p_value:.4f} | {fs.mean_realized_r:+.2f}R | {sig_str} |"
                )
            lines.append("")
        else:
            lines.extend(
                [
                    "_Nenhuma feature técnica disponível para avaliação._",
                    "",
                ]
            )

        if narrative:
            status_desc = "Fallback Operacional" if narrative.is_fallback else "Análise Concluída"
            lines.extend(
                [
                    "## 6. Parecer Qualitativo do Narrador LLM",
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

    @staticmethod
    def save_report(
        report: LearningReviewReport,
        output_dir: str | Path,
        narrative: NarrativeReport | None = None,
    ) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts_str = report.reviewed_at.strftime("%Y%m%d_%H%M%S")
        file_path = out_path / f"learning_review_{ts_str}.md"
        content = LearningReporter.generate_markdown_report(report, narrative=narrative)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    @staticmethod
    def generate_human_discretion_markdown(
        report: HumanDiscretionReport,
        narrative: NarrativeReport | None = None,
    ) -> str:
        lines: list[str] = [
            "# Relatório de Discricionariedade Humana (Human Discretion & Edge Analysis)",
            "",
            "## 1. Resumo Executivo e Veredito",
            "",
            "| Parâmetro | Valor |",
            "| :--- | :--- |",
            f"| Data da Análise (UTC) | {report.analyzed_at.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Veredito Global | **`{report.verdict.value}`** |",
            f"| Parecer | {report.verdict_summary} |",
            f"| Total de Propostas | {report.total_proposals} |",
            f"| Taxa de Aceitação Humana | {report.acceptance_rate * 100:.1f}% |",
            f"| Limiar Mínimo Amostral (Gating) | N >= {report.min_samples} |",
            f"| Nível de Significância (alpha) | {report.alpha:.2f} |",
            "",
            "## 2. Comparativo de Coortes (Aprovadas vs Ignoradas vs Total)",
            "",
            (
                "| Coorte | Propostas | Avaliados | Vitórias | Derrotas | Win Rate | "
                "Expectativa (R) | Desvio (R) | Profit Factor |"
            ),
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        cohorts = [
            report.approved_metrics,
            report.ignored_metrics,
            report.combined_metrics,
        ]
        for c in cohorts:
            pf_str = f"{c.profit_factor:.2f}" if c.profit_factor < 999.0 else "Inf"
            lines.append(
                f"| `{c.cohort_name}` | {c.total_proposals} | {c.evaluated_trades} | "
                f"{c.win_count} | {c.loss_count} | {c.win_rate * 100:.1f}% | "
                f"{c.expectancy_r:+.2f}R | {c.std_dev_r:.2f}R | {pf_str} |"
            )
        lines.append("")

        p_exp_str = (
            f"{report.p_value_expectancy:.4f}" if report.p_value_expectancy is not None else "N/A"
        )
        p_wr_str = (
            f"{report.p_value_win_rate:.4f}" if report.p_value_win_rate is not None else "N/A"
        )

        sig_exp = (
            "**SIM**"
            if report.p_value_expectancy is not None and report.p_value_expectancy < report.alpha
            else "Não"
        )
        sig_wr = (
            "**SIM**"
            if report.p_value_win_rate is not None and report.p_value_win_rate < report.alpha
            else "Não"
        )

        lines.extend(
            [
                "## 3. Mensuração de Human Edge e Discricionariedade",
                "",
                (
                    "| Métrica Comparativa | Diferença (Aprovadas - Ignoradas) | p-valor | "
                    "Significante? |"
                ),
                "| :--- | :---: | :---: | :---: |",
                (
                    f"| Delta Expectativa (Delta R) | {report.delta_expectancy_r:+.2f}R | "
                    f"{p_exp_str} | {sig_exp} |"
                ),
                (
                    f"| Delta Taxa de Acerto (Delta Win Rate) | "
                    f"{report.delta_win_rate * 100:+.1f}% | {p_wr_str} | {sig_wr} |"
                ),
                "",
                "## 4. Segmentação Multifatorial (Breakdown por Dimensão)",
                "",
            ]
        )

        if report.segments:
            lines.extend(
                [
                    (
                        "| Dimensão | Segmento | Total | Aprovados (%) | R Aprovado | "
                        "R Ignorado | Delta R | WR Aprovado | WR Ignorado | Delta WR |"
                    ),
                    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for s in report.segments:
                seg_line = (
                    f"| `{s.dimension}` | `{s.segment_value}` | {s.total_count} | "
                    f"{s.approval_rate * 100:.1f}% | {s.approved_expectancy_r:+.2f}R | "
                    f"{s.ignored_expectancy_r:+.2f}R | {s.delta_expectancy_r:+.2f}R | "
                    f"{s.approved_win_rate * 100:.1f}% | {s.ignored_win_rate * 100:.1f}% | "
                    f"{s.delta_win_rate * 100:+.1f}% |"
                )
                lines.append(seg_line)
            lines.append("")
        else:
            lines.extend(
                [
                    "_Nenhum segmento com amostra suficiente para exibição._",
                    "",
                ]
            )

        if narrative:
            status_desc = "Fallback Operacional" if narrative.is_fallback else "Análise Concluída"
            lines.extend(
                [
                    "## 5. Parecer Qualitativo do Narrador LLM",
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

    @staticmethod
    def save_human_report(
        report: HumanDiscretionReport,
        output_dir: str | Path,
        narrative: NarrativeReport | None = None,
    ) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts_str = report.analyzed_at.strftime("%Y%m%d_%H%M%S")
        file_path = out_path / f"human_discretion_{ts_str}.md"
        content = LearningReporter.generate_human_discretion_markdown(report, narrative=narrative)
        file_path.write_text(content, encoding="utf-8")
        return file_path
