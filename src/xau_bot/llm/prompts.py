"""Prompt templates and formatting utilities for the LLM Narrator."""

from __future__ import annotations

from xau_bot.journal.models import JournalEntry, TradeOutcome
from xau_bot.learning.models import HumanDiscretionReport, LearningReviewReport

SYSTEM_PROMPT_NARRATOR = (
    "Você é um analista quantitativo e de risco sênior especializado em trading institucional "
    "de XAU/USD baseado em Smart Money Concepts (SMC).\n"
    "Sua função é fornecer análises qualitativas e sínteses executivas em linguagem clara e "
    "técnica sobre post-mortems de trades, revisões estatísticas periódicas e impacto da "
    "discricionariedade humana.\n\n"
    "REGRAS INVIOLÁVEIS:\n"
    "1. DO NOT invent numbers or statistics. NÃO CRIE OU INVENTE NENHUM NÚMERO OU ESTATÍSTICA "
    "(zero alucinações quantitativas). Todas as métricas, contagens, taxas de acerto e valores de "
    "R citados devem provir estritamente dos dados fornecidos no contexto.\n"
    "2. Seja direto, conciso e evite chavões genéricos. Concentre-se nas confluências técnicas, "
    "fatores de risco e comportamento de mercado documentados.\n"
    "3. Estruture sua resposta com exatamente 3 seções nomeadas:\n"
    "   RESUMO EXECUTIVO:\n"
    "   (2 a 4 frases explicando a síntese do evento ou período)\n"
    "   INSIGHTS CHAVE:\n"
    "   - (Item 1)\n"
    "   - (Item 2)\n"
    "   - (Item 3)\n"
    "   AÇÃO SUGERIDA:\n"
    "   (1 recomendação prática e acionável para o operador ou time)"
)


def format_trade_post_mortem_prompt(entry: JournalEntry) -> str:
    """Formats deterministic context for a single trade post-mortem analysis."""
    snap = entry.snapshot
    out = entry.outcome_data

    lines = [
        f"Analise o trade XAU/USD da proposta {entry.proposal_id}:",
        f"- Data de Criação: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Direção: {entry.direction.value}",
        f"- Status da Proposta: {entry.status.value}",
        f"- Desfecho: {entry.outcome.value}",
    ]

    if out:
        exec_desc = (
            "Executado em mercado" if out.executed else "Não executado (Contrafactual/Simulado)"
        )
        lines.extend(
            [
                f"- Tipo de Execução: {exec_desc}",
                f"- Preço Entrada: {out.entry_price:.2f}",
                f"- Stop Loss: {out.stop_loss:.2f}",
                f"- Take Profit: {out.take_profit:.2f}",
                f"- Preço Saída: {out.exit_price:.2f}" if out.exit_price else "- Preço Saída: N/A",
                f"- Resultado em R: {out.realized_r:+.2f}R",
                f"- PnL USD: ${out.pnl_usd:+.2f}",
                f"- MFE (Máximo Fator Favorável): +{out.mfe_r:.2f}R",
                f"- MAE (Máximo Fator Adverso): -{out.mae_r:.2f}R",
                f"- Duração: {out.duration_seconds:.1f}s",
                f"- Motivo de Saída: {out.exit_reason or 'N/A'}",
            ]
        )
        if out.post_exit_behavior:
            lines.append(f"- Comportamento Pós-Saída: {out.post_exit_behavior}")

    if entry.outcome == TradeOutcome.LOSS:
        cat_val = entry.loss_category.value if entry.loss_category else "N/A"
        cause_val = entry.loss_cause.value if entry.loss_cause else "N/A"
        lines.extend(
            [
                f"- Categoria da Perda: {cat_val}",
                f"- Causa Raiz Diagnosticada: {cause_val}",
            ]
        )

    confluences_str = (
        ", ".join(snap.confluences) if snap.confluences else "Nenhuma confluência extra"
    )
    news_str = (
        f"{snap.news_distance_min:.1f} min"
        if snap.news_distance_min is not None
        else "Sem notícias"
    )

    lines.extend(
        [
            "- Contexto Técnico do Snapshot:",
            f"  * Sessão: {snap.session}",
            f"  * Viés HTF: {snap.htf_bias}",
            f"  * Tipo POI: {snap.poi_type} @ {snap.poi_level:.2f}",
            f"  * Deslocamento ATR: {snap.displacement_atr:.2f}x",
            f"  * Tamanho FVG ATR: {snap.fvg_size_atr:.2f}x",
            f"  * Nível ChoCH: {snap.choch_level:.2f}",
            f"  * Spread: ${snap.spread_price:.2f}",
            f"  * ATR M5: ${snap.atr_m5:.2f}",
            f"  * Distância de Notícia: {news_str}",
            f"  * Confluências: {confluences_str}",
            f"  * Decisão do Usuário: {snap.user_decision}",
        ]
    )

    lines.extend(
        [
            "",
            "IMPORTANTE: NÃO invente números, estatísticas ou taxas de acerto não fornecidas "
            "acima. Sua análise deve se basear estritamente nos dados listados.",
        ]
    )
    return "\n".join(lines)


def format_learning_review_prompt(report: LearningReviewReport) -> str:
    """Formats deterministic context for periodic learning review and hypotheses synthesis."""
    lines = [
        "Analise o relatório de revisão periódica de aprendizado do robô XAU/USD:",
        f"- Data da Revisão: {report.reviewed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Total de Trades Analisados: {report.total_trades_analyzed}",
        f"- Total de Vitórias: {report.total_wins} ({report.win_rate * 100:.1f}%)",
        f"- Total de Perdas: {report.total_losses} ({report.loss_rate * 100:.1f}%)",
        f"- Total de Empates: {report.total_breakevens}",
        f"- Expectativa Global em R: {report.global_expectancy_r:+.2f}R",
        f"- Limiar Mínimo Amostral (Gating): N >= {report.min_samples}",
        f"- Nível de Significância (Alpha): {report.alpha:.2f}",
        "",
        "Distribuição de Causas Raiz de Perda:",
    ]

    if report.loss_cause_distribution:
        total_losses = report.total_losses if report.total_losses > 0 else 1
        for cause_name, count in report.loss_cause_distribution.items():
            pct = (count / total_losses) * 100
            lines.append(f"- Causa `{cause_name}`: {count} ocorrências ({pct:.1f}% das perdas)")
    else:
        lines.append("- Nenhuma causa de perda registrada.")

    lines.extend(["", "Hipóteses Estatísticas Propostas:"])
    if report.hypotheses:
        for hyp in report.hypotheses:
            lines.append(
                f"- Hipótese `{hyp.id}` ({hyp.parameter_name}): "
                f"valor atual={hyp.current_value}, valor proposto={hyp.proposed_value}, "
                f"N={hyp.sample_size}, taxa de perda={hyp.condition_loss_rate * 100:.1f}%, "
                f"R médio={hyp.observed_expectancy_r:+.2f}R, p-valor={hyp.p_value:.4f}, "
                f"racional={hyp.rationale}"
            )
    else:
        lines.append("- Nenhuma hipótese formal proposta no período.")

    lines.extend(["", "Observações em Acumulação (sem significância ou amostra insuficiente):"])
    if report.accumulating_observations:
        for obs in report.accumulating_observations:
            lines.append(
                f"- `{obs.feature_name}` ({obs.condition_key}): N={obs.sample_size}, "
                f"taxa de perda={obs.loss_rate * 100:.1f}%, p-valor={obs.p_value:.4f}, "
                f"R médio={obs.mean_realized_r:+.2f}R"
            )
    else:
        lines.append("- Nenhuma observação em acumulação pendente.")

    lines.extend(
        [
            "",
            "IMPORTANTE: NÃO invente números. Analise estritamente as hipóteses e dados acima.",
        ]
    )
    return "\n".join(lines)


def format_human_discretion_prompt(report: HumanDiscretionReport) -> str:
    """Formats deterministic context for human discretion and edge analysis."""
    p_exp_str = (
        f"{report.p_value_expectancy:.4f}" if report.p_value_expectancy is not None else "N/A"
    )
    p_wr_str = f"{report.p_value_win_rate:.4f}" if report.p_value_win_rate is not None else "N/A"

    app = report.approved_metrics
    ign = report.ignored_metrics

    lines = [
        "Analise o relatório de discricionariedade humana (propostas aprovadas vs ignoradas):",
        f"- Data da Análise: {report.analyzed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Veredito Global: {report.verdict.value}",
        f"- Parecer Técnico: {report.verdict_summary}",
        f"- Total de Propostas: {report.total_proposals}",
        f"- Taxa de Aceitação Humana: {report.acceptance_rate * 100:.1f}%",
        f"- Delta Expectativa (Delta R): {report.delta_expectancy_r:+.2f}R (p-valor: {p_exp_str})",
        f"- Delta Win Rate: {report.delta_win_rate * 100:+.1f}% (p-valor: {p_wr_str})",
        "",
        "Comparativo de Coortes:",
        (
            f"- Aprovadas: N={app.evaluated_trades}, Vitórias={app.win_count}, "
            f"Derrotas={app.loss_count}, Win Rate={app.win_rate * 100:.1f}%, "
            f"R Médio={app.expectancy_r:+.2f}R, Desvio={app.std_dev_r:.2f}R"
        ),
        (
            f"- Ignoradas: N={ign.evaluated_trades}, Vitórias={ign.win_count}, "
            f"Derrotas={ign.loss_count}, Win Rate={ign.win_rate * 100:.1f}%, "
            f"R Médio={ign.expectancy_r:+.2f}R, Desvio={ign.std_dev_r:.2f}R"
        ),
    ]

    if report.segments:
        lines.extend(["", "Segmentação por Dimensões Técnicas:"])
        for seg in report.segments:
            lines.append(
                f"- Dimensão `{seg.dimension}` = `{seg.segment_value}`: "
                f"Total={seg.total_count}, Concordância={seg.approval_rate * 100:.1f}%, "
                f"R Aprovado={seg.approved_expectancy_r:+.2f}R, "
                f"R Ignorado={seg.ignored_expectancy_r:+.2f}R, "
                f"Delta R={seg.delta_expectancy_r:+.2f}R"
            )

    lines.extend(
        [
            "",
            (
                "IMPORTANTE: NÃO invente números. "
                "Interprete estritamente o veredito e o comparativo acima."
            ),
        ]
    )
    return "\n".join(lines)
