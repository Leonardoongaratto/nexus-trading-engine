"""LLM Narrator engine for synthesizing qualitative insights from quantitative reports."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

from xau_bot.config import Settings
from xau_bot.journal.models import JournalEntry, TradeOutcome
from xau_bot.learning.models import HumanDiscretionReport, LearningReviewReport
from xau_bot.llm.client import HttpLLMClient, LLMClientProtocol
from xau_bot.llm.models import NarrativeReport, NarratorMode
from xau_bot.llm.prompts import (
    SYSTEM_PROMPT_NARRATOR,
    format_human_discretion_prompt,
    format_learning_review_prompt,
    format_trade_post_mortem_prompt,
)

logger = logging.getLogger(__name__)


class LLMNarrator:
    """Orchestrates qualitative trade post-mortems and learning syntheses via LLM."""

    def __init__(
        self,
        client: LLMClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client
        if self.client is None and self.settings.llm_enabled and self.settings.llm_api_key:
            self.client = HttpLLMClient(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )

    @property
    def is_available(self) -> bool:
        """Indicates whether an operational LLM client is configured."""
        return self.client is not None and self.settings.llm_enabled

    def narrate_trade_post_mortem(self, entry: JournalEntry) -> NarrativeReport:
        """Generates a post-mortem narrative for a given journal entry."""
        mode = NarratorMode.POST_MORTEM
        if not self.is_available:
            return self._build_deterministic_fallback_post_mortem(entry)

        user_prompt = format_trade_post_mortem_prompt(entry)
        return self._execute_narrative(
            mode=mode,
            system_prompt=SYSTEM_PROMPT_NARRATOR,
            user_prompt=user_prompt,
            fallback_builder=lambda err: self._build_deterministic_fallback_post_mortem(
                entry, error_message=err
            ),
        )

    def narrate_learning_review(self, report: LearningReviewReport) -> NarrativeReport:
        """Generates an executive narrative for a periodic learning review."""
        mode = NarratorMode.LEARNING_REVIEW
        if not self.is_available:
            return self._build_deterministic_fallback_learning_review(report)

        user_prompt = format_learning_review_prompt(report)
        return self._execute_narrative(
            mode=mode,
            system_prompt=SYSTEM_PROMPT_NARRATOR,
            user_prompt=user_prompt,
            fallback_builder=lambda err: self._build_deterministic_fallback_learning_review(
                report, error_message=err
            ),
        )

    def narrate_human_discretion(self, report: HumanDiscretionReport) -> NarrativeReport:
        """Generates an executive narrative for human discretion and edge analysis."""
        mode = NarratorMode.HUMAN_DISCRETION
        if not self.is_available:
            return self._build_deterministic_fallback_human_discretion(report)

        user_prompt = format_human_discretion_prompt(report)
        return self._execute_narrative(
            mode=mode,
            system_prompt=SYSTEM_PROMPT_NARRATOR,
            user_prompt=user_prompt,
            fallback_builder=lambda err: self._build_deterministic_fallback_human_discretion(
                report, error_message=err
            ),
        )

    def _execute_narrative(
        self,
        mode: NarratorMode,
        system_prompt: str,
        user_prompt: str,
        fallback_builder: Callable[[str | None], NarrativeReport],
    ) -> NarrativeReport:
        if self.client is None:
            return fallback_builder(None)
        start_time = time.monotonic()
        try:
            raw_text = self.client.complete(system_prompt, user_prompt)
            elapsed = time.monotonic() - start_time
            summary, insights, action = self._parse_completion(raw_text)
            return NarrativeReport(
                summary=summary,
                key_insights=insights,
                suggested_action=action,
                model=self.settings.llm_model,
                elapsed_seconds=elapsed,
                mode=mode,
                is_fallback=False,
                error_message=None,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            logger.warning("Falha na chamada ou processamento do LLM Narrator: %s", e)
            report = fallback_builder(str(e))
            return NarrativeReport(
                summary=report.summary,
                key_insights=report.key_insights,
                suggested_action=report.suggested_action,
                model="deterministic-fallback",
                elapsed_seconds=elapsed,
                mode=mode,
                is_fallback=True,
                error_message=str(e),
            )

    @staticmethod
    def _parse_completion(text: str) -> tuple[str, tuple[str, ...], str | None]:
        summary_match = re.search(
            r"RESUMO EXECUTIVO:\s*\n?(.*?)(?=(INSIGHTS CHAVE:|AÇÃO SUGERIDA:|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        insights_match = re.search(
            r"INSIGHTS CHAVE:\s*\n?(.*?)(?=(AÇÃO SUGERIDA:|\Z))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        action_match = re.search(
            r"AÇÃO SUGERIDA:\s*\n?(.*?)\Z",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        summary = summary_match.group(1).strip() if summary_match else text.strip()

        insights: list[str] = []
        if insights_match:
            raw_bullets = insights_match.group(1).strip().splitlines()
            for line in raw_bullets:
                clean = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
                if clean:
                    insights.append(clean)

        action = action_match.group(1).strip() if action_match else None
        if action == "":
            action = None

        return summary, tuple(insights), action

    @staticmethod
    def _build_deterministic_fallback_post_mortem(
        entry: JournalEntry,
        error_message: str | None = None,
    ) -> NarrativeReport:
        r_val = f"{entry.outcome_data.realized_r:+.2f}R" if entry.outcome_data else "N/A"
        cause = (
            f" com causa identificada como `{entry.loss_cause.value}`"
            if entry.outcome == TradeOutcome.LOSS and entry.loss_cause
            else ""
        )
        summary = (
            f"Operação {entry.direction.value} em {entry.snapshot.session} finalizada com "
            f"desfecho {entry.outcome.value} ({r_val}){cause}. "
            "Parecer qualitativo LLM não executado."
        )
        insights = (
            f"Sessão: {entry.snapshot.session} | "
            f"POI: {entry.snapshot.poi_type} @ {entry.snapshot.poi_level:.2f}",
            f"Deslocamento: {entry.snapshot.displacement_atr:.2f}x ATR | "
            f"FVG: {entry.snapshot.fvg_size_atr:.2f}x ATR",
        )
        return NarrativeReport(
            summary=summary,
            key_insights=insights,
            suggested_action=(
                "Verificar conformidade operacional no journal e histórico de backtest."
            ),
            model="deterministic-fallback",
            elapsed_seconds=0.0,
            mode=NarratorMode.POST_MORTEM,
            is_fallback=True,
            error_message=error_message,
        )

    @staticmethod
    def _build_deterministic_fallback_learning_review(
        report: LearningReviewReport,
        error_message: str | None = None,
    ) -> NarrativeReport:
        summary = (
            f"Revisão periódica de aprendizado concluída com {report.total_trades_analyzed} "
            f"trades avaliados ({report.total_losses} perdas, {report.loss_rate * 100:.1f}% "
            f"de perda global) e {len(report.hypotheses)} hipóteses estatísticas propostas."
        )
        insights = (
            f"Limiar amostral N >= {report.min_samples} e "
            f"nível de significância alpha={report.alpha:.2f}.",
            f"Observações em acumulação: {len(report.accumulating_observations)} "
            "condições pendentes de dados.",
        )
        action = (
            f"Executar validação em backtest/sombra para a hipótese `{report.hypotheses[0].id}`."
            if report.hypotheses
            else "Continuar acumulando observações no diário para significância estatística."
        )
        return NarrativeReport(
            summary=summary,
            key_insights=insights,
            suggested_action=action,
            model="deterministic-fallback",
            elapsed_seconds=0.0,
            mode=NarratorMode.LEARNING_REVIEW,
            is_fallback=True,
            error_message=error_message,
        )

    @staticmethod
    def _build_deterministic_fallback_human_discretion(
        report: HumanDiscretionReport,
        error_message: str | None = None,
    ) -> NarrativeReport:
        summary = (
            f"Análise de discricionariedade humana resultou no veredito {report.verdict.value} "
            f"(Delta R = {report.delta_expectancy_r:+.2f}R, "
            f"Delta Win Rate = {report.delta_win_rate * 100:+.1f}%). {report.verdict_summary}"
        )
        insights = (
            f"Taxa de aceitação humana: {report.acceptance_rate * 100:.1f}% em "
            f"{report.total_proposals} propostas.",
            f"Aprovadas: WR={report.approved_metrics.win_rate * 100:.1f}%, "
            f"R={report.approved_metrics.expectancy_r:+.2f}R | "
            f"Ignoradas: WR={report.ignored_metrics.win_rate * 100:.1f}%, "
            f"R={report.ignored_metrics.expectancy_r:+.2f}R.",
        )
        action = (
            "Manter filtros discricionários ativos e monitorar estabilidade do edge."
            if report.verdict.value == "POSITIVE_EDGE"
            else "Revisar critérios discricionários para evitar inserção de viés prejudicial."
        )
        return NarrativeReport(
            summary=summary,
            key_insights=insights,
            suggested_action=action,
            model="deterministic-fallback",
            elapsed_seconds=0.0,
            mode=NarratorMode.HUMAN_DISCRETION,
            is_fallback=True,
            error_message=error_message,
        )
