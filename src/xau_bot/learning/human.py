"""Human discretion analysis engine comparing trade proposals (T-A8)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.journal.models import JournalEntry, TradeOutcome
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.models import (
    CohortMetrics,
    HumanDiscretionReport,
    HumanDiscretionVerdict,
    HumanSegmentStats,
)
from xau_bot.learning.stats import calculate_expectancy, two_proportion_z_test, welch_t_test


class HumanDiscretionAnalyzer:
    """Analyzes operator discretion and performance divergence in semiautomatic mode."""

    def __init__(
        self,
        repository: JournalRepository,
        min_samples: int = 30,
        alpha: float = 0.05,
    ) -> None:
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        self.repository = repository
        self.min_samples = min_samples
        self.alpha = alpha

    def analyze(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> HumanDiscretionReport:
        """Executes discretion analysis across trade journal entries within a time window."""
        entries = self.repository.list_entries(
            start_time=start_time,
            end_time=end_time,
            limit=None,
        )
        return self.from_entries(entries)

    @staticmethod
    def _is_eligible_factual(entry: JournalEntry) -> bool:
        return entry.outcome_data is not None and entry.outcome in (
            TradeOutcome.WIN,
            TradeOutcome.LOSS,
            TradeOutcome.BREAKEVEN,
        )

    def from_entries(self, entries: Sequence[JournalEntry]) -> HumanDiscretionReport:
        """Processes a sequence of journal entries into a HumanDiscretionReport."""
        now = datetime.now(UTC)
        raw_total = len(entries)
        eligible_entries = [e for e in entries if self._is_eligible_factual(e)]

        approved_entries: list[JournalEntry] = []
        ignored_entries: list[JournalEntry] = []

        for entry in eligible_entries:
            if self._is_approved(entry):
                approved_entries.append(entry)
            elif self._is_ignored(entry):
                ignored_entries.append(entry)

        approved_metrics = self._calculate_cohort_metrics(
            cohort_name="APPROVED",
            entries=approved_entries,
        )
        ignored_metrics = self._calculate_cohort_metrics(
            cohort_name="IGNORED_OR_REJECTED",
            entries=ignored_entries,
        )
        combined_metrics = self._calculate_cohort_metrics(
            cohort_name="COMBINED",
            entries=approved_entries + ignored_entries,
        )

        total_proposals = len(approved_entries) + len(ignored_entries)
        acceptance_rate = len(approved_entries) / total_proposals if total_proposals > 0 else 0.0

        n_app = approved_metrics.evaluated_trades
        n_ign = ignored_metrics.evaluated_trades

        delta_expectancy_r = (
            approved_metrics.expectancy_r - ignored_metrics.expectancy_r
            if n_app > 0 and n_ign > 0
            else 0.0
        )
        delta_win_rate = (
            approved_metrics.win_rate - ignored_metrics.win_rate if n_app > 0 and n_ign > 0 else 0.0
        )

        p_val_exp: float | None = None
        if n_app >= 2 and n_ign >= 2:
            _, p_val_exp = welch_t_test(
                approved_metrics.realized_r_values,
                ignored_metrics.realized_r_values,
            )

        p_val_wr: float | None = None
        if n_app >= 2 and n_ign >= 2:
            _, p_val_wr = two_proportion_z_test(
                count1=approved_metrics.win_count,
                n1=n_app,
                count2=ignored_metrics.win_count,
                n2=n_ign,
            )

        total_evaluated = n_app + n_ign

        verdict, summary = self._determine_verdict(
            approved_count=n_app,
            ignored_count=n_ign,
            total_evaluated=total_evaluated,
            delta_expectancy_r=delta_expectancy_r,
            p_value_expectancy=p_val_exp,
        )

        segments = self._build_segment_stats(
            approved_entries=approved_entries,
            ignored_entries=ignored_entries,
        )

        return HumanDiscretionReport(
            analyzed_at=now,
            min_samples=self.min_samples,
            alpha=self.alpha,
            total_proposals=total_proposals,
            approved_metrics=approved_metrics,
            ignored_metrics=ignored_metrics,
            combined_metrics=combined_metrics,
            acceptance_rate=acceptance_rate,
            delta_expectancy_r=delta_expectancy_r,
            delta_win_rate=delta_win_rate,
            p_value_expectancy=p_val_exp,
            p_value_win_rate=p_val_wr,
            verdict=verdict,
            verdict_summary=summary,
            segments=tuple(segments),
            raw_proposals=raw_total,
        )

    def _determine_verdict(
        self,
        approved_count: int,
        ignored_count: int,
        total_evaluated: int,
        delta_expectancy_r: float,
        p_value_expectancy: float | None,
    ) -> tuple[HumanDiscretionVerdict, str]:
        if approved_count < 2 or ignored_count < 2:
            verdict = HumanDiscretionVerdict.INSUFFICIENT_DATA
            summary = (
                f"Amostra insuficiente em uma das coortes "
                f"(Aprovadas: {approved_count}, Ignoradas: {ignored_count}) "
                "para inferência estatística comparativa."
            )
            return verdict, summary

        if total_evaluated < self.min_samples:
            verdict = HumanDiscretionVerdict.INSUFFICIENT_DATA
            summary = (
                f"Amostra insuficiente ({total_evaluated} < {self.min_samples} trades avaliados) "
                "para inferência estatística conclusiva sobre a discricionariedade humana."
            )
            return verdict, summary

        if p_value_expectancy is not None and p_value_expectancy < self.alpha:
            if delta_expectancy_r > 0.0:
                verdict = HumanDiscretionVerdict.POSITIVE_EDGE
                summary = (
                    "Discricionariedade humana adiciona valor significativo "
                    f"(Delta R = +{delta_expectancy_r:.2f}R, "
                    f"p = {p_value_expectancy:.4f} < {self.alpha:.2f})."
                )
            else:
                verdict = HumanDiscretionVerdict.NEGATIVE_BIAS
                summary = (
                    "Discricionariedade humana insere viés prejudicial sistemático "
                    f"(Delta R = {delta_expectancy_r:.2f}R, "
                    f"p = {p_value_expectancy:.4f} < {self.alpha:.2f})."
                )
        else:
            verdict = HumanDiscretionVerdict.NEUTRAL
            p_txt = f"p = {p_value_expectancy:.4f}" if p_value_expectancy is not None else "p = N/A"
            summary = (
                "Diferença sem significância estatística entre decisões humanas e descartadas "
                f"(Delta R = {delta_expectancy_r:+.2f}R, {p_txt} >= {self.alpha:.2f})."
            )

        return verdict, summary

    def _calculate_cohort_metrics(
        self,
        cohort_name: str,
        entries: Sequence[JournalEntry],
    ) -> CohortMetrics:
        total_proposals = len(entries)
        n_eval = total_proposals

        wins = sum(1 for e in entries if e.outcome == TradeOutcome.WIN)
        losses = sum(1 for e in entries if e.outcome == TradeOutcome.LOSS)
        breakevens = sum(1 for e in entries if e.outcome == TradeOutcome.BREAKEVEN)

        win_rate = wins / n_eval if n_eval > 0 else 0.0
        loss_rate = losses / n_eval if n_eval > 0 else 0.0

        r_values: list[float] = [
            float(e.outcome_data.realized_r) for e in entries if e.outcome_data is not None
        ]

        tot_r = sum(r_values)
        exp_r = calculate_expectancy(r_values)

        std_dev = 0.0
        if n_eval >= 2:
            variance = sum((x - exp_r) ** 2 for x in r_values) / (n_eval - 1)
            std_dev = math.sqrt(variance)

        gross_profit = sum(x for x in r_values if x > 0.0)
        gross_loss = abs(sum(x for x in r_values if x < 0.0))
        if gross_loss > 0.0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0.0:
            pf = 999.0
        else:
            pf = 0.0

        return CohortMetrics(
            cohort_name=cohort_name,
            total_proposals=total_proposals,
            evaluated_trades=n_eval,
            win_count=wins,
            loss_count=losses,
            breakeven_count=breakevens,
            win_rate=win_rate,
            loss_rate=loss_rate,
            expectancy_r=exp_r,
            std_dev_r=std_dev,
            total_realized_r=tot_r,
            profit_factor=pf,
            realized_r_values=tuple(r_values),
        )

    def _build_segment_stats(
        self,
        approved_entries: Sequence[JournalEntry],
        ignored_entries: Sequence[JournalEntry],
    ) -> list[HumanSegmentStats]:
        def _get_direction(e: JournalEntry) -> str:
            return str(e.direction.value if hasattr(e.direction, "value") else e.direction)

        def _get_session(e: JournalEntry) -> str:
            return str(e.snapshot.session or "Unknown")

        def _get_poi(e: JournalEntry) -> str:
            return str(e.snapshot.poi_type or "Unknown")

        dimensions: list[tuple[str, Callable[[JournalEntry], str]]] = [
            ("direction", _get_direction),
            ("session", _get_session),
            ("poi_type", _get_poi),
        ]

        results: list[HumanSegmentStats] = []

        for dim_name, extractor in dimensions:
            app_map: dict[str, list[JournalEntry]] = defaultdict(list)
            ign_map: dict[str, list[JournalEntry]] = defaultdict(list)
            all_keys: set[str] = set()

            for e in approved_entries:
                key = extractor(e)
                app_map[key].append(e)
                all_keys.add(key)

            for e in ignored_entries:
                key = extractor(e)
                ign_map[key].append(e)
                all_keys.add(key)

            for key in sorted(all_keys):
                app_list = app_map[key]
                ign_list = ign_map[key]
                tot_count = len(app_list) + len(ign_list)
                approval_rate = len(app_list) / tot_count

                app_r_vals = [
                    float(e.outcome_data.realized_r) for e in app_list if e.outcome_data is not None
                ]
                ign_r_vals = [
                    float(e.outcome_data.realized_r) for e in ign_list if e.outcome_data is not None
                ]

                app_r = calculate_expectancy(app_r_vals)
                ign_r = calculate_expectancy(ign_r_vals)
                delta_r = app_r - ign_r if app_list and ign_list else 0.0

                app_wr = (
                    sum(1 for e in app_list if e.outcome == TradeOutcome.WIN) / len(app_list)
                    if app_list
                    else 0.0
                )
                ign_wr = (
                    sum(1 for e in ign_list if e.outcome == TradeOutcome.WIN) / len(ign_list)
                    if ign_list
                    else 0.0
                )
                delta_wr = app_wr - ign_wr if app_list and ign_list else 0.0

                results.append(
                    HumanSegmentStats(
                        dimension=dim_name,
                        segment_value=key,
                        total_count=tot_count,
                        approved_count=len(app_list),
                        ignored_count=len(ign_list),
                        approval_rate=approval_rate,
                        approved_expectancy_r=app_r,
                        ignored_expectancy_r=ign_r,
                        delta_expectancy_r=delta_r,
                        approved_win_rate=app_wr,
                        ignored_win_rate=ign_wr,
                        delta_win_rate=delta_wr,
                    )
                )

        return results

    def _is_approved(self, entry: JournalEntry) -> bool:
        if entry.status in (ProposalStatus.APPROVED, ProposalStatus.DISPATCHED):
            return True
        decision = (entry.snapshot.user_decision or "").strip().upper()
        return decision == "APPROVED"

    def _is_ignored(self, entry: JournalEntry) -> bool:
        if self._is_approved(entry):
            return False
        if entry.status in (ProposalStatus.REJECTED, ProposalStatus.EXPIRED):
            return True
        decision = (entry.snapshot.user_decision or "").strip().upper()
        return decision in ("REJECTED", "IGNORED", "EXPIRED")
