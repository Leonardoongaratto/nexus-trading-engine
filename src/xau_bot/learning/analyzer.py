"""Analysis engine for trade journal aggregation, loss patterns, and hypothesis generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from xau_bot.filters.models import NewsFilterConfig, SpreadFilterConfig
from xau_bot.journal.models import JournalEntry, TradeOutcome
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.models import (
    FeatureStats,
    Hypothesis,
    HypothesisStatus,
    LearningReviewReport,
)
from xau_bot.learning.stats import (
    calculate_expectancy,
    calculate_win_loss_rates,
    two_proportion_z_test,
)


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    feature_name: str
    condition_key: str
    description: str
    predicate: Callable[[JournalEntry], bool]
    suggested_parameter: str
    current_value: str
    proposed_value: str
    proposed_action: str


class LearningAnalyzer:
    """Aggregates journal entries and formulates parameter tuning hypotheses."""

    def __init__(
        self,
        repository: JournalRepository,
        min_samples: int = 30,
        alpha: float = 0.05,
        spread_config: SpreadFilterConfig | None = None,
        news_config: NewsFilterConfig | None = None,
    ) -> None:
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        self.repository = repository
        self.min_samples = min_samples
        self.alpha = alpha
        self.spread_config = spread_config or SpreadFilterConfig()
        self.news_config = news_config or NewsFilterConfig()

    def run_review(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> LearningReviewReport:
        """Executes a full periodic review across journal entries."""
        now = datetime.now(UTC)
        all_entries = self.repository.list_entries(
            start_time=start_time,
            end_time=end_time,
            limit=None,
        )

        valid_trades = [
            e
            for e in all_entries
            if e.outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN)
        ]

        if not valid_trades:
            return LearningReviewReport(
                reviewed_at=now,
                total_trades_analyzed=0,
                total_wins=0,
                total_losses=0,
                total_breakevens=0,
                win_rate=0.0,
                loss_rate=0.0,
                global_expectancy_r=0.0,
                min_samples=self.min_samples,
                alpha=self.alpha,
                loss_cause_distribution={},
                feature_stats=(),
                hypotheses=(),
                accumulating_observations=(),
            )

        total_trades = len(valid_trades)
        wins = sum(1 for e in valid_trades if e.outcome == TradeOutcome.WIN)
        losses = sum(1 for e in valid_trades if e.outcome == TradeOutcome.LOSS)
        breakevens = sum(1 for e in valid_trades if e.outcome == TradeOutcome.BREAKEVEN)

        win_rate, loss_rate = calculate_win_loss_rates(wins, losses, total_trades)

        r_values = [e.outcome_data.realized_r for e in valid_trades if e.outcome_data is not None]
        global_expectancy = calculate_expectancy(r_values)

        loss_cause_counter: Counter[str] = Counter()
        for e in valid_trades:
            if e.outcome == TradeOutcome.LOSS:
                cause_str = e.loss_cause.value if e.loss_cause else "UNKNOWN"
                loss_cause_counter[cause_str] += 1
        loss_cause_distribution = dict(loss_cause_counter.most_common())

        condition_specs = self._build_condition_specs(valid_trades)

        feature_stats_list: list[FeatureStats] = []
        hypotheses_list: list[Hypothesis] = []
        accumulating_list: list[FeatureStats] = []

        hyp_counter = 1
        for spec in condition_specs:
            matching = [e for e in valid_trades if spec.predicate(e)]
            n_cond = len(matching)
            if n_cond == 0:
                continue

            loss_cond = sum(1 for e in matching if e.outcome == TradeOutcome.LOSS)
            win_cond = sum(1 for e in matching if e.outcome == TradeOutcome.WIN)
            be_cond = sum(1 for e in matching if e.outcome == TradeOutcome.BREAKEVEN)
            loss_rate_cond = loss_cond / n_cond

            r_cond = [e.outcome_data.realized_r for e in matching if e.outcome_data is not None]
            mean_r_cond = calculate_expectancy(r_cond)

            n_comp = total_trades - n_cond
            loss_comp = losses - loss_cond

            if n_comp <= 0:
                z_score, p_value = 0.0, 1.0
                baseline_rate = loss_rate
            else:
                baseline_rate = loss_comp / n_comp
                z_score, p_value = two_proportion_z_test(
                    count1=loss_cond,
                    n1=n_cond,
                    count2=loss_comp,
                    n2=n_comp,
                )

            is_significant = (
                n_cond >= self.min_samples
                and p_value < self.alpha
                and loss_rate_cond > baseline_rate
            )

            fs = FeatureStats(
                feature_name=spec.feature_name,
                condition_key=spec.condition_key,
                sample_size=n_cond,
                loss_count=loss_cond,
                win_count=win_cond,
                breakeven_count=be_cond,
                loss_rate=round(loss_rate_cond, 4),
                baseline_loss_rate=round(baseline_rate, 4),
                z_score=z_score,
                p_value=p_value,
                mean_realized_r=mean_r_cond,
                is_statistically_significant=is_significant,
                description=spec.description,
            )
            feature_stats_list.append(fs)

            if loss_rate_cond > baseline_rate:
                if n_cond < self.min_samples or not is_significant:
                    accumulating_list.append(fs)
                elif is_significant:
                    date_str = now.strftime("%Y%m%d")
                    hyp = Hypothesis(
                        id=f"HYP-{spec.feature_name.upper()[:4]}-{date_str}-{hyp_counter:02d}",
                        created_at=now,
                        parameter_name=spec.suggested_parameter,
                        current_value=spec.current_value,
                        proposed_value=spec.proposed_value,
                        sample_size=n_cond,
                        condition_loss_rate=round(loss_rate_cond, 4),
                        baseline_loss_rate=round(baseline_rate, 4),
                        p_value=p_value,
                        z_score=z_score,
                        observed_expectancy_r=mean_r_cond,
                        status=HypothesisStatus.PROPOSED,
                        description=spec.description,
                        rationale=(
                            f"Taxa de perda observada de {loss_rate_cond * 100:.1f}% "
                            f"vs complemento de {baseline_rate * 100:.1f}% "
                            f"(p={p_value:.4f}, N={n_cond}, R={mean_r_cond:+.2f}). "
                            f"{spec.proposed_action}."
                        ),
                    )
                    hypotheses_list.append(hyp)
                    hyp_counter += 1

        return LearningReviewReport(
            reviewed_at=now,
            total_trades_analyzed=total_trades,
            total_wins=wins,
            total_losses=losses,
            total_breakevens=breakevens,
            win_rate=win_rate,
            loss_rate=loss_rate,
            global_expectancy_r=global_expectancy,
            min_samples=self.min_samples,
            alpha=self.alpha,
            loss_cause_distribution=loss_cause_distribution,
            feature_stats=tuple(feature_stats_list),
            hypotheses=tuple(hypotheses_list),
            accumulating_observations=tuple(accumulating_list),
        )

    def _build_condition_specs(self, trades: list[JournalEntry]) -> list[ConditionSpec]:
        specs: list[ConditionSpec] = []

        def _make_session_pred(target: str) -> Callable[[JournalEntry], bool]:
            return lambda e: e.snapshot.session == target

        def _make_poi_pred(target: str) -> Callable[[JournalEntry], bool]:
            return lambda e: e.snapshot.poi_type == target

        def _make_conf_pred(target: str) -> Callable[[JournalEntry], bool]:
            return lambda e: target in e.snapshot.confluences

        unique_sessions = {e.snapshot.session for e in trades if e.snapshot.session}
        for s in sorted(unique_sessions):
            specs.append(
                ConditionSpec(
                    feature_name="session",
                    condition_key=s,
                    description=f"Operações executadas na sessão {s}",
                    predicate=_make_session_pred(s),
                    suggested_parameter="allowed_sessions",
                    current_value="ALL",
                    proposed_value=f"RESTRICT_{s}",
                    proposed_action=f"Desativar ou restringir operações na sessão {s}",
                )
            )

        unique_pois = {e.snapshot.poi_type for e in trades if e.snapshot.poi_type}
        for p in sorted(unique_pois):
            specs.append(
                ConditionSpec(
                    feature_name="poi_type",
                    condition_key=p,
                    description=f"Operações baseadas no POI {p}",
                    predicate=_make_poi_pred(p),
                    suggested_parameter="poi_selection_mode",
                    current_value="ALL",
                    proposed_value=f"RESTRICT_{p}",
                    proposed_action=f"Filtrar ou exigir confluência adicional para POI {p}",
                )
            )

        unique_confluences: set[str] = set()
        for e in trades:
            unique_confluences.update(e.snapshot.confluences)
        for c in sorted(unique_confluences):
            specs.append(
                ConditionSpec(
                    feature_name="confluence",
                    condition_key=c,
                    description=f"Operações contendo a confluência {c}",
                    predicate=_make_conf_pred(c),
                    suggested_parameter=f"filter_confluence_{c.lower()}",
                    current_value="ENABLED",
                    proposed_value="DISABLED" if "COUNTER" in c else "OPTIONAL",
                    proposed_action=f"Revisar relevância e pesos da confluência {c}",
                )
            )

        if self.spread_config.max_spread_points is not None:
            spread_param = "max_spread_points"
            current_spread = str(self.spread_config.max_spread_points)
            proposed_spread = str(round(self.spread_config.max_spread_points * 0.75, 1))
        else:
            spread_param = "max_spread_atr_ratio"
            current_spread = str(self.spread_config.max_spread_atr_ratio)
            proposed_spread = str(round(self.spread_config.max_spread_atr_ratio * 0.75, 3))

        current_news = str(self.news_config.blackout_minutes_before)
        proposed_news = str(round(self.news_config.blackout_minutes_before + 15.0, 1))

        specs.extend(
            [
                ConditionSpec(
                    feature_name="displacement",
                    condition_key="DISPLACEMENT < 1.0 ATR",
                    description="Operações com deslocamento em ATR < 1.0",
                    predicate=lambda e: e.snapshot.displacement_atr < 1.0,
                    suggested_parameter="min_displacement_multiplier",
                    current_value="1.0",
                    proposed_value="1.2",
                    proposed_action="Aumentar deslocamento mínimo para validação de CHoCH",
                ),
                ConditionSpec(
                    feature_name="spread",
                    condition_key="SPREAD >= 0.30",
                    description="Operações executadas com spread_price >= 0.30",
                    predicate=lambda e: e.snapshot.spread_price >= 0.30,
                    suggested_parameter=spread_param,
                    current_value=current_spread,
                    proposed_value=proposed_spread,
                    proposed_action="Reduzir limite máximo de spread tolerado",
                ),
                ConditionSpec(
                    feature_name="news",
                    condition_key="NEWS_DISTANCE <= 30m",
                    description="Operações abertas a menos de 30min de notícias de alto impacto",
                    predicate=lambda e: (
                        e.snapshot.news_distance_min is not None
                        and e.snapshot.news_distance_min <= 30.0
                    ),
                    suggested_parameter="blackout_minutes_before",
                    current_value=current_news,
                    proposed_value=proposed_news,
                    proposed_action="Aumentar janela de blackout para eventos macroeconômicos",
                ),
                ConditionSpec(
                    feature_name="liquidity",
                    condition_key="NO_SWEEP",
                    description="Entradas sem confirmação de varredura prévia de liquidez",
                    predicate=lambda e: not e.snapshot.swept_liquidity,
                    suggested_parameter="require_sweep_confirmation",
                    current_value="False",
                    proposed_value="True",
                    proposed_action="Tornar obrigatória a confirmação de varredura prévia",
                ),
                ConditionSpec(
                    feature_name="bias",
                    condition_key="HTF_COUNTER",
                    description="Operações executadas contra a tendência do HTF",
                    predicate=lambda e: (
                        "HTF_COUNTER" in e.snapshot.confluences
                        or (bool(e.snapshot.htf_bias) and e.snapshot.htf_bias != e.direction.value)
                    ),
                    suggested_parameter="require_htf_alignment",
                    current_value="False",
                    proposed_value="True",
                    proposed_action="Restringir entradas que operem contra o viés HTF",
                ),
            ]
        )

        return specs
