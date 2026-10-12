"""Unit and integration tests for the Learning and Periodic Review module (T-A6)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xau_bot.domain.models import Direction
from xau_bot.journal.models import (
    JournalEntry,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)
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


def test_two_proportion_z_test_significant() -> None:
    # 20 losses in 30 trades (66.7%) vs 30 losses in 100 trades (30.0%)
    z_score, p_value = two_proportion_z_test(count1=20, n1=30, count2=30, n2=100)
    assert z_score > 3.0
    assert p_value < 0.001


def test_two_proportion_z_test_equal_proportions() -> None:
    # 10/50 (20%) vs 20/100 (20%)
    z_score, p_value = two_proportion_z_test(count1=10, n1=50, count2=20, n2=100)
    assert z_score == 0.0
    assert p_value == 1.0


def test_two_proportion_z_test_edge_cases() -> None:
    # n1 or n2 zero
    z, p = two_proportion_z_test(count1=0, n1=0, count2=5, n2=10)
    assert z == 0.0
    assert p == 1.0

    z, p = two_proportion_z_test(count1=5, n1=10, count2=0, n2=0)
    assert z == 0.0
    assert p == 1.0

    # Both zero events
    z, p = two_proportion_z_test(count1=0, n1=30, count2=0, n2=50)
    assert z == 0.0
    assert p == 1.0

    # Both 100% events
    z, p = two_proportion_z_test(count1=30, n1=30, count2=50, n2=50)
    assert z == 0.0
    assert p == 1.0


def test_statistical_helpers() -> None:
    assert calculate_expectancy([]) == 0.0
    assert calculate_expectancy([2.0, -1.0, 1.0, -1.0]) == 0.25

    assert calculate_win_loss_rates(wins=0, losses=0, total=0) == (0.0, 0.0)
    w_rate, l_rate = calculate_win_loss_rates(wins=15, losses=10, total=30)
    assert w_rate == 0.5
    assert round(l_rate, 4) == 0.3333


def test_hypothesis_model_serialization() -> None:
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="HYP-001",
        created_at=now,
        parameter_name="min_displacement_atr",
        current_value="0.8",
        proposed_value="1.2",
        sample_size=35,
        condition_loss_rate=0.75,
        baseline_loss_rate=0.35,
        p_value=0.002,
        z_score=3.09,
        observed_expectancy_r=-0.65,
        status=HypothesisStatus.PROPOSED,
        description="Filtro de deslocamento mínimo em ATR",
        rationale="Alta taxa de perdas observada quando displacement < 1.0 ATR.",
    )

    data = hyp.to_dict()
    assert data["id"] == "HYP-001"
    assert data["status"] == "PROPOSED"
    assert data["sample_size"] == 35
    assert data["p_value"] == 0.002

    loaded_hyp = Hypothesis.from_dict(data)
    assert loaded_hyp.id == hyp.id
    assert loaded_hyp.parameter_name == hyp.parameter_name
    assert loaded_hyp.sample_size == hyp.sample_size
    assert loaded_hyp.status == hyp.status

    # Test from_dict with string without tz
    data_no_tz = dict(data)
    data_no_tz["created_at"] = "2026-10-08T14:00:00"
    loaded_no_tz = Hypothesis.from_dict(data_no_tz)
    assert loaded_no_tz.created_at.tzinfo == UTC


def test_feature_stats_serialization() -> None:
    fs = FeatureStats(
        feature_name="session",
        condition_key="ASIA",
        sample_size=40,
        loss_count=28,
        win_count=10,
        breakeven_count=2,
        loss_rate=0.70,
        baseline_loss_rate=0.35,
        z_score=3.25,
        p_value=0.0011,
        mean_realized_r=-0.45,
        is_statistically_significant=True,
        description="Sessão asiática com taxa de perda anormal",
    )

    d = fs.to_dict()
    assert d["feature_name"] == "session"
    assert d["condition_key"] == "ASIA"
    assert d["is_statistically_significant"] is True

    loaded_fs = FeatureStats.from_dict(d)
    assert loaded_fs.feature_name == fs.feature_name
    assert loaded_fs.condition_key == fs.condition_key
    assert loaded_fs.sample_size == fs.sample_size


def test_learning_review_report_serialization() -> None:
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="HYP-001",
        created_at=now,
        parameter_name="min_displacement_atr",
        current_value="0.8",
        proposed_value="1.2",
        sample_size=35,
        condition_loss_rate=0.75,
        baseline_loss_rate=0.35,
        p_value=0.002,
        z_score=3.09,
        observed_expectancy_r=-0.65,
        status=HypothesisStatus.PROPOSED,
        description="Filtro",
        rationale="Motivo",
    )
    fs = FeatureStats(
        feature_name="session",
        condition_key="ASIA",
        sample_size=40,
        loss_count=28,
        win_count=10,
        breakeven_count=2,
        loss_rate=0.70,
        baseline_loss_rate=0.35,
        z_score=3.25,
        p_value=0.0011,
        mean_realized_r=-0.45,
        is_statistically_significant=True,
    )
    report = LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=100,
        total_wins=60,
        total_losses=35,
        total_breakevens=5,
        win_rate=0.60,
        loss_rate=0.35,
        global_expectancy_r=0.45,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={"COUNTER_BIAS": 15, "HIGH_SPREAD": 10},
        feature_stats=(fs,),
        hypotheses=(hyp,),
        accumulating_observations=(fs,),
    )

    d = report.to_dict()
    assert d["total_trades_analyzed"] == 100
    assert d["win_rate"] == 0.60
    assert d["loss_cause_distribution"]["COUNTER_BIAS"] == 15

    loaded_report = LearningReviewReport.from_dict(d)
    assert loaded_report.total_trades_analyzed == 100
    assert len(loaded_report.hypotheses) == 1
    assert len(loaded_report.feature_stats) == 1
    assert len(loaded_report.accumulating_observations) == 1


def test_analyzer_parameter_validation() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        LearningAnalyzer(repo, min_samples=0)

    with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
        LearningAnalyzer(repo, alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
        LearningAnalyzer(repo, alpha=1.0)


def _create_entry(
    entry_id: str,
    outcome: TradeOutcome,
    session: str = "LONDON",
    poi_type: str = "FVG",
    displacement_atr: float = 1.5,
    spread_price: float = 0.2,
    news_distance_min: float | None = 60.0,
    loss_cause: LossRootCause | None = None,
    realized_r: float = 1.0,
    confluences: tuple[str, ...] = ("HTF_ALIGNED",),
    swept_liquidity: bool = True,
    htf_bias: str = "BULLISH",
    direction: Direction = Direction.BULLISH,
) -> JournalEntry:
    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session=session,
        htf_bias=htf_bias,
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type=poi_type,
        poi_level=2630.0,
        swept_liquidity=swept_liquidity,
        displacement_atr=displacement_atr,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=spread_price,
        atr_m5=2.0,
        news_distance_min=news_distance_min,
        confluences=confluences,
    )
    outcome_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        realized_r=realized_r,
    )
    return JournalEntry(
        id=f"jrn_{entry_id}",
        proposal_id=f"prop_{entry_id}",
        snapshot=snap,
        direction=direction,
        outcome=outcome,
        loss_cause=loss_cause,
        loss_category=LossCategory.BAD_DECISION if loss_cause else None,
        outcome_data=outcome_data,
        created_at=now,
    )


def test_analyzer_empty_repository() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    assert report.total_trades_analyzed == 0
    assert report.win_rate == 0.0
    assert report.loss_rate == 0.0
    assert report.global_expectancy_r == 0.0
    assert len(report.hypotheses) == 0
    assert len(report.loss_cause_distribution) == 0


def test_analyzer_aggregates_loss_causes_and_r_expectancy() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 20 wins (+2.0R), 10 losses (-1.0R): 6 COUNTER_BIAS, 4 HIGH_SPREAD
    for i in range(20):
        repo.save_entry(_create_entry(f"win_{i}", TradeOutcome.WIN, realized_r=2.0))
    for i in range(6):
        repo.save_entry(
            _create_entry(
                f"loss_bias_{i}",
                TradeOutcome.LOSS,
                loss_cause=LossRootCause.COUNTER_BIAS,
                realized_r=-1.0,
            )
        )
    for i in range(4):
        repo.save_entry(
            _create_entry(
                f"loss_spread_{i}",
                TradeOutcome.LOSS,
                loss_cause=LossRootCause.HIGH_SPREAD,
                realized_r=-1.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    assert report.total_trades_analyzed == 30
    assert report.total_wins == 20
    assert report.total_losses == 10
    assert round(report.win_rate, 4) == 0.6667
    assert round(report.loss_rate, 4) == 0.3333
    # Expectancy: (20*2.0 - 10*1.0) / 30 = 30 / 30 = 1.0R
    assert report.global_expectancy_r == 1.0
    assert report.loss_cause_distribution["COUNTER_BIAS"] == 6
    assert report.loss_cause_distribution["HIGH_SPREAD"] == 4
    fs_london = next((fs for fs in report.feature_stats if fs.condition_key == "LONDON"), None)
    assert fs_london is not None
    assert fs_london.mean_realized_r == 1.0
    assert fs_london.sample_size == 30


def test_analyzer_generates_hypothesis_when_n_ge_30_and_p_lt_005() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # Global population: 100 trades.
    # Baseline: 30 losses total (30% loss rate).
    # Condition: low displacement (< 1.0 ATR). 35 trades match this condition:
    # 26 losses and 9 wins (loss rate = 26/35 = 74.3%).
    # Remaining 65 trades: 4 losses and 61 wins.
    # Total losses = 26 + 4 = 30 out of 100.
    for i in range(26):
        repo.save_entry(
            _create_entry(
                f"low_disp_loss_{i}",
                TradeOutcome.LOSS,
                displacement_atr=0.7,
                loss_cause=LossRootCause.INSUFFICIENT_DISPLACEMENT,
                realized_r=-1.0,
            )
        )
    for i in range(9):
        repo.save_entry(
            _create_entry(
                f"low_disp_win_{i}",
                TradeOutcome.WIN,
                displacement_atr=0.7,
                realized_r=1.5,
            )
        )
    for i in range(4):
        repo.save_entry(
            _create_entry(
                f"normal_loss_{i}",
                TradeOutcome.LOSS,
                displacement_atr=1.8,
                loss_cause=LossRootCause.NORMAL_VARIANCE,
                realized_r=-1.0,
            )
        )
    for i in range(61):
        repo.save_entry(
            _create_entry(
                f"normal_win_{i}",
                TradeOutcome.WIN,
                displacement_atr=1.8,
                realized_r=2.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    assert report.total_trades_analyzed == 100
    assert report.total_losses == 30
    assert report.loss_rate == 0.30

    # There must be a proposed hypothesis for displacement
    hyp_disp = next((h for h in report.hypotheses if "displacement" in h.parameter_name), None)
    assert hyp_disp is not None
    assert hyp_disp.status == HypothesisStatus.PROPOSED
    assert hyp_disp.sample_size == 35
    assert hyp_disp.p_value < 0.05
    assert hyp_disp.condition_loss_rate > 0.70
    assert hyp_disp.baseline_loss_rate < 0.10
    assert hyp_disp.proposed_value != hyp_disp.current_value


def test_analyzer_classifies_as_accumulating_when_n_lt_30() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 100 trades, 30 losses baseline.
    # High spread condition (spread_price >= 0.30): only 15 trades (N < 30),
    # but 12 of them are losses (80% loss rate).
    for i in range(12):
        repo.save_entry(
            _create_entry(
                f"spread_loss_{i}",
                TradeOutcome.LOSS,
                spread_price=0.40,
                loss_cause=LossRootCause.HIGH_SPREAD,
                realized_r=-1.0,
            )
        )
    for i in range(3):
        repo.save_entry(
            _create_entry(
                f"spread_win_{i}",
                TradeOutcome.WIN,
                spread_price=0.40,
                realized_r=1.5,
            )
        )
    for i in range(18):
        repo.save_entry(
            _create_entry(
                f"norm_loss_{i}",
                TradeOutcome.LOSS,
                spread_price=0.20,
                realized_r=-1.0,
            )
        )
    for i in range(67):
        repo.save_entry(
            _create_entry(
                f"norm_win_{i}",
                TradeOutcome.WIN,
                spread_price=0.20,
                realized_r=2.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    # Must NOT have a hypothesis for spread since N=15 < 30
    assert not any("spread" in h.parameter_name for h in report.hypotheses)
    # Must be in accumulating observations
    obs_spread = next(
        (obs for obs in report.accumulating_observations if "SPREAD" in obs.condition_key), None
    )
    assert obs_spread is not None
    assert obs_spread.sample_size == 15
    assert obs_spread.loss_rate == 0.80


def test_analyzer_does_not_generate_hypothesis_when_p_ge_005() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 100 trades, 30 losses baseline (30%).
    # Condition has N=35 trades, with 12 losses (12/35 = 34.3% loss rate).
    # 34.3% vs 30% with N=35 is not statistically significant (p > 0.05).
    for i in range(12):
        repo.save_entry(
            _create_entry(
                f"asian_loss_{i}",
                TradeOutcome.LOSS,
                session="ASIA",
                realized_r=-1.0,
            )
        )
    for i in range(23):
        repo.save_entry(
            _create_entry(
                f"asian_win_{i}",
                TradeOutcome.WIN,
                session="ASIA",
                realized_r=1.5,
            )
        )
    for i in range(18):
        repo.save_entry(
            _create_entry(
                f"london_loss_{i}",
                TradeOutcome.LOSS,
                session="LONDON",
                realized_r=-1.0,
            )
        )
    for i in range(47):
        repo.save_entry(
            _create_entry(
                f"london_win_{i}",
                TradeOutcome.WIN,
                session="LONDON",
                realized_r=2.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    # N=35 >= 30, but p >= 0.05, so NO hypothesis for session
    assert not any("session" in h.parameter_name for h in report.hypotheses)
    fs_asia = next((fs for fs in report.feature_stats if fs.condition_key == "ASIA"), None)
    assert fs_asia is not None
    assert fs_asia.sample_size == 35
    assert fs_asia.p_value >= 0.05
    assert fs_asia.is_statistically_significant is False
    assert any(obs.condition_key == "ASIA" for obs in report.accumulating_observations)


def test_reporter_empty_report() -> None:
    from xau_bot.learning.reporter import LearningReporter

    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    report = LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=0,
        total_wins=0,
        total_losses=0,
        total_breakevens=0,
        win_rate=0.0,
        loss_rate=0.0,
        global_expectancy_r=0.0,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={},
        feature_stats=(),
        hypotheses=(),
        accumulating_observations=(),
    )

    md = LearningReporter.generate_markdown_report(report)
    assert "# Relatório de Aprendizado e Revisão Periódica" in md
    assert "Total de Trades Analisados | 0" in md
    assert "Nenhuma hipótese formal proposta" in md


def test_reporter_with_hypotheses_and_observations() -> None:
    from xau_bot.learning.reporter import LearningReporter

    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="HYP-DISP-01",
        created_at=now,
        parameter_name="min_displacement_atr",
        current_value="0.8",
        proposed_value="1.2",
        sample_size=35,
        condition_loss_rate=0.7429,
        baseline_loss_rate=0.3000,
        p_value=0.0001,
        z_score=3.75,
        observed_expectancy_r=-0.65,
        status=HypothesisStatus.PROPOSED,
        description="Operações com deslocamento em ATR < 1.0",
        rationale="Aumentar deslocamento mínimo em ATR para validação de CHoCH.",
    )
    obs = FeatureStats(
        feature_name="spread",
        condition_key="SPREAD >= 0.30",
        sample_size=15,
        loss_count=12,
        win_count=3,
        breakeven_count=0,
        loss_rate=0.80,
        baseline_loss_rate=0.30,
        z_score=3.5,
        p_value=0.0005,
        mean_realized_r=-0.75,
        is_statistically_significant=False,
        description="Operações com spread elevado",
    )
    fs = FeatureStats(
        feature_name="session",
        condition_key="LONDON",
        sample_size=50,
        loss_count=15,
        win_count=35,
        breakeven_count=0,
        loss_rate=0.30,
        baseline_loss_rate=0.30,
        z_score=0.0,
        p_value=1.0,
        mean_realized_r=0.55,
        is_statistically_significant=False,
        description="Sessão de Londres",
    )
    report = LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=100,
        total_wins=65,
        total_losses=30,
        total_breakevens=5,
        win_rate=0.65,
        loss_rate=0.30,
        global_expectancy_r=0.45,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={"COUNTER_BIAS": 15, "INSUFFICIENT_DISPLACEMENT": 15},
        feature_stats=(fs,),
        hypotheses=(hyp,),
        accumulating_observations=(obs,),
    )

    md = LearningReporter.generate_markdown_report(report)
    assert "# Relatório de Aprendizado e Revisão Periódica" in md
    assert "HYP-DISP-01" in md
    assert "min_displacement_atr" in md
    assert "1.2" in md
    assert "COUNTER_BIAS" in md
    assert "SPREAD >= 0.30" in md
    assert "LONDON" in md


def test_reporter_save_report(tmp_path: Path) -> None:
    from xau_bot.learning.reporter import LearningReporter

    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    report = LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=10,
        total_wins=6,
        total_losses=4,
        total_breakevens=0,
        win_rate=0.6,
        loss_rate=0.4,
        global_expectancy_r=0.2,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={},
        feature_stats=(),
        hypotheses=(),
        accumulating_observations=(),
    )

    saved_path = LearningReporter.save_report(report, tmp_path)
    assert saved_path.exists()
    assert saved_path.name.startswith("learning_review_")
    assert saved_path.suffix == ".md"
    content = saved_path.read_text(encoding="utf-8")
    assert "# Relatório de Aprendizado e Revisão Periódica" in content


def test_reproduce_m01_complement_comparison() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 60 trades matching low displacement: 30 losses (50%), 30 wins
    for i in range(30):
        repo.save_entry(
            _create_entry(
                f"disp_loss_{i}",
                TradeOutcome.LOSS,
                displacement_atr=0.7,
                realized_r=-1.0,
            )
        )
    for i in range(30):
        repo.save_entry(
            _create_entry(
                f"disp_win_{i}",
                TradeOutcome.WIN,
                displacement_atr=0.7,
                realized_r=1.5,
            )
        )
    # Complement: 40 trades with displacement=1.8: only 5 losses (12.5%), 35 wins
    for i in range(5):
        repo.save_entry(
            _create_entry(
                f"comp_loss_{i}",
                TradeOutcome.LOSS,
                displacement_atr=1.8,
                realized_r=-1.0,
            )
        )
    for i in range(35):
        repo.save_entry(
            _create_entry(
                f"comp_win_{i}",
                TradeOutcome.WIN,
                displacement_atr=1.8,
                realized_r=2.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    # In M-01, comparing against pooled total resulted in p=0.0614 (NO hypothesis!).
    # Comparing against the independent complement (30/60 vs 5/40) yields p=0.0001 < 0.05!
    hyp = next((h for h in report.hypotheses if "displacement" in h.parameter_name), None)
    assert hyp is not None, (
        "Hypothesis must be proposed when condition loss rate is 50% vs complement 12.5%"
    )
    assert hyp.p_value < 0.001


def test_reproduce_m02_no_tautological_loss_cause_condition() -> None:
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 30 trades with loss cause TIGHT_SL in 200 trades
    for i in range(30):
        repo.save_entry(
            _create_entry(
                f"tight_loss_{i}",
                TradeOutcome.LOSS,
                loss_cause=LossRootCause.TIGHT_SL,
                realized_r=-1.0,
            )
        )
    for i in range(170):
        repo.save_entry(
            _create_entry(
                f"win_{i}",
                TradeOutcome.WIN,
                realized_r=2.0,
            )
        )

    analyzer = LearningAnalyzer(repo, min_samples=30, alpha=0.05)
    report = analyzer.run_review()

    # Must NOT produce hypothesis based on tautological outcome filter
    assert not any("min_sl_atr_buffer" in h.parameter_name for h in report.hypotheses)
    # But loss cause must be present in loss_cause_distribution
    assert report.loss_cause_distribution.get("TIGHT_SL") == 30


def test_reproduce_m03_real_config_current_values() -> None:
    from xau_bot.filters.models import NewsFilterConfig, SpreadFilterConfig
    from xau_bot.learning.analyzer import LearningAnalyzer

    repo = JournalRepository(":memory:")
    # 35 trades with high spread (0.40) and 25 losses, plus 65 normal trades (10 losses)
    for i in range(25):
        repo.save_entry(
            _create_entry(
                f"spread_loss_{i}",
                TradeOutcome.LOSS,
                spread_price=0.40,
                realized_r=-1.0,
            )
        )
    for i in range(10):
        repo.save_entry(
            _create_entry(
                f"spread_win_{i}",
                TradeOutcome.WIN,
                spread_price=0.40,
                realized_r=1.5,
            )
        )
    for i in range(10):
        repo.save_entry(
            _create_entry(
                f"norm_loss_{i}",
                TradeOutcome.LOSS,
                spread_price=0.20,
                realized_r=-1.0,
            )
        )
    for i in range(55):
        repo.save_entry(
            _create_entry(
                f"norm_win_{i}",
                TradeOutcome.WIN,
                spread_price=0.20,
                realized_r=2.0,
            )
        )

    spread_cfg = SpreadFilterConfig(max_spread_points=35.0)
    news_cfg = NewsFilterConfig(blackout_minutes_before=30.0, blackout_minutes_after=30.0)
    analyzer = LearningAnalyzer(
        repo,
        min_samples=30,
        alpha=0.05,
        spread_config=spread_cfg,
        news_config=news_cfg,
    )
    report = analyzer.run_review()

    hyp_spread = next((h for h in report.hypotheses if "spread" in h.parameter_name), None)
    assert hyp_spread is not None
    # Current value must reflect the actual injected configuration
    assert "35" in hyp_spread.current_value
