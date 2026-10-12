"""Unit tests for human discretion and edge analysis (T-A8)."""

from __future__ import annotations

import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.cli.main import main
from xau_bot.domain.models import Direction
from xau_bot.journal import (
    JournalEntry,
    JournalRepository,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)
from xau_bot.learning.human import HumanDiscretionAnalyzer
from xau_bot.learning.models import (
    CohortMetrics,
    HumanDiscretionReport,
    HumanDiscretionVerdict,
    HumanSegmentStats,
)
from xau_bot.learning.reporter import LearningReporter


def _create_entry(
    entry_id: str,
    status: ProposalStatus = ProposalStatus.APPROVED,
    user_decision: str | None = None,
    outcome: TradeOutcome = TradeOutcome.WIN,
    realized_r: float = 2.0,
    direction: Direction = Direction.BULLISH,
    session: str = "LONDON",
    poi_type: str = "FVG",
    executed: bool = True,
    created_at: datetime | None = None,
) -> JournalEntry:
    if user_decision is None:
        user_decision = (
            "APPROVED"
            if status in (ProposalStatus.APPROVED, ProposalStatus.DISPATCHED)
            else "REJECTED"
        )
    now = created_at or datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session=session,
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type=poi_type,
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        user_decision=user_decision,
    )
    outcome_data = (
        OutcomeData(
            executed=executed,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            realized_r=realized_r,
        )
        if outcome in (TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN)
        else None
    )
    return JournalEntry(
        id=entry_id,
        proposal_id=f"prop_{entry_id}",
        snapshot=snap,
        direction=direction,
        status=status,
        outcome=outcome,
        outcome_data=outcome_data,
        created_at=now,
    )


def test_human_models_serialization() -> None:
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    cohort = CohortMetrics(
        cohort_name="APPROVED",
        total_proposals=20,
        evaluated_trades=18,
        win_count=12,
        loss_count=5,
        breakeven_count=1,
        win_rate=0.6667,
        loss_rate=0.2778,
        expectancy_r=1.25,
        std_dev_r=1.10,
        total_realized_r=22.5,
        profit_factor=2.5,
        realized_r_values=(2.0, -1.0, 3.0),
    )
    c_dict = cohort.to_dict()
    reconstructed_cohort = CohortMetrics.from_dict(c_dict)
    assert reconstructed_cohort.cohort_name == "APPROVED"
    assert reconstructed_cohort.total_proposals == 20
    assert reconstructed_cohort.evaluated_trades == 18
    assert reconstructed_cohort.win_count == 12
    assert reconstructed_cohort.loss_count == 5
    assert reconstructed_cohort.breakeven_count == 1
    assert reconstructed_cohort.win_rate == 0.6667
    assert reconstructed_cohort.expectancy_r == 1.25
    assert reconstructed_cohort.realized_r_values == (2.0, -1.0, 3.0)

    inf_cohort = CohortMetrics(
        cohort_name="INF_PF",
        total_proposals=5,
        evaluated_trades=5,
        win_count=5,
        loss_count=0,
        breakeven_count=0,
        win_rate=1.0,
        loss_rate=0.0,
        expectancy_r=2.0,
        std_dev_r=0.0,
        total_realized_r=10.0,
        profit_factor=math.inf,
        realized_r_values=(2.0, 2.0),
    )
    assert inf_cohort.to_dict()["profit_factor"] == 999.0

    segment = HumanSegmentStats(
        dimension="session",
        segment_value="LONDON",
        total_count=30,
        approved_count=18,
        ignored_count=12,
        approval_rate=0.60,
        approved_expectancy_r=1.20,
        ignored_expectancy_r=-0.40,
        delta_expectancy_r=1.60,
        approved_win_rate=0.65,
        ignored_win_rate=0.35,
        delta_win_rate=0.30,
    )
    s_dict = segment.to_dict()
    reconstructed_seg = HumanSegmentStats.from_dict(s_dict)
    assert reconstructed_seg.dimension == "session"
    assert reconstructed_seg.segment_value == "LONDON"
    assert reconstructed_seg.delta_expectancy_r == 1.60

    report = HumanDiscretionReport(
        analyzed_at=now,
        min_samples=30,
        alpha=0.05,
        total_proposals=20,
        approved_metrics=cohort,
        ignored_metrics=cohort,
        combined_metrics=cohort,
        acceptance_rate=0.60,
        delta_expectancy_r=0.50,
        delta_win_rate=0.15,
        p_value_expectancy=0.025,
        p_value_win_rate=0.040,
        verdict=HumanDiscretionVerdict.POSITIVE_EDGE,
        verdict_summary="Significant edge detected.",
        segments=(segment,),
    )
    r_dict = report.to_dict()
    reconstructed_rep = HumanDiscretionReport.from_dict(r_dict)
    assert reconstructed_rep.verdict == HumanDiscretionVerdict.POSITIVE_EDGE
    assert reconstructed_rep.min_samples == 30
    assert reconstructed_rep.delta_expectancy_r == 0.50
    assert len(reconstructed_rep.segments) == 1


def test_analyzer_parameter_validation() -> None:
    repo = JournalRepository(":memory:")
    try:
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            HumanDiscretionAnalyzer(repo, min_samples=0)

        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            HumanDiscretionAnalyzer(repo, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            HumanDiscretionAnalyzer(repo, alpha=1.0)
    finally:
        repo.close()


def test_cohort_partitioning_ac_human_01() -> None:
    repo = JournalRepository(":memory:")
    try:
        entries: list[JournalEntry] = []

        entries.append(_create_entry("app_1", status=ProposalStatus.APPROVED, realized_r=2.0))
        entries.append(
            _create_entry(
                "app_2",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
        entries.append(
            _create_entry(
                "app_3",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.BREAKEVEN,
                realized_r=0.0,
            )
        )
        entries.append(
            _create_entry(
                "app_4",
                status=ProposalStatus.DISPATCHED,
                outcome=TradeOutcome.WIN,
                realized_r=3.0,
            )
        )
        entries.append(
            _create_entry(
                "app_5",
                status=ProposalStatus.CANCELLED,
                user_decision="APPROVED",
                outcome=TradeOutcome.WIN,
                realized_r=1.5,
            )
        )
        entries.append(
            _create_entry(
                "app_pending",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.UNEXECUTED,
                realized_r=0.0,
            )
        )

        entries.append(
            _create_entry(
                "ign_1",
                status=ProposalStatus.REJECTED,
                user_decision="REJECTED",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
        entries.append(
            _create_entry(
                "ign_2",
                status=ProposalStatus.EXPIRED,
                user_decision="EXPIRED",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
        entries.append(
            _create_entry(
                "ign_3",
                status=ProposalStatus.CANCELLED,
                user_decision="IGNORED",
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
        )
        entries.append(
            _create_entry(
                "ign_pending",
                status=ProposalStatus.REJECTED,
                user_decision="REJECTED",
                outcome=TradeOutcome.UNEXECUTED,
                realized_r=0.0,
            )
        )

        entries.append(
            _create_entry(
                "cancelled_system",
                status=ProposalStatus.CANCELLED,
                user_decision="CANCELLED",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
        entries.append(
            _create_entry(
                "pending_system",
                status=ProposalStatus.PENDING_APPROVAL,
                user_decision="PENDING",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )

        analyzer = HumanDiscretionAnalyzer(repo, min_samples=5)
        report = analyzer.from_entries(entries)

        assert report.raw_proposals == 12
        assert report.total_proposals == 8
        assert report.approved_metrics.total_proposals == 5
        assert report.approved_metrics.evaluated_trades == 5
        assert report.approved_metrics.win_count == 3
        assert report.approved_metrics.loss_count == 1
        assert report.approved_metrics.breakeven_count == 1
        assert report.approved_metrics.win_rate == 3 / 5
        assert report.approved_metrics.loss_rate == 1 / 5
        assert math.isclose(report.approved_metrics.expectancy_r, 1.10, rel_tol=1e-3)
        assert report.approved_metrics.std_dev_r > 0.0

        assert report.ignored_metrics.total_proposals == 3
        assert report.ignored_metrics.evaluated_trades == 3
        assert report.ignored_metrics.win_count == 1
        assert report.ignored_metrics.loss_count == 2
        assert report.ignored_metrics.breakeven_count == 0
        assert math.isclose(report.ignored_metrics.expectancy_r, 0.0, rel_tol=1e-3)

        assert report.acceptance_rate == 5 / 8
        assert math.isclose(report.delta_expectancy_r, 1.10, rel_tol=1e-3)
        assert math.isclose(report.delta_win_rate, (3 / 5) - (1 / 3), rel_tol=1e-3)
        assert analyzer._is_ignored(entries[0]) is False
    finally:
        repo.close()


def test_win_rate_z_test_ac_human_02() -> None:
    repo = JournalRepository(":memory:")
    try:
        from xau_bot.learning.stats import two_proportion_z_test

        app_entries = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN if i < 18 else TradeOutcome.LOSS,
                realized_r=2.0 if i < 18 else -1.0,
            )
            for i in range(25)
        ]
        ign_entries = [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.WIN if i < 7 else TradeOutcome.LOSS,
                realized_r=2.0 if i < 7 else -1.0,
            )
            for i in range(25)
        ]

        analyzer = HumanDiscretionAnalyzer(repo, min_samples=30, alpha=0.05)
        report = analyzer.from_entries(app_entries + ign_entries)

        assert report.p_value_win_rate is not None
        _, expected_p = two_proportion_z_test(18, 25, 7, 25)
        assert math.isclose(report.p_value_win_rate, expected_p, rel_tol=1e-5)
        assert report.p_value_win_rate < 0.05
        assert math.isclose(report.delta_win_rate, (18 / 25) - (7 / 25), rel_tol=1e-4)

        md = LearningReporter.generate_human_discretion_markdown(report)
        assert f"{report.p_value_win_rate:.4f}" in md
        assert "**SIM**" in md

        empty_report = analyzer.from_entries(app_entries)
        assert empty_report.p_value_win_rate is None
        assert empty_report.verdict == HumanDiscretionVerdict.INSUFFICIENT_DATA
    finally:
        repo.close()


def test_verdicts_ac_human_03() -> None:
    repo = JournalRepository(":memory:")
    try:
        analyzer = HumanDiscretionAnalyzer(repo, min_samples=30, alpha=0.05)

        small_cohort = [
            _create_entry(
                f"e_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
            for i in range(10)
        ] + [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
            for i in range(10)
        ]
        report_small = analyzer.from_entries(small_cohort)
        assert report_small.verdict == HumanDiscretionVerdict.INSUFFICIENT_DATA
        assert "Amostra insuficiente" in report_small.verdict_summary

        pos_cohort = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.5 + (i % 3) * 0.2,
            )
            for i in range(25)
        ] + [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0 + (i % 2) * 0.1,
            )
            for i in range(25)
        ]
        report_pos = analyzer.from_entries(pos_cohort)
        assert report_pos.verdict == HumanDiscretionVerdict.POSITIVE_EDGE
        assert report_pos.delta_expectancy_r > 0.0
        assert report_pos.p_value_expectancy is not None
        assert report_pos.p_value_expectancy < 0.05
        assert "adiciona valor significativo" in report_pos.verdict_summary

        neg_cohort = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0 + (i % 2) * 0.1,
            )
            for i in range(25)
        ] + [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.WIN,
                realized_r=2.5 + (i % 3) * 0.2,
            )
            for i in range(25)
        ]
        report_neg = analyzer.from_entries(neg_cohort)
        assert report_neg.verdict == HumanDiscretionVerdict.NEGATIVE_BIAS
        assert report_neg.delta_expectancy_r < 0.0
        assert report_neg.p_value_expectancy is not None
        assert report_neg.p_value_expectancy < 0.05
        assert "viés prejudicial sistemático" in report_neg.verdict_summary

        neutral_cohort = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS,
                realized_r=1.0 if i % 2 == 0 else -1.0,
            )
            for i in range(20)
        ] + [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS,
                realized_r=1.0 if i % 2 == 0 else -1.0,
            )
            for i in range(20)
        ]
        report_neutral = analyzer.from_entries(neutral_cohort)
        assert report_neutral.verdict == HumanDiscretionVerdict.NEUTRAL
        assert report_neutral.p_value_expectancy is not None
        assert report_neutral.p_value_expectancy >= 0.05
        assert "sem significância estatística" in report_neutral.verdict_summary

        one_sided_cohort = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
            for i in range(35)
        ]
        report_one_sided = analyzer.from_entries(one_sided_cohort)
        assert report_one_sided.verdict == HumanDiscretionVerdict.INSUFFICIENT_DATA
        assert "Amostra insuficiente em uma das coortes" in report_one_sided.verdict_summary
    finally:
        repo.close()


def test_segmentation_breakdown_ac_human_04() -> None:
    repo = JournalRepository(":memory:")
    try:
        entries: list[JournalEntry] = []

        for i in range(3):
            entries.append(
                _create_entry(
                    f"lon_bull_ob_app_{i}",
                    status=ProposalStatus.APPROVED,
                    direction=Direction.BULLISH,
                    session="LONDON",
                    poi_type="ORDER_BLOCK",
                    outcome=TradeOutcome.WIN,
                    realized_r=2.0,
                )
            )
        entries.append(
            _create_entry(
                "lon_bull_ob_ign",
                status=ProposalStatus.REJECTED,
                direction=Direction.BULLISH,
                session="LONDON",
                poi_type="ORDER_BLOCK",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )

        entries.append(
            _create_entry(
                "ny_bear_fvg_app",
                status=ProposalStatus.APPROVED,
                direction=Direction.BEARISH,
                session="NEW_YORK",
                poi_type="FVG",
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
        for i in range(2):
            entries.append(
                _create_entry(
                    f"ny_bear_fvg_ign_{i}",
                    status=ProposalStatus.REJECTED,
                    direction=Direction.BEARISH,
                    session="NEW_YORK",
                    poi_type="FVG",
                    outcome=TradeOutcome.WIN,
                    realized_r=2.0,
                )
            )

        analyzer = HumanDiscretionAnalyzer(repo, min_samples=5)
        report = analyzer.from_entries(entries)

        assert len(report.segments) > 0
        dim_map = {(s.dimension, s.segment_value): s for s in report.segments}

        assert ("direction", "BULLISH") in dim_map
        s_bull = dim_map[("direction", "BULLISH")]
        assert s_bull.total_count == 4
        assert s_bull.approved_count == 3
        assert s_bull.ignored_count == 1
        assert s_bull.approval_rate == 0.75
        assert s_bull.approved_expectancy_r == 2.0
        assert s_bull.ignored_expectancy_r == -1.0
        assert s_bull.delta_expectancy_r == 3.0

        assert ("session", "LONDON") in dim_map
        s_lon = dim_map[("session", "LONDON")]
        assert s_lon.total_count == 4
        assert s_lon.approval_rate == 0.75

        assert ("poi_type", "FVG") in dim_map
        s_fvg = dim_map[("poi_type", "FVG")]
        assert s_fvg.total_count == 3
        assert s_fvg.approved_count == 1
        assert s_fvg.ignored_count == 2
        assert s_fvg.approved_expectancy_r == -1.0
        assert s_fvg.ignored_expectancy_r == 2.0
        assert s_fvg.delta_expectancy_r == -3.0
    finally:
        repo.close()


def test_analyzer_analyze_with_time_window(tmp_path: Path) -> None:
    db_path = tmp_path / "test_repo.db"
    repo = JournalRepository(db_path)
    try:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        e_old = _create_entry(
            "old_1",
            status=ProposalStatus.APPROVED,
            created_at=now - timedelta(days=2),
        )
        e_today_1 = _create_entry(
            "today_1",
            status=ProposalStatus.APPROVED,
            created_at=now,
        )
        e_today_2 = _create_entry(
            "today_2",
            status=ProposalStatus.REJECTED,
            created_at=now + timedelta(hours=1),
        )
        repo.save_entry(e_old)
        repo.save_entry(e_today_1)
        repo.save_entry(e_today_2)

        analyzer = HumanDiscretionAnalyzer(repo, min_samples=1)
        report = analyzer.analyze(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=2),
        )
        assert report.total_proposals == 2
    finally:
        repo.close()


def test_reporter_human_discretion_markdown_and_save(tmp_path: Path) -> None:
    repo = JournalRepository(":memory:")
    try:
        entries = [
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
            for i in range(5)
        ] + [
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
            for i in range(5)
        ]
        analyzer = HumanDiscretionAnalyzer(repo, min_samples=5)
        report = analyzer.from_entries(entries)

        md = LearningReporter.generate_human_discretion_markdown(report)
        assert "# Relatório de Discricionariedade Humana" in md
        assert "## 1. Resumo Executivo e Veredito" in md
        assert "## 2. Comparativo de Coortes" in md
        assert "## 3. Mensuração de Human Edge e Discricionariedade" in md
        assert "## 4. Segmentação Multifatorial" in md
        assert report.verdict.value in md

        file_path = LearningReporter.save_human_report(report, tmp_path)
        assert file_path.exists()
        saved_text = file_path.read_text(encoding="utf-8")
        assert "# Relatório de Discricionariedade Humana" in saved_text
    finally:
        repo.close()


def test_cli_learn_human_action_and_flags_ac_human_05(tmp_path: Path) -> None:
    db_path = tmp_path / "cli_human.db"
    repo = JournalRepository(db_path)
    for i in range(15):
        repo.save_entry(
            _create_entry(
                f"app_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
        )
    for i in range(15):
        repo.save_entry(
            _create_entry(
                f"ign_{i}",
                status=ProposalStatus.REJECTED,
                outcome=TradeOutcome.LOSS,
                realized_r=-1.0,
            )
        )
    repo.close()

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["learn", "human", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out_val = stdout.getvalue()
    assert "# Relatório de Discricionariedade Humana" in out_val
    assert "Taxa de Aceitação Humana" in out_val

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(["learn", "--human", "--db", str(db_path)], stdout=stdout, stderr=stderr)
    assert exit_code == 0
    out_val_flag = stdout.getvalue()
    assert "# Relatório de Discricionariedade Humana" in out_val_flag

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["learn", "human", "--db", str(db_path), "--json"],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    parsed = json.loads(stdout.getvalue())
    assert parsed["total_proposals"] == 30
    assert parsed["acceptance_rate"] == 0.5
    assert parsed["verdict"] in [v.value for v in HumanDiscretionVerdict]

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        [
            "learn",
            "human",
            "--db",
            str(db_path),
            "--output-dir",
            str(tmp_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    saved_md_files = list(tmp_path.glob("human_discretion_*.md"))
    assert len(saved_md_files) >= 1

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        [
            "learn",
            "human",
            "--db",
            str(db_path),
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    saved_json_files = list(tmp_path.glob("human_discretion_*.json"))
    assert len(saved_json_files) >= 1

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        ["learn", "human", "--db", "missing_file_xyz.db"],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 1
    assert "não encontrado" in stderr.getvalue()

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        [
            "learn",
            "human",
            "--db",
            str(db_path),
            "--min-samples",
            "0",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 2
    assert "Erro de parâmetro" in stderr.getvalue()

    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = main(
        [
            "learn",
            "review",
            "--human",
            "--db",
            str(db_path),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 2
    assert "mutuamente exclusivas" in stderr.getvalue()


def test_reporter_empty_segments_and_edge_cases() -> None:
    repo = JournalRepository(":memory:")
    try:
        now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
        all_win_entries = [
            _create_entry(
                f"win_{i}",
                status=ProposalStatus.APPROVED,
                outcome=TradeOutcome.WIN,
                realized_r=2.0,
            )
            for i in range(5)
        ]
        analyzer = HumanDiscretionAnalyzer(repo, min_samples=5)
        report = analyzer.from_entries(all_win_entries)
        assert report.approved_metrics.profit_factor == 999.0

        report_empty_segs = HumanDiscretionReport(
            analyzed_at=now,
            min_samples=10,
            alpha=0.05,
            total_proposals=5,
            approved_metrics=report.approved_metrics,
            ignored_metrics=report.ignored_metrics,
            combined_metrics=report.combined_metrics,
            acceptance_rate=1.0,
            delta_expectancy_r=2.0,
            delta_win_rate=1.0,
            p_value_expectancy=None,
            p_value_win_rate=None,
            verdict=HumanDiscretionVerdict.INSUFFICIENT_DATA,
            verdict_summary="Insuficiente",
            segments=(),
        )
        md = LearningReporter.generate_human_discretion_markdown(report_empty_segs)
        assert "Nenhum segmento com amostra suficiente para exibição" in md

        e_exp = _create_entry(
            "e_exp",
            status=ProposalStatus.EXPIRED,
            user_decision="EXPIRED",
            outcome=TradeOutcome.LOSS,
            realized_r=-1.0,
        )
        e_rej = _create_entry(
            "e_rej",
            status=ProposalStatus.REJECTED,
            user_decision="REJECTED",
            outcome=TradeOutcome.LOSS,
            realized_r=-1.0,
        )
        assert analyzer._is_ignored(e_exp)
        assert analyzer._is_ignored(e_rej)
    finally:
        repo.close()
