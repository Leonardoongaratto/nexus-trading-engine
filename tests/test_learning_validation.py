"""Unit tests for hypothesis validation models, parameter versioning, and FDR stats."""

from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from xau_bot.cli.main import main as cli_main
from xau_bot.domain.models import Direction, Tick
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.models import (
    Hypothesis,
    HypothesisStatus,
    ParameterVersion,
    ValidationMetrics,
    ValidationResult,
)
from xau_bot.learning.stats import fdr_benjamini_hochberg
from xau_bot.learning.validator import HypothesisValidator


def test_hypothesis_status_enum_expansion() -> None:
    assert HypothesisStatus.PROPOSED == "PROPOSED"
    assert HypothesisStatus.ACCUMULATING == "ACCUMULATING"
    assert HypothesisStatus.VALIDATED == "VALIDATED"
    assert HypothesisStatus.REJECTED == "REJECTED"
    assert HypothesisStatus.SHADOW == "SHADOW"
    assert HypothesisStatus.PROMOTED == "PROMOTED"
    assert HypothesisStatus.ROLLED_BACK == "ROLLED_BACK"


def test_validation_metrics_serialization() -> None:
    metrics = ValidationMetrics(
        total_trades=50,
        win_count=35,
        loss_count=15,
        win_rate=0.70,
        expectancy_r=1.25,
        max_drawdown_pct=4.5,
        profit_factor=2.1,
    )
    d = metrics.to_dict()
    assert d["total_trades"] == 50
    assert d["win_rate"] == 0.70
    assert d["expectancy_r"] == 1.25
    assert d["max_drawdown_pct"] == 4.5

    reconstructed = ValidationMetrics.from_dict(d)
    assert reconstructed == metrics


def test_validation_result_serialization() -> None:
    now = datetime(2026, 10, 8, 19, 0, tzinfo=UTC)
    baseline = ValidationMetrics(
        total_trades=60,
        win_count=36,
        loss_count=24,
        win_rate=0.60,
        expectancy_r=0.80,
        max_drawdown_pct=6.0,
        profit_factor=1.5,
    )
    candidate = ValidationMetrics(
        total_trades=50,
        win_count=35,
        loss_count=15,
        win_rate=0.70,
        expectancy_r=1.25,
        max_drawdown_pct=5.0,
        profit_factor=2.0,
    )
    result = ValidationResult(
        hypothesis_id="HYP-DISP-01",
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        expectancy_delta_r=0.45,
        drawdown_delta_pct=-1.0,
        trade_retention_rate=0.8333,
        is_statistically_superior=True,
        p_value_improvement=0.008,
        verdict=HypothesisStatus.VALIDATED,
        rejection_reason="",
        validated_at=now,
    )

    d = result.to_dict()
    assert d["hypothesis_id"] == "HYP-DISP-01"
    assert d["verdict"] == "VALIDATED"
    assert d["expectancy_delta_r"] == 0.45
    assert d["baseline_metrics"]["total_trades"] == 60
    assert d["candidate_metrics"]["total_trades"] == 50

    reconstructed = ValidationResult.from_dict(d)
    assert reconstructed == result


def test_parameter_version_serialization() -> None:
    now = datetime(2026, 10, 8, 19, 30, tzinfo=UTC)
    params = {
        "min_displacement_multiplier": 1.2,
        "max_spread_atr_ratio": 0.05,
        "blackout_minutes_before": 45.0,
    }
    pv = ParameterVersion(
        version_id="pv-20261008-001",
        created_at=now,
        parameters=params,
        source_hypothesis_id="HYP-DISP-01",
        applied_by="operator",
        is_active=True,
        rollback_target_id=None,
    )

    d = pv.to_dict()
    assert d["version_id"] == "pv-20261008-001"
    assert d["parameters"]["min_displacement_multiplier"] == 1.2
    assert d["is_active"] is True

    reconstructed = ParameterVersion.from_dict(d)
    assert reconstructed == pv


def test_fdr_benjamini_hochberg_empty() -> None:
    assert fdr_benjamini_hochberg([]) == []


def test_fdr_benjamini_hochberg_single_value() -> None:
    assert fdr_benjamini_hochberg([0.01], q_threshold=0.05) == [True]
    assert fdr_benjamini_hochberg([0.06], q_threshold=0.05) == [False]


def test_fdr_benjamini_hochberg_mixed() -> None:
    p_vals = [0.008, 0.015, 0.025, 0.10, 0.50]
    expected = [True, True, True, False, False]
    assert fdr_benjamini_hochberg(p_vals, q_threshold=0.05) == expected

    p_vals_permuted = [0.10, 0.008, 0.50, 0.025, 0.015]
    expected_permuted = [False, True, False, True, True]
    assert fdr_benjamini_hochberg(p_vals_permuted, q_threshold=0.05) == expected_permuted


def test_fdr_benjamini_hochberg_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="q_threshold must be between 0.0 and 1.0"):
        fdr_benjamini_hochberg([0.01], q_threshold=0.0)

    with pytest.raises(ValueError, match="q_threshold must be between 0.0 and 1.0"):
        fdr_benjamini_hochberg([0.01], q_threshold=1.5)

    with pytest.raises(ValueError, match="p-values must be between 0.0 and 1.0"):
        fdr_benjamini_hochberg([-0.1], q_threshold=0.05)

    with pytest.raises(ValueError, match="p-values must be between 0.0 and 1.0"):
        fdr_benjamini_hochberg([1.2], q_threshold=0.05)


def test_parameter_registry_create_and_get_active(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_file = tmp_path / "params.db"
    with ParameterRegistry(db_file) as reg:
        assert reg.get_active_version() is None

        v1 = reg.create_version(
            parameters={"min_displacement_multiplier": 1.0, "max_spread_atr_ratio": 0.05},
            applied_by="system_init",
        )
        assert v1.is_active is True
        assert v1.source_hypothesis_id is None

        active = reg.get_active_version()
        assert active is not None
        assert active.version_id == v1.version_id
        assert active.parameters["min_displacement_multiplier"] == 1.0


def test_parameter_registry_activation_switching(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_file = tmp_path / "params.db"
    with ParameterRegistry(db_file) as reg:
        v1 = reg.create_version(
            parameters={"min_displacement_multiplier": 1.0},
            applied_by="init",
            activate=True,
        )
        v2 = reg.create_version(
            parameters={"min_displacement_multiplier": 1.2},
            source_hypothesis_id="HYP-01",
            applied_by="operator",
            activate=True,
        )

        active = reg.get_active_version()
        assert active is not None
        assert active.version_id == v2.version_id
        assert active.parameters["min_displacement_multiplier"] == 1.2

        v1_fetched = reg.get_version(v1.version_id)
        assert v1_fetched is not None
        assert v1_fetched.is_active is False

        versions = reg.list_versions()
        assert len(versions) == 2
        assert versions[0].version_id == v2.version_id
        assert versions[1].version_id == v1.version_id


def test_parameter_registry_rollback(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_file = tmp_path / "params.db"
    with ParameterRegistry(db_file) as reg:
        v1 = reg.create_version(parameters={"spread": 0.20}, applied_by="init")
        v2 = reg.create_version(
            parameters={"spread": 0.35}, applied_by="hyp", source_hypothesis_id="H2"
        )

        v3 = reg.rollback(v1.version_id, applied_by="operator_undo")
        assert v3.is_active is True
        assert v3.parameters["spread"] == 0.20
        assert v3.rollback_target_id == v1.version_id

        v2_refetched = reg.get_version(v2.version_id)
        assert v2_refetched is not None
        assert v2_refetched.is_active is False

        with pytest.raises(ValueError, match="is already the active version"):
            reg.rollback(v3.version_id)

        with pytest.raises(ValueError, match="not found"):
            reg.rollback("non_existent_version_id")


def test_hypothesis_validator_evaluates_promotion_success() -> None:
    from xau_bot.learning.models import Hypothesis, HypothesisStatus, ValidationMetrics
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-DISP-01",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        sample_size=35,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.001,
        z_score=3.5,
        observed_expectancy_r=-0.20,
        status=HypothesisStatus.PROPOSED,
        description="Test hyp",
        rationale="Test rationale",
    )

    baseline = ValidationMetrics(
        total_trades=100,
        win_count=60,
        loss_count=40,
        win_rate=0.60,
        expectancy_r=0.50,
        max_drawdown_pct=8.0,
        profit_factor=1.6,
    )
    candidate = ValidationMetrics(
        total_trades=85,
        win_count=58,
        loss_count=27,
        win_rate=0.68,
        expectancy_r=0.85,
        max_drawdown_pct=6.5,
        profit_factor=2.1,
    )

    validator = HypothesisValidator(min_trade_retention=0.70, max_drawdown_tolerance_ratio=1.10)
    result = validator.evaluate_metrics(hyp, baseline, candidate)

    assert result.verdict == HypothesisStatus.VALIDATED
    assert result.is_statistically_superior is True
    assert result.expectancy_delta_r == 0.35
    assert result.drawdown_delta_pct == -1.5
    assert not result.rejection_reason


def test_hypothesis_validator_rejects_low_trade_retention() -> None:
    from xau_bot.learning.models import Hypothesis, HypothesisStatus, ValidationMetrics
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-02",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="2.0",
        sample_size=35,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.001,
        z_score=3.5,
        observed_expectancy_r=-0.20,
        status=HypothesisStatus.PROPOSED,
        description="Overfilter",
        rationale="Overfilter",
    )

    baseline = ValidationMetrics(
        total_trades=100,
        win_count=60,
        loss_count=40,
        win_rate=0.60,
        expectancy_r=0.50,
        max_drawdown_pct=8.0,
        profit_factor=1.6,
    )
    candidate = ValidationMetrics(
        total_trades=50,  # 50% < 70% retention
        win_count=35,
        loss_count=15,
        win_rate=0.70,
        expectancy_r=0.90,
        max_drawdown_pct=5.0,
        profit_factor=2.2,
    )

    validator = HypothesisValidator(min_trade_retention=0.70)
    result = validator.evaluate_metrics(hyp, baseline, candidate)

    assert result.verdict == HypothesisStatus.REJECTED
    assert "retention" in result.rejection_reason.lower()


def test_hypothesis_validator_rejects_degraded_drawdown() -> None:
    from xau_bot.learning.models import Hypothesis, HypothesisStatus, ValidationMetrics
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-03",
        created_at=datetime.now(UTC),
        parameter_name="spread",
        current_value="0.30",
        proposed_value="0.25",
        sample_size=35,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.001,
        z_score=3.5,
        observed_expectancy_r=-0.20,
        status=HypothesisStatus.PROPOSED,
        description="DD spike",
        rationale="DD spike",
    )

    baseline = ValidationMetrics(
        total_trades=100,
        win_count=60,
        loss_count=40,
        win_rate=0.60,
        expectancy_r=0.50,
        max_drawdown_pct=5.0,
        profit_factor=1.6,
    )
    candidate = ValidationMetrics(
        total_trades=90,
        win_count=58,
        loss_count=32,
        win_rate=0.64,
        expectancy_r=0.60,
        max_drawdown_pct=7.5,  # 50% increase in drawdown, exceeds 10% tolerance
        profit_factor=1.7,
    )

    validator = HypothesisValidator(max_drawdown_tolerance_ratio=1.10)
    result = validator.evaluate_metrics(hyp, baseline, candidate)

    assert result.verdict == HypothesisStatus.REJECTED
    assert "drawdown" in result.rejection_reason.lower()


def test_hypothesis_validator_rejects_negative_expectancy_delta() -> None:
    from xau_bot.learning.models import Hypothesis, HypothesisStatus, ValidationMetrics
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-04",
        created_at=datetime.now(UTC),
        parameter_name="spread",
        current_value="0.30",
        proposed_value="0.25",
        sample_size=35,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.001,
        z_score=3.5,
        observed_expectancy_r=-0.20,
        status=HypothesisStatus.PROPOSED,
        description="Worse R",
        rationale="Worse R",
    )

    baseline = ValidationMetrics(
        total_trades=100,
        win_count=60,
        loss_count=40,
        win_rate=0.60,
        expectancy_r=0.80,
        max_drawdown_pct=5.0,
        profit_factor=1.8,
    )
    candidate = ValidationMetrics(
        total_trades=90,
        win_count=52,
        loss_count=38,
        win_rate=0.58,
        expectancy_r=0.65,  # Delta R is negative (-0.15)
        max_drawdown_pct=5.0,
        profit_factor=1.5,
    )

    validator = HypothesisValidator()
    result = validator.evaluate_metrics(hyp, baseline, candidate)

    assert result.verdict == HypothesisStatus.REJECTED
    assert "expectancy" in result.rejection_reason.lower()


def test_shadow_engine_evaluates_divergence() -> None:
    from xau_bot.domain.models import Direction
    from xau_bot.learning.shadow import ShadowDivergence, ShadowEngine

    shadow_engine = ShadowEngine()

    # Case 1: Both allow (ALIGNED)
    res_aligned = shadow_engine.evaluate_filter_decision(
        setup_id="s1",
        direction=Direction.BULLISH,
        active_allowed=True,
        shadow_allowed=True,
        active_reason="",
        shadow_reason="",
    )
    assert res_aligned.divergence == ShadowDivergence.ALIGNED
    assert res_aligned.is_divergent is False

    res_blocked = shadow_engine.evaluate_filter_decision(
        setup_id="s2",
        direction=Direction.BULLISH,
        active_allowed=True,
        shadow_allowed=False,
        active_reason="",
        shadow_reason="HIGH_SPREAD",
    )
    assert res_blocked.divergence == ShadowDivergence.SHADOW_BLOCKED
    assert res_blocked.is_divergent is True

    res_allowed = shadow_engine.evaluate_filter_decision(
        setup_id="s3",
        direction=Direction.BEARISH,
        active_allowed=False,
        shadow_allowed=True,
        active_reason="NEWS_BLACKOUT",
        shadow_reason="",
    )
    assert res_allowed.divergence == ShadowDivergence.SHADOW_ALLOWED
    assert res_allowed.is_divergent is True


def test_hypothesis_validator_init_validation() -> None:
    from xau_bot.learning.validator import HypothesisValidator

    with pytest.raises(ValueError, match="min_trade_retention"):
        HypothesisValidator(min_trade_retention=0.0)

    with pytest.raises(ValueError, match="min_trade_retention"):
        HypothesisValidator(min_trade_retention=1.5)

    with pytest.raises(ValueError, match="max_drawdown_tolerance_ratio"):
        HypothesisValidator(max_drawdown_tolerance_ratio=0.9)


def test_hypothesis_validator_metrics_from_backtest() -> None:
    from xau_bot.backtest.models import BacktestResult
    from xau_bot.learning.validator import HypothesisValidator

    bt_res = BacktestResult(
        trades=(),
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=11800.0,
        total_net_pnl=1800.0,
        total_trades=20,
        winning_trades=14,
        losing_trades=6,
        breakeven_trades=0,
        win_rate=0.70,
        profit_factor=2.1,
        max_drawdown_usd=550.0,
        max_drawdown_pct=5.5,
        sharpe_ratio=1.6,
        expectancy_r=0.45,
        average_trade_duration_seconds=3600.0,
    )
    metrics = HypothesisValidator.metrics_from_backtest(bt_res)
    assert metrics.total_trades == 20
    assert metrics.win_count == 14
    assert metrics.loss_count == 6
    assert metrics.win_rate == 0.70
    assert metrics.expectancy_r == 0.45
    assert metrics.max_drawdown_pct == 5.5
    assert metrics.profit_factor == 2.1


def test_hypothesis_validator_metrics_from_trades() -> None:
    from xau_bot.backtest.models import BacktestTrade, TradeExitReason
    from xau_bot.domain.models import Direction
    from xau_bot.learning.validator import HypothesisValidator

    empty_metrics = HypothesisValidator.metrics_from_trades([])
    assert empty_metrics.total_trades == 0
    assert empty_metrics.win_count == 0

    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    trades = [
        BacktestTrade(
            trade_id="t1",
            order_id="o1",
            direction=Direction.BULLISH,
            volume=1.0,
            entry_time=now,
            entry_price=2650.0,
            exit_time=now,
            exit_price=2660.0,
            exit_reason=TradeExitReason.TAKE_PROFIT,
            initial_sl=2645.0,
            final_sl=2645.0,
            tp=2660.0,
            commission=7.0,
            swap=0.0,
            gross_pnl=207.0,
            net_pnl=200.0,
            return_r=2.0,
            mfe_price=2660.0,
            mae_price=2649.0,
            mfe_r=2.0,
            mae_r=0.2,
            duration_seconds=3600.0,
        ),
        BacktestTrade(
            trade_id="t2",
            order_id="o2",
            direction=Direction.BEARISH,
            volume=1.0,
            entry_time=now,
            entry_price=2660.0,
            exit_time=now,
            exit_price=2665.0,
            exit_reason=TradeExitReason.STOP_LOSS,
            initial_sl=2665.0,
            final_sl=2665.0,
            tp=2650.0,
            commission=7.0,
            swap=0.0,
            gross_pnl=-93.0,
            net_pnl=-100.0,
            return_r=-1.0,
            mfe_price=2659.0,
            mae_price=2665.0,
            mfe_r=0.2,
            mae_r=1.0,
            duration_seconds=1800.0,
        ),
    ]

    metrics = HypothesisValidator.metrics_from_trades(trades, initial_balance=10000.0)
    assert metrics.total_trades == 2
    assert metrics.win_count == 1
    assert metrics.loss_count == 1
    assert metrics.win_rate == 0.5
    assert metrics.expectancy_r == 0.5
    assert metrics.profit_factor == 2.0


def test_hypothesis_validator_rejects_insignificant_p_value() -> None:
    from xau_bot.learning.models import Hypothesis
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-PVAL",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.42,
        p_value=0.15,
        z_score=1.0,
        observed_expectancy_r=0.10,
        status=HypothesisStatus.PROPOSED,
        description="Not significant",
        rationale="Not significant",
    )
    baseline = ValidationMetrics(
        total_trades=100,
        win_count=50,
        loss_count=50,
        win_rate=0.50,
        expectancy_r=0.20,
        max_drawdown_pct=5.0,
        profit_factor=1.2,
    )
    candidate = ValidationMetrics(
        total_trades=90,
        win_count=55,
        loss_count=35,
        win_rate=0.61,
        expectancy_r=0.35,
        max_drawdown_pct=4.8,
        profit_factor=1.5,
    )
    validator = HypothesisValidator(alpha=0.05)
    res = validator.evaluate_metrics(hyp, baseline, candidate)
    assert res.verdict == HypothesisStatus.REJECTED
    assert (
        "significance" in res.rejection_reason.lower() or "p-value" in res.rejection_reason.lower()
    )


def test_hypothesis_validator_drawdown_strict_border() -> None:
    from xau_bot.learning.models import Hypothesis
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-DD",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.01,
        z_score=2.8,
        observed_expectancy_r=0.20,
        status=HypothesisStatus.PROPOSED,
        description="DD test",
        rationale="DD test",
    )
    baseline = ValidationMetrics(
        total_trades=100,
        win_count=60,
        loss_count=40,
        win_rate=0.60,
        expectancy_r=0.40,
        max_drawdown_pct=0.50,
        profit_factor=1.5,
    )
    # 0.50 * 1.10 = 0.55. Candidate at 0.58 (> 10% relative degradation)
    candidate = ValidationMetrics(
        total_trades=90,
        win_count=60,
        loss_count=30,
        win_rate=0.66,
        expectancy_r=0.55,
        max_drawdown_pct=0.58,
        profit_factor=1.7,
    )
    validator = HypothesisValidator(max_drawdown_tolerance_ratio=1.10)
    res = validator.evaluate_metrics(hyp, baseline, candidate)
    assert res.verdict == HypothesisStatus.REJECTED
    assert "drawdown" in res.rejection_reason.lower()


def test_hypothesis_validator_rejects_degraded_oos() -> None:
    from xau_bot.learning.models import Hypothesis
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-OOS",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.01,
        z_score=2.8,
        observed_expectancy_r=0.20,
        status=HypothesisStatus.PROPOSED,
        description="OOS test",
        rationale="OOS test",
    )
    is_base = ValidationMetrics(100, 60, 40, 0.60, 0.40, 5.0, 1.5)
    is_cand = ValidationMetrics(90, 60, 30, 0.66, 0.60, 4.8, 1.8)
    oos_base = ValidationMetrics(50, 30, 20, 0.60, 0.40, 5.0, 1.5)
    oos_cand = ValidationMetrics(45, 20, 25, 0.44, 0.20, 6.0, 0.9)  # Degraded OOS

    validator = HypothesisValidator()
    res = validator.evaluate_metrics(
        hyp,
        is_base,
        is_cand,
        oos_baseline=oos_base,
        oos_candidate=oos_cand,
    )
    assert res.verdict == HypothesisStatus.REJECTED
    assert "out-of-sample" in res.rejection_reason.lower() or "oos" in res.rejection_reason.lower()


def test_shadow_engine_evaluate_setup_integration(tmp_path: Path) -> None:
    from datetime import timedelta

    from xau_bot.domain.models import (
        Candle,
        Direction,
        OrderBlock,
        OrderBlockType,
        Timeframe,
        TradeSetup,
    )
    from xau_bot.journal.repository import JournalRepository
    from xau_bot.learning.shadow import ShadowDivergence, ShadowEngine

    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    c1 = Candle(
        time=now - timedelta(minutes=10),
        open=2630.0,
        high=2635.0,
        low=2628.0,
        close=2634.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    ob = OrderBlock(
        time=c1.time,
        confirmed_time=c1.time + timedelta(minutes=5),
        type=OrderBlockType.BULLISH,
        high=2635.0,
        low=2628.0,
        open=2630.0,
        close=2634.0,
        candle=c1,
        displacement_atr_ratio=1.2,
    )
    setup = TradeSetup(
        id="set-01",
        direction=Direction.BULLISH,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        time=now - timedelta(minutes=5),
        confirmed_time=now,
        risk_reward_ratio=2.0,
        poi=ob,
    )
    engine = ShadowEngine()
    eval_res = engine.evaluate_setup(
        setup=setup,
        active_params={"min_displacement_multiplier": 1.0},
        shadow_params={"min_displacement_multiplier": 1.5},
    )
    assert eval_res.setup_id == "set-01"
    assert eval_res.active_allowed is True
    assert eval_res.shadow_allowed is False
    assert eval_res.divergence == ShadowDivergence.SHADOW_BLOCKED

    db_path = tmp_path / "shadow.db"
    repo = JournalRepository(str(db_path))
    try:
        engine.record_evaluation(repo, eval_res)
        records = repo.list_shadow_evaluations()
        assert len(records) == 1
        assert records[0]["setup_id"] == "set-01"
        assert records[0]["divergence"] == "SHADOW_BLOCKED"
    finally:
        repo.close()


def test_validator_empirical_with_backtest_engines() -> None:
    from unittest.mock import MagicMock

    from xau_bot.backtest.models import BacktestResult
    from xau_bot.domain.models import Tick
    from xau_bot.learning.models import Hypothesis, HypothesisStatus
    from xau_bot.learning.validator import HypothesisValidator

    hyp = Hypothesis(
        id="HYP-EMP",
        created_at=datetime.now(UTC),
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=40,
        condition_loss_rate=0.70,
        baseline_loss_rate=0.30,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=0.3,
        status=HypothesisStatus.PROPOSED,
        description="Empirical test",
        rationale="Empirical test",
    )

    t1 = Tick(time=datetime.now(UTC), bid=2000.0, ask=2000.2)
    ticks = [t1] * 10

    trades_base = tuple(
        MagicMock(return_r=r, net_pnl=r * 50.0)
        for r in [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    trades_cand = tuple(
        MagicMock(return_r=r, net_pnl=r * 50.0)
        for r in [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
    )

    mock_res_base = BacktestResult(
        trades=trades_base,
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=10200.0,
        total_net_pnl=200.0,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        breakeven_trades=0,
        win_rate=0.6,
        profit_factor=1.5,
        max_drawdown_usd=500.0,
        max_drawdown_pct=5.0,
        sharpe_ratio=1.2,
        expectancy_r=0.2,
        average_trade_duration_seconds=3600.0,
    )
    mock_res_cand = BacktestResult(
        trades=trades_cand,
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=10500.0,
        total_net_pnl=500.0,
        total_trades=9,
        winning_trades=7,
        losing_trades=2,
        breakeven_trades=0,
        win_rate=0.77,
        profit_factor=2.0,
        max_drawdown_usd=400.0,
        max_drawdown_pct=4.0,
        sharpe_ratio=1.8,
        expectancy_r=0.5,
        average_trade_duration_seconds=3600.0,
    )

    base_engine = MagicMock()
    base_engine.run.return_value = mock_res_base
    cand_engine = MagicMock()
    cand_engine.run.return_value = mock_res_cand

    validator = HypothesisValidator()
    res = validator.validate_empirical(hyp, base_engine, cand_engine, ticks=ticks)
    assert res.verdict == HypothesisStatus.VALIDATED

    res_oos = validator.validate_empirical(
        hyp, base_engine, cand_engine, ticks=ticks, oos_split_ratio=0.5
    )
    assert res_oos.verdict == HypothesisStatus.VALIDATED


def test_cli_validate_json_and_output_dir(tmp_path: Path) -> None:
    import json
    from io import StringIO

    from xau_bot.cli.main import main as cli_main
    from xau_bot.domain.models import Direction
    from xau_bot.journal.models import JournalEntry, OutcomeData, SnapshotData, TradeOutcome
    from xau_bot.journal.repository import JournalRepository

    db_file = tmp_path / "valid_json.db"
    out_dir = tmp_path / "reports_json"
    repo = JournalRepository(str(db_file))
    try:
        now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
        snap_disp_low = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=0.7,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=1.0,
        )
        snap_disp_high = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=1.5,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=1.0,
        )
        outcome_loss = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            exit_reason="SL",
            realized_r=-1.0,
            pnl_usd=-100.0,
        )
        outcome_win = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            exit_reason="TP",
            realized_r=2.0,
            pnl_usd=200.0,
        )

        for i in range(25):
            repo.save_entry(
                JournalEntry(
                    id=f"loss-{i}",
                    proposal_id=f"prop-l-{i}",
                    snapshot=snap_disp_low,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.LOSS,
                    outcome_data=outcome_loss,
                    created_at=now,
                )
            )
        for i in range(5):
            repo.save_entry(
                JournalEntry(
                    id=f"win-c-{i}",
                    proposal_id=f"prop-wc-{i}",
                    snapshot=snap_disp_low,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.WIN,
                    outcome_data=outcome_win,
                    created_at=now,
                )
            )
        for i in range(80):
            repo.save_entry(
                JournalEntry(
                    id=f"win-{i}",
                    proposal_id=f"prop-w-{i}",
                    snapshot=snap_disp_high,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.WIN,
                    outcome_data=outcome_win,
                    created_at=now,
                )
            )
    finally:
        repo.close()

    stdout = StringIO()
    stderr = StringIO()
    code = cli_main(
        [
            "validate",
            "--db",
            str(db_file),
            "--json",
            "--output-dir",
            str(out_dir),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    raw_json = stdout.getvalue()
    parsed = json.loads(raw_json)
    assert len(parsed) >= 1
    assert "validation_report_" in stderr.getvalue()
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1


def test_cli_validate_filter_spread_atr_ratio(tmp_path: Path) -> None:
    from xau_bot.cli.validate import _filter_entries_for_hypothesis
    from xau_bot.domain.models import Direction
    from xau_bot.journal.models import JournalEntry, OutcomeData, SnapshotData, TradeOutcome

    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    outcome = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_reason="TP",
        realized_r=1.0,
        pnl_usd=100.0,
    )
    snap1 = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.0,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=1.0,
    )
    snap2 = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.0,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.6,
        atr_m5=1.0,
    )
    snap3 = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.0,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=0.0,
    )

    e1 = JournalEntry(
        id="e1",
        proposal_id="p1",
        snapshot=snap1,
        direction=Direction.BULLISH,
        outcome=TradeOutcome.WIN,
        outcome_data=outcome,
        created_at=now,
    )
    e2 = JournalEntry(
        id="e2",
        proposal_id="p2",
        snapshot=snap2,
        direction=Direction.BULLISH,
        outcome=TradeOutcome.LOSS,
        outcome_data=outcome,
        created_at=now,
    )
    e3 = JournalEntry(
        id="e3",
        proposal_id="p3",
        snapshot=snap3,
        direction=Direction.BULLISH,
        outcome=TradeOutcome.WIN,
        outcome_data=outcome,
        created_at=now,
    )

    entries = [e1, e2, e3]
    filtered = _filter_entries_for_hypothesis(entries, "max_spread_atr_ratio", "0.3")
    assert len(filtered) == 2
    with pytest.raises(ValueError, match="Valor numérico inválido"):
        _filter_entries_for_hypothesis(entries, "max_spread_atr_ratio", "invalid")


def test_cli_validate_missing_candles_or_ticks_exit_1(tmp_path: Path) -> None:
    db_file = tmp_path / "test_journal.db"
    repo = JournalRepository(db_file)
    repo.close()

    stdout = StringIO()
    stderr = StringIO()
    code1 = cli_main(
        [
            "validate",
            "--db",
            str(db_file),
            "--candles",
            str(tmp_path / "missing_candles.csv"),
            "--ticks",
            str(tmp_path / "missing_ticks.csv"),
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code1 == 1
    assert "não encontrado" in stderr.getvalue()

    stderr2 = StringIO()
    code2 = cli_main(
        [
            "validate",
            "--db",
            str(db_file),
            "--candles",
            str(tmp_path / "missing_candles.csv"),
        ],
        stdout=StringIO(),
        stderr=stderr2,
    )
    assert code2 == 1


def test_hypothesis_validator_validate_real_backtest_with_oos_split() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup

    t0 = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    ticks = [
        Tick(time=t0 + timedelta(seconds=i * 5), bid=2600.0 + i * 0.1, ask=2600.2 + i * 0.1)
        for i in range(100)
    ]
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2600.0,
            high=2610.0,
            low=2595.0,
            close=2605.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(5)
    ]

    c0 = Candle(
        time=t0,
        open=2600.0,
        high=2605.0,
        low=2598.0,
        close=2603.0,
        volume=100.0,
        timeframe=Timeframe.M5,
    )
    ob = OrderBlock(
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        type=OrderBlockType.BULLISH,
        high=2605.0,
        low=2598.0,
        open=2600.0,
        close=2603.0,
        candle=c0,
        displacement_atr_ratio=1.5,
    )
    setup = TradeSetup(
        id="stp_real",
        direction=Direction.BULLISH,
        poi=ob,
        entry_price=2601.0,
        stop_loss=2597.0,
        take_profit=2610.0,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=2.25,
    )

    validator = HypothesisValidator()
    hyp = Hypothesis(
        id="HYP-REAL-1",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        sample_size=100,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=0.5,
        status=HypothesisStatus.PROPOSED,
        description="Min displacement test",
        rationale="Improve expectancy",
    )

    with pytest.raises(ValueError, match="oos_split_ratio must be strictly between 0.0 and 1.0"):
        validator.validate(hyp, ticks=ticks, candles=candles, setups=[setup], oos_split_ratio=1.5)

    with pytest.raises(ValueError, match="oos_split_ratio must be strictly between 0.0 and 1.0"):
        validator.validate(hyp, ticks=ticks, candles=candles, setups=[setup], oos_split_ratio=-0.1)

    result = validator.validate(
        hyp,
        ticks=ticks,
        candles=candles,
        setups=[setup],
        oos_split_ratio=0.30,
    )
    assert isinstance(result, ValidationResult)
    assert result.hypothesis_id == "HYP-REAL-1"
    assert result.baseline_metrics.total_trades >= 0
    assert result.candidate_metrics.total_trades >= 0


def test_hypothesis_validator_empirical_oos_gain_and_rejection() -> None:
    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    validator = HypothesisValidator()
    hyp = Hypothesis(
        id="HYP-OOS-TEST",
        created_at=now,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=100,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=0.5,
        status=HypothesisStatus.PROPOSED,
        description="OOS test",
        rationale="OOS gain verification",
    )
    is_base = ValidationMetrics(
        total_trades=100,
        win_count=50,
        loss_count=50,
        win_rate=0.50,
        profit_factor=1.5,
        expectancy_r=0.2,
        max_drawdown_pct=5.0,
    )
    is_cand = ValidationMetrics(
        total_trades=80,
        win_count=48,
        loss_count=32,
        win_rate=0.60,
        profit_factor=2.0,
        expectancy_r=0.5,
        max_drawdown_pct=4.0,
    )
    oos_base = ValidationMetrics(
        total_trades=50,
        win_count=25,
        loss_count=25,
        win_rate=0.50,
        profit_factor=1.5,
        expectancy_r=0.3,
        max_drawdown_pct=4.0,
    )
    oos_cand_degraded = ValidationMetrics(
        total_trades=40,
        win_count=18,
        loss_count=22,
        win_rate=0.45,
        profit_factor=1.2,
        expectancy_r=0.1,
        max_drawdown_pct=6.0,
    )
    res_rejected = validator.evaluate_metrics(
        hypothesis=hyp,
        baseline=is_base,
        candidate=is_cand,
        oos_baseline=oos_base,
        oos_candidate=oos_cand_degraded,
    )
    assert res_rejected.verdict == HypothesisStatus.REJECTED
    assert "Out-of-sample (OOS) expectancy delta" in res_rejected.rejection_reason

    oos_cand_improved = ValidationMetrics(
        total_trades=40,
        win_count=22,
        loss_count=18,
        win_rate=0.55,
        profit_factor=1.8,
        expectancy_r=0.45,
        max_drawdown_pct=3.5,
    )
    res_validated = validator.evaluate_metrics(
        hypothesis=hyp,
        baseline=is_base,
        candidate=is_cand,
        oos_baseline=oos_base,
        oos_candidate=oos_cand_improved,
    )
    assert res_validated.verdict == HypothesisStatus.VALIDATED
    assert not res_validated.rejection_reason


def test_hypothesis_validator_backtest_causal_full_cycle_validated() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2602.0,
            high=2608.0,
            low=2596.0,
            close=2603.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(45)
    ]

    t_is_1 = t0 + timedelta(minutes=70)
    t_is_2 = t0 + timedelta(minutes=85)
    t_is_3 = t0 + timedelta(minutes=100)
    t_is_4 = t0 + timedelta(minutes=115)
    t_oos_1 = t0 + timedelta(minutes=150)
    t_oos_2 = t0 + timedelta(minutes=165)

    def _make_setup(st_id: str, t_time: datetime, c_idx: int, disp: float) -> TradeSetup:
        ob = OrderBlock(
            time=t_time - timedelta(minutes=5),
            confirmed_time=t_time,
            type=OrderBlockType.BULLISH,
            high=candles[c_idx].high,
            low=candles[c_idx].low,
            open=candles[c_idx].open,
            close=candles[c_idx].close,
            candle=candles[c_idx],
            displacement_atr_ratio=disp,
        )
        return TradeSetup(
            id=st_id,
            direction=Direction.BULLISH,
            poi=ob,
            entry_price=2600.0,
            stop_loss=2595.0,
            take_profit=2610.0,
            time=t_time - timedelta(minutes=5),
            confirmed_time=t_time,
            risk_reward_ratio=2.0,
        )

    setup_is_win_1 = _make_setup("stp_is_w1", t_is_1, 13, 1.5)
    setup_is_win_2 = _make_setup("stp_is_w2", t_is_2, 16, 1.5)
    setup_is_win_3 = _make_setup("stp_is_w3", t_is_3, 19, 1.5)
    setup_is_loss = _make_setup("stp_is_l1", t_is_4, 22, 1.0)

    setup_oos_win = _make_setup("stp_oos_w1", t_oos_1, 29, 1.5)
    setup_oos_loss = _make_setup("stp_oos_l1", t_oos_2, 32, 1.0)

    ticks: list[Tick] = []
    ticks.append(Tick(time=t_is_1, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_is_1 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_is_1 + timedelta(seconds=20), bid=2610.2, ask=2610.4))

    ticks.append(Tick(time=t_is_2, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_is_2 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_is_2 + timedelta(seconds=20), bid=2610.2, ask=2610.4))

    ticks.append(Tick(time=t_is_3, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_is_3 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_is_3 + timedelta(seconds=20), bid=2610.2, ask=2610.4))

    ticks.append(Tick(time=t_is_4, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_is_4 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_is_4 + timedelta(seconds=20), bid=2594.0, ask=2594.2))

    ticks.append(Tick(time=t_oos_1, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_oos_1 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_oos_1 + timedelta(seconds=20), bid=2610.2, ask=2610.4))

    ticks.append(Tick(time=t_oos_2, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_oos_2 + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_oos_2 + timedelta(seconds=20), bid=2594.0, ask=2594.2))

    hyp = Hypothesis(
        id="HYP-CAUSAL-WIN",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Causal backtest win",
        rationale="Expectancy gain",
    )

    validator = HypothesisValidator(
        min_trade_retention=0.70, min_expectancy_delta_r=0.0, alpha=0.50
    )
    res = validator.validate(
        hypothesis=hyp,
        candles=candles,
        ticks=ticks,
        setups=[
            setup_is_win_1,
            setup_is_win_2,
            setup_is_win_3,
            setup_is_loss,
            setup_oos_win,
            setup_oos_loss,
        ],
        oos_split_ratio=0.35,
    )
    assert isinstance(res, ValidationResult)
    assert res.baseline_metrics.total_trades == 4
    assert res.candidate_metrics.total_trades == 3
    assert res.trade_retention_rate == 0.75
    assert res.expectancy_delta_r == 0.51
    assert res.is_statistically_superior is True
    assert res.p_value_improvement == 0.396066
    assert res.verdict == HypothesisStatus.VALIDATED


def test_hypothesis_validator_backtest_causal_oos_rejection() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2602.0,
            high=2608.0,
            low=2596.0,
            close=2603.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]

    t_is = t0 + timedelta(minutes=30)
    t_oos = t0 + timedelta(minutes=90)

    ob_is = OrderBlock(
        time=t_is - timedelta(minutes=5),
        confirmed_time=t_is,
        type=OrderBlockType.BULLISH,
        high=candles[5].high,
        low=candles[5].low,
        open=candles[5].open,
        close=candles[5].close,
        candle=candles[5],
        displacement_atr_ratio=1.5,
    )
    setup_is = TradeSetup(
        id="stp_is_win",
        direction=Direction.BULLISH,
        poi=ob_is,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t_is - timedelta(minutes=5),
        confirmed_time=t_is,
        risk_reward_ratio=2.0,
    )

    ob_oos = OrderBlock(
        time=t_oos - timedelta(minutes=5),
        confirmed_time=t_oos,
        type=OrderBlockType.BULLISH,
        high=candles[17].high,
        low=candles[17].low,
        open=candles[17].open,
        close=candles[17].close,
        candle=candles[17],
        displacement_atr_ratio=1.5,
    )
    setup_oos = TradeSetup(
        id="stp_oos_loss",
        direction=Direction.BULLISH,
        poi=ob_oos,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t_oos - timedelta(minutes=5),
        confirmed_time=t_oos,
        risk_reward_ratio=2.0,
    )

    ticks: list[Tick] = []
    ticks.append(Tick(time=t_is, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_is + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_is + timedelta(seconds=20), bid=2610.2, ask=2610.4))

    ticks.append(Tick(time=t_oos, bid=2604.0, ask=2604.2))
    ticks.append(Tick(time=t_oos + timedelta(seconds=10), bid=2599.8, ask=2600.0))
    ticks.append(Tick(time=t_oos + timedelta(seconds=20), bid=2594.0, ask=2594.2))

    hyp = Hypothesis(
        id="HYP-CAUSAL-REJ",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Causal backtest oos loss",
        rationale="Degradation test",
    )

    validator = HypothesisValidator()
    res = validator.validate(
        hypothesis=hyp,
        candles=candles,
        ticks=ticks,
        setups=[setup_is, setup_oos],
        oos_split_ratio=0.40,
    )
    assert res.verdict == HypothesisStatus.REJECTED
    assert "out-of-sample (oos) expectancy delta" in res.rejection_reason.lower()


def test_hypothesis_validator_strict_exceptions() -> None:
    from xau_bot.learning.validator import HypothesisValidator

    validator = HypothesisValidator()
    now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

    hyp_ok = Hypothesis(
        id="H-1",
        created_at=now,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.2",
        sample_size=30,
        condition_loss_rate=0.4,
        baseline_loss_rate=0.5,
        p_value=0.01,
        z_score=2.0,
        observed_expectancy_r=0.5,
        status=HypothesisStatus.PROPOSED,
        description="test",
        rationale="test",
    )

    with pytest.raises(ValueError, match="at least 2 ticks"):
        validator.validate(hyp_ok, ticks=[])

    with pytest.raises(ValueError, match="at least 2 ticks"):
        validator.validate(
            hyp_ok,
            ticks=[Tick(time=now, bid=2600.0, ask=2600.2)],
        )

    hyp_unsupported = Hypothesis(
        id="H-2",
        created_at=now,
        parameter_name="unknown_param",
        current_value="1.0",
        proposed_value="2.0",
        sample_size=30,
        condition_loss_rate=0.4,
        baseline_loss_rate=0.5,
        p_value=0.01,
        z_score=2.0,
        observed_expectancy_r=0.5,
        status=HypothesisStatus.PROPOSED,
        description="test",
        rationale="test",
    )
    dummy_ticks = [
        Tick(time=now, bid=2600.0, ask=2600.2),
        Tick(time=now + timedelta(seconds=1), bid=2600.1, ask=2600.3),
    ]
    with pytest.raises(ValueError, match="não suportado"):
        validator.validate(hyp_unsupported, ticks=dummy_ticks)

    hyp_invalid_val = Hypothesis(
        id="H-3",
        created_at=now,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="not_a_number",
        sample_size=30,
        condition_loss_rate=0.4,
        baseline_loss_rate=0.5,
        p_value=0.01,
        z_score=2.0,
        observed_expectancy_r=0.5,
        status=HypothesisStatus.PROPOSED,
        description="test",
        rationale="test",
    )
    with pytest.raises(ValueError, match="Valores num[eé]ricos inv[aá]lidos"):
        validator.validate(hyp_invalid_val, ticks=dummy_ticks)


def test_backtest_engine_spread_filter_with_atr() -> None:
    from xau_bot.backtest.engine import BacktestEngine
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.filters.composite import CompositeTradeFilter
    from xau_bot.filters.models import SpreadFilterConfig
    from xau_bot.filters.spread import SpreadFilter

    t0 = datetime(2026, 10, 8, 7, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2600.0,
            high=2604.0,
            low=2596.0,
            close=2602.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]

    t_stp = t0 + timedelta(minutes=16 * 5)
    ob = OrderBlock(
        time=t_stp - timedelta(minutes=5),
        confirmed_time=t_stp,
        type=OrderBlockType.BULLISH,
        high=candles[15].high,
        low=candles[15].low,
        open=candles[15].open,
        close=candles[15].close,
        candle=candles[15],
        displacement_atr_ratio=1.5,
    )
    setup = TradeSetup(
        id="stp_filter_test",
        direction=Direction.BULLISH,
        poi=ob,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t_stp - timedelta(minutes=5),
        confirmed_time=t_stp,
        risk_reward_ratio=2.0,
    )

    filter_lenient = CompositeTradeFilter(
        spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=0.20)),
        session_filter=None,
    )
    engine_lenient = BacktestEngine(trade_filter=filter_lenient)

    ticks = [
        Tick(time=t_stp, bid=2604.0, ask=2604.3),
        Tick(time=t_stp + timedelta(seconds=10), bid=2599.7, ask=2600.0),
        Tick(time=t_stp + timedelta(seconds=20), bid=2610.2, ask=2610.5),
    ]

    res_lenient = engine_lenient.run(ticks=ticks, setups=[setup], candles=candles)
    assert res_lenient.total_trades == 1

    filter_strict = CompositeTradeFilter(
        spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=0.01)),
        session_filter=None,
    )
    engine_strict = BacktestEngine(trade_filter=filter_strict)
    res_strict = engine_strict.run(ticks=ticks, setups=[setup], candles=candles)
    assert res_strict.total_trades == 0


@pytest.mark.asyncio
async def test_live_trading_engine_shadow_mode_integration(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from xau_bot.advisor.modes import AdvisorDispatcher, ExecutionMode, ProposalManager
    from xau_bot.broker.models import AccountInfo, SymbolInfo
    from xau_bot.domain.models import Candle, SetupStatus, Timeframe, TradeSetup
    from xau_bot.journal.repository import JournalRepository
    from xau_bot.learning.shadow import ShadowEngine
    from xau_bot.runtime.live_loop import LiveTradingEngine
    from xau_bot.runtime.models import CandleCloseEvent

    db_path = tmp_path / "journal_shadow.db"
    repo = JournalRepository(db_path)

    broker = MagicMock()
    broker.get_candles = AsyncMock(return_value=())
    broker.get_account_info = AsyncMock(
        return_value=AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=100.0,
            leverage=100.0,
        )
    )
    broker.get_symbol_info = AsyncMock(
        return_value=SymbolInfo(name="XAUUSD", bid=2650.0, ask=2650.25, spread_points=0.25)
    )
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
    )
    shadow_engine = ShadowEngine()
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        journal_repo=repo,
        shadow_engine=shadow_engine,
        shadow_params={"max_spread_atr_ratio": 0.05},
    )

    t0 = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2650.0,
            high=2655.0,
            low=2645.0,
            close=2652.0,
            volume=100.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_shadow_live",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    evals = repo.list_shadow_evaluations()
    assert len(evals) == 1
    assert evals[0]["setup_id"] == "setup_shadow_live"
    repo.close()


def test_welch_t_test_edge_cases_and_branches() -> None:
    from xau_bot.learning.stats import welch_t_test

    assert welch_t_test([1.0], [2.0, 3.0]) == (0.0, 1.0)
    assert welch_t_test([1.0, 2.0], [3.0]) == (0.0, 1.0)
    assert welch_t_test([2.0, 2.0], [2.0, 2.0]) == (0.0, 1.0)
    assert welch_t_test([1.0, 1.0], [2.0, 2.0]) == (0.0, 0.0)

    t_stat, p_val = welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert t_stat == 0.0
    assert p_val == 1.0

    t_small, p_small = welch_t_test([1.0, 2.0], [10.0, 20.0])
    assert t_small != 0.0
    assert 0.0 <= p_small <= 1.0

    t_stat, p_val = welch_t_test([1.0, 1.1, 0.9, 1.2, 0.8], [5.0, 5.2, 4.8, 5.1, 4.9])
    assert t_stat > 10.0
    assert p_val < 0.001


def test_hypothesis_validator_validate_spread_atr_ratio_branch() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2600.0,
            high=2605.0,
            low=2595.0,
            close=2601.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    t_stp = t0 + timedelta(minutes=60)
    ob = OrderBlock(
        time=t_stp - timedelta(minutes=5),
        confirmed_time=t_stp,
        type=OrderBlockType.BULLISH,
        high=candles[12].high,
        low=candles[12].low,
        open=candles[12].open,
        close=candles[12].close,
        candle=candles[12],
        displacement_atr_ratio=1.5,
    )
    setup = TradeSetup(
        id="stp_spread_branch",
        direction=Direction.BULLISH,
        poi=ob,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t_stp - timedelta(minutes=5),
        confirmed_time=t_stp,
        risk_reward_ratio=2.0,
    )
    ticks = [
        Tick(time=t_stp, bid=2604.0, ask=2604.2),
        Tick(time=t_stp + timedelta(seconds=10), bid=2599.8, ask=2600.0),
        Tick(time=t_stp + timedelta(seconds=20), bid=2610.2, ask=2610.4),
    ]
    hyp = Hypothesis(
        id="HYP-SPREAD",
        created_at=t0,
        parameter_name="max_spread_atr_ratio",
        current_value="0.30",
        proposed_value="0.25",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Spread filter test",
        rationale="Spread test",
    )
    validator = HypothesisValidator()
    res = validator.validate(
        hypothesis=hyp,
        candles=candles,
        ticks=ticks,
        setups=[setup],
    )
    assert isinstance(res, ValidationResult)
    assert res.parameter_name == "max_spread_atr_ratio"
    assert res.verdict in (HypothesisStatus.VALIDATED, HypothesisStatus.REJECTED)
    assert isinstance(res.expectancy_delta_r, float)
    assert isinstance(res.baseline_metrics.win_rate, float)
    assert isinstance(res.candidate_metrics.win_rate, float)
    assert isinstance(res.drawdown_delta_pct, float)


def test_hypothesis_validator_displacement_without_poi_raises() -> None:
    from xau_bot.domain.models import TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    setup_no_poi = TradeSetup(
        id="stp_no_poi",
        direction=Direction.BULLISH,
        poi=None,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t0,
        confirmed_time=t0,
        risk_reward_ratio=2.0,
    )
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    hyp = Hypothesis(
        id="HYP-NO-POI",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Test without POI",
        rationale="Test",
    )
    validator = HypothesisValidator()
    with pytest.raises(ValueError, match="sem displacement_atr_ratio"):
        validator.validate(hyp, ticks=ticks, setups=[setup_no_poi])


def test_hypothesis_validator_displacement_zero_ratio_raises_fail_closed() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    c = Candle(
        time=t0,
        open=2600.0,
        high=2605.0,
        low=2595.0,
        close=2602.0,
        volume=10.0,
        timeframe=Timeframe.M5,
    )
    ob_zero = OrderBlock(
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        type=OrderBlockType.BULLISH,
        high=2605.0,
        low=2595.0,
        open=2600.0,
        close=2602.0,
        candle=c,
        displacement_atr_ratio=0.0,
    )
    setup = TradeSetup(
        id="stp_zero_ratio",
        direction=Direction.BULLISH,
        poi=ob_zero,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=2.0,
    )
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    hyp = Hypothesis(
        id="HYP-ZERO-RATIO",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Zero ratio test",
        rationale="Fail closed",
    )
    validator = HypothesisValidator()
    with pytest.raises(ValueError, match="sem displacement_atr_ratio mensurável"):
        validator.validate(hyp, ticks=ticks, setups=[setup])


def test_hypothesis_validator_displacement_preserves_non_ob_setups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xau_bot.domain.models import (
        FVG,
        Candle,
        FVGType,
        OrderBlock,
        OrderBlockType,
        Timeframe,
        TradeSetup,
    )
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    c0 = Candle(
        time=t0,
        open=2598.0,
        high=2600.0,
        low=2595.0,
        close=2599.0,
        volume=10.0,
        timeframe=Timeframe.M5,
    )
    c1 = Candle(
        time=t0 + timedelta(minutes=5),
        open=2600.0,
        high=2608.0,
        low=2599.0,
        close=2607.0,
        volume=20.0,
        timeframe=Timeframe.M5,
    )
    c2 = Candle(
        time=t0 + timedelta(minutes=10),
        open=2607.0,
        high=2612.0,
        low=2605.0,
        close=2610.0,
        volume=15.0,
        timeframe=Timeframe.M5,
    )
    fvg = FVG(
        time=t0 + timedelta(minutes=10),
        confirmed_time=t0 + timedelta(minutes=15),
        type=FVGType.BULLISH,
        top=2605.0,
        bottom=2600.0,
        candle_0=c0,
        candle_1=c1,
        candle_2=c2,
    )
    setup_fvg = TradeSetup(
        id="stp_fvg_preserved",
        direction=Direction.BULLISH,
        poi=fvg,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t0 + timedelta(minutes=10),
        confirmed_time=t0 + timedelta(minutes=15),
        risk_reward_ratio=2.0,
    )
    ob = OrderBlock(
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        type=OrderBlockType.BULLISH,
        high=c0.high,
        low=c0.low,
        open=c0.open,
        close=c0.close,
        candle=c0,
        displacement_atr_ratio=1.2,
    )
    setup_ob = TradeSetup(
        id="stp_ob_filtered",
        direction=Direction.BULLISH,
        poi=ob,
        entry_price=2600.0,
        stop_loss=2595.0,
        take_profit=2610.0,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=2.0,
    )
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    hyp = Hypothesis(
        id="HYP-PRESERVE-FVG",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Preserve FVG setups",
        rationale="Semantic test",
    )
    validator = HypothesisValidator()
    captured_active: list[Any] = []
    captured_candidate: list[Any] = []
    orig_validate_empirical = validator.validate_empirical

    def spy_validate_empirical(**kwargs: Any) -> ValidationResult:
        nonlocal captured_active, captured_candidate
        captured_active = list(kwargs.get("setups", []))
        captured_candidate = list(kwargs.get("candidate_setups", []))
        return orig_validate_empirical(**kwargs)

    monkeypatch.setattr(validator, "validate_empirical", spy_validate_empirical)
    res = validator.validate(hyp, ticks=ticks, setups=[setup_fvg, setup_ob])
    assert isinstance(res, ValidationResult)
    assert [s.id for s in captured_active] == ["stp_fvg_preserved", "stp_ob_filtered"]
    assert [s.id for s in captured_candidate] == ["stp_fvg_preserved"]


def test_hypothesis_validator_validate_fallback_detect_setups() -> None:
    from xau_bot.domain.models import Candle, Timeframe
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2600.0,
            high=2605.0,
            low=2595.0,
            close=2601.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    hyp = Hypothesis(
        id="HYP-FALLBACK",
        created_at=t0,
        parameter_name="max_spread_atr_ratio",
        current_value="0.30",
        proposed_value="0.25",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Fallback detect setups",
        rationale="Test",
    )
    validator = HypothesisValidator()
    res = validator.validate(
        hypothesis=hyp,
        candles=candles,
        ticks=ticks,
        setups=None,
    )
    assert isinstance(res, ValidationResult)
    assert res.verdict in (HypothesisStatus.VALIDATED, HypothesisStatus.REJECTED)
    assert isinstance(res.trade_retention_rate, float)
    assert isinstance(res.expectancy_delta_r, float)
    assert isinstance(res.drawdown_delta_pct, float)


def test_cli_validate_end_to_end_backtest_execution(tmp_path: Path) -> None:
    import json
    from io import StringIO

    from xau_bot.cli.main import main as cli_main
    from xau_bot.domain.models import Direction
    from xau_bot.journal.models import JournalEntry, OutcomeData, SnapshotData, TradeOutcome
    from xau_bot.journal.repository import JournalRepository

    db_file = tmp_path / "valid_bt.db"
    repo = JournalRepository(str(db_file))
    try:
        now = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
        snap = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=0.7,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=1.0,
        )
        out_loss = OutcomeData(
            executed=True,
            entry_price=2630.0,
            stop_loss=2625.0,
            take_profit=2640.0,
            exit_reason="SL",
            realized_r=-1.0,
            pnl_usd=-100.0,
        )
        for i in range(25):
            repo.save_entry(
                JournalEntry(
                    id=f"loss-bt-{i}",
                    proposal_id=f"prop-bt-{i}",
                    snapshot=snap,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.LOSS,
                    outcome_data=out_loss,
                    created_at=now,
                )
            )
        for i in range(5):
            repo.save_entry(
                JournalEntry(
                    id=f"win-bt-{i}",
                    proposal_id=f"prop-bt-w-{i}",
                    snapshot=snap,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.WIN,
                    outcome_data=OutcomeData(
                        executed=True,
                        entry_price=2630.0,
                        stop_loss=2625.0,
                        take_profit=2640.0,
                        exit_reason="TP",
                        realized_r=2.0,
                        pnl_usd=200.0,
                    ),
                    created_at=now,
                )
            )
        snap_high = SnapshotData(
            timestamp=now,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0,
            swept_liquidity=True,
            displacement_atr=1.5,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=1.0,
        )
        for i in range(80):
            repo.save_entry(
                JournalEntry(
                    id=f"win-bt-high-{i}",
                    proposal_id=f"prop-bt-wh-{i}",
                    snapshot=snap_high,
                    direction=Direction.BULLISH,
                    outcome=TradeOutcome.WIN,
                    outcome_data=OutcomeData(
                        executed=True,
                        entry_price=2630.0,
                        stop_loss=2625.0,
                        take_profit=2640.0,
                        exit_reason="TP",
                        realized_r=2.0,
                        pnl_usd=200.0,
                    ),
                    created_at=now,
                )
            )
    finally:
        repo.close()

    candles_file = tmp_path / "candles.csv"
    with open(candles_file, "w", encoding="utf-8") as f:
        f.write("time,open,high,low,close,volume\n")
        for i in range(20):
            t_str = (now + timedelta(minutes=i * 5)).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{t_str},2600.0,2605.0,2595.0,2602.0,100\n")

    ticks_file = tmp_path / "ticks.csv"
    with open(ticks_file, "w", encoding="utf-8") as f:
        f.write("time,bid,ask\n")
        for i in range(5):
            t_str = (now + timedelta(minutes=i * 5)).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{t_str},2600.0,2600.2\n")

    setups_file = tmp_path / "setups.json"
    with open(setups_file, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": "setup_cli_1",
                    "direction": "BULLISH",
                    "entry_price": 2600.0,
                    "stop_loss": 2595.0,
                    "take_profit": 2610.0,
                    "time": now.isoformat(),
                    "confirmed_time": (now + timedelta(minutes=5)).isoformat(),
                    "risk_reward_ratio": 2.0,
                    "displacement_atr_ratio": 1.5,
                }
            ],
            f,
        )

    stdout = StringIO()
    stderr = StringIO()
    code = cli_main(
        [
            "validate",
            "--db",
            str(db_file),
            "--candles",
            str(candles_file),
            "--ticks",
            str(ticks_file),
            "--setups",
            str(setups_file),
            "--json",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    raw_json = stdout.getvalue()
    parsed = json.loads(raw_json)
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert parsed[0]["parameter_name"] == "min_displacement_multiplier"
    assert parsed[0]["verdict"] in ("VALIDATED", "REJECTED")
    if parsed[0]["verdict"] == "REJECTED":
        assert len(parsed[0]["rejection_reason"]) > 0

    import sqlite3

    conn = sqlite3.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT hypothesis_id, parameter_name, verdict, rejection_reason "
            "FROM validation_results WHERE parameter_name = 'min_displacement_multiplier';"
        ).fetchone()
        assert row is not None
        assert row[0] == parsed[0]["hypothesis_id"]
        assert row[1] == "min_displacement_multiplier"
        assert row[2] == parsed[0]["verdict"]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_live_trading_engine_shadow_divergence_blocked_active(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from xau_bot.advisor.modes import AdvisorDispatcher, ExecutionMode, ProposalManager
    from xau_bot.broker.models import AccountInfo, SymbolInfo
    from xau_bot.domain.models import Candle, SetupStatus, Timeframe, TradeSetup
    from xau_bot.filters.composite import CompositeTradeFilter
    from xau_bot.filters.models import SpreadFilterConfig
    from xau_bot.filters.spread import SpreadFilter
    from xau_bot.journal.repository import JournalRepository
    from xau_bot.learning.shadow import ShadowEngine
    from xau_bot.runtime.live_loop import LiveTradingEngine
    from xau_bot.runtime.models import CandleCloseEvent

    db_path = tmp_path / "journal_div.db"
    repo = JournalRepository(db_path)

    broker = MagicMock()
    broker.get_candles = AsyncMock(return_value=())
    broker.get_account_info = AsyncMock(
        return_value=AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=100.0,
            leverage=100.0,
        )
    )
    broker.get_symbol_info = AsyncMock(
        return_value=SymbolInfo(
            name="XAUUSD",
            bid=2650.0,
            ask=2651.0,
            spread_points=100.0,
            point_size=0.01,
        )
    )
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
    )
    active_filter = CompositeTradeFilter(
        spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=0.05)),
        session_filter=None,
    )
    shadow_engine = ShadowEngine()
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        journal_repo=repo,
        composite_filter=active_filter,
        shadow_engine=shadow_engine,
        shadow_params={"max_spread_atr_ratio": 0.50},
    )

    t0 = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2650.0,
            high=2655.0,
            low=2645.0,
            close=2652.0,
            volume=100.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_div_test",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    evals = repo.list_shadow_evaluations()
    assert len(evals) == 1
    ev = evals[0]
    assert ev["setup_id"] == "setup_div_test"
    assert ev["active_allowed"] is False
    assert ev["shadow_allowed"] is True
    assert ev["divergence"] == "SHADOW_ALLOWED"
    assert ev["is_divergent"] is True
    assert "composite_filter" in ev["active_reason"]
    broker.place_order.assert_not_called()
    repo.close()


@pytest.mark.asyncio
async def test_live_trading_engine_shadow_aligned_when_session_blocks(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from xau_bot.advisor.modes import AdvisorDispatcher, ExecutionMode, ProposalManager
    from xau_bot.broker.models import AccountInfo, SymbolInfo
    from xau_bot.domain.models import Candle, SetupStatus, Timeframe, TradeSetup
    from xau_bot.filters.composite import CompositeTradeFilter
    from xau_bot.filters.models import SpreadFilterConfig
    from xau_bot.filters.sessions import SessionFilter
    from xau_bot.filters.spread import SpreadFilter
    from xau_bot.journal.repository import JournalRepository
    from xau_bot.learning.shadow import ShadowEngine
    from xau_bot.runtime.live_loop import LiveTradingEngine
    from xau_bot.runtime.models import CandleCloseEvent

    db_path = tmp_path / "journal_session_block.db"
    repo = JournalRepository(db_path)

    broker = MagicMock()
    broker.get_candles = AsyncMock(return_value=())
    broker.get_account_info = AsyncMock(
        return_value=AccountInfo(
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=100.0,
            leverage=100.0,
        )
    )
    broker.get_symbol_info = AsyncMock(
        return_value=SymbolInfo(
            name="XAUUSD",
            bid=2650.0,
            ask=2650.2,
            spread_points=20.0,
            point_size=0.01,
        )
    )
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
    )
    active_filter = CompositeTradeFilter(
        spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=0.50)),
        session_filter=SessionFilter(),
    )
    shadow_engine = ShadowEngine()
    engine = LiveTradingEngine(
        broker=broker,
        proposal_manager=proposal_manager,
        advisor_dispatcher=dispatcher,
        journal_repo=repo,
        composite_filter=active_filter,
        shadow_engine=shadow_engine,
        shadow_params={"max_spread_atr_ratio": 0.60},
    )

    t_out_of_session = datetime(2026, 10, 8, 17, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t_out_of_session + timedelta(minutes=i * 5),
            open=2650.0,
            high=2655.0,
            low=2645.0,
            close=2652.0,
            volume=100.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    engine.candle_loop.feed_candles(Timeframe.M5, candles, has_forming_candle=False)

    fake_setup = TradeSetup(
        id="setup_session_block",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2665.0,
        time=t_out_of_session,
        confirmed_time=t_out_of_session + timedelta(minutes=5),
        risk_reward_ratio=3.0,
        status=SetupStatus.PENDING,
        timeframe=Timeframe.M5,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "xau_bot.runtime.live_loop.detect_setups",
            lambda **kwargs: [fake_setup],
        )

        event = CandleCloseEvent(
            timeframe=Timeframe.M5,
            candle=candles[-1],
            closed_series=tuple(candles),
            forming_candle=None,
            closed_at=candles[-1].time + timedelta(minutes=5),
        )
        await engine.on_candle_close(event)

    evals = repo.list_shadow_evaluations()
    assert len(evals) == 1
    ev = evals[0]
    assert ev["setup_id"] == "setup_session_block"
    assert ev["active_allowed"] is False
    assert ev["shadow_allowed"] is False
    assert ev["divergence"] == "ALIGNED"
    assert ev["is_divergent"] is False
    assert "composite_filter" in ev["active_reason"]
    assert "composite_filter" in ev["shadow_reason"]
    broker.place_order.assert_not_called()
    repo.close()


def test_hypothesis_validator_causal_full_cycle_default_alpha_validated() -> None:
    from xau_bot.domain.models import Candle, OrderBlock, OrderBlockType, Timeframe, TradeSetup
    from xau_bot.learning.validator import HypothesisValidator

    t0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2602.0,
            high=2608.0,
            low=2596.0,
            close=2603.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(100)
    ]

    setups: list[TradeSetup] = []
    ticks: list[Tick] = []
    loss_indices = {3, 7, 11, 15, 23}

    for i in range(25):
        c = candles[i]
        t_trade = c.time + timedelta(minutes=5)
        is_loss = i in loss_indices
        disp = 1.0 if is_loss else 1.5
        ob = OrderBlock(
            time=c.time,
            confirmed_time=t_trade,
            type=OrderBlockType.BULLISH,
            high=c.high,
            low=c.low,
            open=c.open,
            close=c.close,
            candle=c,
            displacement_atr_ratio=disp,
        )
        stp = TradeSetup(
            id=f"stp_causal_{i}",
            direction=Direction.BULLISH,
            poi=ob,
            entry_price=2600.0,
            stop_loss=2595.0,
            take_profit=2610.0,
            time=c.time,
            confirmed_time=t_trade,
            risk_reward_ratio=2.0,
        )
        setups.append(stp)

        ticks.append(Tick(time=t_trade, bid=2604.0, ask=2604.2))
        ticks.append(Tick(time=t_trade + timedelta(seconds=10), bid=2599.8, ask=2600.0))
        ticks.append(
            Tick(
                time=t_trade + timedelta(seconds=20),
                bid=2594.0 if is_loss else 2610.2,
                ask=2594.2 if is_loss else 2610.4,
            )
        )

    hyp = Hypothesis(
        id="HYP-DEFAULT-ALPHA-WIN",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="0.9",
        proposed_value="1.2",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Default alpha validation",
        rationale="Expectancy gain with alpha 0.05",
    )

    validator = HypothesisValidator(
        min_trade_retention=0.70,
        min_expectancy_delta_r=0.0,
    )
    res = validator.validate(
        hypothesis=hyp,
        candles=candles,
        ticks=ticks,
        setups=setups,
        oos_split_ratio=0.30,
    )
    assert isinstance(res, ValidationResult)
    assert res.baseline_metrics.total_trades == 17
    assert res.candidate_metrics.total_trades == 13
    assert res.trade_retention_rate == pytest.approx(13 / 17, 0.001)
    assert res.expectancy_delta_r > 0.0
    assert res.is_statistically_superior is True
    assert res.p_value_improvement < 0.05
    assert res.verdict == HypothesisStatus.VALIDATED
    assert res.baseline_metrics.win_rate < res.candidate_metrics.win_rate
    assert res.candidate_metrics.win_rate == 1.0
    assert res.drawdown_delta_pct <= 0.0
    assert res.rejection_reason == ""


def test_hypothesis_validator_displacement_detect_setups_receives_current_and_proposed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import xau_bot.smc.setup
    from xau_bot.domain.models import Candle, Timeframe

    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    candles = [
        Candle(
            time=t0 + timedelta(minutes=i * 5),
            open=2600.0,
            high=2605.0,
            low=2595.0,
            close=2601.0,
            volume=50.0,
            timeframe=Timeframe.M5,
        )
        for i in range(25)
    ]
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    hyp = Hypothesis(
        id="HYP-DISP-CALLS",
        created_at=t0,
        parameter_name="min_displacement_multiplier",
        current_value="1.2",
        proposed_value="1.8",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Check detect_setups args",
        rationale="R6-01",
    )
    captured_calls: list[float] = []

    def spy_detect_setups(*args: Any, **kwargs: Any) -> list[Any]:
        captured_calls.append(kwargs.get("min_ob_displacement_atr", 0.0))
        return []

    monkeypatch.setattr(xau_bot.smc.setup, "detect_setups", spy_detect_setups)

    dummy_m = ValidationMetrics(
        total_trades=0,
        win_count=0,
        loss_count=0,
        win_rate=0.0,
        expectancy_r=0.0,
        max_drawdown_pct=0.0,
        profit_factor=1.0,
    )
    validator = HypothesisValidator()
    monkeypatch.setattr(
        validator,
        "validate_empirical",
        lambda **kwargs: ValidationResult(
            hypothesis_id=hyp.id,
            parameter_name=hyp.parameter_name,
            current_value=hyp.current_value,
            proposed_value=hyp.proposed_value,
            baseline_metrics=dummy_m,
            candidate_metrics=dummy_m,
            expectancy_delta_r=0.0,
            drawdown_delta_pct=0.0,
            trade_retention_rate=0.0,
            is_statistically_superior=False,
            p_value_improvement=1.0,
            verdict=HypothesisStatus.REJECTED,
            rejection_reason="Spy",
            validated_at=datetime.now(UTC),
        ),
    )
    res = validator.validate(hyp, ticks=ticks, candles=candles, setups=None)
    assert isinstance(res, ValidationResult)
    assert captured_calls == [1.2, 1.8]


def test_hypothesis_validator_rejects_risk_per_trade_pct_explicitly() -> None:
    t0 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="HYP-RISK-PCT",
        created_at=t0,
        parameter_name="risk_per_trade_pct",
        current_value="0.01",
        proposed_value="0.02",
        sample_size=30,
        condition_loss_rate=0.40,
        baseline_loss_rate=0.50,
        p_value=0.01,
        z_score=2.5,
        observed_expectancy_r=1.0,
        status=HypothesisStatus.PROPOSED,
        description="Invalid param for R-expectancy",
        rationale="Dimensioning only",
    )
    validator = HypothesisValidator()
    ticks = [
        Tick(time=t0, bid=2600.0, ask=2600.2),
        Tick(time=t0 + timedelta(seconds=10), bid=2601.0, ask=2601.2),
    ]
    with pytest.raises(ValueError, match="dimensionamento"):
        validator.validate(hyp, ticks=ticks)


def test_parameter_registry_immutability(tmp_path: Path) -> None:
    from xau_bot.learning.registry import ParameterRegistry

    db_file = tmp_path / "immutability.db"
    with ParameterRegistry(db_file) as registry:
        v1 = registry.create_version({"max_spread_atr_ratio": 0.08}, applied_by="test")
        v2 = registry.create_version({"max_spread_atr_ratio": 0.06}, applied_by="test")

        versions = registry.list_versions()
        assert len(versions) == 2
        assert versions[0].version_id == v2.version_id
        assert versions[1].version_id == v1.version_id
        assert versions[0].parameters == {"max_spread_atr_ratio": 0.06}
        assert versions[1].parameters == {"max_spread_atr_ratio": 0.08}
