"""Domain models and data structures for the Learning and Periodic Review module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class HypothesisStatus(StrEnum):
    PROPOSED = "PROPOSED"
    ACCUMULATING = "ACCUMULATING"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    SHADOW = "SHADOW"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class FeatureStats:
    feature_name: str
    condition_key: str
    sample_size: int
    loss_count: int
    win_count: int
    breakeven_count: int
    loss_rate: float
    baseline_loss_rate: float
    z_score: float
    p_value: float
    mean_realized_r: float
    is_statistically_significant: bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "condition_key": self.condition_key,
            "sample_size": self.sample_size,
            "loss_count": self.loss_count,
            "win_count": self.win_count,
            "breakeven_count": self.breakeven_count,
            "loss_rate": self.loss_rate,
            "baseline_loss_rate": self.baseline_loss_rate,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "mean_realized_r": self.mean_realized_r,
            "is_statistically_significant": self.is_statistically_significant,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureStats:
        return cls(
            feature_name=str(data["feature_name"]),
            condition_key=str(data["condition_key"]),
            sample_size=int(data["sample_size"]),
            loss_count=int(data["loss_count"]),
            win_count=int(data["win_count"]),
            breakeven_count=int(data.get("breakeven_count", 0)),
            loss_rate=float(data["loss_rate"]),
            baseline_loss_rate=float(data["baseline_loss_rate"]),
            z_score=float(data["z_score"]),
            p_value=float(data["p_value"]),
            mean_realized_r=float(data["mean_realized_r"]),
            is_statistically_significant=bool(data["is_statistically_significant"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    created_at: datetime
    parameter_name: str
    current_value: str
    proposed_value: str
    sample_size: int
    condition_loss_rate: float
    baseline_loss_rate: float
    p_value: float
    z_score: float
    observed_expectancy_r: float
    status: HypothesisStatus
    description: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "sample_size": self.sample_size,
            "condition_loss_rate": self.condition_loss_rate,
            "baseline_loss_rate": self.baseline_loss_rate,
            "p_value": self.p_value,
            "z_score": self.z_score,
            "observed_expectancy_r": self.observed_expectancy_r,
            "status": self.status.value,
            "description": self.description,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        raw_ts = data["created_at"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            id=str(data["id"]),
            created_at=ts,
            parameter_name=str(data["parameter_name"]),
            current_value=str(data["current_value"]),
            proposed_value=str(data["proposed_value"]),
            sample_size=int(data["sample_size"]),
            condition_loss_rate=float(data["condition_loss_rate"]),
            baseline_loss_rate=float(data["baseline_loss_rate"]),
            p_value=float(data["p_value"]),
            z_score=float(data["z_score"]),
            observed_expectancy_r=float(data["observed_expectancy_r"]),
            status=HypothesisStatus(data["status"]),
            description=str(data.get("description", "")),
            rationale=str(data.get("rationale", "")),
        )


@dataclass(frozen=True, slots=True)
class LearningReviewReport:
    reviewed_at: datetime
    total_trades_analyzed: int
    total_wins: int
    total_losses: int
    total_breakevens: int
    win_rate: float
    loss_rate: float
    global_expectancy_r: float
    min_samples: int
    alpha: float
    loss_cause_distribution: dict[str, int]
    feature_stats: tuple[FeatureStats, ...]
    hypotheses: tuple[Hypothesis, ...]
    accumulating_observations: tuple[FeatureStats, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewed_at": self.reviewed_at.astimezone(UTC).isoformat(),
            "total_trades_analyzed": self.total_trades_analyzed,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "total_breakevens": self.total_breakevens,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "global_expectancy_r": self.global_expectancy_r,
            "min_samples": self.min_samples,
            "alpha": self.alpha,
            "loss_cause_distribution": dict(self.loss_cause_distribution),
            "feature_stats": [fs.to_dict() for fs in self.feature_stats],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "accumulating_observations": [obs.to_dict() for obs in self.accumulating_observations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningReviewReport:
        raw_ts = data["reviewed_at"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            reviewed_at=ts,
            total_trades_analyzed=int(data["total_trades_analyzed"]),
            total_wins=int(data["total_wins"]),
            total_losses=int(data["total_losses"]),
            total_breakevens=int(data.get("total_breakevens", 0)),
            win_rate=float(data["win_rate"]),
            loss_rate=float(data["loss_rate"]),
            global_expectancy_r=float(data["global_expectancy_r"]),
            min_samples=int(data["min_samples"]),
            alpha=float(data["alpha"]),
            loss_cause_distribution=dict(data.get("loss_cause_distribution", {})),
            feature_stats=tuple(
                FeatureStats.from_dict(item) for item in data.get("feature_stats", [])
            ),
            hypotheses=tuple(Hypothesis.from_dict(item) for item in data.get("hypotheses", [])),
            accumulating_observations=tuple(
                FeatureStats.from_dict(item) for item in data.get("accumulating_observations", [])
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    expectancy_r: float
    max_drawdown_pct: float
    profit_factor: float

    def to_dict(self) -> dict[str, Any]:
        pf = (
            999.0
            if (
                math.isinf(self.profit_factor)
                or math.isnan(self.profit_factor)
                or self.profit_factor > 999.0
            )
            else self.profit_factor
        )
        return {
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "expectancy_r": self.expectancy_r,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": round(pf, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationMetrics:
        return cls(
            total_trades=int(data["total_trades"]),
            win_count=int(data["win_count"]),
            loss_count=int(data["loss_count"]),
            win_rate=float(data["win_rate"]),
            expectancy_r=float(data["expectancy_r"]),
            max_drawdown_pct=float(data["max_drawdown_pct"]),
            profit_factor=float(data["profit_factor"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    hypothesis_id: str
    parameter_name: str
    current_value: str
    proposed_value: str
    baseline_metrics: ValidationMetrics
    candidate_metrics: ValidationMetrics
    expectancy_delta_r: float
    drawdown_delta_pct: float
    trade_retention_rate: float
    is_statistically_superior: bool
    p_value_improvement: float
    verdict: HypothesisStatus
    rejection_reason: str = ""
    validated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        val_at = self.validated_at or datetime.now(UTC)
        return {
            "hypothesis_id": self.hypothesis_id,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
            "expectancy_delta_r": self.expectancy_delta_r,
            "drawdown_delta_pct": self.drawdown_delta_pct,
            "trade_retention_rate": self.trade_retention_rate,
            "is_statistically_superior": self.is_statistically_superior,
            "p_value_improvement": self.p_value_improvement,
            "verdict": self.verdict.value,
            "rejection_reason": self.rejection_reason,
            "validated_at": val_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        raw_ts = data.get("validated_at")
        ts = None
        if raw_ts is not None:
            ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            parameter_name=str(data["parameter_name"]),
            current_value=str(data["current_value"]),
            proposed_value=str(data["proposed_value"]),
            baseline_metrics=ValidationMetrics.from_dict(data["baseline_metrics"]),
            candidate_metrics=ValidationMetrics.from_dict(data["candidate_metrics"]),
            expectancy_delta_r=float(data["expectancy_delta_r"]),
            drawdown_delta_pct=float(data["drawdown_delta_pct"]),
            trade_retention_rate=float(data["trade_retention_rate"]),
            is_statistically_superior=bool(data["is_statistically_superior"]),
            p_value_improvement=float(data["p_value_improvement"]),
            verdict=HypothesisStatus(str(data["verdict"])),
            rejection_reason=str(data.get("rejection_reason", "")),
            validated_at=ts,
        )


@dataclass(frozen=True, slots=True)
class ParameterVersion:
    version_id: str
    created_at: datetime
    parameters: dict[str, Any]
    source_hypothesis_id: str | None = None
    applied_by: str = "operator"
    is_active: bool = False
    rollback_target_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "parameters": dict(self.parameters),
            "source_hypothesis_id": self.source_hypothesis_id,
            "applied_by": self.applied_by,
            "is_active": self.is_active,
            "rollback_target_id": self.rollback_target_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterVersion:
        raw_ts = data["created_at"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            version_id=str(data["version_id"]),
            created_at=ts,
            parameters=dict(data["parameters"]),
            source_hypothesis_id=(
                str(data["source_hypothesis_id"]) if data.get("source_hypothesis_id") else None
            ),
            applied_by=str(data.get("applied_by", "operator")),
            is_active=bool(data.get("is_active", False)),
            rollback_target_id=(
                str(data["rollback_target_id"]) if data.get("rollback_target_id") else None
            ),
        )


class HumanDiscretionVerdict(StrEnum):
    POSITIVE_EDGE = "POSITIVE_EDGE"
    NEGATIVE_BIAS = "NEGATIVE_BIAS"
    NEUTRAL = "NEUTRAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class CohortMetrics:
    cohort_name: str
    total_proposals: int
    evaluated_trades: int
    win_count: int
    loss_count: int
    breakeven_count: int
    win_rate: float
    loss_rate: float
    expectancy_r: float
    std_dev_r: float
    total_realized_r: float
    profit_factor: float
    realized_r_values: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        pf = (
            999.0
            if (
                math.isinf(self.profit_factor)
                or math.isnan(self.profit_factor)
                or self.profit_factor > 999.0
            )
            else self.profit_factor
        )
        return {
            "cohort_name": self.cohort_name,
            "total_proposals": self.total_proposals,
            "evaluated_trades": self.evaluated_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "breakeven_count": self.breakeven_count,
            "win_rate": round(self.win_rate, 4),
            "loss_rate": round(self.loss_rate, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "std_dev_r": round(self.std_dev_r, 4),
            "total_realized_r": round(self.total_realized_r, 4),
            "profit_factor": round(pf, 4),
            "realized_r_values": [round(v, 4) for v in self.realized_r_values],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CohortMetrics:
        return cls(
            cohort_name=str(data["cohort_name"]),
            total_proposals=int(data["total_proposals"]),
            evaluated_trades=int(data["evaluated_trades"]),
            win_count=int(data["win_count"]),
            loss_count=int(data["loss_count"]),
            breakeven_count=int(data.get("breakeven_count", 0)),
            win_rate=float(data["win_rate"]),
            loss_rate=float(data["loss_rate"]),
            expectancy_r=float(data["expectancy_r"]),
            std_dev_r=float(data["std_dev_r"]),
            total_realized_r=float(data["total_realized_r"]),
            profit_factor=float(data["profit_factor"]),
            realized_r_values=tuple(float(v) for v in data.get("realized_r_values", [])),
        )


@dataclass(frozen=True, slots=True)
class HumanSegmentStats:
    dimension: str
    segment_value: str
    total_count: int
    approved_count: int
    ignored_count: int
    approval_rate: float
    approved_expectancy_r: float
    ignored_expectancy_r: float
    delta_expectancy_r: float
    approved_win_rate: float
    ignored_win_rate: float
    delta_win_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "segment_value": self.segment_value,
            "total_count": self.total_count,
            "approved_count": self.approved_count,
            "ignored_count": self.ignored_count,
            "approval_rate": round(self.approval_rate, 4),
            "approved_expectancy_r": round(self.approved_expectancy_r, 4),
            "ignored_expectancy_r": round(self.ignored_expectancy_r, 4),
            "delta_expectancy_r": round(self.delta_expectancy_r, 4),
            "approved_win_rate": round(self.approved_win_rate, 4),
            "ignored_win_rate": round(self.ignored_win_rate, 4),
            "delta_win_rate": round(self.delta_win_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanSegmentStats:
        return cls(
            dimension=str(data["dimension"]),
            segment_value=str(data["segment_value"]),
            total_count=int(data["total_count"]),
            approved_count=int(data["approved_count"]),
            ignored_count=int(data["ignored_count"]),
            approval_rate=float(data["approval_rate"]),
            approved_expectancy_r=float(data["approved_expectancy_r"]),
            ignored_expectancy_r=float(data["ignored_expectancy_r"]),
            delta_expectancy_r=float(data["delta_expectancy_r"]),
            approved_win_rate=float(data["approved_win_rate"]),
            ignored_win_rate=float(data["ignored_win_rate"]),
            delta_win_rate=float(data["delta_win_rate"]),
        )


@dataclass(frozen=True, slots=True)
class HumanDiscretionReport:
    analyzed_at: datetime
    min_samples: int
    alpha: float
    total_proposals: int
    approved_metrics: CohortMetrics
    ignored_metrics: CohortMetrics
    combined_metrics: CohortMetrics
    acceptance_rate: float
    delta_expectancy_r: float
    delta_win_rate: float
    p_value_expectancy: float | None
    p_value_win_rate: float | None
    verdict: HumanDiscretionVerdict
    verdict_summary: str
    segments: tuple[HumanSegmentStats, ...] = ()
    raw_proposals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzed_at": self.analyzed_at.astimezone(UTC).isoformat(),
            "min_samples": self.min_samples,
            "alpha": self.alpha,
            "total_proposals": self.total_proposals,
            "raw_proposals": self.raw_proposals,
            "approved_metrics": self.approved_metrics.to_dict(),
            "ignored_metrics": self.ignored_metrics.to_dict(),
            "combined_metrics": self.combined_metrics.to_dict(),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "delta_expectancy_r": round(self.delta_expectancy_r, 4),
            "delta_win_rate": round(self.delta_win_rate, 4),
            "p_value_expectancy": (
                round(self.p_value_expectancy, 6) if self.p_value_expectancy is not None else None
            ),
            "p_value_win_rate": (
                round(self.p_value_win_rate, 6) if self.p_value_win_rate is not None else None
            ),
            "verdict": self.verdict.value,
            "verdict_summary": self.verdict_summary,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanDiscretionReport:
        raw_ts = data["analyzed_at"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return cls(
            analyzed_at=ts,
            min_samples=int(data["min_samples"]),
            alpha=float(data["alpha"]),
            total_proposals=int(data["total_proposals"]),
            approved_metrics=CohortMetrics.from_dict(data["approved_metrics"]),
            ignored_metrics=CohortMetrics.from_dict(data["ignored_metrics"]),
            combined_metrics=CohortMetrics.from_dict(data["combined_metrics"]),
            acceptance_rate=float(data["acceptance_rate"]),
            delta_expectancy_r=float(data["delta_expectancy_r"]),
            delta_win_rate=float(data["delta_win_rate"]),
            p_value_expectancy=(
                float(data["p_value_expectancy"])
                if data.get("p_value_expectancy") is not None
                else None
            ),
            p_value_win_rate=(
                float(data["p_value_win_rate"])
                if data.get("p_value_win_rate") is not None
                else None
            ),
            verdict=HumanDiscretionVerdict(data["verdict"]),
            verdict_summary=str(data.get("verdict_summary", "")),
            segments=tuple(HumanSegmentStats.from_dict(s) for s in data.get("segments", [])),
            raw_proposals=int(data.get("raw_proposals", data.get("total_proposals", 0))),
        )
