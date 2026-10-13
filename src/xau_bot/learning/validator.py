"""Hypothesis validator evaluating candidate parameter configurations via backtesting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from xau_bot.backtest.engine import BacktestEngine
from xau_bot.backtest.models import BacktestResult, BacktestTrade
from xau_bot.domain.models import OrderBlock
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import SpreadFilterConfig
from xau_bot.filters.spread import SpreadFilter
from xau_bot.learning.models import (
    Hypothesis,
    HypothesisStatus,
    ValidationMetrics,
    ValidationResult,
)
from xau_bot.learning.stats import calculate_expectancy, welch_t_test
from xau_bot.risk.guards import RiskGuard

if TYPE_CHECKING:
    from xau_bot.backtest.models import BacktestOrder
    from xau_bot.domain.models import Candle, Tick, TradeSetup


class HypothesisValidator:
    """Evaluates proposed parameter adjustments against empirical baseline backtest metrics."""

    def __init__(
        self,
        min_trade_retention: float = 0.70,
        max_drawdown_tolerance_ratio: float = 1.10,
        min_expectancy_delta_r: float = 0.0,
        alpha: float = 0.05,
    ) -> None:
        if not (0.0 < min_trade_retention <= 1.0):
            raise ValueError(
                f"min_trade_retention must be between 0.0 and 1.0, got {min_trade_retention}"
            )
        if max_drawdown_tolerance_ratio < 1.0:
            raise ValueError(
                f"max_drawdown_tolerance_ratio must be >= 1.0, got {max_drawdown_tolerance_ratio}"
            )
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        self.min_trade_retention = min_trade_retention
        self.max_drawdown_tolerance_ratio = max_drawdown_tolerance_ratio
        self.min_expectancy_delta_r = min_expectancy_delta_r
        self.alpha = alpha

    def evaluate_metrics(
        self,
        hypothesis: Hypothesis,
        baseline: ValidationMetrics,
        candidate: ValidationMetrics,
        p_value_improvement: float | None = None,
        oos_baseline: ValidationMetrics | None = None,
        oos_candidate: ValidationMetrics | None = None,
        validated_at: datetime | None = None,
        require_empirical_p: bool = False,
    ) -> ValidationResult:
        now = validated_at or datetime.now(UTC)

        expectancy_delta_r = round(candidate.expectancy_r - baseline.expectancy_r, 4)
        drawdown_delta_pct = round(candidate.max_drawdown_pct - baseline.max_drawdown_pct, 4)
        trade_retention_rate = (
            round(candidate.total_trades / baseline.total_trades, 4)
            if baseline.total_trades > 0
            else 0.0
        )

        reasons: list[str] = []

        if require_empirical_p:
            if p_value_improvement is None:
                reasons.append(
                    f"Amostra insuficiente para teste de significância do Delta R "
                    f"(mínimo 2 trades por braço: n_base={baseline.total_trades}, "
                    f"n_cand={candidate.total_trades})."
                )
                p_val = None
            else:
                p_val = p_value_improvement
        else:
            p_val = p_value_improvement if p_value_improvement is not None else hypothesis.p_value

        if p_val is not None and p_val >= self.alpha:
            reasons.append(
                f"Hypothesis p-value ({p_val:.4f}) does not meet significance threshold "
                f"(alpha={self.alpha:.4f})."
            )

        if expectancy_delta_r <= self.min_expectancy_delta_r:
            reasons.append(
                f"Expectancy delta ({expectancy_delta_r:+.2f}R) does not improve upon baseline "
                f"by required threshold ({self.min_expectancy_delta_r:+.2f}R)."
            )

        allowed_max_dd = round(baseline.max_drawdown_pct * self.max_drawdown_tolerance_ratio, 4)
        if candidate.max_drawdown_pct > allowed_max_dd and not math.isclose(
            candidate.max_drawdown_pct, allowed_max_dd, rel_tol=1e-5
        ):
            reasons.append(
                f"Max drawdown increased from {baseline.max_drawdown_pct:.1f}% to "
                f"{candidate.max_drawdown_pct:.1f}%, exceeding allowable limit of "
                f"{allowed_max_dd:.1f}%."
            )

        if trade_retention_rate < self.min_trade_retention:
            reasons.append(
                f"Trade retention rate {trade_retention_rate * 100:.1f}% is below minimum "
                f"threshold {self.min_trade_retention * 100:.1f}%."
            )

        if oos_baseline is not None and oos_candidate is not None:
            oos_delta_r = round(oos_candidate.expectancy_r - oos_baseline.expectancy_r, 4)
            if oos_delta_r <= 0.0:
                reasons.append(
                    f"Out-of-sample (OOS) expectancy delta ({oos_delta_r:+.2f}R) does not show "
                    "positive gain."
                )

        is_superior = len(reasons) == 0
        verdict = HypothesisStatus.VALIDATED if is_superior else HypothesisStatus.REJECTED
        rejection_reason_str = "; ".join(reasons) if reasons else ""
        p_val_float = p_val if p_val is not None else 1.0

        return ValidationResult(
            hypothesis_id=hypothesis.id,
            parameter_name=hypothesis.parameter_name,
            current_value=hypothesis.current_value,
            proposed_value=hypothesis.proposed_value,
            baseline_metrics=baseline,
            candidate_metrics=candidate,
            expectancy_delta_r=expectancy_delta_r,
            drawdown_delta_pct=drawdown_delta_pct,
            trade_retention_rate=trade_retention_rate,
            is_statistically_superior=is_superior,
            p_value_improvement=p_val_float,
            verdict=verdict,
            rejection_reason=rejection_reason_str,
            validated_at=now,
        )

    @staticmethod
    def _clone_engine(engine: Any) -> Any:
        if isinstance(engine, BacktestEngine):
            cloned_filter = None
            if engine.trade_filter is not None:
                cloned_filter = CompositeTradeFilter(
                    session_filter=engine.trade_filter.session_filter,
                    news_filter=engine.trade_filter.news_filter,
                    spread_filter=engine.trade_filter.spread_filter,
                )
            cloned_guard = None
            if engine.risk_guard is not None:
                cloned_guard = RiskGuard(
                    limits=engine.risk_guard.limits,
                    ny_tz=engine.risk_guard.ny_tz,
                )
            return BacktestEngine(
                config=engine.config,
                slippage_model=engine.slippage_model,
                position_risk_config=engine.position_risk_config,
                trade_filter=cloned_filter,
                risk_guard=cloned_guard,
                close_positions_at_end=engine.close_positions_at_end,
            )
        return engine

    def validate(
        self,
        hypothesis: Hypothesis,
        candles: Sequence[Candle] | None = None,
        ticks: Sequence[Tick] | None = None,
        setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        oos_split_ratio: float = 0.30,
        validated_at: datetime | None = None,
    ) -> ValidationResult:
        if not ticks or len(ticks) < 2:
            raise ValueError("ticks sequence must contain at least 2 ticks for backtest validation")

        supported_params = {
            "max_spread_atr_ratio",
            "min_displacement_multiplier",
        }
        if hypothesis.parameter_name == "risk_per_trade_pct":
            raise ValueError(
                "Parâmetro 'risk_per_trade_pct' altera o dimensionamento financeiro em USD, "
                "mas não a expectativa estatística expressa em múltiplos de R"
            )
        if hypothesis.parameter_name not in supported_params:
            raise ValueError(
                f"Parâmetro não suportado para validação em backtest: '{hypothesis.parameter_name}'"
            )

        base_trade_filter: CompositeTradeFilter | None = None
        cand_trade_filter: CompositeTradeFilter | None = None
        active_setups = setups
        candidate_setups = setups

        if hypothesis.parameter_name == "max_spread_atr_ratio":
            try:
                base_ratio = float(hypothesis.current_value)
                cand_ratio = float(hypothesis.proposed_value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Valores numéricos inválidos para max_spread_atr_ratio: "
                    f"current='{hypothesis.current_value}', proposed='{hypothesis.proposed_value}'"
                ) from exc
            if not math.isfinite(cand_ratio) or cand_ratio <= 0.0:
                raise ValueError(
                    f"Valor proposto para max_spread_atr_ratio deve ser finito e positivo, "
                    f"got {cand_ratio}"
                )
            base_trade_filter = CompositeTradeFilter(
                spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=base_ratio))
            )
            cand_trade_filter = CompositeTradeFilter(
                spread_filter=SpreadFilter(SpreadFilterConfig(max_spread_atr_ratio=cand_ratio))
            )
            if active_setups is None and candles:
                from xau_bot.config import Settings
                from xau_bot.domain.series import resample_to_h1
                from xau_bot.smc.setup import detect_setups

                settings = Settings()
                htf_candles = resample_to_h1(candles)
                active_setups = detect_setups(
                    candles=candles,
                    htf_candles=htf_candles if htf_candles else None,
                    **settings.smc_kwargs(),
                )
                candidate_setups = active_setups
        elif hypothesis.parameter_name == "min_displacement_multiplier":
            try:
                base_min_disp = float(hypothesis.current_value)
                cand_min_disp = float(hypothesis.proposed_value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Valores numéricos inválidos para min_displacement_multiplier: "
                    f"current='{hypothesis.current_value}', proposed='{hypothesis.proposed_value}'"
                ) from exc
            if not math.isfinite(base_min_disp) or base_min_disp < 0.0:
                raise ValueError(
                    f"Valor atual para min_displacement_multiplier deve ser finito e >= 0, "
                    f"got {base_min_disp}"
                )
            if not math.isfinite(cand_min_disp) or cand_min_disp < 0.0:
                raise ValueError(
                    f"Valor proposto para min_displacement_multiplier deve ser finito e >= 0, "
                    f"got {cand_min_disp}"
                )

            if setups is None and candles:
                from xau_bot.config import Settings
                from xau_bot.domain.series import resample_to_h1
                from xau_bot.smc.setup import detect_setups

                settings = Settings()
                htf_candles = resample_to_h1(candles)
                active_setups = detect_setups(
                    candles=candles,
                    htf_candles=htf_candles if htf_candles else None,
                    min_ob_displacement_atr=base_min_disp,
                    **settings.smc_kwargs(),
                )
                candidate_setups = detect_setups(
                    candles=candles,
                    htf_candles=htf_candles if htf_candles else None,
                    min_ob_displacement_atr=cand_min_disp,
                    **settings.smc_kwargs(),
                )
            elif active_setups is not None:
                has_measurable = any(
                    isinstance(s.poi, OrderBlock) and s.poi.displacement_atr_ratio > 0.0
                    for s in active_setups
                )
                if not has_measurable:
                    raise ValueError(
                        "Setups sem displacement_atr_ratio mensurável; "
                        "impossível validar min_displacement_multiplier"
                    )
                base_filtered = [
                    s
                    for s in active_setups
                    if not isinstance(s.poi, OrderBlock)
                    or s.poi.displacement_atr_ratio >= base_min_disp
                ]
                cand_filtered = [
                    s
                    for s in active_setups
                    if not isinstance(s.poi, OrderBlock)
                    or s.poi.displacement_atr_ratio >= cand_min_disp
                ]
                active_setups = base_filtered
                candidate_setups = cand_filtered

        base_engine = BacktestEngine(
            trade_filter=base_trade_filter,
        )
        cand_engine = BacktestEngine(
            trade_filter=cand_trade_filter,
        )

        return self.validate_empirical(
            hypothesis=hypothesis,
            baseline_engine=base_engine,
            candidate_engine=cand_engine,
            ticks=ticks,
            setups=active_setups,
            candidate_setups=candidate_setups,
            orders=orders,
            candles=candles,
            oos_split_ratio=oos_split_ratio,
            validated_at=validated_at,
        )

    def validate_empirical(
        self,
        hypothesis: Hypothesis,
        baseline_engine: Any,
        candidate_engine: Any,
        ticks: Sequence[Tick],
        setups: Sequence[TradeSetup] | None = None,
        candidate_setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        candles: Sequence[Candle] | None = None,
        oos_split_ratio: float | None = None,
        validated_at: datetime | None = None,
    ) -> ValidationResult:
        cand_setups = setups if candidate_setups is None else candidate_setups

        if oos_split_ratio is not None:
            if not (0.0 < oos_split_ratio < 1.0):
                raise ValueError(
                    f"oos_split_ratio must be strictly between 0.0 and 1.0, got {oos_split_ratio}"
                )

            split_idx = int(len(ticks) * (1.0 - oos_split_ratio))
            if split_idx <= 0 or split_idx >= len(ticks):
                split_idx = max(1, min(len(ticks) - 1, split_idx))

            is_ticks = ticks[:split_idx]
            oos_ticks = ticks[split_idx:]
            split_time = oos_ticks[0].time

            is_candles = [c for c in candles if c.time < split_time] if candles else None
            oos_candles = [c for c in candles if c.time >= split_time] if candles else None

            is_setups = [s for s in setups if s.confirmed_time < split_time] if setups else None
            oos_setups = [s for s in setups if s.confirmed_time >= split_time] if setups else None

            is_cand_setups = (
                [s for s in cand_setups if s.confirmed_time < split_time] if cand_setups else None
            )
            oos_cand_setups = (
                [s for s in cand_setups if s.confirmed_time >= split_time] if cand_setups else None
            )

            is_orders = [o for o in orders if o.created_at < split_time] if orders else None
            oos_orders = [o for o in orders if o.created_at >= split_time] if orders else None

            is_base_engine = self._clone_engine(baseline_engine)
            is_cand_engine = self._clone_engine(candidate_engine)
            oos_base_engine = self._clone_engine(baseline_engine)
            oos_cand_engine = self._clone_engine(candidate_engine)

            is_base_res = is_base_engine.run(
                ticks=is_ticks, setups=is_setups, orders=is_orders, candles=is_candles
            )
            is_cand_res = is_cand_engine.run(
                ticks=is_ticks, setups=is_cand_setups, orders=is_orders, candles=is_candles
            )

            oos_base_res = oos_base_engine.run(
                ticks=oos_ticks, setups=oos_setups, orders=oos_orders, candles=oos_candles
            )
            oos_cand_res = oos_cand_engine.run(
                ticks=oos_ticks, setups=oos_cand_setups, orders=oos_orders, candles=oos_candles
            )

            p_val = None
            if is_base_res.trades and is_cand_res.trades:
                base_r = [t.return_r for t in is_base_res.trades]
                cand_r = [t.return_r for t in is_cand_res.trades]
                if len(base_r) >= 2 and len(cand_r) >= 2:
                    _, p_val = welch_t_test(base_r, cand_r)

            return self.evaluate_metrics(
                hypothesis=hypothesis,
                baseline=self.metrics_from_backtest(is_base_res),
                candidate=self.metrics_from_backtest(is_cand_res),
                p_value_improvement=p_val,
                oos_baseline=self.metrics_from_backtest(oos_base_res),
                oos_candidate=self.metrics_from_backtest(oos_cand_res),
                validated_at=validated_at,
                require_empirical_p=True,
            )

        base_res = baseline_engine.run(ticks=ticks, setups=setups, orders=orders, candles=candles)
        cand_res = candidate_engine.run(
            ticks=ticks, setups=cand_setups, orders=orders, candles=candles
        )

        p_val = None
        if base_res.trades and cand_res.trades:
            base_r = [t.return_r for t in base_res.trades]
            cand_r = [t.return_r for t in cand_res.trades]
            if len(base_r) >= 2 and len(cand_r) >= 2:
                _, p_val = welch_t_test(base_r, cand_r)

        return self.evaluate_metrics(
            hypothesis=hypothesis,
            baseline=self.metrics_from_backtest(base_res),
            candidate=self.metrics_from_backtest(cand_res),
            p_value_improvement=p_val,
            validated_at=validated_at,
            require_empirical_p=True,
        )

    @staticmethod
    def metrics_from_backtest(result: BacktestResult) -> ValidationMetrics:
        pf = result.profit_factor
        if math.isinf(pf) or math.isnan(pf) or pf > 999.0:
            pf = 999.0
        return ValidationMetrics(
            total_trades=result.total_trades,
            win_count=result.winning_trades,
            loss_count=result.losing_trades,
            win_rate=round(result.win_rate, 4),
            expectancy_r=round(result.expectancy_r, 4),
            max_drawdown_pct=round(result.max_drawdown_pct, 4),
            profit_factor=round(pf, 4),
        )

    @staticmethod
    def metrics_from_trades(
        trades: Sequence[BacktestTrade], initial_balance: float = 10000.0
    ) -> ValidationMetrics:
        if not trades:
            return ValidationMetrics(
                total_trades=0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                expectancy_r=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
            )

        total = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        losses = sum(1 for t in trades if t.net_pnl < 0)
        win_rate = round(wins / total, 4)
        expectancy_r = calculate_expectancy([t.return_r for t in trades])

        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        if gross_loss > 0:
            raw_pf = gross_profit / gross_loss
            profit_factor = round(min(raw_pf, 999.0), 4)
        else:
            profit_factor = 999.0 if gross_profit > 0 else 0.0

        peak = initial_balance
        curr = initial_balance
        max_dd_pct = 0.0
        for t in trades:
            curr += t.net_pnl
            if curr > peak:
                peak = curr
            dd_pct = ((peak - curr) / peak) * 100.0 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        return ValidationMetrics(
            total_trades=total,
            win_count=wins,
            loss_count=losses,
            win_rate=win_rate,
            expectancy_r=expectancy_r,
            max_drawdown_pct=round(max_dd_pct, 4),
            profit_factor=profit_factor,
        )
