import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.backtest.analysis import (
    BacktestAnalysisReport,
    BacktestAnalyzer,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloSimulator,
    RandomBaselineAnalyzer,
    RandomBaselineConfig,
    RandomBaselineResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    _percentile,
)
from xau_bot.backtest.engine import BacktestEngine
from xau_bot.backtest.models import (
    BacktestConfig,
    BacktestOrder,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    OrderType,
    TradeExitReason,
)
from xau_bot.backtest.slippage import FixedSlippage
from xau_bot.domain.models import Candle, Direction, Tick, TradeSetup


def _make_utc(
    hour: int = 10,
    minute: int = 0,
    second: int = 0,
    day: int = 24,
    month: int = 9,
    year: int = 2026,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _make_tick(bid: float, ask: float, time: datetime) -> Tick:
    return Tick(time=time, bid=bid, ask=ask)


def _create_mock_trade(
    trade_id: str,
    net_pnl: float,
    return_r: float,
    entry_time: datetime,
    exit_time: datetime,
    exit_reason: TradeExitReason = TradeExitReason.TAKE_PROFIT,
) -> BacktestTrade:
    return BacktestTrade(
        trade_id=trade_id,
        order_id=f"ORD-{trade_id}",
        direction=Direction.BULLISH,
        volume=0.1,
        entry_time=entry_time,
        entry_price=2600.0,
        exit_time=exit_time,
        exit_price=2610.0 if net_pnl > 0 else 2595.0,
        exit_reason=exit_reason,
        initial_sl=2595.0,
        final_sl=2595.0,
        tp=2610.0,
        commission=0.7,
        swap=0.0,
        gross_pnl=net_pnl + 0.7,
        net_pnl=net_pnl,
        return_r=return_r,
        mfe_price=2610.0,
        mae_price=2598.0,
        mfe_r=2.0,
        mae_r=0.4,
        duration_seconds=(exit_time - entry_time).total_seconds(),
    )


def _create_sample_backtest_result(trades: list[BacktestTrade]) -> BacktestResult:
    initial_balance = 10000.0
    total_trades = len(trades)
    winning = sum(1 for t in trades if t.net_pnl > 0)
    losing = sum(1 for t in trades if t.net_pnl < 0)
    breakeven = total_trades - winning - losing
    total_net = sum(t.net_pnl for t in trades)
    final_balance = initial_balance + total_net

    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    expectancy = sum(t.return_r for t in trades) / total_trades if total_trades > 0 else 0.0
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    equity_points = [
        EquityPoint(
            timestamp=_make_utc(9, 0),
            balance=initial_balance,
            equity=initial_balance,
            drawdown_usd=0.0,
            drawdown_pct=0.0,
        )
    ]
    cur_bal = initial_balance
    peak = cur_bal
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for t in trades:
        cur_bal += t.net_pnl
        if cur_bal > peak:
            peak = cur_bal
        dd = peak - cur_bal
        dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd_usd:
            max_dd_usd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        equity_points.append(
            EquityPoint(
                timestamp=t.exit_time,
                balance=cur_bal,
                equity=cur_bal,
                drawdown_usd=dd,
                drawdown_pct=dd_pct,
            )
        )

    return BacktestResult(
        trades=tuple(trades),
        equity_curve=tuple(equity_points),
        initial_balance=initial_balance,
        final_balance=final_balance,
        total_net_pnl=total_net,
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        breakeven_trades=breakeven,
        win_rate=win_rate,
        profit_factor=pf,
        max_drawdown_usd=max_dd_usd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=1.5 if total_trades > 0 else 0.0,
        expectancy_r=expectancy,
        average_trade_duration_seconds=600.0,
    )


class TestMonteCarloSimulator:
    def test_ac_1201_monte_carlo_metrics_and_percentiles(self) -> None:
        t0 = _make_utc(10, 0)
        trades = [
            _create_mock_trade(
                "T1",
                net_pnl=100.0,
                return_r=2.0,
                entry_time=t0,
                exit_time=t0 + timedelta(minutes=10),
            ),
            _create_mock_trade(
                "T2",
                net_pnl=-50.0,
                return_r=-1.0,
                entry_time=t0 + timedelta(minutes=15),
                exit_time=t0 + timedelta(minutes=25),
            ),
            _create_mock_trade(
                "T3",
                net_pnl=150.0,
                return_r=3.0,
                entry_time=t0 + timedelta(minutes=30),
                exit_time=t0 + timedelta(minutes=40),
            ),
            _create_mock_trade(
                "T4",
                net_pnl=-50.0,
                return_r=-1.0,
                entry_time=t0 + timedelta(minutes=45),
                exit_time=t0 + timedelta(minutes=55),
            ),
            _create_mock_trade(
                "T5",
                net_pnl=80.0,
                return_r=1.6,
                entry_time=t0 + timedelta(minutes=60),
                exit_time=t0 + timedelta(minutes=70),
            ),
        ]
        simulator = MonteCarloSimulator()
        cfg = MonteCarloConfig(simulations=500, ruin_drawdown_pct=10.0, seed=123)
        result = simulator.simulate(trades=trades, initial_balance=10000.0, config=cfg)

        assert isinstance(result, MonteCarloResult)
        assert result.simulations == 500
        assert 0.0 <= result.ruin_probability <= 1.0
        assert result.median_max_drawdown_usd >= 0.0
        assert result.p95_max_drawdown_usd >= result.median_max_drawdown_usd
        assert result.p99_max_drawdown_usd >= result.p95_max_drawdown_usd
        assert result.p95_max_drawdown_pct >= result.median_max_drawdown_pct
        assert result.p99_max_drawdown_pct >= result.p95_max_drawdown_pct
        assert result.p5_final_balance <= result.median_final_balance <= result.p95_final_balance
        result_repeat = simulator.simulate(trades=trades, initial_balance=10000.0, config=cfg)
        assert result == result_repeat

    def test_ac_1202_monte_carlo_edge_cases_empty_and_single_trade(self) -> None:
        simulator = MonteCarloSimulator()
        cfg = MonteCarloConfig(simulations=100, seed=42)

        res_empty = simulator.simulate(trades=[], initial_balance=10000.0, config=cfg)
        assert res_empty.simulations == 100
        assert res_empty.ruin_probability == 0.0
        assert res_empty.median_max_drawdown_usd == 0.0
        assert res_empty.p95_max_drawdown_usd == 0.0
        assert res_empty.median_final_balance == 10000.0

        t0 = _make_utc(10, 0)
        single_trade = [_create_mock_trade("T1", 50.0, 1.0, t0, t0 + timedelta(minutes=5))]
        res_single = simulator.simulate(trades=single_trade, initial_balance=10000.0, config=cfg)
        assert res_single.simulations == 100
        assert res_single.median_final_balance == 10050.0
        assert res_single.ruin_probability == 0.0

    def test_monte_carlo_config_validation(self) -> None:
        with pytest.raises(ValueError, match="simulations must be >= 1"):
            MonteCarloConfig(simulations=0)
        with pytest.raises(ValueError, match="ruin_drawdown_pct must be between 0 and 100"):
            MonteCarloConfig(ruin_drawdown_pct=-1.0)
        with pytest.raises(ValueError, match="ruin_drawdown_pct must be between 0 and 100"):
            MonteCarloConfig(ruin_drawdown_pct=105.0)


class TestRandomBaselineAnalyzer:
    def test_ac_1203_random_baseline_generation_and_p_value(self) -> None:
        t0 = _make_utc(10, 0)
        ticks = [
            _make_tick(bid=2600.0, ask=2600.25, time=t0),
            _make_tick(bid=2605.0, ask=2605.25, time=t0 + timedelta(minutes=5)),
            _make_tick(bid=2595.0, ask=2595.25, time=t0 + timedelta(minutes=10)),
            _make_tick(bid=2610.0, ask=2610.25, time=t0 + timedelta(minutes=15)),
            _make_tick(bid=2590.0, ask=2590.25, time=t0 + timedelta(minutes=20)),
        ]
        strategy_trades = [
            _create_mock_trade("WIN-1", 100.0, 2.0, t0, t0 + timedelta(minutes=5)),
            _create_mock_trade(
                "WIN-2", 100.0, 2.0, t0 + timedelta(minutes=5), t0 + timedelta(minutes=15)
            ),
        ]
        strategy_result = _create_sample_backtest_result(strategy_trades)

        analyzer = RandomBaselineAnalyzer()
        cfg = RandomBaselineConfig(simulations=20, seed=42, trades_per_simulation=2)
        result = analyzer.analyze(
            ticks=ticks,
            strategy_result=strategy_result,
            config=cfg,
        )

        assert isinstance(result, RandomBaselineResult)
        assert result.simulations == 20
        assert math.isfinite(result.baseline_expectancy_r_mean)
        assert math.isfinite(result.baseline_expectancy_r_std)
        assert 0.0 <= result.p_value_expectancy_r <= 1.0
        assert 0.0 <= result.p_value_profit_factor <= 1.0
        assert result.strategy_profit_factor == strategy_result.profit_factor
        assert isinstance(result.statistically_significant, bool)

    def test_ac_1204_random_baseline_statistical_significance_flags(self) -> None:
        analyzer = RandomBaselineAnalyzer()
        t0 = _make_utc(10, 0)
        ticks: list[Tick] = []
        p = 2600.0
        delta = 1.0
        for i in range(200):
            if p >= 2615.0:
                delta = -1.0
            elif p <= 2585.0:
                delta = 1.0
            p += delta
            ticks.append(_make_tick(bid=p, ask=p + 0.25, time=t0 + timedelta(seconds=i)))

        outstanding_trades = [
            _create_mock_trade(f"G{i}", 500.0, 10.0, t0, t0 + timedelta(minutes=1))
            for i in range(5)
        ]
        res_outstanding = _create_sample_backtest_result(outstanding_trades)
        cfg = RandomBaselineConfig(simulations=20, seed=42, trades_per_simulation=5)
        res_stat = analyzer.analyze(ticks=ticks, strategy_result=res_outstanding, config=cfg)
        assert res_stat.strategy_expectancy_r == 10.0
        assert res_stat.p_value_expectancy_r <= 0.05
        assert res_stat.p_value_profit_factor <= 0.05
        assert res_stat.statistically_significant is True

        losing_trades = [
            _create_mock_trade(f"L{i}", -500.0, -10.0, t0, t0 + timedelta(minutes=1))
            for i in range(5)
        ]
        res_losing = _create_sample_backtest_result(losing_trades)
        res_stat_loss = analyzer.analyze(ticks=ticks, strategy_result=res_losing, config=cfg)
        assert res_stat_loss.p_value_expectancy_r >= 0.05
        assert res_stat_loss.statistically_significant is False

    def test_random_baseline_sequential_non_overlapping_execution(self) -> None:
        t0 = _make_utc(10, 0)
        ticks: list[Tick] = []
        p = 2600.0
        delta = 1.0
        for i in range(500):
            if p >= 2615.0:
                delta = -1.0
            elif p <= 2585.0:
                delta = 1.0
            p += delta
            ticks.append(_make_tick(bid=p, ask=p + 0.25, time=t0 + timedelta(seconds=i)))

        strategy_trades = [
            _create_mock_trade(f"T{i}", 10.0, 1.0, t0, t0 + timedelta(seconds=1)) for i in range(5)
        ]
        strategy_res = _create_sample_backtest_result(strategy_trades)

        recorded_orders: list[list[BacktestOrder]] = []
        recorded_results: list[BacktestResult] = []
        recorded_brokers: list[object] = []

        def spy_factory() -> BacktestEngine:
            real_engine = BacktestEngine(BacktestConfig())
            recorded_brokers.append(real_engine.broker)

            class _Spy(BacktestEngine):
                def run(
                    self,
                    ticks: Sequence[Tick],
                    setups: Sequence[TradeSetup] | None = None,
                    orders: Sequence[BacktestOrder] | None = None,
                    candles: Sequence[Candle] | None = None,
                ) -> BacktestResult:
                    if orders is not None:
                        recorded_orders.append(list(orders))
                    res = real_engine.run(
                        ticks=ticks, setups=setups, orders=orders, candles=candles
                    )
                    recorded_results.append(res)
                    return res

            return _Spy()

        analyzer = RandomBaselineAnalyzer()
        cfg = RandomBaselineConfig(
            simulations=5,
            trades_per_simulation=8,
            risk_price_distance=1.0,
            risk_reward_ratio=1.0,
            seed=42,
        )
        res = analyzer.analyze(
            ticks=ticks,
            strategy_result=strategy_res,
            config=cfg,
            engine_factory=spy_factory,
        )
        assert res.simulations == 5
        assert len(recorded_orders) == 5
        assert len(recorded_results) == 5
        for sim_idx in range(5):
            orders = recorded_orders[sim_idx]
            result = recorded_results[sim_idx]
            broker = recorded_brokers[sim_idx]
            assert len(orders) == 8
            assert result.total_trades == 8
            assert len(getattr(broker, "rejected_orders", [])) == 0
            for k in range(len(orders) - 1):
                assert orders[k + 1].created_at > orders[k].created_at

    def test_random_baseline_derived_geometry(self) -> None:
        t0 = _make_utc(10, 0)
        ticks: list[Tick] = []
        p = 2600.0
        delta = 1.0
        for i in range(500):
            if p >= 2615.0:
                delta = -1.0
            elif p <= 2585.0:
                delta = 1.0
            p += delta
            ticks.append(_make_tick(bid=p, ask=p + 0.25, time=t0 + timedelta(seconds=i)))

        trade = BacktestTrade(
            trade_id="T1",
            order_id="ORD-T1",
            direction=Direction.BULLISH,
            volume=0.35,
            entry_time=t0,
            entry_price=2600.0,
            exit_time=t0 + timedelta(minutes=1),
            exit_price=2604.0,
            exit_reason=TradeExitReason.TAKE_PROFIT,
            initial_sl=2598.0,
            final_sl=2598.0,
            tp=2604.0,
            commission=0.7,
            swap=0.0,
            gross_pnl=350.7,
            net_pnl=350.0,
            return_r=2.0,
            mfe_price=2604.0,
            mae_price=2598.0,
            mfe_r=2.0,
            mae_r=0.4,
            duration_seconds=60.0,
        )
        strategy_res = _create_sample_backtest_result([trade])

        recorded_orders: list[BacktestOrder] = []

        def spy_factory() -> BacktestEngine:
            real_engine = BacktestEngine(BacktestConfig())

            class _Spy(BacktestEngine):
                def run(
                    self,
                    ticks: Sequence[Tick],
                    setups: Sequence[TradeSetup] | None = None,
                    orders: Sequence[BacktestOrder] | None = None,
                    candles: Sequence[Candle] | None = None,
                ) -> BacktestResult:
                    if orders is not None:
                        recorded_orders.extend(orders)
                    return real_engine.run(
                        ticks=ticks, setups=setups, orders=orders, candles=candles
                    )

            return _Spy()

        analyzer = RandomBaselineAnalyzer()
        res = analyzer.analyze(
            ticks=ticks,
            strategy_result=strategy_res,
            config=RandomBaselineConfig(simulations=2, trades_per_simulation=4, seed=42),
            engine_factory=spy_factory,
        )
        assert res.simulations == 2
        assert len(recorded_orders) == 8
        for o in recorded_orders:
            assert o.volume == 0.35
            if o.direction == Direction.BULLISH:
                assert o.stop_loss is not None and o.take_profit is not None
                assert round(o.limit_price - o.stop_loss, 4) == 2.0
                assert round(o.take_profit - o.limit_price, 4) == 4.0
            else:
                assert o.stop_loss is not None and o.take_profit is not None
                assert round(o.stop_loss - o.limit_price, 4) == 2.0
                assert round(o.limit_price - o.take_profit, 4) == 4.0

    def test_random_baseline_parity_of_costs_and_engine_factory(self) -> None:
        t0 = _make_utc(10, 0)
        ticks: list[Tick] = []
        p = 2600.0
        delta = 1.0
        for i in range(500):
            if p >= 2615.0:
                delta = -1.0
            elif p <= 2585.0:
                delta = 1.0
            p += delta
            ticks.append(_make_tick(bid=p, ask=p + 0.25, time=t0 + timedelta(seconds=i)))

        trades = [
            _create_mock_trade(f"T{i}", 10.0, 1.0, t0, t0 + timedelta(minutes=1)) for i in range(5)
        ]
        strat_res = _create_sample_backtest_result(trades)

        def factory_zero() -> BacktestEngine:
            return BacktestEngine(slippage_model=FixedSlippage(points=0.0))

        def factory_slip() -> BacktestEngine:
            return BacktestEngine(slippage_model=FixedSlippage(points=50.0))

        analyzer = RandomBaselineAnalyzer()
        cfg = RandomBaselineConfig(simulations=5, trades_per_simulation=5, seed=42)
        res_zero = analyzer.analyze(
            ticks=ticks, strategy_result=strat_res, config=cfg, engine_factory=factory_zero
        )
        res_slip = analyzer.analyze(
            ticks=ticks, strategy_result=strat_res, config=cfg, engine_factory=factory_slip
        )

        assert res_slip.baseline_expectancy_r_mean < res_zero.baseline_expectancy_r_mean

        factory_calls = {"count": 0}

        def counting_factory() -> BacktestEngine:
            factory_calls["count"] += 1
            return BacktestEngine(BacktestConfig())

        full_analyzer = BacktestAnalyzer()
        full_analyzer.run_full_analysis(
            ticks=ticks,
            strategy_result=strat_res,
            engine_factory=counting_factory,
            random_baseline_config=RandomBaselineConfig(
                simulations=2, trades_per_simulation=2, seed=42
            ),
            walk_forward_config=WalkForwardConfig(windows_count=2, in_sample_ratio=0.5),
        )
        assert factory_calls["count"] >= 6

    def test_random_baseline_config_validation(self) -> None:
        with pytest.raises(ValueError, match="simulations must be >= 1"):
            RandomBaselineConfig(simulations=0)
        with pytest.raises(ValueError, match="trades_per_simulation must be >= 1"):
            RandomBaselineConfig(trades_per_simulation=0)
        with pytest.raises(ValueError, match="risk_reward_ratio must be positive"):
            RandomBaselineConfig(risk_reward_ratio=-1.0)
        with pytest.raises(ValueError, match="risk_price_distance must be positive"):
            RandomBaselineConfig(risk_price_distance=0.0)
        with pytest.raises(ValueError, match="volume must be positive"):
            RandomBaselineConfig(volume=-0.1)


class TestWalkForwardAnalyzer:
    def test_ac_1205_walk_forward_partitioning_and_causality(self) -> None:
        t0 = _make_utc(10, 0, day=20)
        ticks = [
            _make_tick(bid=2600.0 + i * 0.1, ask=2600.25 + i * 0.1, time=t0 + timedelta(hours=i))
            for i in range(24)
        ]
        cfg = WalkForwardConfig(windows_count=3, in_sample_ratio=0.7)
        analyzer = WalkForwardAnalyzer()
        windows = analyzer.partition_ticks(ticks=ticks, config=cfg)

        assert len(windows) == 3
        for idx, (is_ticks, oos_ticks) in enumerate(windows):
            assert len(is_ticks) > 0
            assert len(oos_ticks) > 0
            assert is_ticks[-1].time <= oos_ticks[0].time
            if idx > 0:
                prev_oos = windows[idx - 1][1]
                assert prev_oos[-1].time <= is_ticks[0].time

    def test_ac_1206_walk_forward_execution_and_efficiency(self) -> None:
        t0 = _make_utc(10, 0, day=20)
        ticks = [
            _make_tick(bid=2600.0, ask=2600.25, time=t0 + timedelta(minutes=10 * i))
            for i in range(30)
        ]
        orders = [
            BacktestOrder(
                order_id=f"ORD-{i}",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_MARKET,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2595.0,
                take_profit=2610.0,
                created_at=t0 + timedelta(minutes=10 * i),
            )
            for i in range(30)
        ]
        analyzer = WalkForwardAnalyzer()
        cfg = WalkForwardConfig(windows_count=2, in_sample_ratio=0.6)
        result = analyzer.analyze(ticks=ticks, orders=orders, config=cfg)

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) == 2
        assert math.isfinite(result.overall_wfe)
        for w in result.windows:
            assert isinstance(w, WalkForwardWindow)
            assert isinstance(w.in_sample_result, BacktestResult)
            assert isinstance(w.out_of_sample_result, BacktestResult)
            assert math.isfinite(w.wfe)

    def test_ac_1207_walk_forward_validation_errors(self) -> None:
        analyzer = WalkForwardAnalyzer()
        t0 = _make_utc(10, 0)
        ticks = [_make_tick(bid=2600.0, ask=2600.25, time=t0)]

        with pytest.raises(ValueError, match="Insufficient ticks to partition"):
            analyzer.partition_ticks(ticks=ticks, config=WalkForwardConfig(windows_count=3))

        with pytest.raises(ValueError, match="windows_count must be >= 1"):
            WalkForwardConfig(windows_count=0)

        with pytest.raises(ValueError, match="in_sample_ratio must be between 0.1 and 0.9"):
            WalkForwardConfig(in_sample_ratio=0.05)


class TestBacktestAnalyzer:
    def test_ac_1208_consolidated_report_and_markdown(self) -> None:
        t0 = _make_utc(10, 0)
        trades = [
            _create_mock_trade("T1", 100.0, 2.0, t0, t0 + timedelta(minutes=10)),
            _create_mock_trade(
                "T2", -50.0, -1.0, t0 + timedelta(minutes=20), t0 + timedelta(minutes=30)
            ),
        ]
        strategy_result = _create_sample_backtest_result(trades)

        ticks = [
            _make_tick(bid=2600.0, ask=2600.25, time=t0 + timedelta(minutes=i)) for i in range(40)
        ]

        analyzer = BacktestAnalyzer()
        report = analyzer.run_full_analysis(
            strategy_result=strategy_result,
            ticks=ticks,
            monte_carlo_config=MonteCarloConfig(simulations=50, seed=42),
            random_baseline_config=RandomBaselineConfig(
                simulations=10, seed=42, trades_per_simulation=2
            ),
            walk_forward_config=WalkForwardConfig(windows_count=2, in_sample_ratio=0.7),
        )

        assert isinstance(report, BacktestAnalysisReport)
        assert report.strategy_result is strategy_result
        assert report.monte_carlo is not None
        assert report.random_baseline is not None
        assert report.walk_forward is not None

        md = report.to_markdown()
        assert isinstance(md, str)
        assert "# Relatório de Análise Estatística de Backtest" in md
        assert "## 1. Desempenho da Estratégia" in md
        assert "## 2. Simulação de Monte Carlo" in md
        assert "## 3. Comparativo com Baseline Aleatório" in md
        assert "## 4. Análise Walk-Forward" in md
        assert "## 5. Parecer do Portão de Decisão Humana 1" in md

    def test_backtest_analyzer_markdown_verdicts(self) -> None:
        t0 = _make_utc(10, 0)
        trades_200 = [
            _create_mock_trade(f"T{i}", 10.0, 1.0, t0, t0 + timedelta(minutes=1))
            for i in range(200)
        ]
        passed_strategy = _create_sample_backtest_result(trades_200)
        passed_base = RandomBaselineResult(
            simulations=100,
            baseline_expectancy_r_mean=0.1,
            baseline_expectancy_r_std=0.05,
            baseline_profit_factor_mean=1.0,
            baseline_win_rate_mean=0.5,
            strategy_expectancy_r=1.0,
            strategy_profit_factor=2.0,
            p_value_expectancy_r=0.01,
            p_value_profit_factor=0.01,
            statistically_significant=True,
        )
        wf_passed = WalkForwardResult(
            windows=(),
            overall_wfe=0.75,
            combined_oos_trades=tuple(trades_200),
            combined_oos_result=passed_strategy,
        )
        mc_passed = MonteCarloResult(
            simulations=100,
            ruin_probability=0.005,
            median_max_drawdown_usd=200.0,
            p95_max_drawdown_usd=400.0,
            p99_max_drawdown_usd=600.0,
            median_max_drawdown_pct=2.0,
            p95_max_drawdown_pct=4.0,
            p99_max_drawdown_pct=6.0,
            median_final_balance=11000.0,
            p5_final_balance=10500.0,
            p95_final_balance=12000.0,
        )

        report_ok = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_passed,
            monte_carlo=mc_passed,
        )
        md_ok = report_ok.to_markdown()
        assert "APROVADO para avanço" in md_ok

        report_no_wf = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            monte_carlo=mc_passed,
        )
        assert "PENDENTE: Análise Walk-Forward ausente" in report_no_wf.to_markdown()

        report_no_mc = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_passed,
        )
        assert "PENDENTE: Simulação de Monte Carlo ausente" in report_no_mc.to_markdown()

        report_no_base = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            walk_forward=wf_passed,
            monte_carlo=mc_passed,
        )
        assert "PENDENTE: Análise de baseline aleatório ausente" in report_no_base.to_markdown()

        failed_base = RandomBaselineResult(
            simulations=100,
            baseline_expectancy_r_mean=0.1,
            baseline_expectancy_r_std=0.05,
            baseline_profit_factor_mean=1.0,
            baseline_win_rate_mean=0.5,
            strategy_expectancy_r=0.05,
            strategy_profit_factor=0.9,
            p_value_expectancy_r=0.8,
            p_value_profit_factor=0.8,
            statistically_significant=False,
        )
        report_failed = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=failed_base,
            walk_forward=wf_passed,
            monte_carlo=mc_passed,
        )
        assert "Estratégia não superou estatisticamente" in report_failed.to_markdown()

        oos_trades_insufficient = trades_200[:150]
        wf_oos_insufficient = WalkForwardResult(
            windows=(),
            overall_wfe=0.75,
            combined_oos_trades=tuple(oos_trades_insufficient),
            combined_oos_result=_create_sample_backtest_result(oos_trades_insufficient),
        )
        report_pending_oos = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_oos_insufficient,
            monte_carlo=mc_passed,
        )
        md_pending_oos = report_pending_oos.to_markdown()
        assert "PENDENTE: Amostra com 150 trades fora da amostra (OOS)" in md_pending_oos

        wf_oos_empty = WalkForwardResult(
            windows=(),
            overall_wfe=0.75,
            combined_oos_trades=(),
            combined_oos_result=None,
        )
        report_empty_oos = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_oos_empty,
            monte_carlo=mc_passed,
        )
        md_empty_oos = report_empty_oos.to_markdown()
        assert "(0 trades fora da amostra (OOS))" in md_empty_oos
        assert "PENDENTE: Amostra com 0 trades fora da amostra (OOS)" in md_empty_oos

        trades_loss = [
            _create_mock_trade(f"TL{i}", -10.0, -1.0, t0, t0 + timedelta(minutes=1))
            for i in range(200)
        ]
        wf_oos_loss = WalkForwardResult(
            windows=(),
            overall_wfe=0.75,
            combined_oos_trades=tuple(trades_loss),
            combined_oos_result=_create_sample_backtest_result(trades_loss),
        )
        rep_unprofitable_oos = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_oos_loss,
            monte_carlo=mc_passed,
        )
        assert "não é lucrativa fora da amostra (OOS)" in rep_unprofitable_oos.to_markdown()

        rep_low_wfe = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=WalkForwardResult(
                windows=(),
                overall_wfe=0.2,
                combined_oos_trades=tuple(trades_200),
                combined_oos_result=passed_strategy,
            ),
            monte_carlo=mc_passed,
        )
        assert "Eficiência Walk-Forward insatisfatória" in rep_low_wfe.to_markdown()

        rep_tail_risk = BacktestAnalysisReport(
            strategy_result=passed_strategy,
            random_baseline=passed_base,
            walk_forward=wf_passed,
            monte_carlo=MonteCarloResult(
                simulations=100,
                ruin_probability=0.05,
                median_max_drawdown_usd=500.0,
                p95_max_drawdown_usd=1000.0,
                p99_max_drawdown_usd=2500.0,
                median_max_drawdown_pct=5.0,
                p95_max_drawdown_pct=10.0,
                p99_max_drawdown_pct=25.0,
                median_final_balance=10000.0,
                p5_final_balance=9000.0,
                p95_final_balance=12000.0,
            ),
        )
        assert "Risco de cauda excessivo" in rep_tail_risk.to_markdown()


def test_percentile_helper() -> None:
    assert _percentile([], 50.0) == 0.0
    assert _percentile([42.0], 50.0) == 42.0
    assert _percentile([10.0, 20.0, 30.0], 50.0) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 0.0) == 10.0
    assert _percentile([10.0, 20.0, 30.0], 100.0) == 30.0


def test_monte_carlo_ruin_triggered() -> None:
    t0 = _make_utc(10, 0)
    trades = [
        _create_mock_trade("T1", -500.0, -1.0, t0, t0 + timedelta(minutes=10)),
        _create_mock_trade(
            "T2", -500.0, -1.0, t0 + timedelta(minutes=20), t0 + timedelta(minutes=30)
        ),
    ]
    sim = MonteCarloSimulator()
    res = sim.simulate(
        trades,
        initial_balance=10000.0,
        config=MonteCarloConfig(simulations=20, ruin_drawdown_pct=5.0),
    )
    assert res.ruin_probability > 0.0


def test_random_baseline_insufficient_ticks_and_single_sim() -> None:
    t0 = _make_utc(10, 0)
    res_sample = _create_sample_backtest_result([])
    analyzer = RandomBaselineAnalyzer()
    res = analyzer.analyze([], res_sample)
    assert res.simulations == 100
    assert res.p_value_expectancy_r == 1.0
    assert res.p_value_profit_factor == 1.0
    assert not res.statistically_significant

    ticks = [
        _make_tick(2600.0, 2600.25, t0),
        _make_tick(2601.0, 2601.25, t0 + timedelta(minutes=1)),
        _make_tick(2602.0, 2602.25, t0 + timedelta(minutes=2)),
    ]
    res_single = analyzer.analyze(
        ticks, res_sample, config=RandomBaselineConfig(simulations=1, trades_per_simulation=1)
    )
    assert res_single.simulations == 1
    assert res_single.baseline_expectancy_r_std == 0.0


def test_walk_forward_negative_and_zero_expectancy() -> None:
    t0 = _make_utc(10, 0)
    ticks = [_make_tick(2600.0, 2600.25, t0 + timedelta(minutes=i)) for i in range(20)]
    analyzer = WalkForwardAnalyzer()
    res = analyzer.analyze(
        ticks, [], [], config=WalkForwardConfig(windows_count=2, in_sample_ratio=0.5)
    )
    assert res.overall_wfe == 0.0
    assert len(res.windows) == 2


def test_walk_forward_split_point_boundary() -> None:
    t0 = _make_utc(10, 0)
    ticks = [
        _make_tick(2600.0, 2600.25, t0),
        _make_tick(2601.0, 2601.25, t0 + timedelta(minutes=1)),
    ]
    analyzer = WalkForwardAnalyzer()
    windows = analyzer.partition_ticks(
        ticks, WalkForwardConfig(windows_count=1, in_sample_ratio=0.9)
    )
    assert len(windows) == 1
    assert len(windows[0][0]) == 1
    assert len(windows[0][1]) == 1


class _MockEngine(BacktestEngine):
    def __init__(self, fn: Callable[[], BacktestResult]) -> None:
        self._fn = fn

    def run(
        self,
        ticks: Sequence[Tick],
        setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        candles: Sequence[Candle] | None = None,
    ) -> BacktestResult:
        return self._fn()


def test_walk_forward_mock_engine_branches() -> None:
    t0 = _make_utc(10, 0)
    ticks = [_make_tick(2600.0, 2600.25, t0 + timedelta(minutes=i)) for i in range(20)]

    trade_win = _create_mock_trade("TW", 100.0, 2.0, t0, t0 + timedelta(days=1))
    trade_loss = _create_mock_trade(
        "TL", -50.0, -1.0, t0 + timedelta(days=2), t0 + timedelta(days=3)
    )
    trade_be = _create_mock_trade("TBE", 0.0, 0.0, t0 + timedelta(days=4), t0 + timedelta(days=5))

    is_res = _create_sample_backtest_result([trade_win])
    oos_res = _create_sample_backtest_result([trade_win, trade_loss, trade_be])

    calls = 0

    def mock_factory_mixed() -> BacktestEngine:
        def _get_res() -> BacktestResult:
            nonlocal calls
            calls += 1
            return is_res if (calls % 2 != 0) else oos_res

        return _MockEngine(_get_res)

    analyzer = WalkForwardAnalyzer()
    res = analyzer.analyze(
        ticks=ticks,
        setups=[],
        config=WalkForwardConfig(windows_count=1, in_sample_ratio=0.5),
        engine_factory=mock_factory_mixed,
    )
    assert res.overall_wfe > 0.0
    assert res.combined_oos_result is not None
    assert res.combined_oos_result.total_trades == 3

    calls_flip = 0

    def mock_factory_flip() -> BacktestEngine:
        def _get_res() -> BacktestResult:
            nonlocal calls_flip
            calls_flip += 1
            return (
                _create_sample_backtest_result([trade_loss])
                if (calls_flip % 2 != 0)
                else _create_sample_backtest_result([trade_win])
            )

        return _MockEngine(_get_res)

    res_flip = analyzer.analyze(
        ticks=ticks,
        setups=[],
        config=WalkForwardConfig(windows_count=1, in_sample_ratio=0.5),
        engine_factory=mock_factory_flip,
    )
    assert res_flip.overall_wfe == 0.0

    def mock_factory_wins() -> BacktestEngine:
        return _MockEngine(lambda: _create_sample_backtest_result([trade_win]))

    res_wins = analyzer.analyze(
        ticks=ticks,
        setups=[],
        config=WalkForwardConfig(windows_count=1, in_sample_ratio=0.5),
        engine_factory=mock_factory_wins,
    )
    assert res_wins.combined_oos_result is not None
    assert res_wins.combined_oos_result.profit_factor == float("inf")

    report_with_oos = BacktestAnalysisReport(
        strategy_result=is_res,
        walk_forward=res_wins,
    )
    md = report_with_oos.to_markdown()
    assert "- **Trades Fora da Amostra (OOS)**: 1" in md


def test_random_baseline_repeated_timestamps_advances_correctly() -> None:
    t0 = _make_utc(10, 0)
    ticks: list[Tick] = []
    p = 2600.0
    delta = 1.0
    for i in range(200):
        if p >= 2615.0:
            delta = -1.0
        elif p <= 2585.0:
            delta = 1.0
        p += delta
        ticks.append(_make_tick(bid=p, ask=p + 0.25, time=t0 + timedelta(seconds=i // 2)))

    strategy_trades = [
        _create_mock_trade(f"T{i}", 10.0, 1.0, t0, t0 + timedelta(seconds=1)) for i in range(2)
    ]
    strategy_res = _create_sample_backtest_result(strategy_trades)

    recorded_brokers: list[object] = []
    recorded_results: list[BacktestResult] = []

    def spy_factory() -> BacktestEngine:
        real_engine = BacktestEngine(BacktestConfig())
        recorded_brokers.append(real_engine.broker)

        class _Spy(BacktestEngine):
            def run(
                self,
                ticks: Sequence[Tick],
                setups: Sequence[TradeSetup] | None = None,
                orders: Sequence[BacktestOrder] | None = None,
                candles: Sequence[Candle] | None = None,
            ) -> BacktestResult:
                res = real_engine.run(ticks=ticks, setups=setups, orders=orders, candles=candles)
                recorded_results.append(res)
                return res

        return _Spy()

    analyzer = RandomBaselineAnalyzer()
    res = analyzer.analyze(
        ticks=ticks,
        strategy_result=strategy_res,
        config=RandomBaselineConfig(simulations=2, trades_per_simulation=3, seed=42),
        engine_factory=spy_factory,
    )
    assert res.simulations == 2
    assert len(recorded_results) == 2
    for sim_idx in range(2):
        assert recorded_results[sim_idx].total_trades == 3
        broker = recorded_brokers[sim_idx]
        assert len(getattr(broker, "rejected_orders", [])) == 0


def test_walk_forward_all_breakeven_oos_trades() -> None:
    t0 = _make_utc(10, 0)
    ticks = [_make_tick(2600.0, 2600.25, t0 + timedelta(minutes=i)) for i in range(20)]
    trade_be = _create_mock_trade("TBE", 0.0, 0.0, t0, t0 + timedelta(days=1))

    analyzer = WalkForwardAnalyzer()
    res = analyzer.analyze(
        ticks=ticks,
        setups=[],
        config=WalkForwardConfig(windows_count=1, in_sample_ratio=0.5),
        engine_factory=lambda: _MockEngine(lambda: _create_sample_backtest_result([trade_be])),
    )
    assert res.combined_oos_result is not None
    assert res.combined_oos_result.profit_factor == 0.0
