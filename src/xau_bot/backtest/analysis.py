import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from xau_bot.backtest.engine import BacktestEngine
from xau_bot.backtest.models import (
    BacktestConfig,
    BacktestOrder,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    OrderType,
)
from xau_bot.domain.models import Direction, Tick, TradeSetup


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    simulations: int = 1000
    ruin_drawdown_pct: float = 10.0
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {self.simulations}")
        if not (0.0 <= self.ruin_drawdown_pct <= 100.0):
            raise ValueError(
                f"ruin_drawdown_pct must be between 0 and 100, got {self.ruin_drawdown_pct}"
            )


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    simulations: int
    ruin_probability: float
    median_max_drawdown_usd: float
    p95_max_drawdown_usd: float
    p99_max_drawdown_usd: float
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    p99_max_drawdown_pct: float
    median_final_balance: float
    p5_final_balance: float
    p95_final_balance: float


class MonteCarloSimulator:
    def simulate(
        self,
        trades: Sequence[BacktestTrade],
        initial_balance: float = 10000.0,
        config: MonteCarloConfig | None = None,
    ) -> MonteCarloResult:
        cfg = config if config is not None else MonteCarloConfig()
        if not trades:
            return MonteCarloResult(
                simulations=cfg.simulations,
                ruin_probability=0.0,
                median_max_drawdown_usd=0.0,
                p95_max_drawdown_usd=0.0,
                p99_max_drawdown_usd=0.0,
                median_max_drawdown_pct=0.0,
                p95_max_drawdown_pct=0.0,
                p99_max_drawdown_pct=0.0,
                median_final_balance=initial_balance,
                p5_final_balance=initial_balance,
                p95_final_balance=initial_balance,
            )

        rng = random.Random(cfg.seed)
        pnls = [t.net_pnl for t in trades]
        n_trades = len(pnls)

        sim_max_dd_usds: list[float] = []
        sim_max_dd_pcts: list[float] = []
        sim_final_bals: list[float] = []
        ruin_count = 0

        for _ in range(cfg.simulations):
            sampled_pnls = rng.choices(pnls, k=n_trades)
            bal = initial_balance
            peak = bal
            max_dd_usd = 0.0
            max_dd_pct = 0.0

            for pnl in sampled_pnls:
                bal += pnl
                if bal > peak:
                    peak = bal
                dd = peak - bal
                dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
                if dd > max_dd_usd:
                    max_dd_usd = dd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct

            if max_dd_pct >= cfg.ruin_drawdown_pct:
                ruin_count += 1

            sim_max_dd_usds.append(max_dd_usd)
            sim_max_dd_pcts.append(max_dd_pct)
            sim_final_bals.append(bal)

        sim_max_dd_usds.sort()
        sim_max_dd_pcts.sort()
        sim_final_bals.sort()

        return MonteCarloResult(
            simulations=cfg.simulations,
            ruin_probability=round(ruin_count / cfg.simulations, 4),
            median_max_drawdown_usd=round(_percentile(sim_max_dd_usds, 50.0), 2),
            p95_max_drawdown_usd=round(_percentile(sim_max_dd_usds, 95.0), 2),
            p99_max_drawdown_usd=round(_percentile(sim_max_dd_usds, 99.0), 2),
            median_max_drawdown_pct=round(_percentile(sim_max_dd_pcts, 50.0), 2),
            p95_max_drawdown_pct=round(_percentile(sim_max_dd_pcts, 95.0), 2),
            p99_max_drawdown_pct=round(_percentile(sim_max_dd_pcts, 99.0), 2),
            median_final_balance=round(_percentile(sim_final_bals, 50.0), 2),
            p5_final_balance=round(_percentile(sim_final_bals, 5.0), 2),
            p95_final_balance=round(_percentile(sim_final_bals, 95.0), 2),
        )


@dataclass(frozen=True, slots=True)
class RandomBaselineConfig:
    simulations: int = 100
    trades_per_simulation: int | None = None
    risk_reward_ratio: float | None = None
    risk_price_distance: float | None = None
    volume: float | None = None
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {self.simulations}")
        if self.trades_per_simulation is not None and self.trades_per_simulation < 1:
            raise ValueError(
                f"trades_per_simulation must be >= 1, got {self.trades_per_simulation}"
            )
        if self.risk_reward_ratio is not None and self.risk_reward_ratio <= 0:
            raise ValueError(f"risk_reward_ratio must be positive, got {self.risk_reward_ratio}")
        if self.risk_price_distance is not None and self.risk_price_distance <= 0:
            raise ValueError(
                f"risk_price_distance must be positive, got {self.risk_price_distance}"
            )
        if self.volume is not None and self.volume <= 0:
            raise ValueError(f"volume must be positive, got {self.volume}")


@dataclass(frozen=True, slots=True)
class RandomBaselineResult:
    simulations: int
    baseline_expectancy_r_mean: float
    baseline_expectancy_r_std: float
    baseline_profit_factor_mean: float
    baseline_win_rate_mean: float
    strategy_expectancy_r: float
    strategy_profit_factor: float
    p_value_expectancy_r: float
    p_value_profit_factor: float
    statistically_significant: bool


class RandomBaselineAnalyzer:
    def analyze(
        self,
        ticks: Sequence[Tick],
        strategy_result: BacktestResult,
        engine_factory: Callable[[], BacktestEngine] | None = None,
        config: RandomBaselineConfig | None = None,
    ) -> RandomBaselineResult:
        cfg = config if config is not None else RandomBaselineConfig()
        if len(ticks) < 2:
            return RandomBaselineResult(
                simulations=cfg.simulations,
                baseline_expectancy_r_mean=0.0,
                baseline_expectancy_r_std=0.0,
                baseline_profit_factor_mean=0.0,
                baseline_win_rate_mean=0.0,
                strategy_expectancy_r=strategy_result.expectancy_r,
                strategy_profit_factor=strategy_result.profit_factor,
                p_value_expectancy_r=1.0,
                p_value_profit_factor=1.0,
                statistically_significant=False,
            )

        rng = random.Random(cfg.seed)

        strat_sl_dists = [
            abs(t.entry_price - t.initial_sl)
            for t in strategy_result.trades
            if t.initial_sl > 0 and abs(t.entry_price - t.initial_sl) > 1e-4
        ]
        risk_dist = (
            cfg.risk_price_distance
            if cfg.risk_price_distance is not None
            else (sum(strat_sl_dists) / len(strat_sl_dists) if strat_sl_dists else 5.0)
        )

        strat_rrs = [
            abs(t.tp - t.entry_price) / abs(t.entry_price - t.initial_sl)
            for t in strategy_result.trades
            if t.tp > 0 and t.initial_sl > 0 and abs(t.entry_price - t.initial_sl) > 1e-4
        ]
        rr_ratio = (
            cfg.risk_reward_ratio
            if cfg.risk_reward_ratio is not None
            else (sum(strat_rrs) / len(strat_rrs) if strat_rrs else 2.0)
        )

        strat_vols = [t.volume for t in strategy_result.trades if t.volume > 0]
        trade_vol = (
            cfg.volume
            if cfg.volume is not None
            else (sum(strat_vols) / len(strat_vols) if strat_vols else 0.1)
        )

        num_trades = cfg.trades_per_simulation or max(1, strategy_result.total_trades)
        reward_dist = risk_dist * rr_ratio
        n_ticks = len(ticks)

        make_engine = (
            engine_factory
            if engine_factory is not None
            else (lambda: BacktestEngine(config=BacktestConfig()))
        )

        expectancies: list[float] = []
        profit_factors: list[float] = []
        win_rates: list[float] = []

        for sim_idx in range(cfg.simulations):
            orders: list[BacktestOrder] = []
            curr_tick_idx = 0

            while len(orders) < num_trades and curr_tick_idx < n_ticks - 1:
                remaining_trades = num_trades - len(orders)
                remaining_ticks = (n_ticks - 1) - curr_tick_idx
                max_gap = max(0, remaining_ticks // (remaining_trades + 1))
                entry_idx = curr_tick_idx + (rng.randint(0, max_gap) if max_gap > 0 else 0)

                ref_tick = ticks[entry_idx]
                direction = rng.choice([Direction.BULLISH, Direction.BEARISH])

                if direction == Direction.BULLISH:
                    entry_price = ref_tick.ask
                    sl = entry_price - risk_dist
                    tp = entry_price + reward_dist
                    order_type = OrderType.BUY_MARKET
                else:
                    entry_price = ref_tick.bid
                    sl = entry_price + risk_dist
                    tp = entry_price - reward_dist
                    order_type = OrderType.SELL_MARKET

                orders.append(
                    BacktestOrder(
                        order_id=f"RND-{sim_idx}-{len(orders)}",
                        direction=direction,
                        order_type=order_type,
                        volume=trade_vol,
                        limit_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        created_at=ref_tick.time,
                    )
                )

                exit_idx = n_ticks - 1
                for j in range(entry_idx + 1, n_ticks):
                    t_j = ticks[j]
                    if direction == Direction.BULLISH:
                        if t_j.bid <= sl or t_j.bid >= tp:
                            exit_idx = j
                            break
                    else:
                        if t_j.ask >= sl or t_j.ask <= tp:
                            exit_idx = j
                            break

                exit_time = ticks[exit_idx].time
                curr_tick_idx = exit_idx + 1
                while curr_tick_idx < n_ticks and ticks[curr_tick_idx].time <= exit_time:
                    curr_tick_idx += 1

            eng = make_engine()
            res = eng.run(ticks=ticks, orders=orders)
            expectancies.append(res.expectancy_r)
            profit_factors.append(res.profit_factor)
            win_rates.append(res.win_rate)

        mean_exp = sum(expectancies) / len(expectancies)
        if len(expectancies) > 1:
            var_exp = sum((x - mean_exp) ** 2 for x in expectancies) / (len(expectancies) - 1)
            std_exp = math.sqrt(var_exp)
        else:
            std_exp = 0.0

        finite_pfs = [pf for pf in profit_factors if math.isfinite(pf)]
        mean_pf = sum(finite_pfs) / len(finite_pfs) if finite_pfs else 10.0
        mean_wr = sum(win_rates) / len(win_rates)

        exceed_exp = sum(1 for exp in expectancies if exp >= strategy_result.expectancy_r)
        p_val_exp = (exceed_exp + 1) / (len(expectancies) + 1)

        strat_pf = strategy_result.profit_factor
        exceed_pf = sum(1 for pf in profit_factors if pf >= strat_pf)
        p_val_pf = (exceed_pf + 1) / (len(profit_factors) + 1)

        sig = (p_val_exp < 0.05) and (p_val_pf < 0.05)

        return RandomBaselineResult(
            simulations=cfg.simulations,
            baseline_expectancy_r_mean=round(mean_exp, 4),
            baseline_expectancy_r_std=round(std_exp, 4),
            baseline_profit_factor_mean=round(mean_pf, 4),
            baseline_win_rate_mean=round(mean_wr, 4),
            strategy_expectancy_r=round(strategy_result.expectancy_r, 4),
            strategy_profit_factor=round(strategy_result.profit_factor, 4),
            p_value_expectancy_r=round(p_val_exp, 4),
            p_value_profit_factor=round(p_val_pf, 4),
            statistically_significant=sig,
        )


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    windows_count: int = 3
    in_sample_ratio: float = 0.7

    def __post_init__(self) -> None:
        if self.windows_count < 1:
            raise ValueError(f"windows_count must be >= 1, got {self.windows_count}")
        if not (0.1 <= self.in_sample_ratio <= 0.9):
            raise ValueError(
                f"in_sample_ratio must be between 0.1 and 0.9, got {self.in_sample_ratio}"
            )


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult
    wfe: float


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    windows: tuple[WalkForwardWindow, ...]
    overall_wfe: float
    combined_oos_trades: tuple[BacktestTrade, ...]
    combined_oos_result: BacktestResult | None


class WalkForwardAnalyzer:
    def partition_ticks(
        self,
        ticks: Sequence[Tick],
        config: WalkForwardConfig | None = None,
    ) -> list[tuple[list[Tick], list[Tick]]]:
        cfg = config if config is not None else WalkForwardConfig()
        if len(ticks) < cfg.windows_count * 2:
            raise ValueError(
                f"Insufficient ticks to partition into {cfg.windows_count} "
                f"windows: got {len(ticks)}"
            )

        block_size = len(ticks) // cfg.windows_count
        windows: list[tuple[list[Tick], list[Tick]]] = []

        for w_idx in range(cfg.windows_count):
            start_i = w_idx * block_size
            end_i = (w_idx + 1) * block_size if w_idx < cfg.windows_count - 1 else len(ticks)
            block = list(ticks[start_i:end_i])

            split_point = max(1, min(len(block) - 1, int(len(block) * cfg.in_sample_ratio)))

            is_slice = block[:split_point]
            oos_slice = block[split_point:]
            windows.append((is_slice, oos_slice))

        return windows

    def analyze(
        self,
        ticks: Sequence[Tick],
        setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        engine_factory: Callable[[], BacktestEngine] | None = None,
        config: WalkForwardConfig | None = None,
    ) -> WalkForwardResult:
        cfg = config if config is not None else WalkForwardConfig()
        tick_windows = self.partition_ticks(ticks=ticks, config=cfg)

        def make_engine() -> BacktestEngine:
            return engine_factory() if engine_factory is not None else BacktestEngine()

        all_setups = list(setups) if setups is not None else []
        all_orders = list(orders) if orders is not None else []

        computed_windows: list[WalkForwardWindow] = []
        combined_oos_trades: list[BacktestTrade] = []

        for idx, (is_ticks, oos_ticks) in enumerate(tick_windows, 1):
            is_start = is_ticks[0].time
            is_end = is_ticks[-1].time
            oos_start = oos_ticks[0].time
            oos_end = oos_ticks[-1].time

            is_setups = [s for s in all_setups if is_start <= s.confirmed_time < oos_start]
            is_orders = [o for o in all_orders if is_start <= o.created_at < oos_start]
            oos_setups = [s for s in all_setups if oos_start <= s.confirmed_time <= oos_end]
            oos_orders = [o for o in all_orders if oos_start <= o.created_at <= oos_end]

            is_eng = make_engine()
            is_res = is_eng.run(ticks=is_ticks, setups=is_setups, orders=is_orders)

            oos_eng = make_engine()
            oos_res = oos_eng.run(ticks=oos_ticks, setups=oos_setups, orders=oos_orders)

            combined_oos_trades.extend(oos_res.trades)

            if is_res.expectancy_r > 0:
                wfe = round(oos_res.expectancy_r / is_res.expectancy_r, 4)
            else:
                wfe = 0.0

            computed_windows.append(
                WalkForwardWindow(
                    window_id=idx,
                    in_sample_start=is_start,
                    in_sample_end=is_end,
                    out_of_sample_start=oos_start,
                    out_of_sample_end=oos_end,
                    in_sample_result=is_res,
                    out_of_sample_result=oos_res,
                    wfe=wfe,
                )
            )

        positive_is_windows = [w for w in computed_windows if w.in_sample_result.expectancy_r > 0]
        overall_wfe = (
            round(sum(w.wfe for w in positive_is_windows) / len(positive_is_windows), 4)
            if positive_is_windows
            else 0.0
        )

        combined_oos_res: BacktestResult | None = None
        if combined_oos_trades:
            init_bal = computed_windows[0].out_of_sample_result.initial_balance
            cur_bal = init_bal
            peak = cur_bal
            max_dd_usd = 0.0
            max_dd_pct = 0.0
            eq_points = [
                EquityPoint(
                    timestamp=computed_windows[0].out_of_sample_start,
                    balance=init_bal,
                    equity=init_bal,
                    drawdown_usd=0.0,
                    drawdown_pct=0.0,
                )
            ]
            winning = 0
            losing = 0
            be = 0

            for t in combined_oos_trades:
                cur_bal += t.net_pnl
                if cur_bal > peak:
                    peak = cur_bal
                dd = peak - cur_bal
                dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
                if dd > max_dd_usd:
                    max_dd_usd = dd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
                eq_points.append(
                    EquityPoint(
                        timestamp=t.exit_time,
                        balance=cur_bal,
                        equity=cur_bal,
                        drawdown_usd=dd,
                        drawdown_pct=dd_pct,
                    )
                )
                if t.net_pnl > 0:
                    winning += 1
                elif t.net_pnl < 0:
                    losing += 1
                else:
                    be += 1

            total_trades = len(combined_oos_trades)
            gross_p = sum(t.net_pnl for t in combined_oos_trades if t.net_pnl > 0)
            gross_l = abs(sum(t.net_pnl for t in combined_oos_trades if t.net_pnl < 0))
            if gross_l > 0:
                pf = gross_p / gross_l
            elif gross_p > 0:
                pf = float("inf")
            else:
                pf = 0.0

            daily_close_equity: dict[date, float] = {}
            for ep in eq_points:
                daily_close_equity[ep.timestamp.date()] = ep.equity

            daily_vals = list(daily_close_equity.values())
            if len(daily_vals) >= 2:
                daily_returns = [
                    (daily_vals[i] - daily_vals[i - 1]) / daily_vals[i - 1]
                    for i in range(1, len(daily_vals))
                    if daily_vals[i - 1] > 0
                ]
                if len(daily_returns) >= 2:
                    mean_ret = sum(daily_returns) / len(daily_returns)
                    var_ret = sum((r - mean_ret) ** 2 for r in daily_returns) / (
                        len(daily_returns) - 1
                    )
                    std_ret = math.sqrt(var_ret)
                    sharpe = (
                        round((mean_ret / std_ret) * math.sqrt(252.0), 4) if std_ret > 0 else 0.0
                    )
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            combined_oos_res = BacktestResult(
                trades=tuple(combined_oos_trades),
                equity_curve=tuple(eq_points),
                initial_balance=init_bal,
                final_balance=round(cur_bal, 2),
                total_net_pnl=round(cur_bal - init_bal, 2),
                total_trades=total_trades,
                winning_trades=winning,
                losing_trades=losing,
                breakeven_trades=be,
                win_rate=round(winning / total_trades, 4) if total_trades > 0 else 0.0,
                profit_factor=round(pf, 4),
                max_drawdown_usd=round(max_dd_usd, 2),
                max_drawdown_pct=round(max_dd_pct, 2),
                sharpe_ratio=sharpe,
                expectancy_r=round(sum(t.return_r for t in combined_oos_trades) / total_trades, 4)
                if total_trades > 0
                else 0.0,
                average_trade_duration_seconds=round(
                    sum(t.duration_seconds for t in combined_oos_trades) / total_trades, 1
                )
                if total_trades > 0
                else 0.0,
            )

        return WalkForwardResult(
            windows=tuple(computed_windows),
            overall_wfe=overall_wfe,
            combined_oos_trades=tuple(combined_oos_trades),
            combined_oos_result=combined_oos_res,
        )


@dataclass(frozen=True, slots=True)
class Gate1Evaluation:
    passed: bool
    label: str
    verdict: str
    checks: dict[str, bool]
    eval_trades: int
    eval_scope: str
    eval_pnl: float
    eval_pf: float
    eval_expectancy_r: float


@dataclass(frozen=True, slots=True)
class BacktestAnalysisReport:
    strategy_result: BacktestResult
    monte_carlo: MonteCarloResult | None = None
    random_baseline: RandomBaselineResult | None = None
    walk_forward: WalkForwardResult | None = None

    def to_markdown(self, min_trades: int = 200) -> str:
        s = self.strategy_result
        lines: list[str] = [
            "# Relatório de Análise Estatística de Backtest",
            "",
            "## 1. Desempenho da Estratégia",
            f"- **Saldo Inicial**: US$ {s.initial_balance:.2f}",
            f"- **Saldo Final**: US$ {s.final_balance:.2f}",
            f"- **Lucro Líquido**: US$ {s.total_net_pnl:.2f}",
            f"- **Total de Trades**: {s.total_trades}",
            f"- **Taxa de Acerto**: {s.win_rate * 100:.1f}%",
            f"- **Fator de Lucro**: {s.profit_factor:.2f}",
            f"- **Expectativa em R**: {s.expectancy_r:.2f}R",
            f"- **Drawdown Máximo**: US$ {s.max_drawdown_usd:.2f} ({s.max_drawdown_pct:.2f}%)",
            f"- **Índice Sharpe Anualizado**: {s.sharpe_ratio:.2f}",
            "",
        ]

        if self.monte_carlo is not None:
            mc = self.monte_carlo
            lines.extend(
                [
                    "## 2. Simulação de Monte Carlo",
                    f"- **Simulações**: {mc.simulations}",
                    f"- **Probabilidade de Ruína (DD Limiar)**: {mc.ruin_probability * 100:.2f}%",
                    f"- **Drawdown Máximo Mediano**: US$ {mc.median_max_drawdown_usd:.2f} "
                    f"({mc.median_max_drawdown_pct:.2f}%)",
                    f"- **Drawdown Máximo P95**: US$ {mc.p95_max_drawdown_usd:.2f} "
                    f"({mc.p95_max_drawdown_pct:.2f}%)",
                    f"- **Drawdown Máximo P99**: US$ {mc.p99_max_drawdown_usd:.2f} "
                    f"({mc.p99_max_drawdown_pct:.2f}%)",
                    f"- **Saldo Final Mediano**: US$ {mc.median_final_balance:.2f}",
                    f"- **Intervalo de Confiança 90% (P5 - P95)**: "
                    f"US$ {mc.p5_final_balance:.2f} - US$ {mc.p95_final_balance:.2f}",
                    "",
                ]
            )

        if self.random_baseline is not None:
            rb = self.random_baseline
            sig_text = "SIM (p < 0.05)" if rb.statistically_significant else "NÃO (p >= 0.05)"
            lines.extend(
                [
                    "## 3. Comparativo com Baseline Aleatório",
                    f"- **Simulações Aleatórias**: {rb.simulations}",
                    f"- **Expectativa Estratégia**: {rb.strategy_expectancy_r:.2f}R",
                    f"- **Fator de Lucro Estratégia**: {rb.strategy_profit_factor:.2f}",
                    f"- **Expectativa Média Baseline**: {rb.baseline_expectancy_r_mean:.2f}R "
                    f"(+/- {rb.baseline_expectancy_r_std:.2f})",
                    f"- **Fator de Lucro Médio Baseline**: {rb.baseline_profit_factor_mean:.2f}",
                    f"- **Taxa de Acerto Média Baseline**: {rb.baseline_win_rate_mean * 100:.1f}%",
                    f"- **p-valor Expectativa**: {rb.p_value_expectancy_r:.4f}",
                    f"- **p-valor Fator de Lucro**: {rb.p_value_profit_factor:.4f}",
                    f"- **Estatisticamente Significativo**: {sig_text}",
                    "",
                ]
            )

        if self.walk_forward is not None:
            wf = self.walk_forward
            lines.extend(
                [
                    "## 4. Análise Walk-Forward",
                    f"- **Janelas Analisadas**: {len(wf.windows)}",
                    f"- **Eficiência Walk-Forward Geral (WFE)**: {wf.overall_wfe:.2f}",
                ]
            )
            if wf.combined_oos_result is not None:
                oos = wf.combined_oos_result
                lines.extend(
                    [
                        f"- **Trades Fora da Amostra (OOS)**: {oos.total_trades}",
                        f"- **PnL Líquido OOS**: US$ {oos.total_net_pnl:.2f}",
                        f"- **Fator de Lucro OOS**: {oos.profit_factor:.2f}",
                        f"- **Drawdown Máximo OOS**: US$ {oos.max_drawdown_usd:.2f} "
                        f"({oos.max_drawdown_pct:.2f}%)",
                    ]
                )
            lines.append("")

        lines.append("## 5. Parecer do Portão de Decisão Humana 1")

        gate_eval = self.evaluate_gate_1(min_trades=min_trades)
        checks = gate_eval.checks
        eval_trades = gate_eval.eval_trades
        eval_scope = gate_eval.eval_scope
        eval_pnl = gate_eval.eval_pnl
        eval_pf = gate_eval.eval_pf
        eval_expectancy_r = gate_eval.eval_expectancy_r
        gate_verdict_text = f"{gate_eval.label} {gate_eval.verdict}"

        lines.extend(
            [
                f"- **Amostra Mínima (>= {min_trades} trades)**: "
                f"{'APROVADO' if checks.get('min_trades', False) else 'REPROVADO'} "
                f"({eval_trades} trades {eval_scope})",
                f"- **Lucratividade Líquida (PnL > 0, PF > 1.0, Exp > 0R)**: "
                f"{'APROVADO' if checks.get('profitable', False) else 'REPROVADO'} "
                f"(US$ {eval_pnl:.2f}, PF {eval_pf:.2f}, Exp {eval_expectancy_r:.2f}R)",
                f"- **Superioridade vs Baseline (p < 0.05)**: "
                f"{'APROVADO' if checks.get('beats_baseline', False) else 'REPROVADO'}"
                + (" (não executado/ausente)" if self.random_baseline is None else ""),
                f"- **Eficiência Walk-Forward (WFE >= 0.50)**: "
                f"{'APROVADO' if checks.get('walk_forward_efficiency', False) else 'REPROVADO'}"
                + (
                    f" ({self.walk_forward.overall_wfe:.2f})"
                    if self.walk_forward is not None
                    else " (não executado/ausente)"
                ),
                f"- **Controle de Risco de Cauda (Ruína <= 1%, DD P99 <= 20%)**: "
                f"{'APROVADO' if checks.get('tail_risk', False) else 'REPROVADO'}"
                + (
                    f" (Ruína {self.monte_carlo.ruin_probability * 100:.1f}%, "
                    f"DD P99 {self.monte_carlo.p99_max_drawdown_pct:.1f}%)"
                    if self.monte_carlo is not None
                    else " (não executado/ausente)"
                ),
            ]
        )

        lines.append(f"- **Veredito Proposto**: {gate_verdict_text}")
        lines.append("")

        return "\n".join(lines)

    def evaluate_gate_1(self, min_trades: int = 200) -> Gate1Evaluation:
        s = self.strategy_result
        if self.walk_forward is not None:
            if self.walk_forward.combined_oos_result is not None:
                eval_trades = self.walk_forward.combined_oos_result.total_trades
                eval_pnl = self.walk_forward.combined_oos_result.total_net_pnl
                eval_pf = self.walk_forward.combined_oos_result.profit_factor
                eval_expectancy_r = self.walk_forward.combined_oos_result.expectancy_r
            else:
                eval_trades = len(self.walk_forward.combined_oos_trades)
                eval_pnl = 0.0
                eval_pf = 0.0
                eval_expectancy_r = 0.0
            eval_scope = "fora da amostra (OOS)"
        else:
            eval_trades = s.total_trades
            eval_pnl = s.total_net_pnl
            eval_pf = s.profit_factor
            eval_expectancy_r = s.expectancy_r
            eval_scope = "amostra completa"

        trades_ok = eval_trades >= min_trades
        profitable_ok = eval_pnl > 0.0 and eval_pf > 1.0 and eval_expectancy_r > 0.0
        baseline_ok = (
            self.random_baseline is not None and self.random_baseline.statistically_significant
        )
        wfe_ok = self.walk_forward is not None and self.walk_forward.overall_wfe >= 0.50
        tail_risk_ok = (
            self.monte_carlo is not None
            and self.monte_carlo.ruin_probability <= 0.01
            and self.monte_carlo.p99_max_drawdown_pct <= 20.0
        )

        checks = {
            "min_trades": trades_ok,
            "profitable": profitable_ok,
            "beats_baseline": baseline_ok,
            "walk_forward_efficiency": wfe_ok,
            "tail_risk": tail_risk_ok,
        }

        if trades_ok and profitable_ok and baseline_ok and wfe_ok and tail_risk_ok:
            verdict = "APROVADO para avanço à fase de execução e testes demo."
            label = "[APROVADO]"
            passed = True
        elif not trades_ok:
            label = "[AMOSTRA_INSUFICIENTE]"
            passed = False
            verdict = (
                f"PENDENTE: Amostra com {eval_trades} trades {eval_scope} "
                f"(< {min_trades} trades mínimos recomendados para o Portão 1)."
            )
        elif not baseline_ok:
            label = "[REPROVADO]"
            passed = False
            if self.random_baseline is None:
                verdict = (
                    "PENDENTE: Análise de baseline aleatório ausente "
                    "(evidência estatística obrigatória para o Portão 1)."
                )
            else:
                verdict = (
                    "REVISÃO NECESSÁRIA: Estratégia não superou estatisticamente "
                    "os custos ou baseline aleatório."
                )
        elif not profitable_ok:
            label = "[REPROVADO]"
            passed = False
            verdict = (
                f"REVISÃO NECESSÁRIA: Estratégia não é lucrativa {eval_scope} "
                f"(PnL US$ {eval_pnl:.2f} <= 0 ou PF {eval_pf:.2f} <= 1.0 "
                f"ou Exp {eval_expectancy_r:.2f}R <= 0)."
            )
        elif not wfe_ok:
            label = "[REPROVADO]"
            passed = False
            if self.walk_forward is None:
                verdict = (
                    "PENDENTE: Análise Walk-Forward ausente "
                    "(evidência OOS obrigatória para o Portão 1)."
                )
            else:
                verdict = (
                    f"REVISÃO NECESSÁRIA: Eficiência Walk-Forward insatisfatória "
                    f"(WFE {self.walk_forward.overall_wfe:.2f} < 0.50)."
                )
        else:
            label = "[REPROVADO]"
            passed = False
            if self.monte_carlo is None:
                verdict = (
                    "PENDENTE: Simulação de Monte Carlo ausente "
                    "(controle de risco de cauda obrigatório para o Portão 1)."
                )
            else:
                verdict = (
                    "REVISÃO NECESSÁRIA: Risco de cauda excessivo no Monte Carlo "
                    f"(Ruína {self.monte_carlo.ruin_probability * 100:.1f}% > 1% "
                    f"ou DD P99 {self.monte_carlo.p99_max_drawdown_pct:.1f}% > 20%)."
                )

        return Gate1Evaluation(
            passed=passed,
            label=label,
            verdict=verdict,
            checks=checks,
            eval_trades=eval_trades,
            eval_scope=eval_scope,
            eval_pnl=eval_pnl,
            eval_pf=eval_pf,
            eval_expectancy_r=eval_expectancy_r,
        )


class BacktestAnalyzer:
    def __init__(
        self,
        monte_carlo_simulator: MonteCarloSimulator | None = None,
        random_baseline_analyzer: RandomBaselineAnalyzer | None = None,
        walk_forward_analyzer: WalkForwardAnalyzer | None = None,
    ) -> None:
        self.monte_carlo_simulator = (
            monte_carlo_simulator if monte_carlo_simulator is not None else MonteCarloSimulator()
        )
        self.random_baseline_analyzer = (
            random_baseline_analyzer
            if random_baseline_analyzer is not None
            else RandomBaselineAnalyzer()
        )
        self.walk_forward_analyzer = (
            walk_forward_analyzer if walk_forward_analyzer is not None else WalkForwardAnalyzer()
        )

    def run_full_analysis(
        self,
        strategy_result: BacktestResult,
        ticks: Sequence[Tick],
        setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        engine_factory: Callable[[], BacktestEngine] | None = None,
        monte_carlo_config: MonteCarloConfig | None = None,
        random_baseline_config: RandomBaselineConfig | None = None,
        walk_forward_config: WalkForwardConfig | None = None,
    ) -> BacktestAnalysisReport:
        mc_res = self.monte_carlo_simulator.simulate(
            trades=strategy_result.trades,
            initial_balance=strategy_result.initial_balance,
            config=monte_carlo_config,
        )

        rb_res = self.random_baseline_analyzer.analyze(
            ticks=ticks,
            strategy_result=strategy_result,
            engine_factory=engine_factory,
            config=random_baseline_config,
        )

        wf_res = self.walk_forward_analyzer.analyze(
            ticks=ticks,
            setups=setups,
            orders=orders,
            engine_factory=engine_factory,
            config=walk_forward_config,
        )

        return BacktestAnalysisReport(
            strategy_result=strategy_result,
            monte_carlo=mc_res,
            random_baseline=rb_res,
            walk_forward=wf_res,
        )
