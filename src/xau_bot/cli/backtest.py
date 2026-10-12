import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from xau_bot.advisor.stats import SetupStatsAggregator, SetupStatsStore
from xau_bot.backtest.analysis import (
    BacktestAnalysisReport,
    MonteCarloConfig,
    MonteCarloSimulator,
    RandomBaselineAnalyzer,
    RandomBaselineConfig,
    WalkForwardAnalyzer,
    WalkForwardConfig,
)
from xau_bot.backtest.engine import BacktestEngine
from xau_bot.backtest.models import BacktestConfig
from xau_bot.backtest.slippage import FixedSlippage, ZeroSlippage
from xau_bot.domain.loader import (
    load_candles_from_file,
    load_setups_from_json,
    load_ticks_from_file,
)
from xau_bot.domain.models import Tick as DomainTick
from xau_bot.domain.models import TradeSetup
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import SpreadFilterConfig
from xau_bot.filters.news import NewsFilter, load_calendar_from_file
from xau_bot.filters.sessions import SessionFilter
from xau_bot.filters.spread import SpreadFilter
from xau_bot.risk.guards import RiskGuard
from xau_bot.risk.models import DailyRiskLimits, PositionRiskConfig, SymbolRiskSpec
from xau_bot.smc.setup import detect_setups


def evaluate_gate_1(
    report: BacktestAnalysisReport,
    min_trades: int = 200,
) -> tuple[bool, str, dict[str, bool]]:
    eval_res = report.evaluate_gate_1(min_trades=min_trades)
    return eval_res.passed, f"{eval_res.label} {eval_res.verdict}", eval_res.checks


def build_backtest_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="xau-bot backtest",
            description=(
                "Executa backtest da estratégia SMC para XAU/USD com análise "
                "estatística e validação do Portão 1."
            ),
        )

    parser.add_argument(
        "--ticks-path",
        required=True,
        type=str,
        help="Caminho para arquivo de ticks históricos (CSV ou JSON).",
    )
    parser.add_argument(
        "--candles-path",
        type=str,
        default=None,
        help="Caminho para arquivo de candles históricos (CSV ou JSON).",
    )
    parser.add_argument(
        "--detect-setups",
        action="store_true",
        help="Detecta automaticamente setups SMC a partir dos candles fornecidos.",
    )
    parser.add_argument(
        "--setups-path",
        type=str,
        default=None,
        help="Caminho para arquivo JSON de setups pré-calculados.",
    )
    parser.add_argument(
        "--news-path",
        type=str,
        default=None,
        help="Caminho para arquivo JSON de calendário econômico.",
    )
    parser.add_argument(
        "--filter-sessions",
        action="store_true",
        help="Habilita filtro de sessões e kill zones de Londres/NY.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Saldo inicial da conta simulada em USD (padrão: 10000.0).",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=100.0,
        help="Alavancagem da conta (padrão: 100.0).",
    )
    parser.add_argument(
        "--fixed-spread-points",
        type=float,
        default=0.0,
        help=(
            "Pontos de spread fixo sintético "
            "(se > 0.0, substitui o spread dos ticks por este valor)."
        ),
    )
    parser.add_argument(
        "--max-spread-points",
        type=float,
        default=0.0,
        help=(
            "Filtro: rejeita entradas quando o spread do tick "
            "exceder este valor em pontos (se > 0.0)."
        ),
    )
    parser.add_argument(
        "--spread-points",
        type=float,
        default=None,
        help="Alias de conveniência para --fixed-spread-points.",
    )
    parser.add_argument(
        "--slippage-points",
        type=float,
        default=0.0,
        help="Pontos de slippage fixo na execução (padrão: 0.0).",
    )
    parser.add_argument(
        "--commission-per-lot",
        type=float,
        default=7.0,
        help="Comissão por lote round-trip em USD (padrão: 7.0).",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=1.0,
        help="Percentual de risco por operação (padrão: 1.0%%).",
    )
    parser.add_argument(
        "--max-daily-loss-pct",
        type=float,
        default=3.0,
        help="Limite diário de perda em percentual (padrão: 3.0%%).",
    )
    parser.add_argument(
        "--max-daily-dd-pct",
        type=float,
        default=4.0,
        help="Drawdown diário máximo permitido (padrão: 4.0%%).",
    )
    parser.add_argument(
        "--max-total-dd-pct",
        type=float,
        default=10.0,
        help="Limiar de ruína do Monte Carlo em percentual (padrão: 10.0%%).",
    )
    parser.add_argument(
        "--max-consecutive-losses",
        type=int,
        default=3,
        help="Máximo de perdas consecutivas antes de pausar o dia (padrão: 3).",
    )
    parser.add_argument(
        "--monte-carlo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Habilita simulação de Monte Carlo por bootstrap (padrão: ativado).",
    )
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=1000,
        help="Número de simulações de Monte Carlo (padrão: 1000).",
    )
    parser.add_argument(
        "--random-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Habilita teste contra baseline aleatório de mesma geometria (padrão: ativado).",
    )
    parser.add_argument(
        "--baseline-iterations",
        type=int,
        default=100,
        help="Número de baselines aleatórios gerados (padrão: 100).",
    )
    parser.add_argument(
        "--walk-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Habilita particionamento e análise Walk-Forward (padrão: desativado).",
    )
    parser.add_argument(
        "--wf-windows",
        type=int,
        default=5,
        help="Número de janelas cronológicas para Walk-Forward (padrão: 5).",
    )
    parser.add_argument(
        "--wf-oos-ratio",
        type=float,
        default=0.3,
        help="Proporção fora da amostra (OOS) em cada janela (padrão: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente determinística para geradores pseudo-aleatórios (padrão: 42).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Caminho do arquivo para salvar o relatório Markdown de saída.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "summary"],
        default="markdown",
        help="Formato da saída do resultado (markdown, json ou summary).",
    )
    parser.add_argument(
        "--export-stats",
        type=str,
        default=None,
        help="Caminho para exportar as estatísticas dos setups em formato JSON.",
    )
    parser.add_argument(
        "--gate-check",
        action="store_true",
        help=(
            "Avalia rigorosamente o Portão de Decisão Humana 1 e encerra com código 1 se reprovado."
        ),
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=200,
        help="Quantidade mínima de trades exigida no Portão de Decisão Humana 1 (padrão: 200).",
    )
    parser.add_argument(
        "--server-tz",
        type=str,
        default="UTC",
        help="Fuso horário do servidor/dados (padrão: UTC).",
    )

    return parser


def run_backtest_cli(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    try:
        tz = ZoneInfo(args.server_tz)
    except (ZoneInfoNotFoundError, ValueError) as err:
        stderr.write(f"Erro: fuso horário inválido '{args.server_tz}': {err}\n")
        return 2

    if args.initial_balance <= 0.0:
        stderr.write("Erro de validação: Saldo inicial deve ser maior que zero.\n")
        return 2
    if args.leverage <= 0.0:
        stderr.write("Erro de validação: Alavancagem deve ser maior que zero.\n")
        return 2
    if args.risk_pct <= 0.0 or args.risk_pct > 100.0:
        stderr.write("Erro de validação: Risco percentual deve estar no intervalo (0, 100].\n")
        return 2
    if args.max_daily_loss_pct <= 0.0 or args.max_daily_loss_pct > 100.0:
        stderr.write(
            "Erro de validação: Limite diário de perda deve estar no intervalo (0, 100].\n"
        )
        return 2
    if args.max_daily_dd_pct <= 0.0 or args.max_daily_dd_pct > 100.0:
        stderr.write(
            "Erro de validação: Drawdown diário máximo deve estar no intervalo (0, 100].\n"
        )
        return 2
    if args.max_total_dd_pct <= 0.0 or args.max_total_dd_pct > 100.0:
        stderr.write(
            "Erro de validação: Limiar de ruína do Monte Carlo deve estar no intervalo (0, 100].\n"
        )
        return 2
    if args.max_consecutive_losses < 1:
        stderr.write("Erro de validação: Máximo de perdas consecutivas deve ser >= 1.\n")
        return 2
    if args.mc_iterations < 1:
        stderr.write("Erro de validação: Iterações de Monte Carlo devem ser >= 1.\n")
        return 2
    if args.baseline_iterations < 1:
        stderr.write("Erro de validação: Iterações de baseline aleatório devem ser >= 1.\n")
        return 2
    if args.wf_windows < 1:
        stderr.write("Erro de validação: Janelas de Walk-Forward devem ser >= 1.\n")
        return 2
    if args.wf_oos_ratio < 0.1 or args.wf_oos_ratio > 0.9:
        stderr.write(
            "Erro de validação: Proporção OOS do Walk-Forward "
            "deve estar entre 0.1 e 0.9 inclusive.\n"
        )
        return 2
    if args.min_trades < 1:
        stderr.write("Erro de validação: Quantidade mínima de trades deve ser >= 1.\n")
        return 2
    if args.fixed_spread_points < 0.0:
        stderr.write("Erro de validação: Spread fixo deve ser >= 0.\n")
        return 2
    if args.max_spread_points < 0.0:
        stderr.write("Erro de validação: Filtro de spread máximo deve ser >= 0.\n")
        return 2
    if args.spread_points is not None and args.spread_points < 0.0:
        stderr.write("Erro de validação: Spread fixo (--spread-points) deve ser >= 0.\n")
        return 2
    if args.slippage_points < 0.0:
        stderr.write("Erro de validação: Slippage deve ser >= 0.\n")
        return 2
    if args.commission_per_lot < 0.0:
        stderr.write("Erro de validação: Comissão por lote deve ser >= 0.\n")
        return 2

    ticks_path = Path(args.ticks_path)
    if not ticks_path.is_file():
        stderr.write(f"Erro: arquivo de ticks não encontrado: {ticks_path}\n")
        return 2

    if args.detect_setups and not args.candles_path:
        stderr.write("Erro: --detect-setups exige a especificação de --candles-path.\n")
        return 2

    candles = None
    if args.candles_path:
        candles_path = Path(args.candles_path)
        if not candles_path.is_file():
            stderr.write(f"Erro: arquivo de candles não encontrado: {candles_path}\n")
            return 2
        try:
            candles = load_candles_from_file(candles_path, server_tz=tz)
        except Exception as err:
            stderr.write(f"Erro ao carregar candles: {err}\n")
            return 2

    try:
        ticks = load_ticks_from_file(ticks_path, server_tz=tz)
    except Exception as err:
        stderr.write(f"Erro ao carregar ticks: {err}\n")
        return 2

    if not ticks:
        stderr.write("Erro: arquivo de ticks está vazio.\n")
        return 2

    fixed_spread = args.fixed_spread_points
    if args.spread_points is not None:
        if args.fixed_spread_points > 0.0 and args.spread_points != args.fixed_spread_points:
            stderr.write(
                "Erro de validação: --spread-points e --fixed-spread-points "
                "fornecidos com valores divergentes.\n"
            )
            return 2
        fixed_spread = args.spread_points

    if fixed_spread > 0.0:
        ticks = [
            DomainTick(
                time=t.time,
                bid=t.bid,
                ask=round(t.bid + fixed_spread * 0.01, 5),
                volume=t.volume,
                flags=t.flags,
            )
            for t in ticks
        ]

    setups: list[TradeSetup] | None = None
    if args.detect_setups and candles is not None:
        try:
            setups = detect_setups(candles.closed_candles)
        except Exception as err:
            stderr.write(f"Erro na detecção de setups: {err}\n")
            return 2
    elif args.setups_path:
        setups_path = Path(args.setups_path)
        if not setups_path.is_file():
            stderr.write(f"Erro: arquivo de setups não encontrado: {setups_path}\n")
            return 2
        try:
            setups = load_setups_from_json(setups_path)
        except Exception as err:
            stderr.write(f"Erro ao carregar setups: {err}\n")
            return 2

    news_filter = None
    if args.news_path:
        news_path = Path(args.news_path)
        if not news_path.is_file():
            stderr.write(f"Erro: arquivo de calendário econômico não encontrado: {news_path}\n")
            return 2
        try:
            events = load_calendar_from_file(news_path)
            news_filter = NewsFilter(events)
        except Exception as err:
            stderr.write(f"Erro ao carregar calendário econômico: {err}\n")
            return 2

    session_filter = SessionFilter(server_tz=tz) if args.filter_sessions else None
    spread_filter = (
        SpreadFilter(config=SpreadFilterConfig(max_spread_points=args.max_spread_points))
        if args.max_spread_points > 0.0
        else None
    )

    trade_filter = None
    if session_filter is not None or news_filter is not None or spread_filter is not None:
        trade_filter = CompositeTradeFilter(
            session_filter=session_filter,
            news_filter=news_filter,
            spread_filter=spread_filter,
        )

    spec = SymbolRiskSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        tick_size=0.01,
        tick_value=1.0,
        lot_min=0.01,
        lot_max=100.0,
        lot_step=0.01,
        point=0.01,
        commission_per_lot=args.commission_per_lot,
        spread_points=fixed_spread if fixed_spread > 0.0 else 25.0,
    )

    engine_config = BacktestConfig(
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        symbol_spec=spec,
        slippage_points=args.slippage_points,
    )

    slippage_model = (
        FixedSlippage(points=args.slippage_points, point_size=spec.point)
        if args.slippage_points > 0.0
        else ZeroSlippage()
    )

    pos_risk_config = PositionRiskConfig(
        risk_per_trade_percent=args.risk_pct / 100.0,
    )

    daily_risk_limits = DailyRiskLimits(
        max_daily_loss_percent=args.max_daily_loss_pct / 100.0,
        max_daily_drawdown_percent=args.max_daily_dd_pct / 100.0,
        max_consecutive_losses=args.max_consecutive_losses,
    )

    def engine_factory() -> BacktestEngine:
        return BacktestEngine(
            config=engine_config,
            slippage_model=slippage_model,
            position_risk_config=pos_risk_config,
            trade_filter=trade_filter,
            risk_guard=RiskGuard(limits=daily_risk_limits),
        )

    primary_engine = engine_factory()
    closed_c = candles.closed_candles if candles is not None else None
    strategy_result = primary_engine.run(
        ticks=ticks,
        setups=setups,
        candles=closed_c,
    )

    mc_res = None
    if args.monte_carlo and strategy_result.trades:
        mc_sim = MonteCarloSimulator()
        mc_cfg = MonteCarloConfig(
            simulations=args.mc_iterations,
            ruin_drawdown_pct=args.max_total_dd_pct,
            seed=args.seed,
        )
        mc_res = mc_sim.simulate(
            trades=strategy_result.trades,
            initial_balance=strategy_result.initial_balance,
            config=mc_cfg,
        )

    rb_res = None
    if args.random_baseline:
        rb_analyzer = RandomBaselineAnalyzer()
        rb_cfg = RandomBaselineConfig(
            simulations=args.baseline_iterations,
            seed=args.seed,
        )
        rb_res = rb_analyzer.analyze(
            ticks=ticks,
            strategy_result=strategy_result,
            engine_factory=engine_factory,
            config=rb_cfg,
        )

    wf_res = None
    if args.walk_forward:
        wf_analyzer = WalkForwardAnalyzer()
        wf_cfg = WalkForwardConfig(
            windows_count=args.wf_windows,
            in_sample_ratio=1.0 - args.wf_oos_ratio,
        )
        wf_res = wf_analyzer.analyze(
            ticks=ticks,
            setups=setups,
            engine_factory=engine_factory,
            config=wf_cfg,
        )

    report = BacktestAnalysisReport(
        strategy_result=strategy_result,
        monte_carlo=mc_res,
        random_baseline=rb_res,
        walk_forward=wf_res,
    )

    passed, verdict, checks = evaluate_gate_1(
        report,
        min_trades=args.min_trades,
    )

    if args.format == "markdown":
        output_content = report.to_markdown(min_trades=args.min_trades)
    elif args.format == "json":
        json_obj: dict[str, Any] = {
            "strategy": {
                "initial_balance": strategy_result.initial_balance,
                "final_balance": strategy_result.final_balance,
                "total_net_pnl": strategy_result.total_net_pnl,
                "total_trades": strategy_result.total_trades,
                "win_rate": strategy_result.win_rate,
                "profit_factor": strategy_result.profit_factor,
                "expectancy_r": strategy_result.expectancy_r,
                "max_drawdown_usd": strategy_result.max_drawdown_usd,
                "max_drawdown_pct": strategy_result.max_drawdown_pct,
                "sharpe_ratio": strategy_result.sharpe_ratio,
            },
            "monte_carlo": None
            if mc_res is None
            else {
                "simulations": mc_res.simulations,
                "ruin_probability": mc_res.ruin_probability,
                "median_max_drawdown_pct": mc_res.median_max_drawdown_pct,
                "p95_max_drawdown_pct": mc_res.p95_max_drawdown_pct,
                "p99_max_drawdown_pct": mc_res.p99_max_drawdown_pct,
                "median_final_balance": mc_res.median_final_balance,
            },
            "random_baseline": None
            if rb_res is None
            else {
                "simulations": rb_res.simulations,
                "strategy_expectancy_r": rb_res.strategy_expectancy_r,
                "strategy_profit_factor": rb_res.strategy_profit_factor,
                "baseline_expectancy_r_mean": rb_res.baseline_expectancy_r_mean,
                "p_value_expectancy_r": rb_res.p_value_expectancy_r,
                "p_value_profit_factor": rb_res.p_value_profit_factor,
                "statistically_significant": rb_res.statistically_significant,
            },
            "walk_forward": None
            if wf_res is None
            else {
                "windows_count": len(wf_res.windows),
                "overall_wfe": wf_res.overall_wfe,
                "combined_oos_trades": wf_res.combined_oos_result.total_trades
                if wf_res.combined_oos_result
                else 0,
                "combined_oos_pnl": wf_res.combined_oos_result.total_net_pnl
                if wf_res.combined_oos_result
                else 0.0,
            },
            "gate_decision": {
                "passed": passed,
                "verdict": verdict,
                "checks": checks,
            },
        }
        output_content = json.dumps(json_obj, indent=2)
    else:
        summary_lines = [
            "============================================================",
            "                   XAU-BOT BACKTEST SUMMARY                 ",
            "============================================================",
            f"Saldo Inicial:     US$ {strategy_result.initial_balance:.2f}",
            f"Saldo Final:       US$ {strategy_result.final_balance:.2f}",
            f"Lucro Líquido:     US$ {strategy_result.total_net_pnl:.2f}",
            f"Total de Trades:   {strategy_result.total_trades}",
            f"Taxa de Acerto:    {strategy_result.win_rate * 100:.1f}%",
            f"Fator de Lucro:    {strategy_result.profit_factor:.2f}",
            f"Expectativa em R:  {strategy_result.expectancy_r:.2f}R",
            (
                f"Drawdown Máximo:   US$ {strategy_result.max_drawdown_usd:.2f} "
                f"({strategy_result.max_drawdown_pct:.2f}%)"
            ),
            "------------------------------------------------------------",
            f"Portão 1 Veredito: {'APROVADO' if passed else 'REPROVADO/PENDENTE'}",
            f"Detalhes:          {verdict}",
            "============================================================",
        ]
        output_content = "\n".join(summary_lines)

    if args.output:
        out_path = Path(args.output)
        if out_path.is_dir():
            stderr.write(f"Erro ao salvar arquivo de saída: {out_path} é um diretório.\n")
            return 2
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_content, encoding="utf-8")
        except OSError as exc:
            stderr.write(f"Erro ao salvar arquivo de saída: {exc}\n")
            return 2
        stdout.write(f"Relatório de backtest gravado com sucesso em: {out_path}\n")
    else:
        stdout.write(output_content + "\n")

    if args.export_stats:
        aggregator = SetupStatsAggregator()
        setups_dict = {s.id: s for s in setups} if setups else {}
        stats_by_trigger = aggregator.aggregate_by_trigger_type(strategy_result.trades, setups_dict)
        stats_by_trigger["GLOBAL"] = aggregator.aggregate_trades(
            strategy_result.trades, setup_type="GLOBAL"
        )
        store = SetupStatsStore(stats_by_trigger)
        try:
            store.save_to_file(args.export_stats)
        except OSError as exc:
            stderr.write(f"Erro ao exportar estatísticas de setups: {exc}\n")
            return 2
        stdout.write(f"Estatísticas de setups exportadas com sucesso em: {args.export_stats}\n")

    if args.gate_check and not passed:
        return 1

    return 0


def backtest_main(
    argv: Sequence[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    parser = build_backtest_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    return run_backtest_cli(args, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    sys.exit(backtest_main())
