import io
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from xau_bot.cli.backtest import (
    backtest_main,
    build_backtest_parser,
    evaluate_gate_1,
    run_backtest_cli,
)
from xau_bot.cli.main import build_parser, main
from xau_bot.domain.loader import load_setups_from_json
from xau_bot.domain.models import Direction, TradeSetup


def _create_synthetic_data(
    tmp_path: Path,
    num_ticks: int = 50,
    start_price: float = 2650.0,
) -> tuple[Path, Path, Path]:
    ticks_file = tmp_path / "ticks.csv"
    start_dt = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)

    tick_lines = ["time,bid,ask,last,volume,flags"]
    curr_time = start_dt
    for i in range(num_ticks):
        p = start_price + (1.5 if i % 2 == 0 else -1.5)
        tick_lines.append(f"{curr_time.isoformat()},{p:.2f},{p + 0.20:.2f},{p:.2f},1.0,6")
        curr_time += timedelta(seconds=1)
    ticks_file.write_text("\n".join(tick_lines) + "\n", encoding="utf-8")

    candles_file = tmp_path / "candles.csv"
    candle_lines = ["time,open,high,low,close,volume"]
    c_time = start_dt
    for _i in range(10):
        candle_lines.append(f"{c_time.isoformat()},2650.0,2655.0,2648.0,2652.0,100.0")
        c_time += timedelta(minutes=5)
    candles_file.write_text("\n".join(candle_lines) + "\n", encoding="utf-8")

    setups_file = tmp_path / "setups.json"
    setups_data = [
        {
            "id": "setup_test_1",
            "direction": "BULLISH",
            "entry_price": 2650.0,
            "stop_loss": 2645.0,
            "take_profit": 2660.0,
            "time": start_dt.isoformat(),
            "confirmed_time": (start_dt + timedelta(seconds=1)).isoformat(),
            "risk_reward_ratio": 2.0,
            "timeframe": "M5",
        }
    ]
    setups_file.write_text(json.dumps(setups_data), encoding="utf-8")

    return ticks_file, candles_file, setups_file


def test_build_parser_help() -> None:
    parser = build_parser()
    assert parser.prog == "xau-bot"

    bt_parser = build_backtest_parser()
    assert "ticks-path" in bt_parser.format_help()


def test_cli_main_missing_subcommand() -> None:
    code = main([])
    assert code == 2


def test_cli_main_invalid_subcommand() -> None:
    code = main(["nonexistent-subcommand"])
    assert code == 2


def test_cli_missing_ticks_path_error() -> None:
    stderr = io.StringIO()
    stdout = io.StringIO()
    code = main(["backtest"], stdout=stdout, stderr=stderr)
    assert code == 2


def test_cli_ticks_file_not_found(tmp_path: Path) -> None:
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(["--ticks-path", str(tmp_path / "missing.csv")])
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "arquivo de ticks não encontrado" in stderr.getvalue()


def test_cli_invalid_server_tz(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--server-tz",
            "Invalid/Timezone_Name_12345",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "fuso horário inválido" in stderr.getvalue()


def test_cli_detect_setups_without_candles(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--detect-setups",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "--detect-setups exige a especificação de --candles-path" in stderr.getvalue()


def test_cli_candles_file_not_found(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--candles-path",
            str(tmp_path / "missing_candles.csv"),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "arquivo de candles não encontrado" in stderr.getvalue()


def test_cli_setups_file_not_found(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(tmp_path / "missing_setups.json"),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "arquivo de setups não encontrado" in stderr.getvalue()


def test_cli_news_file_not_found(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--news-path",
            str(tmp_path / "missing_news.json"),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "arquivo de calendário econômico não encontrado" in stderr.getvalue()


def test_cli_empty_ticks_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("time,bid,ask\n", encoding="utf-8")
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(["--ticks-path", str(empty_file)])
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "arquivo de ticks está vazio" in stderr.getvalue()


def test_cli_malformed_ticks_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("invalid,file,content\n1,2,3\n", encoding="utf-8")
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(["--ticks-path", str(bad_file)])
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "Erro ao carregar ticks" in stderr.getvalue()


def test_cli_malformed_candles_file(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    bad_candles = tmp_path / "bad_candles.csv"
    bad_candles.write_text("invalid,csv\n1,2\n", encoding="utf-8")
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--candles-path",
            str(bad_candles),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "Erro ao carregar candles" in stderr.getvalue()


def test_cli_malformed_setups_file(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    bad_setups = tmp_path / "bad_setups.json"
    bad_setups.write_text("not-a-valid-json", encoding="utf-8")
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(bad_setups),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "Erro ao carregar setups" in stderr.getvalue()


def test_cli_malformed_news_file(tmp_path: Path) -> None:
    ticks_file, _, _ = _create_synthetic_data(tmp_path)
    bad_news = tmp_path / "bad_news.json"
    bad_news.write_text("not-a-valid-json", encoding="utf-8")
    stderr = io.StringIO()
    stdout = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--news-path",
            str(bad_news),
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "Erro ao carregar calendário econômico" in stderr.getvalue()


def test_load_setups_from_json(tmp_path: Path) -> None:
    setups_file = tmp_path / "custom_setups.json"
    now_iso = datetime(2026, 9, 24, 12, 0, tzinfo=UTC).isoformat()
    content = [
        {
            "id": "s1",
            "direction": "BEARISH",
            "entry_price": 2660.0,
            "stop_loss": 2665.0,
            "take_profit": 2650.0,
            "time": now_iso,
            "confirmed_time": now_iso,
            "risk_reward_ratio": 2.0,
            "valid_until": (datetime(2026, 9, 24, 14, 0, tzinfo=UTC)).isoformat(),
            "timeframe": "M5",
        }
    ]
    setups_file.write_text(json.dumps(content), encoding="utf-8")
    setups = load_setups_from_json(setups_file)
    assert len(setups) == 1
    assert setups[0].direction == Direction.BEARISH
    assert setups[0].entry_price == 2660.0
    assert setups[0].valid_until is not None


def test_cli_run_success_markdown(tmp_path: Path) -> None:
    ticks_file, candles_file, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = main(
        [
            "backtest",
            "--ticks-path",
            str(ticks_file),
            "--candles-path",
            str(candles_file),
            "--setups-path",
            str(setups_file),
            "--no-walk-forward",
            "--baseline-iterations",
            "2",
            "--mc-iterations",
            "10",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Relatório de Análise Estatística" in stdout.getvalue()


def test_cli_run_success_json_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)

    exit_code = backtest_main(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--format",
            "json",
            "--no-walk-forward",
            "--baseline-iterations",
            "2",
            "--mc-iterations",
            "5",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "strategy" in payload
    assert "gate_decision" in payload
    assert payload["strategy"]["initial_balance"] == 10000.0


def test_cli_run_success_summary_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)

    exit_code = backtest_main(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--format",
            "summary",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "XAU-BOT BACKTEST SUMMARY" in captured.out
    assert "Portão 1 Veredito" in captured.out


def test_cli_run_output_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    out_report = tmp_path / "reports" / "report.md"

    exit_code = backtest_main(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "-o",
            str(out_report),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    assert exit_code == 0
    assert out_report.is_file()
    content = out_report.read_text(encoding="utf-8")
    assert "# Relatório de Análise Estatística de Backtest" in content
    captured = capsys.readouterr()
    assert f"Relatório de backtest gravado com sucesso em: {out_report}" in captured.out


def test_cli_run_with_detect_setups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ticks_file, candles_file, _ = _create_synthetic_data(tmp_path)
    start_dt = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)
    detected_setup = TradeSetup(
        id="detected_setup_1",
        direction=Direction.BULLISH,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        time=start_dt,
        confirmed_time=start_dt + timedelta(seconds=1),
        risk_reward_ratio=2.0,
    )
    captured_detect_kwargs: dict[str, Any] = {}

    def mock_detect(candles: Any, *args: Any, **kwargs: Any) -> list[TradeSetup]:
        captured_detect_kwargs.update(kwargs)
        return [detected_setup]

    monkeypatch.setattr("xau_bot.cli.backtest.detect_setups", mock_detect)
    monkeypatch.setenv("XAU_SMC_REQUIRE_PREMIUM_DISCOUNT", "false")
    monkeypatch.setenv("XAU_SMC_SWEEP_GUARD_MAX_REJECTION_RATIO", "0.35")

    stdout = io.StringIO()
    stderr = io.StringIO()

    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--candles-path",
            str(candles_file),
            "--detect-setups",
            "--format",
            "summary",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    out = stdout.getvalue()
    assert "Total de Trades:   1" in out

    # Valida regressão: flags de Settings e htf_candles foram propagados a detect_setups
    assert captured_detect_kwargs.get("require_premium_discount") is False
    assert captured_detect_kwargs.get("require_sweep_guard") is True
    assert captured_detect_kwargs.get("sweep_guard_max_rejection_ratio") == 0.35
    assert captured_detect_kwargs.get("htf_alignment_required") is True
    assert captured_detect_kwargs.get("htf_candles") is not None


def test_cli_run_with_filters(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    news_file = tmp_path / "news.json"
    news_content = [
        {
            "title": "US CPI",
            "currency": "USD",
            "impact": "HIGH",
            "time": "2026-09-24T12:30:00Z",
        }
    ]
    news_file.write_text(json.dumps(news_content), encoding="utf-8")

    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--news-path",
            str(news_file),
            "--filter-sessions",
            "--spread-points",
            "25.0",
            "--slippage-points",
            "10.0",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0


def test_cli_run_with_walk_forward(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path, num_ticks=100)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--walk-forward",
            "--wf-windows",
            "2",
            "--wf-oos-ratio",
            "0.3",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    assert "Análise Walk-Forward" in stdout.getvalue()


def test_cli_walk_forward_gate_eval_uses_oos_trades(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--walk-forward",
            "--wf-windows",
            "2",
            "--wf-oos-ratio",
            "0.3",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    out = stdout.getvalue()
    pattern = (
        r"Amostra Mínima \(>= \d+ trades\)\*\*:\s*REPROVADO\s*"
        r"\((\d+)\s+trades fora da amostra \(OOS\)\)"
    )
    match = re.search(pattern, out)
    assert match is not None
    assert int(match.group(1)) == 0


def test_cli_conflicting_spread_options_error(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--fixed-spread-points",
            "10.0",
            "--spread-points",
            "20.0",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "valores divergentes" in stderr.getvalue()


def test_cli_invalid_wf_oos_ratio_error(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    for bad_ratio in ["1.5", "0.05"]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        parser = build_backtest_parser()
        args = parser.parse_args(
            [
                "--ticks-path",
                str(ticks_file),
                "--setups-path",
                str(setups_file),
                "--walk-forward",
                "--wf-oos-ratio",
                bad_ratio,
            ]
        )
        code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
        assert code == 2
        assert "Proporção OOS" in stderr.getvalue()


def test_gate_check_fails_on_insufficient_sample(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--gate-check",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 1


def test_gate_check_passes_when_criteria_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--gate-check",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    from xau_bot.backtest.analysis import BacktestAnalysisReport, Gate1Evaluation

    monkeypatch.setattr(
        BacktestAnalysisReport,
        "evaluate_gate_1",
        lambda self, min_trades=200: Gate1Evaluation(
            passed=True,
            label="[APROVADO]",
            verdict="Estratégia aprovada no Portão de Decisão Humana 1.",
            checks={
                "min_trades": True,
                "profitable": True,
                "beats_baseline": True,
                "walk_forward_efficiency": True,
                "tail_risk": True,
            },
            eval_trades=250,
            eval_scope="amostra completa",
            eval_pnl=1500.0,
            eval_pf=1.8,
            eval_expectancy_r=0.45,
        ),
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    assert "[APROVADO]" in stdout.getvalue()


def test_gate_check_fails_when_criteria_not_met(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--gate-check",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    from xau_bot.backtest.analysis import BacktestAnalysisReport, Gate1Evaluation

    monkeypatch.setattr(
        BacktestAnalysisReport,
        "evaluate_gate_1",
        lambda self, min_trades=200: Gate1Evaluation(
            passed=False,
            label="[REPROVADO]",
            verdict="Estratégia não superou baseline aleatório.",
            checks={
                "min_trades": True,
                "profitable": True,
                "beats_baseline": False,
                "walk_forward_efficiency": True,
                "tail_risk": True,
            },
            eval_trades=250,
            eval_scope="amostra completa",
            eval_pnl=1500.0,
            eval_pf=1.8,
            eval_expectancy_r=0.45,
        ),
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 1
    assert "[REPROVADO]" in stdout.getvalue()


def test_evaluate_gate_1_unit_branches() -> None:
    from xau_bot.backtest.analysis import (
        BacktestAnalysisReport,
        MonteCarloResult,
        RandomBaselineResult,
        WalkForwardResult,
    )
    from xau_bot.backtest.models import BacktestResult

    base_res = BacktestResult(
        trades=(),
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=11000.0,
        total_net_pnl=1000.0,
        total_trades=250,
        winning_trades=150,
        losing_trades=100,
        breakeven_trades=0,
        win_rate=0.60,
        profit_factor=1.8,
        max_drawdown_usd=500.0,
        max_drawdown_pct=5.0,
        sharpe_ratio=1.5,
        expectancy_r=0.4,
        average_trade_duration_seconds=300.0,
    )

    report_no_rb = BacktestAnalysisReport(strategy_result=base_res)
    passed, verdict, checks = evaluate_gate_1(report_no_rb, min_trades=200)
    assert not passed
    assert not checks["beats_baseline"]
    assert "baseline aleatório ausente" in verdict
    assert "[REPROVADO]" in verdict

    rb_fail = RandomBaselineResult(
        simulations=10,
        baseline_expectancy_r_mean=0.3,
        baseline_expectancy_r_std=0.1,
        baseline_profit_factor_mean=1.5,
        baseline_win_rate_mean=0.5,
        strategy_expectancy_r=0.4,
        strategy_profit_factor=1.8,
        p_value_expectancy_r=0.15,
        p_value_profit_factor=0.20,
        statistically_significant=False,
    )
    report_rb_fail = BacktestAnalysisReport(strategy_result=base_res, random_baseline=rb_fail)
    passed, verdict, _ = evaluate_gate_1(report_rb_fail, min_trades=200)
    assert not passed
    assert "não superou estatisticamente" in verdict
    assert "[REPROVADO]" in verdict

    rb_ok = RandomBaselineResult(
        simulations=10,
        baseline_expectancy_r_mean=0.1,
        baseline_expectancy_r_std=0.1,
        baseline_profit_factor_mean=1.1,
        baseline_win_rate_mean=0.5,
        strategy_expectancy_r=0.4,
        strategy_profit_factor=1.8,
        p_value_expectancy_r=0.01,
        p_value_profit_factor=0.01,
        statistically_significant=True,
    )
    few_trades_res = BacktestResult(
        trades=(),
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=10500.0,
        total_net_pnl=500.0,
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        breakeven_trades=0,
        win_rate=0.60,
        profit_factor=1.5,
        max_drawdown_usd=200.0,
        max_drawdown_pct=2.0,
        sharpe_ratio=1.5,
        expectancy_r=0.4,
        average_trade_duration_seconds=300.0,
    )
    report_few = BacktestAnalysisReport(strategy_result=few_trades_res, random_baseline=rb_ok)
    passed, verdict, checks = evaluate_gate_1(report_few, min_trades=200)
    assert not passed
    assert not checks["min_trades"]
    assert "Amostra com 50 trades" in verdict
    assert "[AMOSTRA_INSUFICIENTE]" in verdict

    loss_res = BacktestResult(
        trades=(),
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=9500.0,
        total_net_pnl=-500.0,
        total_trades=250,
        winning_trades=100,
        losing_trades=150,
        breakeven_trades=0,
        win_rate=0.40,
        profit_factor=0.8,
        max_drawdown_usd=800.0,
        max_drawdown_pct=8.0,
        sharpe_ratio=-0.5,
        expectancy_r=-0.2,
        average_trade_duration_seconds=300.0,
    )
    report_loss = BacktestAnalysisReport(strategy_result=loss_res, random_baseline=rb_ok)
    passed, verdict, checks = evaluate_gate_1(report_loss, min_trades=200)
    assert not passed
    assert not checks["profitable"]
    assert "não é lucrativa" in verdict or "não lucrativa" in verdict
    assert "[REPROVADO]" in verdict

    wf_fail = WalkForwardResult(
        windows=(),
        overall_wfe=0.30,
        combined_oos_trades=(),
        combined_oos_result=base_res,
    )
    report_wf_fail = BacktestAnalysisReport(
        strategy_result=base_res,
        random_baseline=rb_ok,
        walk_forward=wf_fail,
    )
    passed, verdict, checks = evaluate_gate_1(report_wf_fail, min_trades=200)
    assert not passed
    assert not checks["walk_forward_efficiency"]
    assert "Eficiência Walk-Forward insatisfatória" in verdict
    assert "[REPROVADO]" in verdict

    mc_fail = MonteCarloResult(
        simulations=100,
        ruin_probability=0.05,
        median_max_drawdown_usd=500.0,
        p95_max_drawdown_usd=800.0,
        p99_max_drawdown_usd=1200.0,
        median_max_drawdown_pct=5.0,
        p95_max_drawdown_pct=8.0,
        p99_max_drawdown_pct=12.0,
        median_final_balance=11000.0,
        p5_final_balance=9000.0,
        p95_final_balance=13000.0,
    )
    wf_ok = WalkForwardResult(
        windows=(),
        overall_wfe=0.85,
        combined_oos_trades=(),
        combined_oos_result=base_res,
    )
    report_mc_fail = BacktestAnalysisReport(
        strategy_result=base_res,
        random_baseline=rb_ok,
        walk_forward=wf_ok,
        monte_carlo=mc_fail,
    )
    passed, verdict, checks = evaluate_gate_1(report_mc_fail, min_trades=200)
    assert not passed
    assert not checks["tail_risk"]
    assert "Risco de cauda excessivo" in verdict
    assert "[REPROVADO]" in verdict

    mc_ok = MonteCarloResult(
        simulations=100,
        ruin_probability=0.0,
        median_max_drawdown_usd=300.0,
        p95_max_drawdown_usd=500.0,
        p99_max_drawdown_usd=700.0,
        median_max_drawdown_pct=3.0,
        p95_max_drawdown_pct=5.0,
        p99_max_drawdown_pct=7.0,
        median_final_balance=11000.0,
        p5_final_balance=10200.0,
        p95_final_balance=12500.0,
    )
    report_all_ok = BacktestAnalysisReport(
        strategy_result=base_res,
        random_baseline=rb_ok,
        monte_carlo=mc_ok,
        walk_forward=wf_ok,
    )
    passed, verdict, checks = evaluate_gate_1(report_all_ok, min_trades=200)
    assert passed
    assert all(checks.values())
    assert "[APROVADO]" in verdict


def test_evaluate_gate_1_insufficient_sample_takes_precedence_over_missing_baseline() -> None:
    from xau_bot.backtest.analysis import (
        BacktestAnalysisReport,
        RandomBaselineResult,
    )
    from xau_bot.backtest.models import BacktestResult

    few_trades_res = BacktestResult(
        trades=(),
        equity_curve=(),
        initial_balance=10000.0,
        final_balance=10500.0,
        total_net_pnl=500.0,
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        breakeven_trades=0,
        win_rate=0.60,
        profit_factor=1.5,
        max_drawdown_usd=200.0,
        max_drawdown_pct=2.0,
        sharpe_ratio=1.5,
        expectancy_r=0.4,
        average_trade_duration_seconds=300.0,
    )

    report_no_rb = BacktestAnalysisReport(strategy_result=few_trades_res, random_baseline=None)
    passed_1, verdict_1, checks_1 = evaluate_gate_1(report_no_rb, min_trades=200)
    assert not passed_1
    assert not checks_1["min_trades"]
    assert not checks_1["beats_baseline"]
    assert "[AMOSTRA_INSUFICIENTE]" in verdict_1
    assert "Amostra com 50 trades" in verdict_1

    rb_fail = RandomBaselineResult(
        simulations=10,
        baseline_expectancy_r_mean=0.3,
        baseline_expectancy_r_std=0.1,
        baseline_profit_factor_mean=1.5,
        baseline_win_rate_mean=0.5,
        strategy_expectancy_r=0.4,
        strategy_profit_factor=1.8,
        p_value_expectancy_r=0.15,
        p_value_profit_factor=0.20,
        statistically_significant=False,
    )
    report_rb_fail = BacktestAnalysisReport(strategy_result=few_trades_res, random_baseline=rb_fail)
    passed_2, verdict_2, checks_2 = evaluate_gate_1(report_rb_fail, min_trades=200)
    assert not passed_2
    assert not checks_2["min_trades"]
    assert not checks_2["beats_baseline"]
    assert "[AMOSTRA_INSUFICIENTE]" in verdict_2
    assert "Amostra com 50 trades" in verdict_2


def test_cli_invocation_via_python_module(tmp_path: Path) -> None:
    import os
    import subprocess

    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    res_cli_help = subprocess.run(
        [sys.executable, "-m", "xau_bot.cli", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert res_cli_help.returncode == 0
    assert "xau-bot" in res_cli_help.stdout

    res_bt_help = subprocess.run(
        [sys.executable, "-m", "xau_bot.cli.backtest", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert res_bt_help.returncode == 0
    assert "--ticks-path" in res_bt_help.stdout

    res_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "xau_bot.cli",
            "backtest",
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert res_run.returncode == 0
    assert "Relatório de Análise Estatística" in res_run.stdout

    scripts_dir = Path(sys.prefix) / ("Scripts" if sys.platform == "win32" else "bin")
    xau_bot_exe = scripts_dir / ("xau-bot.exe" if sys.platform == "win32" else "xau-bot")
    if xau_bot_exe.is_file():
        res_exe = subprocess.run(
            [str(xau_bot_exe), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        assert res_exe.returncode == 0
        assert "xau-bot" in res_exe.stdout


def test_cli_run_with_ticks_json(tmp_path: Path) -> None:
    _, _, setups_file = _create_synthetic_data(tmp_path)
    ticks_json = tmp_path / "ticks.json"
    ticks_json.write_text(
        json.dumps(
            [
                {
                    "time": "2026-09-24T12:00:00Z",
                    "bid": 2650.0,
                    "ask": 2650.20,
                    "last": 2650.0,
                    "volume": 1.0,
                },
                {
                    "time": "2026-09-24T12:00:01Z",
                    "bid": 2651.5,
                    "ask": 2651.70,
                    "last": 2651.5,
                    "volume": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_json),
            "--setups-path",
            str(setups_file),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    assert "Relatório de Análise Estatística" in stdout.getvalue()


def test_cli_domain_validation_errors(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)

    invalid_args_cases = [
        (["--initial-balance", "-100.0"], "Saldo inicial deve ser maior que zero"),
        (["--leverage", "0.0"], "Alavancagem deve ser maior que zero"),
        (["--risk-pct", "0.0"], "Risco percentual"),
        (["--risk-pct", "150.0"], "Risco percentual"),
        (["--max-daily-loss-pct", "0.0"], "Limite diário de perda"),
        (["--max-daily-dd-pct", "0.0"], "Drawdown diário máximo"),
        (["--max-total-dd-pct", "0.0"], "Limiar de ruína"),
        (["--max-consecutive-losses", "0"], "Máximo de perdas consecutivas"),
        (["--mc-iterations", "0"], "Iterações de Monte Carlo"),
        (["--baseline-iterations", "0"], "Iterações de baseline aleatório"),
        (["--wf-windows", "0"], "Janelas de Walk-Forward"),
        (["--wf-oos-ratio", "1.5"], "Proporção OOS"),
        (["--min-trades", "0"], "Quantidade mínima de trades"),
        (["--fixed-spread-points", "-1.0"], "Spread fixo deve ser >= 0"),
        (["--max-spread-points", "-1.0"], "Filtro de spread máximo"),
        (["--spread-points", "-5.0"], "Spread fixo (--spread-points)"),
        (["--slippage-points", "-1.0"], "Slippage deve ser >= 0"),
        (["--commission-per-lot", "-2.0"], "Comissão por lote"),
    ]

    for flags, expected_err in invalid_args_cases:
        parser = build_backtest_parser()
        args = parser.parse_args(
            [
                "--ticks-path",
                str(ticks_file),
                "--setups-path",
                str(setups_file),
                *flags,
            ]
        )
        stderr = io.StringIO()
        stdout = io.StringIO()
        code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
        assert code == 2, f"Expected code 2 for {flags}, got {code}"
        assert expected_err in stderr.getvalue(), (
            f"Expected '{expected_err}' in '{stderr.getvalue()}'"
        )

    out_dir = tmp_path / "some_dir"
    out_dir.mkdir()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--output",
            str(out_dir),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    stderr = io.StringIO()
    stdout = io.StringIO()
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "é um diretório" in stderr.getvalue()


def test_cli_fixed_and_max_spread(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--fixed-spread-points",
            "30.0",
            "--max-spread-points",
            "50.0",
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ]
    )
    stderr = io.StringIO()
    stdout = io.StringIO()
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    assert "Relatório de Análise Estatística" in stdout.getvalue()


def test_ac_1301_cli_backtest_with_ticks_json_and_csv(tmp_path: Path) -> None:
    test_cli_run_success_markdown(tmp_path)
    test_cli_run_with_ticks_json(tmp_path)


def test_ac_1302_cli_backtest_detect_setups_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_cli_run_with_detect_setups(tmp_path, monkeypatch)


def test_ac_1303_statistical_analyzers_flags(tmp_path: Path) -> None:
    ticks_file, _, setups_file = _create_synthetic_data(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--walk-forward",
            "--wf-windows",
            "2",
            "--wf-oos-ratio",
            "0.3",
            "--baseline-iterations",
            "2",
            "--mc-iterations",
            "5",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 0
    out = stdout.getvalue()
    assert "Simulação de Monte Carlo" in out
    assert "Comparativo com Baseline Aleatório" in out
    assert "Análise Walk-Forward" in out


def test_ac_1304_markdown_output_file_export(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    test_cli_run_output_file(tmp_path, capsys)


def test_ac_1305_gate_check_verdicts_and_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_gate_check_fails_on_insufficient_sample(tmp_path)
    test_gate_check_passes_when_criteria_met(tmp_path, monkeypatch)
    test_gate_check_fails_when_criteria_not_met(tmp_path, monkeypatch)


def test_ac_1306_invalid_arguments_and_missing_files(tmp_path: Path) -> None:
    test_cli_ticks_file_not_found(tmp_path)
    test_cli_candles_file_not_found(tmp_path)
    test_cli_setups_file_not_found(tmp_path)
    test_cli_news_file_not_found(tmp_path)
    test_cli_domain_validation_errors(tmp_path)


def test_ac_1307_module_and_entrypoint_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_cli_invocation_via_python_module(tmp_path)
    test_cli_main_and_package_entrypoint(monkeypatch)


def test_cli_detect_setups_error_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xau_bot.cli import backtest as bt_mod

    ticks_file, candles_file, _ = _create_synthetic_data(tmp_path)
    stderr = io.StringIO()
    stdout = io.StringIO()

    def _mock_failing_detect_setups(*args: object, **kwargs: object) -> list[TradeSetup]:
        raise RuntimeError("Synthetic detection error")

    monkeypatch.setattr(bt_mod, "detect_setups", _mock_failing_detect_setups)

    parser = build_backtest_parser()
    args = parser.parse_args(
        [
            "--ticks-path",
            str(ticks_file),
            "--candles-path",
            str(candles_file),
            "--detect-setups",
        ]
    )
    code = run_backtest_cli(args, stdout=stdout, stderr=stderr)
    assert code == 2
    assert "Erro na detecção de setups: Synthetic detection error" in stderr.getvalue()


def test_cli_main_and_package_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["xau-bot", "backtest", "--help"])
    monkeypatch.delitem(sys.modules, "xau_bot.cli.main", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("xau_bot.cli.main", run_name="__main__")
    assert exc_info.value.code == 0

    monkeypatch.delitem(sys.modules, "xau_bot.cli.__main__", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("xau_bot.cli.__main__", run_name="__main__")
    assert exc_info.value.code == 0

    monkeypatch.setattr(sys, "argv", ["xau-bot", "--help"])
    monkeypatch.delitem(sys.modules, "xau_bot.cli.backtest", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("xau_bot.cli.backtest", run_name="__main__")
    assert exc_info.value.code == 0
