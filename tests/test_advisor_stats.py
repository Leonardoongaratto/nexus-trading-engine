"""Tests for Advisor Setup Historical Statistics and Gate 1 Integration."""

from __future__ import annotations

import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from xau_bot.advisor.modes import ExecutionMode, ProposalManager, TradeProposal
from xau_bot.advisor.stats import (
    SetupHistoricalStats,
    SetupStatsAggregator,
    SetupStatsStore,
)
from xau_bot.backtest.models import BacktestTrade, TradeExitReason
from xau_bot.cli.backtest import backtest_main
from xau_bot.domain.models import (
    Direction,
    SetupStatus,
    SetupTriggerType,
    Timeframe,
    TradeSetup,
)
from xau_bot.notify.telegram import format_proposal_message


def _create_synthetic_backtest_data(
    tmp_path: Path,
) -> tuple[Path, Path]:
    ticks_file = tmp_path / "ticks.csv"
    start_dt = datetime(2026, 9, 24, 12, 0, 0, tzinfo=UTC)

    tick_lines = ["time,bid,ask,last,volume,flags"]
    curr_time = start_dt
    for i in range(20):
        p = 2650.0 + (1.5 if i % 2 == 0 else -1.5)
        tick_lines.append(f"{curr_time.isoformat()},{p:.2f},{p + 0.20:.2f},{p:.2f},1.0,6")
        curr_time += timedelta(seconds=1)
    ticks_file.write_text("\n".join(tick_lines) + "\n", encoding="utf-8")

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
            "trigger_type": "FVG_CE",
            "timeframe": "M5",
        }
    ]
    setups_file.write_text(json.dumps(setups_data), encoding="utf-8")
    return ticks_file, setups_file


def _create_dummy_setup(
    setup_id: str = "setup_1",
    trigger_type: SetupTriggerType = SetupTriggerType.FVG_CE,
    direction: Direction = Direction.BULLISH,
) -> TradeSetup:
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    entry = 2650.0
    sl = 2645.0 if direction == Direction.BULLISH else 2655.0
    tp = 2660.0 if direction == Direction.BULLISH else 2640.0
    return TradeSetup(
        id=setup_id,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=2.0,
        confluences=("H4_OB", "M15_CHOCH"),
        trigger_type=trigger_type,
        timeframe=Timeframe.M5,
        status=SetupStatus.PENDING,
    )


def _create_dummy_trade(
    trade_id: str,
    setup_id: str,
    return_r: float,
    direction: Direction = Direction.BULLISH,
) -> BacktestTrade:
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    is_win = return_r > 0
    return BacktestTrade(
        trade_id=trade_id,
        order_id=f"order_{trade_id}",
        direction=direction,
        volume=0.1,
        entry_time=t0,
        entry_price=2650.0,
        exit_time=t1,
        exit_price=2660.0 if is_win else 2645.0,
        exit_reason=TradeExitReason.TAKE_PROFIT if is_win else TradeExitReason.STOP_LOSS,
        initial_sl=2645.0,
        final_sl=2645.0,
        tp=2660.0,
        commission=1.5,
        swap=0.0,
        gross_pnl=100.0 if is_win else -50.0,
        net_pnl=98.5 if is_win else -51.5,
        return_r=return_r,
        mfe_price=2662.0,
        mae_price=2648.0,
        mfe_r=2.2,
        mae_r=0.4,
        duration_seconds=3600.0,
        setup_id=setup_id,
    )


def test_setup_historical_stats_invariants_and_serialization() -> None:
    stats = SetupHistoricalStats(
        setup_type="FVG_CE",
        sample_size=250,
        win_count=150,
        loss_count=100,
        win_rate=0.6,
        expectancy_r=0.45,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=2.25,
        gate_1_validated=True,
        validation_warning=None,
    )
    d = stats.to_dict()
    assert d["setup_type"] == "FVG_CE"
    assert d["sample_size"] == 250
    assert d["gate_1_validated"] is True
    assert d["validation_warning"] is None

    reconstructed = SetupHistoricalStats.from_dict(d)
    assert reconstructed == stats

    with pytest.raises(ValueError, match="sample_size must be non-negative"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=-1,
            win_count=0,
            loss_count=0,
            win_rate=0.5,
            expectancy_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=1.0,
            gate_1_validated=False,
            validation_warning="error",
        )

    with pytest.raises(ValueError, match="win_count and loss_count must be non-negative"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=1,
            win_count=-1,
            loss_count=2,
            win_rate=0.5,
            expectancy_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=1.0,
            gate_1_validated=False,
            validation_warning="error",
        )

    with pytest.raises(ValueError, match="win_rate must be between 0.0 and 1.0"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=10,
            win_count=6,
            loss_count=4,
            win_rate=1.5,
            expectancy_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=1.0,
            gate_1_validated=False,
            validation_warning="error",
        )

    with pytest.raises(ValueError, match="must be finite"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=2,
            win_count=1,
            loss_count=1,
            win_rate=0.5,
            expectancy_r=float("nan"),
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=1.0,
            gate_1_validated=False,
            validation_warning="error",
        )

    with pytest.raises(ValueError, match="must equal sample_size"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=10,
            win_count=5,
            loss_count=2,
            win_rate=0.5,
            expectancy_r=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            profit_factor=1.0,
            gate_1_validated=False,
            validation_warning="error",
        )

    with pytest.raises(ValueError, match="gate_1_validated invariant violation"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=3,
            win_count=2,
            loss_count=1,
            win_rate=0.66,
            expectancy_r=-0.5,
            avg_win_r=1.0,
            avg_loss_r=-1.0,
            profit_factor=1.0,
            gate_1_validated=True,
            validation_warning=None,
        )

    forged = {
        "setup_type": "FORGED",
        "sample_size": 3,
        "win_count": 2,
        "loss_count": 1,
        "win_rate": 0.66,
        "expectancy_r": -0.5,
        "avg_win_r": 1.0,
        "avg_loss_r": -1.0,
        "profit_factor": 1.0,
        "gate_1_validated": True,
        "validation_warning": None,
    }
    loaded_forged = SetupHistoricalStats.from_dict(forged)
    assert loaded_forged.gate_1_validated is False
    assert loaded_forged.validation_warning is not None

    tiny_exp = SetupHistoricalStats.evaluate_gate_1(
        setup_type="TINY",
        sample_size=250,
        win_count=125,
        loss_count=125,
        win_rate=0.5,
        expectancy_r=0.00004,
        avg_win_r=1.0,
        avg_loss_r=-1.0,
        profit_factor=1.0,
    )
    assert tiny_exp.expectancy_r == 0.0
    assert tiny_exp.gate_1_validated is False

    with pytest.raises(ValueError, match="validation_warning invariant violation"):
        SetupHistoricalStats(
            setup_type="TEST",
            sample_size=250,
            win_count=150,
            loss_count=100,
            win_rate=0.6,
            expectancy_r=0.45,
            avg_win_r=1.5,
            avg_loss_r=-1.0,
            profit_factor=2.25,
            gate_1_validated=True,
            validation_warning="Invalid warning for validated stats",
        )

    forged_win_rate = {
        "setup_type": "FORGED_WR",
        "sample_size": 10,
        "win_count": 2,
        "loss_count": 8,
        "win_rate": 0.9,
        "expectancy_r": -0.5,
        "avg_win_r": 1.0,
        "avg_loss_r": -1.0,
        "profit_factor": 1.0,
    }
    loaded_wr = SetupHistoricalStats.from_dict(forged_win_rate)
    assert loaded_wr.win_rate == 0.2


def test_gate_1_validation_evaluation() -> None:
    s1 = SetupHistoricalStats.evaluate_gate_1(
        setup_type="OB_EXTREME",
        sample_size=150,
        win_count=90,
        loss_count=60,
        win_rate=0.6,
        expectancy_r=0.5,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=2.25,
    )
    assert s1.gate_1_validated is False
    assert s1.validation_warning is not None
    assert "Portão 1" in s1.validation_warning

    s2 = SetupHistoricalStats.evaluate_gate_1(
        setup_type="OB_EXTREME",
        sample_size=250,
        win_count=100,
        loss_count=150,
        win_rate=0.4,
        expectancy_r=-0.1,
        avg_win_r=1.2,
        avg_loss_r=-1.0,
        profit_factor=0.8,
    )
    assert s2.gate_1_validated is False
    assert s2.validation_warning is not None

    s3 = SetupHistoricalStats.evaluate_gate_1(
        setup_type="OB_EXTREME",
        sample_size=210,
        win_count=120,
        loss_count=90,
        win_rate=0.5714,
        expectancy_r=0.35,
        avg_win_r=1.6,
        avg_loss_r=-1.0,
        profit_factor=2.13,
    )
    assert s3.gate_1_validated is True
    assert s3.validation_warning is None


def test_setup_stats_aggregator_empty_and_populated_trades() -> None:
    aggregator = SetupStatsAggregator()

    empty_stats = aggregator.aggregate_trades([], setup_type="FVG_CE")
    assert empty_stats.sample_size == 0
    assert empty_stats.win_rate == 0.0
    assert empty_stats.expectancy_r == 0.0
    assert empty_stats.gate_1_validated is False
    assert empty_stats.validation_warning is not None

    trades = [
        _create_dummy_trade("t1", "setup_A", return_r=2.0),
        _create_dummy_trade("t2", "setup_A", return_r=2.0),
        _create_dummy_trade("t3", "setup_A", return_r=2.0),
        _create_dummy_trade("t4", "setup_A", return_r=-1.0),
    ]
    st = aggregator.aggregate_trades(trades, setup_type="FVG_CE")
    assert st.sample_size == 4
    assert st.win_count == 3
    assert st.loss_count == 1
    assert math.isclose(st.win_rate, 0.75, rel_tol=1e-5)
    assert math.isclose(st.expectancy_r, 1.25, rel_tol=1e-5)
    assert math.isclose(st.avg_win_r, 2.0, rel_tol=1e-5)
    assert math.isclose(st.avg_loss_r, -1.0, rel_tol=1e-5)
    assert math.isclose(st.profit_factor, 6.0, rel_tol=1e-5)
    assert st.gate_1_validated is False


def test_setup_stats_aggregator_by_setup_map() -> None:
    setup_a = _create_dummy_setup("setup_a", trigger_type=SetupTriggerType.FVG_CE)
    setup_b = _create_dummy_setup("setup_b", trigger_type=SetupTriggerType.OB_EXTREME)

    trades = [
        _create_dummy_trade("t1", "setup_a", return_r=2.0),
        _create_dummy_trade("t2", "setup_a", return_r=-1.0),
        _create_dummy_trade("t3", "setup_b", return_r=1.5),
        _create_dummy_trade("t4", "setup_unknown", return_r=0.5),
    ]

    setups_map = {"setup_a": setup_a, "setup_b": setup_b}
    aggregator = SetupStatsAggregator()
    stats_by_trigger = aggregator.aggregate_by_trigger_type(trades, setups_map)

    assert "FVG_CE" in stats_by_trigger
    assert "OB_EXTREME" in stats_by_trigger
    assert "UNKNOWN" in stats_by_trigger

    fvg_stats = stats_by_trigger["FVG_CE"]
    assert fvg_stats.sample_size == 2
    assert fvg_stats.win_count == 1
    assert fvg_stats.loss_count == 1

    ob_stats = stats_by_trigger["OB_EXTREME"]
    assert ob_stats.sample_size == 1
    assert ob_stats.win_count == 1

    unknown_stats = stats_by_trigger["UNKNOWN"]
    assert unknown_stats.sample_size == 1
    assert unknown_stats.win_count == 1


def test_setup_stats_store_persistence_and_fallbacks(tmp_path: Path) -> None:
    store = SetupStatsStore()
    stats = SetupHistoricalStats.evaluate_gate_1(
        setup_type="FVG_CE",
        sample_size=220,
        win_count=130,
        loss_count=90,
        win_rate=0.59,
        expectancy_r=0.4,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=2.16,
    )
    store.set("FVG_CE", stats)

    out_file = tmp_path / "setup_stats.json"
    store.save_to_file(out_file)
    assert out_file.exists()

    loaded_store = SetupStatsStore.load_from_file(out_file)
    loaded_stats = loaded_store.get("FVG_CE")
    assert loaded_stats is not None
    assert loaded_stats.sample_size == 220
    assert loaded_stats.gate_1_validated is True

    missing_stats = loaded_store.get_or_default("UNKNOWN_TRIGGER")
    assert missing_stats.sample_size == 0
    assert missing_stats.gate_1_validated is False

    empty_store = SetupStatsStore.load_from_file(tmp_path / "non_existing.json")
    assert empty_store.count() == 0

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("invalid json")
    fallback_store = SetupStatsStore.load_from_file(corrupt_file)
    assert fallback_store.count() == 0


def test_trade_proposal_integration_with_proposal_manager() -> None:
    store = SetupStatsStore()
    stats = SetupHistoricalStats.evaluate_gate_1(
        setup_type="FVG_CE",
        sample_size=50,
        win_count=28,
        loss_count=22,
        win_rate=0.56,
        expectancy_r=0.25,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=1.9,
    )
    store.set("FVG_CE", stats)

    manager = ProposalManager(default_timeout_s=300.0, stats_store=store)
    setup = _create_dummy_setup(trigger_type=SetupTriggerType.FVG_CE)

    prop = manager.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)
    assert prop.historical_stats is not None
    assert prop.historical_stats.setup_type == "FVG_CE"
    assert prop.historical_stats.sample_size == 50
    assert prop.historical_stats.gate_1_validated is False

    d = prop.to_dict()
    assert "historical_stats" in d
    assert d["historical_stats"]["sample_size"] == 50
    assert d["historical_stats"]["gate_1_validated"] is False


def test_telegram_message_formatting_with_stats() -> None:
    unvalidated_stats = SetupHistoricalStats.evaluate_gate_1(
        setup_type="FVG_CE",
        sample_size=50,
        win_count=28,
        loss_count=22,
        win_rate=0.56,
        expectancy_r=0.25,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=1.9,
    )
    setup = _create_dummy_setup(trigger_type=SetupTriggerType.FVG_CE)
    now = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    prop1 = TradeProposal(
        id="prop_test_1",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        volume=0.05,
        risk_usd=50.0,
        risk_pct=0.01,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        historical_stats=unvalidated_stats,
    )
    msg1 = format_proposal_message(prop1)
    assert "Estatística Histórica (N=50)" in msg1
    assert "Win Rate: `56.0%`" in msg1
    assert "Exp: `+0.25 R`" in msg1
    assert "PF: `1.90`" in msg1
    assert "Portão 1" in msg1

    validated_stats = SetupHistoricalStats.evaluate_gate_1(
        setup_type="FVG_CE",
        sample_size=250,
        win_count=150,
        loss_count=100,
        win_rate=0.6,
        expectancy_r=0.45,
        avg_win_r=1.5,
        avg_loss_r=-1.0,
        profit_factor=2.25,
    )
    prop2 = TradeProposal(
        id="prop_test_2",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        volume=0.05,
        risk_usd=50.0,
        risk_pct=0.01,
        created_at=now,
        expires_at=now + timedelta(minutes=5),
        historical_stats=validated_stats,
    )
    msg2 = format_proposal_message(prop2)
    assert "Estatística Histórica (N=250)" in msg2
    assert "Win Rate: `60.0%`" in msg2
    assert "Exp: `+0.45 R`" in msg2
    assert "PF: `2.25`" in msg2
    assert "Estratégia não validada" not in msg2


def test_cli_backtest_export_stats(tmp_path: Path) -> None:
    ticks_file, setups_file = _create_synthetic_backtest_data(tmp_path)
    export_stats_file = tmp_path / "exported_stats.json"

    out = io.StringIO()
    err = io.StringIO()
    exit_code = backtest_main(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--export-stats",
            str(export_stats_file),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ],
        stdout=out,
        stderr=err,
    )
    assert exit_code == 0
    assert export_stats_file.exists()

    loaded = json.loads(export_stats_file.read_text(encoding="utf-8"))
    assert "GLOBAL" in loaded
    assert "sample_size" in loaded["GLOBAL"]
    assert "gate_1_validated" in loaded["GLOBAL"]
    assert "FVG_CE" in loaded
    assert "sample_size" in loaded["FVG_CE"]


def test_cli_backtest_export_stats_os_error(tmp_path: Path) -> None:
    ticks_file, setups_file = _create_synthetic_backtest_data(tmp_path)
    invalid_path = tmp_path / "invalid_dir"
    invalid_path.mkdir()

    out = io.StringIO()
    err = io.StringIO()
    exit_code = backtest_main(
        [
            "--ticks-path",
            str(ticks_file),
            "--setups-path",
            str(setups_file),
            "--export-stats",
            str(invalid_path),
            "--no-walk-forward",
            "--no-random-baseline",
            "--no-monte-carlo",
        ],
        stdout=out,
        stderr=err,
    )
    assert exit_code == 2
    assert "Erro ao exportar estatísticas de setups" in err.getvalue()
