"""Unit and integration tests for the Trade Journal module."""

from __future__ import annotations

import concurrent.futures
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from xau_bot.advisor.modes import ProposalStatus, TradeProposal
from xau_bot.domain.models import (
    Candle,
    Direction,
    OrderIntent,
    Swing,
    SwingType,
    Timeframe,
    TradeSetup,
)
from xau_bot.journal import (
    JournalEntry,
    JournalReporter,
    JournalRepository,
    LossCategory,
    LossRootCause,
    OutcomeData,
    OutcomeEvaluator,
    SnapshotData,
    TradeOutcome,
    create_snapshot_from_proposal,
)


def _make_candle(
    time: datetime,
    o: float,
    h: float,
    low: float,
    c: float,
    tf: Timeframe = Timeframe.M5,
) -> Candle:
    return Candle(
        time=time,
        open=o,
        high=h,
        low=low,
        close=c,
        volume=100.0,
        timeframe=tf,
    )


def test_snapshot_data_serialization() -> None:
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session="NY",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2635.0,
        swept_liquidity=True,
        displacement_atr=1.8,
        fvg_size_atr=0.9,
        choch_level=2632.0,
        spread_price=0.2,
        atr_m5=2.5,
        news_distance_min=45.0,
        confluences=("SWEEP", "FVG"),
    )

    d = snap.to_dict()
    assert d["session"] == "NY"
    assert d["poi_level"] == 2635.0
    assert d["swept_liquidity"] is True
    assert d["confluences"] == ["SWEEP", "FVG"]

    loaded = SnapshotData.from_dict(d)
    assert loaded.timestamp == now
    assert loaded.session == "NY"
    assert loaded.news_distance_min == 45.0
    assert loaded.confluences == ("SWEEP", "FVG")


def test_snapshot_data_from_dict_without_tz() -> None:
    d = {
        "timestamp": "2026-10-08T12:00:00",
        "session": "LONDON",
        "htf_bias": "BEARISH",
        "swing_high": 2660.0,
        "swing_low": 2630.0,
        "poi_type": "OrderBlock",
        "poi_level": 2655.0,
        "swept_liquidity": False,
        "displacement_atr": 1.2,
        "fvg_size_atr": 0.5,
        "choch_level": 2650.0,
        "spread_price": 0.25,
        "atr_m5": 2.0,
    }
    loaded = SnapshotData.from_dict(d)
    assert loaded.timestamp.tzinfo is not None
    assert loaded.news_distance_min is None


def test_outcome_data_serialization() -> None:
    out = OutcomeData(
        executed=True,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        fill_price=2640.1,
        exit_price=2660.0,
        slippage_points=1.0,
        mfe_r=2.1,
        mae_r=0.2,
        max_r=2.1,
        realized_r=2.0,
        pnl_usd=500.0,
        duration_seconds=3600.0,
        exit_reason="Take Profit Hit",
        post_exit_behavior="TP reached",
    )
    d = out.to_dict()
    assert d["executed"] is True
    assert d["realized_r"] == 2.0
    assert d["exit_price"] == 2660.0

    loaded = OutcomeData.from_dict(d)
    assert loaded.executed is True
    assert loaded.realized_r == 2.0
    assert loaded.exit_price == 2660.0
    assert loaded.fill_price == 2640.1


def test_journal_repository_crud() -> None:
    repo = JournalRepository(":memory:")
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
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
        atr_m5=2.0,
    )

    entry = JournalEntry(
        id="jrn_001",
        proposal_id="prop_001",
        snapshot=snap,
        ticket=12345,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        status=ProposalStatus.DISPATCHED,
        outcome=TradeOutcome.UNEXECUTED,
        created_at=now,
        updated_at=now,
    )

    repo.save_entry(entry)

    # Fetch by ID
    fetched = repo.get_entry("jrn_001")
    assert fetched is not None
    assert fetched.proposal_id == "prop_001"
    assert fetched.ticket == 12345
    assert fetched.outcome == TradeOutcome.UNEXECUTED

    # Fetch by proposal ID
    by_prop = repo.get_by_proposal_id("prop_001")
    assert by_prop is not None
    assert by_prop.id == "jrn_001"

    # Fetch by ticket
    by_ticket = repo.get_by_ticket(12345)
    assert by_ticket is not None
    assert by_ticket.proposal_id == "prop_001"

    # Count
    assert repo.count_entries() == 1
    assert repo.count_entries(outcome=TradeOutcome.UNEXECUTED) == 1
    assert repo.count_entries(outcome=TradeOutcome.WIN) == 0

    # Upsert on conflict
    entry.outcome = TradeOutcome.WIN
    entry.outcome_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2640.0,
        realized_r=2.0,
        pnl_usd=400.0,
    )
    repo.save_entry(entry)

    updated = repo.get_by_proposal_id("prop_001")
    assert updated is not None
    assert updated.outcome == TradeOutcome.WIN
    assert updated.outcome_data is not None
    assert updated.outcome_data.realized_r == 2.0
    assert repo.count_entries() == 1


def test_journal_repository_list_and_filters() -> None:
    repo = JournalRepository(":memory:")
    base_time = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)

    for i in range(5):
        t = base_time + timedelta(hours=i)
        snap = SnapshotData(
            timestamp=t,
            session="LONDON",
            htf_bias="BULLISH",
            swing_high=2650.0,
            swing_low=2620.0,
            poi_type="FVG",
            poi_level=2630.0 + i,
            swept_liquidity=True,
            displacement_atr=1.5,
            fvg_size_atr=0.8,
            choch_level=2625.0,
            spread_price=0.2,
            atr_m5=2.0,
        )
        outcome_choice = TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS
        entry = JournalEntry(
            id=f"jrn_{i}",
            proposal_id=f"prop_{i}",
            snapshot=snap,
            ticket=1000 + i,
            outcome=outcome_choice,
            created_at=t,
            updated_at=t,
        )
        repo.save_entry(entry)

    all_entries = repo.list_entries(limit=10)
    assert len(all_entries) == 5
    # verify descending order
    assert all_entries[0].id == "jrn_4"
    assert all_entries[-1].id == "jrn_0"

    # Filter by outcome
    wins = repo.list_entries(outcome=TradeOutcome.WIN)
    assert len(wins) == 3
    losses = repo.list_entries(outcome=TradeOutcome.LOSS)
    assert len(losses) == 2


def test_journal_repository_concurrency() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_concurrent.db"
        repo = JournalRepository(db_file)
        now = datetime.now(UTC)

        def worker(idx: int) -> None:
            snap = SnapshotData(
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
                atr_m5=2.0,
            )
            entry = JournalEntry(
                id=f"jrn_c_{idx}",
                proposal_id=f"prop_c_{idx}",
                snapshot=snap,
                ticket=2000 + idx,
                created_at=now,
                updated_at=now,
            )
            repo.save_entry(entry)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(20)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        count = repo.count_entries()
        repo.close()
        assert count == 20


def test_create_snapshot_from_proposal() -> None:
    now = datetime(2026, 10, 8, 14, 30, tzinfo=UTC)
    swing_high = Swing(
        time=now - timedelta(minutes=30),
        price=2655.0,
        type=SwingType.HIGH,
        confirmed_time=now - timedelta(minutes=15),
        index=10,
    )
    setup = TradeSetup(
        id="setup_test_01",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        time=now - timedelta(minutes=10),
        confirmed_time=now,
        risk_reward_ratio=2.0,
        target_swing=swing_high,
        confluences=("SWEEP", "FVG"),
    )
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.05,
        price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        magic=123456,
        comment="test",
        expiration=now + timedelta(minutes=30),
    )
    prop = TradeProposal(
        id="prop_snap_01",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        volume=0.05,
        risk_usd=50.0,
        risk_pct=1.0,
        created_at=now,
        expires_at=now + timedelta(minutes=30),
        setup=setup,
        intent=intent,
        atr=2.5,
    )

    snapshot = create_snapshot_from_proposal(prop)
    assert snapshot.session == "NY"
    assert snapshot.htf_bias == "BULLISH"
    assert snapshot.swing_high == 2655.0
    assert snapshot.poi_level == 2640.0
    assert snapshot.atr_m5 == 2.5
    assert snapshot.confluences == ("SWEEP", "FVG")


def test_outcome_evaluator_evaluate_position_bullish_win() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
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
        atr_m5=2.0,
    )
    entry = JournalEntry(
        id="jrn_w",
        proposal_id="prop_w",
        snapshot=snap,
        direction=Direction.BULLISH,
        created_at=now,
    )

    # 3 candles during position: entry at 2630, SL=2625 (risk=5), reaches 2642
    c1 = _make_candle(now + timedelta(minutes=5), 2630.0, 2635.0, 2628.0, 2634.0)
    c2 = _make_candle(now + timedelta(minutes=10), 2634.0, 2642.0, 2632.0, 2640.0)

    # exit at 2640 (realized gain = 10, risk=5 => +2.0R)
    out_data, outcome, loss_cause, loss_cat = evaluator.evaluate_position(
        entry=entry,
        exit_price=2640.0,
        exit_time=now + timedelta(minutes=15),
        position_candles=(c1, c2),
        fill_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        pnl_usd=500.0,
        exit_reason="Take Profit",
    )

    assert outcome == TradeOutcome.WIN
    assert out_data.realized_r == 2.0
    assert out_data.mfe_r == 2.4  # (2642 - 2630) / 5 = 12 / 5 = 2.4
    assert out_data.mae_r == 0.4  # (2630 - 2628) / 5 = 2 / 5 = 0.4
    assert loss_cause is None
    assert loss_cat is None


def test_outcome_evaluator_evaluate_position_breakeven() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
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
        atr_m5=2.0,
    )
    entry = JournalEntry(
        id="jrn_be",
        proposal_id="prop_be",
        snapshot=snap,
        direction=Direction.BULLISH,
        created_at=now,
    )

    c1 = _make_candle(now + timedelta(minutes=5), 2630.0, 2634.0, 2628.0, 2631.0)
    out_data, outcome, loss_cause, loss_cat = evaluator.evaluate_position(
        entry=entry,
        exit_price=2630.5,
        exit_time=now + timedelta(minutes=10),
        position_candles=(c1,),
        fill_price=2630.0,
        pnl_usd=25.0,
        exit_reason="Break Even",
    )

    assert outcome == TradeOutcome.BREAKEVEN
    assert 0.0 <= out_data.realized_r < 0.5
    assert loss_cause is None


def test_outcome_evaluator_loss_taxonomy() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    # 1. COUNTER_BIAS: HTF Bearish vs User Bullish
    snap_bias = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BEARISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        user_decision="BULLISH",
    )
    cause, cat = evaluator.diagnose_loss(snap_bias, risk_dist=5.0)
    assert cause == LossRootCause.COUNTER_BIAS
    assert cat == LossCategory.BAD_DECISION

    # 2. INSUFFICIENT_DISPLACEMENT
    snap_disp = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=0.5,  # < 1.0
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        user_decision="BULLISH",
    )
    cause, cat = evaluator.diagnose_loss(snap_disp, risk_dist=5.0)
    assert cause == LossRootCause.INSUFFICIENT_DISPLACEMENT
    assert cat == LossCategory.BAD_DECISION

    # 3. NEWS_PROXIMITY
    snap_news = SnapshotData(
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
        atr_m5=2.0,
        user_decision="BULLISH",
        news_distance_min=15.0,  # <= 30.0
    )
    cause, cat = evaluator.diagnose_loss(snap_news, risk_dist=5.0)
    assert cause == LossRootCause.NEWS_PROXIMITY
    assert cat == LossCategory.BAD_DECISION

    # 4. HIGH_SPREAD
    snap_spread = SnapshotData(
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
        spread_price=1.0,  # spread 1.0 / atr 2.0 = 0.5 > 0.35
        atr_m5=2.0,
        user_decision="BULLISH",
    )
    cause, cat = evaluator.diagnose_loss(snap_spread, risk_dist=5.0)
    assert cause == LossRootCause.HIGH_SPREAD
    assert cat == LossCategory.BAD_DECISION

    # 5. TIGHT_SL
    snap_sl = SnapshotData(
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
        atr_m5=5.0,  # risk_dist 1.0 / atr 5.0 = 0.2 < 0.5
        user_decision="BULLISH",
    )
    cause, cat = evaluator.diagnose_loss(snap_sl, risk_dist=1.0)
    assert cause == LossRootCause.TIGHT_SL
    assert cat == LossCategory.BAD_DECISION

    # 6. EXECUTION_FAILURE
    snap_ok = SnapshotData(
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
        atr_m5=2.0,
        user_decision="BULLISH",
    )
    cause, cat = evaluator.diagnose_loss(snap_ok, risk_dist=5.0, exit_reason="REJECTED by broker")
    assert cause == LossRootCause.EXECUTION_FAILURE

    # 7. PREMATURE_EXIT
    cause, cat = evaluator.diagnose_loss(
        snap_ok, risk_dist=5.0, exit_reason="Trailing stop hit early"
    )
    assert cause == LossRootCause.PREMATURE_EXIT

    # 8. NORMAL_VARIANCE
    cause, cat = evaluator.diagnose_loss(snap_ok, risk_dist=5.0, exit_reason="Stop Loss hit")
    assert cause == LossRootCause.NORMAL_VARIANCE
    assert cat == LossCategory.NORMAL_VARIANCE


def test_outcome_evaluator_unexecuted_trajectory() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
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
        atr_m5=2.0,
        user_decision="BULLISH",
    )
    entry = JournalEntry(
        id="jrn_unex",
        proposal_id="prop_unex",
        snapshot=snap,
        direction=Direction.BULLISH,
        created_at=now,
    )

    # 1. Entry never reached
    c_never = [
        _make_candle(now + timedelta(minutes=5), 2635.0, 2640.0, 2632.0, 2638.0),
    ]
    out_data, outcome, _, _ = evaluator.evaluate_unexecuted(
        entry=entry,
        candles=c_never,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
    )
    assert outcome == TradeOutcome.UNEXECUTED
    assert out_data.exit_reason == "Entry level never reached"

    # 2. Entry reached then TP hit (Win)
    c_win = [
        _make_candle(
            now + timedelta(minutes=5), 2632.0, 2634.0, 2629.0, 2631.0
        ),  # low 2629 <= 2630 entry
        _make_candle(
            now + timedelta(minutes=10), 2631.0, 2641.0, 2630.0, 2640.0
        ),  # high 2641 >= 2640 TP
    ]
    out_data, outcome, _, _ = evaluator.evaluate_unexecuted(
        entry=entry,
        candles=c_win,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
    )
    assert outcome == TradeOutcome.WIN
    assert out_data.realized_r == 2.0

    # 3. Entry reached then SL hit (Loss)
    c_loss = [
        _make_candle(now + timedelta(minutes=5), 2632.0, 2634.0, 2629.0, 2631.0),
        _make_candle(
            now + timedelta(minutes=10), 2631.0, 2632.0, 2624.0, 2625.0
        ),  # low 2624 <= 2625 SL
    ]
    out_data, outcome, cause, cat = evaluator.evaluate_unexecuted(
        entry=entry,
        candles=c_loss,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
    )
    assert outcome == TradeOutcome.LOSS
    assert out_data.realized_r == -1.0
    assert cause == LossRootCause.NORMAL_VARIANCE


def test_journal_reporter_markdown_and_svg() -> None:
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
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
        atr_m5=2.0,
        user_decision="APPROVED",
    )
    out_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2640.0,
        mfe_r=2.2,
        mae_r=0.3,
        realized_r=2.0,
        pnl_usd=400.0,
        duration_seconds=1200.0,
        exit_reason="Take Profit Hit",
    )
    entry = JournalEntry(
        id="jrn_rep_01",
        proposal_id="prop_rep_01",
        snapshot=snap,
        ticket=9999,
        outcome=TradeOutcome.WIN,
        outcome_data=out_data,
        created_at=now,
    )

    svg = JournalReporter.generate_svg_chart(entry)
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "2630.00" in svg

    report = JournalReporter.generate_markdown_report(entry)
    assert "# Trade Post-Mortem Report" in report
    assert "prop_rep_01" in report
    assert "+2.00R" in report
    assert "+400.00" in report
    assert "<svg" in report

    with tempfile.TemporaryDirectory() as tmp_dir:
        exported = JournalReporter.export_report_file(entry, tmp_dir)
        p = Path(exported)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "prop_rep_01" in content


def test_outcome_evaluator_bearish_trades_and_post_exit() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BEARISH",
        swing_high=2660.0,
        swing_low=2630.0,
        poi_type="OrderBlock",
        poi_level=2650.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2645.0,
        spread_price=0.2,
        atr_m5=2.0,
    )
    entry_bear = JournalEntry(
        id="jrn_bear",
        proposal_id="prop_bear",
        snapshot=snap,
        direction=Direction.BEARISH,
        created_at=now,
    )

    # Candles during position
    c1 = _make_candle(now + timedelta(minutes=5), 2650.0, 2652.0, 2640.0, 2642.0)
    c2 = _make_candle(now + timedelta(minutes=10), 2642.0, 2645.0, 2628.0, 2630.0)

    # 1. Bearish WIN: entry 2650, SL 2660 (risk 10), exit at 2630 => +2.0R
    out_win, outcome_win, _, _ = evaluator.evaluate_position(
        entry=entry_bear,
        exit_price=2630.0,
        exit_time=now + timedelta(minutes=15),
        position_candles=(c1, c2),
        fill_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
        pnl_usd=500.0,
    )
    assert outcome_win == TradeOutcome.WIN
    assert out_win.realized_r == 2.0
    assert out_win.mfe_r == 2.2  # (2650 - 2628) / 10 = 2.2
    assert out_win.mae_r == 0.2  # (2652 - 2650) / 10 = 0.2

    # 2. Bearish subsequent TP reached after early exit
    c_sub = [
        _make_candle(now + timedelta(minutes=20), 2640.0, 2642.0, 2625.0, 2626.0),
    ]
    out_early, _, _, _ = evaluator.evaluate_position(
        entry=entry_bear,
        exit_price=2640.0,
        exit_time=now + timedelta(minutes=15),
        position_candles=(c1,),
        subsequent_candles=c_sub,
        fill_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
    )
    assert out_early.post_exit_behavior == "TP reached after exit"


def test_outcome_evaluator_unexecuted_bearish_and_expiration() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BEARISH",
        swing_high=2660.0,
        swing_low=2630.0,
        poi_type="OrderBlock",
        poi_level=2650.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2645.0,
        spread_price=0.2,
        atr_m5=2.0,
    )
    entry_bear = JournalEntry(
        id="jrn_unex_bear",
        proposal_id="prop_unex_bear",
        snapshot=snap,
        direction=Direction.BEARISH,
        created_at=now,
    )

    # 1. Bearish entry hit and TP hit
    c_win = [
        _make_candle(
            now + timedelta(minutes=5), 2648.0, 2652.0, 2645.0, 2648.0
        ),  # high 2652 >= 2650 entry
        _make_candle(
            now + timedelta(minutes=10), 2648.0, 2649.0, 2628.0, 2630.0
        ),  # low 2628 <= 2630 TP
    ]
    out_w, outc_w, _, _ = evaluator.evaluate_unexecuted(
        entry=entry_bear,
        candles=c_win,
        entry_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
    )
    assert outc_w == TradeOutcome.WIN
    assert out_w.realized_r == 2.0

    # 2. Bearish entry hit and SL hit
    c_loss = [
        _make_candle(now + timedelta(minutes=5), 2648.0, 2652.0, 2645.0, 2648.0),
        _make_candle(
            now + timedelta(minutes=10), 2652.0, 2662.0, 2650.0, 2660.0
        ),  # high 2662 >= 2660 SL
    ]
    out_l, outc_l, _, _ = evaluator.evaluate_unexecuted(
        entry=entry_bear,
        candles=c_loss,
        entry_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
    )
    assert outc_l == TradeOutcome.LOSS
    assert out_l.realized_r == -1.0

    # 3. Bearish entry hit and neither hit (expired)
    c_exp = [
        _make_candle(now + timedelta(minutes=5), 2648.0, 2652.0, 2645.0, 2648.0),
        _make_candle(now + timedelta(minutes=10), 2648.0, 2654.0, 2646.0, 2648.0),
    ]
    out_e, outc_e, _, _ = evaluator.evaluate_unexecuted(
        entry=entry_bear,
        candles=c_exp,
        entry_price=2650.0,
        stop_loss=2660.0,
        take_profit=2630.0,
    )
    assert outc_e == TradeOutcome.UNEXECUTED
    assert out_e.exit_reason == "Hypothetical position expired"


def test_create_snapshot_from_proposal_bearish_and_sessions() -> None:
    now_asia = datetime(2026, 10, 8, 4, 0, tzinfo=UTC)
    swing_low = Swing(
        time=now_asia - timedelta(minutes=30),
        price=2620.0,
        type=SwingType.LOW,
        confirmed_time=now_asia - timedelta(minutes=15),
        index=5,
    )
    setup = TradeSetup(
        id="setup_asia",
        direction=Direction.BEARISH,
        entry_price=2645.0,
        stop_loss=2655.0,
        take_profit=2625.0,
        time=now_asia,
        confirmed_time=now_asia,
        risk_reward_ratio=2.0,
        target_swing=swing_low,
    )
    prop = TradeProposal(
        id="prop_asia",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=Direction.BEARISH,
        entry_price=2645.0,
        stop_loss=2655.0,
        take_profit=2625.0,
        volume=0.01,
        risk_usd=10.0,
        risk_pct=1.0,
        created_at=now_asia,
        expires_at=now_asia + timedelta(minutes=30),
        setup=setup,
    )

    snap = create_snapshot_from_proposal(prop)
    assert snap.session == "ASIA"
    assert snap.htf_bias == "BEARISH"
    assert snap.swing_low == 2620.0


def test_journal_reporter_loss_and_breakeven_reports() -> None:
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
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
        atr_m5=2.0,
    )

    # 1. Loss report
    out_loss = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2625.0,
        realized_r=-1.0,
        pnl_usd=-200.0,
        exit_reason="Stop Loss",
    )
    entry_loss = JournalEntry(
        id="jrn_loss_rep",
        proposal_id="prop_loss_rep",
        snapshot=snap,
        outcome=TradeOutcome.LOSS,
        loss_cause=LossRootCause.NORMAL_VARIANCE,
        loss_category=LossCategory.NORMAL_VARIANCE,
        outcome_data=out_loss,
        created_at=now,
    )
    rep_loss = JournalReporter.generate_markdown_report(entry_loss)
    assert "NORMAL_VARIANCE" in rep_loss
    assert "-1.00R" in rep_loss

    # 2. Breakeven report
    entry_be = JournalEntry(
        id="jrn_be_rep",
        proposal_id="prop_be_rep",
        snapshot=snap,
        outcome=TradeOutcome.BREAKEVEN,
        created_at=now,
    )
    rep_be = JournalReporter.generate_markdown_report(entry_be)
    assert "BREAKEVEN" in rep_be

    # 3. Unexecuted with post exit report
    out_unex = OutcomeData(
        executed=False,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        realized_r=2.0,
        mfe_r=2.5,
        mae_r=0.2,
        exit_reason="Expired",
        post_exit_behavior="TP reached later",
    )
    entry_unex = JournalEntry(
        id="jrn_unex_rep",
        proposal_id="prop_unex_rep",
        snapshot=snap,
        outcome=TradeOutcome.UNEXECUTED,
        outcome_data=out_unex,
        created_at=now,
    )
    rep_unex = JournalReporter.generate_markdown_report(entry_unex)
    assert "Hipotético" in rep_unex

    # 4. Empty SVG
    entry_empty = JournalEntry(
        id="jrn_emp",
        proposal_id="prop_emp",
        snapshot=SnapshotData(
            timestamp=now,
            session="",
            htf_bias="",
            swing_high=0.0,
            swing_low=0.0,
            poi_type="",
            poi_level=0.0,
            swept_liquidity=False,
            displacement_atr=0.0,
            fvg_size_atr=0.0,
            choch_level=0.0,
            spread_price=0.0,
            atr_m5=0.0,
        ),
        created_at=now,
    )
    assert JournalReporter.generate_svg_chart(entry_empty) == ""


def test_journal_repository_update_outcome_and_time_filters() -> None:
    repo = JournalRepository(":memory:")
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
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
        atr_m5=2.0,
    )
    entry = JournalEntry(
        id="jrn_filt_01",
        proposal_id="prop_filt_01",
        snapshot=snap,
        created_at=now,
    )
    repo.save_entry(entry)

    # Update outcome
    out_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2640.0,
        realized_r=2.0,
    )
    updated = repo.update_outcome(
        proposal_id="prop_filt_01",
        outcome_data=out_data,
        outcome=TradeOutcome.WIN,
        status=ProposalStatus.DISPATCHED,
    )
    assert updated is True

    # Filter with start_time and end_time
    found = repo.list_entries(
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        outcome=TradeOutcome.WIN,
    )
    assert len(found) == 1
    assert found[0].proposal_id == "prop_filt_01"

    empty_filt = repo.list_entries(start_time=now + timedelta(days=1))
    assert len(empty_filt) == 0

    repo.close()


def test_outcome_evaluator_simultaneous_sl_tp_candle_conservative() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2660.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
    )
    entry = JournalEntry(
        id="jrn_ambiguous",
        proposal_id="prop_ambiguous",
        snapshot=snap,
        direction=Direction.BULLISH,
        created_at=now,
    )

    # Candle 0 touches entry (2630.0)
    c0 = _make_candle(now, o=2632.0, h=2633.0, low=2629.0, c=2631.0)
    # Candle 1 touches BOTH stop_loss (2625.0) and take_profit (2640.0) in the same bar
    c1 = _make_candle(now + timedelta(minutes=5), o=2631.0, h=2645.0, low=2620.0, c=2635.0)

    out_data, outcome, loss_cause, loss_cat = evaluator.evaluate_unexecuted(
        entry=entry,
        candles=[c0, c1],
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
    )

    # Conservative backtest convention: SL hit first
    assert outcome == TradeOutcome.LOSS
    assert out_data.realized_r == -1.0
    assert out_data.exit_price == 2625.0
    # Crucially, exit candle c1 must be included in MFE and MAE before break!
    assert out_data.mfe_r >= 2.0  # high was 2645.0, +15.0 / 5.0 risk_dist = 3.0R
    assert out_data.mae_r >= 2.0  # low was 2620.0, -10.0 / 5.0 risk_dist = 2.0R


def test_create_snapshot_from_proposal_real_setup_attributes() -> None:
    now = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)

    class DummyPOI:
        displacement_atr_ratio = 2.4
        size = 3.6

    setup = TradeSetup(
        id="setup_rich_context",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        time=now,
        confirmed_time=now,
        risk_reward_ratio=2.0,
        confluences=("HTF_COUNTER", "SWEEP_CONFIRMED"),
    )
    object.__setattr__(setup, "poi", DummyPOI())

    prop = TradeProposal(
        id="prop_rich_context",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        entry_price=2640.0,
        stop_loss=2630.0,
        take_profit=2660.0,
        volume=0.1,
        risk_usd=100.0,
        risk_pct=1.0,
        created_at=now,
        expires_at=now + timedelta(minutes=30),
        setup=setup,
        atr=2.0,
        status=ProposalStatus.APPROVED,
    )

    snap = create_snapshot_from_proposal(
        prop,
        spread_price=0.35,
    )

    assert snap.displacement_atr == 2.4
    assert snap.fvg_size_atr == 1.8  # 3.6 / 2.0 ATR = 1.8
    assert snap.htf_bias == "BEARISH"  # HTF_COUNTER on BULLISH direction
    assert snap.spread_price == 0.35
    assert snap.user_decision == "APPROVED"


def test_outcome_evaluator_loss_counter_bias_with_direction() -> None:
    evaluator = OutcomeEvaluator()
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BEARISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="FVG",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        user_decision="PENDING",
    )
    entry = JournalEntry(
        id="jrn_cb",
        proposal_id="prop_cb",
        snapshot=snap,
        direction=Direction.BULLISH,
        created_at=now,
    )

    # Bullish trade against Bearish HTF bias resulting in loss
    out_data, outcome, loss_cause, loss_cat = evaluator.evaluate_position(
        entry=entry,
        exit_price=2620.0,
        exit_time=now + timedelta(minutes=30),
        position_candles=[_make_candle(now, 2630.0, 2632.0, 2618.0, 2620.0)],
        fill_price=2630.0,
        stop_loss=2625.0,
    )

    assert outcome == TradeOutcome.LOSS
    assert loss_cause == LossRootCause.COUNTER_BIAS
    assert loss_cat == LossCategory.BAD_DECISION


def test_journal_repository_get_entry_supports_proposal_id_or_id() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_get.db")
        repo = JournalRepository(db_path=db_path)
        now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

        snap = SnapshotData(
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
            atr_m5=2.0,
        )
        entry = JournalEntry(
            id="jrn_unique_id",
            proposal_id="prop_unique_id",
            snapshot=snap,
            created_at=now,
        )
        repo.save_entry(entry)

        # Lookup by entry ID
        found_by_id = repo.get_entry("jrn_unique_id")
        assert found_by_id is not None
        assert found_by_id.proposal_id == "prop_unique_id"

        # Lookup by proposal ID
        found_by_prop = repo.get_entry("prop_unique_id")
        assert found_by_prop is not None
        repo.close()


def test_outcome_data_effective_and_theoretical_rr_serialization() -> None:
    outcome = OutcomeData(
        executed=True,
        entry_price=2650.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        fill_price=2648.0,
        effective_rr=2.75,
        theoretical_rr=2.0,
    )
    d = outcome.to_dict()
    assert d["effective_rr"] == 2.75
    assert d["theoretical_rr"] == 2.0

    restored = OutcomeData.from_dict(d)
    assert restored.effective_rr == 2.75
    assert restored.theoretical_rr == 2.0
