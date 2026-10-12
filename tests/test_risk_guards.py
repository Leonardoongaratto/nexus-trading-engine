from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from xau_bot.risk.guards import RiskGuard, get_trading_day
from xau_bot.risk.models import (
    AccountState,
    DailyRiskLimits,
    DailyTradingState,
    GuardCheckResult,
    GuardHaltReason,
)
from xau_bot.server import HealthStatusState


@pytest.fixture
def base_account() -> AccountState:
    return AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )


def test_guard_models_validation() -> None:
    with pytest.raises(ValueError, match="max_daily_loss_usd must be positive"):
        DailyRiskLimits(max_daily_loss_usd=-10.0)

    with pytest.raises(ValueError, match="max_daily_loss_usd must be finite"):
        DailyRiskLimits(max_daily_loss_usd=float("nan"))

    with pytest.raises(ValueError, match="max_daily_loss_percent must be between 0 and 1"):
        DailyRiskLimits(max_daily_loss_percent=0.0)

    with pytest.raises(ValueError, match="max_daily_loss_percent must be between 0 and 1"):
        DailyRiskLimits(max_daily_loss_percent=1.5)

    with pytest.raises(ValueError, match="max_daily_drawdown_usd must be positive"):
        DailyRiskLimits(max_daily_drawdown_usd=0.0)

    with pytest.raises(ValueError, match="max_daily_drawdown_percent must be between 0 and 1"):
        DailyRiskLimits(max_daily_drawdown_percent=-0.05)

    with pytest.raises(ValueError, match="max_consecutive_losses must be at least 1"):
        DailyRiskLimits(max_consecutive_losses=0)

    limits = DailyRiskLimits(
        max_daily_loss_usd=300.0,
        max_daily_loss_percent=0.03,
        max_daily_drawdown_usd=500.0,
        max_daily_drawdown_percent=0.05,
        max_consecutive_losses=3,
        reset_consecutive_losses_on_new_day=True,
    )
    assert limits.max_daily_loss_usd == 300.0
    assert limits.reset_consecutive_losses_on_new_day is True

    with pytest.raises(ValueError, match="start_balance must be positive"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=0.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=10000.0,
        )

    with pytest.raises(ValueError, match="start_equity must be positive"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=10000.0,
            start_equity=-1.0,
            peak_equity=10000.0,
            current_equity=10000.0,
        )

    with pytest.raises(ValueError, match="peak_equity must be positive"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=0.0,
            current_equity=10000.0,
        )

    with pytest.raises(ValueError, match="current_equity must be finite"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=float("nan"),
        )

    valid_zero_equity_state = DailyTradingState(
        trading_day=date(2026, 9, 25),
        start_balance=10000.0,
        start_equity=10000.0,
        peak_equity=10000.0,
        current_equity=0.0,
    )
    assert valid_zero_equity_state.current_equity == 0.0
    assert valid_zero_equity_state.daily_pnl == -10000.0

    with pytest.raises(ValueError, match="consecutive_losses cannot be negative"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=10000.0,
            consecutive_losses=-1,
        )

    with pytest.raises(ValueError, match="halt_time must be timezone-aware"):
        DailyTradingState(
            trading_day=date(2026, 9, 25),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=10000.0,
            halted=True,
            halt_time=datetime(2026, 9, 25, 12, 0),
        )

    with pytest.raises(ValueError, match="daily_pnl must be finite"):
        GuardCheckResult(
            can_trade=True,
            halted=False,
            daily_pnl=float("inf"),
        )


def test_get_trading_day_rollover() -> None:
    ny_tz = ZoneInfo("America/New_York")

    dt_midday = datetime(2026, 9, 23, 14, 0, tzinfo=ny_tz)
    assert get_trading_day(dt_midday, ny_tz=ny_tz) == date(2026, 9, 23)

    dt_pre_roll = datetime(2026, 9, 23, 16, 59, tzinfo=ny_tz)
    assert get_trading_day(dt_pre_roll, ny_tz=ny_tz) == date(2026, 9, 23)

    dt_roll = datetime(2026, 9, 23, 17, 0, tzinfo=ny_tz)
    assert get_trading_day(dt_roll, ny_tz=ny_tz) == date(2026, 9, 24)

    dt_post_roll = datetime(2026, 9, 23, 18, 5, tzinfo=ny_tz)
    assert get_trading_day(dt_post_roll, ny_tz=ny_tz) == date(2026, 9, 24)

    dt_fri_close = datetime(2026, 9, 25, 17, 5, tzinfo=ny_tz)
    assert get_trading_day(dt_fri_close, ny_tz=ny_tz) == date(2026, 9, 28)

    dt_sun_open = datetime(2026, 9, 27, 18, 5, tzinfo=ny_tz)
    assert get_trading_day(dt_sun_open, ny_tz=ny_tz) == date(2026, 9, 28)


def test_daily_loss_usd_limit_triggers_halt(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_daily_loss_usd=300.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    res = guard.check_guards(now, base_account)
    assert res.can_trade is True
    assert res.halted is False

    guard.register_closed_trade(pnl=-250.0, exit_time=now + timedelta(minutes=30))
    account_after_trade = AccountState(
        balance=9750.0,
        equity=9750.0,
        margin=0.0,
        free_margin=9750.0,
        leverage=100.0,
    )
    res_under = guard.check_guards(now + timedelta(minutes=31), account_after_trade)
    assert res_under.can_trade is True
    assert res_under.halted is False
    assert res_under.daily_pnl == -250.0

    guard.register_closed_trade(pnl=-60.0, exit_time=now + timedelta(hours=1))
    account_breached = AccountState(
        balance=9690.0,
        equity=9690.0,
        margin=0.0,
        free_margin=9690.0,
        leverage=100.0,
    )
    res_halted = guard.check_guards(now + timedelta(hours=1, minutes=1), account_breached)
    assert res_halted.can_trade is False
    assert res_halted.halted is True
    assert res_halted.halt_reason == GuardHaltReason.MAX_DAILY_LOSS_USD
    assert "Daily loss limit" in res_halted.details


def test_daily_loss_percent_limit_triggers_halt(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_daily_loss_percent=0.03)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    acc_under_limit = AccountState(
        balance=10000.0,
        equity=9800.0,
        margin=100.0,
        free_margin=9700.0,
        leverage=100.0,
    )
    res_under = guard.check_guards(now + timedelta(minutes=10), acc_under_limit)
    assert res_under.can_trade is True
    assert res_under.halted is False
    assert res_under.daily_pnl == -200.0

    floating_loss_account = AccountState(
        balance=10000.0,
        equity=9690.0,
        margin=100.0,
        free_margin=9590.0,
        leverage=100.0,
    )
    res = guard.check_guards(now + timedelta(minutes=15), floating_loss_account)
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.MAX_DAILY_LOSS_PERCENT

    acc_floating_open = AccountState(
        balance=10000.0,
        equity=10200.0,
        margin=100.0,
        free_margin=10100.0,
        leverage=100.0,
    )
    guard_floating = RiskGuard(limits=limits)
    guard_floating.initialize_day(now, acc_floating_open)

    acc_under_relative = AccountState(
        balance=10000.0,
        equity=9900.0,
        margin=100.0,
        free_margin=9800.0,
        leverage=100.0,
    )
    res_rel_under = guard_floating.check_guards(now + timedelta(minutes=20), acc_under_relative)
    assert res_rel_under.can_trade is True
    assert res_rel_under.halted is False

    acc_breach_relative = AccountState(
        balance=10000.0,
        equity=9880.0,
        margin=100.0,
        free_margin=9780.0,
        leverage=100.0,
    )
    res_rel_breach = guard_floating.check_guards(now + timedelta(minutes=25), acc_breach_relative)
    assert res_rel_breach.can_trade is False
    assert res_rel_breach.halted is True
    assert res_rel_breach.halt_reason == GuardHaltReason.MAX_DAILY_LOSS_PERCENT


def test_daily_drawdown_usd_limit_from_intraday_peak(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_daily_drawdown_usd=400.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    peak_account = AccountState(
        balance=10600.0,
        equity=10600.0,
        margin=0.0,
        free_margin=10600.0,
        leverage=100.0,
    )
    res_peak = guard.check_guards(now + timedelta(hours=1), peak_account)
    assert res_peak.can_trade is True
    assert guard.state.peak_equity == 10600.0

    pullback_account = AccountState(
        balance=10600.0,
        equity=10250.0,
        margin=100.0,
        free_margin=10150.0,
        leverage=100.0,
    )
    res_pullback = guard.check_guards(now + timedelta(hours=2), pullback_account)
    assert res_pullback.can_trade is True
    assert res_pullback.current_drawdown_usd == 350.0

    deep_dd_account = AccountState(
        balance=10600.0,
        equity=10150.0,
        margin=100.0,
        free_margin=10050.0,
        leverage=100.0,
    )
    res_halt = guard.check_guards(now + timedelta(hours=3), deep_dd_account)
    assert res_halt.can_trade is False
    assert res_halt.halted is True
    assert res_halt.halt_reason == GuardHaltReason.MAX_DAILY_DRAWDOWN_USD
    assert res_halt.current_drawdown_usd == 450.0


def test_daily_drawdown_usd_limit_with_open_floating_at_session_start() -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_open = AccountState(
        balance=10000.0,
        equity=10300.0,
        margin=100.0,
        free_margin=10200.0,
        leverage=100.0,
    )

    limits = DailyRiskLimits(max_daily_drawdown_usd=400.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, acc_open)

    acc_peak = AccountState(
        balance=10000.0,
        equity=10500.0,
        margin=100.0,
        free_margin=10400.0,
        leverage=100.0,
    )
    res_peak = guard.check_guards(now + timedelta(hours=1), acc_peak)
    assert res_peak.can_trade is True
    assert guard.state.peak_equity == 10500.0

    acc_reversal = AccountState(
        balance=10000.0,
        equity=10050.0,
        margin=100.0,
        free_margin=9950.0,
        leverage=100.0,
    )
    res_halt = guard.check_guards(now + timedelta(hours=2), acc_reversal)
    assert res_halt.can_trade is False
    assert res_halt.halted is True
    assert res_halt.halt_reason == GuardHaltReason.MAX_DAILY_DRAWDOWN_USD
    assert res_halt.current_drawdown_usd == 450.0


def test_daily_drawdown_percent_limit_from_intraday_peak(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_daily_drawdown_percent=0.04)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    dd_account = AccountState(
        balance=10000.0,
        equity=9580.0,
        margin=0.0,
        free_margin=9580.0,
        leverage=100.0,
    )
    res = guard.check_guards(now + timedelta(minutes=10), dd_account)
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.MAX_DAILY_DRAWDOWN_PERCENT


def test_consecutive_losses_circuit_breaker(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_consecutive_losses=3)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    guard.register_closed_trade(pnl=-20.0, exit_time=now + timedelta(minutes=10))
    assert guard.state.consecutive_losses == 1
    assert guard.check_guards(now + timedelta(minutes=11), base_account).can_trade is True

    guard.register_closed_trade(pnl=-25.0, exit_time=now + timedelta(minutes=20))
    assert guard.state.consecutive_losses == 2
    assert guard.check_guards(now + timedelta(minutes=21), base_account).can_trade is True

    guard.register_closed_trade(pnl=50.0, exit_time=now + timedelta(minutes=30))
    assert guard.state.consecutive_losses == 0
    assert guard.check_guards(now + timedelta(minutes=31), base_account).can_trade is True

    guard.register_closed_trade(pnl=-30.0, exit_time=now + timedelta(minutes=40))
    assert guard.state.consecutive_losses == 1

    guard.register_closed_trade(pnl=-30.0, exit_time=now + timedelta(minutes=50))
    assert guard.state.consecutive_losses == 2

    guard.register_closed_trade(pnl=-30.0, exit_time=now + timedelta(hours=1))
    assert guard.state.consecutive_losses == 3

    res = guard.check_guards(now + timedelta(hours=1, minutes=1), base_account)
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.MAX_CONSECUTIVE_LOSSES


def test_manual_halt_and_resume(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    guard.initialize_day(now, base_account)

    assert guard.check_guards(now, base_account).can_trade is True

    guard.trigger_manual_halt("User clicked emergency stop", halt_time=now)
    res_halt = guard.check_guards(now + timedelta(seconds=1), base_account)
    assert res_halt.can_trade is False
    assert res_halt.halted is True
    assert res_halt.halt_reason == GuardHaltReason.MANUAL_HALT
    assert "User clicked emergency stop" in res_halt.details

    guard.resume_trading()
    res_resume = guard.check_guards(now + timedelta(minutes=1), base_account)
    assert res_resume.can_trade is True
    assert res_resume.halted is False


@pytest.mark.parametrize(
    ("day2_balance", "day2_equity"),
    [
        (9850.0, 9790.0),
        (9850.0, 9900.0),
    ],
)
def test_day_transition_resets_limits(
    base_account: AccountState, day2_balance: float, day2_equity: float
) -> None:
    ny_tz = ZoneInfo("America/New_York")
    day1_time = datetime(2026, 9, 23, 15, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(
        max_daily_loss_usd=200.0,
        max_consecutive_losses=2,
        reset_consecutive_losses_on_new_day=True,
    )
    guard = RiskGuard(limits=limits)
    guard.initialize_day(day1_time, base_account)

    guard.register_closed_trade(pnl=-100.0, exit_time=day1_time + timedelta(minutes=10))
    guard.register_closed_trade(pnl=-110.0, exit_time=day1_time + timedelta(minutes=20))
    account_halted = AccountState(
        balance=9790.0,
        equity=9790.0,
        margin=0.0,
        free_margin=9790.0,
        leverage=100.0,
    )
    res_day1 = guard.check_guards(day1_time + timedelta(minutes=25), account_halted)
    assert res_day1.halted is True

    day2_time = datetime(2026, 9, 23, 18, 5, tzinfo=ny_tz)
    account_day2_open_with_floating = AccountState(
        balance=day2_balance,
        equity=day2_equity,
        margin=100.0,
        free_margin=day2_equity - 100.0,
        leverage=100.0,
    )
    res_day2 = guard.check_guards(day2_time, account_day2_open_with_floating)

    assert res_day2.can_trade is True
    assert res_day2.halted is False
    assert guard.state.trading_day == date(2026, 9, 24)
    assert guard.state.realized_pnl == 0.0
    assert guard.state.start_equity == day2_equity
    assert guard.state.current_equity == day2_equity
    assert guard.state.peak_equity == day2_equity
    assert guard.state.daily_pnl == 0.0
    assert guard.state.consecutive_losses == 0


def test_consecutive_losses_preserved_across_days_when_configured(
    base_account: AccountState,
) -> None:
    ny_tz = ZoneInfo("America/New_York")
    day1_time = datetime(2026, 9, 23, 15, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(
        max_consecutive_losses=3,
        reset_consecutive_losses_on_new_day=False,
    )
    guard = RiskGuard(limits=limits)
    guard.initialize_day(day1_time, base_account)

    guard.register_closed_trade(pnl=-50.0, exit_time=day1_time)
    guard.register_closed_trade(pnl=-50.0, exit_time=day1_time + timedelta(minutes=10))
    assert guard.state.consecutive_losses == 2

    day2_time = datetime(2026, 9, 23, 18, 5, tzinfo=ny_tz)
    guard.check_guards(day2_time, base_account)
    assert guard.state.consecutive_losses == 2


def test_intraday_equity_peak_ratchet(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    guard.initialize_day(now, base_account)
    assert guard.state.peak_equity == 10000.0

    acc_rise = AccountState(
        balance=10500.0, equity=10500.0, margin=0.0, free_margin=10500.0, leverage=100.0
    )
    guard.check_guards(now + timedelta(minutes=10), acc_rise)
    assert guard.state.peak_equity == 10500.0

    acc_drop = AccountState(
        balance=10500.0, equity=10200.0, margin=0.0, free_margin=10200.0, leverage=100.0
    )
    guard.check_guards(now + timedelta(minutes=20), acc_drop)
    assert guard.state.peak_equity == 10500.0


def test_sync_to_health_state(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    limits = DailyRiskLimits(max_daily_loss_usd=200.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, base_account)

    health_state = HealthStatusState()
    guard.sync_to_health_state(health_state, current_time=now, account=base_account)

    assert health_state.trading_halted is False
    assert health_state.kill_switches_active is False
    assert health_state.daily_pnl == 0.0

    guard.register_closed_trade(pnl=-250.0, exit_time=now + timedelta(minutes=5))
    halted_account = AccountState(
        balance=9750.0, equity=9750.0, margin=0.0, free_margin=9750.0, leverage=100.0
    )
    guard.check_guards(now + timedelta(minutes=6), halted_account)
    guard.sync_to_health_state(
        health_state, current_time=now + timedelta(minutes=6), account=halted_account
    )

    assert health_state.trading_halted is True
    assert health_state.kill_switches_active is True
    assert health_state.daily_pnl == -250.0


def test_guard_models_edge_validation() -> None:
    with pytest.raises(ValueError, match="max_daily_loss_percent must be finite"):
        DailyRiskLimits(max_daily_loss_percent=float("nan"))

    with pytest.raises(ValueError, match="realized_pnl must be finite"):
        DailyTradingState(
            trading_day=date(2026, 9, 23),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=10000.0,
            realized_pnl=float("nan"),
        )

    with pytest.raises(ValueError, match="trade_count cannot be negative"):
        DailyTradingState(
            trading_day=date(2026, 9, 23),
            start_balance=10000.0,
            start_equity=10000.0,
            peak_equity=10000.0,
            current_equity=10000.0,
            trade_count=-1,
        )

    with pytest.raises(ValueError, match="current_drawdown_usd cannot be negative"):
        GuardCheckResult(
            can_trade=True,
            halted=False,
            current_drawdown_usd=-1.0,
        )

    with pytest.raises(ValueError, match="consecutive_losses cannot be negative"):
        GuardCheckResult(
            can_trade=True,
            halted=False,
            consecutive_losses=-1,
        )


def test_get_trading_day_weekend_and_naive() -> None:
    ny_tz = ZoneInfo("America/New_York")

    with pytest.raises(ValueError, match="datetime must be timezone-aware"):
        get_trading_day(datetime(2026, 9, 23, 10, 0))

    sat_dt = datetime(2026, 9, 26, 12, 0, tzinfo=ny_tz)
    assert get_trading_day(sat_dt, ny_tz=ny_tz) == date(2026, 9, 28)

    sat_late = datetime(2026, 9, 26, 18, 0, tzinfo=ny_tz)
    assert get_trading_day(sat_late, ny_tz=ny_tz) == date(2026, 9, 28)


def test_risk_guard_uninitialized_lifecycle(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    uninit_guard = RiskGuard()
    with pytest.raises(RuntimeError, match="RiskGuard has not been initialized"):
        _ = uninit_guard.state

    with pytest.raises(RuntimeError, match="RiskGuard has not been initialized"):
        uninit_guard.register_closed_trade(pnl=10.0, exit_time=now)

    with pytest.raises(RuntimeError, match="RiskGuard has not been initialized"):
        uninit_guard.trigger_manual_halt("stop")

    uninit_guard.resume_trading()

    res_state = uninit_guard.update_account_state(now, base_account)
    assert res_state.trading_day == date(2026, 9, 23)


def test_risk_guard_timezone_awareness_and_validation(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        guard.initialize_day(datetime(2026, 9, 23, 10, 0), base_account)

    guard.initialize_day(now, base_account)

    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        guard.reset_day(datetime(2026, 9, 23, 10, 0), base_account)

    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        guard.update_account_state(datetime(2026, 9, 23, 10, 0), base_account)

    with pytest.raises(ValueError, match="current_time must be timezone-aware"):
        guard.check_guards(datetime(2026, 9, 23, 10, 0), base_account)

    with pytest.raises(ValueError, match="pnl must be finite"):
        guard.register_closed_trade(float("nan"), exit_time=now)

    with pytest.raises(ValueError, match="exit_time must be timezone-aware"):
        guard.register_closed_trade(10.0, exit_time=datetime(2026, 9, 23, 10, 0))

    with pytest.raises(ValueError, match="halt_time must be timezone-aware"):
        guard.trigger_manual_halt("stop", halt_time=datetime(2026, 9, 23, 10, 0))


def test_risk_guard_breakeven_trade_preserves_streak(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    guard.initialize_day(now, base_account)

    guard.register_closed_trade(-10.0, exit_time=now + timedelta(minutes=1))
    assert guard.state.consecutive_losses == 1
    guard.register_closed_trade(0.0, exit_time=now + timedelta(minutes=2))
    assert guard.state.consecutive_losses == 1


def test_risk_guard_trade_across_rollover_requires_session_update(
    base_account: AccountState,
) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    guard.initialize_day(now, base_account)

    next_day_time = datetime(2026, 9, 24, 10, 0, tzinfo=ny_tz)
    with pytest.raises(ValueError, match="Trade exit_time .* is in trading day"):
        guard.register_closed_trade(-20.0, exit_time=next_day_time)

    reset_state = guard.reset_day(next_day_time, base_account)
    assert reset_state.trade_count == 0
    assert reset_state.trading_day == date(2026, 9, 24)


def test_risk_guard_already_halted_returns_latched_result(base_account: AccountState) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    guard = RiskGuard()
    guard.initialize_day(now, base_account)
    guard.trigger_manual_halt("Emergency", halt_time=now + timedelta(hours=1))

    halt_res = guard.check_guards(now + timedelta(hours=1, minutes=1), base_account)
    assert halt_res.can_trade is False
    assert halt_res.halted is True
    assert halt_res.halt_reason == GuardHaltReason.MANUAL_HALT


def test_repro_bug_t09_01_negative_floating_at_session_open_does_not_false_halt() -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_open = AccountState(
        balance=10000.0,
        equity=9800.0,
        margin=100.0,
        free_margin=9700.0,
        leverage=100.0,
    )

    limits = DailyRiskLimits(max_daily_loss_usd=150.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, acc_open)

    res_open = guard.check_guards(now, acc_open)
    assert res_open.daily_pnl == 0.0
    assert res_open.current_drawdown_usd == 0.0
    assert res_open.can_trade is True
    assert res_open.halted is False


def test_repro_bug_t09_01_positive_floating_at_session_open_triggers_kill_switch_on_real_loss() -> (
    None
):
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_open = AccountState(
        balance=10000.0,
        equity=10300.0,
        margin=100.0,
        free_margin=10200.0,
        leverage=100.0,
    )

    limits = DailyRiskLimits(max_daily_loss_usd=300.0, max_daily_drawdown_usd=400.0)
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, acc_open)

    acc_reversal = AccountState(
        balance=10000.0,
        equity=9800.0,
        margin=100.0,
        free_margin=9700.0,
        leverage=100.0,
    )

    res = guard.check_guards(now + timedelta(hours=1), acc_reversal)
    assert res.daily_pnl == -500.0
    assert res.current_drawdown_usd == 500.0
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.MAX_DAILY_LOSS_USD


@pytest.mark.parametrize("depleted_equity", [0.0, -500.0])
def test_repro_bug_t09_02_zero_or_negative_equity_triggers_fail_closed_halt(
    depleted_equity: float,
) -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_open = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )

    limits = DailyRiskLimits(
        max_daily_loss_usd=300.0,
        max_daily_drawdown_usd=400.0,
        max_daily_loss_percent=0.03,
    )
    guard = RiskGuard(limits=limits)
    guard.initialize_day(now, acc_open)

    acc_depleted = AccountState(
        balance=10000.0,
        equity=depleted_equity,
        margin=0.0,
        free_margin=depleted_equity,
        leverage=100.0,
    )

    res = guard.check_guards(now + timedelta(hours=1), acc_depleted)
    expected_pnl = depleted_equity - 10000.0
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.ACCOUNT_EQUITY_DEPLETED
    assert "Account equity depleted" in res.details
    assert res.daily_pnl == expected_pnl
    assert guard.state.current_equity == depleted_equity


def test_equity_depletion_halts_even_without_daily_limits() -> None:
    ny_tz = ZoneInfo("America/New_York")
    now = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_open = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin=0.0,
        free_margin=10000.0,
        leverage=100.0,
    )

    guard = RiskGuard(limits=DailyRiskLimits())
    guard.initialize_day(now, acc_open)

    acc_depleted = AccountState(
        balance=10000.0,
        equity=0.0,
        margin=0.0,
        free_margin=0.0,
        leverage=100.0,
    )

    res = guard.check_guards(now + timedelta(hours=1), acc_depleted)
    assert res.can_trade is False
    assert res.halted is True
    assert res.halt_reason == GuardHaltReason.ACCOUNT_EQUITY_DEPLETED
    assert "Account equity depleted: 0.00 <= 0" in res.details


def test_rollover_and_initialization_with_depleted_equity_halts_fail_closed() -> None:
    ny_tz = ZoneInfo("America/New_York")
    day1_time = datetime(2026, 9, 23, 10, 0, tzinfo=ny_tz)

    acc_depleted = AccountState(
        balance=10000.0,
        equity=-500.0,
        margin=0.0,
        free_margin=-500.0,
        leverage=100.0,
    )

    guard = RiskGuard()
    guard.initialize_day(day1_time, acc_depleted)
    res_init = guard.check_guards(day1_time, acc_depleted)
    assert res_init.can_trade is False
    assert res_init.halted is True
    assert res_init.halt_reason == GuardHaltReason.ACCOUNT_EQUITY_DEPLETED

    day2_time = datetime(2026, 9, 23, 18, 5, tzinfo=ny_tz)
    res_rollover = guard.check_guards(day2_time, acc_depleted)
    assert res_rollover.can_trade is False
    assert res_rollover.halted is True
    assert res_rollover.halt_reason == GuardHaltReason.ACCOUNT_EQUITY_DEPLETED
    assert guard.state.trading_day == date(2026, 9, 24)
