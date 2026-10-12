import math
from dataclasses import replace
from datetime import date, datetime, timedelta, tzinfo
from typing import Protocol
from zoneinfo import ZoneInfo

from xau_bot.risk.models import (
    AccountState,
    DailyRiskLimits,
    DailyTradingState,
    GuardCheckResult,
    GuardHaltReason,
)

DEFAULT_NY_TZ: tzinfo = ZoneInfo("America/New_York")


class HealthStateSyncTarget(Protocol):
    trading_halted: bool
    kill_switches_active: bool
    daily_pnl: float


def get_trading_day(
    dt: datetime,
    ny_tz: tzinfo = DEFAULT_NY_TZ,
) -> date:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    dt_ny = dt.astimezone(ny_tz)
    candidate = dt_ny.date() + timedelta(days=1) if dt_ny.hour >= 17 else dt_ny.date()

    if candidate.weekday() == 5:
        candidate += timedelta(days=2)
    elif candidate.weekday() == 6:
        candidate += timedelta(days=1)

    return candidate


class RiskGuard:
    def __init__(
        self,
        limits: DailyRiskLimits | None = None,
        ny_tz: tzinfo = DEFAULT_NY_TZ,
    ) -> None:
        self.limits: DailyRiskLimits = limits if limits is not None else DailyRiskLimits()
        self._ny_tz: tzinfo = ny_tz
        self._state: DailyTradingState | None = None
        self._halt_details: str = ""

    @property
    def ny_tz(self) -> tzinfo:
        return self._ny_tz

    @property
    def state(self) -> DailyTradingState:
        if self._state is None:
            raise RuntimeError("RiskGuard has not been initialized with initialize_day")
        return self._state

    def _create_day_state(
        self,
        trading_day: date,
        account: AccountState,
        consecutive_losses: int,
    ) -> DailyTradingState:
        is_depleted = account.equity <= 0.0
        base_equity = (
            account.equity
            if not is_depleted
            else (account.balance if account.balance > 0.0 else 1.0)
        )
        base_balance = account.balance if account.balance > 0.0 else 1.0
        self._halt_details = (
            f"Account equity depleted: {account.equity:.2f} <= 0" if is_depleted else ""
        )
        return DailyTradingState(
            trading_day=trading_day,
            start_balance=base_balance,
            start_equity=base_equity,
            peak_equity=base_equity,
            current_equity=account.equity,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            consecutive_losses=consecutive_losses,
            trade_count=0,
            halted=is_depleted,
            halt_reason=GuardHaltReason.ACCOUNT_EQUITY_DEPLETED if is_depleted else None,
            halt_time=None,
        )

    def initialize_day(
        self,
        current_time: datetime,
        account: AccountState,
    ) -> DailyTradingState:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        trading_day = get_trading_day(current_time, self._ny_tz)
        self._state = self._create_day_state(trading_day, account, consecutive_losses=0)
        return self._state

    def reset_day(
        self,
        current_time: datetime,
        account: AccountState,
    ) -> DailyTradingState:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        trading_day = get_trading_day(current_time, self._ny_tz)
        return self._roll_to_new_day(account, trading_day)

    def _roll_to_new_day(
        self,
        account: AccountState,
        new_trading_day: date,
    ) -> DailyTradingState:
        consecutive_losses = (
            0
            if (self.limits.reset_consecutive_losses_on_new_day or self._state is None)
            else self._state.consecutive_losses
        )
        self._state = self._create_day_state(
            new_trading_day, account, consecutive_losses=consecutive_losses
        )
        return self._state

    def update_account_state(
        self,
        current_time: datetime,
        account: AccountState,
    ) -> DailyTradingState:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")
        if self._state is None:
            return self.initialize_day(current_time, account)

        current_trading_day = get_trading_day(current_time, self._ny_tz)
        if current_trading_day != self._state.trading_day:
            return self._roll_to_new_day(account, current_trading_day)

        state = self.state
        new_peak = max(state.peak_equity, account.equity)
        daily_pnl = account.equity - state.start_equity
        unrealized = daily_pnl - state.realized_pnl

        self._state = replace(
            state,
            current_equity=account.equity,
            peak_equity=new_peak,
            unrealized_pnl=unrealized,
        )
        return self._state

    def register_closed_trade(self, pnl: float, exit_time: datetime) -> None:
        if not math.isfinite(pnl):
            raise ValueError(f"pnl must be finite, got {pnl}")
        if exit_time.tzinfo is None:
            raise ValueError("exit_time must be timezone-aware")
        if self._state is None:
            raise RuntimeError("RiskGuard has not been initialized with initialize_day")

        current_trading_day = get_trading_day(exit_time, self._ny_tz)
        if current_trading_day != self._state.trading_day:
            raise ValueError(
                f"Trade exit_time {exit_time} is in trading day {current_trading_day}, "
                f"but current session is {self._state.trading_day}. "
                "Call update_account_state or check_guards first with "
                "the new session account state."
            )

        state = self.state
        new_realized = state.realized_pnl + pnl
        new_unrealized = state.daily_pnl - new_realized
        new_count = state.trade_count + 1

        if pnl < 0.0:
            new_losses = state.consecutive_losses + 1
        elif pnl > 0.0:
            new_losses = 0
        else:
            new_losses = state.consecutive_losses

        self._state = replace(
            state,
            realized_pnl=new_realized,
            unrealized_pnl=new_unrealized,
            trade_count=new_count,
            consecutive_losses=new_losses,
        )

    @property
    def has_state(self) -> bool:
        return self._state is not None

    @property
    def is_halted(self) -> bool:
        return self._state is not None and self._state.halted

    def trigger_manual_halt(
        self,
        reason: str = "Manual halt",
        halt_time: datetime | None = None,
    ) -> None:
        if self._state is None:
            raise RuntimeError("RiskGuard has not been initialized with initialize_day")
        if halt_time is not None and halt_time.tzinfo is None:
            raise ValueError("halt_time must be timezone-aware")

        self._halt_details = f"Manual halt: {reason}"
        self._state = replace(
            self.state,
            halted=True,
            halt_reason=GuardHaltReason.MANUAL_HALT,
            halt_time=halt_time or datetime.now(self._ny_tz),
        )

    def resume_trading(self) -> None:
        state = self._state
        if state is not None:
            self._halt_details = ""
            self._state = replace(
                state,
                halted=False,
                halt_reason=None,
                halt_time=None,
            )

    def _create_halt(
        self,
        state: DailyTradingState,
        current_time: datetime,
        reason: GuardHaltReason,
        details: str,
    ) -> GuardCheckResult:
        self._halt_details = details
        self._state = replace(
            state,
            halted=True,
            halt_reason=reason,
            halt_time=current_time,
        )
        return GuardCheckResult(
            can_trade=False,
            halted=True,
            halt_reason=reason,
            details=details,
            daily_pnl=state.daily_pnl,
            current_drawdown_usd=state.current_drawdown_usd,
            current_drawdown_percent=state.current_drawdown_percent,
            consecutive_losses=state.consecutive_losses,
        )

    def check_guards(
        self,
        current_time: datetime,
        account: AccountState,
    ) -> GuardCheckResult:
        if current_time.tzinfo is None:
            raise ValueError("current_time must be timezone-aware")

        self.update_account_state(current_time, account)
        state = self.state

        daily_pnl = state.daily_pnl
        dd_usd = state.current_drawdown_usd
        dd_pct = state.current_drawdown_percent

        if state.halted:
            return GuardCheckResult(
                can_trade=False,
                halted=True,
                halt_reason=state.halt_reason,
                details=self._halt_details,
                daily_pnl=daily_pnl,
                current_drawdown_usd=dd_usd,
                current_drawdown_percent=dd_pct,
                consecutive_losses=state.consecutive_losses,
            )

        if state.current_equity <= 0.0:
            details = f"Account equity depleted: {state.current_equity:.2f} <= 0"
            return self._create_halt(
                state, current_time, GuardHaltReason.ACCOUNT_EQUITY_DEPLETED, details
            )

        if (
            self.limits.max_consecutive_losses is not None
            and state.consecutive_losses >= self.limits.max_consecutive_losses
        ):
            details = (
                f"Consecutive losses limit reached: "
                f"{state.consecutive_losses} >= {self.limits.max_consecutive_losses}"
            )
            return self._create_halt(
                state, current_time, GuardHaltReason.MAX_CONSECUTIVE_LOSSES, details
            )

        if (
            self.limits.max_daily_loss_usd is not None
            and daily_pnl <= -self.limits.max_daily_loss_usd
        ):
            details = (
                f"Daily loss limit (USD) reached: "
                f"{daily_pnl:.2f} <= -{self.limits.max_daily_loss_usd:.2f}"
            )
            return self._create_halt(
                state, current_time, GuardHaltReason.MAX_DAILY_LOSS_USD, details
            )

        if self.limits.max_daily_loss_percent is not None:
            max_loss_usd = state.start_equity * self.limits.max_daily_loss_percent
            if daily_pnl <= -max_loss_usd:
                details = (
                    f"Daily loss limit (%) reached: {daily_pnl:.2f} <= -{max_loss_usd:.2f} "
                    f"({self.limits.max_daily_loss_percent * 100:.1f}%)"
                )
                return self._create_halt(
                    state, current_time, GuardHaltReason.MAX_DAILY_LOSS_PERCENT, details
                )

        if (
            self.limits.max_daily_drawdown_usd is not None
            and dd_usd >= self.limits.max_daily_drawdown_usd
        ):
            details = (
                f"Daily drawdown limit (USD) reached: "
                f"{dd_usd:.2f} >= {self.limits.max_daily_drawdown_usd:.2f}"
            )
            return self._create_halt(
                state, current_time, GuardHaltReason.MAX_DAILY_DRAWDOWN_USD, details
            )

        if (
            self.limits.max_daily_drawdown_percent is not None
            and dd_pct >= self.limits.max_daily_drawdown_percent
        ):
            details = (
                f"Daily drawdown limit (%) reached: {dd_pct * 100:.2f}% >= "
                f"{self.limits.max_daily_drawdown_percent * 100:.2f}%"
            )
            return self._create_halt(
                state, current_time, GuardHaltReason.MAX_DAILY_DRAWDOWN_PERCENT, details
            )

        return GuardCheckResult(
            can_trade=True,
            halted=False,
            halt_reason=None,
            details="All guards passed",
            daily_pnl=daily_pnl,
            current_drawdown_usd=dd_usd,
            current_drawdown_percent=dd_pct,
            consecutive_losses=state.consecutive_losses,
        )

    def sync_to_health_state(
        self,
        health_state: HealthStateSyncTarget,
        current_time: datetime,
        account: AccountState,
    ) -> None:
        res = self.check_guards(current_time, account)
        health_state.trading_halted = res.halted
        health_state.kill_switches_active = res.halted
        health_state.daily_pnl = res.daily_pnl
