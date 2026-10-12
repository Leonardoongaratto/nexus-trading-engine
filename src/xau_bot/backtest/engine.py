import math
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from xau_bot.backtest.execution import SimulatedBroker
from xau_bot.backtest.models import (
    BacktestConfig,
    BacktestOrder,
    BacktestResult,
    EquityPoint,
    OrderType,
    TradeExitReason,
)
from xau_bot.backtest.slippage import FixedSlippage, SlippageModel, ZeroSlippage
from xau_bot.domain.models import Candle, Direction, Tick, TradeSetup
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.risk.guards import RiskGuard
from xau_bot.risk.models import AccountState, PositionRiskConfig
from xau_bot.risk.position import calculate_lot_size
from xau_bot.smc.swings import current_atr


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        broker: SimulatedBroker | None = None,
        trade_filter: CompositeTradeFilter | None = None,
        risk_guard: RiskGuard | None = None,
        position_risk_config: PositionRiskConfig | None = None,
        slippage_model: SlippageModel | None = None,
        close_positions_at_end: bool = True,
    ) -> None:
        if broker is not None and (
            config is not None or slippage_model is not None or position_risk_config is not None
        ):
            raise ValueError(
                "Cannot specify both broker and config/slippage_model/position_risk_config; "
                "configure the broker directly"
            )

        if broker is not None:
            self.broker = broker
            self.config = broker.config
            self.slippage_model = broker.slippage_model
            self.position_risk_config = broker.risk_config
        else:
            self.config = config if config is not None else BacktestConfig()
            if slippage_model is not None:
                self.slippage_model = slippage_model
            elif self.config.slippage_points > 0:
                self.slippage_model = FixedSlippage(
                    points=self.config.slippage_points,
                    point_size=self.config.symbol_spec.point,
                )
            else:
                self.slippage_model = ZeroSlippage()

            self.position_risk_config = (
                position_risk_config if position_risk_config is not None else PositionRiskConfig()
            )
            self.broker = SimulatedBroker(
                config=self.config,
                slippage_model=self.slippage_model,
                risk_config=self.position_risk_config,
            )

        self.trade_filter: CompositeTradeFilter | None = trade_filter
        self.risk_guard: RiskGuard | None = risk_guard
        self.close_positions_at_end: bool = close_positions_at_end

    def _account_state(self) -> AccountState:
        return AccountState(
            balance=self.broker.balance,
            equity=self.broker.equity,
            margin=self.broker.used_margin,
            free_margin=self.broker.free_margin,
            leverage=self.broker.config.leverage,
        )

    def _equity_point(self, timestamp: datetime) -> EquityPoint:
        peak = self.broker.peak_equity
        dd_usd = max(0.0, peak - self.broker.equity)
        dd_pct = (dd_usd / peak) * 100.0 if peak > 0 else 0.0
        return EquityPoint(
            timestamp=timestamp,
            balance=round(self.broker.balance, 2),
            equity=round(self.broker.equity, 2),
            drawdown_usd=round(dd_usd, 2),
            drawdown_pct=round(dd_pct, 4),
        )

    def _cancel_pending_on_halt(self) -> None:
        for pending_order in list(self.broker.pending_orders):
            self.broker.cancel_order(pending_order.order_id, reason="KILL_SWITCH")

    def run(
        self,
        ticks: Sequence[Tick],
        setups: Sequence[TradeSetup] | None = None,
        orders: Sequence[BacktestOrder] | None = None,
        candles: Sequence[Candle] | None = None,
    ) -> BacktestResult:
        if not ticks:
            raise ValueError("ticks sequence cannot be empty")

        for idx in range(1, len(ticks)):
            if ticks[idx].time < ticks[idx - 1].time:
                raise ValueError(
                    f"Ticks must be in non-decreasing chronological order: "
                    f"{ticks[idx].time} < {ticks[idx - 1].time} at index {idx}"
                )

        sorted_setups: list[TradeSetup] = []
        if setups:
            sorted_setups = sorted(setups, key=lambda s: s.confirmed_time)

        sorted_orders: list[BacktestOrder] = []
        if orders:
            sorted_orders = sorted(orders, key=lambda o: o.created_at)

        sorted_candles: list[Candle] = []
        if candles:
            sorted_candles = sorted(candles, key=lambda c: c.time)

        initial_account = self._account_state()
        if self.risk_guard is not None:
            self.risk_guard.initialize_day(ticks[0].time, initial_account)

        equity_curve: list[EquityPoint] = [self._equity_point(ticks[0].time)]

        candle_idx = 0
        total_candles = len(sorted_candles)
        closed_candles: list[Candle] = []
        current_atr_val: float | None = (
            current_atr(sorted_candles[:50], period=14) if total_candles >= 14 else None
        )

        setup_idx = 0
        order_idx = 0
        total_setups = len(sorted_setups)
        total_orders = len(sorted_orders)
        daily_close_equity: dict[date, float] = {}

        for tick in ticks:
            spread = tick.ask - tick.bid
            curr_account = self._account_state()

            candle_advanced = False
            while candle_idx < total_candles:
                candle = sorted_candles[candle_idx]
                close_time = candle.time + timedelta(seconds=candle.timeframe.duration_seconds)
                if close_time <= tick.time:
                    closed_candles.append(candle)
                    candle_idx += 1
                    candle_advanced = True
                else:
                    break

            if candle_advanced and len(closed_candles) >= 14:
                current_atr_val = current_atr(closed_candles[-50:], period=14)

            if self.risk_guard is not None:
                self.risk_guard.update_account_state(tick.time, curr_account)
                guard_res = self.risk_guard.check_guards(tick.time, curr_account)
                if not guard_res.can_trade:
                    self._cancel_pending_on_halt()

            while setup_idx < total_setups and sorted_setups[setup_idx].confirmed_time <= tick.time:
                setup = sorted_setups[setup_idx]
                setup_idx += 1

                if setup.valid_until is not None and tick.time >= setup.valid_until:
                    continue

                if self.trade_filter is not None:
                    filter_res = self.trade_filter.check(
                        current_time=tick.time, spread=spread, atr=current_atr_val
                    )
                    if not filter_res.can_trade:
                        continue

                if self.risk_guard is not None:
                    guard_res = self.risk_guard.check_guards(tick.time, curr_account)
                    if not guard_res.can_trade:
                        self._cancel_pending_on_halt()
                        continue

                lot_res = calculate_lot_size(
                    account=curr_account,
                    spec=self.config.symbol_spec,
                    config=self.position_risk_config,
                    direction=setup.direction,
                    entry_price=setup.entry_price,
                    stop_loss=setup.stop_loss,
                    spread=spread,
                )
                if not lot_res.success or lot_res.lot <= 0:
                    continue

                direction = setup.direction
                order_type = (
                    OrderType.BUY_LIMIT if direction == Direction.BULLISH else OrderType.SELL_LIMIT
                )
                new_order = BacktestOrder(
                    order_id=f"ORD-{setup.id}",
                    setup_id=setup.id,
                    direction=direction,
                    order_type=order_type,
                    volume=lot_res.lot,
                    limit_price=setup.entry_price,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit,
                    created_at=setup.confirmed_time,
                    expires_at=setup.valid_until,
                )
                self.broker.submit_order(new_order)

            while order_idx < total_orders and sorted_orders[order_idx].created_at <= tick.time:
                order = sorted_orders[order_idx]
                order_idx += 1

                if self.trade_filter is not None:
                    filter_res = self.trade_filter.check(
                        current_time=tick.time, spread=spread, atr=current_atr_val
                    )
                    if not filter_res.can_trade:
                        continue

                if self.risk_guard is not None:
                    guard_res = self.risk_guard.check_guards(tick.time, curr_account)
                    if not guard_res.can_trade:
                        self._cancel_pending_on_halt()
                        continue

                self.broker.submit_order(order)

            newly_closed = self.broker.process_tick(tick, atr=current_atr_val)

            if newly_closed:
                for closed_trade in newly_closed:
                    if self.risk_guard is not None:
                        post_trade_acc = self._account_state()
                        self.risk_guard.update_account_state(closed_trade.exit_time, post_trade_acc)
                        self.risk_guard.register_closed_trade(
                            closed_trade.net_pnl, closed_trade.exit_time
                        )
                        guard_res = self.risk_guard.check_guards(
                            closed_trade.exit_time, post_trade_acc
                        )
                        if not guard_res.can_trade:
                            self._cancel_pending_on_halt()

                    equity_curve.append(self._equity_point(closed_trade.exit_time))

            daily_close_equity[tick.time.date()] = self.broker.equity

        if self.close_positions_at_end and self.broker.active_positions:
            end_closed = self.broker.close_all_positions(
                time=ticks[-1].time,
                exit_reason=TradeExitReason.END_OF_DATA,
            )
            for closed_trade in end_closed:
                if self.risk_guard is not None:
                    post_trade_acc = self._account_state()
                    self.risk_guard.update_account_state(closed_trade.exit_time, post_trade_acc)
                    self.risk_guard.register_closed_trade(
                        closed_trade.net_pnl, closed_trade.exit_time
                    )
                equity_curve.append(self._equity_point(closed_trade.exit_time))
            daily_close_equity[ticks[-1].time.date()] = self.broker.equity

        if equity_curve[-1].timestamp != ticks[-1].time:
            equity_curve.append(self._equity_point(ticks[-1].time))

        trades = tuple(self.broker.closed_trades)
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in trades if t.net_pnl < 0)
        breakeven_trades = sum(1 for t in trades if t.net_pnl == 0)
        win_rate = round(winning_trades / total_trades, 4) if total_trades > 0 else 0.0
        total_net_pnl = round(sum(t.net_pnl for t in trades), 2)

        net_profits = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        net_losses = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        if net_losses > 0:
            profit_factor = round(net_profits / net_losses, 4)
        elif net_profits > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        daily_vals = list(daily_close_equity.values())
        if len(daily_vals) >= 2:
            daily_returns = [
                (daily_vals[i] - daily_vals[i - 1]) / daily_vals[i - 1]
                for i in range(1, len(daily_vals))
                if daily_vals[i - 1] > 0
            ]
            if len(daily_returns) >= 2:
                mean_ret = sum(daily_returns) / len(daily_returns)
                var_ret = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
                std_ret = math.sqrt(var_ret)
                sharpe_ratio = (
                    round((mean_ret / std_ret) * math.sqrt(252), 4) if std_ret > 0 else 0.0
                )
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        expectancy_r = (
            round(sum(t.return_r for t in trades) / total_trades, 4) if total_trades > 0 else 0.0
        )
        average_trade_duration = (
            round(sum(t.duration_seconds for t in trades) / total_trades, 1)
            if total_trades > 0
            else 0.0
        )

        return BacktestResult(
            trades=trades,
            equity_curve=tuple(equity_curve),
            initial_balance=self.config.initial_balance,
            final_balance=round(self.broker.balance, 2),
            total_net_pnl=total_net_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown_usd=round(self.broker.max_drawdown_usd, 2),
            max_drawdown_pct=round(self.broker.max_drawdown_pct, 4),
            sharpe_ratio=sharpe_ratio,
            expectancy_r=expectancy_r,
            average_trade_duration_seconds=average_trade_duration,
        )
