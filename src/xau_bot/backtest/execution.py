from dataclasses import dataclass, replace
from datetime import datetime

from xau_bot.backtest.models import (
    BacktestConfig,
    BacktestOrder,
    BacktestTrade,
    OrderStatus,
    OrderType,
    TradeExitReason,
)
from xau_bot.backtest.slippage import FixedSlippage, SlippageModel, ZeroSlippage
from xau_bot.domain.models import Direction, Swing, Tick
from xau_bot.risk.margin import calculate_margin
from xau_bot.risk.models import OpenPosition, PositionRiskConfig
from xau_bot.risk.protection import check_breakeven, check_trailing_stop


@dataclass(slots=True)
class _ActiveTradeState:
    ticket: int
    order: BacktestOrder
    entry_time: datetime
    entry_price: float
    volume: float
    direction: Direction
    current_sl: float
    current_tp: float
    initial_sl: float
    initial_risk_price: float
    initial_risk_usd: float
    mfe_price: float
    mae_price: float
    be_applied: bool = False
    trailing_applied: bool = False


class SimulatedBroker:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        slippage_model: SlippageModel | None = None,
        risk_config: PositionRiskConfig | None = None,
    ) -> None:
        self.config: BacktestConfig = config if config is not None else BacktestConfig()
        if slippage_model is not None:
            self.slippage_model: SlippageModel = slippage_model
        elif self.config.slippage_points > 0:
            self.slippage_model = FixedSlippage(
                points=self.config.slippage_points,
                point_size=self.config.symbol_spec.point,
            )
        else:
            self.slippage_model = ZeroSlippage()
        self.risk_config: PositionRiskConfig = (
            risk_config if risk_config is not None else PositionRiskConfig()
        )

        self.balance: float = self.config.initial_balance
        self.equity: float = self.config.initial_balance
        self.peak_equity: float = self.config.initial_balance
        self.max_drawdown_usd: float = 0.0
        self.max_drawdown_pct: float = 0.0

        self.pending_orders: list[BacktestOrder] = []
        self.filled_orders: list[BacktestOrder] = []
        self.expired_orders: list[BacktestOrder] = []
        self.cancelled_orders: list[BacktestOrder] = []
        self.rejected_orders: list[BacktestOrder] = []
        self.active_positions: list[_ActiveTradeState] = []
        self.closed_trades: list[BacktestTrade] = []
        self._next_ticket: int = 1
        self._last_tick: Tick | None = None

    @property
    def used_margin(self) -> float:
        margin_sum = 0.0
        for pos in self.active_positions:
            margin_sum += calculate_margin(
                lots=pos.volume,
                price=pos.entry_price,
                contract_size=self.config.symbol_spec.contract_size,
                leverage=self.config.leverage,
            )
        return margin_sum

    @property
    def free_margin(self) -> float:
        return max(0.0, self.equity - self.used_margin)

    def submit_order(self, order: BacktestOrder) -> bool:
        if len(self.active_positions) >= self.config.max_open_positions:
            rejected = replace(
                order,
                status=OrderStatus.REJECTED,
                cancellation_reason="Max open positions reached",
            )
            self.rejected_orders.append(rejected)
            return False

        req_margin = calculate_margin(
            lots=order.volume,
            price=order.limit_price,
            contract_size=self.config.symbol_spec.contract_size,
            leverage=self.config.leverage,
        )
        if req_margin > self.free_margin:
            rejected = replace(
                order,
                status=OrderStatus.REJECTED,
                cancellation_reason=(
                    f"Insufficient free margin: {req_margin:.2f} > {self.free_margin:.2f}"
                ),
            )
            self.rejected_orders.append(rejected)
            return False

        self.pending_orders.append(order)
        return True

    def cancel_order(self, order_id: str, reason: str = "CANCELLED") -> bool:
        for idx, o in enumerate(self.pending_orders):
            if o.order_id == order_id:
                cancelled = replace(
                    o,
                    status=OrderStatus.CANCELLED,
                    cancellation_reason=reason,
                )
                self.pending_orders.pop(idx)
                self.cancelled_orders.append(cancelled)
                return True
        return False

    def process_tick(
        self,
        tick: Tick,
        atr: float | None = None,
        structural_swing: Swing | None = None,
    ) -> list[BacktestTrade]:
        if self._last_tick is not None and tick.time < self._last_tick.time:
            raise ValueError(f"Tick timestamp went backwards: {tick.time} < {self._last_tick.time}")
        self._last_tick = tick
        spread = tick.ask - tick.bid

        newly_closed: list[BacktestTrade] = []

        remaining_pending: list[BacktestOrder] = []
        for order in self.pending_orders:
            if order.expires_at is not None and tick.time >= order.expires_at:
                expired = replace(order, status=OrderStatus.EXPIRED)
                self.expired_orders.append(expired)
                continue

            if len(self.active_positions) >= self.config.max_open_positions:
                remaining_pending.append(order)
                continue

            filled = False
            fill_price = 0.0

            if order.order_type == OrderType.BUY_LIMIT:
                condition = (
                    tick.ask <= order.limit_price
                    if self.config.fill_limit_on_touch
                    else tick.ask < order.limit_price
                )
                if condition:
                    filled = True
                    fill_price = min(order.limit_price, tick.ask)
            elif order.order_type == OrderType.SELL_LIMIT:
                condition = (
                    tick.bid >= order.limit_price
                    if self.config.fill_limit_on_touch
                    else tick.bid > order.limit_price
                )
                if condition:
                    filled = True
                    fill_price = max(order.limit_price, tick.bid)
            elif order.order_type == OrderType.BUY_MARKET:
                slip = self.slippage_model.calculate_slippage(
                    price=tick.ask,
                    direction=Direction.BULLISH,
                    volume=order.volume,
                    spread=spread,
                    is_stop_or_market=True,
                )
                filled = True
                fill_price = tick.ask + slip
            elif order.order_type == OrderType.SELL_MARKET:
                slip = self.slippage_model.calculate_slippage(
                    price=tick.bid,
                    direction=Direction.BEARISH,
                    volume=order.volume,
                    spread=spread,
                    is_stop_or_market=True,
                )
                filled = True
                fill_price = tick.bid - slip

            if filled:
                req_margin = calculate_margin(
                    lots=order.volume,
                    price=fill_price,
                    contract_size=self.config.symbol_spec.contract_size,
                    leverage=self.config.leverage,
                )
                if req_margin > self.free_margin:
                    remaining_pending.append(order)
                    continue

                init_risk_dist = abs(fill_price - order.stop_loss)
                init_risk_usd = (
                    init_risk_dist
                    / self.config.symbol_spec.tick_size
                    * self.config.symbol_spec.tick_value
                    * order.volume
                )

                ticket = self._next_ticket
                self._next_ticket += 1

                pos_state = _ActiveTradeState(
                    ticket=ticket,
                    order=order,
                    entry_time=tick.time,
                    entry_price=fill_price,
                    volume=order.volume,
                    direction=order.direction,
                    current_sl=order.stop_loss,
                    current_tp=order.take_profit,
                    initial_sl=order.stop_loss,
                    initial_risk_price=max(1e-9, init_risk_dist),
                    initial_risk_usd=max(1e-9, init_risk_usd),
                    mfe_price=fill_price,
                    mae_price=fill_price,
                )
                self.active_positions.append(pos_state)

                filled_order = replace(
                    order,
                    status=OrderStatus.FILLED,
                    filled_at=tick.time,
                    fill_price=fill_price,
                )
                self.filled_orders.append(filled_order)
            else:
                remaining_pending.append(order)

        self.pending_orders = remaining_pending

        remaining_positions: list[_ActiveTradeState] = []
        for pos in self.active_positions:
            is_closed = False
            exit_price = 0.0
            exit_reason = TradeExitReason.STOP_LOSS

            if pos.direction == Direction.BULLISH:
                pos.mfe_price = max(pos.mfe_price, tick.bid)
                pos.mae_price = min(pos.mae_price, tick.bid)

                if tick.bid <= pos.current_sl:
                    slip = self.slippage_model.calculate_slippage(
                        price=tick.bid,
                        direction=pos.direction,
                        volume=pos.volume,
                        spread=spread,
                        is_stop_or_market=True,
                    )
                    is_closed = True
                    exit_price = tick.bid - slip
                    if pos.trailing_applied:
                        exit_reason = TradeExitReason.TRAILING_STOP
                    elif pos.be_applied:
                        exit_reason = TradeExitReason.BREAKEVEN
                    else:
                        exit_reason = TradeExitReason.STOP_LOSS
                elif tick.bid >= pos.current_tp:
                    is_closed = True
                    exit_price = pos.current_tp
                    exit_reason = TradeExitReason.TAKE_PROFIT
                else:
                    self._check_position_modifications(
                        pos=pos,
                        current_price=tick.bid,
                        spread=spread,
                        atr=atr,
                        structural_swing=structural_swing,
                    )

            else:
                pos.mfe_price = min(pos.mfe_price, tick.ask)
                pos.mae_price = max(pos.mae_price, tick.ask)

                if tick.ask >= pos.current_sl:
                    slip = self.slippage_model.calculate_slippage(
                        price=tick.ask,
                        direction=pos.direction,
                        volume=pos.volume,
                        spread=spread,
                        is_stop_or_market=True,
                    )
                    is_closed = True
                    exit_price = tick.ask + slip
                    if pos.trailing_applied:
                        exit_reason = TradeExitReason.TRAILING_STOP
                    elif pos.be_applied:
                        exit_reason = TradeExitReason.BREAKEVEN
                    else:
                        exit_reason = TradeExitReason.STOP_LOSS
                elif tick.ask <= pos.current_tp:
                    is_closed = True
                    exit_price = pos.current_tp
                    exit_reason = TradeExitReason.TAKE_PROFIT
                else:
                    self._check_position_modifications(
                        pos=pos,
                        current_price=tick.ask,
                        spread=spread,
                        atr=atr,
                        structural_swing=structural_swing,
                    )

            if is_closed:
                trade = self._create_closed_trade(pos, tick.time, exit_price, exit_reason)
                self.balance += trade.net_pnl
                self.closed_trades.append(trade)
                newly_closed.append(trade)
            else:
                remaining_positions.append(pos)

        self.active_positions = remaining_positions

        unrealized = self._calculate_unrealized_pnl(tick)
        self.equity = self.balance + unrealized
        self._update_drawdown()

        return newly_closed

    def _update_drawdown(self) -> None:
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd_usd = max(0.0, self.peak_equity - self.equity)
        dd_pct = (dd_usd / self.peak_equity) * 100.0 if self.peak_equity > 0 else 0.0

        if dd_usd > self.max_drawdown_usd:
            self.max_drawdown_usd = dd_usd
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

    def _check_position_modifications(
        self,
        pos: _ActiveTradeState,
        current_price: float,
        spread: float,
        atr: float | None = None,
        structural_swing: Swing | None = None,
    ) -> None:
        open_pos = OpenPosition(
            ticket=pos.ticket,
            symbol=self.config.symbol_spec.symbol,
            direction=pos.direction,
            volume=pos.volume,
            entry_price=pos.entry_price,
            current_stop_loss=pos.current_sl,
            current_take_profit=pos.current_tp,
            initial_stop_loss=pos.initial_sl,
            initial_take_profit=pos.order.take_profit,
            open_time=pos.entry_time,
        )

        be_res = check_breakeven(
            position=open_pos,
            current_price=current_price,
            spec=self.config.symbol_spec,
            config=self.risk_config,
            spread=spread,
        )
        if be_res.should_modify and be_res.new_stop_loss is not None:
            pos.current_sl = be_res.new_stop_loss
            pos.be_applied = True
            open_pos = OpenPosition(
                ticket=pos.ticket,
                symbol=self.config.symbol_spec.symbol,
                direction=pos.direction,
                volume=pos.volume,
                entry_price=pos.entry_price,
                current_stop_loss=pos.current_sl,
                current_take_profit=pos.current_tp,
                initial_stop_loss=pos.initial_sl,
                initial_take_profit=pos.order.take_profit,
                open_time=pos.entry_time,
            )

        if atr is not None or structural_swing is not None:
            ts_res = check_trailing_stop(
                position=open_pos,
                current_price=current_price,
                spec=self.config.symbol_spec,
                config=self.risk_config,
                atr=atr,
                structural_swing=structural_swing,
                spread=spread,
            )
            if ts_res.should_modify and ts_res.new_stop_loss is not None:
                pos.current_sl = ts_res.new_stop_loss
                pos.trailing_applied = True

    def _create_closed_trade(
        self,
        pos: _ActiveTradeState,
        exit_time: datetime,
        exit_price: float,
        exit_reason: TradeExitReason,
    ) -> BacktestTrade:
        spec = self.config.symbol_spec

        if pos.direction == Direction.BULLISH:
            price_delta = exit_price - pos.entry_price
            mfe_dist = max(0.0, pos.mfe_price - pos.entry_price)
            mae_dist = max(0.0, pos.entry_price - pos.mae_price)
        else:
            price_delta = pos.entry_price - exit_price
            mfe_dist = max(0.0, pos.entry_price - pos.mfe_price)
            mae_dist = max(0.0, pos.mae_price - pos.entry_price)

        gross_pnl = (price_delta / spec.tick_size) * spec.tick_value * pos.volume
        comm_rate = (
            self.config.commission_per_lot
            if self.config.commission_per_lot is not None
            else spec.commission_per_lot
        )
        commission = comm_rate * pos.volume
        swap = 0.0
        net_pnl = gross_pnl - commission - swap

        return_r = net_pnl / pos.initial_risk_usd if pos.initial_risk_usd > 0 else 0.0
        mfe_r = mfe_dist / pos.initial_risk_price if pos.initial_risk_price > 0 else 0.0
        mae_r = mae_dist / pos.initial_risk_price if pos.initial_risk_price > 0 else 0.0

        duration = max(0.0, (exit_time - pos.entry_time).total_seconds())

        trade = BacktestTrade(
            trade_id=f"T-{len(self.closed_trades) + 1}",
            order_id=pos.order.order_id,
            setup_id=pos.order.setup_id,
            direction=pos.direction,
            volume=pos.volume,
            entry_time=pos.entry_time,
            entry_price=round(pos.entry_price, 4),
            exit_time=exit_time,
            exit_price=round(exit_price, 4),
            exit_reason=exit_reason,
            initial_sl=round(pos.initial_sl, 4),
            final_sl=round(pos.current_sl, 4),
            tp=round(pos.current_tp, 4),
            commission=round(commission, 2),
            swap=round(swap, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            return_r=round(return_r, 4),
            mfe_price=round(pos.mfe_price, 4),
            mae_price=round(pos.mae_price, 4),
            mfe_r=round(mfe_r, 4),
            mae_r=round(mae_r, 4),
            duration_seconds=round(duration, 1),
        )
        return trade

    def _calculate_unrealized_pnl(self, tick: Tick) -> float:
        spec = self.config.symbol_spec
        total_unrealized = 0.0
        for pos in self.active_positions:
            if pos.direction == Direction.BULLISH:
                delta = tick.bid - pos.entry_price
            else:
                delta = pos.entry_price - tick.ask
            total_unrealized += (delta / spec.tick_size) * spec.tick_value * pos.volume
        return total_unrealized

    def close_all_positions(
        self,
        time: datetime,
        exit_reason: TradeExitReason = TradeExitReason.END_OF_DATA,
    ) -> list[BacktestTrade]:
        if not self.active_positions or self._last_tick is None:
            return []

        closed: list[BacktestTrade] = []
        spread = self._last_tick.ask - self._last_tick.bid
        for pos in self.active_positions:
            if pos.direction == Direction.BULLISH:
                slip = self.slippage_model.calculate_slippage(
                    price=self._last_tick.bid,
                    direction=pos.direction,
                    volume=pos.volume,
                    spread=spread,
                    is_stop_or_market=True,
                )
                exit_price = self._last_tick.bid - slip
            else:
                slip = self.slippage_model.calculate_slippage(
                    price=self._last_tick.ask,
                    direction=pos.direction,
                    volume=pos.volume,
                    spread=spread,
                    is_stop_or_market=True,
                )
                exit_price = self._last_tick.ask + slip

            trade = self._create_closed_trade(pos, time, exit_price, exit_reason)
            self.balance += trade.net_pnl
            self.closed_trades.append(trade)
            closed.append(trade)

        self.active_positions.clear()
        self.equity = self.balance
        self._update_drawdown()
        return closed
