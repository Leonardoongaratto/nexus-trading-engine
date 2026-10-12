import math
from datetime import UTC, datetime, timedelta

import pytest

from xau_bot.backtest.engine import BacktestEngine
from xau_bot.backtest.execution import SimulatedBroker
from xau_bot.backtest.models import (
    BacktestConfig,
    BacktestOrder,
    BacktestResult,
    BacktestTrade,
    EquityPoint,
    OrderStatus,
    OrderType,
    TradeExitReason,
)
from xau_bot.backtest.slippage import (
    FixedSlippage,
    NormalSlippage,
    ZeroSlippage,
)
from xau_bot.domain.models import Candle, Direction, Swing, Tick, Timeframe, TradeSetup
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.sessions import SessionFilter
from xau_bot.risk.guards import RiskGuard
from xau_bot.risk.models import (
    AccountState,
    DailyRiskLimits,
    PositionRiskConfig,
)


def _make_utc(
    hour: int = 10,
    minute: int = 0,
    second: int = 0,
    day: int = 24,
    month: int = 9,
    year: int = 2026,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def _make_tick(
    bid: float,
    ask: float,
    time: datetime,
) -> Tick:
    return Tick(time=time, bid=bid, ask=ask)


class TestBacktestModels:
    def test_backtest_order_validations(self) -> None:
        t0 = _make_utc(10, 0)
        t_exp = _make_utc(10, 30)

        order = BacktestOrder(
            order_id="ORD-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.0,
            stop_loss=2590.0,
            take_profit=2620.0,
            created_at=t0,
            expires_at=t_exp,
        )
        assert order.order_id == "ORD-1"
        assert order.status == OrderStatus.PENDING

        with pytest.raises(ValueError, match="order_id must not be empty"):
            BacktestOrder(
                order_id="",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="volume must be positive"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.0,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="limit_price must be positive"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=-10.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="created_at must be timezone-aware"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=datetime(2026, 9, 24, 10, 0),
            )

        with pytest.raises(ValueError, match="expires_at must be timezone-aware"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
                expires_at=datetime(2026, 9, 24, 11, 0),
            )

        with pytest.raises(ValueError, match="expires_at must be >= created_at"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
                expires_at=_make_utc(9, 0),
            )

        with pytest.raises(ValueError, match="filled_at must be timezone-aware"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
                filled_at=datetime(2026, 9, 24, 10, 5),
            )

        with pytest.raises(ValueError, match="fill_price must be positive"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2620.0,
                created_at=t0,
                fill_price=-5.0,
            )

        with pytest.raises(ValueError, match="Buy order stop_loss.*must be < limit_price"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2605.0,
                take_profit=2620.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="Buy order take_profit.*must be > limit_price"):
            BacktestOrder(
                order_id="ORD-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2595.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="Sell order stop_loss.*must be > limit_price"):
            BacktestOrder(
                order_id="ORD-2",
                direction=Direction.BEARISH,
                order_type=OrderType.SELL_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2590.0,
                take_profit=2580.0,
                created_at=t0,
            )

        with pytest.raises(ValueError, match="Sell order take_profit.*must be < limit_price"):
            BacktestOrder(
                order_id="ORD-2",
                direction=Direction.BEARISH,
                order_type=OrderType.SELL_LIMIT,
                volume=0.1,
                limit_price=2600.0,
                stop_loss=2610.0,
                take_profit=2615.0,
                created_at=t0,
            )

    def test_backtest_trade_validations(self) -> None:
        t0 = _make_utc(10, 0)
        t1 = _make_utc(10, 15)

        trade = BacktestTrade(
            trade_id="TR-1",
            order_id="ORD-1",
            direction=Direction.BULLISH,
            volume=0.1,
            entry_time=t0,
            entry_price=2600.0,
            exit_time=t1,
            exit_price=2610.0,
            exit_reason=TradeExitReason.TAKE_PROFIT,
            initial_sl=2595.0,
            final_sl=2600.0,
            tp=2610.0,
            commission=0.7,
            swap=0.0,
            gross_pnl=100.0,
            net_pnl=99.3,
            return_r=2.0,
            mfe_price=2612.0,
            mae_price=2599.0,
            mfe_r=2.4,
            mae_r=0.2,
            duration_seconds=900.0,
        )
        assert trade.net_pnl == 99.3

        with pytest.raises(ValueError, match="trade_id must not be empty"):
            BacktestTrade(
                trade_id="",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=t0,
                entry_price=2600.0,
                exit_time=t1,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="order_id must not be empty"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=t0,
                entry_price=2600.0,
                exit_time=t1,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="volume must be positive"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=-0.1,
                entry_time=t0,
                entry_price=2600.0,
                exit_time=t1,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="entry_time must be timezone-aware"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=datetime(2026, 9, 24, 10, 0),
                entry_price=2600.0,
                exit_time=t1,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="exit_time must be timezone-aware"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=t0,
                entry_price=2600.0,
                exit_time=datetime(2026, 9, 24, 10, 15),
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="exit_time must be >= entry_time"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=t1,
                entry_price=2600.0,
                exit_time=t0,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=100.0,
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

        with pytest.raises(ValueError, match="gross_pnl must be finite"):
            BacktestTrade(
                trade_id="TR-1",
                order_id="ORD-1",
                direction=Direction.BULLISH,
                volume=0.1,
                entry_time=t0,
                entry_price=2600.0,
                exit_time=t1,
                exit_price=2610.0,
                exit_reason=TradeExitReason.TAKE_PROFIT,
                initial_sl=2595.0,
                final_sl=2600.0,
                tp=2610.0,
                commission=0.7,
                swap=0.0,
                gross_pnl=float("nan"),
                net_pnl=99.3,
                return_r=2.0,
                mfe_price=2612.0,
                mae_price=2599.0,
                mfe_r=2.4,
                mae_r=0.2,
                duration_seconds=900.0,
            )

    def test_equity_point_and_config_validations(self) -> None:
        t0 = _make_utc(10, 0)
        ep = EquityPoint(
            timestamp=t0,
            balance=10000.0,
            equity=10000.0,
            drawdown_usd=0.0,
            drawdown_pct=0.0,
        )
        assert ep.balance == 10000.0

        with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
            EquityPoint(
                timestamp=datetime(2026, 9, 24, 10, 0),
                balance=10000.0,
                equity=10000.0,
                drawdown_usd=0.0,
                drawdown_pct=0.0,
            )

        with pytest.raises(ValueError, match="balance must be finite"):
            EquityPoint(
                timestamp=t0,
                balance=float("nan"),
                equity=10000.0,
                drawdown_usd=0.0,
                drawdown_pct=0.0,
            )

        cfg = BacktestConfig(initial_balance=5000.0)
        assert cfg.initial_balance == 5000.0

        with pytest.raises(ValueError, match="initial_balance must be positive"):
            BacktestConfig(initial_balance=-10.0)

        with pytest.raises(ValueError, match="leverage must be >= 1.0"):
            BacktestConfig(leverage=0.5)

        with pytest.raises(ValueError, match="commission_per_lot cannot be negative"):
            BacktestConfig(commission_per_lot=-1.0)

        with pytest.raises(ValueError, match="slippage_points cannot be negative"):
            BacktestConfig(slippage_points=-1.0)

        with pytest.raises(ValueError, match="max_open_positions must be at least 1"):
            BacktestConfig(max_open_positions=0)


class TestSlippageModels:
    def test_zero_slippage(self) -> None:
        zs = ZeroSlippage()
        assert zs.calculate_slippage(2600.0, Direction.BULLISH, 1.0, 0.25) == 0.0

    def test_fixed_slippage(self) -> None:
        fs = FixedSlippage(points=2.0, point_size=0.01)
        assert (
            fs.calculate_slippage(2600.0, Direction.BULLISH, 1.0, 0.25, is_stop_or_market=True)
            == 0.02
        )
        assert (
            fs.calculate_slippage(2600.0, Direction.BULLISH, 1.0, 0.25, is_stop_or_market=False)
            == 0.0
        )

        with pytest.raises(ValueError, match="points must be non-negative"):
            FixedSlippage(points=-1.0)

        with pytest.raises(ValueError, match="point_size must be positive"):
            FixedSlippage(point_size=0.0)

    def test_normal_slippage(self) -> None:
        ns = NormalSlippage(mean_points=1.0, std_points=0.2, point_size=0.01, seed=42)
        slip1 = ns.calculate_slippage(2600.0, Direction.BULLISH, 1.0, 0.25, is_stop_or_market=True)
        assert slip1 >= 0.0
        assert (
            ns.calculate_slippage(2600.0, Direction.BULLISH, 1.0, 0.25, is_stop_or_market=False)
            == 0.0
        )

        with pytest.raises(ValueError, match="mean_points must be non-negative"):
            NormalSlippage(mean_points=-1.0)

        with pytest.raises(ValueError, match="std_points must be non-negative"):
            NormalSlippage(std_points=-0.1)

        with pytest.raises(ValueError, match="point_size must be positive"):
            NormalSlippage(point_size=0.0)


class TestSimulatedBrokerMatching:
    def test_ac_1101_limit_order_matching_and_touch_mode(self) -> None:
        t0 = _make_utc(10, 0)
        t1 = _make_utc(10, 1)
        t2 = _make_utc(10, 2)

        broker = SimulatedBroker(config=BacktestConfig(fill_limit_on_touch=True))
        order_buy = BacktestOrder(
            order_id="BUY-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        assert broker.submit_order(order_buy) is True

        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=t1))
        assert len(broker.active_positions) == 1
        assert broker.active_positions[0].entry_price == 2600.00
        assert len(broker.filled_orders) == 1
        assert broker.filled_orders[0].fill_price == 2600.00

        broker_no_touch = SimulatedBroker(config=BacktestConfig(fill_limit_on_touch=False))
        order_buy2 = BacktestOrder(
            order_id="BUY-2",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker_no_touch.submit_order(order_buy2)

        broker_no_touch.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=t1))
        assert len(broker_no_touch.active_positions) == 0

        broker_no_touch.process_tick(_make_tick(bid=2599.65, ask=2599.90, time=t2))
        assert len(broker_no_touch.active_positions) == 1
        assert broker_no_touch.active_positions[0].entry_price == 2599.90

        broker_sell = SimulatedBroker()
        order_sell = BacktestOrder(
            order_id="SELL-1",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2650.00,
            stop_loss=2660.00,
            take_profit=2630.00,
            created_at=t0,
        )
        broker_sell.submit_order(order_sell)
        broker_sell.process_tick(_make_tick(bid=2650.10, ask=2650.35, time=t1))
        assert len(broker_sell.active_positions) == 1
        assert broker_sell.active_positions[0].entry_price == 2650.10

    def test_ac_1102_order_expiration(self) -> None:
        t0 = _make_utc(10, 0)
        t_exp = _make_utc(10, 15)
        t_past = _make_utc(10, 16)

        broker = SimulatedBroker()
        order = BacktestOrder(
            order_id="EXP-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
            expires_at=t_exp,
        )
        broker.submit_order(order)

        broker.process_tick(_make_tick(bid=2605.00, ask=2605.25, time=_make_utc(10, 5)))
        assert len(broker.pending_orders) == 1
        assert len(broker.expired_orders) == 0

        broker.process_tick(_make_tick(bid=2605.00, ask=2605.25, time=t_past))
        assert len(broker.pending_orders) == 0
        assert len(broker.expired_orders) == 1
        assert broker.expired_orders[0].status == OrderStatus.EXPIRED
        assert len(broker.active_positions) == 0
        assert len(broker.closed_trades) == 0

    def test_ac_1103_ac_1105_long_sl_and_tp(self) -> None:
        t0 = _make_utc(10, 0)
        t1 = _make_utc(10, 1)
        t2 = _make_utc(10, 2)

        broker_sl = SimulatedBroker(slippage_model=FixedSlippage(points=5.0, point_size=0.01))
        order_buy = BacktestOrder(
            order_id="BUY-SL",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker_sl.submit_order(order_buy)
        broker_sl.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=t1))
        assert len(broker_sl.active_positions) == 1

        exit_tick_bid = 2589.50
        slippage_amount = 0.05
        expected_sl_exit_price = exit_tick_bid - slippage_amount
        closed = broker_sl.process_tick(_make_tick(bid=exit_tick_bid, ask=2589.75, time=t2))
        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == TradeExitReason.STOP_LOSS
        assert trade.exit_price == expected_sl_exit_price
        assert trade.net_pnl < 0

        broker_tp = SimulatedBroker()
        order_tp = BacktestOrder(
            order_id="BUY-TP",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker_tp.submit_order(order_tp)
        broker_tp.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=t1))

        closed_tp = broker_tp.process_tick(_make_tick(bid=2620.50, ask=2620.75, time=t2))
        assert len(closed_tp) == 1
        assert closed_tp[0].exit_reason == TradeExitReason.TAKE_PROFIT
        assert closed_tp[0].exit_price == 2620.00
        assert closed_tp[0].net_pnl > 0

    def test_ac_1104_ac_1106_short_sl_and_tp(self) -> None:
        t0 = _make_utc(10, 0)
        t1 = _make_utc(10, 1)
        t2 = _make_utc(10, 2)

        broker_sl = SimulatedBroker(slippage_model=FixedSlippage(points=5.0, point_size=0.01))
        order_sell = BacktestOrder(
            order_id="SELL-SL",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2580.00,
            created_at=t0,
        )
        broker_sl.submit_order(order_sell)
        broker_sl.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=t1))
        assert len(broker_sl.active_positions) == 1

        exit_tick_ask = 2610.50
        slippage_amount = 0.05
        expected_sl_exit_price = exit_tick_ask + slippage_amount
        closed = broker_sl.process_tick(_make_tick(bid=2610.25, ask=exit_tick_ask, time=t2))
        assert len(closed) == 1
        trade = closed[0]
        assert trade.exit_reason == TradeExitReason.STOP_LOSS
        assert trade.exit_price == expected_sl_exit_price
        assert trade.net_pnl < 0

        broker_tp = SimulatedBroker()
        order_tp = BacktestOrder(
            order_id="SELL-TP",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2580.00,
            created_at=t0,
        )
        broker_tp.submit_order(order_tp)
        broker_tp.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=t1))

        closed_tp = broker_tp.process_tick(_make_tick(bid=2579.50, ask=2579.75, time=t2))
        assert len(closed_tp) == 1
        assert closed_tp[0].exit_reason == TradeExitReason.TAKE_PROFIT
        assert closed_tp[0].exit_price == 2580.00
        assert closed_tp[0].net_pnl > 0

    def test_market_orders_and_slippage(self) -> None:
        t0 = _make_utc(10, 0)
        t1 = _make_utc(10, 1)
        broker = SimulatedBroker(slippage_model=FixedSlippage(points=10.0, point_size=0.01))

        order_buy_mkt = BacktestOrder(
            order_id="MKT-B",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(order_buy_mkt)
        expected_buy_fill = 2600.25 + 0.10
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=t1))
        assert len(broker.active_positions) == 1
        assert broker.active_positions[0].entry_price == expected_buy_fill

        broker.close_all_positions(time=t1)
        assert len(broker.active_positions) == 0

        order_sell_mkt = BacktestOrder(
            order_id="MKT-S",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2580.00,
            created_at=t0,
        )
        broker.submit_order(order_sell_mkt)
        expected_sell_fill = 2600.00 - 0.10
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=_make_utc(10, 2)))
        assert len(broker.active_positions) == 1
        assert broker.active_positions[0].entry_price == expected_sell_fill

    def test_ac_1108_costs_and_pnl_accounting(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(commission_per_lot=7.0))
        order = BacktestOrder(
            order_id="FEE-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=1.0,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2610.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))

        expected_gross_pnl = 1000.00
        expected_commission = 7.00
        expected_net_pnl = expected_gross_pnl - expected_commission
        closed = broker.process_tick(_make_tick(bid=2610.00, ask=2610.25, time=_make_utc(10, 2)))
        assert len(closed) == 1
        trade = closed[0]
        assert trade.gross_pnl == expected_gross_pnl
        assert trade.commission == expected_commission
        assert trade.net_pnl == expected_net_pnl
        assert broker.balance == 10000.0 + expected_net_pnl

    def test_ac_1109_protection_management_be_and_trailing(self) -> None:
        t0 = _make_utc(10, 0)
        risk_cfg = PositionRiskConfig(
            be_trigger_r=1.0,
            be_offset_points=10.0,
            trailing_atr_multiplier=1.0,
            min_trailing_step_points=5.0,
        )
        broker = SimulatedBroker(risk_config=risk_cfg)
        order = BacktestOrder(
            order_id="PROT-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2630.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))
        pos = broker.active_positions[0]
        assert pos.current_sl == 2590.00

        broker.process_tick(_make_tick(bid=2611.00, ask=2611.25, time=_make_utc(10, 2)))
        assert pos.be_applied is True
        assert pos.current_sl > 2600.00

        broker.process_tick(
            _make_tick(bid=2620.00, ask=2620.25, time=_make_utc(10, 3)),
            atr=2.0,
        )
        assert pos.trailing_applied is True
        assert pos.current_sl == 2618.00

        closed = broker.process_tick(_make_tick(bid=2617.90, ask=2618.15, time=_make_utc(10, 4)))
        assert len(closed) == 1
        assert closed[0].exit_reason == TradeExitReason.TRAILING_STOP
        assert closed[0].net_pnl > 0

    def test_broker_order_management_edge_cases(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(max_open_positions=1))

        order1 = BacktestOrder(
            order_id="CANCEL-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(order1)
        assert broker.cancel_order("NON-EXISTENT") is False
        assert broker.cancel_order("CANCEL-1", "User cancelled") is True
        assert len(broker.cancelled_orders) == 1
        assert broker.cancelled_orders[0].cancellation_reason == "User cancelled"

        huge_order = BacktestOrder(
            order_id="HUGE-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=100.0,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        assert broker.submit_order(huge_order) is False
        assert broker.rejected_orders[0].cancellation_reason is not None
        assert "Insufficient free margin" in broker.rejected_orders[0].cancellation_reason

        fill_order = BacktestOrder(
            order_id="FILL-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(fill_order)
        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))
        assert len(broker.active_positions) == 1

        second_order = BacktestOrder(
            order_id="SECOND-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        assert broker.submit_order(second_order) is False
        assert broker.rejected_orders[-1].cancellation_reason == "Max open positions reached"

        with pytest.raises(ValueError, match="Tick timestamp went backwards"):
            broker.process_tick(_make_tick(bid=2600.0, ask=2600.25, time=_make_utc(9, 59)))


class TestBacktestEngine:
    def test_ac_1110_metrics_and_equity_curve(self) -> None:
        t0 = _make_utc(10, 0)
        ticks = [
            _make_tick(bid=2600.0, ask=2600.25, time=t0),
            _make_tick(bid=2599.75, ask=2600.00, time=t0 + timedelta(seconds=10)),
            _make_tick(bid=2610.00, ask=2610.25, time=t0 + timedelta(seconds=30)),
            _make_tick(bid=2609.00, ask=2609.25, time=t0 + timedelta(seconds=60)),
        ]
        order = BacktestOrder(
            order_id="ORD-METRICS",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2595.00,
            take_profit=2610.00,
            created_at=t0,
        )

        engine = BacktestEngine()
        result = engine.run(ticks=ticks, orders=[order])

        assert isinstance(result, BacktestResult)
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.losing_trades == 0
        assert result.win_rate == 1.0
        assert math.isinf(result.profit_factor)
        assert result.expectancy_r > 0
        assert result.average_trade_duration_seconds == 20.0
        assert len(result.equity_curve) >= 2
        assert result.final_balance > result.initial_balance

    def test_engine_validations_and_empty_ticks(self) -> None:
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="ticks sequence cannot be empty"):
            engine.run(ticks=[])

        t0 = _make_utc(10, 0)
        with pytest.raises(ValueError, match="Ticks must be in non-decreasing chronological order"):
            engine.run(
                ticks=[
                    _make_tick(bid=2600.0, ask=2600.25, time=t0),
                    _make_tick(bid=2600.0, ask=2600.25, time=t0 - timedelta(seconds=1)),
                ]
            )

    def test_engine_with_trade_setup_integration(self) -> None:
        t0 = _make_utc(10, 0)
        setup = TradeSetup(
            id="SMC-SETUP-1",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t0,
            confirmed_time=t0,
            risk_reward_ratio=2.0,
            valid_until=t0 + timedelta(minutes=15),
        )

        ticks = [
            _make_tick(bid=2601.00, ask=2601.25, time=t0),
            _make_tick(bid=2599.75, ask=2600.00, time=t0 + timedelta(minutes=5)),
            _make_tick(bid=2620.00, ask=2620.25, time=t0 + timedelta(minutes=10)),
        ]

        engine = BacktestEngine()
        result = engine.run(ticks=ticks, setups=[setup])

        assert result.total_trades == 1
        assert result.trades[0].setup_id == "SMC-SETUP-1"
        assert result.trades[0].exit_reason == TradeExitReason.TAKE_PROFIT

    def test_engine_filters_and_guards_integration(self) -> None:
        t_sunday = _make_utc(10, 0, day=27)
        trade_filter = CompositeTradeFilter(session_filter=SessionFilter())

        setup = TradeSetup(
            id="FILTERED-SETUP",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t_sunday,
            confirmed_time=t_sunday,
            risk_reward_ratio=2.0,
        )

        ticks = [
            _make_tick(bid=2600.00, ask=2600.25, time=t_sunday),
            _make_tick(bid=2599.50, ask=2600.00, time=t_sunday + timedelta(minutes=1)),
        ]

        engine = BacktestEngine(trade_filter=trade_filter)
        result = engine.run(ticks=ticks, setups=[setup])
        assert result.total_trades == 0

    def test_engine_risk_guard_daily_halt(self) -> None:
        t0 = _make_utc(10, 0)
        limits = DailyRiskLimits(max_consecutive_losses=1)
        guard = RiskGuard(limits=limits)

        order1 = BacktestOrder(
            order_id="ORD-LOSS",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2595.00,
            take_profit=2620.00,
            created_at=t0,
        )
        order2 = BacktestOrder(
            order_id="ORD-BLOCKED",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2590.00,
            stop_loss=2580.00,
            take_profit=2610.00,
            created_at=t0 + timedelta(minutes=5),
        )

        ticks = [
            _make_tick(bid=2600.00, ask=2600.00, time=t0),
            _make_tick(bid=2594.00, ask=2594.25, time=t0 + timedelta(minutes=1)),
            _make_tick(bid=2590.00, ask=2590.00, time=t0 + timedelta(minutes=6)),
        ]

        engine = BacktestEngine(risk_guard=guard)
        result = engine.run(ticks=ticks, orders=[order1, order2])

        assert result.total_trades == 1
        assert result.losing_trades == 1
        assert result.trades[0].order_id == "ORD-LOSS"

    def test_engine_close_positions_at_end(self) -> None:
        t0 = _make_utc(10, 0)
        order = BacktestOrder(
            order_id="ORD-END",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2580.00,
            take_profit=2630.00,
            created_at=t0,
        )
        ticks = [
            _make_tick(bid=2599.75, ask=2600.00, time=t0),
            _make_tick(bid=2605.00, ask=2605.25, time=t0 + timedelta(minutes=10)),
        ]

        engine = BacktestEngine(close_positions_at_end=True)
        result = engine.run(ticks=ticks, orders=[order])

        assert result.total_trades == 1
        assert result.trades[0].exit_reason == TradeExitReason.END_OF_DATA
        assert result.trades[0].exit_price == 2605.00
        assert result.trades[0].net_pnl > 0

    def test_engine_sharpe_with_multiple_trades(self) -> None:
        t0 = _make_utc(10, 0, day=20)
        t1 = _make_utc(10, 0, day=21)
        t2 = _make_utc(10, 0, day=22)
        orders = [
            BacktestOrder(
                order_id="WIN-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2600.00,
                stop_loss=2590.00,
                take_profit=2610.00,
                created_at=t0,
            ),
            BacktestOrder(
                order_id="LOSS-1",
                direction=Direction.BULLISH,
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                limit_price=2615.00,
                stop_loss=2610.00,
                take_profit=2630.00,
                created_at=t1,
            ),
        ]
        ticks = [
            _make_tick(bid=2599.75, ask=2600.00, time=t0),
            _make_tick(bid=2610.00, ask=2610.25, time=t0 + timedelta(minutes=2)),
            _make_tick(bid=2614.75, ask=2615.00, time=t1),
            _make_tick(bid=2609.50, ask=2609.75, time=t1 + timedelta(minutes=2)),
            _make_tick(bid=2610.00, ask=2610.25, time=t2),
        ]

        engine = BacktestEngine()
        result = engine.run(ticks=ticks, orders=orders)
        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.profit_factor > 0
        assert math.isclose(result.sharpe_ratio, -11.2250, abs_tol=1e-4)

    def test_engine_single_day_sharpe_is_zero(self) -> None:
        t0 = _make_utc(10, 0)
        order = BacktestOrder(
            order_id="DAY1-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        ticks = [
            _make_tick(bid=2600.00, ask=2600.00, time=t0),
            _make_tick(bid=2620.00, ask=2620.00, time=t0 + timedelta(minutes=1)),
        ]
        res = BacktestEngine().run(ticks=ticks, orders=[order])
        assert res.total_trades == 1
        assert res.sharpe_ratio == 0.0

    def test_engine_rejects_broker_and_config_together(self) -> None:
        broker = SimulatedBroker()
        with pytest.raises(ValueError, match="Cannot specify both broker and config"):
            BacktestEngine(broker=broker, config=BacktestConfig())

    def test_engine_setup_blocked_by_guard_cancels_pending(self) -> None:
        t0 = _make_utc(10, 0)
        limits = DailyRiskLimits(max_consecutive_losses=1)
        guard = RiskGuard(limits=limits)
        loser_order = BacktestOrder(
            order_id="LOSER-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2595.00,
            take_profit=2620.00,
            created_at=t0,
        )
        pending_order = BacktestOrder(
            order_id="PENDING-LIMIT",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2580.00,
            stop_loss=2570.00,
            take_profit=2600.00,
            created_at=t0,
        )
        setup = TradeSetup(
            id="SETUP-LATE",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t0 + timedelta(minutes=2),
            confirmed_time=t0 + timedelta(minutes=2),
            risk_reward_ratio=2.0,
        )
        engine = BacktestEngine(risk_guard=guard)
        ticks = [
            _make_tick(bid=2600.00, ask=2600.00, time=t0),
            _make_tick(bid=2594.00, ask=2594.25, time=t0 + timedelta(minutes=1)),
            _make_tick(bid=2600.00, ask=2600.25, time=t0 + timedelta(minutes=2)),
        ]
        res = engine.run(ticks=ticks, setups=[setup], orders=[loser_order, pending_order])
        assert res.total_trades == 1
        assert res.trades[0].order_id == "LOSER-1"
        cancelled_pending = [
            o for o in engine.broker.cancelled_orders if o.order_id == "PENDING-LIMIT"
        ]
        assert len(cancelled_pending) == 1
        assert cancelled_pending[0].cancellation_reason == "KILL_SWITCH"

    def test_close_all_positions_empty_or_bearish(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker()
        assert broker.close_all_positions(time=t0) == []

        order_sell = BacktestOrder(
            order_id="SELL-END",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2580.00,
            created_at=t0,
        )
        broker.submit_order(order_sell)
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=_make_utc(10, 1)))
        assert len(broker.active_positions) == 1

        closed = broker.close_all_positions(time=_make_utc(10, 2))
        assert len(closed) == 1
        assert closed[0].exit_reason == TradeExitReason.END_OF_DATA
        assert closed[0].exit_price == 2600.25
        assert len(broker.active_positions) == 0

    def test_close_all_positions_with_slippage_and_drawdown_update(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(slippage_points=5.0))
        order_buy = BacktestOrder(
            order_id="BUY-END-SLIP",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2580.00,
            take_profit=2650.00,
            created_at=t0,
        )
        broker.submit_order(order_buy)
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.00, time=_make_utc(10, 1)))
        assert len(broker.active_positions) == 1

        broker.process_tick(_make_tick(bid=2590.00, ask=2590.25, time=_make_utc(10, 2)))

        closed = broker.close_all_positions(time=_make_utc(10, 3))
        assert len(closed) == 1
        assert math.isclose(closed[0].exit_price, 2589.95, rel_tol=1e-5)
        assert broker.max_drawdown_usd > 0
        assert broker.max_drawdown_pct > 0

    def test_process_tick_fill_blocked_by_margin_or_max_positions(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(max_open_positions=1))
        order1 = BacktestOrder(
            order_id="ORD-1",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        order2 = BacktestOrder(
            order_id="ORD-2",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.pending_orders.extend([order1, order2])

        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))
        assert len(broker.active_positions) == 1
        assert len(broker.pending_orders) == 1
        assert broker.pending_orders[0].order_id == "ORD-2"

    def test_bearish_be_and_trailing(self) -> None:
        t0 = _make_utc(10, 0)
        risk_cfg = PositionRiskConfig(
            be_trigger_r=1.0,
            be_offset_points=10.0,
            trailing_atr_multiplier=1.0,
            min_trailing_step_points=5.0,
        )
        broker = SimulatedBroker(risk_config=risk_cfg)
        order = BacktestOrder(
            order_id="PROT-SELL",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2570.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=_make_utc(10, 1)))
        pos = broker.active_positions[0]
        assert pos.current_sl == 2610.00

        broker.process_tick(_make_tick(bid=2588.75, ask=2589.00, time=_make_utc(10, 2)))
        assert pos.be_applied is True
        assert pos.current_sl < 2600.00

        broker.process_tick(
            _make_tick(bid=2579.75, ask=2580.00, time=_make_utc(10, 3)),
            atr=2.0,
        )
        assert pos.trailing_applied is True
        assert pos.current_sl == 2582.00

        closed = broker.process_tick(_make_tick(bid=2582.00, ask=2582.25, time=_make_utc(10, 4)))
        assert len(closed) == 1
        assert closed[0].exit_reason == TradeExitReason.TRAILING_STOP
        assert closed[0].net_pnl > 0

    def test_engine_setups_and_orders_branches(self) -> None:
        t0 = _make_utc(10, 0)
        t_sun = _make_utc(10, 0, day=27)

        expired_setup = TradeSetup(
            id="EXP-SETUP",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t0,
            confirmed_time=t0,
            risk_reward_ratio=2.0,
            valid_until=t0 + timedelta(minutes=5),
        )
        ticks = [
            _make_tick(bid=2600.00, ask=2600.25, time=t0 + timedelta(minutes=6)),
        ]
        engine = BacktestEngine()
        res = engine.run(ticks=ticks, setups=[expired_setup])
        assert res.total_trades == 0

        tiny_risk_cfg = PositionRiskConfig(fixed_risk_usd=0.001)
        valid_setup = TradeSetup(
            id="TINY-RISK-SETUP",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t0,
            confirmed_time=t0,
            risk_reward_ratio=2.0,
        )
        engine_tiny = BacktestEngine(position_risk_config=tiny_risk_cfg)
        res2 = engine_tiny.run(ticks=ticks, setups=[valid_setup])
        assert res2.total_trades == 0

        tf = CompositeTradeFilter(session_filter=SessionFilter())
        blocked_order = BacktestOrder(
            order_id="BLOCKED-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t_sun,
        )
        ticks_sun = [_make_tick(bid=2600.00, ask=2600.25, time=t_sun)]
        engine_filter = BacktestEngine(trade_filter=tf)
        res3 = engine_filter.run(ticks=ticks_sun, orders=[blocked_order])
        assert res3.total_trades == 0

        limits = DailyRiskLimits(max_consecutive_losses=1)
        guard = RiskGuard(limits=limits)
        guard.initialize_day(
            t0,
            AccountState(
                balance=10000.0, equity=10000.0, margin=0.0, free_margin=10000.0, leverage=100.0
            ),
        )
        guard.trigger_manual_halt("Testing halt")
        engine_halt = BacktestEngine(risk_guard=guard)
        valid_order = BacktestOrder(
            order_id="HALTED-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        ticks_halt = [_make_tick(bid=2600.00, ask=2600.25, time=t0)]
        res4 = engine_halt.run(ticks=ticks_halt, orders=[valid_order])
        assert res4.total_trades == 0
        assert res4.win_rate == 0.0
        assert res4.profit_factor == 0.0
        assert res4.sharpe_ratio == 0.0
        assert res4.expectancy_r == 0.0
        assert res4.average_trade_duration_seconds == 0.0

    def test_broker_free_margin_and_used_margin_properties(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(initial_balance=10000.0, leverage=100.0))
        assert broker.used_margin == 0.0
        assert broker.free_margin == 10000.0

        order = BacktestOrder(
            order_id="MARGIN-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=1.0,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=t0))
        expected_used_margin = 1.0 * 100.0 * 2600.00 / 100.0
        expected_equity = 10000.0 - 25.0
        expected_free_margin = expected_equity - expected_used_margin
        assert math.isclose(broker.used_margin, expected_used_margin, rel_tol=1e-5)
        assert math.isclose(broker.free_margin, expected_free_margin, rel_tol=1e-5)

    def test_breakeven_exit_both_directions(self) -> None:
        t0 = _make_utc(10, 0)
        risk_cfg = PositionRiskConfig(
            be_trigger_r=1.0,
            be_offset_points=5.0,
        )

        broker_long = SimulatedBroker(risk_config=risk_cfg)
        order_buy = BacktestOrder(
            order_id="BUY-BE",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2630.00,
            created_at=t0,
        )
        broker_long.submit_order(order_buy)
        broker_long.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))
        broker_long.process_tick(_make_tick(bid=2611.00, ask=2611.25, time=_make_utc(10, 2)))
        pos_long = broker_long.active_positions[0]
        assert pos_long.be_applied is True
        be_sl = pos_long.current_sl
        closed_long = broker_long.process_tick(
            _make_tick(bid=be_sl - 0.10, ask=be_sl + 0.15, time=_make_utc(10, 3))
        )
        assert len(closed_long) == 1
        assert closed_long[0].exit_reason == TradeExitReason.BREAKEVEN

        broker_short = SimulatedBroker(risk_config=risk_cfg)
        order_sell = BacktestOrder(
            order_id="SELL-BE",
            direction=Direction.BEARISH,
            order_type=OrderType.SELL_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2610.00,
            take_profit=2570.00,
            created_at=t0,
        )
        broker_short.submit_order(order_sell)
        broker_short.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=_make_utc(10, 1)))
        broker_short.process_tick(_make_tick(bid=2588.00, ask=2588.25, time=_make_utc(10, 2)))
        pos_short = broker_short.active_positions[0]
        assert pos_short.be_applied is True
        be_sl_short = pos_short.current_sl
        closed_short = broker_short.process_tick(
            _make_tick(bid=be_sl_short - 0.25, ask=be_sl_short + 0.10, time=_make_utc(10, 3))
        )
        assert len(closed_short) == 1
        assert closed_short[0].exit_reason == TradeExitReason.BREAKEVEN

    def test_process_tick_fill_blocked_by_insufficient_margin(self) -> None:
        t0 = _make_utc(10, 0)
        broker = SimulatedBroker(config=BacktestConfig(initial_balance=1000.0, leverage=100.0))
        order = BacktestOrder(
            order_id="BIG-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.35,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.balance = 500.0
        broker.equity = 500.0
        broker.process_tick(_make_tick(bid=2599.75, ask=2600.00, time=_make_utc(10, 1)))
        assert len(broker.active_positions) == 0
        assert len(broker.pending_orders) == 1

    def test_engine_setup_blocked_by_risk_guard_and_end_close_with_risk_guard(self) -> None:
        t0 = _make_utc(10, 0)
        limits = DailyRiskLimits()
        guard = RiskGuard(limits=limits)
        guard.initialize_day(
            t0,
            AccountState(
                balance=10000.0, equity=10000.0, margin=0.0, free_margin=10000.0, leverage=100.0
            ),
        )
        guard.trigger_manual_halt("Manual halt")

        setup = TradeSetup(
            id="HALTED-SETUP",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t0,
            confirmed_time=t0,
            risk_reward_ratio=2.0,
        )
        ticks = [_make_tick(bid=2600.00, ask=2600.25, time=t0)]
        engine = BacktestEngine(risk_guard=guard)
        res = engine.run(ticks=ticks, setups=[setup])
        assert res.total_trades == 0

        guard.resume_trading()
        order = BacktestOrder(
            order_id="END-GUARD-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2580.00,
            take_profit=2640.00,
            created_at=t0,
        )
        ticks2 = [
            _make_tick(bid=2599.75, ask=2600.00, time=t0),
            _make_tick(bid=2610.00, ask=2610.25, time=t0 + timedelta(minutes=5)),
        ]
        engine2 = BacktestEngine(risk_guard=guard, close_positions_at_end=True)
        res2 = engine2.run(ticks=ticks2, orders=[order])
        assert res2.total_trades == 1
        assert res2.trades[0].exit_reason == TradeExitReason.END_OF_DATA
        assert res2.trades[0].net_pnl > 0

    def test_repro_bug_01_slippage_points_initializes_fixed_slippage(self) -> None:
        t0 = _make_utc(10, 0)
        config = BacktestConfig(slippage_points=5.0)
        broker = SimulatedBroker(config=config)
        order = BacktestOrder(
            order_id="BUY-MKT-SLIP",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        broker.submit_order(order)
        broker.process_tick(_make_tick(bid=2600.00, ask=2600.25, time=t0))
        assert len(broker.active_positions) == 1
        expected_fill_price = 2600.25 + 5.0 * 0.01
        assert math.isclose(
            broker.active_positions[0].entry_price, expected_fill_price, rel_tol=1e-5
        )

        engine = BacktestEngine(config=config)
        order_engine = BacktestOrder(
            order_id="BUY-ENG-SLIP",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        ticks = [
            _make_tick(bid=2600.00, ask=2600.25, time=t0),
            _make_tick(bid=2580.00, ask=2580.25, time=t0 + timedelta(minutes=1)),
        ]
        res = engine.run(ticks=ticks, orders=[order_engine])
        assert res.total_trades == 1
        assert math.isclose(res.trades[0].entry_price, 2600.30, rel_tol=1e-5)

    def test_repro_bug_02_engine_runs_trailing_stop_with_candles_and_atr(self) -> None:
        t0 = _make_utc(10, 0)
        candles = [
            Candle(
                time=t0 - timedelta(minutes=15 * (14 - i)),
                open=2600.00,
                high=2602.00,
                low=2598.00,
                close=2600.00,
            )
            for i in range(14)
        ]
        risk_cfg = PositionRiskConfig(
            trailing_atr_multiplier=1.0,
            min_trailing_step_points=5.0,
        )
        order = BacktestOrder(
            order_id="BUY-TRAIL",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2595.00,
            take_profit=2630.00,
            created_at=t0,
        )
        ticks = [
            _make_tick(bid=2600.00, ask=2600.00, time=t0),
            _make_tick(bid=2610.00, ask=2610.00, time=t0 + timedelta(minutes=1)),
            _make_tick(bid=2601.00, ask=2601.25, time=t0 + timedelta(minutes=2)),
        ]
        engine = BacktestEngine(position_risk_config=risk_cfg)
        result = engine.run(ticks=ticks, orders=[order], candles=candles)
        assert result.total_trades == 1
        assert result.trades[0].exit_reason == TradeExitReason.TRAILING_STOP
        assert result.trades[0].final_sl > 2595.00

    def test_repro_bug_03_engine_cancels_pending_orders_on_risk_guard_halt(self) -> None:
        t0 = _make_utc(10, 0)
        limits = DailyRiskLimits(max_daily_loss_usd=50.0)
        guard = RiskGuard(limits=limits)

        order1 = BacktestOrder(
            order_id="ORD-1-LOSER",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        order2 = BacktestOrder(
            order_id="ORD-2-PENDING",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2585.00,
            stop_loss=2575.00,
            take_profit=2610.00,
            created_at=t0,
        )
        ticks = [
            _make_tick(bid=2600.00, ask=2600.00, time=t0),
            _make_tick(bid=2589.50, ask=2589.75, time=t0 + timedelta(minutes=1)),
            _make_tick(bid=2584.00, ask=2584.25, time=t0 + timedelta(minutes=2)),
        ]
        engine = BacktestEngine(risk_guard=guard)
        result = engine.run(ticks=ticks, orders=[order1, order2])

        assert result.total_trades == 1
        assert result.trades[0].order_id == "ORD-1-LOSER"
        cancelled_orders = [
            o for o in engine.broker.cancelled_orders if o.order_id == "ORD-2-PENDING"
        ]
        assert len(cancelled_orders) == 1
        assert cancelled_orders[0].cancellation_reason == "KILL_SWITCH"
        assert len(engine.broker.active_positions) == 0
        assert len(engine.broker.pending_orders) == 0

    def test_repro_bug_04_no_lookahead_bias_in_candle_atr(self) -> None:
        t0 = _make_utc(10, 0)
        candles = [
            Candle(
                time=t0 - timedelta(minutes=5 * (15 - i)),
                open=2600.00,
                high=2601.00,
                low=2599.00,
                close=2600.00,
                timeframe=Timeframe.M5,
            )
            for i in range(14)
        ]
        extreme_candle = Candle(
            time=t0,
            open=2600.00,
            high=2800.00,
            low=2600.00,
            close=2700.00,
            timeframe=Timeframe.M5,
        )
        candles.append(extreme_candle)

        received_atrs: list[float | None] = []

        class SpyBroker(SimulatedBroker):
            def process_tick(
                self,
                tick: Tick,
                atr: float | None = None,
                structural_swing: Swing | None = None,
            ) -> list[BacktestTrade]:
                received_atrs.append(atr)
                return super().process_tick(tick, atr=atr, structural_swing=structural_swing)

        broker = SpyBroker()
        engine = BacktestEngine(broker=broker)

        ticks = [
            _make_tick(bid=2600.00, ask=2600.25, time=t0),
            _make_tick(bid=2600.00, ask=2600.25, time=t0 + timedelta(minutes=5)),
        ]
        engine.run(ticks=ticks, candles=candles)

        assert len(received_atrs) == 2
        assert received_atrs[0] is not None
        assert math.isclose(received_atrs[0], 2.0, rel_tol=1e-2)
        assert received_atrs[1] is not None
        assert received_atrs[1] > 10.0

    def test_engine_custom_slippage_model_without_broker(self) -> None:
        custom_slip = FixedSlippage(points=3.0, point_size=0.01)
        engine = BacktestEngine(slippage_model=custom_slip)
        assert engine.slippage_model is custom_slip
        assert engine.broker.slippage_model is custom_slip

    def test_engine_two_days_returns_zero_sharpe(self) -> None:
        t0 = _make_utc(10, 0, day=20)
        t1 = _make_utc(10, 0, day=21)
        ticks = [
            _make_tick(bid=2600.0, ask=2600.25, time=t0),
            _make_tick(bid=2605.0, ask=2605.25, time=t1),
        ]
        order = BacktestOrder(
            order_id="ORD-2DAYS",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_MARKET,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t0,
        )
        engine = BacktestEngine(close_positions_at_end=True)
        res = engine.run(ticks=ticks, orders=[order])
        assert res.total_trades == 1
        assert res.sharpe_ratio == 0.0

    def test_engine_filter_and_post_trade_guard_allows_trading(self) -> None:
        t_mon = _make_utc(8, 0, day=21)
        trade_filter = CompositeTradeFilter(session_filter=SessionFilter())
        limits = DailyRiskLimits(max_daily_loss_usd=500.0)
        guard = RiskGuard(limits=limits)

        setup = TradeSetup(
            id="ALLOWED-SETUP",
            direction=Direction.BULLISH,
            entry_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            time=t_mon,
            confirmed_time=t_mon,
            risk_reward_ratio=2.0,
            valid_until=t_mon + timedelta(minutes=15),
        )
        ticks = [
            _make_tick(bid=2599.75, ask=2600.00, time=t_mon),
            _make_tick(bid=2620.00, ask=2620.25, time=t_mon + timedelta(minutes=1)),
        ]
        engine = BacktestEngine(trade_filter=trade_filter, risk_guard=guard)
        res = engine.run(ticks=ticks, setups=[setup])
        assert res.total_trades == 1
        assert res.winning_trades == 1

        direct_order = BacktestOrder(
            order_id="ALLOWED-DIR-ORD",
            direction=Direction.BULLISH,
            order_type=OrderType.BUY_LIMIT,
            volume=0.1,
            limit_price=2600.00,
            stop_loss=2590.00,
            take_profit=2620.00,
            created_at=t_mon,
        )
        engine_direct = BacktestEngine(trade_filter=trade_filter, risk_guard=guard)
        res_direct = engine_direct.run(ticks=ticks, orders=[direct_order])
        assert res_direct.total_trades == 1
