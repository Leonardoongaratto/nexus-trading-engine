"""Live Trading Loop engine and runner for XAU/USD SMC Bot."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from xau_bot.advisor.modes import (
    AdvisorDispatcher,
    ProposalManager,
    ProposalStatus,
    TradeProposal,
)
from xau_bot.broker.models import AccountInfo, BrokerPosition, SymbolInfo
from xau_bot.broker.ports import BrokerPort
from xau_bot.domain.models import (
    Direction,
    OrderIntent,
    SwingType,
    Timeframe,
)
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import SpreadFilterConfig
from xau_bot.filters.spread import SpreadFilter
from xau_bot.journal.evaluator import OutcomeEvaluator, create_snapshot_from_proposal
from xau_bot.journal.models import (
    JournalEntry,
    OutcomeData,
    TradeOutcome,
)
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.shadow import ShadowEngine
from xau_bot.risk.models import (
    AccountState,
    OpenPosition,
    PositionRiskConfig,
    SymbolRiskSpec,
)
from xau_bot.risk.position import calculate_lot_size
from xau_bot.risk.protection import (
    check_breakeven,
    check_partial_close,
    check_smart_exit,
    check_trailing_stop,
)
from xau_bot.runtime.candle_loop import CandleLoop
from xau_bot.runtime.models import CandleCloseEvent
from xau_bot.runtime.watchdog import Watchdog
from xau_bot.smc.setup import detect_setups, determine_htf_bias, get_active_setups
from xau_bot.smc.swings import calculate_atr, find_swings

logger = logging.getLogger("xau_bot.runtime.live_loop")


@runtime_checkable
class LiveTelegramNotifier(Protocol):
    async def send_order_filled(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
    ) -> Any: ...

    async def send_order_cancelled(
        self,
        ticket: int,
        symbol: str,
        reason: str,
    ) -> Any: ...

    async def send_trade_closed(
        self,
        ticket: int,
        symbol: str,
        profit_usd: float,
        profit_r: float,
        exit_price: float,
    ) -> Any: ...

    async def send_break_even_activated(
        self,
        ticket: int,
        symbol: str,
        price: float,
    ) -> Any: ...

    async def send_trailing_stop_updated(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
    ) -> Any: ...

    async def send_partial_close_executed(
        self,
        ticket: int,
        symbol: str,
        volume_closed: float,
        remaining_volume: float,
        r_multiple: float,
        new_sl: float | None = None,
        runner_active: bool = False,
    ) -> Any: ...

    async def send_smart_exit_triggered(
        self,
        ticket: int,
        symbol: str,
        reason: str,
        exit_price: float,
    ) -> Any: ...


def symbol_info_to_risk_spec(spec: SymbolInfo) -> SymbolRiskSpec:
    return SymbolRiskSpec(
        symbol=spec.name,
        contract_size=spec.contract_size,
        tick_size=spec.tick_size,
        tick_value=spec.tick_value,
        lot_min=spec.min_volume,
        lot_max=spec.max_volume,
        lot_step=spec.volume_step,
        point=spec.point_size,
        stops_level_points=spec.stops_level_points,
        freeze_level_points=spec.freeze_level_points,
        spread_points=spec.spread_points,
    )


def account_info_to_state(acc: AccountInfo) -> AccountState:
    return AccountState(
        balance=acc.balance,
        equity=acc.equity,
        margin=acc.margin,
        free_margin=acc.free_margin,
        leverage=acc.leverage,
    )


class LiveTradingEngine:
    def __init__(
        self,
        broker: BrokerPort,
        proposal_manager: ProposalManager,
        advisor_dispatcher: AdvisorDispatcher,
        candle_loop: CandleLoop | None = None,
        composite_filter: CompositeTradeFilter | None = None,
        risk_config: PositionRiskConfig | None = None,
        symbol: str = "XAUUSD",
        timeframes: Sequence[Timeframe] = (
            Timeframe.D1,
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
        ),
        trigger_timeframe: Timeframe | Sequence[Timeframe] = Timeframe.M5,
        is_trading_halted_fn: Callable[[], bool] | None = None,
        telegram_service: LiveTelegramNotifier | None = None,
        magic: int = 123456,
        journal_repo: JournalRepository | None = None,
        evaluator: OutcomeEvaluator | None = None,
        shadow_engine: ShadowEngine | None = None,
        shadow_params: dict[str, Any] | None = None,
        active_params: dict[str, Any] | None = None,
        require_premium_discount: bool = True,
        require_sweep_guard: bool = True,
        sweep_guard_max_rejection_ratio: float = 0.40,
        htf_alignment_required: bool = True,
    ) -> None:
        self.broker = broker
        self.proposal_manager = proposal_manager
        self.advisor_dispatcher = advisor_dispatcher
        if isinstance(trigger_timeframe, Timeframe):
            self.trigger_timeframe = trigger_timeframe
            self.trigger_timeframes: tuple[Timeframe, ...] = (trigger_timeframe,)
        else:
            self.trigger_timeframes = tuple(trigger_timeframe)
            self.trigger_timeframe = (
                self.trigger_timeframes[0] if self.trigger_timeframes else Timeframe.M5
            )

        all_tfs = list(timeframes)
        for tf in self.trigger_timeframes:
            if tf not in all_tfs:
                all_tfs.append(tf)
        self.timeframes = tuple(all_tfs)
        self.candle_loop = candle_loop or CandleLoop(timeframes=self.timeframes)
        self.composite_filter = composite_filter
        self.risk_config = risk_config or PositionRiskConfig()
        self.symbol = symbol
        self.is_trading_halted_fn = is_trading_halted_fn
        self.telegram_service: LiveTelegramNotifier | None = telegram_service
        self.magic = magic
        self.require_premium_discount = require_premium_discount
        self.require_sweep_guard = require_sweep_guard
        self.sweep_guard_max_rejection_ratio = sweep_guard_max_rejection_ratio
        self.htf_alignment_required = htf_alignment_required
        self.shadow_engine = shadow_engine
        self.shadow_params = shadow_params or {}
        self.active_params = active_params or {}
        self.shadow_composite_filter: CompositeTradeFilter | None = None
        if self.composite_filter is not None and self.shadow_params:
            shadow_spread_ratio = self.shadow_params.get(
                "max_spread_atr_ratio",
                (
                    self.composite_filter.spread_filter.config.max_spread_atr_ratio
                    if self.composite_filter.spread_filter is not None
                    else 0.05
                ),
            )
            shadow_spread_filter = SpreadFilter(
                config=SpreadFilterConfig(max_spread_atr_ratio=shadow_spread_ratio)
            )
            self.shadow_composite_filter = CompositeTradeFilter(
                session_filter=self.composite_filter.session_filter,
                news_filter=self.composite_filter.news_filter,
                spread_filter=shadow_spread_filter,
            )
        self.seen_setup_ids: set[str] = set()
        self.seen_shadow_setup_ids: set[str] = set()
        self.initial_stops: dict[int, float] = {}
        self.initial_tps: dict[int, float | None] = {}
        self.tracked_positions: dict[int, BrokerPosition] = {}
        self.partially_closed_tickets: set[int] = set()
        self.smart_exited_tickets: set[int] = set()
        self.notified_open_tickets: set[int] = set()
        self._last_spread_price: float = 0.25
        self._last_htf_bias: str | None = None
        self._last_news_distance_min: float | None = None
        self.journal_repo = journal_repo
        self.evaluator: OutcomeEvaluator = (
            evaluator if evaluator is not None else OutcomeEvaluator()
        )
        if self.journal_repo is not None:
            self.proposal_manager.add_listener(self._on_proposal_event)

    def _on_proposal_event(self, proposal: TradeProposal) -> None:
        if self.journal_repo is None:
            return
        entry = self.journal_repo.get_by_proposal_id(proposal.id)
        if entry is None:
            snap = create_snapshot_from_proposal(
                proposal,
                spread_price=self._last_spread_price,
                htf_bias=self._last_htf_bias,
                news_distance_min=self._last_news_distance_min,
            )
            out_data = OutcomeData(
                executed=proposal.status == ProposalStatus.DISPATCHED,
                entry_price=proposal.entry_price,
                stop_loss=proposal.stop_loss,
                take_profit=proposal.take_profit,
            )
            entry = JournalEntry(
                id=f"jrn_{proposal.id}",
                proposal_id=proposal.id,
                snapshot=snap,
                ticket=proposal.order_ticket,
                symbol=proposal.symbol,
                direction=proposal.direction,
                status=proposal.status,
                outcome=TradeOutcome.UNEXECUTED,
                outcome_data=out_data,
                created_at=proposal.created_at,
            )
            self.journal_repo.save_entry(entry)
        else:
            entry.status = proposal.status
            if proposal.order_ticket is not None:
                entry.ticket = proposal.order_ticket
            if proposal.status == ProposalStatus.EXPIRED:
                entry.outcome = TradeOutcome.EXPIRED
                entry.outcome_data = OutcomeData(
                    executed=False,
                    entry_price=proposal.entry_price,
                    stop_loss=proposal.stop_loss,
                    take_profit=proposal.take_profit,
                    exit_reason="Proposal expired by timeout",
                    realized_r=0.0,
                )
            elif proposal.status == ProposalStatus.CANCELLED:
                entry.outcome = TradeOutcome.INVALIDATED
                entry.outcome_data = OutcomeData(
                    executed=False,
                    entry_price=proposal.entry_price,
                    stop_loss=proposal.stop_loss,
                    take_profit=proposal.take_profit,
                    exit_price=proposal.stop_loss,
                    exit_reason=proposal.reason or "invalidation_price_reached",
                    realized_r=0.0,
                )
            elif proposal.status == ProposalStatus.REJECTED:
                entry.outcome = TradeOutcome.UNEXECUTED
                entry.outcome_data = OutcomeData(
                    executed=False,
                    entry_price=proposal.entry_price,
                    stop_loss=proposal.stop_loss,
                    take_profit=proposal.take_profit,
                    exit_reason=proposal.reason or "Proposal rejected by user",
                    realized_r=0.0,
                )
            self.journal_repo.save_entry(entry)

    async def on_candle_close(self, event: CandleCloseEvent) -> None:
        trigger_tf = getattr(
            event,
            "timeframe",
            getattr(getattr(event, "candle", None), "timeframe", self.trigger_timeframe),
        )
        if trigger_tf not in self.trigger_timeframes:
            return

        if self.is_trading_halted_fn and self.is_trading_halted_fn():
            logger.info(
                "LiveTradingEngine: Trading is halted. Skipping setup check for closed candle %s",
                event.candle.time,
            )
            return

        symbol_info = await self.broker.get_symbol_info(self.symbol)
        if symbol_info is None:
            logger.warning("LiveTradingEngine: Symbol info unavailable for %s", self.symbol)
            return

        self._last_spread_price = float(symbol_info.spread_points) * float(symbol_info.point_size)
        entry_candles = self.candle_loop.get_closed_series(trigger_tf)
        if len(entry_candles) < 20:
            return

        atr_vals = calculate_atr(entry_candles, period=14)
        current_atr = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else None
        if current_atr is None or current_atr <= 0:
            logger.info(
                "LiveTradingEngine: ATR unavailable for %s (%s), skipping setup check",
                self.symbol,
                trigger_tf.value,
            )
            return

        if trigger_tf == Timeframe.M1:
            htf_candles = self.candle_loop.get_closed_series(Timeframe.M15)
            if not htf_candles:
                htf_candles = self.candle_loop.get_closed_series(Timeframe.H1)
            if not htf_candles:
                htf_candles = self.candle_loop.get_closed_series(Timeframe.H4)
        else:
            htf_candles = self.candle_loop.get_closed_series(Timeframe.H1)
            if not htf_candles:
                htf_candles = self.candle_loop.get_closed_series(Timeframe.H4)
            if not htf_candles:
                htf_candles = self.candle_loop.get_closed_series(Timeframe.D1)

        if htf_candles:
            computed_bias = determine_htf_bias(htf_candles)
            if computed_bias is not None:
                self._last_htf_bias = computed_bias.value

        if self.composite_filter is not None and self.composite_filter.news_filter is not None:
            self._last_news_distance_min = (
                self.composite_filter.news_filter.get_minutes_to_next_news(event.closed_at)
            )
        else:
            self._last_news_distance_min = None

        filter_res = None
        if self.composite_filter is not None:
            spread_price = self._last_spread_price
            filter_res = self.composite_filter.check(
                current_time=event.closed_at,
                spread=spread_price,
                atr=current_atr,
            )

        shadow_filter_res = None
        if self.shadow_composite_filter is not None:
            spread_price = self._last_spread_price
            shadow_filter_res = self.shadow_composite_filter.check(
                current_time=event.closed_at,
                spread=spread_price,
                atr=current_atr,
            )

        has_shadow = self.shadow_engine is not None and bool(self.shadow_params)
        if filter_res is not None and not filter_res.can_trade and not has_shadow:
            logger.info(
                "LiveTradingEngine: Filter blocked trade at %s: %s",
                event.closed_at,
                filter_res.details,
            )
            return

        detected = detect_setups(
            candles=entry_candles,
            htf_candles=htf_candles if htf_candles else None,
            as_of_time=event.closed_at,
            require_premium_discount=self.require_premium_discount,
            require_sweep_guard=self.require_sweep_guard,
            sweep_guard_max_rejection_ratio=self.sweep_guard_max_rejection_ratio,
            htf_alignment_required=self.htf_alignment_required,
        )
        active_setups = get_active_setups(detected, as_of_time=event.closed_at)
        logger.info(
            "LiveTradingEngine [%s]: Vela fechada %s (close=%.2f, ATR=%.2f) - %d setup(s) ativo(s)",
            trigger_tf.value,
            event.candle.time.strftime("%H:%M:%S"),
            event.candle.close,
            current_atr,
            len(active_setups),
        )

        for setup in active_setups:
            if has_shadow and setup.id not in self.seen_shadow_setup_ids:
                assert self.shadow_engine is not None
                self.seen_shadow_setup_ids.add(setup.id)
                eval_res = self.shadow_engine.evaluate_setup(
                    setup=setup,
                    active_params=self.active_params,
                    shadow_params=self.shadow_params,
                    spread=symbol_info.spread_points * symbol_info.point_size,
                    atr=current_atr,
                    timestamp=event.closed_at,
                )

                active_ok = eval_res.active_allowed and (filter_res is None or filter_res.can_trade)
                shadow_ok = eval_res.shadow_allowed and (
                    shadow_filter_res is None or shadow_filter_res.can_trade
                )

                active_reason = eval_res.active_reason
                if filter_res is not None and not filter_res.can_trade:
                    active_reason = f"composite_filter: {filter_res.details}"

                shadow_reason = eval_res.shadow_reason
                if shadow_filter_res is not None and not shadow_filter_res.can_trade:
                    shadow_reason = f"composite_filter: {shadow_filter_res.details}"

                eval_res = self.shadow_engine.evaluate_filter_decision(
                    setup_id=eval_res.setup_id,
                    direction=eval_res.direction,
                    active_allowed=active_ok,
                    shadow_allowed=shadow_ok,
                    active_reason=active_reason,
                    shadow_reason=shadow_reason,
                    timestamp=eval_res.timestamp,
                )
                if self.journal_repo is not None:
                    self.shadow_engine.record_evaluation(self.journal_repo, eval_res)

        if filter_res is not None and not filter_res.can_trade:
            logger.info(
                "LiveTradingEngine: Filter blocked trade at %s: %s",
                event.closed_at,
                filter_res.details,
            )
            return

        for setup in active_setups:
            if setup.id in self.seen_setup_ids:
                continue
            self.seen_setup_ids.add(setup.id)

            order_type_str = "BUY_LIMIT" if setup.direction == Direction.BULLISH else "SELL_LIMIT"
            execution_price = setup.entry_price

            if setup.direction == Direction.BULLISH:
                if setup.take_profit is not None:
                    total_move = setup.take_profit - setup.entry_price
                    dist_to_tp = setup.take_profit - symbol_info.bid
                    if dist_to_tp <= 0:
                        logger.info(
                            "LiveTradingEngine: Setup compra %s descartado "
                            "(mercado %.2f >= TP %.2f já atingido)",
                            setup.id,
                            symbol_info.bid,
                            setup.take_profit,
                        )
                        continue
                    if total_move > 0 and dist_to_tp < 0.3 * total_move:
                        logger.info(
                            "LiveTradingEngine: Setup compra %s descartado "
                            "(mercado %.2f já percorreu >70%% rumo ao TP %.2f sem pullback)",
                            setup.id,
                            symbol_info.bid,
                            setup.take_profit,
                        )
                        continue
                if setup.stop_loss is not None and symbol_info.bid <= setup.stop_loss:
                    logger.info(
                        "LiveTradingEngine: Setup compra %s descartado "
                        "(mercado %.2f <= SL %.2f rompido)",
                        setup.id,
                        symbol_info.bid,
                        setup.stop_loss,
                    )
                    continue

                if setup.entry_price > symbol_info.ask + 1e-4:
                    order_type_str = "BUY"
                    execution_price = symbol_info.ask
                    logger.info(
                        "LiveTradingEngine: Setup compra %s em desconto favorável "
                        "(ask %.2f <= entrada %.2f); executando a mercado BUY",
                        setup.id,
                        symbol_info.ask,
                        setup.entry_price,
                    )
                else:
                    order_type_str = "BUY_LIMIT"
                    execution_price = setup.entry_price

            elif setup.direction == Direction.BEARISH:
                if setup.take_profit is not None:
                    total_move = setup.entry_price - setup.take_profit
                    dist_to_tp = symbol_info.ask - setup.take_profit
                    if dist_to_tp <= 0:
                        logger.info(
                            "LiveTradingEngine: Setup venda %s descartado "
                            "(mercado %.2f <= TP %.2f já atingido)",
                            setup.id,
                            symbol_info.ask,
                            setup.take_profit,
                        )
                        continue
                    if total_move > 0 and dist_to_tp < 0.3 * total_move:
                        logger.info(
                            "LiveTradingEngine: Setup venda %s descartado "
                            "(mercado %.2f já percorreu >70%% rumo ao TP %.2f sem pullback)",
                            setup.id,
                            symbol_info.ask,
                            setup.take_profit,
                        )
                        continue
                if setup.stop_loss is not None and symbol_info.ask >= setup.stop_loss:
                    logger.info(
                        "LiveTradingEngine: Setup venda %s descartado "
                        "(mercado %.2f >= SL %.2f rompido)",
                        setup.id,
                        symbol_info.ask,
                        setup.stop_loss,
                    )
                    continue

                if setup.entry_price < symbol_info.bid - 1e-4:
                    order_type_str = "SELL"
                    execution_price = symbol_info.bid
                    logger.info(
                        "LiveTradingEngine: Setup venda %s em prêmio favorável "
                        "(bid %.2f >= entrada %.2f); executando a mercado SELL",
                        setup.id,
                        symbol_info.bid,
                        setup.entry_price,
                    )
                else:
                    order_type_str = "SELL_LIMIT"
                    execution_price = setup.entry_price

            account_info = await self.broker.get_account_info()
            account_state = account_info_to_state(account_info)
            risk_spec = symbol_info_to_risk_spec(symbol_info)
            cfg = self.risk_config

            spread_val = symbol_info.spread_points * symbol_info.point_size
            lot_calc = calculate_lot_size(
                account=account_state,
                spec=risk_spec,
                config=cfg,
                direction=setup.direction,
                entry_price=execution_price,
                stop_loss=setup.stop_loss,
                spread=spread_val,
                adjust_sl_for_spread=True,
                buffer_points=max(symbol_info.spread_points * 0.5, 2.0),
                min_sl_spread_multiplier=2.0,
            )

            if not lot_calc.success:
                logger.warning(
                    "LiveTradingEngine: Lot sizing failed for setup %s: %s (%s)",
                    setup.id,
                    lot_calc.details,
                    lot_calc.reject_reason,
                )
                continue

            valid_until = setup.valid_until or (
                event.closed_at + timedelta(seconds=self.proposal_manager.default_timeout_s)
            )

            magic_offset = (
                1 if trigger_tf == Timeframe.M1 else (5 if trigger_tf == Timeframe.M5 else 15)
            )
            magic_number = self.magic + magic_offset
            order_comment = f"SMC-{trigger_tf.value}-{setup.id[:6]}"
            final_sl = (
                lot_calc.effective_sl if lot_calc.effective_sl is not None else setup.stop_loss
            )

            dev_val = 20 if order_type_str in ("BUY", "SELL") else None
            intent = OrderIntent(
                symbol=self.symbol,
                direction=setup.direction,
                order_type=order_type_str,
                volume=lot_calc.lot,
                price=execution_price,
                stop_loss=final_sl,
                take_profit=setup.take_profit,
                magic=magic_number,
                comment=order_comment,
                expiration=valid_until if order_type_str in ("BUY_LIMIT", "SELL_LIMIT") else None,
                deviation=dev_val,
            )

            risk_pct_val = (
                round((lot_calc.loss_at_sl_usd / account_state.balance) * 100.0, 2)
                if account_state.balance > 0
                else 1.0
            )

            await self.advisor_dispatcher.handle_setup(
                setup=setup,
                intent=intent,
                volume=lot_calc.lot,
                risk_usd=lot_calc.loss_at_sl_usd,
                risk_pct=risk_pct_val,
                timeout_s=self.proposal_manager.default_timeout_s,
                atr=current_atr,
            )

    async def maintain_active_proposals_and_positions(
        self,
        current_price: float,
        ask_price: float | None = None,
        current_time: datetime | None = None,
    ) -> None:
        now = current_time or datetime.now(UTC)

        cancelled = self.proposal_manager.cancel_by_price(
            self.symbol, current_price, ask_price=ask_price
        )
        if cancelled:
            logger.info(
                "LiveTradingEngine: Cancelled %d proposal(s) due to price invalidation",
                len(cancelled),
            )

        expired = self.proposal_manager.check_expirations(now)
        if expired:
            logger.info("LiveTradingEngine: Expired %d proposal(s) due to timeout", len(expired))

        if cancelled or expired:
            self.evaluate_unexecuted_proposals()

        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            logger.warning("Error fetching open orders for expiration check: %s", exc)
            open_orders = ()

        sym_orders = [o for o in open_orders if o.symbol == self.symbol]
        for ord_item in sym_orders:
            matching_prop = None
            for p in self.proposal_manager.get_all_proposals():
                if p.order_ticket == ord_item.ticket:
                    matching_prop = p
                    break

            should_cancel = False
            cancel_reason = ""

            if matching_prop is not None:
                if now >= matching_prop.expires_at:
                    should_cancel = True
                    cancel_reason = (
                        f"validade expirada ({matching_prop.expires_at.strftime('%H:%M:%S')})"
                    )
                elif matching_prop.direction == Direction.BULLISH:
                    if (
                        matching_prop.take_profit is not None
                        and current_price >= matching_prop.take_profit
                    ):
                        should_cancel = True
                        cancel_reason = (
                            f"alvo atingido sem pullback (mercado {current_price:.2f} >= "
                            f"TP {matching_prop.take_profit:.2f})"
                        )
                    elif (
                        matching_prop.stop_loss is not None
                        and current_price <= matching_prop.stop_loss
                    ):
                        should_cancel = True
                        cancel_reason = (
                            f"stop rompido antes da entrada (mercado {current_price:.2f} <= "
                            f"SL {matching_prop.stop_loss:.2f})"
                        )
                elif matching_prop.direction == Direction.BEARISH:
                    if (
                        matching_prop.take_profit is not None
                        and current_price <= matching_prop.take_profit
                    ):
                        should_cancel = True
                        cancel_reason = (
                            f"alvo atingido sem pullback (mercado {current_price:.2f} <= "
                            f"TP {matching_prop.take_profit:.2f})"
                        )
                    elif (
                        matching_prop.stop_loss is not None
                        and current_price >= matching_prop.stop_loss
                    ):
                        should_cancel = True
                        cancel_reason = (
                            f"stop rompido antes da entrada (mercado {current_price:.2f} >= "
                            f"SL {matching_prop.stop_loss:.2f})"
                        )

            if should_cancel:
                logger.info(
                    "LiveTradingEngine: Cancelando ordem pendente #%d no broker: %s",
                    ord_item.ticket,
                    cancel_reason,
                )
                try:
                    await self.broker.cancel_order(ord_item.ticket)
                    if matching_prop is not None:
                        matching_prop.status = ProposalStatus.CANCELLED
                        matching_prop.reason = cancel_reason
                        self.proposal_manager._notify_listeners(matching_prop)
                    if self.telegram_service is not None:
                        try:
                            await self.telegram_service.send_order_cancelled(
                                ticket=ord_item.ticket,
                                symbol=ord_item.symbol,
                                reason=cancel_reason,
                            )
                        except Exception as tg_err:
                            logger.warning("Falha ao enviar cancelamento via Telegram: %s", tg_err)
                except Exception as cancel_exc:
                    logger.warning(
                        "Falha ao cancelar ordem pendente #%d: %s",
                        ord_item.ticket,
                        cancel_exc,
                    )

        try:
            open_positions = await self.broker.get_open_positions()
        except Exception as exc:
            logger.warning("Error fetching open positions for protection: %s", exc)
            return

        sym_positions = [p for p in open_positions if p.symbol == self.symbol]
        active_tickets = {p.ticket for p in sym_positions}

        for pos in sym_positions:
            self.tracked_positions[pos.ticket] = pos
            if pos.ticket not in self.notified_open_tickets:
                self.notified_open_tickets.add(pos.ticket)
                if self.telegram_service is not None:
                    try:
                        await self.telegram_service.send_order_filled(
                            ticket=pos.ticket,
                            symbol=pos.symbol,
                            direction=pos.direction.value,
                            volume=pos.volume,
                            price=pos.open_price,
                        )
                    except Exception as tg_err:
                        logger.warning("Falha ao enviar ordem executada via Telegram: %s", tg_err)

        for t in list(self.initial_stops.keys()):
            if t not in active_tickets:
                is_runner = t in self.partially_closed_tickets
                self.notified_open_tickets.discard(t)
                self.partially_closed_tickets.discard(t)
                self.smart_exited_tickets.discard(t)
                initial_sl = self.initial_stops.pop(t, None)
                initial_tp = self.initial_tps.pop(t, None)
                last_pos = self.tracked_positions.pop(t, None)

                exit_p = current_price
                exit_rsn = "Runner closed (inferred)" if is_runner else "Position closed (inferred)"
                is_bull = last_pos.direction == Direction.BULLISH if last_pos else True
                active_sl = (
                    last_pos.stop_loss
                    if (last_pos and last_pos.stop_loss is not None and last_pos.stop_loss > 0)
                    else initial_sl
                )
                if active_sl is not None and (
                    (is_bull and current_price <= active_sl)
                    or (not is_bull and current_price >= active_sl)
                ):
                    exit_rsn = (
                        "Trailing stop hit (runner closed)"
                        if is_runner
                        else "Stop loss hit (inferred)"
                    )
                    exit_p = active_sl
                elif initial_tp is not None and (
                    (is_bull and current_price >= initial_tp)
                    or (not is_bull and current_price <= initial_tp)
                ):
                    exit_rsn = "Take profit hit (inferred)"
                    exit_p = initial_tp

                fill_p = last_pos.open_price if last_pos else exit_p
                pnl_usd = 0.0
                contract_size = 100.0
                if last_pos:
                    sym_info = await self.broker.get_symbol_info(self.symbol)
                    contract_size = sym_info.contract_size if sym_info is not None else 100.0
                    pnl_usd = (
                        (exit_p - fill_p) * last_pos.volume * contract_size
                        if is_bull
                        else (fill_p - exit_p) * last_pos.volume * contract_size
                    )

                if self.telegram_service is not None:
                    risk_dist = abs(fill_p - initial_sl) if initial_sl else 1.0
                    vol_val = last_pos.volume if last_pos else 0.01
                    risk_total_usd = risk_dist * vol_val * contract_size
                    r_mult = pnl_usd / max(1e-4, risk_total_usd)
                    try:
                        await self.telegram_service.send_trade_closed(
                            ticket=t,
                            symbol=self.symbol,
                            profit_usd=pnl_usd,
                            profit_r=r_mult,
                            exit_price=exit_p,
                        )
                    except Exception as tg_err:
                        logger.warning("Falha ao enviar trade encerrado via Telegram: %s", tg_err)

                if self.journal_repo is not None:
                    j_entry = self.journal_repo.get_by_ticket(t)
                    if j_entry is not None:
                        all_candles = self.candle_loop.get_closed_series(Timeframe.M5)
                        pos_open_time = last_pos.open_time if last_pos else j_entry.created_at
                        if not last_pos and j_entry.outcome_data:
                            fill_p = j_entry.outcome_data.entry_price
                        pos_candles = [c for c in all_candles if pos_open_time <= c.time <= now]
                        if not pos_candles and all_candles:
                            pos_candles = [all_candles[-1]]

                        out_data, outcome, loss_cause, loss_cat = self.evaluator.evaluate_position(
                            entry=j_entry,
                            exit_price=exit_p,
                            exit_time=now,
                            position_candles=pos_candles,
                            open_time=pos_open_time,
                            fill_price=fill_p,
                            stop_loss=initial_sl,
                            take_profit=initial_tp,
                            pnl_usd=pnl_usd,
                            exit_reason=exit_rsn,
                        )
                        j_entry.outcome_data = out_data
                        j_entry.outcome = outcome
                        j_entry.loss_cause = loss_cause
                        j_entry.loss_category = loss_cat
                        self.journal_repo.save_entry(j_entry)

        if not sym_positions:
            return

        symbol_info = await self.broker.get_symbol_info(self.symbol)
        if symbol_info is None:
            return

        risk_spec = symbol_info_to_risk_spec(symbol_info)
        cfg = self.risk_config

        m5_candles = self.candle_loop.get_closed_series(Timeframe.M5)
        swings = find_swings(m5_candles) if len(m5_candles) >= 10 else []
        atr_vals = calculate_atr(m5_candles, period=14) if len(m5_candles) >= 14 else []
        current_atr: float | None = atr_vals[-1] if atr_vals and atr_vals[-1] is not None else None

        for pos in sym_positions:
            if pos.stop_loss is None or pos.stop_loss <= 0:
                continue

            if pos.ticket not in self.initial_stops:
                matched_sl = None
                for prop in self.proposal_manager.get_all_proposals():
                    if (
                        prop.order_ticket == pos.ticket
                        and prop.intent is not None
                        and prop.intent.stop_loss is not None
                    ):
                        matched_sl = prop.intent.stop_loss
                        break
                self.initial_stops[pos.ticket] = (
                    matched_sl if matched_sl is not None else pos.stop_loss
                )

            initial_sl = self.initial_stops[pos.ticket]

            if pos.ticket not in self.initial_tps:
                if pos.take_profit is not None and pos.take_profit > 0:
                    self.initial_tps[pos.ticket] = pos.take_profit
                else:
                    risk_dist = abs(pos.open_price - initial_sl)
                    self.initial_tps[pos.ticket] = (
                        pos.open_price + 2.0 * risk_dist
                        if pos.direction == Direction.BULLISH
                        else pos.open_price - 2.0 * risk_dist
                    )

            tp_val = (
                None
                if (cfg.remove_tp_after_partial and pos.ticket in self.partially_closed_tickets)
                else (
                    pos.take_profit
                    if pos.take_profit is not None and pos.take_profit > 0
                    else self.initial_tps.get(pos.ticket)
                )
            )

            open_pos = OpenPosition(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction=pos.direction,
                volume=pos.volume,
                entry_price=pos.open_price,
                current_stop_loss=pos.stop_loss,
                initial_stop_loss=initial_sl,
                open_time=pos.open_time,
                current_take_profit=tp_val,
                initial_take_profit=self.initial_tps.get(pos.ticket),
            )

            # 1. Avalia Saída Parcial (ex: 50% no TP1)
            pc_res = check_partial_close(
                position=open_pos,
                current_price=current_price,
                spec=risk_spec,
                config=cfg,
                already_partially_closed=pos.ticket in self.partially_closed_tickets,
            )

            if pc_res.should_close and pc_res.volume_to_close > 0:
                close_res = await self.broker.close_position(
                    ticket=pos.ticket,
                    volume=pc_res.volume_to_close,
                )
                if close_res.success:
                    self.partially_closed_tickets.add(pos.ticket)
                    logger.info(
                        "Partial close 50%% executed on position %d: closed %.2f, remaining %.2f",
                        pos.ticket,
                        pc_res.volume_to_close,
                        pc_res.remaining_volume,
                    )
                    # Imediatamente posiciona o Stop Loss no Break-Even para a metade restante
                    commission_dist = (
                        risk_spec.commission_per_lot / risk_spec.tick_value
                    ) * risk_spec.tick_size
                    offset_dist = cfg.be_offset_points * risk_spec.point
                    total_cost_buffer = (
                        (symbol_info.spread_points * symbol_info.point_size)
                        + commission_dist
                        + offset_dist
                    )
                    price_decimals = max(0, -int(math.floor(math.log10(risk_spec.point))))

                    new_sl = None
                    if pos.direction == Direction.BULLISH:
                        candidate_sl = round(pos.open_price + total_cost_buffer, price_decimals)
                        if pos.stop_loss is None or pos.stop_loss < candidate_sl:
                            new_sl = candidate_sl
                    else:
                        candidate_sl = round(pos.open_price - total_cost_buffer, price_decimals)
                        if pos.stop_loss is None or pos.stop_loss > candidate_sl:
                            new_sl = candidate_sl

                    target_sl = new_sl if new_sl is not None else pos.stop_loss
                    target_tp = 0.0 if cfg.remove_tp_after_partial else pos.take_profit
                    if cfg.remove_tp_after_partial:
                        self.initial_tps[pos.ticket] = None

                    if target_sl is not None or cfg.remove_tp_after_partial:
                        await self.broker.modify_order(
                            ticket=pos.ticket,
                            stop_loss=target_sl,
                            take_profit=target_tp,
                        )
                    if self.telegram_service is not None:
                        try:
                            await self.telegram_service.send_partial_close_executed(
                                ticket=pos.ticket,
                                symbol=pos.symbol,
                                volume_closed=pc_res.volume_to_close,
                                remaining_volume=pc_res.remaining_volume,
                                r_multiple=pc_res.current_r_multiple,
                                new_sl=target_sl,
                                runner_active=cfg.remove_tp_after_partial,
                            )
                        except Exception as err:
                            logger.warning(
                                "Failed to send partial close notification via Telegram: %s", err
                            )
                    continue

            # 2. Avalia Break-Even normal
            be_res = check_breakeven(
                position=open_pos,
                current_price=current_price,
                spec=risk_spec,
                config=cfg,
                spread=symbol_info.spread_points * symbol_info.point_size,
            )

            if be_res.should_modify and be_res.new_stop_loss is not None:
                mod_res = await self.broker.modify_order(
                    ticket=pos.ticket,
                    stop_loss=be_res.new_stop_loss,
                    take_profit=pos.take_profit,
                )
                if mod_res.success:
                    logger.info(
                        "Break-even activated on position %d at %.2f",
                        pos.ticket,
                        be_res.new_stop_loss,
                    )
                    if self.telegram_service is not None:
                        try:
                            await self.telegram_service.send_break_even_activated(
                                ticket=pos.ticket,
                                symbol=pos.symbol,
                                price=be_res.new_stop_loss,
                            )
                        except Exception as err:
                            logger.warning("Failed to send BE notification via Telegram: %s", err)
                continue

            # 3. Avalia Smart Exit (invalidação estrutural em M5)
            se_res = check_smart_exit(
                position=open_pos,
                current_price=current_price,
                recent_candles=m5_candles,
                swings=swings,
                config=cfg,
            )
            if se_res.should_exit and pos.ticket not in self.smart_exited_tickets:
                self.smart_exited_tickets.add(pos.ticket)
                exit_res = await self.broker.close_position(ticket=pos.ticket)
                if exit_res.success:
                    logger.info(
                        "Smart Exit executed on position %d: %s",
                        pos.ticket,
                        se_res.reason,
                    )
                    if self.telegram_service is not None:
                        try:
                            await self.telegram_service.send_smart_exit_triggered(
                                ticket=pos.ticket,
                                symbol=pos.symbol,
                                reason=se_res.reason,
                                exit_price=current_price,
                            )
                        except Exception as err:
                            logger.warning(
                                "Failed to send Smart Exit notification via Telegram: %s", err
                            )
                    continue

            target_swing_type = (
                SwingType.LOW if pos.direction == Direction.BULLISH else SwingType.HIGH
            )
            matching_swings = [
                s
                for s in swings
                if s.type == target_swing_type and s.confirmed_time > pos.open_time
            ]
            last_swing = matching_swings[-1] if matching_swings else None

            if current_atr is not None or last_swing is not None:
                ts_res = check_trailing_stop(
                    position=open_pos,
                    current_price=current_price,
                    spec=risk_spec,
                    config=cfg,
                    atr=current_atr,
                    structural_swing=last_swing,
                    spread=symbol_info.spread_points * symbol_info.point_size,
                )

                if ts_res.should_modify and ts_res.new_stop_loss is not None:
                    target_tp = (
                        0.0
                        if (
                            cfg.remove_tp_after_partial
                            and pos.ticket in self.partially_closed_tickets
                        )
                        else pos.take_profit
                    )
                    mod_res = await self.broker.modify_order(
                        ticket=pos.ticket,
                        stop_loss=ts_res.new_stop_loss,
                        take_profit=target_tp,
                    )
                    if mod_res.success:
                        logger.info(
                            "Trailing stop updated on position %d to %.2f",
                            pos.ticket,
                            ts_res.new_stop_loss,
                        )
                        if self.telegram_service is not None:
                            try:
                                await self.telegram_service.send_trailing_stop_updated(
                                    ticket=pos.ticket,
                                    symbol=pos.symbol,
                                    new_sl=ts_res.new_stop_loss,
                                )
                            except Exception as err:
                                logger.warning(
                                    "Failed to send Trailing Stop notification via Telegram: %s",
                                    err,
                                )

    def evaluate_unexecuted_proposals(self) -> list[JournalEntry]:
        """Evaluates hypothetical trajectory of unexecuted or expired proposals."""
        if self.journal_repo is None:
            return []
        evaluated: list[JournalEntry] = []
        entries = self.journal_repo.list_entries(limit=50)
        m5_candles = self.candle_loop.get_closed_series(Timeframe.M5)
        if not m5_candles:
            return []

        for entry in entries:
            if entry.status not in (
                ProposalStatus.EXPIRED,
                ProposalStatus.REJECTED,
                ProposalStatus.ADVISORY,
            ):
                continue
            if (
                entry.outcome in (TradeOutcome.UNEXECUTED, TradeOutcome.EXPIRED)
                and entry.outcome_data is not None
            ):
                subsequent = [c for c in m5_candles if c.time >= entry.created_at]
                if len(subsequent) >= 3:
                    out_data, outcome, loss_cause, loss_cat = self.evaluator.evaluate_unexecuted(
                        entry=entry,
                        candles=subsequent,
                        entry_price=entry.outcome_data.entry_price,
                        stop_loss=entry.outcome_data.stop_loss,
                        take_profit=entry.outcome_data.take_profit,
                    )
                    if (
                        outcome == TradeOutcome.UNEXECUTED
                        and entry.status == ProposalStatus.EXPIRED
                    ):
                        outcome = TradeOutcome.EXPIRED
                    if outcome != entry.outcome or (
                        outcome in (TradeOutcome.WIN, TradeOutcome.LOSS)
                    ):
                        entry.outcome_data = out_data
                        entry.outcome = outcome
                        entry.loss_cause = loss_cause
                        entry.loss_category = loss_cat
                        self.journal_repo.save_entry(entry)
                        evaluated.append(entry)
        return evaluated


class LiveTradingLoopRunner:
    def __init__(
        self,
        engine: LiveTradingEngine | None = None,
        watchdog: Watchdog | None = None,
        poll_interval_s: float = 2.0,
        engines: Sequence[LiveTradingEngine] | None = None,
    ) -> None:
        if engines is not None and len(engines) > 0:
            self.engines: list[LiveTradingEngine] = list(engines)
        elif engine is not None:
            self.engines = [engine]
        else:
            raise ValueError("Either engine or engines must be provided")
        self.engine = self.engines[0]
        self.watchdog = watchdog
        self.poll_interval_s = max(0.1, float(poll_interval_s))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        return (
            self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        symbols_label = ",".join(e.symbol for e in self.engines)
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name=f"LiveTradingLoop[{symbols_label}]",
        )
        self._thread.start()
        logger.info(
            "LiveTradingLoopRunner thread started for symbols=[%s] (poll_interval=%.2fs)",
            symbols_label,
            self.poll_interval_s,
        )

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        logger.info("LiveTradingLoopRunner stopped")

    def _worker(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async())
        except Exception as exc:
            logger.error("LiveTradingLoopRunner encountered fatal error: %s", exc, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                loop.close()

    async def _run_async(self) -> None:
        for engine in self.engines:
            trigger_tfs = getattr(engine, "trigger_timeframes", (engine.trigger_timeframe,))
            for tf in trigger_tfs:
                engine.candle_loop.register_handler(tf, engine.on_candle_close)

            for tf in engine.timeframes:
                try:
                    candles = await engine.broker.get_candles(engine.symbol, tf, count=100)
                    if candles:
                        engine.candle_loop.feed_candles(
                            tf, candles, has_forming_candle=True, dispatch=False
                        )
                except Exception as seed_err:
                    logger.warning(
                        "Failed initial candle seed for %s (%s): %s",
                        engine.symbol,
                        tf.value,
                        seed_err,
                    )

        while not self._stop_event.is_set():
            now = datetime.now(UTC)
            if self.watchdog is not None:
                self.watchdog.beat_loop(now=now)

            any_market_data_received = False
            for engine in self.engines:
                sym_info: SymbolInfo | None = None

                for tf in engine.timeframes:
                    try:
                        candles = await engine.broker.get_candles(engine.symbol, tf, count=10)
                        if candles:
                            any_market_data_received = True
                            engine.candle_loop.feed_candles(
                                tf, candles, has_forming_candle=True, dispatch=True
                            )
                    except Exception as feed_err:
                        logger.warning(
                            "Error fetching candles for %s %s: %s",
                            engine.symbol,
                            tf.value,
                            feed_err,
                        )

                try:
                    sym_info = await engine.broker.get_symbol_info(engine.symbol)
                    if sym_info is not None:
                        await engine.maintain_active_proposals_and_positions(
                            current_price=sym_info.bid,
                            ask_price=sym_info.ask,
                            current_time=now,
                        )
                except Exception as maint_err:
                    logger.warning(
                        "Error during position/proposal maintenance for %s: %s",
                        engine.symbol,
                        maint_err,
                    )

                if sym_info is not None:
                    any_market_data_received = True

            if self.watchdog is not None and any_market_data_received:
                self.watchdog.update_tick(now)

            sleep_chunks = max(1, int(self.poll_interval_s / 0.1))
            for _ in range(sleep_chunks):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(0.1)
