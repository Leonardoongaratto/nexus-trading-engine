import asyncio
import logging
import re
from dataclasses import replace

from xau_bot.broker.models import OrderRequest, OrderType
from xau_bot.broker.mt5_mapping import (
    RETCODE_CONNECTION,
    RETCODE_DONE,
    RETCODE_NO_CHANGES,
    RETCODE_PRICE_CHANGED,
    RETCODE_PRICE_OFF,
    RETCODE_REQUOTE,
)
from xau_bot.broker.ports import BrokerPort
from xau_bot.domain.models import Direction, OrderIntent
from xau_bot.execution.idempotency import InFlightTracker, generate_order_comment
from xau_bot.execution.models import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionStatus,
    ReconciliationResult,
    RetcodeAction,
)
from xau_bot.execution.reconciler import OrderReconciler
from xau_bot.execution.retcodes import classify_retcode

logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(
        self,
        broker: BrokerPort,
        config: ExecutionConfig | None = None,
        tracker: InFlightTracker | None = None,
        reconciler: OrderReconciler | None = None,
    ) -> None:
        self._broker = broker
        self._config = config or ExecutionConfig()
        self._tracker = tracker or InFlightTracker()
        self._reconciler = reconciler or OrderReconciler(broker)

    @staticmethod
    def _sanitize_or_generate_comment(intent: OrderIntent, config: ExecutionConfig) -> str:
        if intent.comment and re.match(r"^[A-Za-z0-9_]{1,31}$", intent.comment):
            return intent.comment
        return generate_order_comment(intent, config.comment_prefix)

    @staticmethod
    def _result_from_reconciliation(
        rec: ReconciliationResult,
        retcode: int,
        attempts: int,
        comment: str,
        magic: int,
    ) -> ExecutionResult:
        fill_price = (
            rec.position.open_price
            if rec.position is not None
            else (rec.order.price if rec.order is not None else 0.0)
        )
        fill_vol = (
            rec.position.volume
            if rec.position is not None
            else (rec.order.volume if rec.order is not None else 0.0)
        )
        return ExecutionResult(
            success=True,
            status=ExecutionStatus.RECONCILED,
            ticket=rec.ticket,
            price=fill_price,
            volume=fill_vol,
            retcode=retcode,
            reconciled=True,
            retries_attempted=attempts,
            comment=comment,
            magic=magic,
        )

    async def execute_intent(self, intent: OrderIntent) -> ExecutionResult:
        magic = intent.magic if intent.magic > 0 else self._config.magic
        comment = self._sanitize_or_generate_comment(intent, self._config)
        key = f"{magic}:{comment}"

        cached = await self._tracker.get_completed(key)
        if cached is not None:
            return cached

        acquired = await self._tracker.acquire(key)
        if not acquired:
            logger.warning(
                "Order intent %s rejected: already in-flight",
                key,
                extra={"event": "execution.already_inflight"},
            )
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.REJECTED,
                error_message=f"Order intent {key} is already in-flight",
                comment=comment,
                magic=magic,
            )

        try:
            try:
                pre_rec = await self._reconciler.reconcile_order(magic=magic, comment=comment)
                if pre_rec.found:
                    res_rec = self._result_from_reconciliation(
                        pre_rec,
                        retcode=RETCODE_DONE,
                        attempts=0,
                        comment=comment,
                        magic=magic,
                    )
                    logger.info(
                        "Order intent %s pre-reconciled with ticket %s",
                        key,
                        pre_rec.ticket,
                        extra={"event": "execution.reconciled"},
                    )
                    await self._tracker.mark_completed(key, res_rec)
                    return res_rec
            except Exception as ex:
                logger.error(
                    "Pre-reconciliation unavailable for intent %s: %s",
                    key,
                    ex,
                    extra={"event": "execution.reconciliation_failed"},
                )
                return ExecutionResult(
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error_message=f"Pre-reconciliation unavailable: {ex}",
                    comment=comment,
                    magic=magic,
                )

            order_type = OrderType(intent.order_type)
            current_price = intent.price
            dev = (
                intent.deviation
                if intent.deviation is not None
                else int(self._config.max_slippage_points)
            )
            req = OrderRequest(
                symbol=intent.symbol,
                order_type=order_type,
                volume=intent.volume,
                price=current_price,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                comment=comment,
                magic=magic,
                deviation=dev,
            )

            attempts = 0
            last_retcode = 0
            last_error = ""

            while attempts < self._config.max_retries:
                attempts += 1
                try:
                    order_res = await self._broker.send_order(req)
                    last_retcode = order_res.retcode
                    last_error = order_res.error_message
                    action = classify_retcode(last_retcode)
                except Exception as ex:
                    last_retcode = RETCODE_CONNECTION
                    last_error = str(ex)
                    action = RetcodeAction.RETRY_WITH_RECONCILIATION

                if action == RetcodeAction.SUCCESS:
                    res_success = ExecutionResult(
                        success=True,
                        status=ExecutionStatus.FILLED,
                        ticket=order_res.ticket,
                        price=order_res.price if order_res.price > 0 else current_price,
                        volume=order_res.volume if order_res.volume > 0 else intent.volume,
                        retcode=last_retcode,
                        reconciled=False,
                        retries_attempted=attempts,
                        comment=comment,
                        magic=magic,
                    )
                    logger.info(
                        "Order filled: ticket=%s retcode=%s",
                        order_res.ticket,
                        last_retcode,
                        extra={"event": "execution.filled"},
                    )
                    await self._tracker.mark_completed(key, res_success)
                    return res_success

                if action == RetcodeAction.FATAL_ABORT:
                    logger.warning(
                        "Order aborted fatally: retcode=%s error=%s",
                        last_retcode,
                        last_error,
                        extra={"event": "execution.aborted"},
                    )
                    return ExecutionResult(
                        success=False,
                        status=ExecutionStatus.ABORTED,
                        retcode=last_retcode,
                        error_message=last_error,
                        retries_attempted=attempts,
                        comment=comment,
                        magic=magic,
                    )

                try:
                    rec = await self._reconciler.reconcile_order(magic=magic, comment=comment)
                except Exception as ex:
                    logger.error(
                        "Reconciliation unavailable after retcode %s: %s",
                        last_retcode,
                        ex,
                        extra={"event": "execution.reconciliation_failed"},
                    )
                    return ExecutionResult(
                        success=False,
                        status=ExecutionStatus.FAILED,
                        retcode=last_retcode,
                        error_message=f"Reconciliation unavailable: {ex}",
                        retries_attempted=attempts,
                        comment=comment,
                        magic=magic,
                    )

                if rec.found:
                    res_reconciled = self._result_from_reconciliation(
                        rec,
                        retcode=last_retcode,
                        attempts=attempts,
                        comment=comment,
                        magic=magic,
                    )
                    logger.info(
                        "Order reconciled after error: ticket=%s",
                        rec.ticket,
                        extra={"event": "execution.reconciled"},
                    )
                    await self._tracker.mark_completed(key, res_reconciled)
                    return res_reconciled

                if attempts >= self._config.max_retries:
                    logger.warning(
                        "Max retries (%s) exceeded after retcode %s: %s",
                        self._config.max_retries,
                        last_retcode,
                        last_error,
                        extra={"event": "execution.max_retries_exceeded"},
                    )
                    return ExecutionResult(
                        success=False,
                        status=ExecutionStatus.FAILED,
                        retcode=last_retcode,
                        error_message=(
                            f"Max retries ({self._config.max_retries}) exceeded after "
                            f"retcode {last_retcode}: {last_error}"
                        ),
                        retries_attempted=attempts,
                        comment=comment,
                        magic=magic,
                    )

                if last_retcode in (
                    RETCODE_REQUOTE,
                    RETCODE_PRICE_CHANGED,
                    RETCODE_PRICE_OFF,
                ) and order_type in (OrderType.BUY, OrderType.SELL):
                    sym_info = await self._broker.get_symbol_info(intent.symbol)
                    if sym_info is None:
                        return ExecutionResult(
                            success=False,
                            status=ExecutionStatus.FAILED,
                            retcode=last_retcode,
                            error_message="Symbol info unavailable for price update",
                            retries_attempted=attempts,
                            comment=comment,
                            magic=magic,
                        )

                    is_buy = intent.direction == Direction.BULLISH
                    new_price = sym_info.ask if is_buy else sym_info.bid
                    point_size = sym_info.point_size if sym_info.point_size > 0 else 0.01
                    slippage_points = abs(new_price - intent.price) / point_size

                    if slippage_points > self._config.max_slippage_points:
                        logger.warning(
                            "Slippage %s points exceeds maximum allowed %s points",
                            slippage_points,
                            self._config.max_slippage_points,
                            extra={"event": "execution.slippage_abort"},
                        )
                        return ExecutionResult(
                            success=False,
                            status=ExecutionStatus.ABORTED,
                            retcode=last_retcode,
                            error_message=(
                                f"Slippage {slippage_points:.1f} points exceeds "
                                f"maximum allowed ({self._config.max_slippage_points:.1f})"
                            ),
                            retries_attempted=attempts,
                            comment=comment,
                            magic=magic,
                        )
                    current_price = new_price
                    req = replace(req, price=new_price)

                logger.info(
                    "Retrying order attempt %s after retcode %s",
                    attempts + 1,
                    last_retcode,
                    extra={"event": "execution.retry"},
                )
                await asyncio.sleep(self._config.retry_delay_seconds)

            return ExecutionResult(
                success=False,
                status=ExecutionStatus.FAILED,
                retcode=last_retcode,
                error_message=f"Execution failed after {attempts} attempts",
                retries_attempted=attempts,
                comment=comment,
                magic=magic,
            )
        finally:
            await self._tracker.release(key)

    async def modify_position(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> ExecutionResult:
        try:
            res = await self._broker.modify_order(
                ticket=ticket,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if res.success:
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.FILLED,
                    ticket=ticket,
                    retcode=res.retcode,
                )
            retcode = res.retcode
            err_msg = res.error_message
        except Exception as ex:
            retcode = RETCODE_CONNECTION
            err_msg = str(ex)

        action = classify_retcode(retcode)
        if action == RetcodeAction.FATAL_ABORT and retcode != RETCODE_NO_CHANGES:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.ABORTED,
                ticket=ticket,
                retcode=retcode,
                error_message=err_msg,
            )

        try:
            positions = await self._broker.get_open_positions()
            for pos in positions:
                if pos.ticket == ticket:
                    sl_ok = (stop_loss is None) or (
                        pos.stop_loss is not None and abs(pos.stop_loss - stop_loss) < 1e-4
                    )
                    tp_ok = (take_profit is None) or (
                        pos.take_profit is not None and abs(pos.take_profit - take_profit) < 1e-4
                    )
                    if sl_ok and tp_ok:
                        return ExecutionResult(
                            success=True,
                            status=ExecutionStatus.RECONCILED,
                            ticket=ticket,
                            reconciled=True,
                            retcode=retcode,
                        )
        except Exception as ex:
            logger.warning(
                "Modify position reconciliation unavailable: %s",
                ex,
                extra={"event": "execution.reconciliation_failed"},
            )

        return ExecutionResult(
            success=False,
            status=ExecutionStatus.FAILED,
            ticket=ticket,
            retcode=retcode,
            error_message=err_msg,
        )

    async def close_position(
        self,
        ticket: int,
        volume: float | None = None,
    ) -> ExecutionResult:
        volume_before = volume or 0.0
        try:
            positions_before = await self._broker.get_open_positions()
            pos_before = next((p for p in positions_before if p.ticket == ticket), None)
            if pos_before is not None:
                volume_before = pos_before.volume
        except Exception as ex:
            logger.warning(
                "Error querying positions before close: %s",
                ex,
                extra={"event": "execution.query_failed"},
            )

        try:
            res = await self._broker.close_position(ticket=ticket, volume=volume)
            if res.success:
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.FILLED,
                    ticket=ticket,
                    retcode=res.retcode,
                )
            retcode = res.retcode
            err_msg = res.error_message
        except Exception as ex:
            retcode = RETCODE_CONNECTION
            err_msg = str(ex)

        action = classify_retcode(retcode)
        if action == RetcodeAction.FATAL_ABORT:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.ABORTED,
                ticket=ticket,
                retcode=retcode,
                error_message=err_msg,
            )

        try:
            positions = await self._broker.get_open_positions()
            pos_found = next((p for p in positions if p.ticket == ticket), None)
            if pos_found is None:
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.RECONCILED,
                    ticket=ticket,
                    reconciled=True,
                    retcode=retcode,
                )

            if volume is not None and pos_found.volume <= (volume_before - volume) + 1e-5:
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.RECONCILED,
                    ticket=ticket,
                    reconciled=True,
                    retcode=retcode,
                )
        except Exception as ex:
            logger.warning(
                "Close position reconciliation unavailable: %s",
                ex,
                extra={"event": "execution.reconciliation_failed"},
            )

        return ExecutionResult(
            success=False,
            status=ExecutionStatus.FAILED,
            ticket=ticket,
            retcode=retcode,
            error_message=err_msg,
        )

    async def cancel_order(self, ticket: int) -> ExecutionResult:
        try:
            res = await self._broker.cancel_order(ticket=ticket)
            if res.success:
                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.FILLED,
                    ticket=ticket,
                    retcode=res.retcode,
                )
            retcode = res.retcode
            err_msg = res.error_message
        except Exception as ex:
            retcode = RETCODE_CONNECTION
            err_msg = str(ex)

        action = classify_retcode(retcode)
        if action == RetcodeAction.FATAL_ABORT:
            return ExecutionResult(
                success=False,
                status=ExecutionStatus.ABORTED,
                ticket=ticket,
                retcode=retcode,
                error_message=err_msg,
            )

        try:
            orders = await self._broker.get_open_orders()
            order_found = next((o for o in orders if o.ticket == ticket), None)
            if order_found is None:
                positions = await self._broker.get_open_positions()
                pos_found = next((p for p in positions if p.ticket == ticket), None)
                if pos_found is not None:
                    logger.warning(
                        "Order %s was triggered and filled as position, not canceled",
                        ticket,
                        extra={"event": "execution.order_triggered"},
                    )
                    return ExecutionResult(
                        success=False,
                        status=ExecutionStatus.FAILED,
                        ticket=ticket,
                        retcode=retcode,
                        error_message="Order was triggered and filled as position, not canceled",
                    )

                return ExecutionResult(
                    success=True,
                    status=ExecutionStatus.RECONCILED,
                    ticket=ticket,
                    reconciled=True,
                    retcode=retcode,
                )
        except Exception as ex:
            logger.warning(
                "Cancel order reconciliation unavailable: %s",
                ex,
                extra={"event": "execution.reconciliation_failed"},
            )

        return ExecutionResult(
            success=False,
            status=ExecutionStatus.FAILED,
            ticket=ticket,
            retcode=retcode,
            error_message=err_msg,
        )
