from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from xau_bot.advisor.stats import SetupHistoricalStats, SetupStatsStore
from xau_bot.domain.models import Direction, OrderIntent, TradeSetup

if TYPE_CHECKING:
    from xau_bot.llm.veto import LLMVetoDecision, LLMVetoService

logger = logging.getLogger("xau_bot.advisor.modes")


class ExecutionMode(StrEnum):
    ANALYST = "analyst"
    SEMIAUTO = "semiauto"
    AUTO = "auto"


class ProposalStatus(StrEnum):
    ADVISORY = "ADVISORY"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    DISPATCHED = "DISPATCHED"
    DISPATCH_FAILED = "DISPATCH_FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class TradeProposal:
    id: str
    setup_id: str
    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    volume: float
    risk_usd: float
    risk_pct: float
    created_at: datetime
    expires_at: datetime
    status: ProposalStatus = ProposalStatus.PENDING_APPROVAL
    mode: ExecutionMode = ExecutionMode.SEMIAUTO
    reason: str | None = None
    order_ticket: int | None = None
    dispatched_at: datetime | None = None
    setup: TradeSetup | None = None
    intent: OrderIntent | None = None
    atr: float | None = None
    historical_stats: SetupHistoricalStats | None = None
    llm_veto: LLMVetoDecision | None = None

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be strictly greater than created_at")

    def to_dict(self) -> dict[str, Any]:
        dir_label = "BUY" if self.direction == Direction.BULLISH else "SELL"
        return {
            "id": self.id,
            "setup_id": self.setup_id,
            "symbol": self.symbol,
            "direction": dir_label,
            "raw_direction": str(self.direction.value),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "volume": self.volume,
            "risk_usd": self.risk_usd,
            "risk_pct": self.risk_pct,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": str(self.status.value),
            "mode": str(self.mode.value),
            "reason": self.reason,
            "order_ticket": self.order_ticket,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "atr": self.atr,
            "historical_stats": self.historical_stats.to_dict()
            if self.historical_stats is not None
            else None,
            "llm_veto": self.llm_veto.to_dict() if self.llm_veto is not None else None,
        }


class ProposalManager:
    def __init__(
        self,
        default_timeout_s: float = 300.0,
        stats_store: SetupStatsStore | None = None,
        veto_service: LLMVetoService | None = None,
    ) -> None:
        self.default_timeout_s: float = max(1.0, float(default_timeout_s))
        self.stats_store: SetupStatsStore | None = stats_store
        self.veto_service: LLMVetoService | None = veto_service
        self._proposals: dict[str, TradeProposal] = {}
        self._lock: threading.Lock = threading.Lock()
        self._listeners: list[Callable[[TradeProposal], None]] = []

    def add_listener(self, listener: Callable[[TradeProposal], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def _notify_listeners(self, proposal: TradeProposal) -> None:
        for listener in list(self._listeners):
            try:
                listener(proposal)
            except Exception as exc:
                logger.warning("Error in proposal listener: %s", exc)

    def create_proposal(
        self,
        setup: TradeSetup,
        intent: OrderIntent | None = None,
        volume: float = 0.01,
        risk_usd: float = 0.0,
        risk_pct: float = 0.0,
        mode: ExecutionMode = ExecutionMode.SEMIAUTO,
        timeout_s: float | None = None,
        now: datetime | None = None,
        atr: float | None = None,
        historical_stats: SetupHistoricalStats | None = None,
        llm_veto: LLMVetoDecision | None = None,
    ) -> TradeProposal:
        resolved_veto = llm_veto
        if resolved_veto is None and self.veto_service is not None:
            resolved_veto = self.veto_service.evaluate_setup(setup, atr=atr)

        with self._lock:
            current_time = now or datetime.now(UTC)
            effective_timeout = timeout_s if timeout_s is not None else self.default_timeout_s
            expires_at = current_time + timedelta(seconds=max(1.0, effective_timeout))

            symbol = intent.symbol if intent is not None else "XAUUSD"
            initial_status = (
                ProposalStatus.ADVISORY
                if mode == ExecutionMode.ANALYST
                else ProposalStatus.PENDING_APPROVAL
            )

            resolved_stats = historical_stats
            if resolved_stats is None and self.stats_store is not None:
                trigger_key = str(setup.trigger_type.value)
                resolved_stats = self.stats_store.get_or_default(trigger_key)

            reject_reason: str | None = None
            if resolved_veto is not None and not resolved_veto.shadow_mode and resolved_veto.veto:
                initial_status = ProposalStatus.REJECTED
                reject_reason = f"Veto ativo pelo LLM: {resolved_veto.reason}"

            prop_id = f"prop_{uuid.uuid4().hex[:12]}"
            proposal = TradeProposal(
                id=prop_id,
                setup_id=setup.id,
                symbol=symbol,
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                volume=volume,
                risk_usd=risk_usd,
                risk_pct=risk_pct,
                created_at=current_time,
                expires_at=expires_at,
                status=initial_status,
                mode=mode,
                reason=reject_reason,
                setup=setup,
                intent=intent,
                atr=atr,
                historical_stats=resolved_stats,
                llm_veto=resolved_veto,
            )
            self._proposals[prop_id] = proposal
            logger.info(
                "Created trade proposal %s for setup %s (mode=%s, status=%s, expires_at=%s)",
                prop_id,
                setup.id,
                mode.value,
                initial_status.value,
                expires_at.isoformat(),
            )
        self._notify_listeners(proposal)
        return proposal

    def get_proposal(self, proposal_id: str) -> TradeProposal | None:
        with self._lock:
            return self._proposals.get(proposal_id)

    def get_proposals(self, status: ProposalStatus | None = None) -> list[TradeProposal]:
        with self._lock:
            if status is None:
                return list(self._proposals.values())
            return [p for p in self._proposals.values() if p.status == status]

    def get_pending_proposals(self) -> list[TradeProposal]:
        return self.get_proposals(status=ProposalStatus.PENDING_APPROVAL)

    def approve_proposal(
        self,
        proposal_id: str,
        now: datetime | None = None,
    ) -> tuple[bool, str, TradeProposal | None]:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if prop is None:
                return False, "not_found", None

            if prop.mode == ExecutionMode.ANALYST or prop.status == ProposalStatus.ADVISORY:
                logger.warning(
                    "Attempt to approve proposal %s rejected: analyst mode prohibits execution",
                    proposal_id,
                )
                return False, "invalid_mode", prop

            current_time = now or datetime.now(UTC)
            if prop.status != ProposalStatus.PENDING_APPROVAL:
                return False, "invalid_state", prop

            if current_time >= prop.expires_at:
                prop.status = ProposalStatus.EXPIRED
                prop.reason = "proposal_expired_before_approval"
                return False, "expired", prop

            prop.status = ProposalStatus.APPROVED
            logger.info("Trade proposal %s successfully approved", proposal_id)
        if prop is not None:
            self._notify_listeners(prop)
        return True, "approved", prop

    def mark_dispatched(
        self,
        proposal_id: str,
        order_ticket: int | None = None,
        now: datetime | None = None,
    ) -> TradeProposal | None:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if prop is None:
                return None
            prop.status = ProposalStatus.DISPATCHED
            prop.order_ticket = order_ticket
            prop.dispatched_at = now or datetime.now(UTC)
            logger.info(
                "Trade proposal %s marked as dispatched (ticket=%s)", proposal_id, order_ticket
            )
        if prop is not None:
            self._notify_listeners(prop)
        return prop

    def mark_dispatch_failed(
        self,
        proposal_id: str,
        reason: str = "",
    ) -> TradeProposal | None:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if prop is None:
                return None
            prop.status = ProposalStatus.DISPATCH_FAILED
            prop.reason = reason or "dispatch_failed"
            logger.warning(
                "Trade proposal %s marked as dispatch_failed (reason=%s)",
                proposal_id,
                prop.reason,
            )
        if prop is not None:
            self._notify_listeners(prop)
        return prop

    def reject_proposal(
        self,
        proposal_id: str,
        reason: str = "",
    ) -> tuple[bool, str, TradeProposal | None]:
        with self._lock:
            prop = self._proposals.get(proposal_id)
            if prop is None:
                return False, "not_found", None

            if prop.status != ProposalStatus.PENDING_APPROVAL:
                return False, "invalid_state", prop

            prop.status = ProposalStatus.REJECTED
            prop.reason = reason or "rejected_by_operator"
            logger.info("Trade proposal %s rejected: %s", proposal_id, prop.reason)
        if prop is not None:
            self._notify_listeners(prop)
        return True, "rejected", prop

    def check_expirations(self, now: datetime | None = None) -> list[TradeProposal]:
        with self._lock:
            current_time = now or datetime.now(UTC)
            expired: list[TradeProposal] = []
            for prop in self._proposals.values():
                if (
                    prop.status == ProposalStatus.PENDING_APPROVAL
                    and current_time >= prop.expires_at
                ):
                    prop.status = ProposalStatus.EXPIRED
                    prop.reason = "expired_by_timeout"
                    expired.append(prop)
                    logger.info("Trade proposal %s expired due to timeout", prop.id)
        for p in expired:
            self._notify_listeners(p)
        return expired

    def cancel_invalidated(
        self,
        setup_id: str,
        reason: str = "setup_invalidated",
    ) -> list[TradeProposal]:
        with self._lock:
            cancelled: list[TradeProposal] = []
            for prop in self._proposals.values():
                if prop.setup_id == setup_id and prop.status == ProposalStatus.PENDING_APPROVAL:
                    prop.status = ProposalStatus.CANCELLED
                    prop.reason = reason
                    cancelled.append(prop)
                    logger.info("Trade proposal %s cancelled: %s", prop.id, reason)
        for p in cancelled:
            self._notify_listeners(p)
        return cancelled

    def cancel_by_price(
        self,
        symbol: str,
        current_price: float,
        ask_price: float | None = None,
    ) -> list[TradeProposal]:
        bid_price = current_price
        ask_p = ask_price if ask_price is not None else current_price
        with self._lock:
            cancelled: list[TradeProposal] = []
            for prop in self._proposals.values():
                if prop.status != ProposalStatus.PENDING_APPROVAL:
                    continue
                if prop.symbol != symbol:
                    continue

                is_invalid = (
                    prop.direction == Direction.BULLISH and bid_price <= prop.stop_loss
                ) or (prop.direction == Direction.BEARISH and ask_p >= prop.stop_loss)

                if is_invalid:
                    prop.status = ProposalStatus.CANCELLED
                    prop.reason = "invalidation_price_reached"
                    cancelled.append(prop)
                    logger.warning(
                        "Trade proposal %s cancelled due to price invalidation "
                        "(bid=%f, ask=%f, sl=%f)",
                        prop.id,
                        bid_price,
                        ask_p,
                        prop.stop_loss,
                    )
        for p in cancelled:
            self._notify_listeners(p)
        return cancelled

    def get_all_proposals(self) -> list[TradeProposal]:
        with self._lock:
            return list(self._proposals.values())

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            stats = {
                "total": len(self._proposals),
                "advisory": 0,
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "dispatched": 0,
                "dispatch_failed": 0,
                "cancelled": 0,
            }
            for p in self._proposals.values():
                match p.status:
                    case ProposalStatus.ADVISORY:
                        stats["advisory"] += 1
                    case ProposalStatus.PENDING_APPROVAL:
                        stats["pending"] += 1
                    case ProposalStatus.APPROVED:
                        stats["approved"] += 1
                    case ProposalStatus.REJECTED:
                        stats["rejected"] += 1
                    case ProposalStatus.EXPIRED:
                        stats["expired"] += 1
                    case ProposalStatus.DISPATCHED:
                        stats["dispatched"] += 1
                    case ProposalStatus.DISPATCH_FAILED:
                        stats["dispatch_failed"] += 1
                    case ProposalStatus.CANCELLED:
                        stats["cancelled"] += 1
            return stats


class AdvisorDispatcher:
    def __init__(
        self,
        proposal_manager: ProposalManager,
        mode: ExecutionMode = ExecutionMode.SEMIAUTO,
        alert_manager: Any | None = None,
        executor: Any | None = None,
        telegram_service: Any | None = None,
        is_trading_halted_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.proposal_manager = proposal_manager
        self.mode = mode
        self.alert_manager = alert_manager
        self.executor = executor
        self.telegram_service = telegram_service
        self.is_trading_halted_fn = is_trading_halted_fn

    async def handle_setup(
        self,
        setup: TradeSetup,
        intent: OrderIntent | None = None,
        volume: float = 0.01,
        risk_usd: float = 0.0,
        risk_pct: float = 0.0,
        timeout_s: float | None = None,
        atr: float | None = None,
    ) -> TradeProposal | None:
        if self.is_trading_halted_fn and self.is_trading_halted_fn():
            logger.warning(
                "AdvisorDispatcher: setup %s rejected because trading is halted", setup.id
            )
            return None

        veto: LLMVetoDecision | None = None
        if self.proposal_manager.veto_service is not None:
            veto = await asyncio.to_thread(
                self.proposal_manager.veto_service.evaluate_setup,
                setup,
                atr=atr,
            )

        proposal = self.proposal_manager.create_proposal(
            setup=setup,
            intent=intent,
            volume=volume,
            risk_usd=risk_usd,
            risk_pct=risk_pct,
            mode=self.mode,
            timeout_s=timeout_s,
            atr=atr,
            llm_veto=veto,
        )

        dir_label = "BUY" if proposal.direction == Direction.BULLISH else "SELL"
        msg_header = ""
        alert_level = "INFO"
        validity_str = ""

        if proposal.status == ProposalStatus.REJECTED:
            msg_header = "[VETADO PELO LLM]"
            alert_level = "WARNING"
            logger.warning(
                "Proposal %s for setup %s was rejected by active LLM veto (%s); "
                "submission inhibited",
                proposal.id,
                setup.id,
                proposal.reason,
            )
        else:
            match self.mode:
                case ExecutionMode.ANALYST:
                    msg_header = "[SUGESTÃO ANALISTA]"
                    logger.info(
                        "Analyst mode: setup detected %s, order submission inhibited",
                        setup.id,
                    )

                case ExecutionMode.SEMIAUTO:
                    msg_header = "[APROVAÇÃO NECESSÁRIA]"
                    timeout_val = int((proposal.expires_at - proposal.created_at).total_seconds())
                    validity_str = (
                        f" | Validade: até {proposal.expires_at.strftime('%H:%M:%S UTC')} "
                        f"({timeout_val}s)"
                    )
                    logger.info(
                        "Semiauto mode: proposal %s created, awaiting operator approval",
                        proposal.id,
                    )

                case ExecutionMode.AUTO:
                    logger.info(
                        "Auto mode: auto-approving proposal %s for setup %s",
                        proposal.id,
                        setup.id,
                    )
                    ok, rej_reason, _ = self.proposal_manager.approve_proposal(proposal.id)
                    if not ok:
                        logger.warning(
                            "Auto mode: proposal %s could not be approved (%s), submission aborted",
                            proposal.id,
                            rej_reason,
                        )
                        msg_header = "[PROPOSTA NÃO APROVADA]"
                        alert_level = "WARNING"
                    elif self.executor is None or intent is None:
                        err_reason = "Missing executor or intent in AUTO mode"
                        self.proposal_manager.mark_dispatch_failed(proposal.id, reason=err_reason)
                        msg_header = "[FALHA NO DISPARO AUTOMÁTICO]"
                        alert_level = "ERROR"
                    else:
                        try:
                            res = await self.executor.execute_intent(intent)
                            if getattr(res, "success", False):
                                ticket = getattr(res, "ticket", None)
                                self.proposal_manager.mark_dispatched(
                                    proposal.id, order_ticket=ticket
                                )
                                msg_header = "[ORDEM AUTOMÁTICA ENVIADA]"
                            else:
                                err_msg = (
                                    getattr(res, "error_message", None)
                                    or f"Retcode {getattr(res, 'retcode', 'unknown')}"
                                )
                                self.proposal_manager.mark_dispatch_failed(
                                    proposal.id, reason=str(err_msg)
                                )
                                msg_header = "[FALHA NO DISPARO AUTOMÁTICO]"
                                alert_level = "ERROR"
                        except Exception as exc:
                            logger.error(
                                "Failed to auto-dispatch order for proposal %s: %s",
                                proposal.id,
                                exc,
                            )
                            self.proposal_manager.mark_dispatch_failed(proposal.id, reason=str(exc))
                            msg_header = "[FALHA NO DISPARO AUTOMÁTICO]"
                            alert_level = "ERROR"

        if self.telegram_service is not None:
            try:
                res_tel = self.telegram_service.send_proposal(proposal)
                if inspect.isawaitable(res_tel):
                    await res_tel
            except Exception as tel_err:
                logger.warning("Failed to emit telegram proposal for %s: %s", proposal.id, tel_err)
        elif self.alert_manager is not None:
            risk_info = f"Risco: ${proposal.risk_usd:.2f} ({proposal.risk_pct:.1f}%)"
            if alert_level in ("ERROR", "WARNING"):
                reason_str = f" | Motivo: {proposal.reason}" if proposal.reason else ""
                alert_text = (
                    f"{msg_header} {proposal.symbol} {dir_label} @ "
                    f"{proposal.entry_price:.2f}{reason_str}"
                )
            else:
                alert_text = (
                    f"{msg_header} {proposal.symbol} {dir_label} @ {proposal.entry_price:.2f} | "
                    f"SL: {proposal.stop_loss:.2f} | TP: {proposal.take_profit:.2f} | "
                    f"Lote: {proposal.volume:.2f} | {risk_info}{validity_str}"
                )
            title = f"{msg_header} {proposal.symbol}"
            try:
                if hasattr(self.alert_manager, "notify"):
                    res = self.alert_manager.notify(
                        level=alert_level,
                        title=title,
                        message=alert_text,
                        metadata=proposal.to_dict(),
                    )
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(self.alert_manager, "emit"):
                    self.alert_manager.emit(title=title, message=alert_text)
            except Exception as alert_err:
                logger.warning("Failed to emit alert for proposal %s: %s", proposal.id, alert_err)

        return proposal
