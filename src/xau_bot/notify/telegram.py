"""Telegram Bot client, command handler, and interactive polling service."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import SecretStr

from xau_bot.advisor.modes import ExecutionMode, ProposalManager, ProposalStatus, TradeProposal
from xau_bot.broker.ports import BrokerPort
from xau_bot.domain.models import Direction, OrderIntent
from xau_bot.execution.models import ExecutionResult
from xau_bot.notify.models import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    TelegramMessage,
    TelegramUpdate,
)

logger = logging.getLogger(__name__)

LEGACY_MARKDOWN_SPECIAL_CHARS = "_*`["


def escape_markdown(text: str) -> str:
    escaped = []
    for char in text:
        if char in LEGACY_MARKDOWN_SPECIAL_CHARS:
            escaped.append(f"\\{char}")
        else:
            escaped.append(char)
    return "".join(escaped)


def format_proposal_message(proposal: TradeProposal) -> str:
    direction_emoji = "🟢" if proposal.direction == Direction.BULLISH else "🔴"
    direction_text = "COMPRA (BUY)" if proposal.direction == Direction.BULLISH else "VENDA (SELL)"

    if proposal.mode == ExecutionMode.ANALYST:
        mode_header = "💡 *SUGESTÃO DE ANÁLISE SMC* 💡\n_(Modo Analista - Sem Execução Automática)_"
    elif proposal.mode == ExecutionMode.SEMIAUTO:
        mode_header = (
            "⚡ *PROPOSTA DE TRADE PENDENTE* ⚡\n_(Modo Semiautomático - Aguardando Aprovação)_"
        )
    elif proposal.status == ProposalStatus.DISPATCH_FAILED:
        mode_header = "🤖 *FALHA NO ENVIO AUTOMÁTICO* ⚠️\n_(Modo Automático - Erro ao Despachar)_"
    else:
        mode_header = "🤖 *ORDEM AUTOMÁTICA ENVIADA* 🤖\n_(Modo Automático - Execução no Broker)_"

    status_map = {
        ProposalStatus.ADVISORY: "ℹ️ Informativa",
        ProposalStatus.PENDING_APPROVAL: "⏳ Aguardando Aprovação",
        ProposalStatus.APPROVED: "✅ Aprovada pelo Operador",
        ProposalStatus.REJECTED: "❌ Rejeitada pelo Operador",
        ProposalStatus.EXPIRED: "⏱️ Expirada por Timeout",
        ProposalStatus.DISPATCHED: "🚀 Enviada ao Broker",
        ProposalStatus.DISPATCH_FAILED: "⚠️ Falha no Envio ao Broker",
        ProposalStatus.CANCELLED: "🚫 Cancelada (Invalidação de Preço)",
    }
    status_text = status_map.get(proposal.status, str(proposal.status))

    risk_points = abs(proposal.entry_price - proposal.stop_loss)
    reward_points = abs(proposal.take_profit - proposal.entry_price)
    rr_ratio = reward_points / risk_points if risk_points > 0 else 0.0

    lines = [
        mode_header,
        "",
        f"• *ID Proposta:* `{proposal.id}`",
        f"• *Ativo:* `{proposal.symbol}`",
        f"• *Direção:* {direction_emoji} *{direction_text}*",
        f"• *Preço Entrada:* `{proposal.entry_price:.2f}`",
        f"• *Stop Loss:* `{proposal.stop_loss:.2f}`",
        f"• *Take Profit:* `{proposal.take_profit:.2f}`",
        f"• *Risco / Retorno:* `1:{rr_ratio:.2f}`",
        f"• *Risco Estimado:* `${proposal.risk_usd:.2f} ({proposal.risk_pct * 100:.1f}%)`",
        f"• *Lote Calculado:* `{proposal.volume:.2f}`",
        f"• *Validade:* `{proposal.expires_at.strftime('%H:%M:%S UTC')}`",
        f"• *Status:* *{status_text}*",
    ]

    if proposal.mode == ExecutionMode.ANALYST and proposal.atr is not None:
        lines.append(f"• *ATR:* `{proposal.atr:.2f}`")

    if proposal.setup and proposal.setup.confluences:
        confl_str = ", ".join(proposal.setup.confluences)
        lines.append(f"• *Confluências:* `{confl_str}`")

    if proposal.order_ticket:
        protection = " (SL/TP Confirmados)" if proposal.status == ProposalStatus.DISPATCHED else ""
        lines.append(f"• *Ticket Broker:* `#{proposal.order_ticket}`{protection}")

    if proposal.historical_stats is not None:
        hs = proposal.historical_stats
        lines.append(
            f"• *Estatística Histórica (N={hs.sample_size}):* "
            f"Win Rate: `{hs.win_rate * 100:.1f}%` | "
            f"Exp: `{hs.expectancy_r:+.2f} R` | "
            f"PF: `{hs.profit_factor:.2f}`"
        )
        if hs.validation_warning:
            clean_warn = escape_markdown(hs.validation_warning)
            lines.append(f"⚠️ _{clean_warn}_")

    if proposal.reason:
        clean_reason = escape_markdown(proposal.reason.replace("_", " "))
        lines.append(f"• *Observação:* _{clean_reason}_")

    return "\n".join(lines)


def create_proposal_keyboard(proposal_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Executar Ordem",
                    callback_data=f"approve:{proposal_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Ignorar / Rejeitar",
                    callback_data=f"reject:{proposal_id}",
                ),
            ]
        ]
    )


class TelegramBotClient:
    def __init__(
        self,
        token: SecretStr | str,
        chat_id: SecretStr | str,
        base_url: str = "https://api.telegram.org",
        timeout_s: float = 10.0,
    ) -> None:
        self._token: str = token.get_secret_value() if isinstance(token, SecretStr) else str(token)
        self._chat_id: str = (
            chat_id.get_secret_value() if isinstance(chat_id, SecretStr) else str(chat_id)
        )
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    @property
    def chat_id(self) -> str:
        return self._chat_id

    @property
    def is_configured(self) -> bool:
        return bool(self._token and self._chat_id)

    async def send_message(
        self,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        chat_id: str | int | None = None,
        parse_mode: str | None = "Markdown",
    ) -> TelegramMessage | None:
        if not self.is_configured:
            return None

        target_chat = str(chat_id) if chat_id is not None else self._chat_id
        url = f"{self._base_url}/bot{self._token}/sendMessage"

        payload: dict[str, Any] = {
            "chat_id": target_chat,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup.to_dict()

        def _do_request() -> TelegramMessage | None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    if res_json.get("ok") and "result" in res_json:
                        return TelegramMessage.from_dict(res_json["result"])
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and parse_mode is not None:
                    try:
                        raw_payload = dict(payload)
                        raw_payload.pop("parse_mode", None)
                        raw_data = json.dumps(raw_payload, allow_nan=False).encode("utf-8")
                        raw_req = urllib.request.Request(
                            url,
                            data=raw_data,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(raw_req, timeout=self._timeout_s) as resp:
                            res_json = json.loads(resp.read().decode("utf-8"))
                            if res_json.get("ok") and "result" in res_json:
                                return TelegramMessage.from_dict(res_json["result"])
                    except Exception as retry_exc:
                        logger.warning("Retry without parse_mode failed: %s", retry_exc)
                logger.warning("Telegram sendMessage HTTP error: %s", exc)
            except Exception as exc:
                logger.warning("Telegram sendMessage error: %s", exc)
            return None

        return await asyncio.to_thread(_do_request)

    async def edit_message_text(
        self,
        message_id: int,
        text: str,
        reply_markup: InlineKeyboardMarkup | None = None,
        chat_id: str | int | None = None,
        parse_mode: str | None = "Markdown",
    ) -> bool:
        if not self.is_configured:
            return False

        target_chat = str(chat_id) if chat_id is not None else self._chat_id
        url = f"{self._base_url}/bot{self._token}/editMessageText"

        payload: dict[str, Any] = {
            "chat_id": target_chat,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup.to_dict()
        else:
            payload["reply_markup"] = {"inline_keyboard": []}

        def _do_request() -> bool:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    return bool(res_json.get("ok", False))
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and parse_mode is not None:
                    try:
                        raw_payload = dict(payload)
                        raw_payload.pop("parse_mode", None)
                        raw_data = json.dumps(raw_payload, allow_nan=False).encode("utf-8")
                        raw_req = urllib.request.Request(
                            url,
                            data=raw_data,
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(raw_req, timeout=self._timeout_s) as resp:
                            res_json = json.loads(resp.read().decode("utf-8"))
                            return bool(res_json.get("ok", False))
                    except Exception as retry_exc:
                        logger.warning(
                            "Retry editMessageText without parse_mode failed: %s", retry_exc
                        )
                logger.warning("Telegram editMessageText HTTP error: %s", exc)
                return False
            except Exception as exc:
                logger.warning("Telegram editMessageText error: %s", exc)
                return False

        return await asyncio.to_thread(_do_request)

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
        show_alert: bool = False,
    ) -> bool:
        if not self.is_configured:
            return False

        url = f"{self._base_url}/bot{self._token}/answerCallbackQuery"
        payload = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert,
        }

        def _do_request() -> bool:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    return bool(res_json.get("ok", False))
            except Exception as exc:
                logger.warning("Telegram answerCallbackQuery error: %s", exc)
                return False

        return await asyncio.to_thread(_do_request)

    async def get_updates(
        self,
        offset: int = 0,
        timeout: int = 10,
    ) -> list[TelegramUpdate]:
        if not self.is_configured:
            return []

        url = f"{self._base_url}/bot{self._token}/getUpdates"
        payload = {
            "offset": offset,
            "timeout": timeout,
            "allowed_updates": ["message", "callback_query"],
        }

        def _do_request() -> list[TelegramUpdate]:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                # HTTP timeout must be slightly larger than polling timeout
                with urllib.request.urlopen(req, timeout=timeout + 5.0) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    if res_json.get("ok") and "result" in res_json:
                        return [
                            TelegramUpdate.from_dict(item)
                            for item in res_json["result"]
                            if isinstance(item, dict)
                        ]
            except Exception as exc:
                logger.debug("Telegram getUpdates poll returned or timed out: %s", exc)
            return []

        return await asyncio.to_thread(_do_request)


class OrderExecutorProtocol(Protocol):
    async def execute_intent(self, intent: OrderIntent) -> ExecutionResult: ...


class TelegramCommandHandler:
    def __init__(
        self,
        bot_client: TelegramBotClient,
        authorized_chat_id: str,
        proposal_manager: ProposalManager | None = None,
        order_executor: OrderExecutorProtocol | None = None,
        broker: BrokerPort | None = None,
        get_status_fn: Callable[[], Mapping[str, Any]] | None = None,
        set_trading_halt_fn: Callable[[bool, str], None] | None = None,
        is_trading_halted_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.client = bot_client
        self.authorized_chat_id = str(authorized_chat_id)
        self.proposal_manager = proposal_manager
        self.order_executor = order_executor
        self.broker = broker
        self.get_status_fn = get_status_fn
        self.set_trading_halt_fn = set_trading_halt_fn
        self.is_trading_halted_fn = is_trading_halted_fn

    def is_authorized(self, chat_id: int | str) -> bool:
        return str(chat_id) == self.authorized_chat_id

    async def handle_update(self, update: TelegramUpdate) -> None:
        if update.callback_query is not None:
            await self._process_callback_query(update.callback_query)
        elif update.message is not None and update.message.text is not None:
            await self._process_message(update.message)

    async def _process_message(self, message: TelegramMessage) -> None:
        chat_id = str(message.chat.id)
        if not self.is_authorized(chat_id):
            logger.warning(
                "Unauthorized Telegram access attempt from chat_id=%s",
                chat_id,
                extra={"event": "security_unauthorized_telegram_command", "chat_id": chat_id},
            )
            return

        text = (message.text or "").strip()
        cmd = text.split()[0].lower() if text else ""

        if cmd in ("/ajuda", "/help", "/start"):
            await self._cmd_help(message)
        elif cmd in ("/status", "/stats"):
            await self._cmd_status(message)
        elif cmd in ("/posicoes", "/positions"):
            await self._cmd_positions(message)
        elif cmd in ("/pausar", "/pause"):
            await self._cmd_pause(message)
        elif cmd in ("/retomar", "/resume"):
            await self._cmd_resume(message)
        elif cmd in ("/fechar_tudo", "/close_all"):
            await self._cmd_close_all(message)
        elif cmd in ("/propostas", "/proposals"):
            await self._cmd_proposals(message)
        else:
            await self.client.send_message(
                text=(
                    "❓ *Comando não reconhecido.*\n"
                    "Envie `/ajuda` para ver a lista de comandos disponíveis."
                ),
                chat_id=chat_id,
            )

    async def _cmd_help(self, message: TelegramMessage) -> None:
        help_text = (
            "🤖 *Comandos do Robô Institucional XAU/USD:*\n\n"
            "• `/status` - Telemetria e estado operacional em tempo real\n"
            "• `/posicoes` - Lista detalhada de posições abertas no broker\n"
            "• `/propostas` - Lista propostas de trade pendentes de aprovação\n"
            "• `/pausar` - Pausa abertura de novas ordens (mantém posições)\n"
            "• `/retomar` - Restaura operações normais de trading\n"
            "• `/fechar_tudo` - 🚨 *Kill Switch:* Fecha posições a mercado e paralisa o bot\n"
            "• `/ajuda` - Exibe esta mensagem de ajuda"
        )
        await self.client.send_message(text=help_text, chat_id=message.chat.id)

    async def _cmd_status(self, message: TelegramMessage) -> None:
        status_data: Mapping[str, Any] = {}
        if self.get_status_fn:
            try:
                status_data = self.get_status_fn()
            except Exception as exc:
                logger.warning("Error getting status dict: %s", exc)

        broker_connected = status_data.get("broker_connected", False)
        trading_halted = (
            self.is_trading_halted_fn()
            if self.is_trading_halted_fn
            else status_data.get("trading_halted", False)
        )
        mode = status_data.get("execution_mode", "N/A")
        daily_pnl = float(status_data.get("daily_pnl", 0.0))
        kill_switches_active = bool(status_data.get("kill_switches_active", False))
        proposals_stat = status_data.get("proposals", {})

        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        broker_icon = "🟢 Conectado" if broker_connected else "🔴 Desconectado"
        halt_icon = "🛑 HALTED / PAUSADO" if trading_halted else "🟢 ATIVO"
        kill_switch_icon = "🚨 ATIVADO" if kill_switches_active else "🟢 NORMAL"

        status_text = (
            "📊 *STATUS OPERACIONAL DO ROBÔ*\n\n"
            f"• *Modo de Execução:* `{mode.upper()}`\n"
            f"• *Broker:* {broker_icon}\n"
            f"• *Estado Operacional:* {halt_icon}\n"
            f"• *Kill Switches:* {kill_switch_icon}\n"
            f"• *PnL do Dia:* {pnl_emoji} `${daily_pnl:.2f}`\n\n"
            "📋 *Propostas de Trade:*\n"
            f"• Pendentes: `{proposals_stat.get('pending', 0)}`\n"
            f"• Aprovadas: `{proposals_stat.get('approved', 0)}`\n"
            f"• Rejeitadas: `{proposals_stat.get('rejected', 0)}`\n"
            f"• Despachadas: `{proposals_stat.get('dispatched', 0)}`\n"
            f"• Expiradas: `{proposals_stat.get('expired', 0)}`\n"
            f"• Falhas: `{proposals_stat.get('dispatch_failed', 0)}`"
        )
        await self.client.send_message(text=status_text, chat_id=message.chat.id)

    async def _cmd_positions(self, message: TelegramMessage) -> None:
        if not self.broker:
            await self.client.send_message(
                text="⚠️ *Broker não configurado ou indisponível.*",
                chat_id=message.chat.id,
            )
            return

        try:
            positions = await self.broker.get_open_positions()
        except Exception as exc:
            logger.warning("Failed to get open positions: %s", exc)
            await self.client.send_message(
                text=(
                    "⚠️ *Erro ao consultar posições no broker.* Verifique a conexão com o terminal."
                ),
                chat_id=message.chat.id,
            )
            return

        if not positions:
            await self.client.send_message(
                text="ℹ️ *Nenhuma posição aberta no momento.*",
                chat_id=message.chat.id,
            )
            return

        lines = ["📈 *POSIÇÕES ABERTAS NO BROKER:*\n"]
        for p in positions:
            pnl_icon = "🟢" if p.floating_pnl >= 0 else "🔴"
            dir_str = "BUY" if p.direction == Direction.BULLISH else "SELL"
            sl_str = f"{p.stop_loss:.2f}" if p.stop_loss is not None else "N/A"
            tp_str = f"{p.take_profit:.2f}" if p.take_profit is not None else "N/A"
            lines.append(
                f"• *#{p.ticket}* {p.symbol} | *{dir_str}* `{p.volume:.2f}` lotes\n"
                f"  Abertura: `{p.open_price:.2f}` | Atual: `{p.current_price:.2f}`\n"
                f"  SL: `{sl_str}` | TP: `{tp_str}`\n"
                f"  Lucro: {pnl_icon} `${p.floating_pnl:.2f}`\n"
            )

        await self.client.send_message(text="\n".join(lines), chat_id=message.chat.id)

    async def _cmd_pause(self, message: TelegramMessage) -> None:
        if self.set_trading_halt_fn:
            self.set_trading_halt_fn(True, "pausado_manualmente_via_telegram")
            await self.client.send_message(
                text=(
                    "⏸️ *TRADING PAUSADO COM SUCESSO!*\n\n"
                    "O robô não abrirá novas ordens nem propostas.\n"
                    "As ordens e posições abertas continuam protegidas por SL/TP.\n"
                    "Para reativar, envie `/retomar`."
                ),
                chat_id=message.chat.id,
            )
        else:
            await self.client.send_message(
                text="⚠️ *Função de controle de pausa não disponível.*",
                chat_id=message.chat.id,
            )

    async def _cmd_resume(self, message: TelegramMessage) -> None:
        if self.get_status_fn:
            try:
                status_data = self.get_status_fn()
                if status_data.get("kill_switches_active", False):
                    await self.client.send_message(
                        text=(
                            "🚨 *RETOMADA BLOQUEADA: KILL SWITCH ATIVO!*\n\n"
                            "O robô atingiu limites de risco operacionais. "
                            "A retomada automática é proibida por segurança."
                        ),
                        chat_id=message.chat.id,
                    )
                    return
            except Exception as exc:
                logger.warning("Falha ao obter status para comando /retomar: %s", exc)
                await self.client.send_message(
                    text="⚠️ *Não foi possível verificar o estado operacional para retomada.*",
                    chat_id=message.chat.id,
                )
                return

        if self.set_trading_halt_fn:
            self.set_trading_halt_fn(False, "retomado_manualmente_via_telegram")
            await self.client.send_message(
                text=(
                    "▶️ *TRADING RETOMADO!*\n\n"
                    "O robô está ativo e monitorando novas oportunidades de entrada."
                ),
                chat_id=message.chat.id,
            )
        else:
            await self.client.send_message(
                text="⚠️ *Função de controle de pausa não disponível.*",
                chat_id=message.chat.id,
            )

    async def _cmd_close_all(self, message: TelegramMessage) -> None:
        if not self.broker:
            await self.client.send_message(
                text="⚠️ *Broker não disponível para fechamento de emergência.*",
                chat_id=message.chat.id,
            )
            return

        halt_activated = False
        if self.set_trading_halt_fn:
            self.set_trading_halt_fn(True, "kill_switch_fechar_tudo_telegram")
            halt_activated = True
        elif self.is_trading_halted_fn and self.is_trading_halted_fn():
            halt_activated = True

        closed_count = 0
        failed_count = 0
        pos_error = False

        try:
            positions = await self.broker.get_open_positions()
        except Exception as exc:
            pos_error = True
            positions = ()
            logger.error("Error querying positions in close_all: %s", exc)

        for pos in positions:
            try:
                res = await self.broker.close_position(pos.ticket)
                if res.success:
                    closed_count += 1
                else:
                    failed_count += 1
                    logger.error(
                        "Kill switch failed to close position %s: %s",
                        pos.ticket,
                        res.error_message,
                    )
            except Exception as exc:
                failed_count += 1
                pos_error = True
                logger.error("Exception closing position %s in close_all: %s", pos.ticket, exc)

        cancelled_orders = 0
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            open_orders = ()
            logger.warning("Error querying open orders during close_all: %s", exc)

        for ord_item in open_orders:
            try:
                res = await self.broker.cancel_order(ord_item.ticket)
                if res.success:
                    cancelled_orders += 1
                else:
                    logger.warning(
                        "Failed to cancel order %s during close_all: %s",
                        ord_item.ticket,
                        res.error_message,
                    )
            except Exception as exc:
                logger.warning(
                    "Exception cancelling order %s during close_all: %s",
                    ord_item.ticket,
                    exc,
                )

        halt_status_str = (
            "🛑 *TRADING PARALISADO*"
            if halt_activated
            else "⚠️ *HALT INDISPONÍVEL (Apenas Fechamento Realizado)*"
        )

        header = (
            "🚨 *FALHA NO FECHAMENTO DE EMERGÊNCIA:* Erro ao consultar ou encerrar posições.\n\n"
            if (pos_error or failed_count > 0)
            else "🚨 *KILL SWITCH EXECUTADO COM SUCESSO!* 🚨\n\n"
        )

        footer = (
            "O robô está seguro em halt. Para reativar, avalie a situação e envie `/retomar`."
            if halt_activated
            else "⚠️ Atenção: O halt não pôde ser ativado automaticamente. Verifique os serviços."
        )

        summary_msg = (
            f"{header}"
            f"• Posições Encerradas: `{closed_count}`\n"
            f"• Falhas no Fechamento: `{failed_count}`\n"
            f"• Ordens Limite Canceladas: `{cancelled_orders}`\n"
            f"• Estado do Bot: {halt_status_str}\n\n"
            f"{footer}"
        )
        await self.client.send_message(text=summary_msg, chat_id=message.chat.id)

    async def _cmd_proposals(self, message: TelegramMessage) -> None:
        if not self.proposal_manager:
            await self.client.send_message(
                text="ℹ️ *Gerenciador de propostas não disponível.*",
                chat_id=message.chat.id,
            )
            return

        if hasattr(self.proposal_manager, "get_pending_proposals"):
            pending = self.proposal_manager.get_pending_proposals()
        elif hasattr(self.proposal_manager, "get_proposals"):
            pending = self.proposal_manager.get_proposals(status=ProposalStatus.PENDING_APPROVAL)
        else:
            pending = []
        if not pending:
            await self.client.send_message(
                text="ℹ️ *Nenhuma proposta pendente no momento.*",
                chat_id=message.chat.id,
            )
            return

        for prop in pending:
            msg_text = format_proposal_message(prop)
            keyboard = create_proposal_keyboard(prop.id)
            await self.client.send_message(
                text=msg_text,
                reply_markup=keyboard,
                chat_id=message.chat.id,
            )

    async def _process_callback_query(self, cb: CallbackQuery) -> None:
        chat_id = (
            str(cb.message.chat.id) if cb.message and cb.message.chat else str(cb.from_user.id)
        )
        user_id = str(cb.from_user.id)
        if not self.is_authorized(chat_id) and not self.is_authorized(user_id):
            logger.warning(
                "Unauthorized callback query from user_id=%s, chat_id=%s",
                user_id,
                chat_id,
                extra={"event": "security_unauthorized_telegram_command", "user_id": user_id},
            )
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="Acesso não autorizado.",
                show_alert=True,
            )
            return

        data = cb.data or ""
        if data.startswith("approve:"):
            proposal_id = data.split(":", 1)[1]
            await self._handle_approve_callback(cb, proposal_id)
        elif data.startswith("reject:"):
            proposal_id = data.split(":", 1)[1]
            await self._handle_reject_callback(cb, proposal_id)
        else:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="Comando não reconhecido.",
                show_alert=False,
            )

    async def _handle_approve_callback(self, cb: CallbackQuery, proposal_id: str) -> None:
        if not self.proposal_manager:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="Gerenciador de propostas não disponível.",
                show_alert=True,
            )
            return

        if self.is_trading_halted_fn and self.is_trading_halted_fn():
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="⚠️ Operação bloqueada: Trading está em HALT.",
                show_alert=True,
            )
            return

        ok, reason, proposal = self.proposal_manager.approve_proposal(proposal_id)
        if not ok or proposal is None:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text=f"Não foi possível aprovar: {reason}",
                show_alert=True,
            )
            if cb.message:
                current_prop = self.proposal_manager.get_proposal(proposal_id)
                if current_prop:
                    await self.client.edit_message_text(
                        message_id=cb.message.message_id,
                        text=format_proposal_message(current_prop),
                        chat_id=cb.message.chat.id,
                    )
            return

        await self.client.answer_callback_query(
            callback_query_id=cb.id,
            text="Proposta aprovada! Despachando ordem...",
            show_alert=False,
        )

        if self.order_executor and proposal.intent:
            try:
                result = await self.order_executor.execute_intent(proposal.intent)
                if result.success:
                    self.proposal_manager.mark_dispatched(
                        proposal.id,
                        order_ticket=result.ticket,
                    )
                else:
                    err_msg = result.error_message or "Execution failed"
                    retcode = getattr(result, "retcode", "unknown")
                    self.proposal_manager.mark_dispatch_failed(
                        proposal.id,
                        reason=f"Broker rejected: {retcode} ({err_msg})",
                    )
            except Exception as exc:
                logger.error("Error executing approved order %s: %s", proposal.id, exc)
                self.proposal_manager.mark_dispatch_failed(
                    proposal.id,
                    reason="Falha interna na execução da ordem",
                )

        latest = self.proposal_manager.get_proposal(proposal_id)
        if latest and cb.message:
            await self.client.edit_message_text(
                message_id=cb.message.message_id,
                text=format_proposal_message(latest),
                chat_id=cb.message.chat.id,
            )

    async def _handle_reject_callback(self, cb: CallbackQuery, proposal_id: str) -> None:
        if not self.proposal_manager:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="Gerenciador de propostas não disponível.",
                show_alert=True,
            )
            return

        ok, reason, proposal = self.proposal_manager.reject_proposal(
            proposal_id,
            reason="Rejeitado pelo operador via Telegram",
        )
        if not ok or proposal is None:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text=f"Não foi possível rejeitar: {reason}",
                show_alert=True,
            )
        else:
            await self.client.answer_callback_query(
                callback_query_id=cb.id,
                text="Proposta rejeitada com sucesso.",
                show_alert=False,
            )

        latest = self.proposal_manager.get_proposal(proposal_id)
        if latest and cb.message:
            await self.client.edit_message_text(
                message_id=cb.message.message_id,
                text=format_proposal_message(latest),
                chat_id=cb.message.chat.id,
            )


class TelegramService:
    def __init__(
        self,
        bot_client: TelegramBotClient,
        command_handler: TelegramCommandHandler,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.client = bot_client
        self.handler = command_handler
        self.poll_interval_s = poll_interval_s
        self._proposal_messages: dict[str, int] = {}
        self._polling_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._offset = 0

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if not self.client.is_configured:
            logger.info("TelegramService not configured. Skipping startup.")
            return

        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._polling_task = asyncio.create_task(self._poll_loop())
        logger.info("TelegramService background polling started.")

    async def stop(self) -> None:
        self._running = False
        if self._polling_task:
            self._polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._polling_task
            self._polling_task = None
        self._loop = None
        logger.info("TelegramService stopped.")

    async def send_proposal(self, proposal: TradeProposal) -> TelegramMessage | None:
        if not self.client.is_configured:
            return None

        text = format_proposal_message(proposal)
        keyboard = (
            create_proposal_keyboard(proposal.id)
            if proposal.mode == ExecutionMode.SEMIAUTO
            and proposal.status == ProposalStatus.PENDING_APPROVAL
            else None
        )

        msg = await self.client.send_message(text=text, reply_markup=keyboard)
        if msg:
            self._proposal_messages[proposal.id] = msg.message_id
        return msg

    async def update_proposal(self, proposal: TradeProposal) -> bool:
        msg_id = self._proposal_messages.get(proposal.id)
        if not msg_id:
            return False

        text = format_proposal_message(proposal)
        keyboard = (
            create_proposal_keyboard(proposal.id)
            if proposal.mode == ExecutionMode.SEMIAUTO
            and proposal.status == ProposalStatus.PENDING_APPROVAL
            else None
        )

        return await self.client.edit_message_text(
            message_id=msg_id,
            text=text,
            reply_markup=keyboard,
        )

    async def send_order_filled(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
    ) -> TelegramMessage | None:
        text = (
            "🎯 *ORDEM EXECUTADA (FILL)* 🎯\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}` | *{direction.upper()}* `{volume:.2f}` lotes\n"
            f"• *Preço de Execução:* `{price:.2f}`\n"
            "• *Status:* Ordem ativa no broker com proteções armadas."
        )
        return await self.client.send_message(text=text)

    async def send_break_even_activated(
        self,
        ticket: int,
        symbol: str,
        price: float,
    ) -> TelegramMessage | None:
        text = (
            "🛡️ *BREAK-EVEN ATIVADO* 🛡️\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Novo Stop Loss:* `{price:.2f}` (Preço de Entrada)\n"
            "• *Status:* Risco da operação zerado com sucesso."
        )
        return await self.client.send_message(text=text)

    async def send_trailing_stop_updated(
        self,
        ticket: int,
        symbol: str,
        new_sl: float,
    ) -> TelegramMessage | None:
        text = (
            "📈 *TRAILING STOP AJUSTADO* 📈\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Novo Stop Loss:* `{new_sl:.2f}`\n"
            "• *Status:* Protegendo lucros acumulados da operação."
        )
        return await self.client.send_message(text=text)

    async def send_partial_close_executed(
        self,
        ticket: int,
        symbol: str,
        volume_closed: float,
        remaining_volume: float,
        r_multiple: float,
        new_sl: float | None = None,
        runner_active: bool = False,
    ) -> TelegramMessage | None:
        be_info = (
            f"\n• *Stop Loss:* Ajustado para `{new_sl:.2f}` (Break-Even / Risco Zero)"
            if new_sl is not None
            else ""
        )
        runner_info = (
            "\n• *Estratégia Runner:* TP removido — surfando tendência via Trailing Stop 🏄"
            if runner_active
            else ""
        )
        status_text = (
            "Lucro no bolso, risco zero e restante surfando a tendência sem teto de ganho!"
            if runner_active
            else "Lucro parcial garantido e posição protegida."
        )
        text = (
            "🎯 *PARCIAL 50% EXECUTADA* 🎯\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Volume Liquidado:* `{volume_closed:.2f}` lotes\n"
            f"• *Volume Remanescente:* `{remaining_volume:.2f}` lotes\n"
            f"• *Alvo Atingido:* `+{r_multiple:.2f}R`"
            f"{be_info}"
            f"{runner_info}\n"
            f"• *Status:* {status_text}"
        )
        return await self.client.send_message(text=text)

    async def send_smart_exit_triggered(
        self,
        ticket: int,
        symbol: str,
        reason: str,
        exit_price: float,
    ) -> TelegramMessage | None:
        text = (
            "⚡ *SMART EXIT EXECUTADO* ⚡\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Preço de Saída:* `{exit_price:.2f}`\n"
            f"• *Motivo:* _{escape_markdown(reason)}_\n"
            "• *Status:* Trade encerrado antecipadamente para proteger o capital."
        )
        return await self.client.send_message(text=text)

    async def send_order_cancelled(
        self,
        ticket: int,
        symbol: str,
        reason: str,
    ) -> TelegramMessage | None:
        text = (
            "⏳ *ORDEM CANCELADA NO BROKER* ⏳\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Motivo:* _{escape_markdown(reason)}_\n"
            "• *Status:* Ordem pendente removida com segurança."
        )
        return await self.client.send_message(text=text)

    async def send_trade_closed(
        self,
        ticket: int,
        symbol: str,
        profit_usd: float,
        profit_r: float,
        exit_price: float,
    ) -> TelegramMessage | None:
        if profit_usd > 0.05:
            header = "🎉 *TRADE ENCERRADO (TAKE PROFIT / WIN)* 💰"
            status = "Alvo atingido com sucesso! Lucro garantido no bolso."
        elif profit_usd < -0.05:
            header = "🛑 *TRADE ENCERRADO (STOP LOSS)* 🔻"
            status = "Stop Loss executado conforme gerenciamento de risco."
        else:
            header = "🛡️ *TRADE ENCERRADO (BREAK-EVEN)* ⚖️"
            status = "Operação encerrada no zero a zero (risco protegido)."

        text = (
            f"{header}\n\n"
            f"• *Ticket:* `#{ticket}`\n"
            f"• *Ativo:* `{symbol}`\n"
            f"• *Preço de Saída:* `{exit_price:.2f}`\n"
            f"• *Resultado USD:* `${profit_usd:+.2f}`\n"
            f"• *Resultado R:* `{profit_r:+.2f}R`\n"
            f"• *Status:* {status}"
        )
        return await self.client.send_message(text=text)

    def notify_proposal_change_sync(self, proposal: TradeProposal) -> None:
        if proposal.id not in self._proposal_messages:
            return
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None

            if current_loop is loop:
                loop.create_task(self.update_proposal(proposal))
            else:
                asyncio.run_coroutine_threadsafe(self.update_proposal(proposal), loop)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = await self.client.get_updates(offset=self._offset, timeout=5)
                for update in updates:
                    self._offset = max(self._offset, update.update_id + 1)
                    try:
                        await self.handler.handle_update(update)
                    except Exception as exc:
                        logger.error(
                            "Error processing Telegram update %s: %s", update.update_id, exc
                        )
                if not updates and self.poll_interval_s > 0:
                    await asyncio.sleep(self.poll_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Error in Telegram poll loop: %s", exc)
                await asyncio.sleep(self.poll_interval_s)
