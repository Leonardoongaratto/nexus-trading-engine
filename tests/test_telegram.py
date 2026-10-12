import asyncio
import email.message
import io
import json
import logging
import threading
import time
import urllib.error
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from xau_bot.advisor.modes import (
    AdvisorDispatcher,
    ExecutionMode,
    ProposalManager,
    ProposalStatus,
    TradeProposal,
)
from xau_bot.broker.models import (
    BrokerOrder,
    BrokerPosition,
    OrderResult,
    OrderType,
)
from xau_bot.cli.run import create_alert_manager
from xau_bot.config import Settings
from xau_bot.domain.models import Direction, OrderIntent, SetupTriggerType, Timeframe, TradeSetup
from xau_bot.execution.models import ExecutionResult, ExecutionStatus
from xau_bot.notify.models import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    TelegramChat,
    TelegramMessage,
    TelegramUpdate,
    TelegramUser,
)
from xau_bot.notify.service import TelegramRunner, build_telegram_service
from xau_bot.notify.telegram import (
    TelegramBotClient,
    TelegramCommandHandler,
    TelegramService,
    create_proposal_keyboard,
    escape_markdown,
    format_proposal_message,
)
from xau_bot.server import HealthStatusState


def make_test_setup(
    setup_id: str = "setup_telegram_1",
    direction: Direction = Direction.BULLISH,
    entry: float = 2650.0,
    sl: float = 2640.0,
    tp: float = 2670.0,
) -> TradeSetup:
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    risk_points = abs(entry - sl)
    rr = abs(tp - entry) / risk_points if risk_points > 0 else 0.0
    return TradeSetup(
        id=setup_id,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=rr,
        trigger_type=SetupTriggerType.FVG_CE,
        timeframe=Timeframe.M5,
        confluences=("SWEEP", "CHOCH"),
    )


def make_test_intent(setup: TradeSetup, volume: float = 0.1) -> OrderIntent:
    return OrderIntent(
        symbol="XAUUSD",
        direction=setup.direction,
        order_type="BUY_LIMIT" if setup.direction == Direction.BULLISH else "SELL_LIMIT",
        volume=volume,
        price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        magic=123456,
        comment="test_intent",
    )


def make_test_proposal(
    prop_id: str = "prop_123",
    mode: ExecutionMode = ExecutionMode.SEMIAUTO,
    status: ProposalStatus = ProposalStatus.PENDING_APPROVAL,
    direction: Direction = Direction.BULLISH,
    entry: float = 2650.0,
    sl: float = 2640.0,
    tp: float = 2670.0,
    volume: float = 0.05,
    risk_usd: float = 50.0,
    risk_pct: float = 0.01,
    reason: str = "",
    order_ticket: int | None = None,
    atr: float | None = None,
) -> TradeProposal:
    setup_sl = (
        sl if sl != entry else (entry - 10.0 if direction == Direction.BULLISH else entry + 10.0)
    )
    setup = make_test_setup(direction=direction, entry=entry, sl=setup_sl, tp=tp)
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=direction,
        order_type="BUY_LIMIT" if direction == Direction.BULLISH else "SELL_LIMIT",
        volume=volume,
        price=entry,
        stop_loss=sl,
        take_profit=tp,
        magic=123456,
        comment="test_intent",
    )
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    return TradeProposal(
        id=prop_id,
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        volume=volume,
        risk_usd=risk_usd,
        risk_pct=risk_pct,
        created_at=t0,
        expires_at=t0 + timedelta(seconds=300),
        status=status,
        mode=mode,
        setup=setup,
        intent=intent,
        reason=reason,
        order_ticket=order_ticket,
        atr=atr,
    )


# ---------------------------------------------------------------------------
# 1. Models Serialization Tests
# ---------------------------------------------------------------------------


def test_models_inline_keyboard() -> None:
    btn = InlineKeyboardButton(text="Test", callback_data="cb_1")
    btn_dict = btn.to_dict()
    assert btn_dict == {"text": "Test", "callback_data": "cb_1"}
    btn_restored = InlineKeyboardButton.from_dict(btn_dict)
    assert btn_restored == btn

    markup = InlineKeyboardMarkup(inline_keyboard=[[btn]])
    markup_dict = markup.to_dict()
    assert markup_dict == {"inline_keyboard": [[{"text": "Test", "callback_data": "cb_1"}]]}
    markup_restored = InlineKeyboardMarkup.from_dict(markup_dict)
    assert markup_restored == markup
    assert InlineKeyboardMarkup.from_dict({}).inline_keyboard == []


def test_models_user_and_chat() -> None:
    user = TelegramUser(
        id=123, is_bot=False, first_name="Leo", last_name="O", username="leo_trader"
    )
    user_dict = user.to_dict()
    assert user_dict["id"] == 123
    assert user_dict["username"] == "leo_trader"
    user_restored = TelegramUser.from_dict(user_dict)
    assert user_restored == user

    user_min = TelegramUser(id=456)
    assert user_min.to_dict() == {"id": 456, "is_bot": False, "first_name": ""}
    assert TelegramUser.from_dict({"id": 456}) == user_min

    chat = TelegramChat(id=789, type="private", title="ChatTitle", username="chat_u")
    chat_dict = chat.to_dict()
    assert chat_dict["id"] == 789
    chat_restored = TelegramChat.from_dict(chat_dict)
    assert chat_restored == chat

    chat_min = TelegramChat(id=999, type="supergroup")
    assert chat_min.to_dict() == {"id": 999, "type": "supergroup"}


def test_models_message_and_callbacks() -> None:
    chat = TelegramChat(id=100, type="private")
    user = TelegramUser(id=200, first_name="Trader")
    msg = TelegramMessage(
        message_id=50,
        chat=chat,
        date=1700000000,
        from_user=user,
        text="Hello Bot",
    )
    msg_dict = msg.to_dict()
    assert msg_dict["message_id"] == 50
    assert msg_dict["text"] == "Hello Bot"
    msg_restored = TelegramMessage.from_dict(msg_dict)
    assert msg_restored == msg

    msg_min = TelegramMessage(message_id=51, chat=chat, date=1700000000)
    assert "text" not in msg_min.to_dict()
    assert "from" not in msg_min.to_dict()

    cb = CallbackQuery(
        id="cb_q_1",
        from_user=user,
        data="approve:prop_1",
        message=msg,
    )
    cb_dict = cb.to_dict()
    assert cb_dict["id"] == "cb_q_1"
    assert cb_dict["data"] == "approve:prop_1"
    cb_restored = CallbackQuery.from_dict(cb_dict)
    assert cb_restored == cb

    cb_min = CallbackQuery(id="cb_q_2", from_user=user)
    assert "data" not in cb_min.to_dict()

    update = TelegramUpdate(update_id=1, message=msg, callback_query=cb)
    update_dict = update.to_dict()
    assert update_dict["update_id"] == 1
    assert "message" in update_dict
    assert "callback_query" in update_dict
    update_restored = TelegramUpdate.from_dict(update_dict)
    assert update_restored == update

    update_empty = TelegramUpdate(update_id=2)
    assert update_empty.to_dict() == {"update_id": 2}


# ---------------------------------------------------------------------------
# 2. Formatting & Markdown Tests
# ---------------------------------------------------------------------------


def test_escape_markdown() -> None:
    text = "Hello_World! *bold* [link] (1.5 + 2.0 = 3.5) `code`"
    escaped = escape_markdown(text)
    assert "\\_" in escaped
    assert "\\*" in escaped
    assert "\\[" in escaped
    assert "\\`" in escaped
    assert "\\(" not in escaped
    assert "\\!" not in escaped
    assert "\\." not in escaped


def test_format_proposal_message_all_modes_and_statuses() -> None:
    # 1. Analyst Mode
    p_analyst = make_test_proposal(
        mode=ExecutionMode.ANALYST,
        status=ProposalStatus.ADVISORY,
        atr=2.45,
    )
    text_analyst = format_proposal_message(p_analyst)
    assert "SUGESTÃO DE ANÁLISE SMC" in text_analyst
    assert "Modo Analista" in text_analyst
    assert "COMPRA (BUY)" in text_analyst
    assert "Informativa" in text_analyst
    assert "• *ATR:* `2.45`" in text_analyst

    # 2. Semiauto Mode Bearish
    p_semiauto = make_test_proposal(
        mode=ExecutionMode.SEMIAUTO,
        status=ProposalStatus.PENDING_APPROVAL,
        direction=Direction.BEARISH,
        entry=2670.0,
        sl=2680.0,
        tp=2650.0,
        reason="FVG M5 confluência",
    )
    text_semiauto = format_proposal_message(p_semiauto)
    assert "PROPOSTA DE TRADE PENDENTE" in text_semiauto
    assert "VENDA (SELL)" in text_semiauto
    assert "Aguardando Aprovação" in text_semiauto
    assert "FVG M5 confluência" in text_semiauto

    # 3. Auto Mode with Ticket
    p_auto = make_test_proposal(
        mode=ExecutionMode.AUTO,
        status=ProposalStatus.DISPATCHED,
        order_ticket=88889999,
    )
    text_auto = format_proposal_message(p_auto)
    assert "ORDEM AUTOMÁTICA ENVIADA" in text_auto
    assert "88889999" in text_auto
    assert "Enviada ao Broker" in text_auto

    # 4. Status mappings coverage
    statuses = [
        (ProposalStatus.APPROVED, "Aprovada pelo Operador"),
        (ProposalStatus.REJECTED, "Rejeitada pelo Operador"),
        (ProposalStatus.EXPIRED, "Expirada por Timeout"),
        (ProposalStatus.DISPATCH_FAILED, "Falha no Envio ao Broker"),
        (ProposalStatus.CANCELLED, "Cancelada (Invalidação de Preço)"),
    ]
    for st, label in statuses:
        p = make_test_proposal(status=st)
        assert label in format_proposal_message(p)

    # 5. Zero risk points
    p_zero_risk = make_test_proposal(entry=2650.0, sl=2650.0, tp=2670.0)
    assert "1:0.00" in format_proposal_message(p_zero_risk)


def test_create_proposal_keyboard() -> None:
    kb = create_proposal_keyboard("prop_abc")
    assert len(kb.inline_keyboard) == 1
    row = kb.inline_keyboard[0]
    assert len(row) == 2
    assert row[0].callback_data == "approve:prop_abc"
    assert "Executar" in row[0].text
    assert row[1].callback_data == "reject:prop_abc"
    assert "Ignorar" in row[1].text


# ---------------------------------------------------------------------------
# 3. TelegramBotClient Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_client_unconfigured() -> None:
    client = TelegramBotClient(token="", chat_id="")
    assert not client.is_configured
    assert client.chat_id == ""

    assert await client.send_message("test") is None
    assert await client.edit_message_text(1, "test") is False
    assert await client.answer_callback_query("cb_1") is False
    assert await client.get_updates() == []


@pytest.mark.asyncio
async def test_telegram_client_send_message_success() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")
    assert client.is_configured
    assert client.chat_id == "chat999"

    mock_resp = {
        "ok": True,
        "result": {
            "message_id": 42,
            "date": 1700000000,
            "chat": {"id": 999, "type": "private"},
            "text": "Sent text",
        },
    }

    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
        mock_resp
    ).encode("utf-8")

    with patch("urllib.request.urlopen", mock_urlopen):
        msg = await client.send_message("Test message")
        assert msg is not None
        assert msg.message_id == 42
        assert msg.text == "Sent text"


@pytest.mark.asyncio
async def test_telegram_client_send_message_retry_without_parse_mode() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")

    # First call raises HTTP 400 (markdown parse error), second call succeeds
    mock_resp_retry = {
        "ok": True,
        "result": {
            "message_id": 43,
            "date": 1700000000,
            "chat": {"id": 999, "type": "private"},
            "text": "Raw text",
        },
    }

    http_400 = urllib.error.HTTPError(
        url="http://test", code=400, msg="Bad Request", hdrs=MagicMock(), fp=io.BytesIO()
    )

    retry_cm = MagicMock()
    retry_cm.__enter__.return_value.read.return_value = json.dumps(mock_resp_retry).encode("utf-8")

    with patch("urllib.request.urlopen", side_effect=[http_400, retry_cm]):
        msg = await client.send_message("Message with *broken markdown")
        assert msg is not None
        assert msg.message_id == 43


@pytest.mark.asyncio
async def test_telegram_client_send_message_failure() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")

    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        msg = await client.send_message("Fail")
        assert msg is None


@pytest.mark.asyncio
async def test_telegram_client_edit_message_text() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")

    # Success
    mock_resp = {"ok": True, "result": True}
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
        mock_resp
    ).encode("utf-8")

    with patch("urllib.request.urlopen", mock_urlopen):
        success = await client.edit_message_text(
            message_id=42, text="Updated text", reply_markup=None
        )
        assert success is True

    # Error
    with patch("urllib.request.urlopen", side_effect=Exception("Timeout")):
        fail = await client.edit_message_text(message_id=42, text="Updated text")
        assert fail is False


@pytest.mark.asyncio
async def test_telegram_client_answer_callback_query() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")

    mock_resp = {"ok": True}
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
        mock_resp
    ).encode("utf-8")

    with patch("urllib.request.urlopen", mock_urlopen):
        assert await client.answer_callback_query("cb_1", text="OK", show_alert=False) is True

    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        assert await client.answer_callback_query("cb_1") is False


@pytest.mark.asyncio
async def test_telegram_client_get_updates() -> None:
    client = TelegramBotClient(token="token123", chat_id="chat999")

    mock_resp = {
        "ok": True,
        "result": [
            {
                "update_id": 101,
                "message": {
                    "message_id": 1,
                    "date": 1700000000,
                    "chat": {"id": 999, "type": "private"},
                    "text": "/status",
                },
            }
        ],
    }
    mock_urlopen = MagicMock()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
        mock_resp
    ).encode("utf-8")

    with patch("urllib.request.urlopen", mock_urlopen):
        updates = await client.get_updates(offset=100)
        assert len(updates) == 1
        assert updates[0].update_id == 101
        assert updates[0].message is not None
        assert updates[0].message.text == "/status"

    with patch("urllib.request.urlopen", side_effect=Exception("Read timeout")):
        empty_updates = await client.get_updates(offset=100)
        assert empty_updates == []


# ---------------------------------------------------------------------------
# 4. Command Handler & Security Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_handler_security_unauthorized(caplog: pytest.LogCaptureFixture) -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)
    client.send_message = AsyncMock(return_value=None)

    handler = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    assert handler.is_authorized("12345")
    assert not handler.is_authorized("99999")

    # Unauthorized Message
    unauth_msg = TelegramMessage(
        message_id=1,
        date=1700000000,
        chat=TelegramChat(id=99999, type="private"),
        text="/status",
    )
    with caplog.at_level(logging.WARNING):
        await handler.handle_update(TelegramUpdate(update_id=1, message=unauth_msg))
    assert "Unauthorized Telegram access attempt" in caplog.text
    client.send_message.assert_not_called()

    # Unauthorized Callback Query
    unauth_cb = CallbackQuery(
        id="cb_bad",
        from_user=TelegramUser(id=99999),
        data="approve:prop_1",
    )
    await handler.handle_update(TelegramUpdate(update_id=2, callback_query=unauth_cb))
    client.answer_callback_query.assert_called_once_with(
        callback_query_id="cb_bad",
        text="Acesso não autorizado.",
        show_alert=True,
    )


@pytest.mark.asyncio
async def test_command_handler_help_and_unknown() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)
    handler = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")

    auth_chat = TelegramChat(id=12345, type="private")

    # Help commands
    for cmd in ["/ajuda", "/help", "/start"]:
        msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text=cmd)
        await handler.handle_update(TelegramUpdate(update_id=1, message=msg))
        assert "Comandos do Robô Institucional" in client.send_message.call_args[1]["text"]

    # Unknown command
    msg_unknown = TelegramMessage(message_id=2, date=1700000000, chat=auth_chat, text="/foo")
    await handler.handle_update(TelegramUpdate(update_id=2, message=msg_unknown))
    assert "Comando não reconhecido" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_command_handler_status() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    status_data = {
        "broker_connected": True,
        "trading_halted": False,
        "execution_mode": "semiauto",
        "daily_pnl": 125.50,
        "proposals": {"pending": 2, "approved": 3, "dispatched": 3},
    }

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        get_status_fn=lambda: status_data,
        is_trading_halted_fn=lambda: False,
    )

    msg = TelegramMessage(
        message_id=1,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="/status",
    )
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg))
    text = client.send_message.call_args[1]["text"]
    assert "STATUS OPERACIONAL DO ROBÔ" in text
    assert "SEMIAUTO" in text
    assert "Conectado" in text
    assert "$125.50" in text
    assert "Pendentes: `2`" in text


@pytest.mark.asyncio
async def test_command_handler_positions() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    # 1. Broker is None
    handler_no_broker = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/posicoes")
    await handler_no_broker.handle_update(TelegramUpdate(update_id=1, message=msg))
    assert "Broker não configurado" in client.send_message.call_args[1]["text"]

    # 2. Broker error
    mock_broker = MagicMock()
    mock_broker.get_open_positions = AsyncMock(side_effect=Exception("MT5 disconnected"))
    handler_err = TelegramCommandHandler(
        bot_client=client, authorized_chat_id="12345", broker=mock_broker
    )
    await handler_err.handle_update(TelegramUpdate(update_id=2, message=msg))
    assert "Erro ao consultar posições" in client.send_message.call_args[1]["text"]

    # 3. Empty positions
    mock_broker.get_open_positions = AsyncMock(return_value=())
    await handler_err.handle_update(TelegramUpdate(update_id=3, message=msg))
    assert "Nenhuma posição aberta" in client.send_message.call_args[1]["text"]

    # 4. Active positions
    pos1 = BrokerPosition(
        ticket=101,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        current_price=2655.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        floating_pnl=50.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    pos2 = BrokerPosition(
        ticket=102,
        symbol="XAUUSD",
        direction=Direction.BEARISH,
        volume=0.05,
        open_price=2660.0,
        current_price=2665.0,
        stop_loss=None,
        take_profit=None,
        floating_pnl=-25.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    mock_broker.get_open_positions = AsyncMock(return_value=(pos1, pos2))
    await handler_err.handle_update(TelegramUpdate(update_id=4, message=msg))
    text_pos = client.send_message.call_args[1]["text"]
    assert "#101" in text_pos
    assert "BUY" in text_pos
    assert "$50.00" in text_pos
    assert "#102" in text_pos
    assert "SELL" in text_pos
    assert "$-25.00" in text_pos


@pytest.mark.asyncio
async def test_command_handler_pause_and_resume() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    halt_state = {"halted": False, "reason": ""}

    def set_halt(halted: bool, reason: str) -> None:
        halt_state["halted"] = halted
        halt_state["reason"] = reason

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        set_trading_halt_fn=set_halt,
    )
    auth_chat = TelegramChat(id=12345, type="private")

    # /pausar
    msg_pause = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/pausar")
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg_pause))
    assert halt_state["halted"] is True
    assert "TRADING PAUSADO COM SUCESSO" in client.send_message.call_args[1]["text"]

    # /retomar
    msg_resume = TelegramMessage(message_id=2, date=1700000000, chat=auth_chat, text="/retomar")
    await handler.handle_update(TelegramUpdate(update_id=2, message=msg_resume))
    assert halt_state["halted"] is False
    assert "TRADING RETOMADO" in client.send_message.call_args[1]["text"]

    # Without set_trading_halt_fn
    handler_no_fn = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    await handler_no_fn.handle_update(TelegramUpdate(update_id=3, message=msg_pause))
    assert "não disponível" in client.send_message.call_args[1]["text"]
    await handler_no_fn.handle_update(TelegramUpdate(update_id=4, message=msg_resume))
    assert "não disponível" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_command_handler_close_all() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)
    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/fechar_tudo")

    # 1. No broker
    handler_no_broker = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    await handler_no_broker.handle_update(TelegramUpdate(update_id=1, message=msg))
    assert "Broker não disponível" in client.send_message.call_args[1]["text"]

    # 2. Successful close all
    mock_broker = MagicMock()
    pos = BrokerPosition(
        ticket=101,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        current_price=2655.0,
        stop_loss=2640.0,
        take_profit=2670.0,
        floating_pnl=50.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    mock_broker.get_open_positions = AsyncMock(return_value=(pos,))
    mock_broker.close_position = AsyncMock(
        return_value=OrderResult(success=True, ticket=101, retcode=10009)
    )

    open_order = BrokerOrder(
        ticket=201,
        symbol="XAUUSD",
        order_type=OrderType.BUY_LIMIT,
        volume=0.1,
        price=2645.0,
        stop_loss=2635.0,
        take_profit=2665.0,
        time_setup=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    mock_broker.get_open_orders = AsyncMock(return_value=(open_order,))
    mock_broker.cancel_order = AsyncMock(
        return_value=OrderResult(success=True, ticket=201, retcode=10009)
    )

    halt_called = False

    def on_halt(val: bool, reason: str) -> None:
        nonlocal halt_called
        halt_called = val

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        broker=mock_broker,
        set_trading_halt_fn=on_halt,
    )
    await handler.handle_update(TelegramUpdate(update_id=2, message=msg))
    assert halt_called is True
    summary_text = client.send_message.call_args[1]["text"]
    assert "KILL SWITCH EXECUTADO COM SUCESSO" in summary_text
    assert "Posições Encerradas: `1`" in summary_text
    assert "Ordens Limite Canceladas: `1`" in summary_text

    # 3. Close position failure
    mock_broker.close_position = AsyncMock(
        return_value=OrderResult(success=False, error_message="Market closed", retcode=10018)
    )
    await handler.handle_update(TelegramUpdate(update_id=3, message=msg))
    summary_fail = client.send_message.call_args[1]["text"]
    assert "Falhas no Fechamento: `1`" in summary_fail

    # 4. Exception in get_open_positions
    mock_broker.get_open_positions = AsyncMock(side_effect=Exception("Terminal crash"))
    await handler.handle_update(TelegramUpdate(update_id=4, message=msg))
    assert "FALHA NO FECHAMENTO DE EMERGÊNCIA" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_command_handler_proposals() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)
    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/propostas")

    # 1. No proposal manager
    handler_no_pm = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    await handler_no_pm.handle_update(TelegramUpdate(update_id=1, message=msg))
    assert "não disponível" in client.send_message.call_args[1]["text"]

    # 2. Empty proposals
    pm = ProposalManager()
    handler = TelegramCommandHandler(
        bot_client=client, authorized_chat_id="12345", proposal_manager=pm
    )
    await handler.handle_update(TelegramUpdate(update_id=2, message=msg))
    assert "Nenhuma proposta pendente" in client.send_message.call_args[1]["text"]

    # 3. Active proposals
    setup = make_test_setup()
    pm.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)
    await handler.handle_update(TelegramUpdate(update_id=3, message=msg))
    assert "PROPOSTA DE TRADE PENDENTE" in client.send_message.call_args[1]["text"]
    assert client.send_message.call_args[1]["reply_markup"] is not None


# ---------------------------------------------------------------------------
# 5. Callbacks: Approve and Reject Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_handler_approve_callback_flow() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)
    client.edit_message_text = AsyncMock(return_value=True)

    pm = ProposalManager()
    setup = make_test_setup()
    intent = make_test_intent(setup)
    proposal = pm.create_proposal(setup=setup, intent=intent, mode=ExecutionMode.SEMIAUTO)

    mock_executor = MagicMock()
    mock_executor.execute_intent = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            status=ExecutionStatus.FILLED,
            ticket=55556666,
            price=2650.0,
            volume=0.1,
            retcode=10009,
        )
    )

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        proposal_manager=pm,
        order_executor=mock_executor,
    )

    msg = TelegramMessage(
        message_id=99, date=1700000000, chat=TelegramChat(id=12345, type="private")
    )
    cb = CallbackQuery(
        id="cb_app_1",
        from_user=TelegramUser(id=12345),
        data=f"approve:{proposal.id}",
        message=msg,
    )

    await handler.handle_update(TelegramUpdate(update_id=1, callback_query=cb))

    # Verify order execution and proposal dispatched
    mock_executor.execute_intent.assert_called_once_with(intent)
    assert proposal.status == ProposalStatus.DISPATCHED
    assert proposal.order_ticket == 55556666
    client.answer_callback_query.assert_called_with(
        callback_query_id="cb_app_1",
        text="Proposta aprovada! Despachando ordem...",
        show_alert=False,
    )
    client.edit_message_text.assert_called_once()
    assert "Enviada ao Broker" in client.edit_message_text.call_args[1]["text"]

    # Second approval fails (invalid_state)
    await handler.handle_update(TelegramUpdate(update_id=2, callback_query=cb))
    assert "Não foi possível aprovar" in client.answer_callback_query.call_args_list[-1][1]["text"]


@pytest.mark.asyncio
async def test_command_handler_approve_callback_executor_failure() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)
    client.edit_message_text = AsyncMock(return_value=True)

    pm = ProposalManager()
    setup = make_test_setup()
    intent = make_test_intent(setup)
    proposal = pm.create_proposal(setup=setup, intent=intent, mode=ExecutionMode.SEMIAUTO)

    # 1. Executor returns failure
    mock_executor = MagicMock()
    mock_executor.execute_intent = AsyncMock(
        return_value=ExecutionResult(
            success=False,
            status=ExecutionStatus.REJECTED,
            retcode=10015,
            error_message="Invalid price",
        )
    )

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        proposal_manager=pm,
        order_executor=mock_executor,
    )

    msg = TelegramMessage(
        message_id=99, date=1700000000, chat=TelegramChat(id=12345, type="private")
    )
    cb = CallbackQuery(
        id="cb_fail_1",
        from_user=TelegramUser(id=12345),
        data=f"approve:{proposal.id}",
        message=msg,
    )

    await handler.handle_update(TelegramUpdate(update_id=1, callback_query=cb))
    assert proposal.status == ProposalStatus.DISPATCH_FAILED
    assert proposal.reason is not None and "Broker rejected: 10015" in proposal.reason

    # 2. Executor raises exception
    proposal2 = pm.create_proposal(setup=setup, intent=intent, mode=ExecutionMode.SEMIAUTO)
    mock_executor.execute_intent = AsyncMock(side_effect=RuntimeError("Socket closed"))
    cb2 = CallbackQuery(
        id="cb_fail_2",
        from_user=TelegramUser(id=12345),
        data=f"approve:{proposal2.id}",
        message=msg,
    )
    await handler.handle_update(TelegramUpdate(update_id=2, callback_query=cb2))
    assert proposal2.status == ProposalStatus.DISPATCH_FAILED
    assert proposal2.reason is not None and "Falha interna na execução da ordem" in proposal2.reason


@pytest.mark.asyncio
async def test_command_handler_reject_callback_flow() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)
    client.edit_message_text = AsyncMock(return_value=True)

    pm = ProposalManager()
    setup = make_test_setup()
    proposal = pm.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)

    handler = TelegramCommandHandler(
        bot_client=client, authorized_chat_id="12345", proposal_manager=pm
    )

    msg = TelegramMessage(
        message_id=99, date=1700000000, chat=TelegramChat(id=12345, type="private")
    )
    cb = CallbackQuery(
        id="cb_rej_1",
        from_user=TelegramUser(id=12345),
        data=f"reject:{proposal.id}",
        message=msg,
    )

    await handler.handle_update(TelegramUpdate(update_id=1, callback_query=cb))
    assert proposal.status == ProposalStatus.REJECTED
    client.answer_callback_query.assert_called_with(
        callback_query_id="cb_rej_1",
        text="Proposta rejeitada com sucesso.",
        show_alert=False,
    )
    client.edit_message_text.assert_called_once()
    assert "Rejeitada pelo Operador" in client.edit_message_text.call_args[1]["text"]

    # Rejection of nonexistent proposal
    cb_bad = CallbackQuery(
        id="cb_rej_bad",
        from_user=TelegramUser(id=12345),
        data="reject:nonexistent",
        message=msg,
    )
    await handler.handle_update(TelegramUpdate(update_id=2, callback_query=cb_bad))
    assert "Não foi possível rejeitar" in client.answer_callback_query.call_args_list[-1][1]["text"]


# ---------------------------------------------------------------------------
# 6. TelegramService Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_service_lifecycle() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    client.get_updates = AsyncMock(return_value=[])

    handler = MagicMock(spec=TelegramCommandHandler)
    handler.handle_update = AsyncMock()

    service = TelegramService(bot_client=client, command_handler=handler, poll_interval_s=0.01)
    assert not service.is_running

    await service.start()
    assert service.is_running
    # Idempotent start
    await service.start()
    assert service.is_running

    await asyncio.sleep(0.05)
    await service.stop()
    assert not service.is_running
    assert service._polling_task is None

    # Unconfigured client skips startup
    client.is_configured = False
    await service.start()
    assert not service.is_running


@pytest.mark.asyncio
async def test_telegram_service_send_and_update_proposal() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    sent_msg = TelegramMessage(
        message_id=77,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="Proposal text",
    )
    client.send_message = AsyncMock(return_value=sent_msg)
    client.edit_message_text = AsyncMock(return_value=True)

    handler = MagicMock(spec=TelegramCommandHandler)
    service = TelegramService(bot_client=client, command_handler=handler)

    # 1. Send Semiauto Proposal
    prop = make_test_proposal(prop_id="prop_77", mode=ExecutionMode.SEMIAUTO)
    result_msg = await service.send_proposal(prop)
    assert result_msg == sent_msg
    assert service._proposal_messages["prop_77"] == 77
    assert client.send_message.call_args[1]["reply_markup"] is not None

    # 2. Update Proposal when approved (removes keyboard)
    prop.status = ProposalStatus.APPROVED
    update_res = await service.update_proposal(prop)
    assert update_res is True
    assert client.edit_message_text.call_args[1]["message_id"] == 77
    assert client.edit_message_text.call_args[1]["reply_markup"] is None

    # 3. Update untracked proposal
    untracked = make_test_proposal(prop_id="prop_untracked")
    assert await service.update_proposal(untracked) is False

    # 4. Unconfigured client
    client.is_configured = False
    assert await service.send_proposal(prop) is None


@pytest.mark.asyncio
async def test_telegram_service_notify_proposal_change_sync() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    sent_msg = TelegramMessage(
        message_id=88,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="Proposal text",
    )
    client.send_message = AsyncMock(return_value=sent_msg)
    client.edit_message_text = AsyncMock(return_value=True)
    client.get_updates = AsyncMock(return_value=[])

    handler = MagicMock(spec=TelegramCommandHandler)
    service = TelegramService(bot_client=client, command_handler=handler, poll_interval_s=0.01)
    await service.start()

    prop = make_test_proposal(prop_id="prop_88", mode=ExecutionMode.SEMIAUTO)
    await service.send_proposal(prop)

    # Trigger synchronous proposal update from another context
    prop.status = ProposalStatus.EXPIRED
    service.notify_proposal_change_sync(prop)

    # Allow event loop to process scheduled coroutine
    await asyncio.sleep(0.05)
    assert client.edit_message_text.called

    await service.stop()


@pytest.mark.asyncio
async def test_telegram_service_trade_management_messages() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    sent_msg = TelegramMessage(
        message_id=99,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="Trade notification",
    )
    client.send_message = AsyncMock(return_value=sent_msg)

    handler = MagicMock(spec=TelegramCommandHandler)
    service = TelegramService(bot_client=client, command_handler=handler)

    # 1. Partial close with break-even SL
    msg1 = await service.send_partial_close_executed(
        ticket=1001,
        symbol="XAUUSD",
        volume_closed=0.05,
        remaining_volume=0.05,
        r_multiple=1.0,
        new_sl=2650.0,
    )
    assert msg1 == sent_msg
    text1 = client.send_message.call_args[1]["text"]
    assert "PARCIAL 50% EXECUTADA" in text1
    assert "#1001" in text1
    assert "`0.05` lotes" in text1
    assert "Stop Loss" in text1
    assert "2650.00" in text1

    # 1b. Partial close without new SL
    await service.send_partial_close_executed(
        ticket=1002,
        symbol="XAUUSD",
        volume_closed=0.05,
        remaining_volume=0.05,
        r_multiple=1.0,
        new_sl=None,
    )
    text1b = client.send_message.call_args[1]["text"]
    assert "#1002" in text1b
    assert "Stop Loss:" not in text1b

    # 2. Smart exit
    msg2 = await service.send_smart_exit_triggered(
        ticket=1001,
        symbol="XAUUSD",
        reason="Opposite M5 ChoCH detected",
        exit_price=2652.5,
    )
    assert msg2 == sent_msg
    text2 = client.send_message.call_args[1]["text"]
    assert "SMART EXIT EXECUTADO" in text2
    assert "#1001" in text2
    assert "Opposite M5 ChoCH detected" in text2

    # 3. Trade closed (profit and loss)
    msg3 = await service.send_trade_closed(
        ticket=1001,
        symbol="XAUUSD",
        profit_usd=50.0,
        profit_r=1.0,
        exit_price=2660.0,
    )
    assert msg3 == sent_msg
    text3 = client.send_message.call_args[1]["text"]
    assert "TRADE ENCERRADO" in text3
    assert "$+50.00" in text3
    assert "+1.00R" in text3

    await service.send_trade_closed(
        ticket=1001,
        symbol="XAUUSD",
        profit_usd=-25.0,
        profit_r=-0.5,
        exit_price=2645.0,
    )
    text3_loss = client.send_message.call_args[1]["text"]
    assert "$-25.00" in text3_loss
    assert "-0.50R" in text3_loss


@pytest.mark.asyncio
async def test_telegram_service_poll_loop_processes_updates() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True

    up1 = TelegramUpdate(
        update_id=1,
        message=TelegramMessage(
            message_id=1,
            date=1700000000,
            chat=TelegramChat(id=12345, type="private"),
            text="/status",
        ),
    )
    up2 = TelegramUpdate(
        update_id=2,
        callback_query=CallbackQuery(
            id="cb_1",
            from_user=TelegramUser(id=12345),
            data="approve:p1",
        ),
    )

    client.get_updates = AsyncMock(side_effect=[[up1, up2], Exception("Polling error"), []])
    handler = MagicMock(spec=TelegramCommandHandler)
    handler.handle_update = AsyncMock()

    service = TelegramService(bot_client=client, command_handler=handler, poll_interval_s=0.01)
    await service.start()
    await asyncio.sleep(0.05)
    await service.stop()

    assert handler.handle_update.call_count == 2
    assert service._offset == 3


# ---------------------------------------------------------------------------
# 7. TelegramRunner & Factory Tests
# ---------------------------------------------------------------------------


def test_telegram_runner_lifecycle() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    client.get_updates = AsyncMock(return_value=[])

    handler = MagicMock(spec=TelegramCommandHandler)
    service = TelegramService(bot_client=client, command_handler=handler, poll_interval_s=0.01)

    runner = TelegramRunner(service)
    runner.start()
    assert runner._thread is not None
    assert runner._thread.is_alive()

    # Second start is no-op
    runner.start()

    time.sleep(0.1)
    runner.stop()
    assert not runner._thread.is_alive()

    # Unconfigured client skips worker start
    client.is_configured = False
    runner_unconfigured = TelegramRunner(service)
    runner_unconfigured.start()
    assert runner_unconfigured._thread is None


def test_build_telegram_service_factory() -> None:
    # 1. Missing credentials
    s_none = Settings(telegram_token=None, telegram_chat_id=None)
    assert build_telegram_service(s_none) is None

    # 2. Configured credentials
    s_ok = Settings(
        telegram_token=SecretStr("bot_token_secret"),
        telegram_chat_id=SecretStr("12345678"),
    )
    pm = ProposalManager()
    service = build_telegram_service(s_ok, proposal_manager=pm)
    assert service is not None
    assert service.client.is_configured
    assert service.client.chat_id == "12345678"
    assert service.handler.authorized_chat_id == "12345678"
    # Listener registered
    assert service.notify_proposal_change_sync in pm._listeners


# ---------------------------------------------------------------------------
# 8. CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_run_telegram_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    from xau_bot.cli.main import build_parser
    from xau_bot.cli.run import run_bot_cli

    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim"])

    # Configure Telegram credentials in environment
    monkeypatch.setenv("XAU_TELEGRAM_TOKEN", "bot123456:ABC-DEF")
    monkeypatch.setenv("XAU_TELEGRAM_CHAT_ID", "999888")

    started_runner = False
    stopped_runner = False

    class MockRunner:
        def __init__(self, service: Any) -> None:
            self.service = service

        def start(self) -> None:
            nonlocal started_runner
            started_runner = True

        def stop(self) -> None:
            nonlocal stopped_runner
            stopped_runner = True

    monkeypatch.setattr("xau_bot.cli.run.TelegramRunner", MockRunner)
    monkeypatch.setattr("xau_bot.cli.run.run_server", lambda **kwargs: None)

    exit_code = run_bot_cli(args, stdout=io.StringIO(), stderr=io.StringIO())
    assert exit_code == 0
    assert started_runner is True
    assert stopped_runner is True


def test_cli_run_wires_telegram_service_executor_and_halt(monkeypatch: pytest.MonkeyPatch) -> None:
    from xau_bot.cli.main import build_parser
    from xau_bot.cli.run import run_bot_cli

    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim"])

    monkeypatch.setenv("XAU_TELEGRAM_TOKEN", "bot123456:ABC-DEF")
    monkeypatch.setenv("XAU_TELEGRAM_CHAT_ID", "999888")

    captured_service: Any = None
    captured_server_kwargs: dict[str, Any] = {}

    class MockRunner:
        def __init__(self, service: Any) -> None:
            nonlocal captured_service
            captured_service = service
            self.service = service

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    def mock_run_server(**kwargs: Any) -> None:
        captured_server_kwargs.update(kwargs)

    monkeypatch.setattr("xau_bot.cli.run.TelegramRunner", MockRunner)
    monkeypatch.setattr("xau_bot.cli.run.run_server", mock_run_server)

    exit_code = run_bot_cli(args, stdout=io.StringIO(), stderr=io.StringIO())
    assert exit_code == 0
    assert captured_service is not None
    assert captured_service.handler.order_executor is not None
    assert captured_service.handler.set_trading_halt_fn is not None
    assert captured_service.handler.is_trading_halted_fn is not None
    assert captured_service.handler.is_trading_halted_fn() is False
    captured_service.handler.set_trading_halt_fn(True, "test_pause")
    assert captured_service.handler.is_trading_halted_fn() is True
    assert "proposal_manager" in captured_server_kwargs


def test_cli_run_wires_status_fn_executes_without_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    from xau_bot.cli.main import build_parser
    from xau_bot.cli.run import run_bot_cli

    parser = build_parser()
    args = parser.parse_args(["run", "--broker", "sim"])

    monkeypatch.setenv("XAU_TELEGRAM_TOKEN", "bot123456:ABC-DEF")
    monkeypatch.setenv("XAU_TELEGRAM_CHAT_ID", "999888")

    captured_service: Any = None

    class MockRunner:
        def __init__(self, service: Any) -> None:
            nonlocal captured_service
            captured_service = service

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("xau_bot.cli.run.TelegramRunner", MockRunner)
    monkeypatch.setattr("xau_bot.cli.run.run_server", lambda **kwargs: None)

    exit_code = run_bot_cli(args, stdout=io.StringIO(), stderr=io.StringIO())
    assert exit_code == 0
    assert captured_service is not None
    assert captured_service.handler.get_status_fn is not None

    status_result: dict[str, Any] = {}
    thread_error: list[Exception] = []

    def _call_status() -> None:
        try:
            status_result.update(captured_service.handler.get_status_fn())
        except Exception as exc:
            thread_error.append(exc)

    thread = threading.Thread(target=_call_status, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive(), "get_status_fn deadlocked!"
    assert not thread_error
    assert "broker_connected" in status_result
    assert "trading_halted" in status_result


@pytest.mark.asyncio
async def test_close_all_cancels_orders_even_if_positions_fail() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)
    mock_broker = MagicMock()
    mock_broker.get_open_positions = AsyncMock(side_effect=RuntimeError("Positions query failed"))

    cancelled = []

    async def mock_cancel(ticket: int) -> OrderResult:
        cancelled.append(ticket)
        return OrderResult(success=True, ticket=ticket, retcode=10009)

    mock_broker.get_open_orders = AsyncMock(
        return_value=[
            BrokerOrder(
                ticket=555,
                symbol="XAUUSD",
                order_type=OrderType.BUY_LIMIT,
                volume=0.1,
                price=2640.0,
                stop_loss=2630.0,
                take_profit=2660.0,
                time_setup=datetime.now(UTC),
            )
        ]
    )
    mock_broker.cancel_order = mock_cancel

    halt_called = False

    def mock_set_halt(halt: bool, reason: str) -> None:
        nonlocal halt_called
        halt_called = True

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        broker=mock_broker,
        set_trading_halt_fn=mock_set_halt,
    )

    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/fechar_tudo")
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg))

    assert halt_called is True
    # Orders should have been cancelled even though get_open_positions failed!
    assert 555 in cancelled
    assert client.send_message.call_count >= 1
    # Check that exception text is sanitized (not exposing raw internal traceback)
    for call in client.send_message.call_args_list:
        text = call[1]["text"]
        assert "Positions query failed" not in text


@pytest.mark.asyncio
async def test_status_displays_kill_switches_and_real_telemetry() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        get_status_fn=lambda: {
            "broker_connected": True,
            "trading_halted": True,
            "execution_mode": "auto",
            "daily_pnl": -45.20,
            "kill_switches_active": True,
            "proposals": {"pending": 1, "approved": 2},
        },
        is_trading_halted_fn=lambda: True,
    )

    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="/status")
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg))

    text = client.send_message.call_args[1]["text"]
    assert "Kill Switches" in text
    assert "ATIVADO" in text
    assert "$-45.20" in text
    assert "HALTED / PAUSADO" in text


@pytest.mark.asyncio
async def test_edit_message_text_retries_without_markdown_on_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TelegramBotClient(token="token123", chat_id="chat123")
    calls = []

    class MockHTTPError(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(
                "http://api.telegram.org",
                400,
                "Bad Request",
                email.message.Message(),
                io.BytesIO(b""),
            )

    def mock_urlopen(req: urllib.request.Request, timeout: float = 10.0) -> Any:
        calls.append(req)
        assert isinstance(req.data, bytes)
        data = json.loads(req.data.decode("utf-8"))
        if "parse_mode" in data:
            raise MockHTTPError()
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.read.return_value = json.dumps({"ok": True, "result": True}).encode("utf-8")
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    success = await client.edit_message_text(
        message_id=10,
        text="Text with _unescaped_underscore_test_",
        parse_mode="Markdown",
    )
    assert success is True
    assert len(calls) == 2
    # First call had parse_mode, second retry had parse_mode stripped
    assert isinstance(calls[0].data, bytes)
    assert isinstance(calls[1].data, bytes)
    data1 = json.loads(calls[0].data.decode("utf-8"))
    assert "parse_mode" in data1
    data2 = json.loads(calls[1].data.decode("utf-8"))
    assert "parse_mode" not in data2


@pytest.mark.asyncio
async def test_callback_query_group_chat_authorization() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)
    pm = MagicMock(spec=ProposalManager)
    mock_prop = MagicMock(spec=TradeProposal)
    mock_prop.id = "prop_100"
    mock_prop.mode = ExecutionMode.SEMIAUTO
    mock_prop.status = ProposalStatus.APPROVED
    mock_prop.direction = Direction.BULLISH
    mock_prop.symbol = "XAUUSD"
    mock_prop.entry_price = 2650.0
    mock_prop.stop_loss = 2640.0
    mock_prop.take_profit = 2670.0
    mock_prop.volume = 0.1
    mock_prop.risk_usd = 100.0
    mock_prop.risk_pct = 0.01
    mock_prop.expires_at = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
    mock_prop.order_ticket = None
    mock_prop.reason = None
    mock_prop.setup = None
    mock_prop.historical_stats = None
    pm.approve_proposal = MagicMock(return_value=(True, "approved", mock_prop))
    pm.get_proposal = MagicMock(return_value=mock_prop)

    # Group chat ID is negative
    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="-100123456789",
        proposal_manager=pm,
    )

    # Callback where from_user.id is individual user (999)
    # but message.chat.id is authorized group (-100123456789)
    group_chat = TelegramChat(id=-100123456789, type="group")
    msg = TelegramMessage(message_id=42, date=1700000000, chat=group_chat, text="Proposal msg")
    cb = CallbackQuery(
        id="cb_group_1",
        from_user=TelegramUser(id=999, first_name="Trader"),
        message=msg,
        data="approve:prop_100",
    )

    await handler.handle_update(TelegramUpdate(update_id=1, callback_query=cb))
    # Should NOT be rejected as unauthorized
    assert pm.approve_proposal.called


@pytest.mark.asyncio
async def test_unknown_callback_query_answers() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=True)

    handler = TelegramCommandHandler(bot_client=client, authorized_chat_id="12345")
    auth_chat = TelegramChat(id=12345, type="private")
    msg = TelegramMessage(message_id=1, date=1700000000, chat=auth_chat, text="Some msg")
    cb = CallbackQuery(
        id="cb_unknown",
        from_user=TelegramUser(id=12345),
        message=msg,
        data="unknown_action:foo",
    )
    await handler.handle_update(TelegramUpdate(update_id=1, callback_query=cb))
    assert client.answer_callback_query.called
    assert "não reconhecido" in client.answer_callback_query.call_args[1]["text"]


@pytest.mark.asyncio
async def test_lifecycle_notifications() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.is_configured = True
    client.send_message = AsyncMock(
        return_value=TelegramMessage(
            message_id=99,
            date=1700000000,
            chat=TelegramChat(id=12345, type="private"),
            text="alert",
        )
    )

    service = TelegramService(bot_client=client, command_handler=MagicMock())

    # 1. Order filled
    res_fill = await service.send_order_filled(
        ticket=12345,
        symbol="XAUUSD",
        direction="BUY",
        volume=0.05,
        price=2650.50,
    )
    assert res_fill is not None
    assert "FILL" in client.send_message.call_args[1]["text"]

    # 2. Break-even activated
    res_be = await service.send_break_even_activated(
        ticket=12345,
        symbol="XAUUSD",
        price=2650.50,
    )
    assert res_be is not None
    assert "BREAK-EVEN" in client.send_message.call_args[1]["text"]

    # 3. Trailing stop updated
    res_ts = await service.send_trailing_stop_updated(
        ticket=12345,
        symbol="XAUUSD",
        new_sl=2655.00,
    )
    assert res_ts is not None
    assert "TRAILING STOP" in client.send_message.call_args[1]["text"]

    # 4. Trade closed
    res_close = await service.send_trade_closed(
        ticket=12345,
        symbol="XAUUSD",
        profit_usd=150.00,
        profit_r=2.5,
        exit_price=2675.00,
    )
    assert res_close is not None
    assert "TRADE ENCERRADO" in client.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_status_displays_broker_connected_with_real_health_state() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)
    health_state = HealthStatusState(broker_connected=True, execution_mode="semiauto")
    health_state.trading_halted = False
    health_state.daily_pnl = 150.0

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        get_status_fn=lambda: dict(health_state.get_status()[1]),
    )
    msg = TelegramMessage(
        message_id=1,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="/status",
    )
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg))
    sent_text = client.send_message.call_args[1]["text"]
    assert "• *Broker:* 🟢 Conectado" in sent_text
    assert "• *Kill Switches:* 🟢 NORMAL" in sent_text


@pytest.mark.asyncio
async def test_close_all_resilient_with_exception_on_first_of_three_positions() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    pos1 = BrokerPosition(
        ticket=101,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        current_price=2655.0,
        floating_pnl=50.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    pos2 = BrokerPosition(
        ticket=102,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        current_price=2655.0,
        floating_pnl=50.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )
    pos3 = BrokerPosition(
        ticket=103,
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        volume=0.1,
        open_price=2650.0,
        current_price=2655.0,
        floating_pnl=50.0,
        open_time=datetime(2026, 10, 4, 12, 0, tzinfo=UTC),
    )

    closed_tickets = []

    async def mock_close(ticket: int) -> OrderResult:
        if ticket == 101:
            raise RuntimeError("MT5 IPC timeout")
        closed_tickets.append(ticket)
        return OrderResult(success=True, ticket=ticket, retcode=10009)

    mock_broker = MagicMock()
    mock_broker.get_open_positions = AsyncMock(return_value=[pos1, pos2, pos3])
    mock_broker.close_position = AsyncMock(side_effect=mock_close)
    mock_broker.get_open_orders = AsyncMock(return_value=[])

    halt_calls = []

    def mock_set_halt(halted: bool, reason: str) -> None:
        halt_calls.append((halted, reason))

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        broker=mock_broker,
        set_trading_halt_fn=mock_set_halt,
    )

    msg = TelegramMessage(
        message_id=2,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="/fechar_tudo",
    )
    await handler.handle_update(TelegramUpdate(update_id=2, message=msg))

    assert closed_tickets == [102, 103]
    assert len(halt_calls) == 1
    sent_text = client.send_message.call_args[1]["text"]
    assert "FALHA NO FECHAMENTO DE EMERGÊNCIA" in sent_text
    assert "• Posições Encerradas: `2`" in sent_text
    assert "• Falhas no Fechamento: `1`" in sent_text
    assert "O robô está seguro em halt" in sent_text


@pytest.mark.asyncio
async def test_approve_callback_blocked_when_trading_halted() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.answer_callback_query = AsyncMock(return_value=None)
    client.edit_message_text = AsyncMock(return_value=None)

    pm = ProposalManager(default_timeout_s=300)
    setup = make_test_setup("setup_halt_1")
    intent = make_test_intent(setup)
    prop = pm.create_proposal(setup=setup, intent=intent, mode=ExecutionMode.SEMIAUTO)

    executor = MagicMock()
    executor.execute_intent = AsyncMock()

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        proposal_manager=pm,
        order_executor=executor,
        is_trading_halted_fn=lambda: True,
    )

    cb = CallbackQuery(
        id="cb_halt",
        from_user=TelegramUser(id=12345, is_bot=False, first_name="Test"),
        message=TelegramMessage(
            message_id=10,
            date=1700000000,
            chat=TelegramChat(id=12345, type="private"),
            text="Proposal",
        ),
        data=f"approve:{prop.id}",
    )
    await handler.handle_update(TelegramUpdate(update_id=3, callback_query=cb))

    assert client.answer_callback_query.call_count == 1
    call_args = client.answer_callback_query.call_args[1]
    assert "HALT" in call_args["text"]
    assert call_args["show_alert"] is True
    saved = pm.get_proposal(prop.id)
    assert saved is not None
    assert saved.status == ProposalStatus.PENDING_APPROVAL


@pytest.mark.asyncio
async def test_resume_blocked_when_kill_switches_active() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    halt_state = {"halted": True}

    def mock_set_halt(val: bool, reason: str) -> None:
        halt_state["halted"] = val

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        set_trading_halt_fn=mock_set_halt,
        is_trading_halted_fn=lambda: halt_state["halted"],
        get_status_fn=lambda: {"kill_switches_active": True},
    )

    msg = TelegramMessage(
        message_id=3,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="/retomar",
    )
    await handler.handle_update(TelegramUpdate(update_id=4, message=msg))

    assert halt_state["halted"] is True
    sent_text = client.send_message.call_args[1]["text"]
    assert "RETOMADA BLOQUEADA: KILL SWITCH ATIVO" in sent_text


@pytest.mark.asyncio
async def test_dispatcher_ignores_setup_when_trading_halted() -> None:
    pm = ProposalManager(default_timeout_s=300)
    dispatcher = AdvisorDispatcher(
        proposal_manager=pm,
        mode=ExecutionMode.SEMIAUTO,
        is_trading_halted_fn=lambda: True,
    )
    setup = make_test_setup("setup_halt_ignored")
    prop = await dispatcher.handle_setup(setup)
    assert prop is None
    assert len(pm.get_proposals()) == 0


def test_watchdog_alerts_reach_telegram_alert_channel_with_telegram_service_active() -> None:
    settings = Settings(
        telegram_token=SecretStr("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"),
        telegram_chat_id=SecretStr("987654321"),
    )
    out = io.StringIO()
    alert_mgr = create_alert_manager(settings, stdout=out, enable_telegram=True)
    channel_types = [type(ch).__name__ for ch in alert_mgr.channels]
    assert "TelegramAlertChannel" in channel_types


@pytest.mark.asyncio
async def test_command_handler_resume_handles_get_status_exception() -> None:
    client = MagicMock(spec=TelegramBotClient)
    client.send_message = AsyncMock(return_value=None)

    def failing_status() -> dict[str, Any]:
        raise RuntimeError("Status service offline")

    handler = TelegramCommandHandler(
        bot_client=client,
        authorized_chat_id="12345",
        get_status_fn=failing_status,
    )
    msg = TelegramMessage(
        message_id=1,
        date=1700000000,
        chat=TelegramChat(id=12345, type="private"),
        text="/retomar",
    )
    await handler.handle_update(TelegramUpdate(update_id=1, message=msg))
    client.send_message.assert_awaited_once()
    sent_text = client.send_message.call_args[1]["text"]
    assert "Não foi possível verificar o estado operacional" in sent_text
