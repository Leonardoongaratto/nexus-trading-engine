"""Telegram notification and remote intervention module for XAU/USD trading bot."""

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

__all__ = [
    "CallbackQuery",
    "InlineKeyboardButton",
    "InlineKeyboardMarkup",
    "TelegramBotClient",
    "TelegramChat",
    "TelegramCommandHandler",
    "TelegramMessage",
    "TelegramRunner",
    "TelegramService",
    "TelegramUpdate",
    "TelegramUser",
    "build_telegram_service",
    "create_proposal_keyboard",
    "escape_markdown",
    "format_proposal_message",
]
