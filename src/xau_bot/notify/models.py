"""Data models for Telegram Bot API integration and remote intervention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class InlineKeyboardButton:
    text: str
    callback_data: str

    def to_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "callback_data": self.callback_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InlineKeyboardButton:
        return cls(
            text=str(data.get("text", "")),
            callback_data=str(data.get("callback_data", "")),
        )


@dataclass(slots=True, frozen=True)
class InlineKeyboardMarkup:
    inline_keyboard: list[list[InlineKeyboardButton]] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[list[dict[str, str]]]]:
        return {"inline_keyboard": [[btn.to_dict() for btn in row] for row in self.inline_keyboard]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InlineKeyboardMarkup:
        rows_raw = data.get("inline_keyboard", [])
        rows = [
            [InlineKeyboardButton.from_dict(btn) for btn in row]
            for row in rows_raw
            if isinstance(row, list)
        ]
        return cls(inline_keyboard=rows)


@dataclass(slots=True, frozen=True)
class TelegramUser:
    id: int
    is_bot: bool = False
    first_name: str = ""
    last_name: str | None = None
    username: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "is_bot": self.is_bot,
            "first_name": self.first_name,
        }
        if self.last_name is not None:
            result["last_name"] = self.last_name
        if self.username is not None:
            result["username"] = self.username
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramUser:
        return cls(
            id=int(data.get("id", 0)),
            is_bot=bool(data.get("is_bot", False)),
            first_name=str(data.get("first_name", "")),
            last_name=data.get("last_name"),
            username=data.get("username"),
        )


@dataclass(slots=True, frozen=True)
class TelegramChat:
    id: int | str
    type: str = "private"
    title: str | None = None
    username: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
        }
        if self.title is not None:
            result["title"] = self.title
        if self.username is not None:
            result["username"] = self.username
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramChat:
        return cls(
            id=data.get("id", 0),
            type=str(data.get("type", "private")),
            title=data.get("title"),
            username=data.get("username"),
        )


@dataclass(slots=True, frozen=True)
class TelegramMessage:
    message_id: int
    chat: TelegramChat
    date: int = 0
    from_user: TelegramUser | None = None
    text: str | None = None
    reply_markup: InlineKeyboardMarkup | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "message_id": self.message_id,
            "chat": self.chat.to_dict(),
            "date": self.date,
        }
        if self.from_user is not None:
            result["from"] = self.from_user.to_dict()
        if self.text is not None:
            result["text"] = self.text
        if self.reply_markup is not None:
            result["reply_markup"] = self.reply_markup.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramMessage:
        chat = TelegramChat.from_dict(data.get("chat", {}))
        from_user = (
            TelegramUser.from_dict(data["from"])
            if "from" in data and isinstance(data["from"], dict)
            else None
        )
        reply_markup = (
            InlineKeyboardMarkup.from_dict(data["reply_markup"])
            if "reply_markup" in data and isinstance(data["reply_markup"], dict)
            else None
        )
        return cls(
            message_id=int(data.get("message_id", 0)),
            chat=chat,
            date=int(data.get("date", 0)),
            from_user=from_user,
            text=data.get("text"),
            reply_markup=reply_markup,
        )


@dataclass(slots=True, frozen=True)
class CallbackQuery:
    id: str
    from_user: TelegramUser
    message: TelegramMessage | None = None
    data: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "from": self.from_user.to_dict(),
        }
        if self.message is not None:
            result["message"] = self.message.to_dict()
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallbackQuery:
        from_user = TelegramUser.from_dict(data.get("from", {}))
        message = (
            TelegramMessage.from_dict(data["message"])
            if "message" in data and isinstance(data["message"], dict)
            else None
        )
        return cls(
            id=str(data.get("id", "")),
            from_user=from_user,
            message=message,
            data=data.get("data"),
        )


@dataclass(slots=True, frozen=True)
class TelegramUpdate:
    update_id: int
    message: TelegramMessage | None = None
    callback_query: CallbackQuery | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "update_id": self.update_id,
        }
        if self.message is not None:
            result["message"] = self.message.to_dict()
        if self.callback_query is not None:
            result["callback_query"] = self.callback_query.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelegramUpdate:
        message = (
            TelegramMessage.from_dict(data["message"])
            if "message" in data and isinstance(data["message"], dict)
            else None
        )
        callback_query = (
            CallbackQuery.from_dict(data["callback_query"])
            if "callback_query" in data and isinstance(data["callback_query"], dict)
            else None
        )
        return cls(
            update_id=int(data.get("update_id", 0)),
            message=message,
            callback_query=callback_query,
        )
