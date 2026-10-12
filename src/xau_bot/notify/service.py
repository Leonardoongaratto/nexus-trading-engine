"""Telegram Service factory and threading runner."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any

from xau_bot.broker.ports import BrokerPort
from xau_bot.config import Settings
from xau_bot.notify.telegram import TelegramBotClient, TelegramCommandHandler, TelegramService

logger = logging.getLogger(__name__)


class TelegramRunner:
    def __init__(self, service: TelegramService) -> None:
        self.service = service
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.service.client.is_configured:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._loop = asyncio.new_event_loop()

        def _worker(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.create_task(self.service.start())
                loop.run_forever()
            except Exception as exc:
                logger.error("Error in TelegramRunner thread: %s", exc)
            finally:
                with contextlib.suppress(Exception):
                    loop.close()

        self._thread = threading.Thread(
            target=_worker,
            args=(self._loop,),
            daemon=True,
            name="TelegramRunner",
        )
        self._thread.start()
        logger.info("TelegramRunner thread started.")

    def stop(self, timeout: float = 2.0) -> None:
        if self._loop is not None and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.service.stop(), self._loop)
                future.result(timeout=timeout)
            except Exception as exc:
                logger.debug("Failed awaiting service.stop in TelegramRunner: %s", exc)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("TelegramRunner thread stopped.")


def build_telegram_service(
    settings: Settings,
    proposal_manager: Any | None = None,
    order_executor: Any | None = None,
    broker: BrokerPort | None = None,
    get_status_fn: Callable[[], Mapping[str, Any]] | None = None,
    set_trading_halt_fn: Callable[[bool, str], None] | None = None,
    is_trading_halted_fn: Callable[[], bool] | None = None,
) -> TelegramService | None:
    if not (settings.telegram_token and settings.telegram_chat_id):
        return None

    bot_client = TelegramBotClient(
        token=settings.telegram_token,
        chat_id=settings.telegram_chat_id,
    )
    handler = TelegramCommandHandler(
        bot_client=bot_client,
        authorized_chat_id=settings.telegram_chat_id.get_secret_value(),
        proposal_manager=proposal_manager,
        order_executor=order_executor,
        broker=broker,
        get_status_fn=get_status_fn,
        set_trading_halt_fn=set_trading_halt_fn,
        is_trading_halted_fn=is_trading_halted_fn,
    )
    service = TelegramService(bot_client=bot_client, command_handler=handler)
    if proposal_manager is not None and hasattr(proposal_manager, "add_listener"):
        proposal_manager.add_listener(service.notify_proposal_change_sync)
    return service
