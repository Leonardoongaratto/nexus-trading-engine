from typing import Protocol

from xau_bot.broker.models import (
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    OrderRequest,
    OrderResult,
    SymbolInfo,
)
from xau_bot.domain.models import Candle, Timeframe


class BrokerPort(Protocol):
    async def connect(self) -> bool: ...

    async def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    async def get_account_info(self) -> AccountInfo: ...

    async def get_symbol_info(self, symbol: str) -> SymbolInfo | None: ...

    async def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int = 100,
    ) -> tuple[Candle, ...]: ...

    async def send_order(self, request: OrderRequest) -> OrderResult: ...

    async def modify_order(
        self,
        ticket: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrderResult: ...

    async def cancel_order(self, ticket: int) -> OrderResult: ...

    async def close_position(self, ticket: int, volume: float | None = None) -> OrderResult: ...

    async def get_open_positions(self) -> tuple[BrokerPosition, ...]: ...

    async def get_open_orders(self) -> tuple[BrokerOrder, ...]: ...
