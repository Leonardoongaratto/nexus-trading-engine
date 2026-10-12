from xau_bot.broker.ports import BrokerPort
from xau_bot.execution.models import ReconciliationResult


class OrderReconciler:
    def __init__(self, broker: BrokerPort) -> None:
        self._broker = broker

    async def reconcile_order(
        self,
        magic: int,
        comment: str,
        ticket: int | None = None,
    ) -> ReconciliationResult:
        positions = await self._broker.get_open_positions()
        for pos in positions:
            if ticket is not None and pos.ticket == ticket:
                return ReconciliationResult(
                    found=True,
                    is_position=True,
                    is_order=False,
                    ticket=pos.ticket,
                    position=pos,
                )
            if pos.magic == magic and pos.comment == comment:
                return ReconciliationResult(
                    found=True,
                    is_position=True,
                    is_order=False,
                    ticket=pos.ticket,
                    position=pos,
                )

        orders = await self._broker.get_open_orders()
        for ord_item in orders:
            if ticket is not None and ord_item.ticket == ticket:
                return ReconciliationResult(
                    found=True,
                    is_position=False,
                    is_order=True,
                    ticket=ord_item.ticket,
                    order=ord_item,
                )
            if ord_item.magic == magic and ord_item.comment == comment:
                return ReconciliationResult(
                    found=True,
                    is_position=False,
                    is_order=True,
                    ticket=ord_item.ticket,
                    order=ord_item,
                )

        return ReconciliationResult(found=False)
