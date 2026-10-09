"""
Execution & Broker Ports (Clean Architecture).
Defines abstract contracts for order execution, order management,
and portfolio state tracking across simulated paper trading and live exchanges.
"""

from typing import Dict, List, Optional, Protocol
from src.domain.entities import Order, Portfolio, Trade
from src.domain.enums import OrderSide, OrderType


class IExecutionClient(Protocol):
    """
    Abstract interface for order execution engines and exchange brokers.
    """

    def get_portfolio(self) -> Portfolio:
        """Retrieves current portfolio state (cash, positions, unrealized PnL)."""
        ...

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """Submits an order to the execution venue."""
        ...

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels an active open order."""
        ...

    def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Queries the current status of an order."""
        ...

    def get_trade_history(self) -> List[Trade]:
        """Retrieves history of executed trades."""
        ...
