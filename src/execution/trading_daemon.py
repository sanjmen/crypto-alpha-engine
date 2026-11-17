"""
Live Trading Daemon & Portfolio Rebalancer.
Periodically fetches real-time market data, invokes the Meta-Strategy Allocator,
reconciles target positions with current broker inventory, and routes rebalance orders.
"""

from datetime import datetime, timezone
import logging

import pandas as pd

from src.domain.entities import Order
from src.domain.enums import OrderSide, OrderType, PositionSide
from src.ports.execution_ports import IExecutionClient
from src.strategies.meta_allocator import MetaStrategyAllocator

logger = logging.getLogger(__name__)


class TradingDaemon:
    """
    Automated execution daemon for crypto systematic alphas.
    """

    def __init__(
        self,
        broker: IExecutionClient,
        allocator: MetaStrategyAllocator | None = None,
        min_rebalance_notional: float = 50.0,
    ) -> None:
        self.broker = broker
        self.allocator = allocator or MetaStrategyAllocator()
        self.min_rebalance_notional = min_rebalance_notional
        self.last_rebalance_time: datetime | None = None

    def rebalance_cycle(
        self,
        market_data: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        volatilities: dict[str, float],
        funding_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[Order]:
        """
        Executes a single rebalancing iteration:
        1. Queries current broker portfolio
        2. Computes meta-strategy target dollar allocations
        3. Diff-checks target vs current positions
        4. Submits market orders for rebalancing
        """
        portfolio = self.broker.get_portfolio()
        target_dollars, _regime, _weights = self.allocator.generate_portfolio_allocations(
            market_data=market_data,
            portfolio=portfolio,
            volatilities=volatilities,
            funding_data=funding_data,
        )

        executed_orders: list[Order] = []
        all_symbols = set(target_dollars.keys()) | set(portfolio.positions.keys())

        for sym in all_symbols:
            price = current_prices.get(sym)
            if not price or price <= 0:
                continue

            target_val = target_dollars.get(sym, 0.0)
            current_pos = portfolio.positions.get(sym)
            current_qty = current_pos.quantity if current_pos else 0.0
            if current_pos and current_pos.side == PositionSide.SHORT:
                current_val = -abs(current_qty) * price
            elif current_pos and current_pos.side == PositionSide.LONG:
                current_val = abs(current_qty) * price
            else:
                current_val = 0.0

            dollar_delta = target_val - current_val
            if abs(dollar_delta) < self.min_rebalance_notional:
                continue

            order_qty = abs(dollar_delta) / price
            side = OrderSide.BUY if dollar_delta > 0 else OrderSide.SELL

            order = self.broker.create_order(
                symbol=sym,
                side=side,
                order_type=OrderType.MARKET,
                quantity=order_qty,
                price=price,
            )
            executed_orders.append(order)

        self.last_rebalance_time = datetime.now(tz=timezone.utc)
        return executed_orders
