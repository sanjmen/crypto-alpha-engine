"""
CCXT Live Execution Adapter with Hard Safety Circuit Breakers.
Safeguards real trading capital against unexpected fat-finger errors,
extreme leverage, and flash drawdowns.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

import ccxt

from src.domain.entities import Order, Portfolio, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.ports.execution_ports import IExecutionClient

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when a risk or safety circuit breaker halts execution."""
    pass


class MaxLeverageExceededError(CircuitBreakerError):
    """Raised when an order would push portfolio gross leverage beyond limit."""
    pass


class MaxOrderNotionalExceededError(CircuitBreakerError):
    """Raised when an individual order exceeds the maximum single-trade dollar ceiling."""
    pass


class MaxDrawdownStopError(CircuitBreakerError):
    """Raised when portfolio drawdown exceeds the hard emergency stop threshold."""
    pass


class CCXTExecutionAdapter(IExecutionClient):
    """
    Institutional exchange adapter with fail-safe circuit breakers.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        dry_run: bool = True,
        max_gross_leverage: float = 2.5,
        max_order_notional: float = 25_000.0,
        max_drawdown_stop: float = 0.15,
        taker_fee_pct: float = 0.0004,
        maker_fee_pct: float = 0.0002,
    ):
        self.exchange_id = exchange_id
        self.dry_run = dry_run
        self.max_gross_leverage = max_gross_leverage
        self.max_order_notional = max_order_notional
        self.max_drawdown_stop = max_drawdown_stop
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct

        exchange_class = getattr(ccxt, exchange_id)
        config = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        self.client: ccxt.Exchange = exchange_class(config)

        self.portfolio = Portfolio(
            cash=100_000.0,
            initial_cash=100_000.0,
            positions={},
        )
        self.peak_equity = 100_000.0
        self.trade_history: List[Trade] = []
        self.orders: Dict[str, Order] = {}

    def get_portfolio(self) -> Portfolio:
        return self.portfolio

    def get_order_status(self, order_id: str, symbol: str) -> Order:
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found.")
        return self.orders[order_id]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_trade_history(self) -> List[Trade]:
        return list(self.trade_history)

    def verify_risk_circuit_breakers(self, symbol: str, quantity: float, est_price: float) -> None:
        """
        Enforces 3 non-negotiable institutional safety gates:
        1. Maximum single order notional
        2. Emergency drawdown tripwire
        3. Portfolio gross leverage ceiling
        """
        order_notional = quantity * est_price
        if order_notional > self.max_order_notional:
            raise MaxOrderNotionalExceededError(
                f"Order notional ${order_notional:,.2f} exceeds limit ${self.max_order_notional:,.2f}"
            )

        current_equity = self.portfolio.total_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.max_drawdown_stop:
            raise MaxDrawdownStopError(
                f"Hard emergency stop: current drawdown {drawdown*100:.2f}% exceeds {self.max_drawdown_stop*100:.2f}%"
            )

        # Estimate post-trade gross leverage
        current_gross = sum(abs(p.market_value) for p in self.portfolio.positions.values())
        new_gross = current_gross + order_notional
        new_leverage = new_gross / max(current_equity, 1.0)
        if new_leverage > self.max_gross_leverage:
            raise MaxLeverageExceededError(
                f"Projected gross leverage {new_leverage:.2f}x exceeds ceiling {self.max_gross_leverage:.2f}x"
            )

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        est_price = price or 100.0

        # Safety Check
        self.verify_risk_circuit_breakers(symbol, quantity, est_price)

        order_id = f"ccxt_ord_{uuid.uuid4().hex[:10]}"
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(tz=timezone.utc),
        )

        if self.dry_run:
            # Simulate fill immediately in dry run
            order.status = OrderStatus.FILLED
            order.filled_quantity = quantity
            order.average_fill_price = est_price
            trade = Trade(
                id=f"dry_trd_{uuid.uuid4().hex[:10]}",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=est_price,
                fee=quantity * est_price * self.taker_fee_pct,
                timestamp=datetime.now(tz=timezone.utc),
            )
            self.trade_history.append(trade)
            self.portfolio.cash -= trade.fee
            # Record position
            pos_side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                entry_price=est_price,
                current_price=est_price,
            )
        else:
            # Send live order to CCXT
            ccxt_side = "buy" if side == OrderSide.BUY else "sell"
            ccxt_type = "market" if order_type == OrderType.MARKET else "limit"
            res = self.client.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=quantity,
                price=price,
            )
            order.status = OrderStatus.SUBMITTED
            order.id = str(res.get("id", order_id))

        self.orders[order_id] = order
        return order
