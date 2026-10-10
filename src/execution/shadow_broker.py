"""
Institutional Shadow Trading Broker.
Simulates realistic matching against live Binance Mainnet prices,
enforcing hard safety circuit breakers, passive Maker touch execution,
taker slippage, 8h funding cashflows, and local SQLite persistence.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from src.domain.entities import Order, Portfolio, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.live_broker import (
    CircuitBreakerError,
    MaxDrawdownStopError,
    MaxLeverageExceededError,
    MaxOrderNotionalExceededError,
)
from src.ports.execution_ports import IExecutionClient

logger = logging.getLogger(__name__)


class ShadowExecutionBroker(IExecutionClient):
    """
    Simulated matching broker backed by SQLite WAL persistence.
    Consumes live market prices without risking capital.
    """

    def __init__(
        self,
        database: Optional[ShadowDatabase] = None,
        initial_cash: float = 100_000.0,
        taker_fee_pct: float = 0.0004,     # 0.04% taker
        maker_fee_pct: float = 0.0002,     # 0.02% maker
        slippage_bps: float = 1.0,         # 1 bp slippage for market orders
        max_gross_leverage: float = 2.5,   # Hard circuit breaker
        max_order_notional: float = 25_000.0,
        max_drawdown_stop: float = 0.15,   # 15% drawdown halts execution
    ):
        self.db = database or ShadowDatabase()
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.slippage = slippage_bps / 10_000.0
        self.max_gross_leverage = max_gross_leverage
        self.max_order_notional = max_order_notional
        self.max_drawdown_stop = max_drawdown_stop

        # Restore from database or initialize fresh
        latest_acc = self.db.get_latest_account()
        if latest_acc:
            cash = float(latest_acc["cash_balance"])
            init_cash = float(latest_acc["initial_capital"])
        else:
            cash = initial_cash
            init_cash = initial_cash

        open_positions = self.db.get_open_positions()

        self.portfolio = Portfolio(
            cash=cash,
            initial_cash=init_cash,
            positions=open_positions,
        )
        self.peak_equity = max(self.portfolio.total_equity, init_cash)
        self.market_prices: Dict[str, float] = {}
        self.market_bids: Dict[str, float] = {}
        self.market_asks: Dict[str, float] = {}

        # If fresh db, record initial account snapshot
        if not latest_acc:
            self._record_account_snapshot()

    def set_market_price(
        self,
        symbol: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> None:
        """Updates best price levels for a symbol and mark-to-markets open positions."""
        p = float(price)
        self.market_prices[symbol] = p
        self.market_bids[symbol] = float(bid) if bid else p
        self.market_asks[symbol] = float(ask) if ask else p

        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            pos.current_price = p
            self.db.save_position(pos)

    def set_market_prices(self, prices: Dict[str, float]) -> None:
        """Batch updates mark prices."""
        for sym, price in prices.items():
            self.set_market_price(sym, price)

    def get_portfolio(self) -> Portfolio:
        """Returns the current portfolio snapshot."""
        return self.portfolio

    def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Queries order status from database."""
        order = self.db.get_order(order_id)
        if not order:
            raise KeyError(f"Order {order_id} not found in database.")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels a pending order in the database."""
        order = self.db.get_order(order_id)
        if order and order.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED):
            self.db.update_order_fill(
                order_id=order_id,
                status=OrderStatus.CANCELED,
                filled_quantity=order.filled_quantity,
                average_fill_price=order.average_fill_price,
                fee=0.0,
            )
            return True
        return False

    def get_trade_history(self) -> List[Trade]:
        """Returns executed trade records."""
        trade_rows = self.db.get_trades(limit=500)
        trades: List[Trade] = []
        for r in trade_rows:
            trades.append(
                Trade(
                    id=r["id"],
                    order_id=r["order_id"],
                    symbol=r["symbol"],
                    side=OrderSide(r["side"]),
                    quantity=float(r["quantity"]),
                    price=float(r["price"]),
                    fee=float(r["fee_paid"]),
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
            )
        return trades

    def verify_risk_circuit_breakers(self, symbol: str, quantity: float, est_price: float) -> None:
        """
        Enforces 3 non-negotiable safety circuit breakers before order creation.
        """
        order_notional = quantity * est_price
        if order_notional > self.max_order_notional:
            raise MaxOrderNotionalExceededError(
                f"Order notional ${order_notional:,.2f} exceeds ceiling ${self.max_order_notional:,.2f}"
            )

        current_equity = self.portfolio.total_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.max_drawdown_stop:
            raise MaxDrawdownStopError(
                f"Hard emergency stop: drawdown {drawdown*100:.2f}% exceeds {self.max_drawdown_stop*100:.2f}%"
            )

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
        role: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Submits an order through risk verification and matching.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        est_price = price or self.market_prices.get(symbol, 100.0)

        # 1. Enforce safety circuit breakers
        self.verify_risk_circuit_breakers(symbol, quantity, est_price)

        order_id = f"shd_ord_{uuid.uuid4().hex[:10]}"
        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        assigned_role = role or ("MAKER" if order_type == OrderType.LIMIT else "TAKER")
        self.db.save_order(order, role=assigned_role, client_order_id=client_order_id)

        # 2. Execution logic
        if order_type == OrderType.MARKET:
            self._fill_market_order(order)
        elif order_type == OrderType.LIMIT:
            self._evaluate_limit_order(order)

        return order

    def _fill_market_order(self, order: Order) -> None:
        """Executes an aggressive Taker order with slippage and 0.04% fee."""
        symbol = order.symbol
        if order.side == OrderSide.BUY:
            base_price = self.market_asks.get(symbol, self.market_prices.get(symbol, 100.0))
            fill_price = base_price * (1.0 + self.slippage)
        else:
            base_price = self.market_bids.get(symbol, self.market_prices.get(symbol, 100.0))
            fill_price = base_price * (1.0 - self.slippage)

        fee = fill_price * order.quantity * self.taker_fee_pct

        trade = Trade(
            id=f"shd_trd_{uuid.uuid4().hex[:10]}",
            order_id=order.id,
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(timezone.utc),
        )

        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price

        # Update DB order and trade
        self.db.update_order_fill(
            order_id=order.id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_fill_price=fill_price,
            fee=fee,
            role="TAKER",
        )
        realized_pnl = self._update_portfolio_position(trade)
        self.db.save_trade(trade, is_maker=False, realized_pnl=realized_pnl)
        self._record_account_snapshot()

    def _evaluate_limit_order(self, order: Order) -> None:
        """
        Evaluates a passive Maker limit order against current book touch.
        If current market already touches or crosses limit, fill immediately as Maker.
        Otherwise leaves as PENDING in database to be filled on subsequent ticks.
        """
        symbol = order.symbol
        limit_price = order.price or self.market_prices.get(symbol, 100.0)

        should_fill = False
        if order.side == OrderSide.BUY:
            current_ask = self.market_asks.get(symbol, self.market_prices.get(symbol))
            if current_ask is not None and current_ask <= limit_price:
                should_fill = True
        else:
            current_bid = self.market_bids.get(symbol, self.market_prices.get(symbol))
            if current_bid is not None and current_bid >= limit_price:
                should_fill = True

        if should_fill:
            self._fill_maker_order(order, fill_price=limit_price)

    def _fill_maker_order(self, order: Order, fill_price: float) -> None:
        """Executes a passive Maker limit order with zero slippage and 0.02% fee."""
        fee = fill_price * order.quantity * self.maker_fee_pct

        trade = Trade(
            id=f"shd_trd_{uuid.uuid4().hex[:10]}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(timezone.utc),
        )

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price

        self.db.update_order_fill(
            order_id=order.id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_fill_price=fill_price,
            fee=fee,
            role="MAKER",
        )
        realized_pnl = self._update_portfolio_position(trade)
        self.db.save_trade(trade, is_maker=True, realized_pnl=realized_pnl)
        self._record_account_snapshot()

    def check_pending_orders(self, symbol_ticks: Dict[str, Dict[str, float]]) -> List[Order]:
        """
        Evaluates all pending limit orders against fresh market ticks (bid/ask/low/high).
        Fills orders whose limits have been breached.
        """
        pending_orders = self.db.get_pending_orders()
        filled_orders: List[Order] = []

        for order in pending_orders:
            symbol = order.symbol
            tick = symbol_ticks.get(symbol)
            if not tick or order.price is None:
                continue

            limit_price = order.price
            low = tick.get("low", tick.get("bid", limit_price))
            high = tick.get("high", tick.get("ask", limit_price))

            should_fill = False
            if order.side == OrderSide.BUY:
                if low <= limit_price:
                    should_fill = True
            else:
                if high >= limit_price:
                    should_fill = True

            if should_fill:
                self._fill_maker_order(order, fill_price=limit_price)
                filled_orders.append(order)

        return filled_orders

    def settle_funding(
        self,
        symbol: str,
        funding_rate: float,
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Settles 8h perpetual funding payment for an open position.
        Cashflow = - Notional * funding_rate.
        """
        pos = self.portfolio.positions.get(symbol)
        if not pos or pos.side == PositionSide.FLAT or pos.quantity <= 0:
            return 0.0

        current_price = self.market_prices.get(symbol, pos.current_price)
        notional = pos.quantity * current_price

        if pos.side == PositionSide.LONG:
            cashflow = -notional * funding_rate
            side_str = "LONG"
        else:
            cashflow = notional * funding_rate
            side_str = "SHORT"

        self.portfolio.cash += cashflow

        self.db.record_funding_settlement(
            symbol=symbol,
            funding_rate=funding_rate,
            position_side=side_str,
            position_notional=notional,
            cashflow_credited=cashflow,
            timestamp=timestamp,
        )
        self._record_account_snapshot(timestamp)
        return cashflow

    def mark_to_market(
        self,
        prices: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> Portfolio:
        """
        Updates mark prices and refreshes unrealized PnL, equity, and account state.
        """
        self.set_market_prices(prices)
        self._record_account_snapshot(timestamp)
        return self.portfolio

    def _update_portfolio_position(self, trade: Trade) -> float:
        """
        Updates in-memory portfolio position and deducts fee.
        Calculates realized PnL on position reduction or closure.
        """
        symbol = trade.symbol
        pos = self.portfolio.positions.get(symbol)
        self.portfolio.cash -= trade.fee
        realized_pnl = 0.0

        if pos is None or pos.side == PositionSide.FLAT or pos.quantity == 0:
            side = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
            pos = Position(
                symbol=symbol,
                side=side,
                quantity=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
            )
            self.portfolio.positions[symbol] = pos
        else:
            # Existing position
            is_increasing = (
                (pos.side == PositionSide.LONG and trade.side == OrderSide.BUY)
                or (pos.side == PositionSide.SHORT and trade.side == OrderSide.SELL)
            )

            if is_increasing:
                total_qty = pos.quantity + trade.quantity
                avg_entry = (pos.quantity * pos.entry_price + trade.quantity * trade.price) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_entry
                pos.current_price = trade.price
            else:
                # Reducing or reversing position
                closed_qty = min(pos.quantity, trade.quantity)
                if pos.side == PositionSide.LONG:
                    realized_pnl = closed_qty * (trade.price - pos.entry_price)
                else:
                    realized_pnl = closed_qty * (pos.entry_price - trade.price)

                self.portfolio.cash += realized_pnl

                if trade.quantity < pos.quantity:
                    pos.quantity -= trade.quantity
                    pos.current_price = trade.price
                elif trade.quantity == pos.quantity:
                    pos.quantity = 0.0
                    pos.side = PositionSide.FLAT
                    pos.current_price = trade.price
                else:
                    # Reversed side
                    remaining_qty = trade.quantity - pos.quantity
                    pos.side = PositionSide.SHORT if pos.side == PositionSide.LONG else PositionSide.LONG
                    pos.quantity = remaining_qty
                    pos.entry_price = trade.price
                    pos.current_price = trade.price

        # Update DB position record
        self.db.save_position(pos, realized_pnl=realized_pnl)
        return realized_pnl

    def _record_account_snapshot(self, timestamp: Optional[datetime] = None) -> None:
        """Calculates portfolio aggregates and persists an account snapshot."""
        equity = self.portfolio.total_equity
        margin_used = sum(abs(p.market_value) for p in self.portfolio.positions.values())
        free_margin = max(0.0, equity - margin_used)
        leverage = self.portfolio.gross_leverage
        unrealized = sum(p.unrealized_pnl for p in self.portfolio.positions.values())
        realized = equity - self.portfolio.initial_cash - unrealized

        self.db.save_account_snapshot(
            initial_capital=self.portfolio.initial_cash,
            cash_balance=self.portfolio.cash,
            unrealized_pnl=unrealized,
            realized_pnl=realized,
            total_equity=equity,
            margin_used=margin_used,
            free_margin=free_margin,
            leverage=leverage,
            timestamp=timestamp,
        )
