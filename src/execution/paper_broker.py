"""
Institutional Paper Trading Engine.
Simulates realistic exchange order execution with configurable slippage,
maker/taker fee schedules, and position accounting.
"""

import uuid
from datetime import datetime, timezone

from src.domain.entities import Order, Portfolio, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.features.market_impact import (
    CapitalCapacityCeiling,
    NonLinearMarketImpactModel,
)
from src.ports.execution_ports import IExecutionClient


class PaperExecutionBroker(IExecutionClient):
    """
    In-memory simulation broker providing realistic order fills,
    PnL accounting, and trade reporting without risking capital.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        taker_fee_pct: float = 0.0004,     # 0.04% taker
        maker_fee_pct: float = 0.0002,     # 0.02% maker
        slippage_bps: float = 1.0,         # 1 bp slippage
        impact_model: NonLinearMarketImpactModel | None = None,
        capacity_ceiling: CapitalCapacityCeiling | None = None,
        enforce_capacity_ceiling: bool = False,
    ):
        self.portfolio = Portfolio(
            cash=initial_cash,
            initial_cash=initial_cash,
            positions={},
        )
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.slippage = slippage_bps / 10_000.0
        self.impact_model = impact_model
        self.capacity_ceiling = capacity_ceiling or (impact_model.capacity_ceiling if impact_model else None)
        self.enforce_capacity_ceiling = enforce_capacity_ceiling
        self.symbol_microstructure: dict[str, dict] = {}

        self.orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self.market_prices: dict[str, float] = {}

    def set_symbol_microstructure(
        self,
        symbol: str,
        volatility: float,
        volume: float,
        depth_1pct_usd: float | None = None,
        cs_spread: float | None = None,
        amihud: float | None = None,
        is_flash_crash: bool = False,
    ) -> None:
        """Configures dynamic microstructure metrics for symbol."""
        self.symbol_microstructure[symbol] = {
            "volatility": float(volatility),
            "volume": float(volume),
            "depth_1pct_usd": float(depth_1pct_usd) if depth_1pct_usd is not None else None,
            "cs_spread": float(cs_spread) if cs_spread is not None else None,
            "amihud": float(amihud) if amihud is not None else None,
            "is_flash_crash": bool(is_flash_crash),
        }

    def set_market_price(self, symbol: str, price: float) -> None:
        """Updates current mark price for a symbol and mark-to-markets existing positions."""
        self.market_prices[symbol] = float(price)
        if symbol in self.portfolio.positions:
            self.portfolio.positions[symbol].current_price = float(price)

    def set_market_prices(self, prices: dict[str, float]) -> None:
        """Batch updates mark prices."""
        for sym, price in prices.items():
            self.set_market_price(sym, price)

    def get_portfolio(self) -> Portfolio:
        """Returns the current portfolio snapshot."""
        return self.portfolio

    def get_order_status(self, order_id: str, symbol: str) -> Order:
        """Queries status of an order."""
        if order_id not in self.orders:
            raise KeyError(f"Order {order_id} not found.")
        return self.orders[order_id]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels an order if it is still pending."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                return True
        return False

    def get_trade_history(self) -> list[Trade]:
        """Returns all executed trades."""
        return list(self.trades)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> Order:
        """
        Creates and executes an order against current market prices.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        order_id = f"paper_ord_{uuid.uuid4().hex[:10]}"
        current_market_price = self.market_prices.get(symbol, price or 100.0)

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
        self.orders[order_id] = order

        # Execute order
        self._execute_order(order, current_market_price)
        return order

    def _execute_order(self, order: Order, mark_price: float) -> None:
        """Internal execution engine with fee and dynamic slippage modeling."""
        if order.order_type == OrderType.MARKET:
            if self.impact_model:
                micro = self.symbol_microstructure.get(order.symbol, {})
                vol = micro.get("volatility", 0.005)
                vol_usd = micro.get("volume", 50_000_000.0)
                depth_1pct = micro.get("depth_1pct_usd")
                cs_spread = micro.get("cs_spread")
                amihud = micro.get("amihud")
                is_flash_crash = micro.get("is_flash_crash", False)

                fill_price, _ = self.impact_model.apply_execution_slippage(
                    order=order,
                    mark_price=mark_price,
                    volatility=vol,
                    volume=vol_usd,
                    depth_1pct_usd=depth_1pct,
                    cs_spread=cs_spread,
                    amihud=amihud,
                    is_flash_crash=is_flash_crash,
                )
            else:
                # Slippage: buy pays higher, sell receives lower
                if order.side == OrderSide.BUY:
                    fill_price = mark_price * (1.0 + self.slippage)
                else:
                    fill_price = mark_price * (1.0 - self.slippage)
            fee_pct = self.taker_fee_pct
        else:
            # Limit order
            fill_price = order.price if order.price is not None else mark_price
            fee_pct = self.maker_fee_pct

        notional = fill_price * order.quantity
        fee = notional * fee_pct

        # Check cash balance for opening purchases
        trade = Trade(
            id=f"trd_{uuid.uuid4().hex[:10]}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self.trades.append(trade)

        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price

        # Update portfolio accounting
        self._update_position(trade)

    def _update_position(self, trade: Trade) -> None:
        """Updates portfolio cash and position state following a filled trade."""
        symbol = trade.symbol
        pos = self.portfolio.positions.get(symbol)

        # Deduct transaction fee from cash
        self.portfolio.cash -= trade.fee

        if pos is None or pos.side == PositionSide.FLAT or pos.quantity == 0:
            # Open new position
            side = PositionSide.LONG if trade.side == OrderSide.BUY else PositionSide.SHORT
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
            )
            return

        if pos.side == PositionSide.LONG:
            if trade.side == OrderSide.BUY:
                # Add to long
                new_qty = pos.quantity + trade.quantity
                new_entry = (pos.quantity * pos.entry_price + trade.quantity * trade.price) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_entry
            else:
                # Sell long
                if trade.quantity >= pos.quantity:
                    # Flat or flipped
                    pnl = pos.quantity * (trade.price - pos.entry_price)
                    self.portfolio.cash += pnl
                    remaining = trade.quantity - pos.quantity
                    if remaining > 0:
                        pos.side = PositionSide.SHORT
                        pos.quantity = remaining
                        pos.entry_price = trade.price
                    else:
                        pos.side = PositionSide.FLAT
                        pos.quantity = 0.0
                else:
                    # Partial close
                    pnl = trade.quantity * (trade.price - pos.entry_price)
                    self.portfolio.cash += pnl
                    pos.quantity -= trade.quantity

        elif pos.side == PositionSide.SHORT:
            if trade.side == OrderSide.SELL:
                # Add to short
                new_qty = pos.quantity + trade.quantity
                new_entry = (pos.quantity * pos.entry_price + trade.quantity * trade.price) / new_qty
                pos.quantity = new_qty
                pos.entry_price = new_entry
            else:
                # Buy to cover short
                if trade.quantity >= pos.quantity:
                    pnl = pos.quantity * (pos.entry_price - trade.price)
                    self.portfolio.cash += pnl
                    remaining = trade.quantity - pos.quantity
                    if remaining > 0:
                        pos.side = PositionSide.LONG
                        pos.quantity = remaining
                        pos.entry_price = trade.price
                    else:
                        pos.side = PositionSide.FLAT
                        pos.quantity = 0.0
                else:
                    pnl = trade.quantity * (pos.entry_price - trade.price)
                    self.portfolio.cash += pnl
                    pos.quantity -= trade.quantity
