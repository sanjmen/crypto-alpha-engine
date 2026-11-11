"""
Institutional Shadow Trading Broker.
Simulates realistic matching against live Binance Mainnet prices,
enforcing hard safety circuit breakers, passive Maker touch execution,
taker slippage, 8h funding cashflows, and local SQLite persistence.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.domain.entities import Order, Portfolio, Position, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.live_broker import (
    BinanceLiveBroker,
    MarketCapacityExceededError,
    MaxDrawdownStopError,
    MaxLeverageExceededError,
    MaxOrderNotionalExceededError,
)
from src.features.market_impact import (
    CapitalCapacityCeiling,
    NonLinearMarketImpactModel,
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
        database: ShadowDatabase | None = None,
        initial_cash: float = 100_000.0,
        taker_fee_pct: float = 0.0004,     # 0.04% taker
        maker_fee_pct: float = 0.0002,     # 0.02% maker
        slippage_bps: float = 1.0,         # 1 bp slippage for market orders
        max_gross_leverage: float = 2.5,   # Hard circuit breaker
        max_order_notional: float = 25_000.0,
        max_drawdown_stop: float = 0.15,   # 15% drawdown halts execution
        impact_model: NonLinearMarketImpactModel | None = None,
        capacity_ceiling: CapitalCapacityCeiling | None = None,
        enforce_capacity_ceiling: bool = False,
        live_broker: Any | None = None,
        enable_yield_collateral: bool = True,
        collateral_yield_apy: float = 0.055,
        collateral_haircut: float = 0.999,
    ):
        self.db = database or ShadowDatabase()
        self.taker_fee_pct = taker_fee_pct
        self.maker_fee_pct = maker_fee_pct
        self.slippage = slippage_bps / 10_000.0
        self.max_gross_leverage = max_gross_leverage
        self.max_order_notional = max_order_notional
        self.max_drawdown_stop = max_drawdown_stop
        self.impact_model = impact_model
        self.capacity_ceiling = capacity_ceiling or (impact_model.capacity_ceiling if impact_model else None)
        self.enforce_capacity_ceiling = enforce_capacity_ceiling
        self.symbol_microstructure: dict[str, dict] = {}
        self.live_broker = live_broker
        self.enable_yield_collateral = bool(enable_yield_collateral)
        self.collateral_yield_apy = float(collateral_yield_apy)
        self.collateral_haircut = float(collateral_haircut)
        self.last_collateral_breakdown: dict[str, Any] = {}
        self._configured_live_symbols: set[str] = set()
        self._current_live_leverage: int = 3

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
        self.capital_rescaled_ratio: float | None = None
        self.peak_equity = max(self.portfolio.total_equity, init_cash)
        self.market_prices: dict[str, float] = {}
        self.market_bids: dict[str, float] = {}
        self.market_asks: dict[str, float] = {}

        if self.is_live:
            self.sync_from_live_exchange(sync_positions=True)

        # If fresh db, record initial account snapshot
        if not latest_acc:
            self._record_account_snapshot()

    def scale_capital(self, target_initial_capital: float) -> dict[str, float]:
        """
        Rescales `initial_cash` and `cash` by injecting/withdrawing the capital delta
        while preserving historical realized PnL (`cash - initial_cash`).
        """
        target_init = float(target_initial_capital)
        old_init = float(self.portfolio.initial_cash)
        old_cash = float(self.portfolio.cash)
        if target_init <= 0 or abs(target_init - old_init) <= 1.0:
            return {
                "old_initial_capital": old_init,
                "new_initial_capital": old_init,
                "delta_capital": 0.0,
                "new_cash_balance": old_cash,
            }

        delta_cap = target_init - old_init
        self.portfolio.initial_cash = target_init
        self.portfolio.cash = max(0.0, old_cash + delta_cap)
        self.capital_rescaled_ratio = target_init / max(old_init, 1.0)
        self.peak_equity = max(self.portfolio.total_equity, target_init)
        self._record_account_snapshot()
        logger.info(
            f"💰 Capital scaled in Shadow Broker: Initial ${old_init:,.2f} -> ${target_init:,.2f} "
            f"(Delta: {delta_cap:+,.2f} USDT | Cash: ${self.portfolio.cash:,.2f} | "
            f"Realized PnL preserved: ${self.portfolio.cash - target_init:+,.2f})"
        )
        return {
            "old_initial_capital": old_init,
            "new_initial_capital": target_init,
            "delta_capital": delta_cap,
            "new_cash_balance": self.portfolio.cash,
        }

    @property
    def is_live(self) -> bool:
        """Returns True when backed by an active live Binance Futures execution broker."""
        return bool(self.live_broker is not None and not getattr(self.live_broker, "dry_run", True))

    def sync_from_live_exchange(self, sync_positions: bool = False) -> float:
        """
        Synchronizes local `self.portfolio.cash` (and optionally open positions) with the
        real Binance USD-M Futures account (`walletBalance` & `positionRisk`).
        """
        if not self.is_live:
            return self.portfolio.cash

        try:
            bal = self.live_broker.client.fetch_balance()
            col = BinanceLiveBroker.compute_effective_margin_collateral(bal)
            self.last_collateral_breakdown = col
            wallet_bal = float(col.get("effective_wallet_usd") or 0.0)
            if wallet_bal <= 0 and isinstance(bal, dict):
                usdt_info = bal.get("USDT", {})
                wallet_bal = float(usdt_info.get("total") or usdt_info.get("free") or self.portfolio.cash)
            if wallet_bal > 0:
                self.portfolio.cash = wallet_bal
                self.live_broker.portfolio.cash = wallet_bal
                if not self.portfolio.positions and abs(self.portfolio.initial_cash - 100_000.0) < 1e-6:
                    self.portfolio.initial_cash = wallet_bal
                self.peak_equity = max(self.peak_equity, self.portfolio.total_equity)
                self.live_broker.peak_equity = max(self.live_broker.peak_equity, self.portfolio.total_equity)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Live USDT/BFUSD/LDUSDT balance sync note: {e}")

        if sync_positions:
            try:
                ex_positions = self.live_broker.reconcile_positions_from_exchange()
                for sym, ex_pos in ex_positions.items():
                    local_pos = self.portfolio.positions.get(sym)
                    if local_pos and local_pos.side == ex_pos.side and local_pos.quantity > 0:
                        local_pos.quantity = ex_pos.quantity
                        local_pos.entry_price = ex_pos.entry_price
                        local_pos.current_price = ex_pos.current_price or ex_pos.entry_price
                        self.db.save_position(local_pos)
                    else:
                        ex_pos.opened_at = datetime.now(timezone.utc)
                        ex_pos.peak_price = ex_pos.entry_price
                        ex_pos.strategy = (local_pos.strategy if local_pos and local_pos.strategy else "7-Factor Core")
                        self.portfolio.positions[sym] = ex_pos
                        self.db.save_position(ex_pos)

                for sym, local_pos in list(self.portfolio.positions.items()):
                    if local_pos.side != PositionSide.FLAT and local_pos.quantity > 0 and sym not in ex_positions:
                        local_pos.quantity = 0.0
                        local_pos.side = PositionSide.FLAT
                        local_pos.take_profit = None
                        local_pos.stop_loss = None
                        local_pos.peak_price = 0.0
                        self._cancel_bracket_orders(sym)
                        self.db.save_position(local_pos)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Live position reconciliation note: {e}")

        self._record_account_snapshot()
        return self.portfolio.cash

    def _route_to_live_exchange(
        self,
        order: Order,
        fallback_price: float,
        prefer_maker: bool,
    ) -> tuple[float, float, bool]:
        """
        Routes a matched order to the real Binance USD-M Futures exchange via `self.live_broker`.
        Returns `(filled_quantity, average_fill_price, was_maker_fill)`.
        """
        if not self.is_live:
            return order.quantity, fallback_price, prefer_maker

        symbol = order.symbol
        if symbol not in self._configured_live_symbols:
            try:
                self.live_broker.client.set_margin_mode("cross", symbol)
            except Exception:
                pass
            try:
                self.live_broker.client.set_leverage(self._current_live_leverage, symbol)
            except Exception:
                pass
            self._configured_live_symbols.add(symbol)

        pos = self.portfolio.positions.get(symbol)
        is_opposite = bool(
            pos is not None
            and pos.side != PositionSide.FLAT
            and pos.quantity > 0
            and (
                (pos.side == PositionSide.LONG and order.side == OrderSide.SELL)
                or (pos.side == PositionSide.SHORT and order.side == OrderSide.BUY)
            )
        )
        reduce_only = bool(
            order.order_type in (OrderType.TAKE_PROFIT, OrderType.STOP_LOSS)
            or (is_opposite and order.quantity <= pos.quantity * 1.01)
        )
        target_qty = (
            min(order.quantity, pos.quantity)
            if (reduce_only and pos is not None and pos.quantity > 0)
            else order.quantity
        )
        exec_qty = self.live_broker.quantize_amount(symbol, target_qty)
        if exec_qty <= 0:
            return 0.0, fallback_price, prefer_maker

        self.live_broker.portfolio.cash = self.portfolio.cash
        self.live_broker.portfolio.initial_cash = self.portfolio.initial_cash
        self.live_broker.peak_equity = max(self.live_broker.peak_equity, self.portfolio.total_equity)
        self.live_broker.max_gross_leverage = max(self.live_broker.max_gross_leverage, self.max_gross_leverage)

        if prefer_maker:
            prev_maker_cnt = self.live_broker.maker_fills_count
            live_ord = self.live_broker.execute_maker_limit_chase(
                symbol=symbol,
                side=order.side,
                quantity=exec_qty,
                fallback_price=fallback_price,
                reduce_only=reduce_only,
            )
            filled_q = float(live_ord.filled_quantity or exec_qty)
            avg_px = float(live_ord.average_fill_price or fallback_price)
            was_maker = self.live_broker.maker_fills_count > prev_maker_cnt
            return filled_q, avg_px, was_maker

        ccxt_side = "buy" if order.side == OrderSide.BUY else "sell"
        params: dict[str, Any] = {"reduceOnly": True} if reduce_only else {}
        res = self.live_broker.client.create_order(
            symbol=symbol,
            type="market",
            side=ccxt_side,
            amount=exec_qty,
            params=params,
        )
        filled_q = float(res.get("filled") or res.get("amount") or exec_qty)
        avg_px = float(res.get("average") or res.get("price") or fallback_price)
        return filled_q, avg_px, False

    def set_market_price(
        self,
        symbol: str,
        price: float,
        bid: float | None = None,
        ask: float | None = None,
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

    def set_market_prices(self, prices: dict[str, float]) -> None:
        """Batch updates mark prices."""
        for sym, price in prices.items():
            self.set_market_price(sym, price)

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
        """Configures dynamic microstructure metrics for realistic slippage calculations."""
        self.symbol_microstructure[symbol] = {
            "volatility": float(volatility),
            "volume": float(volume),
            "depth_1pct_usd": float(depth_1pct_usd) if depth_1pct_usd is not None else None,
            "cs_spread": float(cs_spread) if cs_spread is not None else None,
            "amihud": float(amihud) if amihud is not None else None,
            "is_flash_crash": bool(is_flash_crash),
        }

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

    def get_trade_history(self) -> list[Trade]:
        """Returns executed trade records."""
        trade_rows = self.db.get_trades(limit=500)
        trades: list[Trade] = []
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
        self.peak_equity = max(self.peak_equity, current_equity)

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
        price: float | None = None,
        role: str | None = None,
        client_order_id: str | None = None,
        strategy: str | None = None,
    ) -> Order:
        """
        Submits an order through risk verification and matching.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        est_price = price or self.market_prices.get(symbol, 100.0)

        # 1. Enforce safety circuit breakers
        self.verify_risk_circuit_breakers(symbol, quantity, est_price)

        if self.enforce_capacity_ceiling and self.capacity_ceiling:
            order_notional = est_price * quantity
            micro = self.symbol_microstructure.get(symbol, {})
            cap_report = self.capacity_ceiling.evaluate_capacity(
                symbol=symbol,
                order_size_usd=order_notional,
                depth_1pct_usd=micro.get("depth_1pct_usd"),
                bar_volume_usd=micro.get("volume"),
            )
            if not cap_report.within_capacity:
                raise MarketCapacityExceededError(
                    f"Order notional ${order_notional:,.2f} exceeds capacity ceiling "
                    f"${cap_report.max_allowed_size_usd:,.2f} for {symbol}. "
                    f"Recommended chunks: {cap_report.recommended_chunks}"
                )

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
            strategy=strategy,
        )

        assigned_role = role or ("MAKER" if order_type == OrderType.LIMIT else "TAKER")
        self.db.save_order(order, role=assigned_role, client_order_id=client_order_id)

        # 2. Execution logic (if live exchange call raises, cancel the local PENDING order so it never becomes a stale zombie)
        try:
            if order_type == OrderType.MARKET:
                self._fill_market_order(order)
            elif order_type == OrderType.LIMIT:
                self._evaluate_limit_order(order)
        except Exception:
            if self.is_live and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELED
                self.db.update_order_status(order.id, OrderStatus.CANCELED)
            raise

        return order

    def _fill_market_order(self, order: Order) -> None:
        """Executes an aggressive Taker order with dynamic non-linear slippage and fee."""
        symbol = order.symbol
        if order.side == OrderSide.BUY:
            base_price = self.market_asks.get(symbol, self.market_prices.get(symbol, 100.0))
        else:
            base_price = self.market_bids.get(symbol, self.market_prices.get(symbol, 100.0))

        if self.impact_model:
            micro = self.symbol_microstructure.get(symbol, {})
            vol = micro.get("volatility", 0.005)
            vol_usd = micro.get("volume", 50_000_000.0)
            depth_1pct = micro.get("depth_1pct_usd")
            cs_spread = micro.get("cs_spread")
            amihud = micro.get("amihud")
            is_flash_crash = micro.get("is_flash_crash", False)

            fill_price, estimate = self.impact_model.apply_execution_slippage(
                order=order,
                mark_price=base_price,
                volatility=vol,
                volume=vol_usd,
                depth_1pct_usd=depth_1pct,
                cs_spread=cs_spread,
                amihud=amihud,
                is_flash_crash=is_flash_crash,
            )

            if self.enforce_capacity_ceiling and not estimate.capacity_report.within_capacity:
                raise MarketCapacityExceededError(
                    f"Order notional ${order.quantity * base_price:,.2f} exceeds capacity ceiling "
                    f"${estimate.capacity_report.max_allowed_size_usd:,.2f} for {symbol}. "
                    f"Recommended chunks: {estimate.capacity_report.recommended_chunks}"
                )
        else:
            if order.side == OrderSide.BUY:
                fill_price = base_price * (1.0 + self.slippage)
            else:
                fill_price = base_price * (1.0 - self.slippage)

        self._fill_taker_order(order, fill_price)

    def _fill_taker_order(self, order: Order, fill_price: float) -> None:
        """Fills order as Taker and updates balance and position."""
        symbol = order.symbol
        if self.is_live:
            exec_qty, fill_price, _ = self._route_to_live_exchange(
                order=order,
                fallback_price=fill_price,
                prefer_maker=False,
            )
            if exec_qty <= 0:
                order.status = OrderStatus.CANCELED
                self.db.update_order_status(order.id, OrderStatus.CANCELED)
                return
            order.quantity = exec_qty

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
            strategy=getattr(order, "strategy", None),
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
        self.db.save_trade(trade, is_maker=False, realized_pnl=realized_pnl, strategy=getattr(order, "strategy", None))
        if self.is_live:
            self.sync_from_live_exchange(sync_positions=False)
        else:
            self._record_account_snapshot()

    def _evaluate_limit_order(self, order: Order) -> None:
        """
        Evaluates a passive Maker limit order against current book touch.
        If current market already touches or crosses limit (or live limit-chase is active),
        fills via `_fill_maker_order`. Otherwise leaves as PENDING in database.
        """
        symbol = order.symbol
        limit_price = order.price or self.market_prices.get(symbol, 100.0)

        should_fill = self.is_live
        if not should_fill:
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
        is_maker_fill = True
        if self.is_live:
            exec_qty, fill_price, is_maker_fill = self._route_to_live_exchange(
                order=order,
                fallback_price=fill_price,
                prefer_maker=True,
            )
            if exec_qty <= 0:
                order.status = OrderStatus.CANCELED
                self.db.update_order_status(order.id, OrderStatus.CANCELED)
                return
            order.quantity = exec_qty

        fee_rate = self.maker_fee_pct if is_maker_fill else self.taker_fee_pct
        role_label = "MAKER" if is_maker_fill else "TAKER"
        fee = fill_price * order.quantity * fee_rate

        trade = Trade(
            id=f"shd_trd_{uuid.uuid4().hex[:10]}",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=datetime.now(timezone.utc),
            strategy=getattr(order, "strategy", None),
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
            role=role_label,
        )
        realized_pnl = self._update_portfolio_position(trade)
        self.db.save_trade(
            trade,
            is_maker=is_maker_fill,
            realized_pnl=realized_pnl,
            strategy=getattr(order, "strategy", None),
        )
        if self.is_live:
            self.sync_from_live_exchange(sync_positions=False)
        else:
            self._record_account_snapshot()

    def create_bracket_orders(
        self,
        symbol: str,
        take_profit: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[Order | None, Order | None] | None:
        """
        Attaches Take-Profit and/or Stop-Loss orders to an active position.
        Persists TP/SL levels in the position record and stores pending bracket orders in DB.
        Supports Take-Profit-only initialization (stop_loss=None) for core basket positions
        where Stop-Profit / Trailing Stop is armed dynamically once profit thresholds are reached.
        """
        pos = self.portfolio.positions.get(symbol)
        if not pos or pos.side == PositionSide.FLAT or pos.quantity <= 0:
            return None

        pos.take_profit = float(take_profit) if take_profit is not None else None
        pos.stop_loss = float(stop_loss) if stop_loss is not None else None
        self.db.save_position(pos)

        # Cancel any existing pending TP/SL orders for this symbol first
        for o in self.db.get_pending_orders():
            if o.symbol == symbol and o.order_type in (OrderType.TAKE_PROFIT, OrderType.STOP_LOSS):
                self.db.update_order_status(o.id, OrderStatus.CANCELED)

        # Order side to close position
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY

        tp_order: Order | None = None
        sl_order: Order | None = None

        if pos.take_profit is not None:
            tp_order = Order(
                id=f"shd_tp_{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                side=close_side,
                order_type=OrderType.TAKE_PROFIT,
                quantity=pos.quantity,
                price=pos.take_profit,
                status=OrderStatus.PENDING,
                strategy=getattr(pos, "strategy", None),
            )
            self.db.save_order(tp_order, role="MAKER")

        if pos.stop_loss is not None:
            sl_order = Order(
                id=f"shd_sl_{uuid.uuid4().hex[:10]}",
                symbol=symbol,
                side=close_side,
                order_type=OrderType.STOP_LOSS,
                quantity=pos.quantity,
                price=pos.stop_loss,
                status=OrderStatus.PENDING,
                strategy=getattr(pos, "strategy", None),
            )
            self.db.save_order(sl_order, role="TAKER")

        return tp_order, sl_order

    def update_stop_loss(self, symbol: str, new_stop_loss: float) -> Order | None:
        """
        Ratchets the Stop-Loss order for an active position (Profit-Lock / Trailing Stop).
        Updates the position stop_loss attribute and replaces the pending STOP_LOSS order in DB.
        """
        pos = self.portfolio.positions.get(symbol)
        if not pos or pos.side == PositionSide.FLAT or pos.quantity <= 0:
            return None

        pos.stop_loss = float(new_stop_loss)
        self.db.save_position(pos)

        # Cancel existing pending SL order for this symbol
        for o in self.db.get_pending_orders():
            if o.symbol == symbol and o.order_type == OrderType.STOP_LOSS:
                self.db.update_order_status(o.id, OrderStatus.CANCELED)

        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        new_sl_order = Order(
            id=f"shd_sl_{uuid.uuid4().hex[:10]}",
            symbol=symbol,
            side=close_side,
            order_type=OrderType.STOP_LOSS,
            quantity=pos.quantity,
            price=pos.stop_loss,
            status=OrderStatus.PENDING,
            strategy=getattr(pos, "strategy", None),
        )
        self.db.save_order(new_sl_order, role="TAKER")
        return new_sl_order

    def check_pending_orders(self, symbol_ticks: dict[str, dict[str, float]]) -> list[Order]:
        """
        Evaluates all pending limit, take-profit, and stop-loss orders against fresh market ticks.
        Fills orders whose trigger conditions have been met and enforces OCO bracket cancellation.
        """
        pending_orders = self.db.get_pending_orders()
        filled_orders: list[Order] = []

        for order in pending_orders:
            symbol = order.symbol
            tick = symbol_ticks.get(symbol) or symbol_ticks.get(symbol.replace("/", ""))
            if not tick or order.price is None:
                continue

            limit_price = order.price
            current = float(tick.get("price", limit_price))
            bid = float(tick.get("bid", current))
            ask = float(tick.get("ask", current))
            low = float(tick.get("low", tick.get("bid", limit_price)))
            high = float(tick.get("high", tick.get("ask", limit_price)))

            should_fill = False
            if order.order_type == OrderType.TAKE_PROFIT:
                # Sell TP triggers when current market price rises to/above target; Buy TP triggers when current drops to/below target
                if order.side == OrderSide.SELL and (bid >= limit_price or current >= limit_price):
                    should_fill = True
                elif order.side == OrderSide.BUY and (ask <= limit_price or current <= limit_price):
                    should_fill = True
            elif order.order_type == OrderType.STOP_LOSS:
                # Sell SL triggers when current market price drops to/below stop; Buy SL triggers when current rises to/above stop
                if order.side == OrderSide.SELL and (bid <= limit_price or current <= limit_price):
                    should_fill = True
                elif order.side == OrderSide.BUY and (ask >= limit_price or current >= limit_price):
                    should_fill = True
            else:
                # Standard passive LIMIT order:
                # Buy limit fills when market ask, current, or low drops to or crosses limit
                # Sell limit fills when market bid, current, or high rises to or crosses limit
                if order.side == OrderSide.BUY and (ask <= limit_price or current <= limit_price or low <= limit_price):
                    should_fill = True
                elif order.side == OrderSide.SELL and (bid >= limit_price or current >= limit_price or high >= limit_price):
                    should_fill = True

            if should_fill:
                if order.order_type == OrderType.STOP_LOSS:
                    self._fill_taker_order(order, fill_price=limit_price)
                else:
                    self._fill_maker_order(order, fill_price=limit_price)
                filled_orders.append(order)

                # OCO bracket logic: Cancel opposing bracket order if TP or SL triggered
                if order.order_type in (OrderType.TAKE_PROFIT, OrderType.STOP_LOSS):
                    opposing = OrderType.STOP_LOSS if order.order_type == OrderType.TAKE_PROFIT else OrderType.TAKE_PROFIT
                    for other in self.db.get_pending_orders():
                        if other.symbol == symbol and other.order_type == opposing:
                            self.db.update_order_status(other.id, OrderStatus.CANCELED)

        return filled_orders

    def settle_funding(
        self,
        symbol: str,
        funding_rate: float,
        timestamp: datetime | None = None,
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

        if self.is_live:
            self.db.record_funding_settlement(
                symbol=symbol,
                funding_rate=funding_rate,
                position_side=side_str,
                position_notional=notional,
                cashflow_credited=cashflow,
                timestamp=timestamp,
            )
            self.sync_from_live_exchange(sync_positions=False)
            return cashflow

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

    def settle_collateral_yield(
        self,
        apy: float | None = None,
        interval_hours: float = 8.0,
        timestamp: datetime | None = None,
    ) -> float:
        """
        Tier 1C (Issue #57): Accrues passive yield on Futures Cross-Margin collateral
        (`BFUSD` / `LDUSDT`, default 5.5% APY == ~0.0151%/day == ~0.00502% per 8h funding interval)
        applying the 99.9% margin value haircut (`collateral_haircut = 0.999`).
        """
        if not self.enable_yield_collateral:
            return 0.0

        eff_apy = float(self.collateral_yield_apy if apy is None else apy)
        if eff_apy <= 0.0 or interval_hours <= 0.0:
            return 0.0

        period_rate = eff_apy * (float(interval_hours) / 8760.0)
        if self.is_live:
            self.sync_from_live_exchange(sync_positions=False)
            yield_base = float(self.last_collateral_breakdown.get("yield_collateral_usd") or 0.0)
            if yield_base <= 0.0:
                return 0.0
            cashflow = yield_base * period_rate
            self.db.record_funding_settlement(
                symbol="BFUSD/USDT",
                funding_rate=-period_rate,
                position_side="LONG",
                position_notional=yield_base,
                cashflow_credited=cashflow,
                timestamp=timestamp,
            )
            return cashflow

        collateral_base = max(0.0, float(self.portfolio.cash) * float(self.collateral_haircut))
        if collateral_base <= 0.0:
            return 0.0

        cashflow = collateral_base * period_rate
        self.portfolio.cash += cashflow
        self.db.record_funding_settlement(
            symbol="BFUSD/USDT",
            funding_rate=-period_rate,
            position_side="LONG",
            position_notional=collateral_base,
            cashflow_credited=cashflow,
            timestamp=timestamp,
        )
        self._record_account_snapshot(timestamp)
        return cashflow

    def mark_to_market(
        self,
        prices: dict[str, float],
        timestamp: datetime | None = None,
    ) -> Portfolio:
        """
        Updates mark prices and refreshes unrealized PnL, equity, and account state.
        """
        self.set_market_prices(prices)
        self._record_account_snapshot(timestamp)
        return self.portfolio

    def _cancel_bracket_orders(self, symbol: str) -> None:
        """Cancels any pending Take-Profit or Stop-Loss bracket orders for a symbol."""
        for o in self.db.get_pending_orders(symbol=symbol):
            if o.order_type in (OrderType.TAKE_PROFIT, OrderType.STOP_LOSS):
                self.db.update_order_status(o.id, OrderStatus.CANCELED)

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
                strategy=getattr(trade, "strategy", None),
                opened_at=trade.timestamp,
            )
            self.portfolio.positions[symbol] = pos
        else:
            if not pos.strategy and getattr(trade, "strategy", None):
                pos.strategy = trade.strategy

            # Existing position
            is_increasing = (
                (pos.side == PositionSide.LONG and trade.side == OrderSide.BUY)
                or (pos.side == PositionSide.SHORT and trade.side == OrderSide.SELL)
            )

            if is_increasing:
                old_qty = pos.quantity
                old_peak = pos.peak_price
                total_qty = old_qty + trade.quantity
                avg_entry = (old_qty * pos.entry_price + trade.quantity * trade.price) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_entry
                pos.current_price = trade.price

                # Preserve active Profit-Lock / Trailing Stop and peak_price on minor neutrality scale-ins (< 20% size increase)
                if old_qty > 0 and (trade.quantity / old_qty) < 0.20 and old_peak > 0:
                    if pos.side == PositionSide.LONG:
                        pos.peak_price = max(old_peak, trade.price, avg_entry)
                    else:
                        pos.peak_price = min(old_peak, trade.price, avg_entry)
                else:
                    pos.peak_price = trade.price
                    pos.take_profit = None
                    pos.stop_loss = None
                    self._cancel_bracket_orders(symbol)
            else:
                # Reducing or reversing position
                closed_qty = min(pos.quantity, trade.quantity)
                if pos.side == PositionSide.LONG:
                    realized_pnl = closed_qty * (trade.price - pos.entry_price)
                else:
                    realized_pnl = closed_qty * (pos.entry_price - trade.price)

                self.portfolio.cash += realized_pnl

                if abs(trade.quantity - pos.quantity) <= 1e-9:
                    pos.quantity = 0.0
                    pos.side = PositionSide.FLAT
                    pos.current_price = trade.price
                    pos.take_profit = None
                    pos.stop_loss = None
                    pos.peak_price = 0.0
                    pos.strategy = None
                    pos.opened_at = None
                    self._cancel_bracket_orders(symbol)
                elif trade.quantity < pos.quantity:
                    pos.quantity -= trade.quantity
                    pos.current_price = trade.price
                else:
                    # Reversed side
                    remaining_qty = trade.quantity - pos.quantity
                    pos.side = PositionSide.SHORT if pos.side == PositionSide.LONG else PositionSide.LONG
                    pos.quantity = remaining_qty
                    pos.entry_price = trade.price
                    pos.current_price = trade.price
                    pos.take_profit = None
                    pos.stop_loss = None
                    pos.peak_price = 0.0
                    pos.strategy = getattr(trade, "strategy", None)
                    pos.opened_at = trade.timestamp
                    self._cancel_bracket_orders(symbol)

        # Update DB position record
        self.db.save_position(pos, realized_pnl=realized_pnl)

        # Keep any active TP/SL bracket order quantities synchronized with updated position size
        if pos.side != PositionSide.FLAT and pos.quantity > 0 and (pos.take_profit is not None or pos.stop_loss is not None):
            self.create_bracket_orders(symbol=symbol, take_profit=pos.take_profit, stop_loss=pos.stop_loss)

        return realized_pnl

    def _record_account_snapshot(self, timestamp: datetime | None = None) -> None:
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
