"""
Autonomous Live Shadow Trading Daemon.
Connects to Binance USDS-M Futures Mainnet via free public read-only endpoints,
reconciles live mark-to-market prices, enforces passive Maker execution alpha,
triggers 8h multi-strategy rebalancing, and settles real funding cashflows in SQLite.
"""

from datetime import datetime, timezone
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import ccxt
import numpy as np
import pandas as pd

from src.domain.entities import Order, Portfolio, Position
from src.domain.enums import MarketRegime, OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.execution_router import ExecutionUrgency, SmartExecutionRouter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.features.volatility import parkinson_volatility
from src.strategies.meta_allocator import MetaStrategyAllocator

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "SUI/USDT",
]


class ShadowTradingDaemon:
    """
    Multi-frequency shadow trading daemon operating on live mainnet data.
    """

    def __init__(
        self,
        broker: ShadowExecutionBroker,
        allocator: Optional[MetaStrategyAllocator] = None,
        router: Optional[SmartExecutionRouter] = None,
        exchange_client: Optional[ccxt.Exchange] = None,
        symbols: Optional[List[str]] = None,
        fast_interval_sec: float = 10.0,
        rebalance_interval_sec: float = 28_800.0,  # 8 hours
        min_rebalance_notional: float = 50.0,
        turnover_deadband: float = 0.04,          # 4% deadband to eliminate fee drag
    ):
        self.broker = broker
        self.allocator = allocator or MetaStrategyAllocator()
        self.router = router or SmartExecutionRouter(broker=broker)
        self.symbols = symbols or DEFAULT_SHADOW_SYMBOLS
        self.fast_interval_sec = fast_interval_sec
        self.rebalance_interval_sec = rebalance_interval_sec
        self.min_rebalance_notional = min_rebalance_notional
        self.turnover_deadband = turnover_deadband

        # Free public read-only client for Binance Futures
        if exchange_client:
            self.exchange = exchange_client
        else:
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 20000,
                "options": {"defaultType": "future"},
            })

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.last_rebalance_time: Optional[datetime] = None
        self.last_funding_check_time: Optional[datetime] = None
        self.latest_tickers: Dict[str, Dict[str, float]] = {}

    def fetch_live_tickers(self) -> Dict[str, Dict[str, float]]:
        """
        Fetches current top-of-book L1 prices from Binance Futures public API.
        Returns mapping: symbol -> {price, bid, ask, high, low}
        """
        ticks: Dict[str, Dict[str, float]] = {}
        try:
            raw_tickers = self.exchange.fetch_tickers(self.symbols)
            for sym, t in raw_tickers.items():
                last = float(t.get("last") or t.get("close") or 0.0)
                bid = float(t.get("bid") or last)
                ask = float(t.get("ask") or last)
                high = float(t.get("high") or last)
                low = float(t.get("low") or last)

                if last > 0:
                    ticks[sym] = {
                        "price": last,
                        "bid": bid,
                        "ask": ask,
                        "high": high,
                        "low": low,
                    }
        except Exception as e:
            logger.warning(f"Error fetching live tickers from Binance: {e}")

        self.latest_tickers = ticks
        return ticks

    def fetch_symbol_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetches recent OHLCV bars for alpha calculation."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            return df
        except Exception as e:
            logger.warning(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()

    def run_fast_cycle(self) -> Tuple[Portfolio, List[Order]]:
        """
        Executes fast loop (every 5-10s):
        1. Fetch fresh L1 tickers
        2. Check & fill pending Maker limit orders
        3. Revalue portfolio Mark-to-Market
        """
        ticks = self.fetch_live_tickers()
        if not ticks:
            return self.broker.get_portfolio(), []

        # 1. Update broker prices
        for sym, t in ticks.items():
            self.broker.set_market_price(
                symbol=sym,
                price=t["price"],
                bid=t["bid"],
                ask=t["ask"],
            )

        # 2. Check pending limit orders against fresh ticks
        filled_orders = self.broker.check_pending_orders(ticks)

        # 3. Mark to market
        mark_prices = {sym: t["price"] for sym, t in ticks.items()}
        portfolio = self.broker.mark_to_market(mark_prices)

        return portfolio, filled_orders

    def run_rebalance_cycle(self) -> List[Order]:
        """
        Executes multi-strategy portfolio rebalance:
        1. Download klines for universe
        2. Detect regime & compute target allocations
        3. Net orders through turnover deadband
        4. Route passive Maker limit orders at the touch
        """
        market_data: Dict[str, pd.DataFrame] = {}
        volatilities: Dict[str, float] = {}
        current_prices: Dict[str, float] = {}

        for sym in self.symbols:
            df = self.fetch_symbol_klines(sym, timeframe="1h", limit=100)
            if not df.empty and len(df) >= 30:
                # Standardize key for allocator (e.g. BTCUSDT and BTC/USDT)
                clean_sym = sym.replace("/", "")
                market_data[clean_sym] = df
                market_data[sym] = df

                # Volatility via Parkinson High-Low range estimator
                vol_val = parkinson_volatility(df["high"].values[-30:], df["low"].values[-30:])
                volatilities[clean_sym] = float(vol_val) if vol_val > 0 else 0.02
                volatilities[sym] = volatilities[clean_sym]
                current_prices[sym] = float(df["close"].iloc[-1])

        if not market_data:
            logger.warning("No market data fetched. Aborting rebalance cycle.")
            return []

        portfolio = self.broker.get_portfolio()
        target_dollars, regime, weights = self.allocator.generate_portfolio_allocations(
            market_data=market_data,
            portfolio=portfolio,
            volatilities=volatilities,
        )

        executed_orders: List[Order] = []
        executed_deltas: Dict[str, float] = {}
        total_equity = max(portfolio.total_equity, 1.0)

        all_symbols = set(self.symbols) | set(portfolio.positions.keys())

        for sym in all_symbols:
            price = current_prices.get(sym) or self.latest_tickers.get(sym, {}).get("price", 0.0)
            if price <= 0:
                continue

            clean_sym = sym.replace("/", "")
            target_val = target_dollars.get(clean_sym, target_dollars.get(sym, 0.0))

            pos = portfolio.positions.get(sym)
            current_qty = pos.quantity if pos else 0.0
            if pos and pos.side == PositionSide.SHORT:
                current_val = -current_qty * price
            else:
                current_val = current_qty * price

            dollar_delta = target_val - current_val

            # 1. Minimum notional gate
            if abs(dollar_delta) < self.min_rebalance_notional:
                continue

            # 2. Turnover deadband gate (4% of total equity)
            delta_turnover = abs(dollar_delta) / total_equity
            if delta_turnover < self.turnover_deadband and abs(target_val) > 0:
                continue

            order_qty = abs(dollar_delta) / price
            side = OrderSide.BUY if dollar_delta > 0 else OrderSide.SELL

            # Route as passive Maker limit order at touch
            try:
                order = self.router.route_order(
                    symbol=sym,
                    side=side,
                    quantity=order_qty,
                    current_price=price,
                    urgency=ExecutionUrgency.LOW,
                )
                executed_orders.append(order)
                executed_deltas[sym] = dollar_delta
            except Exception as e:
                logger.error(f"Failed to submit rebalance order for {sym}: {e}")

        # Persist snapshot
        self.broker.db.save_rebalance_snapshot(
            regime_detected=regime.value if hasattr(regime, "value") else str(regime),
            momentum_weight=weights.get("momentum", 0.0),
            stat_arb_weight=weights.get("stat_arb", 0.0),
            carry_weight=weights.get("carry", 0.0),
            target_allocations=target_dollars,
            executed_delta=executed_deltas,
        )
        self.last_rebalance_time = datetime.now(timezone.utc)
        return executed_orders

    def run_funding_cycle(self) -> Dict[str, float]:
        """
        Polls official Binance funding rates and settles cashflows for open positions.
        """
        settled_cashflows: Dict[str, float] = {}
        open_positions = self.broker.portfolio.positions

        for sym, pos in list(open_positions.items()):
            if pos.side == PositionSide.FLAT or pos.quantity <= 0:
                continue
            try:
                funding_info = self.exchange.fetch_funding_rate(sym)
                rate = float(funding_info.get("fundingRate") or 0.0)
                cf = self.broker.settle_funding(sym, rate)
                settled_cashflows[sym] = cf
            except Exception as e:
                logger.warning(f"Error settling funding for {sym}: {e}")

        self.last_funding_check_time = datetime.now(timezone.utc)
        return settled_cashflows

    def step(self) -> Dict[str, Any]:
        """
        Performs a single synchronous step across fast, rebalance, and funding cycles.
        Useful for scheduled cron executions and unit tests.
        """
        now = datetime.now(timezone.utc)

        # 1. Fast cycle
        portfolio, filled_orders = self.run_fast_cycle()

        # 2. Check if rebalance is due
        rebalance_due = False
        if self.last_rebalance_time is None:
            rebalance_due = True
        elif (now - self.last_rebalance_time).total_seconds() >= self.rebalance_interval_sec:
            rebalance_due = True

        rebalance_orders = []
        if rebalance_due:
            rebalance_orders = self.run_rebalance_cycle()

        # 3. Check if funding settlement is due (00:00, 08:00, 16:00 UTC)
        funding_due = False
        if now.minute in (0, 1) and now.hour in (0, 8, 16):
            if (
                self.last_funding_check_time is None
                or (now - self.last_funding_check_time).total_seconds() > 300
            ):
                funding_due = True

        funding_cashflows = {}
        if funding_due:
            funding_cashflows = self.run_funding_cycle()

        return {
            "timestamp": now.isoformat(),
            "total_equity": portfolio.total_equity,
            "cash": portfolio.cash,
            "filled_orders": len(filled_orders),
            "rebalance_orders": len(rebalance_orders),
            "funding_cashflows": funding_cashflows,
        }

    def start(self) -> None:
        """Starts the daemon in a background daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("ShadowTradingDaemon started successfully.")

    def stop(self) -> None:
        """Stops the daemon background thread."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("ShadowTradingDaemon stopped.")

    def _run_loop(self) -> None:
        """Continuous execution loop."""
        while self.is_running:
            try:
                self.step()
            except Exception as e:
                logger.error(f"Error in shadow daemon loop: {e}", exc_info=True)
            time.sleep(self.fast_interval_sec)
