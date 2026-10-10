"""
Unit Tests for ShadowTradingDaemon.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Order, Portfolio, Position
from src.domain.enums import MarketRegime, OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.strategies.meta_allocator import MetaStrategyAllocator


def generate_mock_klines(n_bars: int = 100, base_price: float = 90_000.0) -> pd.DataFrame:
    """Generates synthetic OHLCV data for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    opens = prices * (1 - 0.001)
    highs = prices * (1 + 0.005)
    lows = prices * (1 - 0.005)
    closes = prices
    volumes = np.random.uniform(50, 200, n_bars)

    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="1h")
    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


@pytest.fixture
def mock_exchange():
    """Provides a mock ccxt exchange returning deterministic data."""
    exchange = MagicMock()
    exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"last": 90_000.0, "bid": 89_990.0, "ask": 90_010.0, "high": 91_000.0, "low": 89_500.0},
        "ETH/USDT": {"last": 3_500.0, "bid": 3_499.0, "ask": 3_501.0, "high": 3_550.0, "low": 3_480.0},
    }
    mock_df = generate_mock_klines(60, 90_000.0)
    ohlcv_list = [
        [int(row.timestamp.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
        for row in mock_df.itertuples()
    ]
    exchange.fetch_ohlcv.return_value = ohlcv_list
    exchange.fetch_funding_rate.return_value = {"symbol": "BTC/USDT", "fundingRate": 0.0001}
    return exchange


@pytest.fixture
def daemon(mock_exchange):
    """Initializes daemon with in-memory SQLite broker and mock exchange."""
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=100_000.0)
    router = SmartExecutionRouter(broker=broker)

    d = ShadowTradingDaemon(
        broker=broker,
        router=router,
        exchange_client=mock_exchange,
        symbols=["BTC/USDT", "ETH/USDT"],
        fast_interval_sec=1.0,
        rebalance_interval_sec=28_800.0,
        min_rebalance_notional=50.0,
        turnover_deadband=0.04,
    )
    yield d
    db.close()


def test_daemon_fetch_tickers(daemon: ShadowTradingDaemon):
    """Verifies parsing of live ticker data."""
    ticks = daemon.fetch_live_tickers()
    assert "BTC/USDT" in ticks
    assert ticks["BTC/USDT"]["price"] == 90_000.0
    assert ticks["BTC/USDT"]["bid"] == 89_990.0
    assert ticks["BTC/USDT"]["ask"] == 90_010.0


def test_daemon_fast_cycle(daemon: ShadowTradingDaemon):
    """Verifies fast cycle prices update and mark-to-market snapshot."""
    # Place a pending Maker limit order on ETH
    daemon.broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=3_450.0,  # Below market (pending)
    )
    assert len(daemon.broker.db.get_pending_orders()) == 1

    portfolio, filled_orders = daemon.run_fast_cycle()
    assert portfolio.total_equity == 100_000.0
    # ETH market low was 3,480.0 > 3,450.0 -> Should remain pending
    assert len(filled_orders) == 0

    # Now simulate market dipping to 3,440.0
    daemon.exchange.fetch_tickers.return_value["ETH/USDT"]["low"] = 3_440.0
    portfolio, filled_orders = daemon.run_fast_cycle()
    assert len(filled_orders) == 1
    assert filled_orders[0].symbol == "ETH/USDT"
    assert filled_orders[0].status == OrderStatus.FILLED


def test_daemon_rebalance_cycle(daemon: ShadowTradingDaemon):
    """Verifies multi-strategy portfolio rebalance signal execution and deadband logic."""
    orders = daemon.run_rebalance_cycle()
    assert daemon.last_rebalance_time is not None

    # Verify that rebalance snapshot was saved in database
    latest_snap = daemon.broker.db.get_latest_rebalance_snapshot()
    assert latest_snap is not None
    assert latest_snap["regime_detected"] in ("TRENDING", "RANGING", "VOLATILE")


def test_daemon_funding_cycle(daemon: ShadowTradingDaemon):
    """Verifies 8h funding settlement execution in daemon."""
    # Open an inventory position in BTC
    daemon.broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )
    cashflows = daemon.run_funding_cycle()
    assert "BTC/USDT" in cashflows
    # Long pays positive funding
    assert cashflows["BTC/USDT"] < 0.0


def test_daemon_single_step(daemon: ShadowTradingDaemon):
    """Verifies synchronous step execution."""
    res = daemon.step()
    assert "total_equity" in res
    assert "rebalance_orders" in res
    assert res["total_equity"] > 0.0
