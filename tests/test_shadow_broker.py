"""
Unit Tests for ShadowExecutionBroker with SQLite WAL Persistence.
"""

from datetime import datetime, timezone
import pytest

from src.domain.entities import Order, Position
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.live_broker import (
    MaxDrawdownStopError,
    MaxLeverageExceededError,
    MaxOrderNotionalExceededError,
)
from src.execution.shadow_broker import ShadowExecutionBroker


@pytest.fixture
def broker():
    """Provides an isolated broker with in-memory database."""
    db = ShadowDatabase(db_path=":memory:")
    b = ShadowExecutionBroker(
        database=db,
        initial_cash=100_000.0,
        taker_fee_pct=0.0004,
        maker_fee_pct=0.0002,
        slippage_bps=1.0,
        max_gross_leverage=2.5,
        max_order_notional=25_000.0,
        max_drawdown_stop=0.15,
    )
    yield b
    db.close()


def test_broker_initial_snapshot(broker: ShadowExecutionBroker):
    """Verifies that the broker initializes portfolio and persists initial snapshot."""
    portfolio = broker.get_portfolio()
    assert portfolio.cash == 100_000.0
    assert portfolio.total_equity == 100_000.0
    assert len(portfolio.positions) == 0

    latest_acc = broker.db.get_latest_account()
    assert latest_acc is not None
    assert latest_acc["cash_balance"] == 100_000.0
    assert latest_acc["total_equity"] == 100_000.0


def test_market_order_taker_execution(broker: ShadowExecutionBroker):
    """Verifies immediate execution of market order with slippage and taker fee."""
    broker.set_market_price("BTC/USDT", price=90_000.0, bid=89_990.0, ask=90_010.0)

    # Buy 0.1 BTC market
    order = broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )
    assert order.status == OrderStatus.FILLED
    # Ask is 90,010.0, slippage 1 bp -> 90,010 * 1.0001 = 90,019.001
    assert order.average_fill_price > 90_010.0
    assert order.filled_quantity == 0.1

    # Verify portfolio position
    pos = broker.portfolio.positions["BTC/USDT"]
    assert pos.side == PositionSide.LONG
    assert pos.quantity == 0.1

    # Verify DB order and trades
    db_order = broker.db.get_order(order.id)
    assert db_order is not None
    assert db_order.status == OrderStatus.FILLED

    trades = broker.db.get_trades(symbol="BTC/USDT")
    assert len(trades) == 1
    assert trades[0]["is_maker"] == 0  # Taker
    assert trades[0]["fee_paid"] > 0.0


def test_passive_maker_order_execution(broker: ShadowExecutionBroker):
    """Verifies Maker limit order placement and tick-driven fill."""
    broker.set_market_price("ETH/USDT", price=3_500.0, bid=3_499.0, ask=3_501.0)

    # Place buy limit below market at 3,490.0
    order = broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=2.0,
        price=3_490.0,
    )
    # Market ask is 3,501.0, limit is 3,490.0 -> Should remain PENDING
    assert order.status == OrderStatus.PENDING

    pending = broker.db.get_pending_orders()
    assert len(pending) == 1

    # Tick arrives with low touching 3,485.0
    fresh_ticks = {
        "ETH/USDT": {"low": 3_485.0, "high": 3_510.0, "bid": 3_488.0, "ask": 3_490.0}
    }
    filled = broker.check_pending_orders(fresh_ticks)
    assert len(filled) == 1
    assert filled[0].id == order.id
    assert filled[0].status == OrderStatus.FILLED
    assert filled[0].average_fill_price == 3_490.0

    # Verify trades recorded Maker fee (0.02%)
    trades = broker.db.get_trades(symbol="ETH/USDT")
    assert len(trades) == 1
    assert trades[0]["is_maker"] == 1
    # 3490 * 2 * 0.0002 = 1.396
    assert abs(trades[0]["fee_paid"] - 1.396) < 1e-4


def test_circuit_breaker_max_order_notional(broker: ShadowExecutionBroker):
    """Verifies that an order exceeding max order notional ($25k) is rejected."""
    broker.set_market_price("BTC/USDT", price=90_000.0)

    # 0.5 BTC * $90k = $45k > $25k limit
    with pytest.raises(MaxOrderNotionalExceededError):
        broker.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5,
        )


def test_circuit_breaker_max_leverage(broker: ShadowExecutionBroker):
    """Verifies that exceeding max gross leverage (2.5x) raises an error."""
    broker.set_market_price("SOL/USDT", price=200.0)

    # Place multiple orders up to leverage limit
    # Capital is $100k, max gross is $250k. Max single notional is $25k.
    # Submit 10 orders of $24k each (120 SOL * 200 = 24k) -> total 240k
    for _ in range(10):
        broker.create_order(
            symbol="SOL/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=120.0,  # 24k notional
        )

    # Current gross is 240k, equity ~99.9k.
    # Submitting another 24k would push gross to 264k (> 250k = 2.5x)
    with pytest.raises(MaxLeverageExceededError):
        broker.create_order(
            symbol="SOL/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=120.0,
        )


def test_funding_settlement(broker: ShadowExecutionBroker):
    """Verifies 8h funding settlement accounting."""
    broker.set_market_price("BTC/USDT", price=90_000.0)
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,  # $9,000 notional
    )
    initial_cash = broker.portfolio.cash

    # Positive funding rate (Long pays, Short receives)
    # Cashflow = -9000 * 0.0001 = -0.90 USD
    cashflow = broker.settle_funding(symbol="BTC/USDT", funding_rate=0.0001)
    assert abs(cashflow - (-0.90)) < 0.05
    assert abs(broker.portfolio.cash - (initial_cash - 0.90)) < 0.05

    settlements = broker.db.get_funding_settlements()
    assert len(settlements) == 1
    assert settlements[0]["symbol"] == "BTC/USDT"


def test_mark_to_market_portfolio(broker: ShadowExecutionBroker):
    """Verifies continuous mark-to-market revaluation."""
    broker.set_market_price("ETH/USDT", price=3_000.0)
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0,  # 15,000 notional
    )

    # Price increases to 3,200.0 (Unrealized PnL = +1,000 USD)
    updated_portfolio = broker.mark_to_market({"ETH/USDT": 3_200.0})
    pos = updated_portfolio.positions["ETH/USDT"]
    assert pos.unrealized_pnl > 950.0  # accounted after slippage

    latest_acc = broker.db.get_latest_account()
    assert latest_acc["unrealized_pnl"] > 950.0
