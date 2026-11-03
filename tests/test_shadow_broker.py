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


def test_bracket_orders_and_pnl_estimation(broker: ShadowExecutionBroker):
    """Verifies creation of Take-Profit and Stop-Loss orders with PnL estimations."""
    broker.set_market_price("WAL/USDT", price=0.0600)
    broker.create_order(
        symbol="WAL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
    )

    pos = broker.portfolio.positions["WAL/USDT"]
    assert pos.quantity == 1000.0

    # Attach brackets: TP @ 0.0690 (+15%), SL @ 0.0576 (-4%)
    tp_order, sl_order = broker.create_bracket_orders(
        symbol="WAL/USDT",
        take_profit=0.0690,
        stop_loss=0.0576,
    )

    assert tp_order.order_type == OrderType.TAKE_PROFIT
    assert tp_order.price == 0.0690
    assert sl_order.order_type == OrderType.STOP_LOSS
    assert sl_order.price == 0.0576

    # Verify estimated PnLs
    assert pos.tp_pnl_usd is not None
    assert pos.tp_pnl_pct is not None
    assert pos.sl_pnl_usd is not None
    assert pos.sl_pnl_pct is not None
    assert round(pos.tp_pnl_pct, 1) == 15.0
    assert round(pos.sl_pnl_pct, 1) == -4.0

    # Simulate price hitting TP
    ticks = {"WAL/USDT": {"price": 0.0700, "bid": 0.0695, "ask": 0.0700, "high": 0.0705, "low": 0.0650}}
    filled = broker.check_pending_orders(ticks)
    assert len(filled) == 1
    assert filled[0].order_type == OrderType.TAKE_PROFIT

    # Verify SL order was cancelled (OCO)
    pending_orders = broker.db.get_pending_orders()
    assert len(pending_orders) == 0


def test_bracket_orders_stop_loss_trigger(broker: ShadowExecutionBroker):
    """Verifies Stop-Loss execution as Taker and OCO cancellation of Take-Profit."""
    broker.set_market_price("ETH/USDT", price=3000.0)
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )

    tp_order, sl_order = broker.create_bracket_orders(
        symbol="ETH/USDT",
        take_profit=3300.0,
        stop_loss=2850.0,
    )

    # Price drops to 2840.0 hitting Stop Loss
    ticks = {"ETH/USDT": {"price": 2840.0, "bid": 2839.0, "ask": 2841.0, "high": 2900.0, "low": 2835.0}}
    filled = broker.check_pending_orders(ticks)
    assert len(filled) == 1
    assert filled[0].order_type == OrderType.STOP_LOSS

    # TP order should be cancelled
    pending = broker.db.get_pending_orders()
    assert len(pending) == 0

    # Trade should be recorded with realized loss
    trades = broker.db.get_trades(symbol="ETH/USDT")
    assert len(trades) >= 2
    latest = trades[0]
    assert latest["order_id"] == sl_order.id
    assert latest["is_maker"] == 0  # Taker execution for stop loss
    assert latest["realized_pnl"] < 0  # Realized loss


def test_broker_update_stop_loss(broker: ShadowExecutionBroker):
    """Verifies ratcheting stop-loss replaces pending order and locks profit."""
    broker.set_market_price("ETH/USDT", price=3000.0)
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )

    broker.create_bracket_orders(
        symbol="ETH/USDT",
        take_profit=3300.0,
        stop_loss=2850.0,
    )

    # Ratchet SL above entry to lock in profit
    updated_sl = broker.update_stop_loss("ETH/USDT", 3050.0)
    assert updated_sl is not None
    assert updated_sl.price == 3050.0

    pos = broker.portfolio.positions["ETH/USDT"]
    assert pos.stop_loss == 3050.0
    assert pos.is_profit_locked is True
    assert round(pos.sl_pnl_pct, 1) == 1.7

    # Verify pending orders in DB
    pending = broker.db.get_pending_orders()
    assert len(pending) == 2  # 1 TP + 1 new SL
    sl_orders = [o for o in pending if o.order_type == OrderType.STOP_LOSS]
    assert len(sl_orders) == 1
    assert sl_orders[0].price == 3050.0


def test_position_reversal_clears_brackets_and_cancels_orders(broker: ShadowExecutionBroker):
    """Verifies that closing or flipping a position cancels orphaned TP/SL orders and clears brackets in DB."""
    broker.set_market_price("SOL/USDT", price=200.0)
    broker.create_order(
        symbol="SOL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        strategy="Early Breakout",
    )
    broker.create_bracket_orders(symbol="SOL/USDT", take_profit=230.0, stop_loss=192.0)
    assert len(broker.db.get_pending_orders(symbol="SOL/USDT")) == 2

    # Reverse position from 10 LONG to 5 SHORT via SELL 15
    broker.create_order(
        symbol="SOL/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=15.0,
        strategy="Cross-Sectional Momentum",
    )

    pos = broker.portfolio.positions["SOL/USDT"]
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == 5.0
    assert pos.take_profit is None
    assert pos.stop_loss is None
    assert pos.strategy == "Cross-Sectional Momentum"
    assert len(broker.db.get_pending_orders(symbol="SOL/USDT")) == 0

    db_pos = broker.db.get_position("SOL/USDT")
    assert db_pos["side"] == "SHORT"
    assert db_pos["take_profit"] is None
    assert db_pos["stop_loss"] is None
    assert db_pos["strategy"] == "Cross-Sectional Momentum"


