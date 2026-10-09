"""
Unit tests for PaperExecutionBroker.
"""

import pytest
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.paper_broker import PaperExecutionBroker


def test_paper_broker_initialization():
    broker = PaperExecutionBroker(initial_cash=50_000.0)
    portfolio = broker.get_portfolio()
    assert portfolio.cash == 50_000.0
    assert portfolio.total_equity == 50_000.0
    assert len(portfolio.positions) == 0


def test_market_buy_and_sell_order():
    broker = PaperExecutionBroker(
        initial_cash=100_000.0,
        taker_fee_pct=0.0004,
        slippage_bps=2.0,   # 0.02% slippage
    )
    broker.set_market_price("BTCUSDT", 50_000.0)

    # Buy 1 BTC
    order = broker.create_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )

    assert order.status == OrderStatus.FILLED
    # Slippage: 50,000 * (1 + 0.0002) = 50010.0
    assert order.average_fill_price == pytest.approx(50010.0, rel=1e-5)

    portfolio = broker.get_portfolio()
    pos = portfolio.positions["BTCUSDT"]
    assert pos.side == PositionSide.LONG
    assert pos.quantity == 1.0
    assert pos.entry_price == pytest.approx(50010.0, rel=1e-5)

    # Price moves to 55,000
    broker.set_market_price("BTCUSDT", 55_000.0)
    assert portfolio.positions["BTCUSDT"].unrealized_pnl > 4900.0

    # Sell 1 BTC
    sell_order = broker.create_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )
    assert sell_order.status == OrderStatus.FILLED
    # Position should now be flat
    assert portfolio.positions["BTCUSDT"].side == PositionSide.FLAT
    assert portfolio.positions["BTCUSDT"].quantity == 0.0
    # Portfolio cash should reflect the profit minus fees
    assert portfolio.cash > 104_000.0


def test_short_position_lifecycle():
    broker = PaperExecutionBroker(initial_cash=100_000.0, slippage_bps=0.0, taker_fee_pct=0.0)
    broker.set_market_price("ETHUSDT", 3000.0)

    # Open short: sell 5 ETH
    broker.create_order(
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=5.0,
    )
    portfolio = broker.get_portfolio()
    pos = portfolio.positions["ETHUSDT"]
    assert pos.side == PositionSide.SHORT
    assert pos.quantity == 5.0
    assert pos.entry_price == 3000.0

    # Price drops to 2800 -> profitable short
    broker.set_market_price("ETHUSDT", 2800.0)
    assert pos.unrealized_pnl == pytest.approx(5.0 * 200.0)  # +$1000

    # Cover short: buy 5 ETH
    broker.create_order(
        symbol="ETHUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=5.0,
    )
    assert pos.side == PositionSide.FLAT
    assert portfolio.cash == pytest.approx(101_000.0)
