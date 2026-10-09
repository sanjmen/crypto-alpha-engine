"""
Unit tests for SmartExecutionRouter.
"""

import pytest
from src.domain.enums import OrderSide, OrderType
from src.execution.execution_router import ExecutionUrgency, SmartExecutionRouter
from src.execution.paper_broker import PaperExecutionBroker


def test_smart_execution_router_maker_and_taker():
    broker = PaperExecutionBroker(initial_cash=100_000.0)
    router = SmartExecutionRouter(broker=broker)

    broker.set_market_price("BTCUSDT", 50_000.0)

    # Low urgency -> Maker Limit order at the touch
    order_maker = router.route_order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.1,
        current_price=50_000.0,
        urgency=ExecutionUrgency.LOW,
        bid_ask_spread_bps=2.0,
    )
    assert order_maker.order_type == OrderType.LIMIT
    assert order_maker.price < 50_000.0  # Placed at bid (inside spread)

    # High urgency (emergency / risk brake) -> Taker Market order
    order_taker = router.route_order(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        quantity=0.1,
        current_price=50_000.0,
        urgency=ExecutionUrgency.HIGH,
    )
    assert order_taker.order_type == OrderType.MARKET


def test_maker_fill_probability_decay():
    # Fill prob for tight order (0.5 bps) with 2% vol
    prob_tight = SmartExecutionRouter.estimate_maker_fill_probability(
        distance_to_mid_bps=0.5,
        parkinson_vol=0.02,
    )
    # Fill prob for distant order (300 bps = 3% away)
    prob_far = SmartExecutionRouter.estimate_maker_fill_probability(
        distance_to_mid_bps=300.0,
        parkinson_vol=0.02,
    )
    assert prob_tight > 0.90
    assert prob_far < 0.20
