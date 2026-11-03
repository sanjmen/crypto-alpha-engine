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


def test_m13_issue_47_level2_obi_adaptive_maker_routing():
    """
    Tests Issue #47 (Level-2 Order Book Imbalance OBI_5 Adaptive Maker Routing):
    1) compute_order_book_imbalance accurately computes (V_bid - V_ask) / (V_bid + V_ask) across top 5 levels.
    2) Adverse flow (e.g. BUY with OBI <= -0.25) places limit 2.5x half-spread deeper to capture price improvement.
    3) Strong favorable flow (e.g. BUY with OBI >= +0.30) joins mid-price touch (0.0 offset) for immediate fill.
    """
    broker = PaperExecutionBroker(initial_cash=100_000.0)
    router = SmartExecutionRouter(broker=broker)
    broker.set_market_price("SOLUSDT", 200.0)

    ob_heavy_ask = {
        "bids": [[199.98, 10.0]] * 5,  # total bid = 50
        "asks": [[200.02, 30.0]] * 5,  # total ask = 150 -> OBI = (50 - 150) / 200 = -0.50
    }
    obi_val = SmartExecutionRouter.compute_order_book_imbalance(ob_heavy_ask, levels=5)
    assert pytest.approx(obi_val, abs=1e-6) == -0.50

    # 1. BUY into heavy ask pressure (flow_align = -0.50 <= -0.25) -> 2.5 bps below 200.0 = 199.95
    ord_adverse = router.route_order(
        symbol="SOLUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        current_price=200.0,
        urgency=ExecutionUrgency.LOW,
        bid_ask_spread_bps=2.0,
        order_book=ob_heavy_ask,
    )
    assert pytest.approx(ord_adverse.price, abs=1e-6) == 200.0 - 0.05

    # 2. BUY into strong bid pressure (OBI = +0.50 >= +0.30) -> inside mid touch = 200.00
    ord_favorable = router.route_order(
        symbol="SOLUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        current_price=200.0,
        urgency=ExecutionUrgency.LOW,
        bid_ask_spread_bps=2.0,
        obi=0.50,
    )
    assert pytest.approx(ord_favorable.price, abs=1e-6) == 200.0

