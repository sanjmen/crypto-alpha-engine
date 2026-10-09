"""
Unit tests for TradingDaemon with PaperExecutionBroker.
"""

import numpy as np
import pandas as pd
import pytest

from src.execution.paper_broker import PaperExecutionBroker
from src.execution.trading_daemon import TradingDaemon
from tests.test_meta_allocator import create_synthetic_data


def test_trading_daemon_rebalance_cycle():
    broker = PaperExecutionBroker(initial_cash=100_000.0)
    daemon = TradingDaemon(broker=broker, min_rebalance_notional=100.0)

    data = create_synthetic_data(120)
    current_prices = {
        "BTCUSDT": float(data["BTCUSDT"]["close"].iloc[-1]),
        "ETHUSDT": float(data["ETHUSDT"]["close"].iloc[-1]),
        "SOLUSDT": float(data["SOLUSDT"]["close"].iloc[-1]),
    }
    broker.set_market_prices(current_prices)
    volatilities = {"BTCUSDT": 0.02, "ETHUSDT": 0.03, "SOLUSDT": 0.04}

    orders = daemon.rebalance_cycle(
        market_data=data,
        current_prices=current_prices,
        volatilities=volatilities,
    )

    assert isinstance(orders, list)
    assert len(orders) > 0

    portfolio = broker.get_portfolio()
    # At least some positions were opened
    assert len(portfolio.positions) > 0
    # Portfolio equity remains close to initial (minus minor transaction fees)
    assert portfolio.total_equity == pytest.approx(100_000.0, rel=0.01)
