"""
Unit tests for CCXTExecutionAdapter and safety circuit breakers.
"""

import pytest

from src.domain.enums import OrderSide, OrderStatus, OrderType
from src.execution.live_broker import (
    CCXTExecutionAdapter,
    MaxDrawdownStopError,
    MaxLeverageExceededError,
    MaxOrderNotionalExceededError,
)


def test_max_order_notional_circuit_breaker():
    adapter = CCXTExecutionAdapter(
        dry_run=True,
        max_order_notional=10_000.0,
    )
    # Order worth $20,000 should immediately trip circuit breaker
    with pytest.raises(MaxOrderNotionalExceededError):
        adapter.create_order(
            symbol="BTC/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.5,
            price=40_000.0,  # 0.5 * 40k = 20k > 10k
        )


def test_max_leverage_circuit_breaker():
    adapter = CCXTExecutionAdapter(
        dry_run=True,
        max_gross_leverage=1.5,
        max_order_notional=200_000.0,
    )
    # Portfolio equity is 100k. Limit is 1.5x (150k).
    # Order of 160k should be blocked
    with pytest.raises(MaxLeverageExceededError):
        adapter.create_order(
            symbol="ETH/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=80.0,
            price=2000.0,  # 80 * 2000 = 160,000 > 150,000
        )


def test_drawdown_stop_circuit_breaker():
    adapter = CCXTExecutionAdapter(
        dry_run=True,
        max_drawdown_stop=0.10,  # 10% max DD
    )
    # Simulate a crash: cash drops from 100k to 85k (15% DD)
    adapter.portfolio.cash = 85_000.0

    with pytest.raises(MaxDrawdownStopError):
        adapter.create_order(
            symbol="SOL/USDT:USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=100.0,
        )


def test_dry_run_successful_execution():
    adapter = CCXTExecutionAdapter(
        dry_run=True,
        max_order_notional=25_000.0,
        max_gross_leverage=2.5,
    )
    order = adapter.create_order(
        symbol="BTC/USDT:USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.2,
        price=50_000.0,  # $10,000 notional
    )
    assert order.status == OrderStatus.FILLED
    assert order.filled_quantity == 0.2
    assert "BTC/USDT:USDT" in adapter.portfolio.positions
    assert len(adapter.get_trade_history()) == 1


def test_adapter_initialization_with_credentials():
    from src.security.vault import ExchangeCredentials
    creds = ExchangeCredentials(
        exchange_id="binance",
        api_key="mock_key_12345",
        api_secret="mock_secret_67890",
    )
    adapter = CCXTExecutionAdapter(credentials=creds, dry_run=True)
    assert adapter.client.apiKey == "mock_key_12345"
    assert adapter.client.secret == "mock_secret_67890"

