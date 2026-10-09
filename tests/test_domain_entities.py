from datetime import datetime, timezone
import pytest

from src.domain.entities import Bar, Signal, Order, Position, Portfolio, Trade
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide, MarketRegime


def test_bar_properties():
    bar = Bar(
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC/USDT",
        open=50000.0,
        high=51000.0,
        low=49500.0,
        close=50500.0,
        volume=120.5,
    )
    assert bar.range == 1500.0
    assert bar.log_return > 0.0


def test_position_and_portfolio_pnl():
    long_pos = Position(
        symbol="BTC/USDT",
        side=PositionSide.LONG,
        quantity=1.5,
        entry_price=50000.0,
        current_price=52000.0,
    )
    assert long_pos.market_value == 78000.0
    assert long_pos.unrealized_pnl == 3000.0

    short_pos = Position(
        symbol="ETH/USDT",
        side=PositionSide.SHORT,
        quantity=10.0,
        entry_price=3000.0,
        current_price=2800.0,
    )
    assert short_pos.unrealized_pnl == 2000.0

    portfolio = Portfolio(
        cash=50000.0,
        positions={"BTC/USDT": long_pos, "ETH/USDT": short_pos},
        initial_cash=50000.0,
    )
    assert portfolio.total_equity == 50000.0 + 3000.0 + 2000.0
    assert portfolio.gross_leverage > 0.0
