"""
Domain Entities and Value Objects for Crypto Alpha Engine.
Strictly decoupled from third-party frameworks, adhering to Clean Architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide, MarketRegime


@dataclass(frozen=True)
class Bar:
    """
    Standard OHLCV price bar.
    """
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def log_return(self) -> float:
        return float(np.log(self.close / self.open)) if self.open > 0 else 0.0

    @property
    def range(self) -> float:
        return float(self.high - self.low)


@dataclass(frozen=True)
class Tick:
    """
    Individual market transaction tick.
    """
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: OrderSide


@dataclass(frozen=True)
class FundingRate:
    """
    Perpetual futures funding rate record.
    """
    timestamp: datetime
    symbol: str
    rate: float
    next_funding_time: datetime


@dataclass(frozen=True)
class Signal:
    """
    Quantitative alpha signal emitted by an alpha model.
    """
    timestamp: datetime
    symbol: str
    score: float                # Normalized score (-1.0 to 1.0)
    confidence: float = 1.0     # Estimation certainty [0.0, 1.0]
    horizon_minutes: int = 60
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class Order:
    """
    Exchange trade order.
    """
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """
    Portfolio asset position.
    """
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return float(self.quantity * self.current_price)

    @property
    def unrealized_pnl(self) -> float:
        if self.side == PositionSide.LONG:
            return float(self.quantity * (self.current_price - self.entry_price))
        elif self.side == PositionSide.SHORT:
            return float(self.quantity * (self.entry_price - self.current_price))
        return 0.0


@dataclass
class Portfolio:
    """
    Aggregate portfolio representation.
    """
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    initial_cash: float = 100_000.0

    @property
    def total_equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return float(self.cash + unrealized)

    @property
    def gross_leverage(self) -> float:
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        gross_value = sum(abs(p.market_value) for p in self.positions.values())
        return float(gross_value / equity)


@dataclass(frozen=True)
class Trade:
    """
    Executed trade transaction record.
    """
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: datetime
