"""
Domain Entities and Value Objects for Crypto Alpha Engine.
Strictly decoupled from third-party frameworks, adhering to Clean Architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math

from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide


def _utc_now() -> datetime:
    """Returns timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


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
        return float(math.log(self.close / self.open)) if (self.open > 0 and self.close > 0) else 0.0

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
    score: float  # Normalized score (-1.0 to 1.0)
    confidence: float = 1.0  # Estimation certainty [0.0, 1.0]
    horizon_minutes: int = 60
    metadata: dict[str, float] = field(default_factory=dict)


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
    price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=_utc_now)
    strategy: str | None = None


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
    peak_price: float = 0.0
    take_profit: float | None = None
    stop_loss: float | None = None
    strategy: str | None = None
    opened_at: datetime | None = None

    def _compute_pnl_usd_at_price(self, target_price: float | None) -> float | None:
        """Computes dollar PnL if the position were closed at `target_price`."""
        if target_price is None:
            return None
        if self.side == PositionSide.LONG:
            return float(self.quantity * (target_price - self.entry_price))
        if self.side == PositionSide.SHORT:
            return float(self.quantity * (self.entry_price - target_price))
        return 0.0

    def _compute_pnl_pct_at_price(self, target_price: float | None) -> float | None:
        """Computes percentage PnL if the position were closed at `target_price`."""
        if target_price is None or self.entry_price <= 0:
            return None
        if self.side == PositionSide.LONG:
            return float((target_price / self.entry_price - 1.0) * 100.0)
        if self.side == PositionSide.SHORT:
            return float((1.0 - target_price / self.entry_price) * 100.0)
        return 0.0

    @property
    def is_profit_locked(self) -> bool:
        """Indicates whether stop-loss has ratcheted into profitable territory."""
        if self.stop_loss is None or self.entry_price <= 0:
            return False
        if self.side == PositionSide.LONG:
            return self.stop_loss > self.entry_price
        if self.side == PositionSide.SHORT:
            return self.stop_loss < self.entry_price
        return False

    @property
    def market_value(self) -> float:
        return float(self.quantity * self.current_price)

    @property
    def unrealized_pnl(self) -> float:
        return float(self._compute_pnl_usd_at_price(self.current_price) or 0.0)

    @property
    def take_profit_pnl_usd(self) -> float | None:
        return self._compute_pnl_usd_at_price(self.take_profit)

    @property
    def tp_pnl_usd(self) -> float | None:
        return self.take_profit_pnl_usd

    @property
    def take_profit_pnl_pct(self) -> float | None:
        return self._compute_pnl_pct_at_price(self.take_profit)

    @property
    def tp_pnl_pct(self) -> float | None:
        return self.take_profit_pnl_pct

    @property
    def stop_loss_pnl_usd(self) -> float | None:
        return self._compute_pnl_usd_at_price(self.stop_loss)

    @property
    def sl_pnl_usd(self) -> float | None:
        return self.stop_loss_pnl_usd

    @property
    def stop_loss_pnl_pct(self) -> float | None:
        return self._compute_pnl_pct_at_price(self.stop_loss)

    @property
    def sl_pnl_pct(self) -> float | None:
        return self.stop_loss_pnl_pct


@dataclass
class Portfolio:
    """
    Aggregate portfolio representation.
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    initial_cash: float = 100_000.0

    @property
    def total_equity(self) -> float:
        unrealized = sum(position.unrealized_pnl for position in self.positions.values())
        return float(self.cash + unrealized)

    @property
    def gross_leverage(self) -> float:
        equity = self.total_equity
        if equity <= 0:
            return 0.0
        gross_value = sum(abs(position.market_value) for position in self.positions.values())
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
    strategy: str | None = None
