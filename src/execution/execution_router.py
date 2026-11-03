"""
Execution Alpha & Smart Order Router.
Distinguishes between passive alpha rebalancing (Maker limit orders with zero slippage)
and aggressive emergency stops (Taker market orders for immediate fills).
Models fill probabilities and adverse selection.
"""

from enum import Enum
import logging
from typing import Dict, Optional, Tuple
import numpy as np

from src.domain.entities import Order, Portfolio
from src.domain.enums import OrderSide, OrderType
from src.ports.execution_ports import IExecutionClient

logger = logging.getLogger(__name__)


class ExecutionUrgency(str, Enum):
    LOW = "low"          # Routine rebalancing -> Passive Maker (Post-Only)
    MEDIUM = "medium"    # Target position shift -> Hybrid / TWAP
    HIGH = "high"        # Risk stop, jump brake, or emergency -> Aggressive Taker (Market)


class SmartExecutionRouter:
    """
    Routes orders intelligently between Maker (passive limit at touch)
    and Taker (aggressive market) to maximize execution alpha and minimize fee drag.
    """

    def __init__(
        self,
        broker: IExecutionClient,
        default_urgency: ExecutionUrgency = ExecutionUrgency.LOW,
        maker_fee_pct: float = 0.0002,     # 0.02% maker
        taker_fee_pct: float = 0.0004,     # 0.04% taker
        tick_offset_bps: float = 0.5,      # 0.5 bp offset into book
    ):
        self.broker = broker
        self.default_urgency = default_urgency
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.tick_offset = tick_offset_bps / 10_000.0

    @staticmethod
    def compute_order_book_imbalance(
        order_book: Optional[Dict],
        levels: int = 5,
    ) -> float:
        """
        Computes Top-N Level-2 Order Book Imbalance (OBI):
            OBI_N = (sum(V_bid, 1..N) - sum(V_ask, 1..N)) / (sum(V_bid, 1..N) + sum(V_ask, 1..N))
        Returns a scalar in [-1.0, +1.0], or 0.0 if order book depth is unavailable.
        """
        if not order_book or not isinstance(order_book, dict):
            return 0.0
        bids = order_book.get("bids") or []
        asks = order_book.get("asks") or []
        if not bids and not asks:
            return 0.0
        v_bid = sum(float(level[1]) for level in bids[:levels] if len(level) >= 2)
        v_ask = sum(float(level[1]) for level in asks[:levels] if len(level) >= 2)
        tot = v_bid + v_ask
        if tot <= 1e-12:
            return 0.0
        return float(np.clip((v_bid - v_ask) / tot, -1.0, 1.0))

    def route_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        urgency: Optional[ExecutionUrgency] = None,
        bid_ask_spread_bps: float = 2.0,
        strategy: Optional[str] = None,
        order_book: Optional[Dict] = None,
        obi: Optional[float] = None,
    ) -> Order:
        """
        Routes order according to execution urgency and Level-2 Order Book Imbalance (Issue #47).
        """
        urgency = urgency or self.default_urgency
        extra_kwargs = {"strategy": strategy} if strategy is not None else {}

        if urgency == ExecutionUrgency.HIGH:
            # Aggressive market order
            return self.broker.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=current_price,
                **extra_kwargs,
            )
        else:
            # Passive maker limit order with optional Level-2 OBI adaptive offset (Issue #47)
            spread = (bid_ask_spread_bps / 10_000.0) * current_price
            half_spread = spread / 2.0

            obi_val: Optional[float] = None
            if obi is not None:
                obi_val = float(np.clip(obi, -1.0, 1.0))
            elif order_book is not None:
                obi_val = self.compute_order_book_imbalance(order_book, levels=5)

            if obi_val is not None:
                order_sign = 1.0 if side == OrderSide.BUY else -1.0
                flow_align = order_sign * obi_val
                if flow_align <= -0.25:
                    # Adverse flow pushing into our passive limit: post 2.5x half-spread (2.5 bps) deeper
                    offset = 2.5 * half_spread
                elif flow_align >= 0.30:
                    # Strong favorable flow / breakout: join inside mid-touch (0.0 bps offset) for immediate fill
                    offset = 0.0
                else:
                    offset = half_spread
            else:
                offset = half_spread

            if side == OrderSide.BUY:
                limit_price = current_price - offset
            else:
                limit_price = current_price + offset

            return self.broker.create_order(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
                **extra_kwargs,
            )

    @staticmethod
    def estimate_maker_fill_probability(
        distance_to_mid_bps: float,
        parkinson_vol: float,
        time_horizon_bars: int = 1,
    ) -> float:
        """
        Empirically models fill probability of a passive limit order
        given asset volatility and distance from the mid-price.
        Using the first-passage time distribution of geometric Brownian motion.
        """
        if parkinson_vol <= 1e-6:
            return 0.5

        d = distance_to_mid_bps / 10_000.0
        scaled_vol = parkinson_vol * np.sqrt(time_horizon_bars)
        # Ratio of limit distance to expected range
        z = d / scaled_vol
        # Fill probability decays exponentially with distance
        prob = float(np.exp(-1.5 * z))
        return float(np.clip(prob, 0.05, 0.98))
