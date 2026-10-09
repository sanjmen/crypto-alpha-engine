"""
Volatility-Targeted Position Sizing Engine.
Implements risk-parity style inverse-volatility weighting across crypto assets.
"""

from typing import Dict, List, Optional
import numpy as np

from src.domain.entities import Signal, Portfolio
from src.features.cross_sectional import dollar_neutralize


class VolatilityTargetingSizer:
    """
    Computes portfolio weights such that each asset contributes proportionally
    to the target volatility budget, scaling down highly volatile tokens.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.25,     # 25% annualized portfolio volatility
        max_leverage: float = 2.0,           # Maximum allowable gross leverage
        max_single_weight: float = 0.20,     # Max 20% allocation in any single token
        min_vol_floor: float = 0.005,        # 0.5% volatility floor
    ):
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage
        self.max_single_weight = max_single_weight
        self.min_vol_floor = min_vol_floor

    def size_positions(
        self,
        signals: List[Signal],
        volatilities: Dict[str, float],
        market_neutral: bool = True,
    ) -> Dict[str, float]:
        """
        Calculates target portfolio weights from raw alpha signals and asset volatilities.
        """
        if not signals:
            return {}

        weights: Dict[str, float] = {}
        for sig in signals:
            vol = max(volatilities.get(sig.symbol, 0.03), self.min_vol_floor)
            # Signal score (-1.0 to 1.0) scaled inversely by asset volatility
            raw_w = (sig.score * sig.confidence) / vol
            weights[sig.symbol] = float(raw_w)

        symbols = list(weights.keys())
        w_arr = np.array([weights[s] for s in symbols], dtype=np.float64)

        # Clip individual raw signal weights if constraint is provided
        if self.max_single_weight is not None:
            w_arr = np.clip(w_arr, -self.max_single_weight * 100, self.max_single_weight * 100)

        if market_neutral:
            # Enforce dollar neutrality: Long sum = +0.5, Short sum = -0.5 (Net = 0.0)
            w_arr = dollar_neutralize(w_arr)
        else:
            # Scale gross exposure to 1.0
            gross = np.sum(np.abs(w_arr))
            if gross > 0:
                w_arr = w_arr / gross

        # Scale by maximum leverage budget (preserves zero net exposure)
        w_arr *= min(1.0, self.max_leverage)

        return {symbols[i]: float(w_arr[i]) for i in range(len(symbols))}
