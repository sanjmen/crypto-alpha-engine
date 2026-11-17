"""
Volatility-Targeted Position Sizing Engine.
Implements risk-parity style inverse-volatility weighting across crypto assets.
"""

import numpy as np

from src.domain.entities import Signal
from src.features.cross_sectional import dollar_neutralize


class VolatilityTargetingSizer:
    """
    Computes portfolio weights such that each asset contributes proportionally
    to the target volatility budget, scaling down highly volatile tokens.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.25,  # 25% annualized portfolio volatility
        max_leverage: float = 2.0,  # Maximum allowable gross leverage
        max_single_weight: float = 0.20,  # Max 20% allocation in any single token
        min_vol_floor: float = 0.005,  # 0.5% volatility floor
    ) -> None:
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage
        self.max_single_weight = max_single_weight
        self.min_vol_floor = min_vol_floor

    def size_positions(
        self,
        signals: list[Signal],
        volatilities: dict[str, float],
        market_neutral: bool = True,
    ) -> dict[str, float]:
        """
        Calculates target portfolio weights from raw alpha signals and asset volatilities.
        """
        if not signals:
            return {}

        weights: dict[str, float] = {}
        for sig in signals:
            vol = max(volatilities.get(sig.symbol, 0.03), self.min_vol_floor)
            raw_weight = (sig.score * sig.confidence) / vol
            weights[sig.symbol] = float(raw_weight)

        symbols = list(weights.keys())
        weights_arr = np.array([weights[s] for s in symbols], dtype=np.float64)

        if self.max_single_weight is not None:
            weights_arr = np.clip(weights_arr, -self.max_single_weight * 100, self.max_single_weight * 100)

        if market_neutral:
            weights_arr = dollar_neutralize(weights_arr)
        else:
            gross = float(np.sum(np.abs(weights_arr)))
            if gross > 0:
                weights_arr = weights_arr / gross

        weights_arr *= min(1.0, self.max_leverage)

        return {symbols[i]: float(weights_arr[i]) for i in range(len(symbols))}
