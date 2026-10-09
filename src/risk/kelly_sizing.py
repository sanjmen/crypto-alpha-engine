"""
Fractional Kelly Criterion Position Sizing Engine.
"""

from typing import Dict, List
import numpy as np

from src.domain.entities import Signal


class FractionalKellySizer:
    """
    Computes optimal bet size using the Continuous Fractional Kelly Criterion:
    f^* = kappa * (mu - r_f) / sigma^2
    """

    def __init__(
        self,
        fraction: float = 0.25,              # Quarter-Kelly (conservative, avoids drawdown)
        risk_free_rate: float = 0.0,
        max_position_fraction: float = 0.15, # Max 15% in any single trade
    ):
        self.fraction = fraction
        self.risk_free_rate = risk_free_rate
        self.max_position_fraction = max_position_fraction

    def compute_size(
        self,
        expected_return: float,
        variance: float,
    ) -> float:
        var = max(variance, 1e-6)
        excess_return = expected_return - self.risk_free_rate
        raw_kelly = (excess_return / var) * self.fraction
        return float(np.clip(raw_kelly, -self.max_position_fraction, self.max_position_fraction))
