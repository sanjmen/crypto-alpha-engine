"""
Student-t Heavy-Tail Distribution Estimator and Tail Risk (VaR / CVaR) Engine.
Ported and adapted from CrunchDAO Synth for risk controls.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy import stats


class StudentTDensityEstimator:
    """
    Fits Student-t distribution with calibrated degrees of freedom nu,
    capturing fat tails and extreme kurtosis in crypto return series.
    """

    def __init__(self, default_df: float = 4.5, min_df: float = 2.5, max_df: float = 30.0):
        self.default_df = default_df
        self.min_df = min_df
        self.max_df = max_df
        self.df: float = default_df
        self.loc: float = 0.0
        self.scale: float = 0.02
        self.is_fitted: bool = False

    def fit(self, returns: Union[np.ndarray, list]) -> "StudentTDensityEstimator":
        r = np.asarray(returns, dtype=np.float64).flatten()
        if len(r) < 30:
            self.df = self.default_df
            self.loc = float(np.mean(r)) if len(r) > 0 else 0.0
            self.scale = float(np.std(r)) if len(r) > 1 else 0.02
            self.is_fitted = True
            return self

        try:
            # Maximum likelihood estimation of Student-t parameters (df, loc, scale)
            df, loc, scale = stats.t.fit(r)
            self.df = float(np.clip(df, self.min_df, self.max_df))
            self.loc = float(loc)
            self.scale = float(max(scale, 1e-5))
            self.is_fitted = True
        except Exception:
            self.df = self.default_df
            self.loc = float(np.mean(r))
            self.scale = float(np.std(r))
            self.is_fitted = True

        return self

    def value_at_risk(self, alpha: float = 0.05) -> float:
        """
        Parametric Value at Risk (VaR) at significance level alpha (e.g. 5% or 1%).
        Returns the negative return threshold: P(r < -VaR) = alpha.
        """
        q = stats.t.ppf(alpha, df=self.df, loc=self.loc, scale=self.scale)
        return float(-q)

    def conditional_value_at_risk(self, alpha: float = 0.05, num_simulations: int = 10_000) -> float:
        """
        Parametric Expected Shortfall / Conditional VaR (CVaR).
        Average loss given that the loss exceeds the VaR threshold.
        """
        sims = stats.t.rvs(df=self.df, loc=self.loc, scale=self.scale, size=num_simulations)
        var_thresh = -self.value_at_risk(alpha)
        tail_losses = sims[sims <= var_thresh]
        if len(tail_losses) == 0:
            return float(-var_thresh)
        return float(-np.mean(tail_losses))
