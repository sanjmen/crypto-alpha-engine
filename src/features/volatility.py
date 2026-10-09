"""
High-Efficiency Intraday Volatility Estimators and Online Recursive Variance Filters.
Ported and adapted from CrunchDAO Synth for production crypto trading.

Includes:
1. Realized Volatility (close-to-close)
2. Parkinson Volatility (high-low extreme value)
3. Garman-Klass Volatility (minimum-variance OHLC)
4. Rogers-Satchell Volatility (drift-independent OHLC)
5. Online EWMA Filter (RiskMetrics lambda)
6. Online GARCH(1,1) Recursive Filter with forward projection
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def realized_volatility(
    prices: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,  # 1h bars in crypto year
) -> float:
    """
    Computes classical Close-to-Close Realized Volatility.
    sigma_RV = sqrt( mean(r_t^2) )
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 2:
        return 0.0

    rets = np.diff(np.log(p))
    rv_var = np.mean(rets**2)
    vol = np.sqrt(max(rv_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def parkinson_volatility(
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Parkinson (1980) High-Low Range Volatility estimator.
    sigma_P^2 = (1 / (4 * ln(2) * N)) * sum( (ln(H/L))^2 )
    Statistically ~5x more efficient than Close-to-Close volatility.
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)

    if len(h) == 0 or len(l) == 0 or len(h) != len(l):
        return 0.0

    valid_mask = (h > 0) & (l > 0) & (h >= l)
    if not np.any(valid_mask):
        return 0.0

    h_v = h[valid_mask]
    l_v = l[valid_mask]

    log_hl = np.log(h_v / l_v)
    factor = 1.0 / (4.0 * np.log(2.0))
    p_var = factor * np.mean(log_hl**2)
    vol = np.sqrt(max(p_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def garman_klass_volatility(
    open_: Union[np.ndarray, pd.Series, List[float]],
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    close: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Garman-Klass (1980) OHLC Volatility estimator.
    sigma_GK^2 = mean( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 )
    Statistically ~8x more efficient than Close-to-Close volatility.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = len(o)
    if n == 0 or not (len(h) == n and len(l) == n and len(c) == n):
        return 0.0

    valid_mask = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    if not np.any(valid_mask):
        return 0.0

    o_v = o[valid_mask]
    h_v = h[valid_mask]
    l_v = l[valid_mask]
    c_v = c[valid_mask]

    term1 = 0.5 * (np.log(h_v / l_v))**2
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(c_v / o_v))**2

    gk_var = np.mean(term1 - term2)
    vol = np.sqrt(max(gk_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def rogers_satchell_volatility(
    open_: Union[np.ndarray, pd.Series, List[float]],
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
    close: Union[np.ndarray, pd.Series, List[float]],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Rogers-Satchell (1991) Volatility estimator.
    sigma_RS^2 = mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    Handles non-zero drift robustly.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)

    n = len(o)
    if n == 0 or not (len(h) == n and len(l) == n and len(c) == n):
        return 0.0

    valid_mask = (o > 0) & (h > 0) & (l > 0) & (c > 0) & (h >= l)
    if not np.any(valid_mask):
        return 0.0

    o_v = o[valid_mask]
    h_v = h[valid_mask]
    l_v = l[valid_mask]
    c_v = c[valid_mask]

    u = np.log(h_v / c_v)
    v = np.log(h_v / o_v)
    w = np.log(l_v / c_v)
    x = np.log(l_v / o_v)

    rs_var = np.mean(u * v + w * x)
    vol = np.sqrt(max(rs_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


class OnlineEWMAVolatility:
    """
    Online recursive Exponentially Weighted Moving Average (EWMA) volatility filter.
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    """

    def __init__(self, decay: float = 0.94, initial_vol: float = 0.02):
        self.decay = decay
        self.variance = initial_vol**2
        self.step_count = 0

    def update(self, return_val: float) -> float:
        self.variance = self.decay * self.variance + (1.0 - self.decay) * (return_val**2)
        self.step_count += 1
        return float(np.sqrt(max(self.variance, 1e-12)))

    @property
    def current_volatility(self) -> float:
        return float(np.sqrt(max(self.variance, 1e-12)))


class OnlineGARCH11:
    """
    Online recursive GARCH(1,1) filter.
    sigma_{t+1}^2 = omega + alpha * r_t^2 + beta * sigma_t^2
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.08,
        beta: float = 0.90,
        initial_vol: float = 0.02,
    ):
        assert alpha + beta < 1.0, "GARCH(1,1) requires alpha + beta < 1 for stationarity"
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.variance = initial_vol**2
        self.step_count = 0

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)

    def update(self, return_val: float) -> float:
        self.variance = self.omega + self.alpha * (return_val**2) + self.beta * self.variance
        self.step_count += 1
        return float(np.sqrt(max(self.variance, 1e-12)))

    def forecast_variance(self, horizon_steps: int) -> float:
        """Forecast cumulative variance over horizon_steps."""
        persistence = self.alpha + self.beta
        var_long = self.unconditional_variance

        cum_var = 0.0
        v_h = self.variance
        for _ in range(horizon_steps):
            v_h = var_long + (persistence) * (v_h - var_long)
            cum_var += v_h
        return float(cum_var)
