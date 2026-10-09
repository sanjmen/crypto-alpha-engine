"""
Market Microstructure, Effective Spread, and Liquidity Shock Estimators.
Ported and adapted from CrunchDAO Synth for production crypto trading.

Includes:
1. Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator.
2. Roll (1984) Effective Spread from serial covariance of return increments.
3. Amihud (2002) Illiquidity Ratio.
"""

from typing import List, Union
import numpy as np
import pandas as pd


def corwin_schultz_spread(
    high: Union[np.ndarray, pd.Series, List[float]],
    low: Union[np.ndarray, pd.Series, List[float]],
) -> np.ndarray:
    """
    Computes Corwin-Schultz (2012) Bid-Ask Spread from consecutive high-low pairs.
    Returns effective percentage spread per bar (floored at 0).
    """
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)

    n = len(h)
    if n < 2 or len(l) != n:
        return np.zeros(n, dtype=np.float64)

    spreads = np.zeros(n, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2

    for t in range(1, n):
        h0, l0 = h[t - 1], l[t - 1]
        h1, l1 = h[t], l[t]

        if h0 <= 0 or l0 <= 0 or h1 <= 0 or l1 <= 0 or h0 < l0 or h1 < l1:
            spreads[t] = 0.0
            continue

        hl0 = np.log(h0 / l0)
        hl1 = np.log(h1 / l1)
        beta = hl0**2 + hl1**2

        h2 = max(h0, h1)
        l2 = min(l0, l1)
        gamma = (np.log(h2 / l2)) ** 2

        alpha = (sqrt2 * np.sqrt(beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        if alpha < 0:
            spreads[t] = 0.0
        else:
            spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
            spreads[t] = max(0.0, float(spread))

    return spreads


def roll_effective_spread(prices: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Computes Roll (1984) Effective Spread from serial covariance of price increments.
    S = 2 * sqrt(-cov(dr_t, dr_{t-1})) when cov < 0, else 0.
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 3:
        return 0.0

    rets = np.diff(np.log(p))
    cov = np.cov(rets[1:], rets[:-1])[0, 1] if len(rets) > 2 else 0.0

    if cov < 0:
        return float(2.0 * np.sqrt(-cov))
    return 0.0


def amihud_illiquidity(
    returns: Union[np.ndarray, pd.Series, List[float]],
    volumes: Union[np.ndarray, pd.Series, List[float]],
) -> float:
    """
    Computes Amihud (2002) Illiquidity ratio:
    ILLIQ = mean( |r_t| / Volume_t )
    """
    r = np.asarray(returns, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)

    if len(r) == 0 or len(v) != len(r):
        return 0.0

    valid_mask = v > 0
    if not np.any(valid_mask):
        return 0.0

    r_v = np.abs(r[valid_mask])
    v_v = v[valid_mask]
    illiq = np.mean(r_v / v_v)
    return float(illiq)
