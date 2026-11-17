"""
Market Microstructure, Effective Spread, and Liquidity Shock Estimators.
Ported and adapted from CrunchDAO Synth for production crypto trading.

Includes:
1. Corwin-Schultz (2012) High-Low Bid-Ask Spread Estimator.
2. Roll (1984) Effective Spread from serial covariance of return increments.
3. Amihud (2002) Illiquidity Ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread(
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    """
    Computes Corwin-Schultz (2012) Bid-Ask Spread from consecutive high-low pairs.
    Returns effective percentage spread per bar (floored at 0).
    """
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)

    num_bars = len(high_arr)
    if num_bars < 2 or len(low_arr) != num_bars:
        return np.zeros(num_bars, dtype=np.float64)

    spreads = np.zeros(num_bars, dtype=np.float64)
    sqrt2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * sqrt2

    for bar_idx in range(1, num_bars):
        high_prev, low_prev = high_arr[bar_idx - 1], low_arr[bar_idx - 1]
        high_curr, low_curr = high_arr[bar_idx], low_arr[bar_idx]

        if (
            high_prev <= 0
            or low_prev <= 0
            or high_curr <= 0
            or low_curr <= 0
            or high_prev < low_prev
            or high_curr < low_curr
        ):
            spreads[bar_idx] = 0.0
            continue

        hl_prev = np.log(high_prev / low_prev)
        hl_curr = np.log(high_curr / low_curr)
        beta = hl_prev**2 + hl_curr**2

        high_2bar = max(high_prev, high_curr)
        low_2bar = min(low_prev, low_curr)
        gamma = (np.log(high_2bar / low_2bar)) ** 2

        alpha = (sqrt2 * np.sqrt(beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        if alpha < 0:
            spreads[bar_idx] = 0.0
        else:
            spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
            spreads[bar_idx] = max(0.0, float(spread))

    return spreads


def rolling_corwin_schultz_spread(
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
    window: int = 24,
) -> np.ndarray:
    """
    Computes rolling mean of Corwin-Schultz spread over specified window.
    Useful for smoothing single-bar microstructure noise.
    """
    raw_spreads = corwin_schultz_spread(high, low)
    spread_series = pd.Series(raw_spreads)
    rolling_spread = spread_series.rolling(window=window, min_periods=1).mean().values
    return np.asarray(rolling_spread, dtype=np.float64)


def roll_effective_spread(prices: np.ndarray | pd.Series | list[float]) -> float:
    """
    Computes Roll (1984) Effective Spread from serial covariance of price increments.
    S = 2 * sqrt(-cov(dr_t, dr_{t-1})) when cov < 0, else 0.
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    prices_arr = prices_arr[np.isfinite(prices_arr) & (prices_arr > 0)]
    if len(prices_arr) < 4:
        return 0.0

    rets = np.diff(np.log(prices_arr))
    cov = float(np.cov(rets[1:], rets[:-1])[0, 1]) if len(rets) > 2 else 0.0

    if cov < 0:
        return float(2.0 * np.sqrt(-cov))
    return 0.0


def amihud_illiquidity(
    returns: np.ndarray | pd.Series | list[float],
    volumes: np.ndarray | pd.Series | list[float],
) -> float:
    """
    Computes Amihud (2002) Illiquidity ratio:
    ILLIQ = mean( |r_t| / Volume_t )
    """
    returns_arr = np.asarray(returns, dtype=np.float64)
    volumes_arr = np.asarray(volumes, dtype=np.float64)

    if len(returns_arr) == 0 or len(volumes_arr) != len(returns_arr):
        return 0.0

    valid_mask = (volumes_arr > 0) & np.isfinite(returns_arr) & np.isfinite(volumes_arr)
    if not np.any(valid_mask):
        return 0.0

    abs_rets = np.abs(returns_arr[valid_mask])
    valid_vols = volumes_arr[valid_mask]
    illiq = np.mean(abs_rets / valid_vols)
    return float(illiq)


def rolling_amihud_illiquidity(
    returns: np.ndarray | pd.Series | list[float],
    volumes: np.ndarray | pd.Series | list[float],
    window: int = 24,
) -> np.ndarray:
    """Computes rolling Amihud illiquidity over a specified bar window."""
    returns_arr = np.asarray(returns, dtype=np.float64)
    volumes_arr = np.asarray(volumes, dtype=np.float64)
    num_bars = len(returns_arr)
    if num_bars == 0 or len(volumes_arr) != num_bars:
        return np.zeros(num_bars, dtype=np.float64)

    valid_mask = (volumes_arr > 0) & np.isfinite(returns_arr) & np.isfinite(volumes_arr)
    ratio = np.zeros(num_bars, dtype=np.float64)
    np.divide(np.abs(returns_arr), volumes_arr, out=ratio, where=valid_mask)
    ratio_series = pd.Series(ratio)
    rolling_illiq = ratio_series.rolling(window=window, min_periods=1).mean().values
    return np.asarray(rolling_illiq, dtype=np.float64)
