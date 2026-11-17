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

import numpy as np
import pandas as pd


def _validate_ohlc_arrays(
    open_: np.ndarray | pd.Series | list[float],
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
    close: np.ndarray | pd.Series | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Validates and filters OHLC arrays for strictly positive finite prices with high >= low."""
    open_arr = np.asarray(open_, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)

    num_bars = len(open_arr)
    if num_bars == 0 or not (len(high_arr) == num_bars and len(low_arr) == num_bars and len(close_arr) == num_bars):
        return None

    valid_mask = (
        np.isfinite(open_arr)
        & np.isfinite(high_arr)
        & np.isfinite(low_arr)
        & np.isfinite(close_arr)
        & (open_arr > 0)
        & (high_arr > 0)
        & (low_arr > 0)
        & (close_arr > 0)
        & (high_arr >= low_arr)
    )
    if not np.any(valid_mask):
        return None

    return open_arr[valid_mask], high_arr[valid_mask], low_arr[valid_mask], close_arr[valid_mask]


def realized_volatility(
    prices: np.ndarray | pd.Series | list[float],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,  # 1h bars in crypto year
) -> float:
    """
    Computes classical Close-to-Close Realized Volatility:
    sigma_RV = sqrt( mean(r_t^2) )
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    prices_arr = prices_arr[np.isfinite(prices_arr) & (prices_arr > 0)]
    if len(prices_arr) < 2:
        return 0.0

    rets = np.diff(np.log(prices_arr))
    rv_var = float(np.mean(rets**2))
    vol = np.sqrt(max(rv_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def parkinson_volatility(
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Parkinson (1980) High-Low Range Volatility estimator:
    sigma_P^2 = (1 / (4 * ln(2) * N)) * sum( (ln(H/L))^2 )
    Statistically ~5x more efficient than Close-to-Close volatility.
    """
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)

    if len(high_arr) == 0 or len(low_arr) == 0 or len(high_arr) != len(low_arr):
        return 0.0

    valid_mask = np.isfinite(high_arr) & np.isfinite(low_arr) & (high_arr > 0) & (low_arr > 0) & (high_arr >= low_arr)
    if not np.any(valid_mask):
        return 0.0

    high_valid = high_arr[valid_mask]
    low_valid = low_arr[valid_mask]

    log_hl = np.log(high_valid / low_valid)
    factor = 1.0 / (4.0 * np.log(2.0))
    p_var = float(factor * np.mean(log_hl**2))
    vol = np.sqrt(max(p_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def garman_klass_volatility(
    open_: np.ndarray | pd.Series | list[float],
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
    close: np.ndarray | pd.Series | list[float],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Garman-Klass (1980) OHLC Volatility estimator:
    sigma_GK^2 = mean( 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2 )
    Statistically ~8x more efficient than Close-to-Close volatility.
    """
    validated = _validate_ohlc_arrays(open_, high, low, close)
    if validated is None:
        return 0.0
    open_valid, high_valid, low_valid, close_valid = validated

    term1 = 0.5 * (np.log(high_valid / low_valid)) ** 2
    term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(close_valid / open_valid)) ** 2

    gk_var = float(np.mean(term1 - term2))
    vol = np.sqrt(max(gk_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


def rogers_satchell_volatility(
    open_: np.ndarray | pd.Series | list[float],
    high: np.ndarray | pd.Series | list[float],
    low: np.ndarray | pd.Series | list[float],
    close: np.ndarray | pd.Series | list[float],
    annualize: bool = False,
    periods_per_year: float = 365.25 * 24,
) -> float:
    """
    Computes Rogers-Satchell (1991) Volatility estimator:
    sigma_RS^2 = mean( ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O) )
    Handles non-zero drift robustly.
    """
    validated = _validate_ohlc_arrays(open_, high, low, close)
    if validated is None:
        return 0.0
    open_valid, high_valid, low_valid, close_valid = validated

    log_hc = np.log(high_valid / close_valid)
    log_ho = np.log(high_valid / open_valid)
    log_lc = np.log(low_valid / close_valid)
    log_lo = np.log(low_valid / open_valid)

    rs_var = float(np.mean(log_hc * log_ho + log_lc * log_lo))
    vol = np.sqrt(max(rs_var, 1e-12))

    if annualize:
        vol *= np.sqrt(periods_per_year)

    return float(vol)


class OnlineEWMAVolatility:
    """
    Online recursive Exponentially Weighted Moving Average (EWMA) volatility filter:
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2
    """

    def __init__(self, decay: float = 0.94, initial_vol: float = 0.02) -> None:
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
    Online recursive GARCH(1,1) filter:
    sigma_{t+1}^2 = omega + alpha * r_t^2 + beta * sigma_t^2
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.08,
        beta: float = 0.90,
        initial_vol: float = 0.02,
    ) -> None:
        if not (alpha >= 0.0 and beta >= 0.0 and alpha + beta < 1.0):
            raise ValueError("GARCH(1,1) requires alpha >= 0, beta >= 0, and alpha + beta < 1 for stationarity.")
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
        var_step = self.variance
        for _ in range(horizon_steps):
            var_step = var_long + persistence * (var_step - var_long)
            cum_var += var_step
        return float(cum_var)
