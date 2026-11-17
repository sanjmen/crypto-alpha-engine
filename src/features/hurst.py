"""
Fractal Hurst Exponent Estimation and Anomalous Temporal Diffusion Scaling.
Ported and adapted from CrunchDAO Synth for production crypto trading.

Provides:
- Variance-Time log-log regression Hurst estimation.
- Classical Rescaled Range (R/S) Hurst estimation.
- AssetHurstCalibrator for dynamic persistence estimation per asset.
"""

import numpy as np
import pandas as pd


def estimate_hurst_variance_time(
    prices: np.ndarray | pd.Series | list[float],
    lags: list[int] | None = None,
) -> float:
    """
    Estimates the Hurst exponent using log-log Variance-Time regression:
        ln(Std(r_tau)) = H * ln(tau) + C
    Returns H in [0.05, 0.95].
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    prices_arr = prices_arr[np.isfinite(prices_arr)]
    if len(prices_arr) < 64:
        return 0.5

    min_val = float(np.min(prices_arr))
    if min_val <= 0:
        prices_arr = prices_arr - min_val + 1.0

    log_prices = np.log(prices_arr)
    lags = lags or [1, 2, 4, 8, 16, 32, 64]
    valid_lags = [lag for lag in lags if lag < len(log_prices) // 4]

    if len(valid_lags) < 3:
        return 0.5

    log_lags: list[float] = []
    log_stds: list[float] = []

    for lag in valid_lags:
        rets = log_prices[lag:] - log_prices[:-lag]
        std_ret = float(np.std(rets))
        if std_ret > 1e-12:
            log_lags.append(float(np.log(lag)))
            log_stds.append(float(np.log(std_ret)))

    if len(log_lags) < 3:
        return 0.5

    poly = np.polyfit(log_lags, log_stds, 1)
    hurst_val = float(poly[0])
    return float(np.clip(hurst_val, 0.05, 0.95))


def estimate_hurst_rs(
    series: np.ndarray | pd.Series | list[float],
    min_chunk: int = 16,
    max_chunk: int | None = None,
) -> float:
    """
    Estimates the Hurst exponent via Classical Rescaled Range (R/S) Analysis.
    """
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    num_obs = len(values)
    if num_obs < min_chunk * 4:
        return 0.5

    max_chunk = max_chunk or (num_obs // 2)
    step_scales = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=8).astype(int))

    rs_values: list[float] = []
    scale_sizes: list[float] = []

    for scale in step_scales:
        num_chunks = num_obs // scale
        if num_chunks < 2:
            continue

        rs_chunks: list[float] = []
        for idx in range(num_chunks):
            chunk = values[idx * scale : (idx + 1) * scale]
            chunk_mean = float(np.mean(chunk))
            cumulative_dev = np.cumsum(chunk - chunk_mean)
            chunk_range = float(np.max(cumulative_dev) - np.min(cumulative_dev))
            chunk_std = float(np.std(chunk, ddof=1))
            if chunk_std > 1e-12 and chunk_range > 1e-12:
                rs_chunks.append(chunk_range / chunk_std)

        if rs_chunks:
            scale_sizes.append(float(np.log(scale)))
            rs_values.append(float(np.log(np.mean(rs_chunks))))

    if len(scale_sizes) < 3:
        return 0.5

    poly = np.polyfit(scale_sizes, rs_values, 1)
    return float(np.clip(poly[0], 0.05, 0.95))


class AssetHurstCalibrator:
    """
    Asset-specific Hurst calibration engine.
    Maintains rolling estimates of fractal memory for crypto assets.
    """

    def __init__(self, default_hurst: float = 0.50) -> None:
        self.default_hurst = default_hurst
        self.calibrated_hursts: dict[str, float] = {}

    def calibrate(self, symbol: str, prices: np.ndarray | pd.Series | list[float]) -> float:
        hurst_exponent = estimate_hurst_variance_time(prices)
        self.calibrated_hursts[symbol] = hurst_exponent
        return hurst_exponent

    def get_hurst(self, symbol: str) -> float:
        return self.calibrated_hursts.get(symbol, self.default_hurst)

    def scale_volatility(self, base_vol: float, horizon_ratio: float, symbol: str) -> float:
        """
        Scales base volatility across time horizons using calibrated Hurst exponent:
            sigma(tau) = base_vol * (tau ** H)
        """
        hurst_exponent = self.get_hurst(symbol)
        if horizon_ratio <= 0:
            return base_vol
        return float(base_vol * (horizon_ratio**hurst_exponent))
