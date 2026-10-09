"""
Fractal Hurst Exponent Estimation and Anomalous Temporal Diffusion Scaling.
Ported and adapted from CrunchDAO Synth for production crypto trading.

Provides:
- Variance-Time log-log regression Hurst estimation.
- Classical Rescaled Range (R/S) Hurst estimation.
- AssetHurstCalibrator for dynamic persistence estimation per asset.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def estimate_hurst_variance_time(
    prices: Union[np.ndarray, pd.Series, List[float]],
    lags: Optional[List[int]] = None,
) -> float:
    """
    Estimates the Hurst exponent using log-log Variance-Time regression:
        ln(Std(r_tau)) = H * ln(tau) + C
    Returns H in [0.05, 0.95].
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 64:
        return 0.5

    log_p = np.log(p)
    lags = lags or [1, 2, 4, 8, 16, 32, 64]
    valid_lags = [lag for lag in lags if lag < len(log_p) // 4]

    if len(valid_lags) < 3:
        return 0.5

    log_lags = []
    log_stds = []

    for lag in valid_lags:
        rets = log_p[lag:] - log_p[:-lag]
        std_ret = np.std(rets)
        if std_ret > 1e-12:
            log_lags.append(np.log(lag))
            log_stds.append(np.log(std_ret))

    if len(log_lags) < 3:
        return 0.5

    poly = np.polyfit(log_lags, log_stds, 1)
    hurst_val = float(poly[0])
    return float(np.clip(hurst_val, 0.05, 0.95))


def estimate_hurst_rs(
    series: Union[np.ndarray, pd.Series, List[float]],
    min_chunk: int = 16,
    max_chunk: Optional[int] = None,
) -> float:
    """
    Estimates the Hurst exponent via Classical Rescaled Range (R/S) Analysis.
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    if n < min_chunk * 4:
        return 0.5

    max_chunk = max_chunk or (n // 2)
    step_scales = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=8).astype(int))

    rs_values = []
    scale_sizes = []

    for scale in step_scales:
        num_chunks = n // scale
        if num_chunks < 2:
            continue

        rs_chunks = []
        for i in range(num_chunks):
            chunk = x[i * scale : (i + 1) * scale]
            m = np.mean(chunk)
            z = np.cumsum(chunk - m)
            r = np.max(z) - np.min(z)
            s = np.std(chunk, ddof=1)
            if s > 1e-12 and r > 1e-12:
                rs_chunks.append(r / s)

        if rs_chunks:
            scale_sizes.append(np.log(scale))
            rs_values.append(np.log(np.mean(rs_chunks)))

    if len(scale_sizes) < 3:
        return 0.5

    poly = np.polyfit(scale_sizes, rs_values, 1)
    return float(np.clip(poly[0], 0.05, 0.95))


class AssetHurstCalibrator:
    """
    Asset-specific Hurst calibration engine.
    Maintains rolling estimates of fractal memory for crypto assets.
    """

    def __init__(self, default_hurst: float = 0.50):
        self.default_hurst = default_hurst
        self.calibrated_hursts: Dict[str, float] = {}

    def calibrate(self, symbol: str, prices: Union[np.ndarray, pd.Series, List[float]]) -> float:
        h = estimate_hurst_variance_time(prices)
        self.calibrated_hursts[symbol] = h
        return h

    def get_hurst(self, symbol: str) -> float:
        return self.calibrated_hursts.get(symbol, self.default_hurst)

    def scale_volatility(self, base_vol: float, horizon_ratio: float, symbol: str) -> float:
        """
        Scales base volatility across time horizons using calibrated Hurst exponent:
            sigma(tau) = base_vol * (tau ** H)
        """
        h = self.get_hurst(symbol)
        if horizon_ratio <= 0:
            return base_vol
        return float(base_vol * (horizon_ratio ** h))
