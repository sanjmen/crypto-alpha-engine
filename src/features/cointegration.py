"""
Multi-Asset Cointegration & Ornstein-Uhlenbeck Spread Analysis Engine.
Statistical Arbitrage & Cointegration Testing for Pair Baskets.

Implements:
1. Engle-Granger Two-Step Cointegration Test for synthetic pairs.
2. Johansen Multivariate Cointegration Test (VECM) for multi-asset baskets.
3. Ornstein-Uhlenbeck (OU) Mean-Reversion Speed (theta) and Half-Life Calibration.
4. Sub-Diffusive Fractal Hurst Exponent Filter (H < 0.45) for spread tradeability.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.features.hurst import estimate_hurst_variance_time


@dataclass
class CointegrationResult:
    """Encapsulates the statistical results of a cointegration & mean-reversion test."""

    symbols: list[str]
    method: str  # 'engle_granger' or 'johansen'
    hedge_ratios: dict[str, float]  # Symbol to weight/multiplier
    intercept: float  # Spread constant offset
    test_statistic: float  # t-stat or trace stat
    p_value: float  # Cointegration significance
    critical_value_95: float  # 95% confidence critical value
    is_cointegrated: bool  # Statistically stationary spread
    half_life_bars: float  # OU mean-reversion half life (in bars/hours)
    ou_theta: float  # Mean-reversion velocity
    ou_mu: float  # Equilibrium mean
    hurst_exponent: float  # Anti-persistence metric (H < 0.45)
    is_tradable: bool  # Passed all statistical & fractal filters


class OrnsteinUhlenbeckCalibrator:
    """
    Fits continuous-time Ornstein-Uhlenbeck stochastic process to a spread:
    dS_t = theta * (mu - S_t) * dt + sigma * dW_t

    Discrete-time regression representation:
    S_t - S_{t-1} = a + b * S_{t-1} + eps_t
    where b = e^(-theta * dt) - 1  ==> theta = -ln(1 + b) / dt
    half-life = ln(2) / theta
    """

    @staticmethod
    def calibrate(
        spread: pd.Series | np.ndarray,
        dt: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """
        Calibrates OU parameters via linear regression on spread differentials.

        Returns:
            theta: Mean-reverting speed (> 0 indicates mean reversion).
            mu: Long-term equilibrium mean.
            sigma: Spread diffusion volatility.
            half_life: Time (in bars/dt units) to revert half the deviation.
        """
        spread_arr = np.asarray(spread, dtype=np.float64)
        spread_arr = spread_arr[np.isfinite(spread_arr)]
        num_obs = len(spread_arr)
        if num_obs < 5 or float(np.std(spread_arr)) < 1e-12:
            return 0.0, float(np.mean(spread_arr)) if num_obs > 0 else 0.0, 0.0, float("inf")

        spread_lag = spread_arr[:-1]
        delta_spread = spread_arr[1:] - spread_lag

        poly = np.polyfit(spread_lag, delta_spread, 1)
        slope_b = float(poly[0])
        intercept_a = float(poly[1])

        # Valid continuous-time OU mean-reversion requires -1.0 < slope_b < 0.0
        if slope_b >= 0.0 or slope_b <= -1.0:
            theta = 1e-6
            half_life = float("inf")
            mu = float(np.mean(spread_arr))
        else:
            factor = max(1.0 + slope_b, 1e-6)
            theta = -float(np.log(factor)) / dt
            mu = -intercept_a / slope_b
            half_life = float(np.log(2.0)) / max(theta, 1e-6)

        resids = delta_spread - (intercept_a + slope_b * spread_lag)
        sigma = float(np.std(resids))

        return theta, mu, sigma, half_life


class EngleGrangerTester:
    """
    Two-Step Engle-Granger Cointegration Test for asset pairs.
    Step 1: OLS regression: Y = alpha + beta * X + epsilon
    Step 2: ADF unit-root test on residuals epsilon_t
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        min_half_life: float = 2.0,
        max_half_life: float = 72.0,
        hurst_threshold: float = 0.45,
    ) -> None:
        self.p_value_threshold = p_value_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold

    def test_pair(
        self,
        y_series: pd.Series,
        x_series: pd.Series,
        y_symbol: str = "Y",
        x_symbol: str = "X",
    ) -> CointegrationResult:
        """
        Tests if pair (Y, X) forms a stationary cointegrated spread.
        """
        df = pd.DataFrame({"y": y_series, "x": x_series}).dropna()
        if len(df) < 30:
            return CointegrationResult(
                symbols=[y_symbol, x_symbol],
                method="engle_granger",
                hedge_ratios={y_symbol: 1.0, x_symbol: -1.0},
                intercept=0.0,
                test_statistic=0.0,
                p_value=1.0,
                critical_value_95=-2.86,
                is_cointegrated=False,
                half_life_bars=float("inf"),
                ou_theta=0.0,
                ou_mu=0.0,
                hurst_exponent=0.5,
                is_tradable=False,
            )

        y_vals = df["y"].values
        x_vals = df["x"].values

        t_stat, p_val, crit_vals = coint(y_vals, x_vals)
        crit_95 = float(crit_vals[1])

        poly = np.polyfit(x_vals, y_vals, 1)
        beta = float(poly[0])
        alpha = float(poly[1])

        spread = y_vals - beta * x_vals - alpha
        theta, mu, _, half_life = OrnsteinUhlenbeckCalibrator.calibrate(spread)
        hurst_val = float(estimate_hurst_variance_time(spread))

        is_coint = bool(p_val <= self.p_value_threshold)
        is_mean_rev = bool(hurst_val < self.hurst_threshold)
        is_valid_hl = bool(self.min_half_life <= half_life <= self.max_half_life)
        is_tradable = is_coint and is_mean_rev and is_valid_hl

        return CointegrationResult(
            symbols=[y_symbol, x_symbol],
            method="engle_granger",
            hedge_ratios={y_symbol: 1.0, x_symbol: -beta},
            intercept=alpha,
            test_statistic=float(t_stat),
            p_value=float(p_val),
            critical_value_95=crit_95,
            is_cointegrated=is_coint,
            half_life_bars=float(half_life),
            ou_theta=float(theta),
            ou_mu=float(mu),
            hurst_exponent=float(hurst_val),
            is_tradable=is_tradable,
        )


class JohansenBasketTester:
    """
    Johansen Multivariate Cointegration Test for multi-asset baskets (K >= 2).
    Uses Vector Error Correction Model (VECM) to find the primary cointegrating vector beta.
    Spread: S_t = sum_k (beta_k * P_{k, t})
    """

    def __init__(
        self,
        significance_level: int = 1,  # 0: 90%, 1: 95%, 2: 99%
        min_half_life: float = 2.0,
        max_half_life: float = 72.0,
        hurst_threshold: float = 0.45,
    ) -> None:
        if significance_level not in (0, 1, 2):
            raise ValueError("significance_level must be 0 (90%), 1 (95%), or 2 (99%).")
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.hurst_threshold = hurst_threshold

    @staticmethod
    def _default_untradable_result(symbols: list[str]) -> CointegrationResult:
        k_assets = len(symbols)
        return CointegrationResult(
            symbols=symbols,
            method="johansen",
            hedge_ratios={sym: 1.0 / max(k_assets, 1) for sym in symbols},
            intercept=0.0,
            test_statistic=0.0,
            p_value=1.0,
            critical_value_95=15.0,
            is_cointegrated=False,
            half_life_bars=float("inf"),
            ou_theta=0.0,
            ou_mu=0.0,
            hurst_exponent=0.5,
            is_tradable=False,
        )

    def test_basket(
        self,
        prices_df: pd.DataFrame,
    ) -> CointegrationResult:
        """
        Executes Johansen trace test and extracts principal cointegrating eigenvector.
        """
        clean_df = prices_df.dropna()
        symbols = list(clean_df.columns)
        k_assets = len(symbols)

        if len(clean_df) < 50 or k_assets < 2:
            return self._default_untradable_result(symbols)

        price_matrix = clean_df.values

        try:
            jres = coint_johansen(price_matrix, det_order=0, k_ar_diff=1)
        except (ValueError, np.linalg.LinAlgError):
            return self._default_untradable_result(symbols)

        trace_stat = float(np.real(jres.lr1[0]))
        crit_95 = float(jres.cvt[0, self.significance_level])
        is_coint = bool(trace_stat > crit_95)

        eigenvec = np.real(jres.evec[:, 0])
        norm_factor = eigenvec[0] if abs(eigenvec[0]) > 1e-8 else 1.0
        beta_weights = eigenvec / norm_factor

        hedge_ratios = {symbols[i]: float(beta_weights[i]) for i in range(k_assets)}

        spread = price_matrix @ beta_weights
        alpha = float(np.mean(spread))
        spread_centered = spread - alpha

        theta, mu, _, half_life = OrnsteinUhlenbeckCalibrator.calibrate(spread_centered)
        hurst_val = float(estimate_hurst_variance_time(spread_centered))

        is_mean_rev = bool(hurst_val < self.hurst_threshold)
        is_valid_hl = bool(self.min_half_life <= half_life <= self.max_half_life)
        is_tradable = is_coint and is_mean_rev and is_valid_hl

        p_val_proxy = max(0.001, min(0.99, 0.05 * (crit_95 / max(trace_stat, 1e-6))))

        return CointegrationResult(
            symbols=symbols,
            method="johansen",
            hedge_ratios=hedge_ratios,
            intercept=alpha,
            test_statistic=trace_stat,
            p_value=p_val_proxy,
            critical_value_95=crit_95,
            is_cointegrated=is_coint,
            half_life_bars=float(half_life),
            ou_theta=float(theta),
            ou_mu=float(mu),
            hurst_exponent=float(hurst_val),
            is_tradable=is_tradable,
        )


class SpreadZScoreEngine:
    """
    Computes real-time dynamic rolling z-scores, Bollinger Bands,
    and mean-reversion trading signals for cointegrated pair/basket spreads.
    """

    def __init__(
        self,
        lookback_window: int = 48,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 3.5,
    ) -> None:
        self.lookback_window = lookback_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore

    def compute_spread(
        self,
        prices_df: pd.DataFrame,
        hedge_ratios: dict[str, float],
        intercept: float = 0.0,
    ) -> pd.Series:
        """Constructs synthetic spread time series from prices and hedge ratios."""
        spread = pd.Series(0.0, index=prices_df.index)
        for sym, beta in hedge_ratios.items():
            if sym in prices_df.columns:
                spread += beta * prices_df[sym]
        return spread - intercept

    def compute_rolling_zscore(
        self,
        spread: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes strictly causal rolling mean, rolling standard deviation, and rolling z-score.
        """
        min_p = max(10, self.lookback_window // 3)
        rolling_mean = spread.rolling(self.lookback_window, min_periods=min_p).mean()
        rolling_std = spread.rolling(self.lookback_window, min_periods=min_p).std()
        rolling_std = rolling_std.replace(0.0, np.nan).ffill().fillna(1e-4)

        zscore = (spread - rolling_mean) / rolling_std
        return zscore.fillna(0.0), rolling_mean, rolling_std
