"""
Institutional Derivatives & Sentiment Features.
Extracts alpha signals from Open Interest velocity, Smart-vs-Retail sentiment divergence,
Taker order flow imbalance, and Annualized Funding Rate dynamics.
"""

from typing import Optional
import numpy as np
import pandas as pd


class DerivativesFeatures:
    """
    Computes alpha indicators from derivatives metrics and order flow.
    Designed for 5-minute to 1-hour aggregated metrics.
    """

    @staticmethod
    def open_interest_velocity(
        oi_series: pd.Series,
        window: int = 12
    ) -> pd.Series:
        """
        Computes rate of change (velocity) of Open Interest over a rolling window.
        delta_OI = (OI_t - OI_{t-k}) / (OI_{t-k} + eps)
        """
        eps = 1e-8
        return (oi_series - oi_series.shift(window)) / (oi_series.shift(window) + eps)

    @staticmethod
    def open_interest_acceleration(
        oi_series: pd.Series,
        window: int = 12
    ) -> pd.Series:
        """
        Computes the second derivative (acceleration) of Open Interest.
        acc_OI = delta_OI_t - delta_OI_{t-k}
        """
        velocity = DerivativesFeatures.open_interest_velocity(oi_series, window=window)
        return velocity - velocity.shift(window)

    @staticmethod
    def smart_money_divergence(
        top_trader_ratio: pd.Series,
        retail_ratio: pd.Series
    ) -> pd.Series:
        """
        Measures positioning divergence between Top Traders (Smart Money) and Retail Accounts:
        Divergence = Top_Trader_L/S_Ratio - Retail_L/S_Ratio
        Positive: Smart Money is net longer than retail (bullish edge).
        Negative: Smart Money is net shorter than retail (bearish edge).
        """
        return top_trader_ratio - retail_ratio

    @staticmethod
    def taker_flow_imbalance(
        taker_long_short_vol_ratio: pd.Series
    ) -> pd.Series:
        """
        Normalizes Taker Buy/Sell volume ratio into bounded [-1.0, 1.0] imbalance:
        Imbalance = (Ratio - 1.0) / (Ratio + 1.0)
        +1: 100% aggressive taker buy pressure.
        -1: 100% aggressive taker sell pressure.
        """
        ratio = taker_long_short_vol_ratio.clip(lower=1e-4)
        return (ratio - 1.0) / (ratio + 1.0)

    @staticmethod
    def annualized_funding_rate(
        funding_rate_8h: pd.Series
    ) -> pd.Series:
        """
        Converts 8-hour funding rate settlements into annualized percentage:
        Annualized = Rate * 3 settlements/day * 365 days
        """
        return funding_rate_8h * 3.0 * 365.0

    @staticmethod
    def compute_all_derivatives_features(
        metrics_df: pd.DataFrame,
        window: int = 12
    ) -> pd.DataFrame:
        """
        Extracts full feature matrix from raw metrics DataFrame.
        """
        df = metrics_df.copy()
        features = pd.DataFrame(index=df.index)

        if "open_interest" in df.columns:
            features["oi_velocity"] = DerivativesFeatures.open_interest_velocity(df["open_interest"], window=window)
            features["oi_accel"] = DerivativesFeatures.open_interest_acceleration(df["open_interest"], window=window)

        if "top_trader_ratio" in df.columns and "retail_ratio" in df.columns:
            features["smart_money_div"] = DerivativesFeatures.smart_money_divergence(
                df["top_trader_ratio"], df["retail_ratio"]
            )

        if "taker_long_short_ratio" in df.columns:
            features["taker_imbalance"] = DerivativesFeatures.taker_flow_imbalance(
                df["taker_long_short_ratio"]
            )

        if "funding_rate" in df.columns:
            features["funding_ann"] = DerivativesFeatures.annualized_funding_rate(
                df["funding_rate"]
            )

        return features
