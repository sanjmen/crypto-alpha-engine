"""
Information-Driven Financial Data Structures (Marcos López de Prado).
Implements Dollar Bars, Dollar Imbalance Bars (DIB), and statistical validation routines.
Recovers near-Gaussian, IID properties from tick-level execution data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BarSnapshot:
    """Represents a completed information-driven bar."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    notional: float
    ticks_count: int
    buy_volume: float
    sell_volume: float
    buy_notional: float
    sell_notional: float
    order_flow_imbalance: float


class DollarBarSynthesizer:
    """
    Synthesizes OHLCV Dollar Bars from raw trade ticks.
    Closes a bar whenever cumulative traded notional breaches a constant dollar threshold ($D).
    """

    def __init__(self, dollar_threshold: float = 5_000_000.0, carry_remainder: bool = True):
        self.dollar_threshold = float(dollar_threshold)
        self.carry_remainder = carry_remainder
        self.reset()

    def reset(self) -> None:
        """Resets the accumulator state."""
        self.cum_notional = 0.0
        self.cum_volume = 0.0
        self.cum_buy_vol = 0.0
        self.cum_sell_vol = 0.0
        self.cum_buy_notional = 0.0
        self.cum_sell_notional = 0.0
        self.ticks_count = 0
        self.open_price: Optional[float] = None
        self.high_price = -np.inf
        self.low_price = np.inf
        self.close_price: Optional[float] = None
        self.last_timestamp: Optional[pd.Timestamp] = None

    def process_ticks(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of ticks and returns generated Dollar Bars as a DataFrame.
        Expected columns: price, quantity (or notional), timestamp, tick_sign (or is_buyer_maker).
        """
        if ticks_df.empty:
            return pd.DataFrame()

        bars: List[Dict[str, Any]] = []

        prices = ticks_df["price"].values
        quantities = ticks_df["quantity"].values
        timestamps = ticks_df["timestamp"].values

        if "notional" in ticks_df.columns:
            notionals = ticks_df["notional"].values
        else:
            notionals = prices * quantities

        if "tick_sign" in ticks_df.columns:
            signs = ticks_df["tick_sign"].values
        elif "is_buyer_maker" in ticks_df.columns:
            signs = np.where(ticks_df["is_buyer_maker"].values, -1, 1)
        else:
            signs = np.ones(len(prices))

        n_ticks = len(prices)

        for i in range(n_ticks):
            p = float(prices[i])
            q = float(quantities[i])
            d = float(notionals[i])
            ts = pd.Timestamp(timestamps[i])
            s = int(signs[i])

            if self.open_price is None:
                self.open_price = p
                self.high_price = p
                self.low_price = p

            if p > self.high_price:
                self.high_price = p
            if p < self.low_price:
                self.low_price = p

            self.close_price = p
            self.last_timestamp = ts
            self.cum_notional += d
            self.cum_volume += q
            self.ticks_count += 1

            if s > 0:
                self.cum_buy_vol += q
                self.cum_buy_notional += d
            else:
                self.cum_sell_vol += q
                self.cum_sell_notional += d

            # Check threshold breach
            if self.cum_notional >= self.dollar_threshold:
                tot_vol = self.cum_buy_vol + self.cum_sell_vol
                ofi = (self.cum_buy_vol - self.cum_sell_vol) / max(tot_vol, 1e-8)

                bars.append({
                    "timestamp": self.last_timestamp,
                    "open": self.open_price,
                    "high": self.high_price,
                    "low": self.low_price,
                    "close": self.close_price,
                    "volume": self.cum_volume,
                    "notional": self.cum_notional,
                    "ticks_count": self.ticks_count,
                    "buy_volume": self.cum_buy_vol,
                    "sell_volume": self.cum_sell_vol,
                    "buy_notional": self.cum_buy_notional,
                    "sell_notional": self.cum_sell_notional,
                    "order_flow_imbalance": ofi,
                })

                # Carry remainder or full reset
                if self.carry_remainder:
                    remainder = self.cum_notional - self.dollar_threshold
                    self.reset()
                    self.cum_notional = remainder
                else:
                    self.reset()

        if not bars:
            return pd.DataFrame()

        df_bars = pd.DataFrame(bars)
        df_bars.set_index("timestamp", inplace=True)
        return df_bars


class DollarImbalanceBarSynthesizer:
    """
    Synthesizes Dollar Imbalance Bars (DIB) based on aggressor order flow (Lee-Ready / Binance tick signs).
    Closes a bar when cumulative signed dollar imbalance exceeds an adaptive EWMA threshold.
    """

    def __init__(
        self,
        initial_threshold: float = 1_000_000.0,
        ewma_alpha: float = 0.05,
    ):
        self.initial_threshold = float(initial_threshold)
        self.current_threshold = float(initial_threshold)
        self.ewma_alpha = float(ewma_alpha)
        self.reset()

    def reset(self) -> None:
        """Resets cumulative imbalance counters."""
        self.cum_imbalance = 0.0
        self.cum_notional = 0.0
        self.cum_volume = 0.0
        self.cum_buy_vol = 0.0
        self.cum_sell_vol = 0.0
        self.ticks_count = 0
        self.open_price: Optional[float] = None
        self.high_price = -np.inf
        self.low_price = np.inf
        self.close_price: Optional[float] = None
        self.last_timestamp: Optional[pd.Timestamp] = None

    def process_ticks(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Processes ticks and generates Dollar Imbalance Bars."""
        if ticks_df.empty:
            return pd.DataFrame()

        bars: List[Dict[str, Any]] = []

        prices = ticks_df["price"].values
        quantities = ticks_df["quantity"].values
        timestamps = ticks_df["timestamp"].values

        if "notional" in ticks_df.columns:
            notionals = ticks_df["notional"].values
        else:
            notionals = prices * quantities

        if "tick_sign" in ticks_df.columns:
            signs = ticks_df["tick_sign"].values
        elif "is_buyer_maker" in ticks_df.columns:
            signs = np.where(ticks_df["is_buyer_maker"].values, -1, 1)
        else:
            signs = np.ones(len(prices))

        for i in range(len(prices)):
            p = float(prices[i])
            q = float(quantities[i])
            d = float(notionals[i])
            s = int(signs[i])
            ts = pd.Timestamp(timestamps[i])

            if self.open_price is None:
                self.open_price = p
                self.high_price = p
                self.low_price = p

            if p > self.high_price:
                self.high_price = p
            if p < self.low_price:
                self.low_price = p

            self.close_price = p
            self.last_timestamp = ts
            self.cum_notional += d
            self.cum_volume += q
            self.ticks_count += 1

            # Signed dollar imbalance: theta_t = b_t * P_t * v_t
            signed_dollar = s * d
            self.cum_imbalance += signed_dollar

            if s > 0:
                self.cum_buy_vol += q
            else:
                self.cum_sell_vol += q

            # Check absolute imbalance breach
            if abs(self.cum_imbalance) >= self.current_threshold:
                tot_vol = self.cum_buy_vol + self.cum_sell_vol
                ofi = (self.cum_buy_vol - self.cum_sell_vol) / max(tot_vol, 1e-8)
                imbalance_side = "BUY" if self.cum_imbalance > 0 else "SELL"

                bars.append({
                    "timestamp": self.last_timestamp,
                    "open": self.open_price,
                    "high": self.high_price,
                    "low": self.low_price,
                    "close": self.close_price,
                    "volume": self.cum_volume,
                    "notional": self.cum_notional,
                    "ticks_count": self.ticks_count,
                    "signed_imbalance": self.cum_imbalance,
                    "imbalance_side": imbalance_side,
                    "order_flow_imbalance": ofi,
                    "threshold_at_bar": self.current_threshold,
                })

                # Update adaptive threshold with EWMA
                abs_imb = abs(self.cum_imbalance)
                self.current_threshold = (
                    (1.0 - self.ewma_alpha) * self.current_threshold + self.ewma_alpha * abs_imb
                )
                self.reset()

        if not bars:
            return pd.DataFrame()

        df_bars = pd.DataFrame(bars)
        df_bars.set_index("timestamp", inplace=True)
        return df_bars


class StatisticalValidator:
    """
    Routines to empirically test and benchmark statistical properties of financial bars.
    Evaluates Jarque-Bera normality, kurtosis, skewness, and return autocorrelation.
    """

    @staticmethod
    def compute_log_returns(bars_df: pd.DataFrame) -> pd.Series:
        """Computes continuous log returns from closing prices."""
        closes = bars_df["close"].dropna()
        if len(closes) < 2:
            return pd.Series(dtype=np.float64)
        return np.log(closes / closes.shift(1)).dropna()

    @staticmethod
    def test_normality_jarque_bera(returns: pd.Series) -> Dict[str, float]:
        """
        Computes Jarque-Bera goodness-of-fit test.
        Null hypothesis: Returns are normally distributed with zero skewness and kurtosis=3.
        Lower JB statistic indicates closer adherence to normality.
        """
        clean_rets = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_rets) < 8:
            return {"jb_stat": 0.0, "p_value": 1.0, "skew": 0.0, "excess_kurtosis": 0.0}

        jb_stat, p_val = stats.jarque_bera(clean_rets)
        skew_val = float(stats.skew(clean_rets))
        # excess kurtosis = kurtosis - 3.0 (normal distribution has 0 excess kurtosis)
        kurt_val = float(stats.kurtosis(clean_rets, fisher=True))

        return {
            "jb_stat": float(jb_stat),
            "p_value": float(p_val),
            "skew": skew_val,
            "excess_kurtosis": kurt_val,
            "sample_size": len(clean_rets),
        }

    @staticmethod
    def compute_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
        """Computes lag-k autocorrelation of returns."""
        clean_rets = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_rets) <= lag:
            return 0.0
        return float(clean_rets.autocorr(lag=lag))

    @classmethod
    def compare_time_vs_dollar_bars(
        cls,
        time_bars_df: pd.DataFrame,
        dollar_bars_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Directly compares empirical statistical properties of Time Bars vs Dollar Bars.
        Demonstrates how Dollar Bars crush excess kurtosis and recover Gaussian properties.
        """
        time_rets = cls.compute_log_returns(time_bars_df)
        dollar_rets = cls.compute_log_returns(dollar_bars_df)

        time_stats = cls.test_normality_jarque_bera(time_rets)
        dollar_stats = cls.test_normality_jarque_bera(dollar_rets)

        time_stats["autocorr_lag1"] = cls.compute_autocorrelation(time_rets, lag=1)
        dollar_stats["autocorr_lag1"] = cls.compute_autocorrelation(dollar_rets, lag=1)

        time_stats["variance"] = float(time_rets.var()) if len(time_rets) > 0 else 0.0
        dollar_stats["variance"] = float(dollar_rets.var()) if len(dollar_rets) > 0 else 0.0

        return {
            "Time Bars": time_stats,
            "Dollar Bars": dollar_stats,
        }
