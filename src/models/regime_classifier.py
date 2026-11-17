"""
Dynamic Market Regime Classifier.
Categorizes market dynamics into TRENDING, RANGING, or VOLATILE states.
Ported and adapted from CrunchDAO Synth for crypto regime switching.
"""

import numpy as np
import pandas as pd

from src.domain.entities import Bar
from src.domain.enums import MarketRegime
from src.features.volatility import garman_klass_volatility
from src.ports.model_ports import IRegimeDetector


class MarketRegimeClassifier(IRegimeDetector):
    """
    Classifies market state into TRENDING, RANGING, or VOLATILE regimes
    using rolling volatility z-scores and trend signal-to-noise ratios.
    """

    def __init__(
        self,
        vol_z_threshold: float = 1.25,
        trend_snr_threshold: float = 0.65,
        rolling_window: int = 24,
        vol_ratio_scale: float = 0.35,
    ) -> None:
        self.vol_z_threshold = vol_z_threshold
        self.trend_snr_threshold = trend_snr_threshold
        self.rolling_window = rolling_window
        self.vol_ratio_scale = vol_ratio_scale

    def detect_regime(self, bars: list[Bar]) -> MarketRegime:
        """Implements `IRegimeDetector.detect_regime` over a sequence of `Bar` entities."""
        if not bars:
            return MarketRegime.RANGING
        return self.classify(
            opens=[bar.open for bar in bars],
            highs=[bar.high for bar in bars],
            lows=[bar.low for bar in bars],
            closes=[bar.close for bar in bars],
        )

    def classify(
        self,
        opens: np.ndarray | pd.Series | list[float],
        highs: np.ndarray | pd.Series | list[float],
        lows: np.ndarray | pd.Series | list[float],
        closes: np.ndarray | pd.Series | list[float],
    ) -> MarketRegime:
        open_arr = np.asarray(opens, dtype=np.float64)
        high_arr = np.asarray(highs, dtype=np.float64)
        low_arr = np.asarray(lows, dtype=np.float64)
        close_arr = np.asarray(closes, dtype=np.float64)

        num_bars = len(close_arr)
        if num_bars < 12:
            return MarketRegime.RANGING

        recent_n = min(num_bars, self.rolling_window)
        sub_close = close_arr[-recent_n:]
        sub_open = open_arr[-recent_n:]
        sub_high = high_arr[-recent_n:]
        sub_low = low_arr[-recent_n:]

        curr_gk = garman_klass_volatility(sub_open[-6:], sub_high[-6:], sub_low[-6:], sub_close[-6:])
        hist_gk = (
            garman_klass_volatility(sub_open[:-6], sub_high[:-6], sub_low[:-6], sub_close[:-6])
            if recent_n > 12
            else curr_gk
        )

        if hist_gk > 1e-8:
            vol_ratio = curr_gk / hist_gk
            vol_zscore = (vol_ratio - 1.0) / max(self.vol_ratio_scale, 1e-6)
        else:
            vol_zscore = 0.0

        time_idx = np.arange(recent_n, dtype=np.float64)
        poly = np.polyfit(time_idx, sub_close, 1)
        slope = float(poly[0])
        drift_norm = abs(slope * recent_n) / max(float(sub_close[0]), 1e-6)
        snr = drift_norm / max(curr_gk, 1e-4)

        if vol_zscore > self.vol_z_threshold:
            return MarketRegime.VOLATILE
        if snr > self.trend_snr_threshold:
            return MarketRegime.TRENDING
        return MarketRegime.RANGING
