"""
Dynamic Market Regime Classifier.
Categorizes market dynamics into TRENDING, RANGING, or VOLATILE states.
Ported and adapted from CrunchDAO Synth for crypto regime switching.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.domain.enums import MarketRegime
from src.features.volatility import garman_klass_volatility, parkinson_volatility


class MarketRegimeClassifier:
    """
    Classifies market state into TRENDING, RANGING, or VOLATILE regimes
    using rolling volatility z-scores and trend signal-to-noise ratios.
    """

    def __init__(
        self,
        vol_z_threshold: float = 1.25,
        trend_snr_threshold: float = 0.65,
        rolling_window: int = 24,
    ):
        self.vol_z_threshold = vol_z_threshold
        self.trend_snr_threshold = trend_snr_threshold
        self.rolling_window = rolling_window

    def classify(
        self,
        opens: Union[np.ndarray, pd.Series, List[float]],
        highs: Union[np.ndarray, pd.Series, List[float]],
        lows: Union[np.ndarray, pd.Series, List[float]],
        closes: Union[np.ndarray, pd.Series, List[float]],
    ) -> MarketRegime:
        o = np.asarray(opens, dtype=np.float64)
        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)

        n = len(c)
        if n < 12:
            return MarketRegime.RANGING

        recent_n = min(n, self.rolling_window)
        sub_c = c[-recent_n:]
        sub_o = o[-recent_n:]
        sub_h = h[-recent_n:]
        sub_l = l[-recent_n:]

        # 1. Volatility estimation: compare recent 6 bars vs prior history
        curr_gk = garman_klass_volatility(sub_o[-6:], sub_h[-6:], sub_l[-6:], sub_c[-6:])
        hist_gk = garman_klass_volatility(sub_o[:-6], sub_h[:-6], sub_l[:-6], sub_c[:-6]) if recent_n > 12 else curr_gk

        if hist_gk > 1e-8:
            vol_ratio = curr_gk / hist_gk
            vol_zscore = (vol_ratio - 1.0) / 0.35
        else:
            vol_zscore = 0.0

        # 2. Trend SNR: linear slope normalized by volatility
        x = np.arange(recent_n, dtype=np.float64)
        poly = np.polyfit(x, sub_c, 1)
        slope = poly[0]
        drift_norm = abs(slope * recent_n) / max(sub_c[0], 1e-6)
        snr = drift_norm / max(curr_gk, 1e-4)

        # 3. Decision Boundary
        if vol_zscore > self.vol_z_threshold:
            return MarketRegime.VOLATILE
        elif snr > self.trend_snr_threshold:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.RANGING
