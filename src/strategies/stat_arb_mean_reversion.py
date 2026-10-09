"""
Statistical Arbitrage & Mean-Reversion Strategy.
Leverages Hurst Exponent regime estimation and rolling Ornstein-Uhlenbeck z-scores
to trade mean-reverting dynamics while filtering out momentum breakouts and jump regimes.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.domain.entities import Signal, Portfolio
from src.features.hurst import estimate_hurst_variance_time
from src.models.gmm_regime import GaussianMixtureJumpDetector
from src.risk.vol_targeting import VolatilityTargetingSizer
from src.ports.model_ports import IStrategy


class StatArbMeanReversionStrategy(IStrategy):
    """
    Mean-reversion statistical arbitrage strategy.
    Selects assets with low Hurst exponent (H < 0.45) in ranging regimes,
    fades extreme statistical deviations (z-score > 2.0 or < -2.0),
    and closes positions when prices revert toward the equilibrium mean.
    """

    def __init__(
        self,
        lookback_window: int = 48,          # 48-hour rolling equilibrium
        hurst_threshold: float = 0.45,       # Must be mean-reverting (H < 0.45)
        entry_zscore: float = 2.0,           # Entry threshold at 2 standard deviations
        exit_zscore: float = 0.5,            # Take-profit / mean-reversion exit
        target_annual_vol: float = 0.20,
        max_leverage: float = 1.5,
    ):
        self.lookback_window = lookback_window
        self.hurst_threshold = hurst_threshold
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.regime_detector = GaussianMixtureJumpDetector()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """
        Generates mean-reverting alpha signals for assets that satisfy
        the Ornstein-Uhlenbeck sub-diffusive criteria (H < 0.45).
        """
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)

        raw_scores: Dict[str, float] = {}

        for sym, df in market_data.items():
            if len(df) < self.lookback_window:
                continue

            closes = df["close"].values
            window_closes = closes[-self.lookback_window:]

            # 1. Estimate Hurst exponent to verify mean-reversion
            hurst_val = float(estimate_hurst_variance_time(window_closes))

            # If asset is trending (H >= 0.45), mean-reversion is dangerous (skip)
            if hurst_val >= self.hurst_threshold:
                continue

            # 2. Check for jump / high-volatility shock regime via GMM
            log_rets = np.diff(np.log(window_closes))
            if len(log_rets) >= 20:
                self.regime_detector.fit(log_rets)
                jump_prob = self.regime_detector.predict_jump_probability(float(log_rets[-1]))
                if jump_prob > 0.60:
                    # High volatility jump regime in progress, avoid fading
                    continue

            # 3. Compute rolling Ornstein-Uhlenbeck price z-score
            mean_price = np.mean(window_closes)
            std_price = np.std(window_closes) or 1e-8
            current_price = window_closes[-1]
            z_score = float((current_price - mean_price) / std_price)

            # 4. Generate fading signal
            # Price > +2.0 std -> Short (-1.0)
            # Price < -2.0 std -> Long (+1.0)
            if z_score >= self.entry_zscore:
                score = -min(1.0, (z_score - self.entry_zscore) / 2.0 + 0.5)
                raw_scores[sym] = score
            elif z_score <= -self.entry_zscore:
                score = min(1.0, (abs(z_score) - self.entry_zscore) / 2.0 + 0.5)
                raw_scores[sym] = score
            elif abs(z_score) <= self.exit_zscore:
                # In equilibrium zone, neutral
                raw_scores[sym] = 0.0

        if not raw_scores:
            return []

        # Enforce cross-sectional dollar neutrality across active mean-reverting positions
        symbols = list(raw_scores.keys())
        scores_arr = np.array([raw_scores[s] for s in symbols], dtype=np.float64)

        if len(scores_arr) > 1 and np.any(scores_arr != 0):
            scores_arr = scores_arr - np.mean(scores_arr)
            max_abs = np.max(np.abs(scores_arr)) or 1.0
            scores_arr = scores_arr / max_abs

        for sym, val in zip(symbols, scores_arr):
            signals.append(Signal(
                timestamp=now,
                symbol=sym,
                score=float(val),
                confidence=float(min(1.0, abs(val))),
                horizon_minutes=120,
            ))

        return signals

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculates dollar allocations with inverse-volatility sizing.
        """
        fractional_weights = self.risk_manager.size_positions(
            signals, volatilities, market_neutral=True
        )
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
