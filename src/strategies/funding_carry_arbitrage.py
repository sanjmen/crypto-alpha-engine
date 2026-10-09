"""
Perpetual Funding Rate Carry Arbitrage Strategy.
Harvests perpetual futures funding premia delta-neutrally by pairing
assets with high positive funding (shorting) with high negative funding (longing).
Incorporates GMM jump-risk brake to deleverage during liquidation cascades.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.domain.entities import Signal, Portfolio
from src.models.gmm_regime import GaussianMixtureJumpDetector
from src.risk.vol_targeting import VolatilityTargetingSizer
from src.ports.model_ports import IStrategy


class FundingCarryArbitrageStrategy(IStrategy):
    """
    Delta-neutral perpetual funding rate harvesting strategy.
    Ranks the universe by annualized funding rate:
    - Shorts the highest positive funding rate assets (earns funding payment from longs).
    - Longs the most negative funding rate assets (earns funding payment from shorts).
    Enforces exact dollar neutrality so portfolio market beta is zero.
    """

    def __init__(
        self,
        min_annualized_carry: float = 0.10,   # Minimum 10% annualized carry to open
        target_annual_vol: float = 0.15,      # 15% target portfolio volatility
        max_leverage: float = 2.0,            # Max allowable gross leverage
        jump_safety_threshold: float = 0.65,  # GMM jump risk brake threshold
    ):
        self.min_annualized_carry = min_annualized_carry
        self.jump_safety_threshold = jump_safety_threshold
        self.regime_detector = GaussianMixtureJumpDetector()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Signal]:
        """
        Generates delta-neutral carry signals based on funding rate differentials.
        funding_data: Dict mapping symbol -> DataFrame with ['funding_rate'].
        If funding_data is not passed, checks if 'funding_rate' column exists in market_data.
        """
        signals: List[Signal] = []
        now = datetime.now(timezone.utc)
        rates: Dict[str, float] = {}

        # 1. Extract annualized funding rates
        for sym, df in market_data.items():
            f_series = None
            if funding_data and sym in funding_data:
                f_df = funding_data[sym]
                if "funding_rate" in f_df.columns and len(f_df) > 0:
                    f_series = f_df["funding_rate"]
            elif "funding_rate" in df.columns and len(df) > 0:
                f_series = df["funding_rate"]

            if f_series is not None and len(f_series) > 0:
                # Use rolling average of last 3 to 9 settlements (1 to 3 days)
                window = min(len(f_series), 9)
                mean_f = float(f_series.iloc[-window:].mean())
                ann_rate = mean_f * 3.0 * 365.0  # Annualized
                rates[sym] = ann_rate

        if len(rates) < 2:
            return []

        # 2. Check for market-wide jump regime on benchmark (BTC if present)
        btc_data = market_data.get("BTCUSDT", next(iter(market_data.values())))
        if len(btc_data) >= 30:
            btc_rets = np.diff(np.log(btc_data["close"].values[-30:]))
            self.regime_detector.fit(btc_rets)
            jump_prob = self.regime_detector.predict_jump_probability(float(btc_rets[-1]))
            # If market is experiencing a liquidity crisis / flash crash, disengage
            if jump_prob > self.jump_safety_threshold:
                return []

        # 3. Sort universe by annualized funding rate
        sorted_assets = sorted(rates.items(), key=lambda x: x[1])
        symbols = [x[0] for x in sorted_assets]
        raw_rates = np.array([x[1] for x in sorted_assets], dtype=np.float64)

        # High positive funding -> Short perpetual (receive payments) -> Negative score
        # High negative funding -> Long perpetual (receive payments) -> Positive score
        # Raw carry alpha is -1 * funding rate
        raw_alpha = -raw_rates

        # Filter out assets with carry below threshold
        active_mask = np.abs(raw_rates) >= self.min_annualized_carry
        if not np.any(active_mask):
            return []

        raw_alpha[~active_mask] = 0.0

        # Enforce dollar neutrality: sum of scores = 0
        mean_score = np.mean(raw_alpha[active_mask])
        raw_alpha[active_mask] = raw_alpha[active_mask] - mean_score

        max_val = np.max(np.abs(raw_alpha)) or 1.0
        normalized = raw_alpha / max_val

        for sym, val in zip(symbols, normalized):
            if abs(val) > 1e-4:
                signals.append(Signal(
                    timestamp=now,
                    symbol=sym,
                    score=float(val),
                    confidence=float(min(1.0, abs(val))),
                    horizon_minutes=480,  # 8-hour funding horizon
                ))

        return signals

    def allocate_weights(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculates dollar allocations with inverse-volatility sizing and zero net beta.
        """
        fractional_weights = self.risk_manager.size_positions(
            signals, volatilities, market_neutral=True
        )
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
