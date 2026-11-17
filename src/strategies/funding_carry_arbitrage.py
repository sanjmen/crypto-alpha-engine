"""
Perpetual Funding Rate Carry Arbitrage Strategy.
Harvests perpetual futures funding premia delta-neutrally by pairing
assets with high positive funding (shorting) with high negative funding (longing).
Incorporates GMM jump-risk brake to deleverage during liquidation cascades.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.entities import Portfolio, Signal
from src.models.gmm_regime import GaussianMixtureJumpDetector
from src.ports.model_ports import IStrategy
from src.risk.vol_targeting import VolatilityTargetingSizer


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
        min_annualized_carry: float = 0.10,  # Minimum 10% annualized carry to open
        target_annual_vol: float = 0.15,  # 15% target portfolio volatility
        max_leverage: float = 2.0,  # Max allowable gross leverage
        jump_safety_threshold: float = 0.65,  # GMM jump risk brake threshold
    ) -> None:
        self.min_annualized_carry = min_annualized_carry
        self.jump_safety_threshold = jump_safety_threshold
        self.regime_detector = GaussianMixtureJumpDetector()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        funding_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[Signal]:
        """
        Generates delta-neutral carry signals based on funding rate differentials.
        """
        signals: list[Signal] = []
        now = datetime.now(timezone.utc)
        rates: dict[str, float] = {}

        for symbol, bars_df in market_data.items():
            funding_series = None
            if funding_data and symbol in funding_data:
                funding_df = funding_data[symbol]
                if "funding_rate" in funding_df.columns and len(funding_df) > 0:
                    funding_series = funding_df["funding_rate"]
            elif "funding_rate" in bars_df.columns and len(bars_df) > 0:
                funding_series = bars_df["funding_rate"]

            if funding_series is not None and len(funding_series) > 0:
                window = min(len(funding_series), 9)
                mean_funding_rate = float(funding_series.iloc[-window:].mean())
                annualized_rate = mean_funding_rate * 3.0 * 365.0
                rates[symbol] = annualized_rate

        if len(rates) < 2:
            return []

        btc_data: pd.DataFrame | None = None
        for key in ("BTCUSDT", "BTC/USDT", "BTC/USDT:USDT"):
            if key in market_data and market_data[key] is not None:
                btc_data = market_data[key]
                break
        if btc_data is None:
            btc_data = next(iter(market_data.values()))
        if len(btc_data) >= 30:
            btc_rets = np.diff(np.log(btc_data["close"].values[-30:]))
            self.regime_detector.fit(btc_rets)
            jump_prob = self.regime_detector.predict_jump_probability(float(btc_rets[-1]))
            if jump_prob > self.jump_safety_threshold:
                return []

        sorted_assets = sorted(rates.items(), key=lambda item: item[1])
        symbols = [item[0] for item in sorted_assets]
        raw_rates = np.array([item[1] for item in sorted_assets], dtype=np.float64)

        raw_alpha = -raw_rates

        active_mask = np.abs(raw_rates) >= self.min_annualized_carry
        if not np.any(active_mask):
            return []

        raw_alpha[~active_mask] = 0.0

        if int(np.sum(active_mask)) >= 2:
            mean_score = float(np.mean(raw_alpha[active_mask]))
            raw_alpha[active_mask] = raw_alpha[active_mask] - mean_score

        raw_max_val = float(np.max(np.abs(raw_alpha)))
        max_val = raw_max_val if (np.isfinite(raw_max_val) and raw_max_val > 1e-12) else 1.0
        normalized = raw_alpha / max_val

        for symbol, val in zip(symbols, normalized, strict=False):
            if abs(val) > 1e-4:
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=symbol,
                        score=float(val),
                        confidence=float(min(1.0, abs(val))),
                        horizon_minutes=480,
                    )
                )

        return signals

    def allocate_weights(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculates dollar allocations with inverse-volatility sizing and zero net beta.
        """
        fractional_weights = self.risk_manager.size_positions(signals, volatilities, market_neutral=True)
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
