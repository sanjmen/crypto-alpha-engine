"""
Multi-Asset Cointegrated Pair & Basket Statistical Arbitrage Strategy.
Issue #22 (Milestone M7): Multi-Asset Cointegration & Sector Stat-Arb.

Combines:
1. Sector Taxonomy & Correlation Clustering.
2. Engle-Granger & Johansen Cointegration Stationarity Testing.
3. Ornstein-Uhlenbeck Half-Life and Mean-Reversion Velocity (theta).
4. Sub-Diffusive Hurst Exponent Filtering (H < 0.45).
5. Dynamic Bollinger z-score spread trading with dollar-neutral paired execution.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.entities import Portfolio, Signal
from src.features.cointegration import (
    CointegrationResult,
    EngleGrangerTester,
    JohansenBasketTester,
    SpreadZScoreEngine,
)
from src.features.sector_clustering import SectorTaxonomy
from src.ports.model_ports import IStrategy
from src.risk.vol_targeting import VolatilityTargetingSizer


class CointegratedPairsStrategy(IStrategy):
    """
    Quantitative Statistical Arbitrage strategy trading cointegrated intra-sector pairs.
    Fades stationary spread deviations when verified by sub-diffusive fractal dynamics (H < 0.45).
    """

    def __init__(
        self,
        lookback_bars: int = 240,           # 10 days of 1h bars for cointegration fitting
        zscore_window: int = 48,             # 48h rolling spread z-score
        entry_zscore: float = 2.0,           # Entry threshold (2 standard deviations)
        exit_zscore: float = 0.5,            # Take-profit equilibrium exit
        stop_loss_zscore: float = 3.5,       # Structural breakdown stop loss
        p_value_threshold: float = 0.05,     # Cointegration significance
        hurst_threshold: float = 0.45,       # Mean-reverting requirement
        min_half_life: float = 2.0,          # Minimum half-life in hours
        max_half_life: float = 72.0,         # Maximum half-life in hours
        max_active_pairs: int = 10,          # Maximum concurrent cointegrated pairs
        target_annual_vol: float = 0.20,
        max_leverage: float = 1.5,
    ):
        self.lookback_bars = lookback_bars
        self.zscore_window = zscore_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.max_active_pairs = max_active_pairs

        self.eg_tester = EngleGrangerTester(
            p_value_threshold=p_value_threshold,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            hurst_threshold=hurst_threshold,
        )
        self.johansen_tester = JohansenBasketTester(
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            hurst_threshold=hurst_threshold,
        )
        self.zscore_engine = SpreadZScoreEngine(
            lookback_window=zscore_window,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            stop_loss_zscore=stop_loss_zscore,
        )
        self.taxonomy = SectorTaxonomy()
        self.risk_manager = VolatilityTargetingSizer(
            target_annual_vol=target_annual_vol,
            max_leverage=max_leverage,
        )

        self.calibrated_pairs_: dict[tuple[str, str], CointegrationResult] = {}

    def screen_sector_pairs(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[CointegrationResult]:
        """
        Screens intra-sector candidate pairs across the provided market data.
        Returns a list of tradable, cointegrated pairs sorted by lowest p-value.
        """
        available_symbols = [s for s, df in market_data.items() if len(df) >= self.lookback_bars]
        sector_map = self.taxonomy.filter_universe_by_sector(available_symbols)

        tradable_results: list[CointegrationResult] = []

        for symbols in sector_map.values():
            if len(symbols) < 2:
                continue

            # Screen intra-sector pairs
            for i in range(len(symbols)):
                for j in range(i + 1, min(len(symbols), i + 6)):  # Limit intra-sector combinations
                    s1 = symbols[i]
                    s2 = symbols[j]

                    c1 = market_data[s1]["close"].iloc[-self.lookback_bars:]
                    c2 = market_data[s2]["close"].iloc[-self.lookback_bars:]

                    res = self.eg_tester.test_pair(c1, c2, y_symbol=s1, x_symbol=s2)
                    if res.is_tradable:
                        tradable_results.append(res)
                        self.calibrated_pairs_[(s1, s2)] = res

        # Sort by most statistically significant cointegration (lowest p-value)
        tradable_results.sort(key=lambda r: (r.p_value, -r.ou_theta))
        return tradable_results[:self.max_active_pairs]

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        Generates paired statistical arbitrage signals from screened cointegrated spreads.
        """
        now = datetime.now(timezone.utc)
        # 1. Screen / refresh tradable pairs if needed
        active_pairs = self.screen_sector_pairs(market_data)
        if not active_pairs:
            return []

        raw_signals_by_symbol: dict[str, float] = {}

        for pair_res in active_pairs:
            s1, s2 = pair_res.symbols[0], pair_res.symbols[1]
            df1 = market_data[s1]
            df2 = market_data[s2]

            prices_df = pd.DataFrame({
                s1: df1["close"],
                s2: df2["close"],
            }).dropna().iloc[-self.lookback_bars:]

            if len(prices_df) < self.zscore_window:
                continue

            # 2. Compute synthetic spread
            spread = self.zscore_engine.compute_spread(
                prices_df,
                pair_res.hedge_ratios,
                intercept=pair_res.intercept,
            )

            # 3. Compute rolling z-score
            zscores, _, _ = self.zscore_engine.compute_rolling_zscore(spread)
            current_z = float(zscores.iloc[-1])

            # 4. Generate fading signals
            # Hedge ratio: beta of X against Y
            beta_x = abs(pair_res.hedge_ratios.get(s2, 1.0))
            # Normalize pair relative weight
            norm_y = 1.0 / (1.0 + beta_x)
            norm_x = beta_x / (1.0 + beta_x)

            # Check stop loss
            if abs(current_z) >= self.stop_loss_zscore:
                # Structural regime break: close pair immediately
                raw_signals_by_symbol[s1] = 0.0
                raw_signals_by_symbol[s2] = 0.0
                continue

            # Fading logic
            if current_z >= self.entry_zscore:
                # Spread is overvalued: Short S1, Long S2
                conviction = min(1.0, (current_z - self.entry_zscore) / 1.5 + 0.5)
                raw_signals_by_symbol[s1] = raw_signals_by_symbol.get(s1, 0.0) - conviction * norm_y
                raw_signals_by_symbol[s2] = raw_signals_by_symbol.get(s2, 0.0) + conviction * norm_x

            elif current_z <= -self.entry_zscore:
                # Spread is undervalued: Long S1, Short S2
                conviction = min(1.0, (abs(current_z) - self.entry_zscore) / 1.5 + 0.5)
                raw_signals_by_symbol[s1] = raw_signals_by_symbol.get(s1, 0.0) + conviction * norm_y
                raw_signals_by_symbol[s2] = raw_signals_by_symbol.get(s2, 0.0) - conviction * norm_x

            elif abs(current_z) <= self.exit_zscore:
                # Mean-reversion achieved: neutral
                raw_signals_by_symbol[s1] = 0.0
                raw_signals_by_symbol[s2] = 0.0

        if not raw_signals_by_symbol:
            return []

        # Enforce cross-sectional clipping to [-1.0, 1.0]
        max_abs = max((abs(v) for v in raw_signals_by_symbol.values()), default=1.0) or 1.0
        signals = []
        for sym, score in raw_signals_by_symbol.items():
            norm_score = float(np.clip(score / max_abs, -1.0, 1.0))
            if abs(norm_score) > 0.01:
                signals.append(Signal(
                    timestamp=now,
                    symbol=sym,
                    score=norm_score,
                    confidence=float(min(1.0, abs(norm_score))),
                    horizon_minutes=240,
                ))

        return signals

    def allocate_weights(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculates dollar allocations with dollar-neutral volatility targeting.
        """
        fractional_weights = self.risk_manager.size_positions(
            signals, volatilities, market_neutral=True
        )
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
