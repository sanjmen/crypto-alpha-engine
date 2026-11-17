"""
Multi-Asset Cointegrated Pair & Basket Statistical Arbitrage Strategy.

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
        lookback_bars: int = 240,  # 10 days of 1h bars for cointegration fitting
        zscore_window: int = 48,  # 48h rolling spread z-score
        entry_zscore: float = 2.0,  # Entry threshold (2 standard deviations)
        exit_zscore: float = 0.5,  # Take-profit equilibrium exit
        stop_loss_zscore: float = 3.5,  # Structural breakdown stop loss
        p_value_threshold: float = 0.05,  # Cointegration significance
        hurst_threshold: float = 0.45,  # Mean-reverting requirement
        min_half_life: float = 2.0,  # Minimum half-life in hours
        max_half_life: float = 72.0,  # Maximum half-life in hours
        max_active_pairs: int = 10,  # Maximum concurrent cointegrated pairs
        target_annual_vol: float = 0.20,
        max_leverage: float = 1.5,
    ) -> None:
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

            for i in range(len(symbols)):
                for j in range(i + 1, min(len(symbols), i + 6)):
                    sym_y = symbols[i]
                    sym_x = symbols[j]

                    aligned_df = (
                        pd.DataFrame(
                            {
                                sym_y: market_data[sym_y]["close"],
                                sym_x: market_data[sym_x]["close"],
                            }
                        )
                        .dropna()
                        .iloc[-self.lookback_bars :]
                    )
                    if len(aligned_df) < 30:
                        continue

                    res = self.eg_tester.test_pair(
                        aligned_df[sym_y],
                        aligned_df[sym_x],
                        y_symbol=sym_y,
                        x_symbol=sym_x,
                    )
                    if res.is_tradable:
                        tradable_results.append(res)
                        self.calibrated_pairs_[(sym_y, sym_x)] = res

        tradable_results.sort(key=lambda r: (r.p_value, -r.ou_theta))
        return tradable_results[: self.max_active_pairs]

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        Generates paired statistical arbitrage signals from screened cointegrated spreads.
        """
        now = datetime.now(timezone.utc)
        active_pairs = self.screen_sector_pairs(market_data)
        if not active_pairs:
            return []

        raw_signals_by_symbol: dict[str, float] = {}

        for pair_res in active_pairs:
            sym_y, sym_x = pair_res.symbols[0], pair_res.symbols[1]
            df_y = market_data[sym_y]
            df_x = market_data[sym_x]

            prices_df = (
                pd.DataFrame(
                    {
                        sym_y: df_y["close"],
                        sym_x: df_x["close"],
                    }
                )
                .dropna()
                .iloc[-self.lookback_bars :]
            )

            if len(prices_df) < self.zscore_window:
                continue

            spread = self.zscore_engine.compute_spread(
                prices_df,
                pair_res.hedge_ratios,
                intercept=pair_res.intercept,
            )

            zscores, _, _ = self.zscore_engine.compute_rolling_zscore(spread)
            current_z = float(zscores.iloc[-1])

            beta_x = abs(pair_res.hedge_ratios.get(sym_x, 1.0))
            norm_y = 1.0 / (1.0 + beta_x)
            norm_x = beta_x / (1.0 + beta_x)

            if abs(current_z) >= self.stop_loss_zscore:
                raw_signals_by_symbol.setdefault(sym_y, 0.0)
                raw_signals_by_symbol.setdefault(sym_x, 0.0)
                continue

            if current_z >= self.entry_zscore:
                conviction = min(1.0, (current_z - self.entry_zscore) / 1.5 + 0.5)
                raw_signals_by_symbol[sym_y] = raw_signals_by_symbol.get(sym_y, 0.0) - conviction * norm_y
                raw_signals_by_symbol[sym_x] = raw_signals_by_symbol.get(sym_x, 0.0) + conviction * norm_x

            elif current_z <= -self.entry_zscore:
                conviction = min(1.0, (abs(current_z) - self.entry_zscore) / 1.5 + 0.5)
                raw_signals_by_symbol[sym_y] = raw_signals_by_symbol.get(sym_y, 0.0) + conviction * norm_y
                raw_signals_by_symbol[sym_x] = raw_signals_by_symbol.get(sym_x, 0.0) - conviction * norm_x

            elif abs(current_z) <= self.exit_zscore:
                raw_signals_by_symbol.setdefault(sym_y, 0.0)
                raw_signals_by_symbol.setdefault(sym_x, 0.0)

        if not raw_signals_by_symbol:
            return []

        raw_max_abs = max((abs(v) for v in raw_signals_by_symbol.values()), default=1.0)
        max_abs = raw_max_abs if (np.isfinite(raw_max_abs) and raw_max_abs > 1e-8) else 1.0
        signals: list[Signal] = []
        for sym, score in raw_signals_by_symbol.items():
            norm_score = float(np.clip(score / max_abs, -1.0, 1.0))
            if abs(norm_score) > 0.01:
                signals.append(
                    Signal(
                        timestamp=now,
                        symbol=sym,
                        score=norm_score,
                        confidence=float(min(1.0, abs(norm_score))),
                        horizon_minutes=240,
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
        Calculates dollar allocations with dollar-neutral volatility targeting.
        """
        fractional_weights = self.risk_manager.size_positions(signals, volatilities, market_neutral=True)
        return {sym: w * portfolio.total_equity for sym, w in fractional_weights.items()}
