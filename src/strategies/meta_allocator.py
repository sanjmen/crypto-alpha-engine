"""
Meta-Strategy Ensemble Allocator.
Dynamically allocates risk capital across Cross-Sectional Momentum,
Statistical Arbitrage Mean-Reversion, and Perpetual Funding Carry strategies
conditioned on the prevailing market regime (Trending, Ranging, Volatile).
Nets overlapping position requests to minimize transaction turnover and fees.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.domain.entities import Signal, Portfolio
from src.domain.enums import MarketRegime
from src.models.regime_classifier import MarketRegimeClassifier
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.ports.model_ports import IStrategy


class MetaStrategyAllocator:
    """
    Institutional multi-strategy portfolio allocator.
    Dynamically balances momentum, mean-reversion, and carry strategies
    based on real-time market regime classification.
    """

    def __init__(
        self,
        momentum_strategy: Optional[CrossSectionalMomentumStrategy] = None,
        stat_arb_strategy: Optional[StatArbMeanReversionStrategy] = None,
        carry_strategy: Optional[FundingCarryArbitrageStrategy] = None,
        regime_classifier: Optional[MarketRegimeClassifier] = None,
        base_weights: Optional[Dict[str, float]] = None,
    ):
        self.momentum_strategy = momentum_strategy or CrossSectionalMomentumStrategy()
        self.stat_arb_strategy = stat_arb_strategy or StatArbMeanReversionStrategy()
        self.carry_strategy = carry_strategy or FundingCarryArbitrageStrategy()
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()

        # Default strategy weight baseline: 40% Momentum, 30% StatArb, 30% Carry
        self.base_weights = base_weights or {
            "momentum": 0.40,
            "stat_arb": 0.30,
            "carry": 0.30,
        }

    def compute_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Adjusts strategy allocation weights conditioned on the detected regime.
        - TRENDING: Boost Momentum, reduce Mean-Reversion.
        - RANGING: Boost Mean-Reversion, reduce Momentum.
        - VOLATILE: De-risk directional strategies, preserve Carry and cash buffer.
        """
        if regime == MarketRegime.TRENDING:
            return {"momentum": 0.60, "stat_arb": 0.10, "carry": 0.30}
        elif regime == MarketRegime.RANGING:
            return {"momentum": 0.15, "stat_arb": 0.55, "carry": 0.30}
        elif regime == MarketRegime.VOLATILE:
            # Scaled down (50% gross target) to manage tail-risk volatility
            return {"momentum": 0.10, "stat_arb": 0.10, "carry": 0.30}
        return self.base_weights

    def detect_market_regime(self, market_data: Dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Detects prevailing market regime from benchmark asset (BTC or ETH).
        """
        benchmark_df = market_data.get("BTCUSDT")
        if benchmark_df is None:
            benchmark_df = market_data.get("ETHUSDT")
        if benchmark_df is None and len(market_data) > 0:
            benchmark_df = next(iter(market_data.values()))

        if benchmark_df is None or len(benchmark_df) < 50:
            return MarketRegime.RANGING

        opens = benchmark_df["open"].values
        highs = benchmark_df["high"].values
        lows = benchmark_df["low"].values
        closes = benchmark_df["close"].values

        regime = self.regime_classifier.classify(opens, highs, lows, closes)
        return regime

    def generate_portfolio_allocations(
        self,
        market_data: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
        volatilities: Dict[str, float],
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[Dict[str, float], MarketRegime, Dict[str, float]]:
        """
        Executes end-to-end multi-strategy signal generation, regime adaptation,
        and position netting.
        Returns:
            net_dollar_positions: Dict[symbol, dollar_allocation]
            detected_regime: MarketRegime
            strategy_weights: Dict[strategy_name, weight]
        """
        # 1. Detect market regime
        regime = self.detect_market_regime(market_data)
        strat_weights = self.compute_regime_weights(regime)

        # 2. Generate sub-strategy signals
        signals_mom = self.momentum_strategy.generate_signals(market_data)
        signals_arb = self.stat_arb_strategy.generate_signals(market_data)
        signals_cry = self.carry_strategy.generate_signals(market_data, funding_data=funding_data)

        # 3. Size positions per strategy
        alloc_mom = self.momentum_strategy.allocate_weights(signals_mom, portfolio, volatilities)
        alloc_arb = self.stat_arb_strategy.allocate_weights(signals_arb, portfolio, volatilities)
        alloc_cry = self.carry_strategy.allocate_weights(signals_cry, portfolio, volatilities)

        # 4. Net positions across strategies: Sum(weight_k * pos_k)
        all_symbols = set(alloc_mom.keys()) | set(alloc_arb.keys()) | set(alloc_cry.keys())
        net_dollar_positions: Dict[str, float] = {}

        w_mom = strat_weights.get("momentum", 0.33)
        w_arb = strat_weights.get("stat_arb", 0.33)
        w_cry = strat_weights.get("carry", 0.33)

        for sym in all_symbols:
            pos_m = alloc_mom.get(sym, 0.0) * w_mom
            pos_a = alloc_arb.get(sym, 0.0) * w_arb
            pos_c = alloc_cry.get(sym, 0.0) * w_cry
            net_pos = pos_m + pos_a + pos_c
            if abs(net_pos) > 1.0:  # Ignore microscopic dust
                net_dollar_positions[sym] = float(net_pos)

        return net_dollar_positions, regime, strat_weights
