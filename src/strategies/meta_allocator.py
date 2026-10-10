"""
Meta-Strategy Ensemble Allocator & Netting Orchestrator.
Dynamically allocates risk capital across Cross-Sectional Momentum,
Statistical Arbitrage Mean-Reversion, Perpetual Funding Carry, and ML Alpha strategies.
Integrates Marcos López de Prado's Hierarchical Risk Parity (HRP, AFML Ch. 16)
to optimize multi-strategy risk budgets without ill-conditioned matrix inversion.
Nets overlapping position requests to minimize transaction turnover and exchange fees.
"""

import pandas as pd

from src.domain.entities import Portfolio
from src.domain.enums import MarketRegime
from src.models.regime_classifier import MarketRegimeClassifier
from src.risk.hrp_allocator import HierarchicalRiskParity
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy


class MetaStrategyAllocator:
    """
    Institutional multi-strategy portfolio allocator & netting orchestrator.
    Dynamically balances momentum, mean-reversion, carry, and ML strategies
    using Hierarchical Risk Parity (HRP) and real-time market regime adaptation.
    """

    def __init__(
        self,
        momentum_strategy: CrossSectionalMomentumStrategy | None = None,
        stat_arb_strategy: StatArbMeanReversionStrategy | None = None,
        carry_strategy: FundingCarryArbitrageStrategy | None = None,
        regime_classifier: MarketRegimeClassifier | None = None,
        hrp_allocator: HierarchicalRiskParity | None = None,
        base_weights: dict[str, float] | None = None,
    ):
        self.momentum_strategy = momentum_strategy or CrossSectionalMomentumStrategy()
        self.stat_arb_strategy = stat_arb_strategy or StatArbMeanReversionStrategy()
        self.carry_strategy = carry_strategy or FundingCarryArbitrageStrategy()
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()
        self.hrp_allocator = hrp_allocator or HierarchicalRiskParity(linkage_method="single")

        # Default strategy weight baseline: 40% Momentum, 30% StatArb, 30% Carry
        self.base_weights = base_weights or {
            "momentum": 0.40,
            "stat_arb": 0.30,
            "carry": 0.30,
        }

    def compute_regime_weights(self, regime: MarketRegime) -> dict[str, float]:
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

    def compute_hrp_strategy_weights(
        self,
        strategy_returns: pd.DataFrame,
        regime: MarketRegime | None = None,
        regime_blend_ratio: float = 0.5,
    ) -> dict[str, float]:
        """
        Allocates risk capital across strategies using Hierarchical Risk Parity (HRP).
        Optionally blends HRP risk-balanced weights with market regime tilts.

        Args:
            strategy_returns: DataFrame where columns are strategy names ('momentum', 'stat_arb', 'carry')
                              and rows are historical periodic returns.
            regime: Optional detected market regime to apply conditional tilt.
            regime_blend_ratio: Weight given to regime tilt (0.0 = pure HRP, 1.0 = pure regime).

        Returns:
            Dict mapping strategy names to normalized capital weights summing to 1.0.
        """
        if strategy_returns.empty or strategy_returns.shape[1] < 2:
            return self.compute_regime_weights(regime) if regime else self.base_weights

        # 1. Compute HRP weights from strategy returns covariance
        hrp_weights_s = self.hrp_allocator.allocate_from_returns(strategy_returns)
        hrp_dict = hrp_weights_s.to_dict()

        if regime is None or regime_blend_ratio <= 0.0:
            return hrp_dict

        # 2. Blend with regime weights
        regime_dict = self.compute_regime_weights(regime)
        blended = {}
        all_keys = set(hrp_dict.keys()) | set(regime_dict.keys())
        for k in all_keys:
            w_hrp = hrp_dict.get(k, 1.0 / len(all_keys))
            w_reg = regime_dict.get(k, 1.0 / len(all_keys))
            blended[k] = (1.0 - regime_blend_ratio) * w_hrp + regime_blend_ratio * w_reg

        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else hrp_dict

    def allocate_asset_weights_hrp(
        self,
        asset_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Directly allocates capital across the asset universe using Hierarchical Risk Parity.
        Bypasses Markowitz quadratic programming matrix inversion.
        """
        if asset_returns.empty:
            return {}
        weights_s = self.hrp_allocator.allocate_from_returns(asset_returns)
        return weights_s.to_dict()

    def detect_market_regime(self, market_data: dict[str, pd.DataFrame]) -> MarketRegime:
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
        market_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        volatilities: dict[str, float],
        funding_data: dict[str, pd.DataFrame] | None = None,
        strategy_returns: pd.DataFrame | None = None,
        use_hrp: bool = False,
    ) -> tuple[dict[str, float], MarketRegime, dict[str, float]]:
        """
        Executes end-to-end multi-strategy signal generation, regime/HRP adaptation,
        and position netting.

        Args:
            market_data: Price data per symbol.
            portfolio: Portfolio entity with current cash and positions.
            volatilities: Current volatility estimate per symbol.
            funding_data: Optional funding rates for carry strategy.
            strategy_returns: Optional historical returns DataFrame to compute HRP weights.
            use_hrp: If True and strategy_returns is provided, uses HRP to balance strategy weights.

        Returns:
            Tuple of (net_dollar_positions, detected_regime, strategy_weights)
        """
        # 1. Detect market regime
        regime = self.detect_market_regime(market_data)

        # 2. Compute strategy weights (HRP or Regime-conditioned)
        if use_hrp and strategy_returns is not None and not strategy_returns.empty:
            strat_weights = self.compute_hrp_strategy_weights(strategy_returns, regime=regime)
        else:
            strat_weights = self.compute_regime_weights(regime)

        # 3. Generate sub-strategy signals
        signals_mom = self.momentum_strategy.generate_signals(market_data)
        signals_arb = self.stat_arb_strategy.generate_signals(market_data)
        signals_cry = self.carry_strategy.generate_signals(market_data, funding_data=funding_data)

        # 4. Size positions per strategy
        alloc_mom = self.momentum_strategy.allocate_weights(signals_mom, portfolio, volatilities)
        alloc_arb = self.stat_arb_strategy.allocate_weights(signals_arb, portfolio, volatilities)
        alloc_cry = self.carry_strategy.allocate_weights(signals_cry, portfolio, volatilities)

        # 5. Net positions across strategies: Sum(weight_k * pos_k)
        all_symbols = set(alloc_mom.keys()) | set(alloc_arb.keys()) | set(alloc_cry.keys())
        net_dollar_positions: dict[str, float] = {}

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
