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
from src.risk.stepped_leverage import LeverageStateSnapshot, SteppedLeverageController
from src.strategies.cointegrated_pairs import CointegratedPairsStrategy
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    ExtremeFundingHarvesterStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
)
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.seven_factor_engine import (
    FACTOR_LABELS,
    SevenFactorAllocationResult,
    SevenFactorOnlineEngine,
)
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy


class MetaStrategyAllocator:
    """
    Institutional multi-strategy portfolio allocator & netting orchestrator.
    Dynamically balances momentum, mean-reversion, carry, and ML strategies
    using Hierarchical Risk Parity (HRP) and real-time market regime adaptation,
    or runs the unified 7-Factor Causal Online Engine with Stepped Leverage (2.5x -> 5.0x).
    """

    def __init__(
        self,
        momentum_strategy: CrossSectionalMomentumStrategy | None = None,
        stat_arb_strategy: StatArbMeanReversionStrategy | None = None,
        carry_strategy: FundingCarryArbitrageStrategy | None = None,
        pairs_strategy: CointegratedPairsStrategy | None = None,
        oversold_bounce_strategy: OversoldBounceStrategy | None = None,
        early_breakout_strategy: EarlyBreakoutStrategy | None = None,
        funding_fade_strategy: PumpFundingFadeStrategy | None = None,
        funding_harvester_strategy: ExtremeFundingHarvesterStrategy | None = None,
        regime_classifier: MarketRegimeClassifier | None = None,
        hrp_allocator: HierarchicalRiskParity | None = None,
        base_weights: dict[str, float] | None = None,
        seven_factor_engine: SevenFactorOnlineEngine | None = None,
        stepped_leverage_controller: SteppedLeverageController | None = None,
        use_seven_factor_engine: bool = False,
    ):
        self.momentum_strategy = momentum_strategy or CrossSectionalMomentumStrategy()
        self.stat_arb_strategy = stat_arb_strategy or StatArbMeanReversionStrategy()
        self.carry_strategy = carry_strategy or FundingCarryArbitrageStrategy()
        self.pairs_strategy = pairs_strategy
        self.oversold_bounce_strategy = oversold_bounce_strategy
        self.early_breakout_strategy = early_breakout_strategy
        self.funding_fade_strategy = funding_fade_strategy
        self.funding_harvester_strategy = funding_harvester_strategy
        self.regime_classifier = regime_classifier or MarketRegimeClassifier()
        self.hrp_allocator = hrp_allocator or HierarchicalRiskParity(linkage_method="single")
        self.seven_factor_engine = seven_factor_engine or (
            SevenFactorOnlineEngine() if use_seven_factor_engine else None
        )
        self.stepped_leverage_controller = stepped_leverage_controller or (
            SteppedLeverageController() if (use_seven_factor_engine or seven_factor_engine) else None
        )
        self.use_seven_factor_engine = bool(use_seven_factor_engine or seven_factor_engine is not None)
        self.last_dominant_strategy: dict[str, str] = {}
        self.last_seven_factor_result: SevenFactorAllocationResult | None = None
        self.last_leverage_snapshot: LeverageStateSnapshot | None = None

        if base_weights:
            self.base_weights = base_weights
        elif pairs_strategy:
            self.base_weights = {"momentum": 0.35, "stat_arb": 0.20, "carry": 0.20, "pairs": 0.25}
        else:
            self.base_weights = {"momentum": 0.40, "stat_arb": 0.30, "carry": 0.30}

    def compute_regime_weights(self, regime: MarketRegime) -> dict[str, float]:
        """
        Adjusts strategy allocation weights conditioned on the detected regime.
        - TRENDING: Boost Momentum, reduce Mean-Reversion.
        - RANGING: Boost Mean-Reversion, reduce Momentum.
        - VOLATILE: De-risk directional strategies, preserve Carry and cash buffer.
        """
        if self.pairs_strategy:
            if regime == MarketRegime.TRENDING:
                return {"momentum": 0.50, "stat_arb": 0.10, "carry": 0.20, "pairs": 0.20}
            elif regime == MarketRegime.RANGING:
                return {"momentum": 0.15, "stat_arb": 0.25, "carry": 0.25, "pairs": 0.35}
            elif regime == MarketRegime.VOLATILE:
                return {"momentum": 0.10, "stat_arb": 0.10, "carry": 0.20, "pairs": 0.10}
            return self.base_weights

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
            benchmark_df = market_data.get("BTC/USDT:USDT")
        if benchmark_df is None:
            benchmark_df = market_data.get("ETHUSDT")
        if benchmark_df is None:
            benchmark_df = market_data.get("ETH/USDT:USDT")
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

    def generate_seven_factor_allocations(
        self,
        market_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        funding_data: dict[str, pd.DataFrame | pd.Series | float] | None = None,
        bars_15m: dict[str, pd.DataFrame] | None = None,
        current_weights: dict[str, float] | None = None,
        open_interest_data: dict[str, pd.DataFrame | pd.Series] | None = None,
        taker_flow_data: dict[str, pd.DataFrame] | None = None,
        premium_index_data: dict[str, pd.DataFrame | pd.Series] | None = None,
        smart_money_data: dict[str, pd.DataFrame | pd.Series] | None = None,
    ) -> tuple[dict[str, float], MarketRegime, dict[str, float]]:
        """
        Runs the 7-Factor Causal Online Alpha Engine + Stepped Leverage Controller (2.5x -> 5.0x)
        and returns (net_dollar_positions, detected_regime, online_factor_weights).
        """
        regime = self.detect_market_regime(market_data)
        engine = self.seven_factor_engine or SevenFactorOnlineEngine()
        lev_ctrl = self.stepped_leverage_controller or SteppedLeverageController(
            initial_capital=max(portfolio.total_equity, 1000.0)
        )

        sf_res = engine.allocate_latest(
            bars_1h=market_data,
            funding_rates=funding_data,
            bars_15m=bars_15m,
            open_interest=open_interest_data,
            taker_flow=taker_flow_data,
            premium_index=premium_index_data,
            smart_money=smart_money_data,
        )
        self.last_seven_factor_result = sf_res

        equity = max(float(portfolio.total_equity), 1.0)
        if current_weights is None and portfolio.positions:
            current_weights = {}
            for sym, pos in portfolio.positions.items():
                df_s = market_data.get(sym)
                if df_s is not None and not df_s.empty:
                    px = float(df_s["close"].iloc[-1])
                    current_weights[sym] = (pos.quantity * px) / equity

        final_weights, lev_snap = lev_ctrl.scale_and_deadband_weights(
            unit_weights=sf_res.unit_weights,
            current_weights=current_weights,
            current_equity=equity,
            vol_scalar=sf_res.vol_scalar,
        )
        self.last_leverage_snapshot = lev_snap

        # Identify dominant active factor for telemetry label
        top_factor_key = (
            max(sf_res.factor_weights, key=sf_res.factor_weights.get)
            if sf_res.factor_weights
            else "macro"
        )
        top_factor_label = FACTOR_LABELS.get(top_factor_key, f"7Factor:{top_factor_key}")

        net_dollar_positions: dict[str, float] = {}
        for sym, w in final_weights.items():
            dollar_notional = float(w * equity)
            if abs(dollar_notional) > 1.0:
                net_dollar_positions[sym] = dollar_notional
                self.last_dominant_strategy[sym] = top_factor_label

        return net_dollar_positions, regime, sf_res.factor_weights

    def generate_portfolio_allocations(
        self,
        market_data: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        volatilities: dict[str, float],
        funding_data: dict[str, pd.DataFrame] | None = None,
        strategy_returns: pd.DataFrame | None = None,
        use_hrp: bool = False,
        bars_15m: dict[str, pd.DataFrame] | None = None,
        current_weights: dict[str, float] | None = None,
        open_interest_data: dict[str, pd.DataFrame | pd.Series] | None = None,
        taker_flow_data: dict[str, pd.DataFrame] | None = None,
        premium_index_data: dict[str, pd.DataFrame | pd.Series] | None = None,
        smart_money_data: dict[str, pd.DataFrame | pd.Series] | None = None,
    ) -> tuple[dict[str, float], MarketRegime, dict[str, float]]:
        """
        Executes end-to-end multi-strategy signal generation, regime/HRP adaptation,
        and position netting. When `use_seven_factor_engine=True`, delegates to the
        7-Factor Causal Online Engine with Stepped Leverage (2.5x -> 5.0x).
        """
        if self.use_seven_factor_engine:
            return self.generate_seven_factor_allocations(
                market_data=market_data,
                portfolio=portfolio,
                funding_data=funding_data,
                bars_15m=bars_15m,
                current_weights=current_weights,
                open_interest_data=open_interest_data,
                taker_flow_data=taker_flow_data,
                premium_index_data=premium_index_data,
                smart_money_data=smart_money_data,
            )

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
        signals_prs = self.pairs_strategy.generate_signals(market_data) if self.pairs_strategy else []
        signals_bnc = self.oversold_bounce_strategy.generate_signals(market_data) if self.oversold_bounce_strategy else []
        signals_brk = self.early_breakout_strategy.generate_signals(market_data) if self.early_breakout_strategy else []
        signals_fad = self.funding_fade_strategy.generate_signals(market_data, funding_data=funding_data) if self.funding_fade_strategy else []

        # 4. Size positions per strategy
        alloc_mom = self.momentum_strategy.allocate_weights(signals_mom, portfolio, volatilities)
        alloc_arb = self.stat_arb_strategy.allocate_weights(signals_arb, portfolio, volatilities)
        alloc_cry = self.carry_strategy.allocate_weights(signals_cry, portfolio, volatilities)
        alloc_prs = self.pairs_strategy.allocate_weights(signals_prs, portfolio, volatilities) if self.pairs_strategy else {}
        alloc_bnc = self.oversold_bounce_strategy.allocate_weights(signals_bnc, portfolio, volatilities) if self.oversold_bounce_strategy else {}
        alloc_brk = self.early_breakout_strategy.allocate_weights(signals_brk, portfolio, volatilities) if self.early_breakout_strategy else {}
        alloc_fad = self.funding_fade_strategy.allocate_weights(signals_fad, portfolio, volatilities) if self.funding_fade_strategy else {}

        # 5. Net positions across strategies: Sum(weight_k * pos_k) + anomaly event overlays
        all_symbols = (
            set(alloc_mom.keys()) | set(alloc_arb.keys()) | set(alloc_cry.keys()) |
            set(alloc_prs.keys()) | set(alloc_bnc.keys()) | set(alloc_brk.keys()) | set(alloc_fad.keys())
        )
        net_dollar_positions: dict[str, float] = {}

        w_mom = strat_weights.get("momentum", 0.25 if self.pairs_strategy else 0.33)
        w_arb = strat_weights.get("stat_arb", 0.25 if self.pairs_strategy else 0.33)
        w_cry = strat_weights.get("carry", 0.25 if self.pairs_strategy else 0.33)
        w_prs = strat_weights.get("pairs", 0.25) if self.pairs_strategy else 0.0

        for sym in all_symbols:
            pos_m = alloc_mom.get(sym, 0.0) * w_mom
            pos_a = alloc_arb.get(sym, 0.0) * w_arb
            pos_c = alloc_cry.get(sym, 0.0) * w_cry
            pos_p = alloc_prs.get(sym, 0.0) * w_prs
            pos_bnc = alloc_bnc.get(sym, 0.0)
            pos_brk = alloc_brk.get(sym, 0.0)
            pos_fad = alloc_fad.get(sym, 0.0)
            net_pos = pos_m + pos_a + pos_c + pos_p + pos_bnc + pos_brk + pos_fad
            if abs(net_pos) > 1.0:  # Ignore microscopic dust
                net_dollar_positions[sym] = float(net_pos)
                contribs = {
                    "Cross-Sectional Momentum": abs(pos_m),
                    "StatArb Mean-Reversion": abs(pos_a),
                    "Funding Carry Arbitrage": abs(pos_c),
                    "Cointegrated Pairs": abs(pos_p),
                    "Extreme Drop Bounce": abs(pos_bnc),
                    "Early Breakout": abs(pos_brk),
                    "Pump & Funding Fade": abs(pos_fad),
                }
                self.last_dominant_strategy[sym] = max(contribs, key=contribs.get)

        return net_dollar_positions, regime, strat_weights
