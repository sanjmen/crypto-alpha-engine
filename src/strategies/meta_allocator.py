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
from src.domain.enums import MarketRegime, PositionSide
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
    ) -> None:
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
        """
        if self.pairs_strategy:
            if regime == MarketRegime.TRENDING:
                return {"momentum": 0.50, "stat_arb": 0.10, "carry": 0.20, "pairs": 0.20}
            if regime == MarketRegime.RANGING:
                return {"momentum": 0.15, "stat_arb": 0.25, "carry": 0.25, "pairs": 0.35}
            if regime == MarketRegime.VOLATILE:
                return {"momentum": 0.10, "stat_arb": 0.10, "carry": 0.20, "pairs": 0.10}
            return self.base_weights

        if regime == MarketRegime.TRENDING:
            return {"momentum": 0.60, "stat_arb": 0.10, "carry": 0.30}
        if regime == MarketRegime.RANGING:
            return {"momentum": 0.15, "stat_arb": 0.55, "carry": 0.30}
        if regime == MarketRegime.VOLATILE:
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
        """
        if strategy_returns.empty or strategy_returns.shape[1] < 2:
            return self.compute_regime_weights(regime) if regime else self.base_weights

        hrp_weights_s = self.hrp_allocator.allocate_from_returns(strategy_returns)
        hrp_dict = hrp_weights_s.to_dict()

        if regime is None or regime_blend_ratio <= 0.0:
            return hrp_dict

        regime_dict = self.compute_regime_weights(regime)
        blended: dict[str, float] = {}
        all_keys = set(hrp_dict.keys()) | set(regime_dict.keys())
        for key in all_keys:
            w_hrp = hrp_dict.get(key, 1.0 / len(all_keys))
            w_reg = regime_dict.get(key, 1.0 / len(all_keys))
            blended[key] = (1.0 - regime_blend_ratio) * w_hrp + regime_blend_ratio * w_reg

        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else hrp_dict

    def allocate_asset_weights_hrp(
        self,
        asset_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Directly allocates capital across the asset universe using Hierarchical Risk Parity.
        """
        if asset_returns.empty:
            return {}
        weights_s = self.hrp_allocator.allocate_from_returns(asset_returns)
        return weights_s.to_dict()

    def detect_market_regime(self, market_data: dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Detects prevailing market regime from benchmark asset (BTC or ETH).
        """
        benchmark_df: pd.DataFrame | None = None
        for key in ("BTCUSDT", "BTC/USDT", "BTC/USDT:USDT", "ETHUSDT", "ETH/USDT", "ETH/USDT:USDT"):
            if key in market_data and market_data[key] is not None:
                benchmark_df = market_data[key]
                break
        if benchmark_df is None and len(market_data) > 0:
            benchmark_df = next(iter(market_data.values()))

        if benchmark_df is None or len(benchmark_df) < 50:
            return MarketRegime.RANGING

        return self.regime_classifier.classify(
            benchmark_df["open"].values,
            benchmark_df["high"].values,
            benchmark_df["low"].values,
            benchmark_df["close"].values,
        )

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
        quarterly_basis_data: dict[str, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | float | None = None,
        cooldown_blocked_sides: dict[str, PositionSide] | None = None,
    ) -> tuple[dict[str, float], MarketRegime, dict[str, float]]:
        """
        Runs the 7-Factor Causal Online Alpha Engine + Stepped Leverage Controller (2.5x -> 5.0x)
        and returns (net_dollar_positions, detected_regime, online_factor_weights).
        """
        regime = self.detect_market_regime(market_data)
        engine = self.seven_factor_engine or SevenFactorOnlineEngine()
        leverage_controller = self.stepped_leverage_controller or SteppedLeverageController(
            initial_capital=max(portfolio.total_equity, 1000.0)
        )

        seven_factor_result = engine.allocate_latest(
            bars_1h=market_data,
            funding_rates=funding_data,
            bars_15m=bars_15m,
            open_interest=open_interest_data,
            taker_flow=taker_flow_data,
            premium_index=premium_index_data,
            smart_money=smart_money_data,
            quarterly_basis=quarterly_basis_data,
        )
        self.last_seven_factor_result = seven_factor_result

        effective_unit_weights = dict(seven_factor_result.unit_weights)
        if cooldown_blocked_sides:
            for sym_key, weight_val in list(effective_unit_weights.items()):
                clean_key = sym_key.replace("/", "")
                blocked_side = cooldown_blocked_sides.get(clean_key) or cooldown_blocked_sides.get(sym_key)
                if blocked_side is not None and (
                    (weight_val > 0 and blocked_side == PositionSide.LONG)
                    or (weight_val < 0 and blocked_side == PositionSide.SHORT)
                ):
                    effective_unit_weights[sym_key] = 0.0

        equity = max(float(portfolio.total_equity), 1.0)
        if current_weights is None and portfolio.positions:
            current_weights = {}
            for sym, pos in portfolio.positions.items():
                if pos.side == PositionSide.FLAT or abs(pos.quantity) <= 1e-8:
                    continue
                df_sym = market_data.get(sym)
                if df_sym is not None and not df_sym.empty:
                    price = float(df_sym["close"].iloc[-1])
                    signed_qty = -abs(pos.quantity) if pos.side == PositionSide.SHORT else abs(pos.quantity)
                    current_weights[sym] = (signed_qty * price) / equity

        final_weights, leverage_snapshot = leverage_controller.scale_and_deadband_weights(
            unit_weights=effective_unit_weights,
            current_weights=current_weights,
            current_equity=equity,
            vol_scalar=seven_factor_result.vol_scalar,
        )
        self.last_leverage_snapshot = leverage_snapshot

        top_factor_key = (
            max(seven_factor_result.factor_weights, key=seven_factor_result.factor_weights.get)
            if seven_factor_result.factor_weights
            else "macro"
        )
        top_factor_label = FACTOR_LABELS.get(top_factor_key, f"7Factor:{top_factor_key}")

        net_dollar_positions: dict[str, float] = {}
        for sym, weight in final_weights.items():
            dollar_notional = float(weight * equity)
            if abs(dollar_notional) > 1.0:
                net_dollar_positions[sym] = dollar_notional
                self.last_dominant_strategy[sym] = top_factor_label

        return net_dollar_positions, regime, seven_factor_result.factor_weights

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
        quarterly_basis_data: dict[str, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | float | None = None,
        cooldown_blocked_sides: dict[str, PositionSide] | None = None,
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
                quarterly_basis_data=quarterly_basis_data,
                cooldown_blocked_sides=cooldown_blocked_sides,
            )

        regime = self.detect_market_regime(market_data)

        if use_hrp and strategy_returns is not None and not strategy_returns.empty:
            strat_weights = self.compute_hrp_strategy_weights(strategy_returns, regime=regime)
        else:
            strat_weights = self.compute_regime_weights(regime)

        signals_mom = self.momentum_strategy.generate_signals(market_data)
        signals_arb = self.stat_arb_strategy.generate_signals(market_data)
        signals_cry = self.carry_strategy.generate_signals(market_data, funding_data=funding_data)
        signals_prs = self.pairs_strategy.generate_signals(market_data) if self.pairs_strategy else []
        signals_bnc = (
            self.oversold_bounce_strategy.generate_signals(market_data) if self.oversold_bounce_strategy else []
        )
        signals_brk = self.early_breakout_strategy.generate_signals(market_data) if self.early_breakout_strategy else []
        signals_fad = (
            self.funding_fade_strategy.generate_signals(market_data, funding_data=funding_data)
            if self.funding_fade_strategy
            else []
        )

        alloc_mom = self.momentum_strategy.allocate_weights(signals_mom, portfolio, volatilities)
        alloc_arb = self.stat_arb_strategy.allocate_weights(signals_arb, portfolio, volatilities)
        alloc_cry = self.carry_strategy.allocate_weights(signals_cry, portfolio, volatilities)
        alloc_prs = (
            self.pairs_strategy.allocate_weights(signals_prs, portfolio, volatilities) if self.pairs_strategy else {}
        )
        alloc_bnc = (
            self.oversold_bounce_strategy.allocate_weights(signals_bnc, portfolio, volatilities)
            if self.oversold_bounce_strategy
            else {}
        )
        alloc_brk = (
            self.early_breakout_strategy.allocate_weights(signals_brk, portfolio, volatilities)
            if self.early_breakout_strategy
            else {}
        )
        alloc_fad = (
            self.funding_fade_strategy.allocate_weights(signals_fad, portfolio, volatilities)
            if self.funding_fade_strategy
            else {}
        )

        all_symbols = (
            set(alloc_mom.keys())
            | set(alloc_arb.keys())
            | set(alloc_cry.keys())
            | set(alloc_prs.keys())
            | set(alloc_bnc.keys())
            | set(alloc_brk.keys())
            | set(alloc_fad.keys())
        )
        net_dollar_positions: dict[str, float] = {}

        w_mom = strat_weights.get("momentum", 0.25 if self.pairs_strategy else 0.33)
        w_arb = strat_weights.get("stat_arb", 0.25 if self.pairs_strategy else 0.33)
        w_cry = strat_weights.get("carry", 0.25 if self.pairs_strategy else 0.33)
        w_prs = strat_weights.get("pairs", 0.25) if self.pairs_strategy else 0.0

        for sym in all_symbols:
            pos_mom = alloc_mom.get(sym, 0.0) * w_mom
            pos_arb = alloc_arb.get(sym, 0.0) * w_arb
            pos_cry = alloc_cry.get(sym, 0.0) * w_cry
            pos_prs = alloc_prs.get(sym, 0.0) * w_prs
            pos_bnc = alloc_bnc.get(sym, 0.0)
            pos_brk = alloc_brk.get(sym, 0.0)
            pos_fad = alloc_fad.get(sym, 0.0)
            net_pos = pos_mom + pos_arb + pos_cry + pos_prs + pos_bnc + pos_brk + pos_fad
            if abs(net_pos) > 1.0:
                net_dollar_positions[sym] = float(net_pos)
                contribs = {
                    "Cross-Sectional Momentum": abs(pos_mom),
                    "StatArb Mean-Reversion": abs(pos_arb),
                    "Funding Carry Arbitrage": abs(pos_cry),
                    "Cointegrated Pairs": abs(pos_prs),
                    "Extreme Drop Bounce": abs(pos_bnc),
                    "Early Breakout": abs(pos_brk),
                    "Pump & Funding Fade": abs(pos_fad),
                }
                self.last_dominant_strategy[sym] = max(contribs, key=contribs.get)

        return net_dollar_positions, regime, strat_weights
