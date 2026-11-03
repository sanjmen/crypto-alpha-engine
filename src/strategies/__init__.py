"""
Quantitative Trading Strategies & Alpha Generation Engines.
"""

from src.strategies.cointegrated_pairs import CointegratedPairsStrategy
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.perp_perp_arbitrage import (
    ArbitrageAction,
    CrossExchangeCollateralManager,
    CrossExchangePerpArbitrageEngine,
    PerpPerpSpread,
    TransferEvent,
)
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
    create_illiquid_early_breakout,
    create_illiquid_funding_fade,
    create_illiquid_oversold_bounce,
    create_liquid_early_breakout,
    create_liquid_funding_fade,
    create_liquid_oversold_bounce,
)
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy

__all__ = [
    "ArbitrageAction",
    "CointegratedPairsStrategy",
    "CrossExchangeCollateralManager",
    "CrossExchangePerpArbitrageEngine",
    "CrossSectionalMomentumStrategy",
    "EarlyBreakoutStrategy",
    "FundingCarryArbitrageStrategy",
    "MetaStrategyAllocator",
    "OversoldBounceStrategy",
    "PerpPerpSpread",
    "PumpFundingFadeStrategy",
    "StatArbMeanReversionStrategy",
    "TransferEvent",
    "create_illiquid_early_breakout",
    "create_illiquid_funding_fade",
    "create_illiquid_oversold_bounce",
    "create_liquid_early_breakout",
    "create_liquid_funding_fade",
    "create_liquid_oversold_bounce",
]
