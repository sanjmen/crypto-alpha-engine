"""
Quantitative Strategies & Alpha Allocation Layer for Crypto-Alpha-Engine.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseAnomalyStrategy": ("src.strategies.extreme_anomaly", "BaseAnomalyStrategy"),
    "CausalDollarBarTSMOMEngine": ("src.strategies.dollar_bar_tsmom", "CausalDollarBarTSMOMEngine"),
    "CointegratedPairsStrategy": ("src.strategies.cointegrated_pairs", "CointegratedPairsStrategy"),
    "CrossSectionalMomentumStrategy": (
        "src.strategies.cross_sectional_momentum",
        "CrossSectionalMomentumStrategy",
    ),
    "EarlyBreakoutStrategy": ("src.strategies.extreme_anomaly", "EarlyBreakoutStrategy"),
    "ExtremeFundingHarvesterStrategy": (
        "src.strategies.extreme_anomaly",
        "ExtremeFundingHarvesterStrategy",
    ),
    "FundingCarryArbitrageStrategy": (
        "src.strategies.funding_carry_arbitrage",
        "FundingCarryArbitrageStrategy",
    ),
    "MetaStrategyAllocator": ("src.strategies.meta_allocator", "MetaStrategyAllocator"),
    "OversoldBounceStrategy": ("src.strategies.extreme_anomaly", "OversoldBounceStrategy"),
    "PerpPerpArbitrageStrategy": ("src.strategies.perp_perp_arbitrage", "PerpPerpArbitrageStrategy"),
    "PumpFundingFadeStrategy": ("src.strategies.extreme_anomaly", "PumpFundingFadeStrategy"),
    "SevenFactorAllocationResult": (
        "src.strategies.seven_factor_engine",
        "SevenFactorAllocationResult",
    ),
    "SevenFactorOnlineEngine": ("src.strategies.seven_factor_engine", "SevenFactorOnlineEngine"),
    "StatArbMeanReversionStrategy": (
        "src.strategies.stat_arb_mean_reversion",
        "StatArbMeanReversionStrategy",
    ),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
