"""
Risk Management & Portfolio Sizing Layer for Crypto-Alpha-Engine.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BalanceDiscrepancy": ("src.risk.reconciliation", "BalanceDiscrepancy"),
    "EmergencyKillSwitch": ("src.risk.kill_switch", "EmergencyKillSwitch"),
    "HierarchicalRiskParityAllocator": (
        "src.risk.hrp_allocator",
        "HierarchicalRiskParityAllocator",
    ),
    "KellyPositionSizer": ("src.risk.kelly_sizing", "KellyPositionSizer"),
    "KillSwitchStatus": ("src.risk.kill_switch", "KillSwitchStatus"),
    "OrphanOrder": ("src.risk.reconciliation", "OrphanOrder"),
    "PositionDrift": ("src.risk.reconciliation", "PositionDrift"),
    "PositionReconciliationEngine": ("src.risk.reconciliation", "PositionReconciliationEngine"),
    "ReconciliationReport": ("src.risk.reconciliation", "ReconciliationReport"),
    "ReconciliationStatus": ("src.risk.reconciliation", "ReconciliationStatus"),
    "SteppedLeverageController": ("src.risk.stepped_leverage", "SteppedLeverageController"),
    "SteppedLeverageSnapshot": ("src.risk.stepped_leverage", "SteppedLeverageSnapshot"),
    "VolatilityTargetingSizer": ("src.risk.vol_targeting", "VolatilityTargetingSizer"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
