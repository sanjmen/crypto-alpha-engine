"""
Execution Layer for Crypto-Alpha-Engine.
Provides live exchange adapters, shadow simulation brokers, order routing,
persistence, rate limiting, WebSocket management, and monitoring.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BinanceLiveBroker": ("src.execution.live_broker", "BinanceLiveBroker"),
    "CCXTExecutionAdapter": ("src.execution.live_broker", "CCXTExecutionAdapter"),
    "CircuitBreakerError": ("src.execution.live_broker", "CircuitBreakerError"),
    "DEFAULT_DB_PATH": ("src.execution.database", "DEFAULT_DB_PATH"),
    "DEFAULT_SHADOW_SYMBOLS": ("src.execution.shadow_daemon", "DEFAULT_SHADOW_SYMBOLS"),
    "ExchangeRateLimiter": ("src.execution.rate_limiter", "ExchangeRateLimiter"),
    "ExecutionUrgency": ("src.execution.execution_router", "ExecutionUrgency"),
    "FeatureFlags": ("src.execution.feature_flags", "FeatureFlags"),
    "MarketCapacityExceededError": ("src.execution.live_broker", "MarketCapacityExceededError"),
    "MaxDrawdownStopError": ("src.execution.live_broker", "MaxDrawdownStopError"),
    "MaxLeverageExceededError": ("src.execution.live_broker", "MaxLeverageExceededError"),
    "MaxOrderNotionalExceededError": ("src.execution.live_broker", "MaxOrderNotionalExceededError"),
    "PaperExecutionBroker": ("src.execution.paper_broker", "PaperExecutionBroker"),
    "ResilientWebSocketClient": ("src.execution.ws_manager", "ResilientWebSocketClient"),
    "ShadowDatabase": ("src.execution.database", "ShadowDatabase"),
    "ShadowExecutionBroker": ("src.execution.shadow_broker", "ShadowExecutionBroker"),
    "ShadowMonitor": ("src.execution.monitor", "ShadowMonitor"),
    "ShadowTradingDaemon": ("src.execution.shadow_daemon", "ShadowTradingDaemon"),
    "SmartExecutionRouter": ("src.execution.execution_router", "SmartExecutionRouter"),
    "TradingDaemon": ("src.execution.trading_daemon", "TradingDaemon"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
