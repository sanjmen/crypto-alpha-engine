"""
Port interfaces for Feature Engineering and Alpha Modeling.
"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from src.domain.entities import Bar, Portfolio, Signal
from src.domain.enums import MarketRegime


@runtime_checkable
class IFeatureExtractor(Protocol):
    """
    Interface for quantitative feature extractors.
    """

    def extract(self, bars: list[Bar]) -> pd.DataFrame:
        """Extract feature matrix from a sequence of OHLCV bars."""
        ...


@runtime_checkable
class IAlphaModel(Protocol):
    """
    Interface for alpha generation models.
    """

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> Any:
        """Fit model on historical features and future returns."""
        ...

    def predict(self, features: pd.DataFrame) -> list[Signal]:
        """Generate normalized alpha signals (-1.0 to 1.0)."""
        ...


@runtime_checkable
class IRegimeDetector(Protocol):
    """
    Interface for market regime classification.
    """

    def detect_regime(self, bars: list[Bar]) -> MarketRegime:
        """Classify prevailing market dynamics."""
        ...


@runtime_checkable
class IRiskManager(Protocol):
    """
    Interface for portfolio risk sizing and constraint enforcement.
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio: Portfolio,
        current_volatilities: dict[str, float],
    ) -> dict[str, float]:
        """Compute target dollar or fractional weights per asset."""
        ...


@runtime_checkable
class IStrategy(Protocol):
    """
    Interface for quantitative trading strategies.
    """

    def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """Generate target alpha signals across assets."""
        ...
