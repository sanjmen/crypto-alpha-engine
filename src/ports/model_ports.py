"""
Port interfaces for Feature Engineering and Alpha Modeling.
"""

from abc import abstractmethod
from typing import Dict, List, Protocol, runtime_checkable
import pandas as pd

from src.domain.entities import Bar, Signal, Portfolio
from src.domain.enums import MarketRegime


@runtime_checkable
class IFeatureExtractor(Protocol):
    """
    Interface for quantitative feature extractors.
    """

    @abstractmethod
    def extract(self, bars: List[Bar]) -> pd.DataFrame:
        """Extract feature matrix from a sequence of OHLCV bars."""
        ...


@runtime_checkable
class IAlphaModel(Protocol):
    """
    Interface for alpha generation models.
    """

    @abstractmethod
    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Fit model on historical features and future returns."""
        ...

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> List[Signal]:
        """Generate normalized alpha signals (-1.0 to 1.0)."""
        ...


@runtime_checkable
class IRegimeDetector(Protocol):
    """
    Interface for market regime classification.
    """

    @abstractmethod
    def detect_regime(self, bars: List[Bar]) -> MarketRegime:
        """Classify prevailing market dynamics."""
        ...


@runtime_checkable
class IRiskManager(Protocol):
    """
    Interface for portfolio risk sizing and constraint enforcement.
    """

    @abstractmethod
    def size_positions(
        self,
        signals: List[Signal],
        portfolio: Portfolio,
        current_volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute target dollar or fractional weights per asset."""
        ...


@runtime_checkable
class IStrategy(Protocol):
    """
    Interface for quantitative trading strategies.
    """

    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Signal]:
        """Generate target alpha signals across assets."""
        ...
