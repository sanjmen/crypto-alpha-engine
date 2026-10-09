"""
Port interfaces for Data Ingestion and Persistence.
Adheres strictly to the Dependency Inversion Principle (DIP).
"""

from abc import abstractmethod
from datetime import datetime
from typing import List, Optional, Protocol, runtime_checkable

from src.domain.entities import Bar, FundingRate
from src.domain.enums import Timeframe


@runtime_checkable
class IDataProvider(Protocol):
    """
    Interface for exchange market data providers.
    """

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Bar]:
        """Fetch historical OHLCV bars."""
        ...

    @abstractmethod
    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> List[FundingRate]:
        """Fetch historical and current perpetual funding rates."""
        ...


@runtime_checkable
class IBarStorage(Protocol):
    """
    Interface for market data persistence layer (Parquet/DuckDB).
    """

    @abstractmethod
    def store_bars(self, symbol: str, timeframe: Timeframe, bars: List[Bar]) -> int:
        """Store bars incrementally with deduplication. Returns number of inserted bars."""
        ...

    @abstractmethod
    def load_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Bar]:
        """Retrieve historical bars within a time range."""
        ...
