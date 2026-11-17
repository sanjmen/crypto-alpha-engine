"""
Port interfaces for Data Ingestion and Persistence.
Adheres strictly to the Dependency Inversion Principle (DIP).
"""

from datetime import datetime
from typing import Protocol, runtime_checkable

from src.domain.entities import Bar, FundingRate
from src.domain.enums import Timeframe


@runtime_checkable
class IDataProvider(Protocol):
    """
    Interface for exchange market data providers.
    """

    def fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[Bar]:
        """Fetch historical OHLCV bars."""
        ...

    def fetch_funding_rates(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[FundingRate]:
        """Fetch historical and current perpetual funding rates."""
        ...


@runtime_checkable
class IBarStorage(Protocol):
    """
    Interface for market data persistence layer (Parquet/DuckDB).
    """

    def store_bars(self, symbol: str, timeframe: Timeframe, bars: list[Bar]) -> int:
        """Store bars incrementally with deduplication. Returns number of inserted bars."""
        ...

    def load_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Bar]:
        """Retrieve historical bars within a time range."""
        ...
