"""
High-Throughput Parquet Bar Storage with DuckDB Query Engine.
Partitioned by symbol and timeframe with automatic deduplication.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Optional
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.domain.entities import Bar
from src.domain.enums import Timeframe

logger = logging.getLogger(__name__)


class ParquetBarStorage:
    """
    On-disk storage manager for OHLCV bars using Apache Parquet and DuckDB.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "bars"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, timeframe: Timeframe) -> Path:
        clean_sym = symbol.replace("/", "_").replace(":", "_")
        timeframe_dir = self.base_dir / timeframe.value
        timeframe_dir.mkdir(parents=True, exist_ok=True)
        return timeframe_dir / f"{clean_sym}.parquet"

    def store_bars(self, symbol: str, timeframe: Timeframe, bars: List[Bar]) -> int:
        """
        Appends bars to the parquet store with timestamp deduplication.
        Returns number of total stored records.
        """
        if not bars:
            return 0

        file_path = self._get_file_path(symbol, timeframe)

        new_data = {
            "timestamp": [int(b.timestamp.replace(tzinfo=timezone.utc).timestamp()) for b in bars],
            "symbol": [b.symbol for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
        new_df = pd.DataFrame(new_data)

        if file_path.exists():
            # Merge with existing file using DuckDB for instant deduplication
            query = f"""
                SELECT timestamp, symbol, open, high, low, close, volume
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY timestamp) as rn
                    FROM (
                        SELECT * FROM read_parquet('{file_path}')
                        UNION ALL
                        SELECT * FROM new_df
                    )
                ) WHERE rn = 1
                ORDER BY timestamp ASC
            """
            merged_df = duckdb.query(query).to_df()
        else:
            merged_df = new_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(merged_df)
        pq.write_table(table, file_path, compression="zstd")
        return len(merged_df)

    def load_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Bar]:
        """
        Retrieves bars within an optional time range.
        """
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            return []

        conditions = []
        if start_time is not None:
            ts_start = int(start_time.replace(tzinfo=timezone.utc).timestamp())
            conditions.append(f"timestamp >= {ts_start}")
        if end_time is not None:
            ts_end = int(end_time.replace(tzinfo=timezone.utc).timestamp())
            conditions.append(f"timestamp <= {ts_end}")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM read_parquet('{file_path}') {where_clause} ORDER BY timestamp ASC"

        df = duckdb.query(query).to_df()
        bars: List[Bar] = []
        for _, row in df.iterrows():
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(row["timestamp"], tz=timezone.utc),
                    symbol=row["symbol"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
        return bars
