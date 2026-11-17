"""
High-Throughput Parquet Bar Storage with DuckDB Query Engine.
Partitioned by symbol and timeframe with automatic deduplication.
"""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.domain.entities import Bar
from src.domain.enums import Timeframe


def _to_utc_epoch_seconds(dt: datetime) -> int:
    """Converts a naive or timezone-aware datetime to UTC seconds since epoch."""
    utc_dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return int(utc_dt.timestamp())


class ParquetBarStorage:
    """
    On-disk storage manager for OHLCV bars using Apache Parquet and DuckDB.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent / "data" / "cache" / "bars"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, timeframe: Timeframe) -> Path:
        clean_symbol = symbol.replace("/", "_").replace(":", "_")
        timeframe_dir = self.base_dir / timeframe.value
        timeframe_dir.mkdir(parents=True, exist_ok=True)
        return timeframe_dir / f"{clean_symbol}.parquet"

    def store_bars(self, symbol: str, timeframe: Timeframe, bars: list[Bar]) -> int:
        """
        Appends bars to the parquet store with timestamp deduplication.
        Returns number of total stored records.
        """
        if not bars:
            return 0

        file_path = self._get_file_path(symbol, timeframe)

        rows = [
            {
                "timestamp": _to_utc_epoch_seconds(bar.timestamp),
                "symbol": bar.symbol,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        new_df = pd.DataFrame(rows)

        if file_path.exists():
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
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Bar]:
        """
        Retrieves bars within an optional time range.
        """
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            return []

        conditions: list[str] = []
        if start_time is not None:
            ts_start = _to_utc_epoch_seconds(start_time)
            conditions.append(f"timestamp >= {ts_start}")
        if end_time is not None:
            ts_end = _to_utc_epoch_seconds(end_time)
            conditions.append(f"timestamp <= {ts_end}")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT timestamp, symbol, open, high, low, close, volume FROM read_parquet('{file_path}') {where_clause} ORDER BY timestamp ASC"

        df = duckdb.query(query).to_df()
        return [
            Bar(
                timestamp=datetime.fromtimestamp(float(row.timestamp), tz=timezone.utc),
                symbol=str(row.symbol),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in df.itertuples(index=False)
        ]
