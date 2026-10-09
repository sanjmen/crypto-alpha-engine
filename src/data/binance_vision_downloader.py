"""
High-Speed Bulk Ingestion Engine for Binance Data Vision (S3).
Streams monthly and daily ZIP archives directly from S3, decompresses in memory,
and stores clean, deduplicated Parquet files.
Works seamlessly on Local Mac, Google Colab, and Google Drive mounts.
"""

from datetime import datetime, timezone
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import urllib.request
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.domain.enums import Timeframe

logger = logging.getLogger(__name__)

BINANCE_VISION_S3_BASE = "https://data.binance.vision"

KLINES_COLUMN_NAMES = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

METRICS_COLUMN_NAMES = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
]

FUNDING_COLUMN_NAMES = [
    "calc_time", "funding_interval_hours", "last_funding_rate"
]


class BinanceVisionDownloader:
    """
    Downloads bulk historical data directly from data.binance.vision S3 bucket.
    Zero rate limits, zero API keys, in-memory decompression.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_zip_as_df(self, url: str, column_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                    csv_name = zf.namelist()[0]
                    # Check if file has header
                    with zf.open(csv_name) as f:
                        first_line = f.readline().decode("utf-8")
                        has_header = (
                            "open_time" in first_line
                            or "create_time" in first_line
                            or "calc_time" in first_line
                        )

                    if has_header:
                        df = pd.read_csv(zf.open(csv_name))
                    else:
                        df = pd.read_csv(zf.open(csv_name), header=None, names=column_names)
                    return df
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            logger.warning(f"HTTP Error {e.code} for {url}")
            return None
        except Exception as e:
            logger.error(f"Error streaming {url}: {e}")
            return None

    def download_monthly_klines(
        self,
        symbol: str,
        year: int,
        month: int,
        timeframe: Timeframe = Timeframe.H1,
    ) -> Optional[pd.DataFrame]:
        """Downloads a single monthly klines ZIP archive and parses into a DataFrame."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        month_str = f"{month:02d}"
        url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/klines/{clean_sym}/{timeframe.value}/{clean_sym}-{timeframe.value}-{year}-{month_str}.zip"
        return self._fetch_zip_as_df(url, KLINES_COLUMN_NAMES)

    def download_daily_klines(
        self,
        symbol: str,
        date_str: str,  # YYYY-MM-DD
        timeframe: Timeframe = Timeframe.H1,
    ) -> Optional[pd.DataFrame]:
        """Downloads a single daily klines ZIP archive and parses into a DataFrame."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/klines/{clean_sym}/{timeframe.value}/{clean_sym}-{timeframe.value}-{date_str}.zip"
        return self._fetch_zip_as_df(url, KLINES_COLUMN_NAMES)

    def download_daily_metrics(
        self,
        symbol: str,
        date_str: str,  # YYYY-MM-DD
    ) -> Optional[pd.DataFrame]:
        """Downloads 5-minute derivatives metrics for a specific date."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/metrics/{clean_sym}/{clean_sym}-metrics-{date_str}.zip"
        return self._fetch_zip_as_df(url, METRICS_COLUMN_NAMES)

    def download_monthly_funding_rates(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> Optional[pd.DataFrame]:
        """Downloads monthly 8h funding rate settlements."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        month_str = f"{month:02d}"
        url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/{clean_sym}/{clean_sym}-fundingRate-{year}-{month_str}.zip"
        return self._fetch_zip_as_df(url, FUNDING_COLUMN_NAMES)

    def save_klines_to_parquet(
        self,
        symbol: str,
        timeframe: Timeframe,
        df: pd.DataFrame,
    ) -> int:
        """Saves or appends klines to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_sym = symbol.replace("/", "_").replace(":", "_").upper()
        target_dir = self.output_dir / "bars" / timeframe.value
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_sym}.parquet"

        trade_count = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(np.int64) if "trade_count" in df.columns else np.zeros(len(df), dtype=np.int64)
        taker_vol = pd.to_numeric(df["taker_buy_base_vol"], errors="coerce").fillna(0.0).astype(np.float64) if "taker_buy_base_vol" in df.columns else np.zeros(len(df), dtype=np.float64)

        df_clean = pd.DataFrame({
            "timestamp": (pd.to_numeric(df["open_time"]) // 1000).astype(np.int64),
            "symbol": symbol,
            "open": pd.to_numeric(df["open"], errors="coerce").astype(np.float64),
            "high": pd.to_numeric(df["high"], errors="coerce").astype(np.float64),
            "low": pd.to_numeric(df["low"], errors="coerce").astype(np.float64),
            "close": pd.to_numeric(df["close"], errors="coerce").astype(np.float64),
            "volume": pd.to_numeric(df["volume"], errors="coerce").astype(np.float64),
            "trade_count": trade_count,
            "taker_buy_vol": taker_vol,
        })

        if file_path.exists():
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY timestamp) as rn
                    FROM (
                        SELECT * FROM read_parquet('{file_path}')
                        UNION ALL
                        SELECT * FROM df_clean
                    )
                ) WHERE rn = 1
                ORDER BY timestamp ASC
            """
            merged_df = duckdb.query(query).to_df().drop(columns=["rn"])
        else:
            merged_df = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(merged_df)
        pq.write_table(table, file_path, compression="zstd")
        return len(merged_df)

    def save_metrics_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> int:
        """Saves or appends 5-minute derivatives metrics to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_sym = symbol.replace("/", "_").replace(":", "_").upper()
        target_dir = self.output_dir / "metrics" / "5m"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_sym}.parquet"

        # Convert create_time string or datetime to epoch seconds
        if "create_time" in df.columns:
            dt = pd.to_datetime(df["create_time"], utc=True)
            ts = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        else:
            ts = pd.to_numeric(df.iloc[:, 0]) // 1000

        df_clean = pd.DataFrame({
            "timestamp": ts.astype(np.int64),
            "symbol": symbol,
            "open_interest": pd.to_numeric(df.get("sum_open_interest", 0), errors="coerce").astype(np.float64),
            "open_interest_usd": pd.to_numeric(df.get("sum_open_interest_value", 0), errors="coerce").astype(np.float64),
            "top_trader_ratio": pd.to_numeric(df.get("count_toptrader_long_short_ratio", 0), errors="coerce").astype(np.float64),
            "top_position_ratio": pd.to_numeric(df.get("sum_toptrader_long_short_ratio", 0), errors="coerce").astype(np.float64),
            "retail_ratio": pd.to_numeric(df.get("count_long_short_ratio", 0), errors="coerce").astype(np.float64),
            "taker_long_short_ratio": pd.to_numeric(df.get("sum_taker_long_short_vol_ratio", 0), errors="coerce").astype(np.float64),
        })

        if file_path.exists():
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY timestamp) as rn
                    FROM (
                        SELECT * FROM read_parquet('{file_path}')
                        UNION ALL
                        SELECT * FROM df_clean
                    )
                ) WHERE rn = 1
                ORDER BY timestamp ASC
            """
            merged_df = duckdb.query(query).to_df().drop(columns=["rn"])
        else:
            merged_df = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(merged_df)
        pq.write_table(table, file_path, compression="zstd")
        return len(merged_df)

    def save_funding_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> int:
        """Saves or appends funding rate history to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_sym = symbol.replace("/", "_").replace(":", "_").upper()
        target_dir = self.output_dir / "funding"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_sym}.parquet"

        df_clean = pd.DataFrame({
            "timestamp": (pd.to_numeric(df["calc_time"]) // 1000).astype(np.int64),
            "symbol": symbol,
            "interval_hours": pd.to_numeric(df["funding_interval_hours"], errors="coerce").astype(np.int64),
            "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").astype(np.float64),
        })

        if file_path.exists():
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY timestamp) as rn
                    FROM (
                        SELECT * FROM read_parquet('{file_path}')
                        UNION ALL
                        SELECT * FROM df_clean
                    )
                ) WHERE rn = 1
                ORDER BY timestamp ASC
            """
            merged_df = duckdb.query(query).to_df().drop(columns=["rn"])
        else:
            merged_df = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        table = pa.Table.from_pandas(merged_df)
        pq.write_table(table, file_path, compression="zstd")
        return len(merged_df)
