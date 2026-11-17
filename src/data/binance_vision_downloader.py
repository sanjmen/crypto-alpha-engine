"""
High-Speed Bulk Ingestion Engine for Binance Data Vision (S3).
Streams monthly and daily ZIP archives directly from S3, decompresses in memory,
and stores clean, deduplicated Parquet files.
Works seamlessly on Local Mac, Google Colab, and Google Drive mounts.
"""

import io
import logging
from pathlib import Path
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
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "ignore",
]

METRICS_COLUMN_NAMES = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

FUNDING_COLUMN_NAMES = [
    "calc_time",
    "funding_interval_hours",
    "last_funding_rate",
]

BOOK_DEPTH_COLUMN_NAMES = [
    "timestamp",
    "percentage",
    "depth",
    "notional",
]


def _clean_url_symbol(symbol: str) -> str:
    """Formats symbol for Binance Vision S3 URLs ('BTC/USDT' -> 'BTCUSDT')."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":", "").upper()


def _clean_file_symbol(symbol: str) -> str:
    """Formats symbol for local Parquet filenames ('BTC/USDT' -> 'BTC_USDT')."""
    return symbol.replace("/", "_").replace(":", "_").upper()


def _deduplicate_and_write_parquet(
    df_clean: pd.DataFrame,
    file_path: Path,
    partition_cols: list[str],
) -> int:
    """Merges `df_clean` with existing Parquet file (if present), deduplicating on `partition_cols`."""
    partition_expr = ", ".join(partition_cols)
    order_expr = ", ".join(f"{col} ASC" for col in partition_cols)
    if file_path.exists():
        query = f"""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_expr} ORDER BY timestamp) as rn
                FROM (
                    SELECT * FROM read_parquet('{file_path}')
                    UNION ALL
                    SELECT * FROM df_clean
                )
            ) WHERE rn = 1
            ORDER BY {order_expr}
        """
        merged_df = duckdb.query(query).to_df().drop(columns=["rn"])
    else:
        merged_df = df_clean.drop_duplicates(subset=partition_cols).sort_values(partition_cols).reset_index(drop=True)

    table = pa.Table.from_pandas(merged_df)
    pq.write_table(table, file_path, compression="zstd")
    return len(merged_df)


class BinanceVisionDownloader:
    """
    Downloads bulk historical data directly from data.binance.vision S3 bucket.
    Zero rate limits, zero API keys, in-memory decompression.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_zip_as_df(self, url: str, column_names: list[str] | None = None) -> pd.DataFrame | None:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(data_bytes)) as zip_archive:
                names = zip_archive.namelist()
                if not names:
                    return None
                csv_name = names[0]
                with zip_archive.open(csv_name) as probe_file:
                    first_line = probe_file.readline().decode("utf-8", errors="ignore")
                    has_header = any(
                        token in first_line for token in ("open_time", "create_time", "calc_time", "percentage")
                    )

                with zip_archive.open(csv_name) as csv_file:
                    if has_header:
                        return pd.read_csv(csv_file)
                    return pd.read_csv(csv_file, header=None, names=column_names)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            logger.warning(f"HTTP Error {err.code} for {url}")
            return None
        except Exception as err:  # noqa: BLE001
            logger.error(f"Error streaming {url}: {err}")
            return None

    def download_monthly_klines(
        self,
        symbol: str,
        year: int,
        month: int,
        timeframe: Timeframe = Timeframe.H1,
    ) -> pd.DataFrame | None:
        """Downloads a single monthly klines ZIP archive and parses into a DataFrame."""
        clean_symbol = _clean_url_symbol(symbol)
        month_str = f"{month:02d}"
        url = (
            f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/klines/"
            f"{clean_symbol}/{timeframe.value}/{clean_symbol}-{timeframe.value}-{year}-{month_str}.zip"
        )
        return self._fetch_zip_as_df(url, KLINES_COLUMN_NAMES)

    def download_daily_klines(
        self,
        symbol: str,
        date_str: str,
        timeframe: Timeframe = Timeframe.H1,
    ) -> pd.DataFrame | None:
        """Downloads a single daily klines ZIP archive and parses into a DataFrame."""
        clean_symbol = _clean_url_symbol(symbol)
        url = (
            f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/klines/"
            f"{clean_symbol}/{timeframe.value}/{clean_symbol}-{timeframe.value}-{date_str}.zip"
        )
        return self._fetch_zip_as_df(url, KLINES_COLUMN_NAMES)

    def download_daily_metrics(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame | None:
        """Downloads 5-minute derivatives metrics for a specific date."""
        clean_symbol = _clean_url_symbol(symbol)
        url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/metrics/{clean_symbol}/{clean_symbol}-metrics-{date_str}.zip"
        return self._fetch_zip_as_df(url, METRICS_COLUMN_NAMES)

    def download_monthly_funding_rates(
        self,
        symbol: str,
        year: int,
        month: int,
    ) -> pd.DataFrame | None:
        """Downloads monthly 8h funding rate settlements."""
        clean_symbol = _clean_url_symbol(symbol)
        month_str = f"{month:02d}"
        url = (
            f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/"
            f"{clean_symbol}/{clean_symbol}-fundingRate-{year}-{month_str}.zip"
        )
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

        clean_symbol = _clean_file_symbol(symbol)
        target_dir = self.output_dir / "bars" / timeframe.value
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_symbol}.parquet"

        trade_count = (
            pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(np.int64)
            if "trade_count" in df.columns
            else np.zeros(len(df), dtype=np.int64)
        )
        taker_vol = (
            pd.to_numeric(df["taker_buy_base_vol"], errors="coerce").fillna(0.0).astype(np.float64)
            if "taker_buy_base_vol" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        open_time_sec = (pd.to_numeric(df["open_time"], errors="coerce").fillna(0) // 1000).astype(np.int64)

        df_clean = pd.DataFrame(
            {
                "timestamp": open_time_sec,
                "symbol": symbol,
                "open": pd.to_numeric(df["open"], errors="coerce").astype(np.float64),
                "high": pd.to_numeric(df["high"], errors="coerce").astype(np.float64),
                "low": pd.to_numeric(df["low"], errors="coerce").astype(np.float64),
                "close": pd.to_numeric(df["close"], errors="coerce").astype(np.float64),
                "volume": pd.to_numeric(df["volume"], errors="coerce").astype(np.float64),
                "trade_count": trade_count,
                "taker_buy_vol": taker_vol,
            }
        )
        return _deduplicate_and_write_parquet(df_clean, file_path, ["timestamp"])

    def save_metrics_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> int:
        """Saves or appends 5-minute derivatives metrics to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_symbol = _clean_file_symbol(symbol)
        target_dir = self.output_dir / "metrics" / "5m"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_symbol}.parquet"

        if "create_time" in df.columns:
            dt = pd.to_datetime(df["create_time"], utc=True)
            ts = (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        else:
            ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").fillna(0) // 1000

        df_clean = pd.DataFrame(
            {
                "timestamp": ts.astype(np.int64),
                "symbol": symbol,
                "open_interest": pd.to_numeric(df.get("sum_open_interest", 0), errors="coerce").astype(np.float64),
                "open_interest_usd": pd.to_numeric(df.get("sum_open_interest_value", 0), errors="coerce").astype(
                    np.float64
                ),
                "top_trader_ratio": pd.to_numeric(
                    df.get("count_toptrader_long_short_ratio", 0), errors="coerce"
                ).astype(np.float64),
                "top_position_ratio": pd.to_numeric(
                    df.get("sum_toptrader_long_short_ratio", 0), errors="coerce"
                ).astype(np.float64),
                "retail_ratio": pd.to_numeric(df.get("count_long_short_ratio", 0), errors="coerce").astype(np.float64),
                "taker_long_short_ratio": pd.to_numeric(
                    df.get("sum_taker_long_short_vol_ratio", 0), errors="coerce"
                ).astype(np.float64),
            }
        )
        return _deduplicate_and_write_parquet(df_clean, file_path, ["timestamp"])

    def save_funding_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> int:
        """Saves or appends funding rate history to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_symbol = _clean_file_symbol(symbol)
        target_dir = self.output_dir / "funding"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_symbol}.parquet"

        calc_time_sec = (pd.to_numeric(df["calc_time"], errors="coerce").fillna(0) // 1000).astype(np.int64)
        interval_hrs = pd.to_numeric(df["funding_interval_hours"], errors="coerce").fillna(8).astype(np.int64)

        df_clean = pd.DataFrame(
            {
                "timestamp": calc_time_sec,
                "symbol": symbol,
                "interval_hours": interval_hrs,
                "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").astype(np.float64),
            }
        )
        return _deduplicate_and_write_parquet(df_clean, file_path, ["timestamp"])

    def download_daily_book_depth(
        self,
        symbol: str,
        date_str: str,
    ) -> pd.DataFrame | None:
        """Downloads a single daily L2 bookDepth ZIP archive and parses into a DataFrame."""
        clean_symbol = _clean_url_symbol(symbol)
        url = (
            f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/bookDepth/"
            f"{clean_symbol}/{clean_symbol}-bookDepth-{date_str}.zip"
        )
        return self._fetch_zip_as_df(url, BOOK_DEPTH_COLUMN_NAMES)

    def save_book_depth_to_parquet(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> int:
        """Saves L2 bookDepth snapshots to partitioned Parquet file."""
        if df is None or df.empty:
            return 0

        clean_symbol = _clean_file_symbol(symbol)
        target_dir = self.output_dir / "book_depth" / "daily"
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{clean_symbol}.parquet"

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            ts_epoch = (ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
        else:
            ts_epoch = pd.Series(np.zeros(len(df), dtype=np.int64))

        df_clean = pd.DataFrame(
            {
                "timestamp": ts_epoch.fillna(0).astype(np.int64),
                "symbol": symbol,
                "percentage": pd.to_numeric(df["percentage"], errors="coerce").fillna(0.0).astype(np.float64),
                "depth": pd.to_numeric(df["depth"], errors="coerce").astype(np.float64),
                "notional": pd.to_numeric(df["notional"], errors="coerce").astype(np.float64),
            }
        )
        return _deduplicate_and_write_parquet(df_clean, file_path, ["timestamp", "percentage"])
