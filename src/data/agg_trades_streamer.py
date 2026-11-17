"""
High-Speed Streaming Pipeline for Binance Data Vision aggTrades.
Downloads daily and monthly compressed ZIP archives, parses tick-level trade executions,
computes aggressor side, and converts directly to Snappy-compressed Parquet.
Designed for 0 MB local persistent disk usage (streams via /tmp or Google Drive).
"""

import io
import logging
from pathlib import Path
import subprocess
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

BINANCE_VISION_FUTURES_URL = "https://data.binance.vision/data/futures/um"

# Binance Vision aggTrades CSV column layout
AGG_TRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def _normalize_vision_symbol(symbol: str) -> str:
    """Normalizes a trading pair symbol (e.g., 'BTC/USDT:USDT') into Binance Vision format ('BTCUSDT')."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":", "").upper()


class AggTradesStreamer:
    """
    Streams and converts tick-level aggTrades archives directly from Binance Data Vision S3.
    """

    def __init__(self, cache_dir: str | Path = "/tmp/crypto_alpha_agg_trades") -> None:
        self.cache_dir = Path(cache_dir)

    def build_daily_url(self, symbol: str, date_str: str) -> str:
        """Constructs URL for daily aggTrades ZIP (date_str: YYYY-MM-DD)."""
        clean_symbol = _normalize_vision_symbol(symbol)
        return f"{BINANCE_VISION_FUTURES_URL}/daily/aggTrades/{clean_symbol}/{clean_symbol}-aggTrades-{date_str}.zip"

    def build_monthly_url(self, symbol: str, year: int, month: int) -> str:
        """Constructs URL for monthly aggTrades ZIP."""
        clean_symbol = _normalize_vision_symbol(symbol)
        return (
            f"{BINANCE_VISION_FUTURES_URL}/monthly/aggTrades/"
            f"{clean_symbol}/{clean_symbol}-aggTrades-{year}-{month:02d}.zip"
        )

    def stream_zip_to_df(self, url: str) -> pd.DataFrame | None:
        """
        Streams a compressed ZIP file directly into memory without writing zip to disk.
        Parses CSV ticks with PyArrow engine for maximum throughput.
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CryptoAlphaEngine/1.0)"},
        )
        try:
            with (
                urllib.request.urlopen(req, timeout=60) as resp,
                zipfile.ZipFile(io.BytesIO(resp.read())) as zip_archive,
            ):
                names = zip_archive.namelist()
                if not names:
                    return None
                csv_name = names[0]
                with zip_archive.open(csv_name) as probe_file:
                    first_line = probe_file.readline().decode("utf-8", errors="ignore")
                    has_header = "agg_trade_id" in first_line or "transact_time" in first_line

                with zip_archive.open(csv_name) as csv_file:
                    if has_header:
                        df = pd.read_csv(csv_file, engine="pyarrow")
                    else:
                        df = pd.read_csv(
                            csv_file,
                            names=AGG_TRADES_COLUMNS,
                            engine="pyarrow",
                        )

            return self._enrich_trades_df(df)

        except urllib.error.HTTPError as err:
            if err.code == 404:
                logger.debug(f"Archive not found (404) at {url}")
                return None
            logger.warning(f"HTTP {err.code} error streaming {url}: {err.reason}")
            return None
        except Exception as err:  # noqa: BLE001
            logger.error(f"Failed to stream aggTrades from {url}: {err}")
            return None

    def _enrich_trades_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches raw trade ticks with dollar notional, aggressor side, and UTC timestamps."""
        enriched = df.copy()

        enriched["price"] = pd.to_numeric(enriched["price"], errors="coerce").fillna(0.0).astype(np.float64)
        enriched["quantity"] = pd.to_numeric(enriched["quantity"], errors="coerce").fillna(0.0).astype(np.float64)
        enriched["notional"] = enriched["price"] * enriched["quantity"]

        if "transact_time" in enriched.columns:
            enriched["timestamp"] = pd.to_datetime(enriched["transact_time"], unit="ms", utc=True)

        # Buyer maker flag indicates aggressor:
        # If is_buyer_maker == True -> Buyer was Maker (Seller hit the bid, aggressor is SELL)
        # If is_buyer_maker == False -> Buyer was Taker (Buyer lifted the ask, aggressor is BUY)
        if "is_buyer_maker" in enriched.columns:
            col = enriched["is_buyer_maker"]
            if col.dtype == object or pd.api.types.is_string_dtype(col):
                is_buyer_maker = col.astype(str).str.strip().str.lower().isin(["true", "1", "t"])
            else:
                is_buyer_maker = col.fillna(False).astype(bool)
            enriched["is_buyer_maker"] = is_buyer_maker
            enriched["tick_sign"] = np.where(enriched["is_buyer_maker"], -1, 1).astype(np.int8)

        return enriched

    def fetch_daily_trades(self, symbol: str, date_str: str) -> pd.DataFrame | None:
        """Fetches a single day of tick aggTrades for a symbol."""
        url = self.build_daily_url(symbol, date_str)
        return self.stream_zip_to_df(url)

    def fetch_monthly_trades(self, symbol: str, year: int, month: int) -> pd.DataFrame | None:
        """Fetches an entire month of tick aggTrades for a symbol."""
        url = self.build_monthly_url(symbol, year, month)
        return self.stream_zip_to_df(url)

    def save_trades_parquet(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        """Saves enriched aggTrades DataFrame to Snappy Parquet."""
        parquet_path = Path(output_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(parquet_path), compression="snappy")
        logger.info(f"Saved {len(df):,} aggTrades to {parquet_path}")
        return parquet_path

    def sync_to_google_drive(
        self,
        local_path: Path,
        drive_dest: str = "gdrive:trading/crypto-alpha-engine/data/agg_trades/",
    ) -> bool:
        """Syncs local parquet file directly to Google Drive via rclone."""
        try:
            subprocess.run(
                ["rclone", "copy", str(local_path), drive_dest],
                check=True,
                capture_output=True,
                timeout=120,
            )
            return True
        except Exception as err:  # noqa: BLE001
            logger.warning(f"Failed to sync {local_path} to Google Drive: {err}")
            return False
