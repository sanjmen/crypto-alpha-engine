"""
High-Speed Streaming Pipeline for Binance Data Vision aggTrades.
Downloads daily and monthly compressed ZIP archives, parses tick-level trade executions,
computes aggressor side, and converts directly to Snappy-compressed Parquet.
Designed for 0 MB local persistent disk usage (streams via /tmp or Google Drive).
"""

from datetime import datetime, timezone
import io
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Generator, List, Optional, Union
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


class AggTradesStreamer:
    """
    Streams and converts tick-level aggTrades archives directly from Binance Data Vision S3.
    """

    def __init__(self, cache_dir: Union[str, Path] = "/tmp/crypto_alpha_agg_trades"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_daily_url(self, symbol: str, date_str: str) -> str:
        """Constructs URL for daily aggTrades ZIP (date_str: YYYY-MM-DD)."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        return f"{BINANCE_VISION_FUTURES_URL}/daily/aggTrades/{clean_sym}/{clean_sym}-aggTrades-{date_str}.zip"

    def build_monthly_url(self, symbol: str, year: int, month: int) -> str:
        """Constructs URL for monthly aggTrades ZIP."""
        clean_sym = symbol.replace("/", "").replace(":", "").upper()
        return f"{BINANCE_VISION_FUTURES_URL}/monthly/aggTrades/{clean_sym}/{clean_sym}-aggTrades-{year}-{month:02d}.zip"

    def stream_zip_to_df(self, url: str) -> Optional[pd.DataFrame]:
        """
        Streams a compressed ZIP file directly into memory without writing zip to disk.
        Parses CSV ticks with PyArrow engine for maximum throughput.
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CryptoAlphaEngine/1.0)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data_bytes = resp.read()
                with zipfile.ZipFile(io.BytesIO(data_bytes)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        first_line = f.readline().decode("utf-8")
                        has_header = "agg_trade_id" in first_line or "transact_time" in first_line

                    # Parse with PyArrow engine for C-level speed
                    if has_header:
                        df = pd.read_csv(zf.open(csv_name), engine="pyarrow")
                    else:
                        df = pd.read_csv(
                            zf.open(csv_name),
                            names=AGG_TRADES_COLUMNS,
                            engine="pyarrow",
                        )

            # Normalize column types and compute derived attributes
            return self._enrich_trades_df(df)

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Archive not found (404) at {url}")
                return None
            logger.warning(f"HTTP {e.code} error streaming {url}: {e.reason}")
            return None
        except Exception as e:
            logger.error(f"Failed to stream aggTrades from {url}: {e}")
            return None

    def _enrich_trades_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches raw trade ticks with dollar notional, aggressor side, and UTC timestamps."""
        df = df.copy()

        # Coerce numeric types
        df["price"] = df["price"].astype(np.float64)
        df["quantity"] = df["quantity"].astype(np.float64)
        df["notional"] = df["price"] * df["quantity"]

        # Timestamp conversion
        if "transact_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)

        # Buyer maker flag indicates aggressor:
        # If is_buyer_maker == True -> Buyer was Maker (Seller hit the bid, aggressor is SELL)
        # If is_buyer_maker == False -> Buyer was Taker (Buyer lifted the ask, aggressor is BUY)
        if "is_buyer_maker" in df.columns:
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
            # Tick rule sign: +1 for aggressive buy, -1 for aggressive sell
            df["tick_sign"] = np.where(df["is_buyer_maker"], -1, 1).astype(np.int8)

        return df

    def fetch_daily_trades(self, symbol: str, date_str: str) -> Optional[pd.DataFrame]:
        """Fetches a single day of tick aggTrades for a symbol."""
        url = self.build_daily_url(symbol, date_str)
        return self.stream_zip_to_df(url)

    def fetch_monthly_trades(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Fetches an entire month of tick aggTrades for a symbol."""
        url = self.build_monthly_url(symbol, year, month)
        return self.stream_zip_to_df(url)

    def save_trades_parquet(self, df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
        """Saves enriched aggTrades DataFrame to Snappy Parquet."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(p), compression="snappy")
        logger.info(f"Saved {len(df):,} aggTrades to {p}")
        return p

    def sync_to_google_drive(self, local_path: Path, drive_dest: str = "gdrive:trading/crypto-alpha-engine/data/agg_trades/") -> bool:
        """Syncs local parquet file directly to Google Drive via rclone."""
        try:
            subprocess.run(
                ["rclone", "copy", str(local_path), drive_dest],
                check=True,
                capture_output=True,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to sync {local_path} to Google Drive: {e}")
            return False
