"""
High-Throughput Remote Ingestion Worker for Google Colab.
Streams Binance Vision historical data directly into Google Drive Parquet storage.
Supports both mounted Google Drive (/content/drive/MyDrive) and direct rclone sync.
Zero local disk space on client Mac. Zero browser tab clicks required.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Storage Configuration
DRIVE_MOUNT_POINT = Path("/content/drive/MyDrive")
if DRIVE_MOUNT_POINT.exists():
    BASE_OUTPUT_DIR = DRIVE_MOUNT_POINT / "trading" / "crypto-alpha-engine" / "data"
    USE_RCLONE = False
else:
    BASE_OUTPUT_DIR = Path("/content/data")
    USE_RCLONE = True

BINANCE_VISION_S3_BASE = "https://data.binance.vision"

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT",
    "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"
]

TIMEFRAMES = ["1h", "15m"]
YEARS = [2023, 2024, 2025, 2026]

KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

METRICS_COLS = [
    "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
    "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
    "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
]

FUNDING_COLS = [
    "calc_time", "funding_interval_hours", "last_funding_rate"
]


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sync_to_drive():
    """Syncs local VM data to Google Drive via rclone if not mounted directly."""
    if not USE_RCLONE:
        return
    log("🔄 Sincronizando datos con Google Drive (rclone)...")
    res = subprocess.run(
        ["rclone", "copy", str(BASE_OUTPUT_DIR), "gdrive:trading/crypto-alpha-engine/data", "--transfers", "8", "--checkers", "16", "-q"],
        capture_output=True,
        text=True,
    )
    if res.returncode == 0:
        log("✅ Sincronización con Google Drive exitosa.")
    else:
        log(f"⚠️ Advertencia en sync rclone: {res.stderr}")


def fetch_zip_df(url: str, col_names: list[str]) -> pd.DataFrame | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    first_line = f.readline().decode("utf-8")
                    has_header = any(k in first_line for k in ["open_time", "create_time", "calc_time"])
                if has_header:
                    return pd.read_csv(zf.open(csv_name))
                else:
                    return pd.read_csv(zf.open(csv_name), header=None, names=col_names)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            log(f"⚠️ HTTP {e.code} for {url}")
        return None
    except Exception as e:
        log(f"⚠️ Error downloading {url}: {e}")
        return None


def save_klines(symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    target_dir = BASE_OUTPUT_DIR / "bars" / timeframe
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{symbol}.parquet"

    df_clean = pd.DataFrame({
        "timestamp": (pd.to_numeric(df["open_time"]) // 1000).astype(np.int64),
        "symbol": symbol,
        "open": pd.to_numeric(df["open"], errors="coerce").astype(np.float64),
        "high": pd.to_numeric(df["high"], errors="coerce").astype(np.float64),
        "low": pd.to_numeric(df["low"], errors="coerce").astype(np.float64),
        "close": pd.to_numeric(df["close"], errors="coerce").astype(np.float64),
        "volume": pd.to_numeric(df["volume"], errors="coerce").astype(np.float64),
        "trade_count": pd.to_numeric(df.get("trade_count", 0), errors="coerce").fillna(0).astype(np.int64),
        "taker_buy_vol": pd.to_numeric(df.get("taker_buy_base_vol", 0), errors="coerce").fillna(0).astype(np.float64),
    })

    if file_path.exists():
        q = f"""
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
        merged = duckdb.query(q).to_df().drop(columns=["rn"])
    else:
        merged = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    table = pa.Table.from_pandas(merged)
    pq.write_table(table, file_path, compression="zstd")
    return len(merged)


def save_funding(symbol: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    target_dir = BASE_OUTPUT_DIR / "funding"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{symbol}.parquet"

    df_clean = pd.DataFrame({
        "timestamp": (pd.to_numeric(df["calc_time"]) // 1000).astype(np.int64),
        "symbol": symbol,
        "interval_hours": pd.to_numeric(df["funding_interval_hours"], errors="coerce").astype(np.int64),
        "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").astype(np.float64),
    })

    if file_path.exists():
        q = f"""
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
        merged = duckdb.query(q).to_df().drop(columns=["rn"])
    else:
        merged = df_clean.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    table = pa.Table.from_pandas(merged)
    pq.write_table(table, file_path, compression="zstd")
    return len(merged)


def download_klines_job(symbol: str, timeframe: str, year: int, month: int):
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
    df = fetch_zip_df(url, KLINES_COLS)
    if df is not None:
        count = save_klines(symbol, timeframe, df)
        return symbol, timeframe, year, month, len(df), count
    return symbol, timeframe, year, month, 0, 0


def download_funding_job(symbol: str, year: int, month: int):
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month_str}.zip"
    df = fetch_zip_df(url, FUNDING_COLS)
    if df is not None:
        count = save_funding(symbol, df)
        return symbol, year, month, len(df), count
    return symbol, year, month, 0, 0


def run_pipeline():
    start_time = time.time()
    log("🚀 [COLAB WORKER] Starting High-Throughput Binance Ingestion directly to Google Drive...")
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"📁 Target Output Directory: {BASE_OUTPUT_DIR} (USE_RCLONE={USE_RCLONE})")

    symbols = TOP_SYMBOLS
    log(f"📊 Selected Universe: {len(symbols)} symbols: {symbols}")
    log(f"⏰ Timeframes: {TIMEFRAMES} | Years: {YEARS}")

    total_klines_downloaded = 0
    total_funding_downloaded = 0

    # 1. Monthly Klines Download across all symbols & timeframes
    log("\n📥 Phase 1: Downloading Historical Klines (1h & 15m)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for sym in symbols:
            for tf in TIMEFRAMES:
                for y in YEARS:
                    for m in range(1, 13):
                        tasks.append(executor.submit(download_klines_job, sym, tf, y, m))

        for f in as_completed(tasks):
            sym, tf, y, m, bars, total = f.result()
            if bars > 0:
                total_klines_downloaded += bars
                log(f"  ✓ {sym} [{tf}] {y}-{m:02d}: +{bars:,} bars (Total in Parquet: {total:,})")

    # Sync klines to Drive
    sync_to_drive()

    # 2. Monthly Funding Rates Download
    log("\n📥 Phase 2: Downloading Historical 8h Funding Rates...")
    funding_tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for sym in symbols:
            for y in YEARS:
                for m in range(1, 13):
                    funding_tasks.append(executor.submit(download_funding_job, sym, y, m))

        for f in as_completed(funding_tasks):
            sym, y, m, records, total = f.result()
            if records > 0:
                total_funding_downloaded += records
                log(f"  ✓ {sym} Funding {y}-{m:02d}: +{records:,} settlements (Total: {total:,})")

    elapsed = time.time() - start_time
    log(f"\n🎉 [COMPLETE] Ingestion finished in {elapsed:.1f}s.")
    log(f"📈 Total Klines Bars Downloaded: {total_klines_downloaded:,}")
    log(f"💰 Total Funding Settlements: {total_funding_downloaded:,}")

    # Write Manifest
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "symbols": symbols,
        "timeframes": TIMEFRAMES,
        "years": YEARS,
        "total_klines_bars": total_klines_downloaded,
        "total_funding_settlements": total_funding_downloaded,
        "drive_path": "gdrive:trading/crypto-alpha-engine/data",
    }
    manifest_path = BASE_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"📄 Manifest written to: {manifest_path}")

    # Final sync
    sync_to_drive()

    # Sentinel file for Colab Runner detection
    sentinel = Path("/content/INGEST_FINISHED")
    sentinel.write_text(json.dumps(manifest, indent=2))
    log("🏁 Sentinel file created: /content/INGEST_FINISHED")


if __name__ == "__main__":
    run_pipeline()
