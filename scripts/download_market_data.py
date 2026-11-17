#!/usr/bin/env python3
"""
Unified Market Data Ingestion & Cloud/Colab Sync CLI (Issue #66).

Consolidates all historical and sub-hourly data ingestion pipelines into a single parameterized CLI:
- `--mode ccxt` (default): Downloads historical bars via CCXT (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`) and stores them in Parquet.
- `--mode subhourly`: Syncs 15m Parquet bars from Google Drive via `rclone` and tops up `15m`, `5m`, and `1m` bars from Binance Futures into `/tmp/crypto_alpha_backtest_cache`.
- `--mode vision`: High-throughput multi-threaded Binance Data Vision S3 monthly klines (`1h`, `15m`) + 8h funding rates downloader with optional `rclone` Google Drive sync.
- `--mode colab`: Headless Google Colab remote compute orchestrator (`--colab-action {status,ensure,mount,ingest,stop,heartbeat}`).

Usage Examples:
    python scripts/download_market_data.py --preset top10 --timeframe 1h --days 365
    python scripts/download_market_data.py --mode subhourly --preset top15
    python scripts/download_market_data.py --mode vision --preset all82 --output-dir /tmp/crypto_alpha_backtest_cache
    python scripts/download_market_data.py --mode colab --colab-action status
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ccxt_provider import CCXTMarketDataProvider, TOP_30_SYMBOLS
from src.data.parquet_storage import ParquetBarStorage
from src.domain.enums import Timeframe

TOP_15_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "FETUSDT",
]

ALL_82_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT",
    "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT",
    "SUIUSDT", "APTUSDT", "DOTUSDT", "ATOMUSDT", "FTMUSDT",
    "ALGOUSDT", "EGLDUSDT", "KAVAUSDT", "IOTAUSDT", "MINAUSDT",
    "ICPUSDT", "SEIUSDT", "TIAUSDT", "THETAUSDT", "ZILUSDT",
    "ARBUSDT", "OPUSDT", "MATICUSDT", "STRKUSDT", "ZKUSDT",
    "METUSDT", "ZROUSDT", "UNIUSDT", "AAVEUSDT", "CRVUSDT",
    "MKRUSDT", "SNXUSDT", "COMPUSDT", "1INCHUSDT", "SUSHIUSDT",
    "DYDXUSDT", "JUPUSDT", "ENAUSDT", "INJUSDT", "RUNEUSDT",
    "LDOUSDT", "PENDLEUSDT", "FETUSDT", "RENDERUSDT", "TAOUSDT",
    "WLDUSDT", "GRTUSDT", "RLCUSDT", "LTCUSDT", "BCHUSDT",
    "ETCUSDT", "XMRUSDT", "ZECUSDT", "FILUSDT", "1000PEPEUSDT",
    "1000SHIBUSDT", "1000FLOKIUSDT", "BONKUSDT", "WIFUSDT", "MEMEUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "GALAUSDT", "ENJUSDT",
    "CHZUSDT", "MAGICUSDT", "PYTHUSDT", "ONDOUSDT", "QNTUSDT",
    "BATUSDT", "SKLUSDT", "CTSIUSDT", "GTCUSDT", "OGNUSDT",
    "KSMUSDT", "FLOWUSDT", "WUSDT",
]

PRESETS = {
    "top5": TOP_30_SYMBOLS[:5],
    "top10": TOP_30_SYMBOLS[:10],
    "top15": [f"{s[:-4]}/USDT" if s.endswith("USDT") and "/" not in s else s for s in TOP_15_SYMBOLS],
    "top30": TOP_30_SYMBOLS[:30],
    "all82": [f"{s[:-4]}/USDT" if s.endswith("USDT") and "/" not in s else s for s in ALL_82_SYMBOLS],
}

BINANCE_VISION_S3_BASE = "https://data.binance.vision"
KLINES_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]
FUNDING_COLS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _raw_symbol(sym: str) -> str:
    return sym.replace("/", "").replace(":USDT", "")


# ==============================================================================
# MODE 1: CCXT STANDARD INGESTION
# ==============================================================================
def run_ccxt_mode(args: argparse.Namespace) -> None:
    symbols = PRESETS[args.preset]
    timeframe = Timeframe(args.timeframe)
    since = datetime.now(tz=timezone.utc) - timedelta(days=args.days)

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: MARKET DATA INGESTION PIPELINE [CCXT]")
    print("=" * 70)
    print(f"  Exchange:  {args.exchange.upper()} (Free Public Endpoints)")
    print(f"  Timeframe: {timeframe.value} | Lookback: {args.days} days")
    print(f"  Universe:  {len(symbols)} assets ({', '.join(symbols[:5])}...)")
    print("-" * 70)

    provider = CCXTMarketDataProvider(exchange_id=args.exchange)
    storage = ParquetBarStorage()

    total_bars = 0
    for sym in symbols:
        print(f"  Fetching {sym:<12} ... ", end="", flush=True)
        bars = provider.fetch_bars(symbol=sym, timeframe=timeframe, since=since, limit=1000)
        if bars:
            count = storage.store_bars(symbol=sym, timeframe=timeframe, bars=bars)
            total_bars += len(bars)
            print(f"OK ({len(bars)} bars downloaded, {count} stored in Parquet)")
        else:
            print("FAILED or EMPTY")

    print("-" * 70)
    print(f"  Completed! Total bars ingested: {total_bars:,}")
    print(f"  Stored at: {storage.base_dir}")
    print("=" * 70)


# ==============================================================================
# MODE 2: SUB-HOURLY CACHE TOP-UP (15m, 5m, 1m)
# ==============================================================================
def _parse_klines_to_df(rows: list, symbol: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_vol", "taker_buy_quote_vol", "ignore",
        ],
    )
    df["timestamp"] = (df["open_time"].astype("int64") // 1000).astype("float64")
    df["symbol"] = symbol
    for col in ["open", "high", "low", "close", "volume", "trade_count", "taker_buy_vol"]:
        df[col] = df[col].astype("float64")
    return df[["timestamp", "symbol", "open", "high", "low", "close", "volume", "trade_count", "taker_buy_vol"]]


def _fetch_klines_range(exchange, symbol: str, interval: str, start_ms: int, end_ms: int, step_ms: int) -> pd.DataFrame:
    all_rows = []
    cur_ms = start_ms
    while cur_ms < end_ms:
        try:
            rows = exchange.fapiPublicGetKlines({
                "symbol": symbol,
                "interval": interval,
                "startTime": cur_ms,
                "limit": 1500,
            })
            if not rows:
                break
            all_rows.extend(rows)
            last_open = int(rows[-1][0])
            if last_open >= end_ms or len(rows) < 1500:
                break
            cur_ms = last_open + step_ms
        except Exception as e:
            print(f"  Warning {symbol} {interval} at {cur_ms}: {e}")
            time.sleep(0.5)
            break
    df = _parse_klines_to_df(all_rows, symbol)
    if not df.empty:
        df = df[df["timestamp"] <= end_ms / 1000.0]
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def run_subhourly_mode(args: argparse.Namespace) -> None:
    import ccxt

    t0 = time.time()
    cache_dir = Path(args.output_dir)
    symbols = [_raw_symbol(s) for s in PRESETS[args.preset]]
    bars_15m_dir = cache_dir / "bars" / "15m"
    bars_5m_dir = cache_dir / "bars" / "5m"
    bars_1m_dir = cache_dir / "bars" / "1m"
    for d in [bars_15m_dir, bars_5m_dir, bars_1m_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("1) Syncing 15m Parquet files from Google Drive...")
    pattern = "{" + ",".join([f"{s}.parquet" for s in symbols]) + "}"
    subprocess.run(
        [
            "rclone", "copy",
            "gdrive:trading/crypto-alpha-engine/data/bars/15m/",
            str(bars_15m_dir),
            "--include", pattern,
            "--transfers", "16",
        ],
        check=False,
    )

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    end_ts_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    print("2) Topping up 15m, 5m, and 1m bars from Binance Futures...")
    since_15m_ms = end_ts_ms - 30 * 86400 * 1000
    since_5m_ms = end_ts_ms - 14 * 86400 * 1000
    since_1m_ms = end_ts_ms - 7 * 86400 * 1000

    for sym in symbols:
        p15 = bars_15m_dir / f"{sym}.parquet"
        new_15 = _fetch_klines_range(exchange, sym, "15m", since_15m_ms, end_ts_ms, 15 * 60 * 1000)
        if p15.exists() and not new_15.empty:
            old_df = pd.read_parquet(p15)
            if "timestamp" not in old_df.columns:
                old_df = old_df.reset_index()
            if pd.api.types.is_datetime64_any_dtype(old_df["timestamp"]):
                old_df["timestamp"] = old_df["timestamp"].astype("int64") // 10**9
            combined = pd.concat([old_df, new_15], ignore_index=True)
            combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        else:
            combined = new_15
        if not combined.empty:
            combined.to_parquet(p15, index=False)

        df_5m = _fetch_klines_range(exchange, sym, "5m", since_5m_ms, end_ts_ms, 5 * 60 * 1000)
        if not df_5m.empty:
            df_5m.to_parquet(bars_5m_dir / f"{sym}.parquet", index=False)

        df_1m = _fetch_klines_range(exchange, sym, "1m", since_1m_ms, end_ts_ms, 60 * 1000)
        if not df_1m.empty:
            df_1m.to_parquet(bars_1m_dir / f"{sym}.parquet", index=False)
        print(f"  ✓ {sym:<12} | 15m: {len(combined):6d} | 5m: {len(df_5m):6d} | 1m: {len(df_1m):6d}")

    print(f"Done in {time.time() - t0:.1f}s!")


# ==============================================================================
# MODE 3: BINANCE DATA VISION S3 HIGH-THROUGHPUT WORKER
# ==============================================================================
def _fetch_zip_df(url: str, col_names: list[str]) -> pd.DataFrame | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    first_line = f.readline().decode("utf-8")
                    has_header = any(k in first_line for k in ["open_time", "create_time", "calc_time"])
                if has_header:
                    return pd.read_csv(zf.open(csv_name))
                return pd.read_csv(zf.open(csv_name), header=None, names=col_names)
    except Exception:
        return None


def _fetch_month_kline(symbol: str, timeframe: str, year: int, month: int) -> pd.DataFrame | None:
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-{year}-{month_str}.zip"
    df = _fetch_zip_df(url, KLINES_COLS)
    if df is not None and not df.empty:
        trade_count = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(np.int64) if "trade_count" in df.columns else np.zeros(len(df), dtype=np.int64)
        taker_vol = pd.to_numeric(df["taker_buy_base_vol"], errors="coerce").fillna(0.0).astype(np.float64) if "taker_buy_base_vol" in df.columns else np.zeros(len(df), dtype=np.float64)
        return pd.DataFrame({
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
    return None


def _fetch_month_funding(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year}-{month_str}.zip"
    df = _fetch_zip_df(url, FUNDING_COLS)
    if df is not None and not df.empty:
        return pd.DataFrame({
            "timestamp": (pd.to_numeric(df["calc_time"]) // 1000).astype(np.int64),
            "symbol": symbol,
            "interval_hours": pd.to_numeric(df["funding_interval_hours"], errors="coerce").astype(np.int64),
            "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").astype(np.float64),
        })
    return None


def run_vision_mode(args: argparse.Namespace) -> None:
    start_time = time.time()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    symbols = [_raw_symbol(s) for s in PRESETS[args.preset]]
    years = [int(y) for y in args.years.split(",")]
    timeframes = ["1h", "15m"] if args.include_subhourly else [args.timeframe]

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: BINANCE VISION S3 HIGH-THROUGHPUT INGESTION")
    print("=" * 70)
    print(f"  Output Dir: {base_output_dir} | Symbols: {len(symbols)} | Years: {years} | TFs: {timeframes}")
    print("-" * 70)

    total_klines = 0
    total_funding = 0

    for tf in timeframes:
        for sym in symbols:
            dfs = []
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(_fetch_month_kline, sym, tf, y, m) for y in years for m in range(1, 13)]
                for f in as_completed(futures):
                    res_df = f.result()
                    if res_df is not None:
                        dfs.append(res_df)
            if dfs:
                merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                target_dir = base_output_dir / "bars" / tf
                target_dir.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(merged), target_dir / f"{sym}.parquet", compression="zstd")
                total_klines += len(merged)
                print(f"  ✓ {sym} [{tf}]: {len(merged):,} bars")

    if args.include_funding:
        for sym in symbols:
            dfs = []
            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(_fetch_month_funding, sym, y, m) for y in years for m in range(1, 13)]
                for f in as_completed(futures):
                    res_df = f.result()
                    if res_df is not None:
                        dfs.append(res_df)
            if dfs:
                merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                target_dir = base_output_dir / "funding"
                target_dir.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(merged), target_dir / f"{sym}.parquet", compression="zstd")
                total_funding += len(merged)
                print(f"  ✓ {sym} [funding]: {len(merged):,} settlements")

    if args.sync_drive:
        subprocess.run(
            ["rclone", "copy", str(base_output_dir), "gdrive:trading/crypto-alpha-engine/data", "--transfers", "8", "--checkers", "16", "-q"],
            check=False,
        )

    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "symbols": symbols,
        "timeframes": timeframes,
        "years": years,
        "total_klines_bars": total_klines,
        "total_funding_settlements": total_funding,
    }
    (base_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"🎉 Complete in {manifest['elapsed_seconds']}s | Klines: {total_klines:,} | Funding: {total_funding:,}")


# ==============================================================================
# MODE 4: GOOGLE COLAB REMOTE ORCHESTRATOR
# ==============================================================================
def run_colab_mode(args: argparse.Namespace) -> None:
    from src.cloud.colab_client import ColabClient

    client = ColabClient(session_name=args.colab_session)
    act = args.colab_action
    if act == "status":
        active = client.is_session_active()
        print(f"[*] Session '{args.colab_session}' active: {active}")
        if active:
            print(f"[*] Google Drive mounted: {client.is_drive_mounted()}")
    elif act == "ensure":
        print(f"[*] Ensure session result: {client.ensure_session(gpu=args.colab_gpu)}")
    elif act == "mount":
        rc, out, err = client.mount_drive()
        print(f"[*] Mount Drive result ({rc}): {out or err}")
    elif act == "heartbeat":
        print(f"[*] Heartbeat sent: {client.send_heartbeat()}")
    elif act == "stop":
        print(f"[*] Session stopped: {client.stop()}")
    elif act == "ingest":
        if not client.ensure_session(gpu=args.colab_gpu):
            raise RuntimeError(f"Failed to connect to Colab session '{args.colab_session}'")
        client.install_packages(["duckdb", "pyarrow", "pandas", "requests"])
        script_path = Path(__file__).resolve()
        client.upload(script_path, "/content/download_market_data.py")
        cmd = (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '/content/download_market_data.py', "
            "'--mode', 'vision', '--preset', 'all82', '--output-dir', '/content/data', '--sync-drive'], "
            "start_new_session=True)"
        )
        rc, out, err = client.exec_inline(cmd, timeout=60)
        print(f"[+] Dispatched remote Vision ingestion on Colab ({rc}): {out or err}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Crypto Market Data Ingestion & Cloud/Colab Sync CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["ccxt", "subhourly", "vision", "colab"],
        default="ccxt",
        help="Ingestion mode: ccxt (default), subhourly (15m/5m/1m cache top-up), vision (Binance S3 bulk), or colab (remote VM).",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="top10",
        help="Symbol universe preset (default: top10).",
    )
    parser.add_argument("--exchange", default="binance", help="Exchange ID for --mode ccxt (default: binance).")
    parser.add_argument(
        "--timeframe",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        default="1h",
        help="Bar timeframe (default: 1h).",
    )
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to fetch in --mode ccxt (default: 30).")
    parser.add_argument("--output-dir", type=str, default="/tmp/crypto_alpha_backtest_cache", help="Target directory for subhourly/vision modes.")
    parser.add_argument("--years", type=str, default="2023,2024,2025,2026", help="Comma-separated years for --mode vision.")
    parser.add_argument("--include-subhourly", action="store_true", help="Include 15m klines alongside 1h in --mode vision.")
    parser.add_argument("--no-funding", dest="include_funding", action="store_false", default=True, help="Skip 8h funding rates in --mode vision.")
    parser.add_argument("--sync-drive", action="store_true", help="Sync downloaded Parquet files to Google Drive via rclone.")
    parser.add_argument("--colab-session", type=str, default="crypto_data_ingest", help="Colab session name for --mode colab.")
    parser.add_argument(
        "--colab-action",
        choices=["status", "ensure", "mount", "ingest", "stop", "heartbeat"],
        default="status",
        help="Action to execute in --mode colab (default: status).",
    )
    parser.add_argument("--colab-gpu", type=str, default=None, help="Optional GPU type for --mode colab.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "ccxt":
        run_ccxt_mode(args)
    elif args.mode == "subhourly":
        run_subhourly_mode(args)
    elif args.mode == "vision":
        run_vision_mode(args)
    elif args.mode == "colab":
        run_colab_mode(args)


if __name__ == "__main__":
    main()
