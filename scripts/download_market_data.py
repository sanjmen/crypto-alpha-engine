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

from src.data.ccxt_provider import TOP_30_SYMBOLS, CCXTMarketDataProvider
from src.data.multimodal_vision_ingestor import (
    KLINES_RAW_COLS,
    build_universe_catalog_row,
    fetch_and_assemble_symbol_metrics_5m,
    fetch_vision_zip_csv,
    list_vision_s3_prefixes,
    load_default_pit_220_plan,
    normalize_funding_df,
    normalize_kline_df,
    resolve_spot_candidates,
    stream_multimodal_shards_to_directory,
    verify_spot_price_parity,
    write_parquet_zstd,
)
from src.data.parquet_storage import ParquetBarStorage
from src.domain.enums import Timeframe

TOP_15_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "NEARUSDT",
    "APTUSDT",
    "ARBUSDT",
    "RENDERUSDT",
    "FETUSDT",
]

ALL_82_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "XRPUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "APTUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "FTMUSDT",
    "ALGOUSDT",
    "EGLDUSDT",
    "KAVAUSDT",
    "IOTAUSDT",
    "MINAUSDT",
    "ICPUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "THETAUSDT",
    "ZILUSDT",
    "ARBUSDT",
    "OPUSDT",
    "MATICUSDT",
    "STRKUSDT",
    "ZKUSDT",
    "METUSDT",
    "ZROUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "CRVUSDT",
    "MKRUSDT",
    "SNXUSDT",
    "COMPUSDT",
    "1INCHUSDT",
    "SUSHIUSDT",
    "DYDXUSDT",
    "JUPUSDT",
    "ENAUSDT",
    "INJUSDT",
    "RUNEUSDT",
    "LDOUSDT",
    "PENDLEUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "TAOUSDT",
    "WLDUSDT",
    "GRTUSDT",
    "RLCUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ETCUSDT",
    "XMRUSDT",
    "ZECUSDT",
    "FILUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "1000FLOKIUSDT",
    "BONKUSDT",
    "WIFUSDT",
    "MEMEUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "AXSUSDT",
    "GALAUSDT",
    "ENJUSDT",
    "CHZUSDT",
    "MAGICUSDT",
    "PYTHUSDT",
    "ONDOUSDT",
    "QNTUSDT",
    "BATUSDT",
    "SKLUSDT",
    "CTSIUSDT",
    "GTCUSDT",
    "OGNUSDT",
    "KSMUSDT",
    "FLOWUSDT",
    "WUSDT",
]

PRESETS = {
    "top5": TOP_30_SYMBOLS[:5],
    "top10": TOP_30_SYMBOLS[:10],
    "top15": [f"{s[:-4]}/USDT" if s.endswith("USDT") and "/" not in s else s for s in TOP_15_SYMBOLS],
    "top30": TOP_30_SYMBOLS[:30],
    "all82": [f"{s[:-4]}/USDT" if s.endswith("USDT") and "/" not in s else s for s in ALL_82_SYMBOLS],
    "m16_pit220": sorted(load_default_pit_220_plan().keys()),
}

BINANCE_VISION_S3_BASE = "https://data.binance.vision"
KLINES_COLS = [
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
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_vol",
            "taker_buy_quote_vol",
            "ignore",
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
            rows = exchange.fapiPublicGetKlines(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cur_ms,
                    "limit": 1500,
                }
            )
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
            "rclone",
            "copy",
            "gdrive:trading/crypto-alpha-engine/data/bars/15m/",
            str(bars_15m_dir),
            "--include",
            pattern,
            "--transfers",
            "16",
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
            combined = (
                combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
            )
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
        with urllib.request.urlopen(req, timeout=30) as resp, zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                first_line = f.readline().decode("utf-8")
                has_header = any(k in first_line for k in ["open_time", "create_time", "calc_time"])
            if has_header:
                return pd.read_csv(zf.open(csv_name))
            return pd.read_csv(zf.open(csv_name), header=None, names=col_names)
    except Exception:
        return None


VISION_TICKER_ALIASES: dict[str, list[str]] = {
    "RENDERUSDT": ["RNDRUSDT", "RENDERUSDT"],
    "MATICUSDT": ["MATICUSDT", "POLUSDT"],
    "FTMUSDT": ["FTMUSDT", "SUSDT"],
}


def _normalize_vision_kline_df(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    if df is None or df.empty or df.shape[1] < 6:
        return None
    ts_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(np.int64)
    if ts_raw.empty:
        return None
    df = df.loc[ts_raw.index]
    ts_sec = np.where(ts_raw.values > 10**14, ts_raw.values // 10**6, ts_raw.values // 10**3)
    quote_vol = (
        pd.to_numeric(df.iloc[:, 7], errors="coerce").fillna(0.0).astype(np.float64)
        if df.shape[1] > 7
        else np.zeros(len(df))
    )
    trade_cnt = (
        pd.to_numeric(df.iloc[:, 8], errors="coerce").fillna(0).astype(np.int64)
        if df.shape[1] > 8
        else np.zeros(len(df), dtype=np.int64)
    )
    tbuy_vol = (
        pd.to_numeric(df.iloc[:, 9], errors="coerce").fillna(0.0).astype(np.float64)
        if df.shape[1] > 9
        else np.zeros(len(df))
    )
    tbuy_qvol = (
        pd.to_numeric(df.iloc[:, 10], errors="coerce").fillna(0.0).astype(np.float64)
        if df.shape[1] > 10
        else np.zeros(len(df))
    )
    return pd.DataFrame(
        {
            "timestamp": ts_sec,
            "symbol": symbol,
            "open": pd.to_numeric(df.iloc[:, 1], errors="coerce").astype(np.float64),
            "high": pd.to_numeric(df.iloc[:, 2], errors="coerce").astype(np.float64),
            "low": pd.to_numeric(df.iloc[:, 3], errors="coerce").astype(np.float64),
            "close": pd.to_numeric(df.iloc[:, 4], errors="coerce").astype(np.float64),
            "volume": pd.to_numeric(df.iloc[:, 5], errors="coerce").astype(np.float64),
            "quote_volume": quote_vol,
            "trade_count": trade_cnt,
            "taker_buy_vol": tbuy_vol,
            "taker_buy_quote_volume": tbuy_qvol,
        }
    )


def _fetch_month_kline(
    symbol: str, timeframe: str, year: int, month: int, url_ticker: str | None = None
) -> pd.DataFrame | None:
    tk = url_ticker or symbol
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/klines/{tk}/{timeframe}/{tk}-{timeframe}-{year}-{month_str}.zip"
    return _normalize_vision_kline_df(_fetch_zip_df(url, KLINES_COLS), symbol)


def _fetch_daily_kline(
    symbol: str, timeframe: str, date_str: str, url_ticker: str | None = None
) -> pd.DataFrame | None:
    tk = url_ticker or symbol
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/daily/klines/{tk}/{timeframe}/{tk}-{timeframe}-{date_str}.zip"
    return _normalize_vision_kline_df(_fetch_zip_df(url, KLINES_COLS), symbol)


def _resample_1m_df(df_1m: pd.DataFrame, rule: str, symbol: str) -> pd.DataFrame:
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    d = d.set_index("dt").sort_index()
    res = (
        d.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
                "trade_count": "sum",
                "taker_buy_vol": "sum",
                "taker_buy_quote_volume": "sum",
            }
        )
        .dropna(subset=["open", "close"])
    )
    res["timestamp"] = res.index.as_unit("s").astype(np.int64)
    res["symbol"] = symbol
    return res.reset_index(drop=True)[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_vol",
            "taker_buy_quote_volume",
        ]
    ]


def _fetch_month_funding(symbol: str, year: int, month: int, url_ticker: str | None = None) -> pd.DataFrame | None:
    tk = url_ticker or symbol
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/{tk}/{tk}-fundingRate-{year}-{month_str}.zip"
    df = _fetch_zip_df(url, FUNDING_COLS)
    if df is not None and not df.empty:
        ts_raw = pd.to_numeric(df["calc_time"], errors="coerce").dropna().astype(np.int64)
        df = df.loc[ts_raw.index]
        ts_sec = np.where(ts_raw.values > 10**14, ts_raw.values // 10**6, ts_raw.values // 10**3)
        return pd.DataFrame(
            {
                "timestamp": ts_sec,
                "symbol": symbol,
                "interval_hours": pd.to_numeric(df["funding_interval_hours"], errors="coerce")
                .fillna(8)
                .astype(np.int64),
                "funding_rate": pd.to_numeric(df["last_funding_rate"], errors="coerce").fillna(0.0).astype(np.float64),
            }
        )
    return None


def _fetch_month_premium(symbol: str, year: int, month: int, url_ticker: str | None = None) -> pd.DataFrame | None:
    tk = url_ticker or symbol
    month_str = f"{month:02d}"
    url = f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/premiumIndexKlines/{tk}/1h/{tk}-1h-{year}-{month_str}.zip"
    df = _fetch_zip_df(url, KLINES_COLS)
    if df is not None and not df.empty:
        ts_raw = pd.to_numeric(df["open_time"], errors="coerce").dropna().astype(np.int64)
        df = df.loc[ts_raw.index]
        ts_sec = np.where(ts_raw.values > 10**14, ts_raw.values // 10**6, ts_raw.values // 10**3)
        return pd.DataFrame(
            {
                "timestamp": ts_sec,
                "symbol": symbol,
                "open": pd.to_numeric(df["open"], errors="coerce").astype(np.float64),
                "high": pd.to_numeric(df["high"], errors="coerce").astype(np.float64),
                "low": pd.to_numeric(df["low"], errors="coerce").astype(np.float64),
                "close": pd.to_numeric(df["close"], errors="coerce").astype(np.float64),
            }
        )
    return None


def _fetch_spot_1m_for_symbol(
    perp_sym: str,
    spot_sym: str,
    multiplier: float,
    years: list[int],
    recent_days: list[str],
) -> pd.DataFrame | None:
    dfs: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [
            ex.submit(
                fetch_vision_zip_csv,
                f"{BINANCE_VISION_S3_BASE}/data/spot/monthly/klines/{spot_sym}/1m/{spot_sym}-1m-{y}-{m:02d}.zip",
                KLINES_RAW_COLS,
            )
            for y in years
            for m in range(1, 13)
        ] + [
            ex.submit(
                fetch_vision_zip_csv,
                f"{BINANCE_VISION_S3_BASE}/data/spot/daily/klines/{spot_sym}/1m/{spot_sym}-1m-{ds}.zip",
                KLINES_RAW_COLS,
            )
            for ds in recent_days
        ]
        for fut in as_completed(futs):
            raw_df = fut.result()
            if raw_df is not None and not raw_df.empty:
                dfs.append(raw_df)
    if not dfs:
        return None
    return normalize_kline_df(
        pd.concat(dfs, ignore_index=True),
        symbol=perp_sym,
        price_multiplier=multiplier,
        spot_symbol=spot_sym,
    )


def run_vision_mode(args: argparse.Namespace) -> None:
    import shutil

    start_time = time.time()
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    universe_plan: dict[str, dict] = {}
    plan_path_arg = getattr(args, "universe_plan_json", None)
    if plan_path_arg and Path(plan_path_arg).exists():
        universe_plan = json.loads(Path(plan_path_arg).read_text())
        symbols = sorted(universe_plan.keys())
    elif args.preset == "m16_pit220":
        universe_plan = load_default_pit_220_plan()
        symbols = sorted(universe_plan.keys())
    else:
        symbols = [_raw_symbol(s) for s in PRESETS[args.preset]]
    years = [int(y) for y in args.years.split(",")]
    if getattr(args, "sharded_multimodal", False):
        rclone_target = "gdrive:trading/crypto-alpha-engine/data/m16_multimodal" if args.sync_drive else None
        stream_multimodal_shards_to_directory(
            output_dir=base_output_dir,
            universe_plan=universe_plan or None,
            years=years,
            rclone_dest=rclone_target,
        )
        return
    timeframes = ["1h", "15m"] if args.include_subhourly else [args.timeframe]
    resample_1m = bool(getattr(args, "resample_from_1m", False)) and args.timeframe == "1m"
    include_prem = bool(getattr(args, "include_premium", False))
    include_metrics = bool(getattr(args, "include_metrics_5m", False))
    include_spot = bool(getattr(args, "include_spot_1m", False))
    clean_local = bool(getattr(args, "clean_local", False)) and bool(args.sync_drive)

    spot_set: set[str] = set()
    if include_spot and not universe_plan:
        spot_prefixes = list_vision_s3_prefixes("data/spot/monthly/klines/")
        spot_set = {p.split("/")[-2] for p in spot_prefixes if p.split("/")[-2].endswith("USDT")}

    now_utc = datetime.now(timezone.utc)
    cur_ym_prefix = f"{now_utc.year:04d}-{now_utc.month:02d}"
    recent_days = [f"{cur_ym_prefix}-{d:02d}" for d in range(1, now_utc.day)] if now_utc.year in years else []
    active_cutoff_ts = int((now_utc - timedelta(days=45)).timestamp())

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: BINANCE VISION S3 -> GOOGLE DRIVE INGESTION")
    print("=" * 70)
    print(
        f"  Staging: {base_output_dir} | Symbols: {len(symbols)} | Years: {years} | "
        f"TFs: {timeframes} | Resample1m: {resample_1m} | Metrics5m: {include_metrics} | "
        f"Spot1m: {include_spot} | SyncDrive: {args.sync_drive} | CleanLocal: {clean_local}"
    )
    print("-" * 70)

    total_klines = 0
    total_funding = 0
    total_metrics = 0
    total_spot = 0
    catalog_rows: list[dict] = []
    batch_size = 6 if clean_local else len(symbols)

    for b_start in range(0, len(symbols), max(1, batch_size)):
        batch_syms = symbols[b_start : b_start + batch_size]
        for sym in batch_syms:
            tickers = VISION_TICKER_ALIASES.get(sym, [sym])
            sym_1m_df: pd.DataFrame | None = None
            sym_fnd_df: pd.DataFrame | None = None
            sym_met_df: pd.DataFrame | None = None
            sym_spot_df: pd.DataFrame | None = None
            for tf in timeframes:
                dfs = []
                with ThreadPoolExecutor(max_workers=16) as ex:
                    futs = [
                        ex.submit(_fetch_month_kline, sym, tf, y, m, tk)
                        for tk in tickers
                        for y in years
                        for m in range(1, 13)
                    ] + [ex.submit(_fetch_daily_kline, sym, tf, ds, tk) for tk in tickers for ds in recent_days]
                    for f in as_completed(futs):
                        res_df = f.result()
                        if res_df is not None:
                            dfs.append(res_df)
                if dfs:
                    merged = (
                        pd.concat(dfs, ignore_index=True)
                        .drop_duplicates(subset=["timestamp"], keep="last")
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
                    if tf == "1m":
                        sym_1m_df = merged
                    target_dir = base_output_dir / "bars" / tf
                    target_dir.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pandas(merged), target_dir / f"{sym}.parquet", compression="zstd")
                    total_klines += len(merged)
                    if resample_1m and tf == "1m":
                        for r_tf, r_rule in (("5m", "5min"), ("15m", "15min"), ("1h", "1h")):
                            r_df = _resample_1m_df(merged, r_rule, sym)
                            r_dir = base_output_dir / "bars" / r_tf
                            r_dir.mkdir(parents=True, exist_ok=True)
                            pq.write_table(pa.Table.from_pandas(r_df), r_dir / f"{sym}.parquet", compression="zstd")
                    print(
                        f"  ✓ {sym:<13} [{tf}{'+5m/15m/1h' if resample_1m else ''}]: {len(merged):,} bars", flush=True
                    )

            if args.include_funding:
                f_dfs = []
                with ThreadPoolExecutor(max_workers=16) as ex:
                    futs_f = [
                        ex.submit(
                            fetch_vision_zip_csv,
                            f"{BINANCE_VISION_S3_BASE}/data/futures/um/monthly/fundingRate/{tk}/{tk}-fundingRate-{y}-{m:02d}.zip",
                        )
                        for tk in tickers
                        for y in years
                        for m in range(1, 13)
                    ]
                    for f in as_completed(futs_f):
                        res_df = f.result()
                        if res_df is not None:
                            f_dfs.append(res_df)
                if f_dfs:
                    merged_f = normalize_funding_df(pd.concat(f_dfs, ignore_index=True), symbol=sym)
                    if merged_f is not None and not merged_f.empty:
                        sym_fnd_df = merged_f
                        f_dir = base_output_dir / "funding"
                        f_dir.mkdir(parents=True, exist_ok=True)
                        write_parquet_zstd(merged_f, f_dir / f"{sym}.parquet")
                        total_funding += len(merged_f)

            if include_metrics:
                sym_met_df = fetch_and_assemble_symbol_metrics_5m(sym, min_year=min(years), max_workers=24)
                if sym_met_df is not None and not sym_met_df.empty:
                    m_dir = base_output_dir / "metrics" / "5m"
                    write_parquet_zstd(sym_met_df, m_dir / f"{sym}.parquet")
                    total_metrics += len(sym_met_df)

            sp_sym_used: str | None = None
            sp_mult_used: float = 1.0
            if include_spot:
                if sym in universe_plan:
                    sp_sym_used = universe_plan[sym].get("spot_symbol")
                    sp_mult_used = float(universe_plan[sym].get("spot_multiplier") or 1.0)
                else:
                    cands = resolve_spot_candidates(sym, spot_set)
                    if cands:
                        sp_sym_used, sp_mult_used = cands[0]
                if sp_sym_used:
                    cand_spot_df = _fetch_spot_1m_for_symbol(sym, sp_sym_used, sp_mult_used, years, recent_days)
                    if (
                        cand_spot_df is not None
                        and not cand_spot_df.empty
                        and (
                            sym_1m_df is None
                            or sym_1m_df.empty
                            or verify_spot_price_parity(
                                float(sym_1m_df["close"].iloc[-1]),
                                float(cand_spot_df["close"].iloc[-1]) / sp_mult_used,
                                sp_mult_used,
                            )
                        )
                    ):
                        sym_spot_df = cand_spot_df
                        sp_dir = base_output_dir / "spot_bars" / "1m"
                        write_parquet_zstd(sym_spot_df, sp_dir / f"{sym}.parquet")
                        total_spot += len(sym_spot_df)

            if include_prem:
                p_dfs = []
                with ThreadPoolExecutor(max_workers=16) as ex:
                    futs_p = [
                        ex.submit(_fetch_month_premium, sym, y, m, tk)
                        for tk in tickers
                        for y in years
                        for m in range(1, 13)
                    ]
                    for f in as_completed(futs_p):
                        res_df = f.result()
                        if res_df is not None:
                            p_dfs.append(res_df)
                if p_dfs:
                    merged_p = (
                        pd.concat(p_dfs, ignore_index=True)
                        .drop_duplicates(subset=["timestamp"], keep="last")
                        .sort_values("timestamp")
                        .reset_index(drop=True)
                    )
                    p_dir = base_output_dir / "premium_1h"
                    p_dir.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pandas(merged_p), p_dir / f"{sym}.parquet", compression="zstd")

            catalog_rows.append(
                build_universe_catalog_row(
                    symbol=sym,
                    bars_1m_df=sym_1m_df,
                    metrics_5m_df=sym_met_df,
                    spot_1m_df=sym_spot_df,
                    funding_df=sym_fnd_df,
                    spot_symbol=sp_sym_used,
                    spot_multiplier=sp_mult_used,
                    active_cutoff_ts=active_cutoff_ts,
                )
            )

        if args.sync_drive:
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    str(base_output_dir),
                    "gdrive:trading/crypto-alpha-engine/data",
                    "--transfers",
                    "12",
                    "--checkers",
                    "16",
                    "-q",
                ],
                check=False,
            )
            if clean_local and base_output_dir.exists():
                shutil.rmtree(base_output_dir, ignore_errors=True)
                base_output_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"  ☁️ Synced batch [{b_start+1}..{b_start+len(batch_syms)}/{len(symbols)}] to Google Drive & cleared local staging.",
                    flush=True,
                )

    cat_df = pd.DataFrame(catalog_rows)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "symbols_count": len(symbols),
        "symbols": symbols,
        "timeframes": timeframes,
        "years": years,
        "total_klines_bars": total_klines,
        "total_funding_settlements": total_funding,
        "total_metrics_5m_rows": total_metrics,
        "total_spot_1m_bars": total_spot,
        "spot_matched_symbols": int(cat_df["has_spot"].sum()) if not cat_df.empty else 0,
        "delisted_historical_symbols": (
            int((cat_df["status"] == "HISTORICAL_DELISTED").sum()) if not cat_df.empty else 0
        ),
    }
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if not cat_df.empty:
        write_parquet_zstd(cat_df, base_output_dir / "universe_catalog.parquet")
    (base_output_dir / "universe_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.sync_drive:
        subprocess.run(
            [
                "rclone",
                "copy",
                str(base_output_dir),
                "gdrive:trading/crypto-alpha-engine/data",
                "--transfers",
                "8",
                "-q",
            ],
            check=False,
        )
    if clean_local:
        shutil.rmtree(base_output_dir, ignore_errors=True)
    print(
        f"🎉 Complete in {manifest['elapsed_seconds']}s | Klines: {total_klines:,} | "
        f"Metrics5m: {total_metrics:,} | Spot1m: {total_spot:,} | Funding: {total_funding:,}",
        flush=True,
    )


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
    elif act == "ingest-m16":
        if not client.ensure_session(gpu=args.colab_gpu):
            raise RuntimeError(f"Failed to connect to Colab session '{args.colab_session}'")
        if not client.is_drive_mounted():
            raise RuntimeError(
                f"Google Drive is not mounted on '{args.colab_session}'. "
                f"Run `colab drivemount -s {args.colab_session} /content/drive` first."
            )
        client.upload(
            REPO_ROOT / "src/data/multimodal_vision_ingestor.py", "/content/src/data/multimodal_vision_ingestor.py"
        )
        client.upload(REPO_ROOT / "src/data/m16_pit_220_universe.json", "/content/src/data/m16_pit_220_universe.json")
        client.upload(REPO_ROOT / "src/data/ccxt_provider.py", "/content/src/data/ccxt_provider.py")
        client.upload(REPO_ROOT / "src/data/parquet_storage.py", "/content/src/data/parquet_storage.py")
        client.upload(REPO_ROOT / "src/domain/enums.py", "/content/src/domain/enums.py")
        client.upload(Path(__file__).resolve(), "/content/scripts/download_market_data.py")
        drive_out = "/content/drive/MyDrive/trading/crypto-alpha-engine/data/m16_multimodal"
        launch_cmd = (
            "import os, subprocess, sys; "
            "os.makedirs('/content/src/data', exist_ok=True); "
            "os.makedirs('/content/src/domain', exist_ok=True); "
            "log_f = open('/content/m16_colab.log', 'w', buffering=1); "
            "proc = subprocess.Popen([sys.executable, '-u', '/content/scripts/download_market_data.py', "
            f"'--mode', 'vision', '--preset', 'm16_pit220', '--sharded-multimodal', '--output-dir', '{drive_out}'], "
            "stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True); "
            "open('/content/m16_colab.pid', 'w').write(str(proc.pid)); "
            "print(f'LAUNCHED_PID={proc.pid}')"
        )
        rc, out, err = client.exec_inline(launch_cmd, timeout=60)
        print(f"[+] Dispatched M16-1 sharded ingestion on Colab ({rc}): {(out or err).strip()}", flush=True)
        seen = 0
        while True:
            time.sleep(35.0)
            poll = (
                "import os; "
                "pid = int(open('/content/m16_colab.pid').read().strip()); "
                "alive = os.path.exists(f'/proc/{pid}'); "
                "lines = open('/content/m16_colab.log').read().splitlines() if os.path.exists('/content/m16_colab.log') else []; "
                f"print(f'ALIVE={{alive}} TOTAL={{len(lines)}}'); "
                f"[print('LOG|' + ln) for ln in lines[{seen}:]]"
            )
            rc_p, out_p, _ = client.exec_inline(poll, timeout=60)
            if rc_p != 0:
                continue
            alive, done = True, False
            for ln in out_p.splitlines():
                if ln.startswith("ALIVE="):
                    p = ln.split()
                    alive = p[0].split("=")[1] == "True"
                    seen = int(p[1].split("=")[1])
                elif ln.startswith("LOG|"):
                    msg = ln[4:]
                    print(msg, flush=True)
                    if "ALL_20_SHARDS_COMPLETE!" in msg:
                        done = True
            if done or not alive:
                break
        client.stop()
        print("[+] M16-1 Colab ingestion finished and session stopped cleanly.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Crypto Market Data Ingestion & Cloud/Colab Sync CLI")
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
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days of history to fetch in --mode ccxt (default: 30)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/crypto_alpha_backtest_cache",
        help="Target directory for subhourly/vision modes.",
    )
    parser.add_argument(
        "--years", type=str, default="2023,2024,2025,2026", help="Comma-separated years for --mode vision."
    )
    parser.add_argument(
        "--include-subhourly", action="store_true", help="Include 15m klines alongside 1h in --mode vision."
    )
    parser.add_argument(
        "--no-funding",
        dest="include_funding",
        action="store_false",
        default=True,
        help="Skip 8h funding rates in --mode vision.",
    )
    parser.add_argument(
        "--sync-drive", action="store_true", help="Sync downloaded Parquet files to Google Drive via rclone."
    )
    parser.add_argument(
        "--resample-from-1m",
        action="store_true",
        help="When --timeframe 1m is used in --mode vision, also deterministically generate 5m, 15m, and 1h Parquets.",
    )
    parser.add_argument(
        "--include-premium",
        action="store_true",
        help="Also ingest 1h Spot-Perp Premium Index klines in --mode vision.",
    )
    parser.add_argument(
        "--include-metrics-5m",
        action="store_true",
        help="M16-1: Also ingest 5-minute Derivatives Positioning & Open Interest metrics (`metrics/5m/`).",
    )
    parser.add_argument(
        "--include-spot-1m",
        action="store_true",
        help="M16-1: Also ingest multiplier-adjusted 1-minute Spot klines (`spot_bars/1m/`).",
    )
    parser.add_argument(
        "--universe-plan-json",
        type=str,
        default=None,
        help="M16-1: Path to Point-in-Time survivorship-free 200+ universe JSON plan overriding --preset.",
    )
    parser.add_argument(
        "--clean-local",
        action="store_true",
        help="Stream in 6-symbol batches to Google Drive and immediately delete local staging files (0 local disk used).",
    )
    parser.add_argument(
        "--sharded-multimodal",
        action="store_true",
        help="M16-1: Stream compact zstd-9 row-group shards (`bars_1m`, `bars_1h`, `metrics_5m`, `spot_bars_1m`, `funding`).",
    )
    parser.add_argument(
        "--colab-session", type=str, default="crypto_data_ingest", help="Colab session name for --mode colab."
    )
    parser.add_argument(
        "--colab-action",
        choices=["status", "ensure", "mount", "ingest", "ingest-m16", "stop", "heartbeat"],
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
