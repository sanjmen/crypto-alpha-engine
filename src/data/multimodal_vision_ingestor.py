"""
Multi-Modal Binance Vision S3 Ingestor & Point-in-Time Universe Catalog Builder (Issue #67 / M16-1).

Provides zero-404 S3 XML key discovery, survivorship-bias-free USDS-M perpetual universe filtering,
empirically verified Spot-Perp contract multiplier mapping (`1x`, `1000x`, `1000000x`), and
deterministic schema normalization for:
  1. `bars/1m/{SYMBOL}.parquet` (+ `5m`, `15m`, `1h` resampled Futures OHLCV + Taker Flow)
  2. `metrics/5m/{SYMBOL}.parquet` (`Open Interest`, `Top-Trader vs Retail L/S`, `Taker Vol Ratio`)
  3. `spot_bars/1m/{SYMBOL}.parquet` (Multiplier-adjusted `1m` Spot OHLCV + Spot Taker Flow)
  4. `funding/{SYMBOL}.parquet` (Dynamic `1h / 2h / 4h / 8h` perpetual funding settlements)
  5. `universe_catalog.parquet` & `universe_data_manifest.json`
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gc
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

VISION_CDN_BASE = "https://data.binance.vision"
VISION_S3_LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
S3_XML_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

NON_CRYPTO_TRADFI_BASES: frozenset[str] = frozenset(
    {
        "AAOI",
        "AAPL",
        "AMD",
        "AMZN",
        "ANTHROPIC",
        "APP",
        "AXTI",
        "BE",
        "BILL",
        "BMNR",
        "BZ",
        "CBRS",
        "CHIP",
        "CL",
        "COHR",
        "COIN",
        "CONL",
        "CRCL",
        "CRWV",
        "CSOPSKHYNIX2L",
        "CXMT",
        "DELL",
        "EWZ",
        "FLNC",
        "GOOGL",
        "HOOD",
        "INTC",
        "INTW",
        "IREN",
        "KORU",
        "KSTR",
        "LITE",
        "META",
        "MINIMAX",
        "MRNA",
        "MRVL",
        "MSFT",
        "MSTR",
        "MSTU",
        "MSTZ",
        "MUU",
        "MVLL",
        "NATGAS",
        "NBIS",
        "NFLX",
        "NOW",
        "NVDA",
        "NVDL",
        "NVDS",
        "ORCL",
        "PLTR",
        "QCOM",
        "QQQ",
        "RKLB",
        "SAMSUNG",
        "SKDD",
        "SKHY",
        "SKHYNIX",
        "SKUU",
        "SMCI",
        "SNDK",
        "SNXX",
        "SOXL",
        "SOXS",
        "SPY",
        "SQQQ",
        "STXX",
        "TQQQ",
        "TSLA",
        "TSLL",
        "TSLS",
        "UNITREE",
        "USDBRL",
        "UVXY",
        "WDC",
        "XAG",
        "XAU",
        "ZHIPU",
        "ZHONGJI",
        "BTCDOM",
        "DEFI",
        "USDC",
        "FDUSD",
        "TUSD",
        "BUSD",
    }
)

# Canonical Perp rebrand stitching (ordered chronologically old -> new so latest wins on overlap)
PERP_TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "RENDERUSDT": ("RNDRUSDT", "RENDERUSDT"),
    "MATICUSDT": ("MATICUSDT", "POLUSDT"),
    "FTMUSDT": ("FTMUSDT", "SUSDT"),
}

# Explicit Spot ticker + contract multiplier overrides (`perp_price ~= spot_price * multiplier`)
SPOT_PERP_MULTIPLIER_MAP: dict[str, tuple[tuple[str, float], ...]] = {
    "1000PEPEUSDT": (("PEPEUSDT", 1000.0),),
    "1000SHIBUSDT": (("SHIBUSDT", 1000.0),),
    "1000FLOKIUSDT": (("FLOKIUSDT", 1000.0),),
    "1000BONKUSDT": (("BONKUSDT", 1000.0),),
    "1000LUNCUSDT": (("LUNCUSDT", 1000.0),),
    "1000XECUSDT": (("XECUSDT", 1000.0),),
    "1000BTTCUSDT": (("BTTCUSDT", 1000.0),),
    "1000SATSUSDT": (("1000SATSUSDT", 1.0), ("SATSUSDT", 1000.0)),
    "1000CHEEMSUSDT": (("1000CHEEMSUSDT", 1.0), ("CHEEMSUSDT", 1000.0)),
    "1000CATUSDT": (("1000CATUSDT", 1.0), ("CATUSDT", 1000.0)),
    "RENDERUSDT": (("RNDRUSDT", 1.0), ("RENDERUSDT", 1.0)),
    "MATICUSDT": (("MATICUSDT", 1.0), ("POLUSDT", 1.0)),
    "POLUSDT": (("MATICUSDT", 1.0), ("POLUSDT", 1.0)),
    "FTMUSDT": (("FTMUSDT", 1.0), ("SUSDT", 1.0)),
    "SUSDT": (("FTMUSDT", 1.0), ("SUSDT", 1.0)),
    "LUNA2USDT": (("LUNAUSDT", 1.0),),
}

KLINES_RAW_COLS = [
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
    "taker_buy_quote_volume",
    "ignore",
]

METRICS_RAW_COLS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

FUNDING_RAW_COLS = [
    "calc_time",
    "funding_interval_hours",
    "last_funding_rate",
]


def is_valid_crypto_usdt_perp(symbol: str) -> bool:
    """
    Returns True if `symbol` is a genuine crypto USDS-M linear perpetual contract,
    excluding dated delivery contracts (`_`), non-ASCII tickers, TradFi equities/ETFs/FX/commodities,
    stablecoin-vs-stablecoin pairs, and leveraged decay tokens (`UP`/`DOWN`/`BULL`/`BEAR`).
    """
    sym = symbol.strip().upper()
    if not sym.endswith("USDT") or "_" in sym or "-" in sym:
        return False
    if not sym.isascii() or not sym.isalnum():
        return False
    if sym in ("RNDRUSDT",):
        # Merged into canonical RENDERUSDT via PERP_TICKER_ALIASES
        return False
    base = sym[:-4]
    if not base or base in NON_CRYPTO_TRADFI_BASES:
        return False
    return not (len(base) > 3 and base.endswith(("UP", "DOWN", "BULL", "BEAR")))


def resolve_spot_candidates(perp_symbol: str, available_spot_symbols: set[str]) -> list[tuple[str, float]]:
    """
    Produces ordered `(spot_symbol, price_multiplier)` candidates for a given `perp_symbol`
    against `available_spot_symbols`.
    """
    sym = perp_symbol.strip().upper()
    raw_cands: list[tuple[str, float]] = []
    if sym in SPOT_PERP_MULTIPLIER_MAP:
        raw_cands.extend(SPOT_PERP_MULTIPLIER_MAP[sym])
    if sym in available_spot_symbols:
        raw_cands.append((sym, 1.0))
    if sym.startswith("1000000") and sym[7:] in available_spot_symbols:
        raw_cands.append((sym[7:], 1_000_000.0))
    if sym.startswith("1000") and sym[4:] in available_spot_symbols:
        raw_cands.append((sym[4:], 1000.0))

    deduped: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for sp_sym, mult in raw_cands:
        if sp_sym in available_spot_symbols and (sp_sym, mult) not in seen:
            seen.add((sp_sym, mult))
            deduped.append((sp_sym, float(mult)))
    return deduped


def verify_spot_price_parity(
    perp_close: float,
    spot_close: float,
    multiplier: float,
    tolerance: float = 0.25,
) -> bool:
    """Verifies that `perp_close / (spot_close * multiplier)` lies within `[1 - tolerance, 1 + tolerance]`."""
    if perp_close <= 0.0 or spot_close <= 0.0 or multiplier <= 0.0:
        return False
    ratio = perp_close / (spot_close * multiplier)
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)


def epoch_to_utc_seconds(raw_ts: pd.Series | np.ndarray) -> np.ndarray:
    """
    Normalizes Binance Vision integer timestamps into UTC epoch seconds (`int64`),
    handling seconds (`10` digits), milliseconds (`13` digits), and `>=2025` Spot microseconds (`16` digits).
    """
    arr = np.asarray(raw_ts, dtype=np.int64)
    return np.where(arr >= 10**14, arr // 10**6, np.where(arr >= 10**11, arr // 10**3, arr))


def normalize_kline_df(
    df: pd.DataFrame | None,
    symbol: str,
    price_multiplier: float = 1.0,
    spot_symbol: str | None = None,
) -> pd.DataFrame | None:
    """
    Normalizes raw Binance Vision Futures or Spot klines DataFrame into canonical UTC-seconds schema.
    When `price_multiplier != 1.0` (e.g. `PEPEUSDT` Spot -> `1000PEPEUSDT` Perp parity):
      - `open`, `high`, `low`, `close` are multiplied by `price_multiplier`
      - `volume`, `taker_buy_vol` (base units) are divided by `price_multiplier`
      - `quote_volume`, `taker_buy_quote_volume` (USDT notional) remain invariant.
    """
    if df is None or df.empty or df.shape[1] < 6:
        return None
    ts_num = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    if ts_num.empty:
        return None
    df = df.loc[ts_num.index]
    ts_sec = epoch_to_utc_seconds(ts_num.to_numpy(dtype=np.int64))
    mult = float(price_multiplier)
    inv_mult = 1.0 / mult if mult > 0.0 else 1.0

    qvol = (
        pd.to_numeric(df.iloc[:, 7], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if df.shape[1] > 7
        else np.zeros(len(df), dtype=np.float64)
    )
    trades = (
        pd.to_numeric(df.iloc[:, 8], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        if df.shape[1] > 8
        else np.zeros(len(df), dtype=np.int64)
    )
    tb_vol = (
        pd.to_numeric(df.iloc[:, 9], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) * inv_mult
        if df.shape[1] > 9
        else np.zeros(len(df), dtype=np.float64)
    )
    tb_qvol = (
        pd.to_numeric(df.iloc[:, 10], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if df.shape[1] > 10
        else np.zeros(len(df), dtype=np.float64)
    )

    cols: dict[str, Any] = {
        "timestamp": ts_sec,
        "symbol": symbol,
        "open": pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64) * mult,
        "high": pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64) * mult,
        "low": pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=np.float64) * mult,
        "close": pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy(dtype=np.float64) * mult,
        "volume": pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(dtype=np.float64) * inv_mult,
        "quote_volume": qvol,
        "trade_count": trades,
        "taker_buy_vol": tb_vol,
        "taker_buy_quote_volume": tb_qvol,
    }
    if spot_symbol is not None:
        cols["spot_symbol"] = spot_symbol
        cols["price_multiplier"] = mult

    out = pd.DataFrame(cols).dropna(subset=["open", "close"])
    return out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def normalize_metrics_5m_df(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    """
    Normalizes raw Binance Vision `metrics` (`5m` Open Interest & Whale/Retail L/S ratios)
    into monotonic, deduplicated UTC-seconds schema.
    """
    if df is None or df.empty or df.shape[1] < 4:
        return None
    raw_c0 = df["create_time"] if "create_time" in df.columns else df.iloc[:, 0]
    num_try = pd.to_numeric(raw_c0, errors="coerce")
    if num_try.notna().mean() > 0.80:
        valid = num_try.dropna()
        df = df.loc[valid.index]
        ts_sec = epoch_to_utc_seconds(valid.to_numpy(dtype=np.int64))
    else:
        dt = pd.to_datetime(raw_c0, utc=True, errors="coerce").dropna()
        if dt.empty:
            return None
        df = df.loc[dt.index]
        ts_sec = ((dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)

    def _col(name: str, idx: int) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        if df.shape[1] > idx:
            return pd.to_numeric(df.iloc[:, idx], errors="coerce").to_numpy(dtype=np.float64)
        return np.full(len(df), np.nan, dtype=np.float64)

    out = pd.DataFrame(
        {
            "timestamp": ts_sec,
            "symbol": symbol,
            "sum_open_interest": _col("sum_open_interest", 2),
            "sum_open_interest_value": _col("sum_open_interest_value", 3),
            "count_toptrader_long_short_ratio": _col("count_toptrader_long_short_ratio", 4),
            "sum_toptrader_long_short_ratio": _col("sum_toptrader_long_short_ratio", 5),
            "count_long_short_ratio": _col("count_long_short_ratio", 6),
            "sum_taker_long_short_vol_ratio": _col("sum_taker_long_short_vol_ratio", 7),
        }
    )
    out = out[out["timestamp"] > 0]
    return out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def normalize_funding_df(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    """
    Normalizes perpetual funding rate history, preserving exact dynamic settlement intervals
    (`1h / 2h / 4h / 8h`) both from `funding_interval_hours` and empirical timestamp spacing.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return None
    raw_ts = df["calc_time"] if "calc_time" in df.columns else df.iloc[:, 0]
    ts_num = pd.to_numeric(raw_ts, errors="coerce").dropna()
    if ts_num.empty:
        return None
    df = df.loc[ts_num.index]
    ts_sec = epoch_to_utc_seconds(ts_num.to_numpy(dtype=np.int64))

    if "last_funding_rate" in df.columns:
        fr = pd.to_numeric(df["last_funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        fr = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    if "funding_interval_hours" in df.columns:
        iv_raw = pd.to_numeric(df["funding_interval_hours"], errors="coerce")
    elif df.shape[1] >= 3:
        iv_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        iv_raw = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "timestamp": ts_sec,
            "symbol": symbol,
            "interval_hours_raw": iv_raw.to_numpy(dtype=np.float64),
            "funding_rate": fr,
        }
    )
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    # Infer missing interval_hours from empirical settlement delta (clamped to standard {1, 2, 4, 8})
    dt_hrs = np.diff(out["timestamp"].to_numpy(dtype=np.int64), prepend=out["timestamp"].iloc[0] - 8 * 3600) / 3600.0
    inferred_hrs = np.clip(np.rint(dt_hrs), 1, 8).astype(np.int64)
    raw_arr = out["interval_hours_raw"].to_numpy(dtype=np.float64)
    safe_raw_iv = np.rint(np.nan_to_num(raw_arr, nan=8.0)).astype(np.int64)
    final_iv = np.where(np.isfinite(raw_arr) & (raw_arr >= 1.0), safe_raw_iv, inferred_hrs)
    out["interval_hours"] = final_iv
    return out[["timestamp", "symbol", "interval_hours", "funding_rate"]]


def list_vision_s3_prefixes(prefix: str, timeout: int = 25) -> list[str]:
    """Lists common directory prefixes under `prefix` on Binance Vision S3."""
    prefixes: list[str] = []
    marker = ""
    while True:
        url = f"{VISION_S3_LIST_BASE}?delimiter=/&prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        xml_bytes = urllib.request.urlopen(url, timeout=timeout).read()
        root = ET.fromstring(xml_bytes)
        for cp in root.findall(f"{S3_XML_NS}CommonPrefixes"):
            txt = cp.findtext(f"{S3_XML_NS}Prefix")
            if txt:
                prefixes.append(txt)
        if root.findtext(f"{S3_XML_NS}IsTruncated") != "true":
            break
        marker = root.findtext(f"{S3_XML_NS}NextMarker") or (prefixes[-1] if prefixes else "")
        if not marker:
            break
    return prefixes


def list_vision_s3_keys(
    prefix: str,
    min_year: int = 2023,
    suffix: str = ".zip",
    timeout: int = 25,
) -> list[str]:
    """
    Lists exact S3 object keys under `prefix` ending in `suffix` (skipping `.CHECKSUM` files)
    for years `>= min_year` so bulk downloaders issue zero 404 requests.
    """
    keys: list[str] = []
    marker = ""
    year_tokens = tuple(f"-{y}-" for y in range(min_year, 2030))
    while True:
        url = f"{VISION_S3_LIST_BASE}?prefix={prefix}"
        if marker:
            url += f"&marker={marker}"
        xml_bytes = urllib.request.urlopen(url, timeout=timeout).read()
        root = ET.fromstring(xml_bytes)
        last_key = ""
        for ct in root.findall(f"{S3_XML_NS}Contents"):
            k = ct.findtext(f"{S3_XML_NS}Key") or ""
            last_key = k
            if k.endswith(suffix) and any(yt in k for yt in year_tokens):
                keys.append(k)
        if root.findtext(f"{S3_XML_NS}IsTruncated") != "true":
            break
        marker = root.findtext(f"{S3_XML_NS}NextMarker") or last_key
        if not marker:
            break
    return keys


def fetch_vision_zip_csv(key_or_url: str, raw_cols: list[str] | None = None, timeout: int = 25) -> pd.DataFrame | None:
    """Streams a single Binance Vision `.zip` archive in memory and parses its CSV."""
    url = key_or_url if key_or_url.startswith("http") else f"{VISION_CDN_BASE}/{key_or_url.lstrip('/')}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as probe:
                first_line = probe.readline().decode("utf-8", errors="ignore").strip()
                has_header = bool(first_line) and not first_line[0].isdigit()
            with zf.open(csv_names[0]) as f_csv:
                if has_header:
                    return pd.read_csv(f_csv)
                return pd.read_csv(f_csv, header=None, names=raw_cols)
    except Exception:
        return None


def build_universe_catalog_row(
    symbol: str,
    bars_1m_df: pd.DataFrame | None,
    metrics_5m_df: pd.DataFrame | None,
    spot_1m_df: pd.DataFrame | None,
    funding_df: pd.DataFrame | None,
    spot_symbol: str | None = None,
    spot_multiplier: float = 1.0,
    active_cutoff_ts: int | None = None,
) -> dict[str, Any]:
    """Builds a single point-in-time metadata row for `universe_catalog.parquet`."""
    has_bars = bars_1m_df is not None and not bars_1m_df.empty
    first_ts = int(bars_1m_df["timestamp"].iloc[0]) if has_bars else 0
    last_ts = int(bars_1m_df["timestamp"].iloc[-1]) if has_bars else 0
    n_bars = int(len(bars_1m_df)) if has_bars else 0

    med_daily_qvol = 0.0
    if has_bars and "quote_volume" in bars_1m_df.columns:
        day_bucket = bars_1m_df["timestamp"].to_numpy(dtype=np.int64) // 86400
        daily_sums = pd.Series(bars_1m_df["quote_volume"].to_numpy(dtype=np.float64)).groupby(day_bucket).sum()
        if not daily_sums.empty:
            med_daily_qvol = float(daily_sums.median())

    has_fnd = funding_df is not None and not funding_df.empty
    min_fnd_iv = int(funding_df["interval_hours"].min()) if has_fnd and "interval_hours" in funding_df.columns else 8

    is_active = True
    if active_cutoff_ts is not None and last_ts > 0:
        is_active = bool(last_ts >= active_cutoff_ts)

    return {
        "symbol": symbol,
        "status": "TRADING" if is_active else "HISTORICAL_DELISTED",
        "onboard_timestamp": first_ts,
        "first_bar_ts": first_ts,
        "last_bar_ts": last_ts,
        "bars_1m_count": n_bars,
        "metrics_5m_count": int(len(metrics_5m_df)) if (metrics_5m_df is not None) else 0,
        "spot_1m_count": int(len(spot_1m_df)) if (spot_1m_df is not None) else 0,
        "funding_count": int(len(funding_df)) if has_fnd else 0,
        "has_spot": bool(spot_1m_df is not None and not spot_1m_df.empty),
        "spot_symbol": spot_symbol or "",
        "spot_multiplier": float(spot_multiplier),
        "min_funding_interval_hours": min_fnd_iv,
        "median_daily_quote_vol_usd": round(med_daily_qvol, 2),
    }


def write_parquet_zstd(df: pd.DataFrame, out_path: Path) -> None:
    """Writes `df` to `out_path` using Zstandard compression."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path, compression="zstd")


def write_symbol_rowgroup_shard(symbol_dfs: dict[str, pd.DataFrame], out_path: Path) -> int:
    """
    Writes a dictionary of `{symbol: df}` into a single Zstandard-compressed Parquet shard
    with **one dedicated Row Group per symbol**. This reduces Google Drive file-creation API calls
    by 10x-20x while preserving O(1) single-symbol predicate pushdown (`filters=[('symbol', '==', sym)]`).
    Returns total rows written.
    """
    valid_items = [(sym, df) for sym, df in sorted(symbol_dfs.items()) if df is not None and not df.empty]
    if not valid_items:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_tbl = pa.Table.from_pandas(valid_items[0][1], preserve_index=False)
    total_rows = 0
    with pq.ParquetWriter(out_path, first_tbl.schema, compression="zstd", write_statistics=True) as writer:
        for _sym, df in valid_items:
            tbl = pa.Table.from_pandas(df, schema=first_tbl.schema, preserve_index=False)
            writer.write_table(tbl, row_group_size=max(1, len(df)))
            total_rows += len(df)
    return total_rows


def read_symbol_from_shard(shard_path: Path, symbol: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Reads a single `symbol` from a row-group-partitioned Parquet shard via predicate pushdown."""
    tbl = pq.read_table(shard_path, columns=columns, filters=[("symbol", "==", symbol)])
    return tbl.to_pandas()


def compact_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preserves exact `int64` timestamp & `float64` close (for sub-bps Basis accuracy), `float32` for OHLV/flow."""
    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(dtype=np.int64),
            "symbol": df["symbol"],
            "open": df["open"].to_numpy(dtype=np.float32),
            "high": df["high"].to_numpy(dtype=np.float32),
            "low": df["low"].to_numpy(dtype=np.float32),
            "close": df["close"].to_numpy(dtype=np.float64),
            "volume": df["volume"].to_numpy(dtype=np.float32),
            "quote_volume": df["quote_volume"].to_numpy(dtype=np.float32),
            "trade_count": df["trade_count"].to_numpy(dtype=np.int32),
            "taker_buy_vol": df["taker_buy_vol"].to_numpy(dtype=np.float32),
            "taker_buy_quote_volume": df["taker_buy_quote_volume"].to_numpy(dtype=np.float32),
        }
    )
    if "spot_symbol" in df.columns:
        out["spot_symbol"] = df["spot_symbol"]
        out["price_multiplier"] = df["price_multiplier"].to_numpy(dtype=np.float32)
    return out


def compact_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compacts `5m` Derivatives Open Interest & Whale/Retail L/S ratios to `float32`."""
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(dtype=np.int64),
            "symbol": df["symbol"],
            "sum_open_interest": df["sum_open_interest"].to_numpy(dtype=np.float32),
            "sum_open_interest_value": df["sum_open_interest_value"].to_numpy(dtype=np.float32),
            "count_toptrader_long_short_ratio": df["count_toptrader_long_short_ratio"].to_numpy(dtype=np.float32),
            "sum_toptrader_long_short_ratio": df["sum_toptrader_long_short_ratio"].to_numpy(dtype=np.float32),
            "count_long_short_ratio": df["count_long_short_ratio"].to_numpy(dtype=np.float32),
            "sum_taker_long_short_vol_ratio": df["sum_taker_long_short_vol_ratio"].to_numpy(dtype=np.float32),
        }
    )


def resample_1m_to_1h(df_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Deterministically resamples normalized `1m` bars into `1h` bars (`closed='left', label='left'`)."""
    dt_idx = pd.to_datetime(df_1m["timestamp"], unit="s", utc=True)
    res = (
        df_1m.set_index(dt_idx)
        .resample("1h", label="left", closed="left")
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
    res["timestamp"] = ((res.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(np.int64)
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


def write_compact_rowgroup_shard(symbol_dfs: dict[str, pd.DataFrame], out_path: Path) -> int:
    """
    Writes `{symbol: df}` into a single `zstd-9` + `byte_stream_split` Parquet shard with
    one dedicated Row Group per symbol for O(1) single-symbol predicate pushdown.
    """
    valid_items = [(sym, df) for sym, df in sorted(symbol_dfs.items()) if df is not None and not df.empty]
    if not valid_items:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_df = valid_items[0][1]
    float_cols = [c for c, dt in first_df.dtypes.items() if dt in (np.float32, np.float64, "float32", "float64")]
    first_tbl = pa.Table.from_pandas(first_df, preserve_index=False)
    total_rows = 0
    with pq.ParquetWriter(
        out_path,
        first_tbl.schema,
        compression="zstd",
        compression_level=9,
        use_byte_stream_split=float_cols,
        write_statistics=True,
    ) as writer:
        for _sym, df in valid_items:
            tbl = pa.Table.from_pandas(df, schema=first_tbl.schema, preserve_index=False)
            writer.write_table(tbl, row_group_size=max(1, len(df)))
            total_rows += len(df)
    return total_rows


def load_default_pit_220_plan(plan_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Loads the canonical Point-in-Time survivorship-free 220-symbol USDS-M universe plan."""
    target = plan_path or (Path(__file__).resolve().parent / "m16_pit_220_universe.json")
    return json.loads(target.read_text())


def _make_pooled_session(pool_size: int = 96) -> requests.Session:
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=2)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "Mozilla/5.0"})
    return sess


def _fetch_pooled_zip_csv(
    session: requests.Session,
    url: str,
    raw_cols: list[str] | None = None,
    retries: int = 5,
) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=22)
            if resp.status_code == 404:
                return None
            if resp.status_code != 200:
                time.sleep(0.5 * (attempt + 1))
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csvs = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csvs:
                    return None
                with zf.open(csvs[0]) as probe:
                    first_line = probe.readline().decode("utf-8", errors="ignore").strip()
                    has_header = bool(first_line) and not first_line[0].isdigit()
                with zf.open(csvs[0]) as f_csv:
                    if has_header:
                        return pd.read_csv(f_csv)
                    return pd.read_csv(f_csv, header=None, names=raw_cols)
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))
    return None


def fetch_gap_free_1m_klines(
    market_prefix: str,
    tickers: tuple[str, ...],
    symbol: str,
    all_yms: list[tuple[int, int]],
    recent_days: list[str],
    session: requests.Session | None = None,
    price_multiplier: float = 1.0,
    spot_symbol: str | None = None,
) -> pd.DataFrame | None:
    """
    Fetches monthly + daily `1m` klines and heals any dropped interior month between
    the asset's first and last active month (`0` interior gaps).
    """
    sess = session or _make_pooled_session(32)
    ym_results: dict[tuple[str, int, int], pd.DataFrame] = {}
    daily_parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        ym_futs = {
            pool.submit(
                _fetch_pooled_zip_csv,
                sess,
                f"{VISION_CDN_BASE}/data/{market_prefix}/monthly/klines/{tk}/1m/{tk}-1m-{y}-{m:02d}.zip",
                KLINES_RAW_COLS,
            ): (tk, y, m)
            for tk in tickers
            for (y, m) in all_yms
        }
        day_futs = [
            pool.submit(
                _fetch_pooled_zip_csv,
                sess,
                f"{VISION_CDN_BASE}/data/{market_prefix}/daily/klines/{tk}/1m/{tk}-1m-{ds}.zip",
                KLINES_RAW_COLS,
            )
            for tk in tickers
            for ds in recent_days
        ]
        for fut, key in ym_futs.items():
            df = fut.result()
            if df is not None and not df.empty:
                ym_results[key] = df
        for fut in as_completed(day_futs):
            df = fut.result()
            if df is not None and not df.empty:
                daily_parts.append(df)

    covered_yms = {(y, m) for (_tk, y, m) in ym_results}
    if covered_yms:
        min_ym = min(covered_yms)
        max_ym = max(covered_yms)
        missing_yms = [ym for ym in all_yms if min_ym <= ym <= max_ym and ym not in covered_yms]
        for y, m in missing_yms:
            for tk in tickers:
                healed = _fetch_pooled_zip_csv(
                    sess,
                    f"{VISION_CDN_BASE}/data/{market_prefix}/monthly/klines/{tk}/1m/{tk}-1m-{y}-{m:02d}.zip",
                    KLINES_RAW_COLS,
                    retries=6,
                )
                if healed is not None and not healed.empty:
                    ym_results[(tk, y, m)] = healed
                    break

    all_parts = list(ym_results.values()) + daily_parts
    if not all_parts:
        return None
    return normalize_kline_df(
        pd.concat(all_parts, ignore_index=True),
        symbol=symbol,
        price_multiplier=price_multiplier,
        spot_symbol=spot_symbol,
    )


def stream_multimodal_shards_to_directory(
    output_dir: Path,
    universe_plan: dict[str, dict[str, Any]] | None = None,
    years: list[int] | None = None,
    shard_size: int = 11,
    symbol_workers: int = 4,
    rclone_dest: str | None = None,
    stage_base: Path = Path("/tmp/m16_multimodal_stage"),
) -> dict[str, Any]:
    """
    Streams the Point-in-Time survivorship-free multi-modal dataset (`bars_1m`, `bars_1h`,
    `metrics_5m`, `spot_bars_1m`, `funding`, `universe_catalog.parquet`, `universe_data_manifest.json`)
    as resumable `zstd-9` row-group shards either directly to `output_dir` (e.g. mounted Google Drive
    on Colab `/content/drive/MyDrive/...`) or via `rclone move` to `rclone_dest` with `0` local residue.
    """
    plan = universe_plan or load_default_pit_220_plan()
    yr_list = sorted(years or [2023, 2024, 2025, 2026])
    now_utc = datetime.now(timezone.utc)
    all_yms = [(y, m) for y in yr_list for m in range(1, 13) if (y, m) < (now_utc.year, now_utc.month)]
    recent_days = [date(now_utc.year, now_utc.month, d).isoformat() for d in range(1, now_utc.day)]
    active_cutoff_ts = int((now_utc - timedelta(days=45)).timestamp())
    sess = _make_pooled_session(96)

    all_syms = sorted(plan.keys())
    shards = [
        (idx, f"shard_{idx:02d}", all_syms[s_i : s_i + shard_size])
        for idx, s_i in enumerate(range(0, len(all_syms), shard_size))
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_shards_m16.json"
    ckpt: dict[str, Any] = {"completed_shards": {}, "catalog_rows": {}}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())

    pending = [sh for sh in shards if sh[1] not in ckpt["completed_shards"]]
    print(
        f"🚀 M16-1 Sharded Multi-Modal Streamer | Universe: {len(all_syms)} | "
        f"Done: {len(ckpt['completed_shards'])}/{len(shards)} | Pending: {len(pending)}",
        flush=True,
    )

    def _ingest_one(
        sym: str, info: dict[str, Any], shard_tag: str
    ) -> tuple[str, dict[str, pd.DataFrame | None], dict[str, Any]]:
        t_s = time.time()
        tickers = PERP_TICKER_ALIASES.get(sym, (sym,))
        b1m = fetch_gap_free_1m_klines("futures/um", tickers, sym, all_yms, recent_days, session=sess)
        b1h: pd.DataFrame | None = None
        if b1m is not None and not b1m.empty:
            b1h = compact_kline_df(resample_1m_to_1h(b1m, sym))
            min_dt = max(
                date(yr_list[0], 1, 1),
                datetime.fromtimestamp(int(b1m["timestamp"].iloc[0]), tz=timezone.utc).date(),
            )
            max_dt = min(
                (now_utc - timedelta(days=1)).date(),
                datetime.fromtimestamp(int(b1m["timestamp"].iloc[-1]), tz=timezone.utc).date(),
            )
            b1m = compact_kline_df(b1m)
        else:
            min_dt = date(yr_list[0], 1, 1)
            max_dt = (now_utc - timedelta(days=1)).date()

        act_yms = [(y, m) for (y, m) in all_yms if (min_dt.year, min_dt.month) <= (y, m) <= (max_dt.year, max_dt.month)]
        f_urls = [
            f"{VISION_CDN_BASE}/data/futures/um/monthly/fundingRate/{tk}/{tk}-fundingRate-{y}-{m:02d}.zip"
            for tk in tickers
            for (y, m) in act_yms
        ]
        f_parts: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=12) as p_f:
            for fut in as_completed([p_f.submit(_fetch_pooled_zip_csv, sess, u, FUNDING_RAW_COLS) for u in f_urls]):
                d_f = fut.result()
                if d_f is not None and not d_f.empty:
                    f_parts.append(d_f)
        fnd = normalize_funding_df(pd.concat(f_parts, ignore_index=True), symbol=sym) if f_parts else None

        m_dates: list[str] = []
        cur_d = min_dt
        while cur_d <= max_dt:
            m_dates.append(cur_d.isoformat())
            cur_d += timedelta(days=1)
        m_urls = [
            f"{VISION_CDN_BASE}/data/futures/um/daily/metrics/{tk}/{tk}-metrics-{ds}.zip"
            for tk in tickers
            for ds in m_dates
        ]
        m_parts: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=32) as p_m:
            for fut in as_completed([p_m.submit(_fetch_pooled_zip_csv, sess, u, METRICS_RAW_COLS) for u in m_urls]):
                d_m = fut.result()
                if d_m is not None and not d_m.empty:
                    m_parts.append(d_m)
        m_raw = normalize_metrics_5m_df(pd.concat(m_parts, ignore_index=True), symbol=sym) if m_parts else None
        met = compact_metrics_df(m_raw) if (m_raw is not None and not m_raw.empty) else None

        sp_sym = info.get("spot_symbol")
        sp_mult = float(info.get("spot_multiplier") or 1.0)
        sp_df: pd.DataFrame | None = None
        if sp_sym:
            sp_tks = (
                PERP_TICKER_ALIASES.get(sym, (sp_sym,)) if sp_mult == 1.0 and sym in PERP_TICKER_ALIASES else (sp_sym,)
            )
            cand_sp = fetch_gap_free_1m_klines(
                "spot",
                sp_tks,
                sym,
                all_yms,
                recent_days,
                session=sess,
                price_multiplier=sp_mult,
                spot_symbol=sp_sym,
            )
            if cand_sp is not None and not cand_sp.empty:
                parity_ok = True
                if b1m is not None and not b1m.empty:
                    common = b1m[["timestamp", "close"]].merge(
                        cand_sp[["timestamp", "close"]], on="timestamp", suffixes=("_p", "_s")
                    )
                    if not common.empty:
                        ratio = float((common["close_p"] / common["close_s"]).median())
                        parity_ok = 0.75 <= ratio <= 1.30
                if parity_ok:
                    sp_df = compact_kline_df(cand_sp)

        row = build_universe_catalog_row(
            symbol=sym,
            bars_1m_df=b1m,
            metrics_5m_df=met,
            spot_1m_df=sp_df,
            funding_df=fnd,
            spot_symbol=sp_sym,
            spot_multiplier=sp_mult,
            active_cutoff_ts=active_cutoff_ts,
        )
        row["shard_id"] = shard_tag
        row["ingest_sec"] = round(time.time() - t_s, 1)
        return sym, {"bars_1m": b1m, "bars_1h": b1h, "metrics_5m": met, "spot_bars_1m": sp_df, "funding": fnd}, row

    def _flush_stage(stage_dir: Path) -> float:
        t_up = time.time()
        if rclone_dest:
            subprocess.run(
                [
                    "rclone",
                    "move",
                    str(stage_dir),
                    rclone_dest,
                    "--transfers",
                    "3",
                    "--checkers",
                    "3",
                    "--drive-chunk-size",
                    "512M",
                    "--no-traverse",
                    "--delete-empty-src-dirs",
                ],
                check=True,
            )
        else:
            for src_f in sorted(stage_dir.rglob("*.parquet")) + sorted(stage_dir.rglob("*.json")):
                rel = src_f.relative_to(stage_dir)
                dst_f = output_dir / rel
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                tmp_f = dst_f.with_suffix(dst_f.suffix + ".tmp")
                shutil.copy2(src_f, tmp_f)
                os.replace(tmp_f, dst_f)
            if hasattr(os, "sync"):
                os.sync()
        shutil.rmtree(stage_dir, ignore_errors=True)
        return time.time() - t_up

    try:
        for _idx, shard_tag, sh_syms in pending:
            t0 = time.time()
            stage_dir = stage_base / shard_tag
            shutil.rmtree(stage_dir, ignore_errors=True)
            stage_dir.mkdir(parents=True, exist_ok=True)
            buckets: dict[str, dict[str, pd.DataFrame]] = {
                "bars_1m": {},
                "bars_1h": {},
                "metrics_5m": {},
                "spot_bars_1m": {},
                "funding": {},
            }
            sh_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=symbol_workers) as pool:
                futs = {pool.submit(_ingest_one, s, plan[s], shard_tag): s for s in sh_syms}
                for fut in as_completed(futs):
                    sym, frames, row = fut.result()
                    sh_rows.append(row)
                    for k_mod, df_mod in frames.items():
                        if df_mod is not None and not df_mod.empty:
                            buckets[k_mod][sym] = df_mod
                    print(
                        f"  [{shard_tag}] ✓ {sym:<14} | 1m:{row['bars_1m_count']:>9,} | "
                        f"Met5m:{row['metrics_5m_count']:>7,} | Spot1m:{row['spot_1m_count']:>9,} | "
                        f"Fnd:{row['funding_count']:>5,} (minIv={row['min_funding_interval_hours']}h) | {row['ingest_sec']}s",
                        flush=True,
                    )

            total_mb = 0.0
            for mod_name, sym_dict in buckets.items():
                if sym_dict:
                    out_f = stage_dir / mod_name / f"{shard_tag}.parquet"
                    write_compact_rowgroup_shard(sym_dict, out_f)
                    total_mb += out_f.stat().st_size / (1024 * 1024)
            del buckets
            gc.collect()

            prep_sec = time.time() - t0
            up_sec = _flush_stage(stage_dir)
            for r in sh_rows:
                ckpt["catalog_rows"][r["symbol"]] = r
            ckpt["completed_shards"][shard_tag] = {
                "symbols": sh_syms,
                "size_mb": round(total_mb, 2),
                "upload_sec": round(up_sec, 2),
            }
            ckpt_path.write_text(json.dumps(ckpt, indent=2))
            print(
                f"☁️ [{shard_tag}] FLUSHED ({len(ckpt['completed_shards'])}/{len(shards)}) | "
                f"{total_mb:.1f} MB | prep={prep_sec:.1f}s | write={up_sec:.1f}s",
                flush=True,
            )

        manifest: dict[str, Any] = {}
        if len(ckpt["completed_shards"]) == len(shards):
            fin_dir = stage_base / "final_catalog"
            fin_dir.mkdir(parents=True, exist_ok=True)
            cat_df = pd.DataFrame([ckpt["catalog_rows"][s] for s in all_syms])
            write_parquet_zstd(cat_df, fin_dir / "universe_catalog.parquet")
            manifest = {
                "issue": "M16-1 (#67)",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "target_root": rclone_dest or str(output_dir),
                "num_shards": len(shards),
                "universe_size": len(cat_df),
                "active_2026_symbols": int((cat_df["status"] == "TRADING").sum()),
                "delisted_historical_symbols": int((cat_df["status"] == "HISTORICAL_DELISTED").sum()),
                "spot_matched_symbols": int(cat_df["has_spot"].sum()),
                "shortened_funding_interval_symbols": int((cat_df["min_funding_interval_hours"] < 8).sum()),
                "total_futures_1m_bars": int(cat_df["bars_1m_count"].sum()),
                "total_metrics_5m_rows": int(cat_df["metrics_5m_count"].sum()),
                "total_spot_1m_bars": int(cat_df["spot_1m_count"].sum()),
                "total_funding_rows": int(cat_df["funding_count"].sum()),
                "total_compressed_mb": round(sum(v["size_mb"] for v in ckpt["completed_shards"].values()), 2),
                "local_storage_bytes": 0,
            }
            (fin_dir / "universe_data_manifest.json").write_text(json.dumps(manifest, indent=2))
            _flush_stage(fin_dir)
            print("🎉 ALL_20_SHARDS_COMPLETE!", flush=True)
        return manifest
    finally:
        shutil.rmtree(stage_base, ignore_errors=True)


def fetch_and_assemble_symbol_metrics_5m(
    symbol: str,
    min_year: int = 2023,
    max_workers: int = 32,
) -> pd.DataFrame | None:
    """Lists exact S3 daily `metrics` archives for `symbol` (+ aliases) and merges into one `5m` DataFrame."""
    tickers = PERP_TICKER_ALIASES.get(symbol, (symbol,))
    all_keys: list[str] = []
    for tk in tickers:
        all_keys.extend(list_vision_s3_keys(f"data/futures/um/daily/metrics/{tk}/", min_year=min_year))
    if not all_keys:
        return None
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(fetch_vision_zip_csv, k, METRICS_RAW_COLS) for k in all_keys]
        for fut in as_completed(futs):
            raw_df = fut.result()
            if raw_df is not None and not raw_df.empty:
                parts.append(raw_df)
    if not parts:
        return None
    return normalize_metrics_5m_df(pd.concat(parts, ignore_index=True), symbol=symbol)
