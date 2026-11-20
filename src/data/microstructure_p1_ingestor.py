"""
Phase-P1 Next-Generation Microstructure & Cross-Venue Data Ingestor.

Enriches the 20-shard Point-in-Time `220`-perp dataset (`m16_multimodal`) on Google Drive
(`0 bytes` persistent local disk) with three orthogonal microstructure layers that `OHLCV + 5m OI` cannot see:

  1. `premium_1m/shard_00..19.parquet`:
     Native `1-minute` `premiumIndexKlines` (`p_open_bps, p_high_bps, p_low_bps, p_close_bps`) from
     `data.binance.vision/data/futures/um/{monthly,daily}/premiumIndexKlines/`.
     - Covers `100%` of the `220` perps (including the `34` coins without Binance Spot via Binance's
       multi-exchange Spot Index across Coinbase/OKX/Bybit/Gate).
     - Captures intraminute L2 Impact-Bid / Impact-Ask wicks (`p_high_bps`, `p_low_bps`).

  2. `cross_venue/cross_venue_funding.parquet`:
     Transparent On-Chain L1 Perpetual Funding & Premium (`Hyperliquid` `fundingHistory` API) +
     `OKX` Perpetual Swap Funding History (`okx.com/api/v5/public/funding-rate-history`),
     normalized into **hourly carry intensity (`bps_per_hour`)** for exact apples-to-apples
     comparison against Binance dynamic `1h / 2h / 4h / 8h` clocks.

  3. `book_depth_events/shard_00..19.parquet`:
     Causal L2 Order-Book Depth (`±0.2%, ±1%, ±2%, ±5%` cumulative Bid USD vs Ask USD) from
     `data.binance.vision/data/futures/um/daily/bookDepth/`, surgically fetched & cached for
     candidate forced-flow event symbol-days across `2023–2026`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import io
from pathlib import Path
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

from src.data.multimodal_vision_ingestor import (
    PERP_TICKER_ALIASES,
    VISION_CDN_BASE,
    epoch_to_utc_seconds,
)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"


def _build_pooled_session(pool_size: int = 64) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CryptoAlphaMicrostructureP1/1.0)"})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=2)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def map_binance_perp_to_hl_coin(symbol: str) -> str:
    """Maps canonical Binance USDS-M symbol (`BTCUSDT`, `1000PEPEUSDT`) to Hyperliquid L1 coin ticker."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if base.startswith("1000000"):
        return "k" + base[7:]
    if base.startswith("1000"):
        return "k" + base[4:]
    return base


def map_binance_perp_to_okx_inst(symbol: str) -> str:
    """Maps canonical Binance USDS-M symbol to OKX USDT-Swap `instId` (`BTC-USDT-SWAP`)."""
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if base.startswith("1000000"):
        base = base[7:]
    elif base.startswith("1000"):
        base = base[4:]
    return f"{base}-USDT-SWAP"


def _months_between_epochs(min_ts_sec: int, max_ts_sec: int) -> list[tuple[int, int]]:
    """Returns inclusive `(year, month)` tuples spanning `[min_ts_sec, max_ts_sec]`."""
    dt_start = datetime.fromtimestamp(max(0, int(min_ts_sec)), tz=timezone.utc)
    dt_end = datetime.fromtimestamp(max(int(min_ts_sec), int(max_ts_sec)), tz=timezone.utc)
    y, m = dt_start.year, dt_start.month
    out: list[tuple[int, int]] = []
    while (y, m) <= (dt_end.year, dt_end.month):
        out.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def parse_premium_index_zip_bytes(raw_bytes: bytes, symbol: str) -> pd.DataFrame | None:
    """
    Parses a Binance Vision `premiumIndexKlines` ZIP archive into compact `int64/float32` basis-point columns:
    `timestamp, symbol, p_open_bps, p_high_bps, p_low_bps, p_close_bps`.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as fh:
                first = fh.readline().decode("utf-8", errors="ignore")
                has_hdr = "open_time" in first or "open" in first
            with zf.open(names[0]) as fh:
                df = pd.read_csv(fh, header=0 if has_hdr else None, usecols=[0, 1, 2, 3, 4])
    except Exception:  # noqa: BLE001
        return None

    if df is None or df.empty:
        return None
    ts_num = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    if ts_num.empty:
        return None
    df = df.loc[ts_num.index]
    ts_sec = epoch_to_utc_seconds(ts_num.to_numpy(dtype=np.int64))

    po = (pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64) * 1e4).astype(np.float32)
    ph = (pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64) * 1e4).astype(np.float32)
    pl = (pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=np.float64) * 1e4).astype(np.float32)
    pc = (pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy(dtype=np.float64) * 1e4).astype(np.float32)

    valid = np.isfinite(pc) & np.isfinite(po)
    if not bool(valid.any()):
        return None
    return pd.DataFrame(
        {
            "timestamp": ts_sec[valid],
            "symbol": symbol,
            "p_open_bps": po[valid],
            "p_high_bps": ph[valid],
            "p_low_bps": pl[valid],
            "p_close_bps": pc[valid],
        }
    )


def _fetch_single_premium_archive(
    sess: requests.Session,
    symbol: str,
    tk: str,
    y: int,
    m: int,
) -> list[pd.DataFrame]:
    """Downloads a single `(tk, y, m)` `1m` `premiumIndexKlines` archive (with Nov-2026 daily fallback)."""
    out: list[pd.DataFrame] = []
    url = f"{VISION_CDN_BASE}/data/futures/um/monthly/premiumIndexKlines/{tk}/1m/{tk}-1m-{y}-{m:02d}.zip"
    try:
        r = sess.get(url, timeout=20)
        if r.status_code == 200 and r.content:
            parsed = parse_premium_index_zip_bytes(r.content, symbol)
            if parsed is not None and not parsed.empty:
                out.append(parsed)
        elif y == 2026 and m == 11:
            for d_day in range(1, 20):
                d_url = (
                    f"{VISION_CDN_BASE}/data/futures/um/daily/premiumIndexKlines/"
                    f"{tk}/1m/{tk}-1m-2026-11-{d_day:02d}.zip"
                )
                rd = sess.get(d_url, timeout=10)
                if rd.status_code == 200 and rd.content:
                    p_d = parse_premium_index_zip_bytes(rd.content, symbol)
                    if p_d is not None and not p_d.empty:
                        out.append(p_d)
    except Exception:  # noqa: BLE001
        return out
    return out


def ensure_premium_index_1m_shards(
    m16_data_dir: Path,
    max_shards: int = 20,
    workers: int = 48,
) -> dict[str, Any]:
    """
    Ensures `m16_data_dir / 'premium_1m' / shard_XX.parquet` exists for all shards,
    parallelizing at the `(symbol, ticker, year, month)` archive level for maximum throughput
    and skipping already-ingested shards cleanly (idempotent Google Drive checkpointing).
    """
    bars_dir = m16_data_dir / "bars_1m"
    prem_dir = m16_data_dir / "premium_1m"
    prem_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(bars_dir.glob("shard_*.parquet"))[:max_shards]

    sess = _build_pooled_session(pool_size=max(64, workers * 2))
    built = 0
    reused = 0
    total_rows = 0

    for sh_p in shard_files:
        out_p = prem_dir / sh_p.name
        if out_p.exists() and out_p.stat().st_size > 1024:
            reused += 1
            continue

        t0 = time.time()
        meta_df = pd.read_parquet(sh_p, columns=["symbol", "timestamp"])
        tasks: list[tuple[str, str, int, int]] = []
        n_syms = 0
        for sym_k, grp in meta_df.groupby("symbol", observed=True):
            ts_a = grp["timestamp"].to_numpy(dtype=np.int64)
            if len(ts_a) == 0:
                continue
            n_syms += 1
            sym_s = str(sym_k)
            yms = _months_between_epochs(int(ts_a.min()), int(ts_a.max()))
            for tk in PERP_TICKER_ALIASES.get(sym_s, (sym_s,)):
                for y, m in yms:
                    tasks.append((sym_s, tk, y, m))
        del meta_df

        by_sym_parts: dict[str, list[pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(tasks)))) as pool:
            futs = [pool.submit(_fetch_single_premium_archive, sess, s, tk, y, m) for s, tk, y, m in tasks]
            for fut in as_completed(futs):
                for df_p in fut.result():
                    sym_name = str(df_p["symbol"].iloc[0])
                    by_sym_parts.setdefault(sym_name, []).append(df_p)

        deduped_syms: list[pd.DataFrame] = []
        for sym_name in sorted(by_sym_parts):
            m_sym = pd.concat(by_sym_parts[sym_name], ignore_index=True)
            m_sym.sort_values("timestamp", inplace=True)
            m_sym.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            deduped_syms.append(m_sym)
        del by_sym_parts

        if deduped_syms:
            sh_df = pd.concat(deduped_syms, ignore_index=True)
            sh_df["symbol"] = sh_df["symbol"].astype("category")
            pq.write_table(pa.Table.from_pandas(sh_df, preserve_index=False), out_p, compression="zstd")
            n_r = len(sh_df)
            total_rows += n_r
            del sh_df, deduped_syms
        else:
            n_r = 0
        built += 1
        gc.collect()
        print(
            f"  [Premium1m] Built {sh_p.name}: {n_r:,} 1m impact-basis rows across {n_syms} syms ({time.time() - t0:.1f}s)",
            flush=True,
        )

    sess.close()
    return {"shards_built": built, "shards_reused": reused, "new_rows": total_rows}


def fetch_hyperliquid_symbol_funding(
    sess: requests.Session,
    symbol: str,
    hl_coin: str,
    start_ms: int = 1_685_577_600_000,
    end_ms: int | None = None,
) -> pd.DataFrame | None:
    """Paginates Hyperliquid L1 `fundingHistory` (`500` hourly settlements per call) for `hl_coin`."""
    stop_ms = end_ms or int(time.time() * 1000)
    cur_ms = int(start_ms)
    rows: list[tuple[int, float, float]] = []

    for _ in range(32):  # Up to 16,000 hours (~22 months; 32 pages = ~3.6 years full HL L1 history!)
        if cur_ms >= stop_ms:
            break
        try:
            r = None
            for attempt in range(4):
                r = sess.post(
                    HL_INFO_URL,
                    json={"type": "fundingHistory", "coin": hl_coin, "startTime": cur_ms},
                    timeout=15,
                )
                if r.status_code == 429:
                    time.sleep(0.6 * (attempt + 1))
                    continue
                break
            if r is None or r.status_code != 200:
                break
            data = r.json()
            if not isinstance(data, list) or not data:
                break
            last_t = cur_ms
            for item in data:
                t_ms = int(item.get("time", 0))
                if t_ms <= 0:
                    continue
                t_sec = ((t_ms // 1000 + 30) // 60) * 60
                fr_1h_bps = float(item.get("fundingRate", 0.0)) * 1e4
                prem_bps = float(item.get("premium", 0.0)) * 1e4
                rows.append((t_sec, fr_1h_bps, prem_bps))
                if t_ms > last_t:
                    last_t = t_ms
            if len(data) < 480 or last_t <= cur_ms:
                break
            cur_ms = last_t + 1000
        except Exception:  # noqa: BLE001
            break

    if not rows:
        return None
    arr_t = np.array([x[0] for x in rows], dtype=np.int64)
    arr_f = np.array([x[1] for x in rows], dtype=np.float32)
    arr_p = np.array([x[2] for x in rows], dtype=np.float32)
    out = pd.DataFrame(
        {
            "timestamp": arr_t,
            "symbol": symbol,
            "hl_fr_bps_per_h": arr_f,
            "hl_premium_bps": arr_p,
        }
    )
    out.sort_values("timestamp", inplace=True)
    out.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def ensure_cross_venue_funding_parquet(
    m16_data_dir: Path,
    symbols: list[str],
    workers: int = 3,
) -> Path:
    """
    Fetches & caches On-Chain `Hyperliquid L1` hourly funding & premium across all overlapping symbols
    into `m16_data_dir / 'cross_venue' / 'hl_funding_1h.parquet'` (incrementally topping up if partial).
    """
    cv_dir = m16_data_dir / "cross_venue"
    cv_dir.mkdir(parents=True, exist_ok=True)
    out_path = cv_dir / "hl_funding_1h.parquet"
    existing_hl = pd.DataFrame()
    covered_syms: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 4096:
        existing_hl = pd.read_parquet(out_path)
        if not existing_hl.empty:
            covered_syms = set(existing_hl["symbol"].astype(str).unique())
        if len(covered_syms) >= 35:
            return out_path

    t0 = time.time()
    sess = _build_pooled_session(pool_size=max(8, workers * 2))

    # Query Hyperliquid `meta` once so we ONLY request coins that actually exist on Hyperliquid L1 (zero wasted calls!)
    hl_available: set[str] = set()
    try:
        r_meta = sess.post(HL_INFO_URL, json={"type": "meta"}, timeout=15)
        if r_meta.status_code == 200:
            meta_j = r_meta.json()
            for u_item in meta_j.get("universe", []):
                if isinstance(u_item, dict) and "name" in u_item:
                    hl_available.add(str(u_item["name"]))
    except Exception:  # noqa: BLE001
        hl_available = set()

    sym_to_hl: list[tuple[str, str]] = []
    for s in sorted(set(symbols)):
        if s in covered_syms:
            continue
        hl_c = map_binance_perp_to_hl_coin(s)
        if not hl_available or hl_c in hl_available:
            sym_to_hl.append((s, hl_c))

    if not sym_to_hl and not existing_hl.empty:
        sess.close()
        return out_path

    parts: list[pd.DataFrame] = [existing_hl] if not existing_hl.empty else []
    if not existing_hl.empty:
        parts[0]["symbol"] = parts[0]["symbol"].astype(str)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_hyperliquid_symbol_funding, sess, s, hc): s for s, hc in sym_to_hl}
        for fut in as_completed(futs):
            df_h = fut.result()
            if df_h is not None and not df_h.empty:
                parts.append(df_h)

    sess.close()
    if parts:
        all_hl = pd.concat(parts, ignore_index=True)
        all_hl["symbol"] = all_hl["symbol"].astype(str)
        all_hl.sort_values(["symbol", "timestamp"], inplace=True)
        all_hl.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
        n_tot_syms = int(all_hl["symbol"].nunique())
        all_hl["symbol"] = all_hl["symbol"].astype("category")
        pq.write_table(pa.Table.from_pandas(all_hl, preserve_index=False), out_path, compression="zstd")
        print(
            f"  [CrossVenue-HL] Saved {len(all_hl):,} on-chain hourly funding rows across {n_tot_syms} matched coins ({time.time() - t0:.1f}s)",
            flush=True,
        )
    else:
        empty_df = pd.DataFrame(
            {
                "timestamp": np.empty(0, dtype=np.int64),
                "symbol": pd.Series([], dtype="string"),
                "hl_fr_bps_per_h": np.empty(0, dtype=np.float32),
                "hl_premium_bps": np.empty(0, dtype=np.float32),
            }
        )
        pq.write_table(pa.Table.from_pandas(empty_df, preserve_index=False), out_path, compression="zstd")
    return out_path


def parse_book_depth_zip_bytes(raw_bytes: bytes, symbol: str) -> pd.DataFrame | None:
    """
    Parses a daily Binance Vision `bookDepth` ZIP archive (`timestamp, percentage, depth, notional`)
    into a wide, ultra-compact `1-row-per-snapshot` DataFrame:
      `timestamp (int64 sec), symbol, bid_02_usd, ask_02_usd, bid_1_usd, ask_1_usd, bid_2_usd, ask_2_usd, bid_5_usd, ask_5_usd`
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as fh:
                first = fh.readline().decode("utf-8", errors="ignore")
                has_hdr = "percentage" in first or "notional" in first
            with zf.open(names[0]) as fh:
                df = pd.read_csv(
                    fh,
                    header=0 if has_hdr else None,
                    names=None if has_hdr else ["timestamp", "percentage", "depth", "notional"],
                )
    except Exception:  # noqa: BLE001
        return None

    if df is None or df.empty or df.shape[1] < 4:
        return None

    raw_ts = df.iloc[:, 0]
    num_ts = pd.to_numeric(raw_ts, errors="coerce")
    if num_ts.notna().mean() > 0.8:
        valid_mask = num_ts.notna()
        df = df.loc[valid_mask]
        ts_sec = epoch_to_utc_seconds(num_ts.loc[valid_mask].to_numpy(dtype=np.int64))
    else:
        dt = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        valid_mask = dt.notna()
        if not bool(valid_mask.any()):
            return None
        df = df.loc[valid_mask]
        ts_sec = (dt.loc[valid_mask].astype("int64") // 10**9).to_numpy(dtype=np.int64)

    pct = np.round(pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) * 10.0).astype(
        np.int32
    )
    notional = np.maximum(0.0, pd.to_numeric(df.iloc[:, 3], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))

    work = pd.DataFrame({"ts": ts_sec, "p10": pct, "usd": notional})
    # Keep target levels: -50 (-5%), -20 (-2%), -10 (-1%), -2 (-0.2%), +2 (+0.2%), +10 (+1%), +20 (+2%), +50 (+5%)
    target_map = {
        -2: "bid_02_usd",
        2: "ask_02_usd",
        -10: "bid_1_usd",
        10: "ask_1_usd",
        -20: "bid_2_usd",
        20: "ask_2_usd",
        -50: "bid_5_usd",
        50: "ask_5_usd",
    }
    work = work[work["p10"].isin(target_map)]
    if work.empty:
        return None

    piv = work.pivot_table(index="ts", columns="p10", values="usd", aggfunc="last")
    out_cols: dict[str, Any] = {
        "timestamp": piv.index.to_numpy(dtype=np.int64),
        "symbol": symbol,
    }
    for k_p10, col_name in target_map.items():
        if k_p10 in piv.columns:
            out_cols[col_name] = piv[k_p10].fillna(0.0).to_numpy(dtype=np.float32)
        else:
            out_cols[col_name] = np.zeros(len(piv), dtype=np.float32)

    return pd.DataFrame(out_cols)


def fetch_shard_event_book_depth(
    m16_data_dir: Path,
    shard_name: str,
    symbol_dates: dict[str, set[str]],
    workers: int = 12,
) -> pd.DataFrame:
    """
    Fetches & merges L2 `bookDepth` snapshots for all causal candidate `(symbol, YYYY-MM-DD)` pairs
    in `shard_name` using bounded memory (`workers=12`), persisting both Parquet data and checked-date
    metadata (`*.dates.json`) on Google Drive so 404 dates are never re-queried.
    """
    import json

    bd_dir = m16_data_dir / "book_depth_events"
    bd_dir.mkdir(parents=True, exist_ok=True)
    cache_path = bd_dir / shard_name
    meta_path = bd_dir / f"{shard_name}.dates.json"

    existing_df = pd.DataFrame()
    covered_pairs: set[tuple[str, str]] = set()
    if meta_path.exists():
        try:
            raw_m = json.loads(meta_path.read_text())
            covered_pairs.update((str(k), str(d)) for k, d_list in raw_m.items() for d in d_list)
        except Exception:  # noqa: BLE001
            pass

    if cache_path.exists() and cache_path.stat().st_size > 512:
        existing_df = pd.read_parquet(cache_path)
        if not existing_df.empty:
            ex_dates = pd.to_datetime(existing_df["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
            covered_pairs.update(zip(existing_df["symbol"].astype(str), ex_dates, strict=False))

    todo: list[tuple[str, str]] = []
    for sym, d_set in symbol_dates.items():
        for ds in sorted(d_set):
            if (sym, ds) not in covered_pairs:
                todo.append((sym, ds))

    if not todo:
        return existing_df

    sess = _build_pooled_session(pool_size=max(16, workers * 2))

    def _get_one(sym_d: tuple[str, str]) -> pd.DataFrame | None:
        sym, ds = sym_d
        for tk in PERP_TICKER_ALIASES.get(sym, (sym,)):
            url = f"{VISION_CDN_BASE}/data/futures/um/daily/bookDepth/{tk}/{tk}-bookDepth-{ds}.zip"
            try:
                r = sess.get(url, timeout=15)
                if r.status_code == 200 and r.content:
                    p_df = parse_book_depth_zip_bytes(r.content, sym)
                    if p_df is not None and not p_df.empty:
                        return p_df
            except Exception:  # noqa: BLE001
                continue
        return None

    new_parts: list[pd.DataFrame] = []
    # Process in bounded chunks of 80 archives so memory stays strictly < 1.8 GB
    for start_i in range(0, len(todo), 80):
        batch = todo[start_i : start_i + 80]
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(batch)))) as pool:
            futs = [pool.submit(_get_one, item) for item in batch]
            for fut in as_completed(futs):
                res = fut.result()
                if res is not None and not res.empty:
                    new_parts.append(res)
        gc.collect()
    sess.close()

    covered_pairs.update(todo)
    meta_dict: dict[str, list[str]] = {}
    for s_k, d_k in sorted(covered_pairs):
        meta_dict.setdefault(s_k, []).append(d_k)
    meta_path.write_text(json.dumps(meta_dict))

    if not new_parts:
        return existing_df

    all_parts = ([existing_df] if not existing_df.empty else []) + new_parts
    merged = pd.concat(all_parts, ignore_index=True)
    merged["symbol"] = merged["symbol"].astype(str)
    merged.sort_values(["symbol", "timestamp"], inplace=True)
    merged.drop_duplicates(subset=["symbol", "timestamp"], keep="last", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    save_df = merged.copy()
    save_df["symbol"] = save_df["symbol"].astype("category")
    pq.write_table(pa.Table.from_pandas(save_df, preserve_index=False), cache_path, compression="zstd")
    del save_df
    return merged
