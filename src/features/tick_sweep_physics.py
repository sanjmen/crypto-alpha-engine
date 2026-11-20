"""
Phase-P2 Tick-Level `aggTrades` 1-Millisecond Book-Sweep & Forced-Liquidation Physics.

Extracts strictly causal (`transact_time_ms < signal_ts * 1000`) matching-engine sweep telemetry
from `data.binance.vision/data/futures/um/daily/aggTrades/` over the `15-minute` pre-signal window:

Why `1m OHLCV` + `30s L2 bookDepth` need `1ms aggTrades` Sweep Physics:
  - Resting `L2 bookDepth` Bids can be spoofed (`flash $500k Bid at -1%` to lure retail breakout buyers
    at a `4h` high while distributing via `1-ms` sell sweeps).
  - Executed `1-millisecond` multi-level sweeps (`N_makers = sum(last_trade_id - first_trade_id + 1)`
    at identical `transact_time` ms) CANNOT be spoofed without paying Taker fees + real capital.
  - Separating **Multi-Maker Sweeps (`burst_makers >= 5` or `burst_usd >= $10k`)** from **Atomized
    Retail Clicks (`burst_makers == 1` & `burst_usd < $1k`)** directly distinguishes genuine forced
    short-liquidation / institutional sweeps from retail FOMO into whale distribution (`2025` chop trap).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import io
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import requests
from requests.adapters import HTTPAdapter

from src.data.multimodal_vision_ingestor import PERP_TICKER_ALIASES, VISION_CDN_BASE
from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics

AGG_COLS_NO_HDR = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def _build_session(pool_size: int = 24) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CryptoAlphaTickSweepP2/1.0)"})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=2)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def parse_aggtrades_zip_window(
    raw_bytes: bytes,
    min_ms: int,
    max_ms: int,
) -> pd.DataFrame | None:
    """
    Uses PyArrow multi-threaded C++ CSV parser to read a daily `aggTrades` ZIP archive in memory
    and immediately filter down to `[min_ms, max_ms]` (`transact_time`), keeping peak RAM minimal.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as probe:
                first_line = probe.readline().decode("utf-8", errors="ignore")
                has_hdr = "transact_time" in first_line or "agg_trade_id" in first_line

            read_opts = pacsv.ReadOptions(
                autogenerate_column_names=False,
                column_names=None if has_hdr else AGG_COLS_NO_HDR,
                block_size=16 * 1024 * 1024,
            )
            conv_opts = pacsv.ConvertOptions(
                include_columns=[
                    "price",
                    "quantity",
                    "first_trade_id",
                    "last_trade_id",
                    "transact_time",
                    "is_buyer_maker",
                ]
            )
            with zf.open(names[0]) as fh:
                tbl = pacsv.read_csv(fh, read_options=read_opts, convert_options=conv_opts)
    except Exception:  # noqa: BLE001
        return None

    if tbl.num_rows == 0:
        return None

    ts_col = tbl["transact_time"].to_numpy(zero_copy_only=False)
    # Normalize microseconds (`>= 10^14`) if ever present in newer archives to milliseconds
    if len(ts_col) > 0 and int(ts_col[0]) >= 10**14:
        ts_ms = (ts_col // 1000).astype(np.int64)
    else:
        ts_ms = ts_col.astype(np.int64, copy=False)

    mask = (ts_ms >= int(min_ms)) & (ts_ms <= int(max_ms))
    if not bool(mask.any()):
        return None

    sub = tbl.filter(pa.array(mask))
    px = sub["price"].to_numpy(zero_copy_only=False).astype(np.float64)
    qty = sub["quantity"].to_numpy(zero_copy_only=False).astype(np.float64)
    f_id = sub["first_trade_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    l_id = sub["last_trade_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    sub_ts = ts_ms[mask]

    ibm_raw = sub["is_buyer_maker"]
    if pa.types.is_boolean(ibm_raw.type):
        ibm = ibm_raw.to_numpy(zero_copy_only=False).astype(bool)
    else:
        s_ibm = pd.Series(ibm_raw.to_pylist()).astype(str).str.lower()
        ibm = s_ibm.isin(["true", "1", "t"]).to_numpy(dtype=bool)

    return pd.DataFrame(
        {
            "ts_ms": sub_ts,
            "price": px,
            "usd": px * qty,
            "makers": np.maximum(1, l_id - f_id + 1),
            "is_buy": ~ibm,
        }
    )


def compute_event_tick_sweep_metrics(
    ticks_df: pd.DataFrame | None,
    signal_ts_sec: int,
) -> dict[str, float]:
    """
    Computes strictly causal `1-millisecond` matching-engine sweep metrics on
    `[signal_ts_sec * 1000 - 900_000, signal_ts_sec * 1000 - 1]` (`15m` BEFORE `signal_ts`).
    """
    default_out = {
        "has_agg": 0.0,
        "max_buy_makers_5m": 0.0,
        "max_sell_makers_5m": 0.0,
        "max_buy_jump_bps_5m": 0.0,
        "whale_sweep_buy_share_15m": 0.50,
        "whale_sweep_buy_share_5m": 0.50,
        "retail_atom_buy_share_15m": 0.50,
        "smart_vs_retail_div_15m": 0.0,
        "sweep_conc_share_5m": 0.0,
        "hawkes_buy_sweep_accel": 1.0,
    }
    if ticks_df is None or ticks_df.empty:
        return default_out

    end_ms = int(signal_ts_sec) * 1000 - 1
    start_15m = end_ms - 900_000 + 1
    start_5m = end_ms - 300_000 + 1
    start_3m = end_ms - 180_000 + 1

    w15 = ticks_df[(ticks_df["ts_ms"] >= start_15m) & (ticks_df["ts_ms"] <= end_ms)]
    if len(w15) < 10:
        return default_out

    # Aggregate into exact `1-millisecond` aggressor bursts `(ts_ms, is_buy)`
    grp = w15.groupby(["ts_ms", "is_buy"], sort=False)
    b_usd = grp["usd"].sum().to_numpy(dtype=np.float64)
    b_makers = grp["makers"].sum().to_numpy(dtype=np.int64)
    b_pmax = grp["price"].max().to_numpy(dtype=np.float64)
    b_pmin = grp["price"].min().to_numpy(dtype=np.float64)
    idx_tuples = grp.size().index
    b_ts = idx_tuples.get_level_values(0).to_numpy(dtype=np.int64)
    b_buy = idx_tuples.get_level_values(1).to_numpy(dtype=bool)

    b_jump_bps = ((b_pmax - b_pmin) / np.maximum(b_pmin, 1e-12)) * 1e4

    # Multi-level Sweep (`burst_makers >= 5` OR `burst_usd >= $10,000`) vs Atomized Retail (`burst_makers == 1` & `burst_usd < $1,000`)
    is_whale_sweep = (b_makers >= 5) | (b_usd >= 10_000.0)
    is_retail_atom = (b_makers == 1) & (b_usd < 1_000.0)
    in_5m = b_ts >= start_5m
    in_3m = b_ts >= start_3m

    wh_buy_15 = float(b_usd[is_whale_sweep & b_buy].sum())
    wh_sell_15 = float(b_usd[is_whale_sweep & (~b_buy)].sum())
    wh_tot_15 = wh_buy_15 + wh_sell_15
    wh_share_15 = (wh_buy_15 / wh_tot_15) if wh_tot_15 > 1.0 else 0.50

    wh_buy_5 = float(b_usd[is_whale_sweep & in_5m & b_buy].sum())
    wh_sell_5 = float(b_usd[is_whale_sweep & in_5m & (~b_buy)].sum())
    wh_tot_5 = wh_buy_5 + wh_sell_5
    wh_share_5 = (wh_buy_5 / wh_tot_5) if wh_tot_5 > 1.0 else wh_share_15

    rt_buy_15 = float(b_usd[is_retail_atom & b_buy].sum())
    rt_sell_15 = float(b_usd[is_retail_atom & (~b_buy)].sum())
    rt_tot_15 = rt_buy_15 + rt_sell_15
    rt_share_15 = (rt_buy_15 / rt_tot_15) if rt_tot_15 > 1.0 else 0.50

    tot_usd_5m = float(b_usd[in_5m].sum())
    sweep_conc_5m = (wh_tot_5 / tot_usd_5m) if tot_usd_5m > 1.0 else 0.0

    buy_5m_mask = in_5m & b_buy
    sell_5m_mask = in_5m & (~b_buy)
    max_bm_5m = float(b_makers[buy_5m_mask].max()) if bool(buy_5m_mask.any()) else 0.0
    max_sm_5m = float(b_makers[sell_5m_mask].max()) if bool(sell_5m_mask.any()) else 0.0
    max_bj_5m = float(b_jump_bps[buy_5m_mask].max()) if bool(buy_5m_mask.any()) else 0.0

    n_wh_buy_15 = int(np.sum(is_whale_sweep & b_buy))
    n_wh_buy_3 = int(np.sum(is_whale_sweep & in_3m & b_buy))
    hawkes_acc = float((n_wh_buy_3 * 5.0) / max(1.0, float(n_wh_buy_15)))

    return {
        "has_agg": 1.0,
        "max_buy_makers_5m": round(max_bm_5m, 1),
        "max_sell_makers_5m": round(max_sm_5m, 1),
        "max_buy_jump_bps_5m": round(max_bj_5m, 3),
        "whale_sweep_buy_share_15m": round(wh_share_15, 4),
        "whale_sweep_buy_share_5m": round(wh_share_5, 4),
        "retail_atom_buy_share_15m": round(rt_share_15, 4),
        "smart_vs_retail_div_15m": round(wh_share_15 - rt_share_15, 4),
        "sweep_conc_share_5m": round(sweep_conc_5m, 4),
        "hawkes_buy_sweep_accel": round(hawkes_acc, 3),
    }


def enrich_events_with_aggtrades_sweeps(
    events_df: pd.DataFrame,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Given unique `(symbol, signal_ts)` candidate events, groups them by `(symbol, UTC_date)`,
    downloads each daily `aggTrades` ZIP once in memory, extracts strictly causal `15m` pre-signal
    `1-ms` sweep physics for every event on that day, and immediately frees raw tick memory.
    """
    uniq_evs = events_df[["symbol", "signal_ts"]].drop_duplicates().reset_index(drop=True)
    if uniq_evs.empty:
        return uniq_evs

    # Map every needed `(symbol, date_str)` -> min/max ms needed on that date
    date_windows: dict[tuple[str, str], tuple[int, int]] = {}
    ev_to_dates: dict[tuple[str, int], list[str]] = {}

    for row in uniq_evs.itertuples(index=False):
        sym = str(row.symbol)
        sig_s = int(row.signal_ts)
        end_ms = sig_s * 1000 - 1
        start_ms = end_ms - 900_000 + 1
        dt_end = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
        dt_start = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
        d_set = sorted({dt_start.strftime("%Y-%m-%d"), dt_end.strftime("%Y-%m-%d")})
        ev_to_dates[(sym, sig_s)] = d_set
        for ds in d_set:
            cur = date_windows.get((sym, ds))
            if cur is None:
                date_windows[(sym, ds)] = (start_ms, end_ms)
            else:
                date_windows[(sym, ds)] = (min(cur[0], start_ms), max(cur[1], end_ms))

    # Group events by primary date so we can process in bounded batches without holding all ticks in RAM!
    all_sym_dates = sorted(date_windows.keys())
    sess = _build_session(pool_size=max(16, workers * 2))

    def _fetch_date_slice(item: tuple[str, str]) -> tuple[tuple[str, str], pd.DataFrame | None]:
        sym, ds = item
        mn_ms, mx_ms = date_windows[item]
        for tk in PERP_TICKER_ALIASES.get(sym, (sym,)):
            url = f"{VISION_CDN_BASE}/data/futures/um/daily/aggTrades/{tk}/{tk}-aggTrades-{ds}.zip"
            try:
                r = sess.get(url, timeout=30)
                if r.status_code == 200 and r.content:
                    df_w = parse_aggtrades_zip_window(r.content, mn_ms, mx_ms)
                    if df_w is not None and not df_w.empty:
                        return item, df_w
            except Exception:  # noqa: BLE001
                continue
        return item, None

    sliced_cache: dict[tuple[str, str], pd.DataFrame] = {}
    total_sd = len(all_sym_dates)
    for b_start in range(0, total_sd, 48):
        batch = all_sym_dates[b_start : b_start + 48]
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(batch)))) as pool:
            futs = [pool.submit(_fetch_date_slice, it) for it in batch]
            for fut in as_completed(futs):
                k_it, df_res = fut.result()
                if df_res is not None:
                    # Keep ONLY `[t_sig - 15m, t_sig)` rows across events on this `(sym, date)`!
                    sliced_cache[k_it] = df_res
        gc.collect()
        print(
            f"  [AggTrades-P2] Streamed {min(b_start + 48, total_sd)}/{total_sd} causal daily slices "
            f"(matched={len(sliced_cache)})",
            flush=True,
        )
    sess.close()

    feat_rows: list[dict[str, Any]] = []
    for row in uniq_evs.itertuples(index=False):
        sym = str(row.symbol)
        sig_s = int(row.signal_ts)
        d_list = ev_to_dates[(sym, sig_s)]
        parts = [sliced_cache[(sym, d)] for d in d_list if (sym, d) in sliced_cache]
        merged_ticks = pd.concat(parts, ignore_index=True) if parts else None
        m_dict = compute_event_tick_sweep_metrics(merged_ticks, sig_s)
        m_dict["symbol"] = sym
        m_dict["signal_ts"] = sig_s
        feat_rows.append(m_dict)

    del sliced_cache
    gc.collect()
    return pd.DataFrame(feat_rows)


def evaluate_p2_tick_sweep_ablation(
    p1_trades_df: pd.DataFrame,
    sweep_feats_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merges causal `1-ms` `aggTrades` sweep features onto the candidate trades of
    `P6A_L2_ASK_AIR_POCKET_LONG`, `P2_C2A_BOOK_VACUUM_PAIN_LONG`, and `B2_C2A_BASE_UNCHOKED_LONG`
    and evaluates first-principles Anti-Spoof / Whale-Sweep / Kyle-Jump variants under
    True In-Fold Refit `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits).
    """
    target_mechs = (
        "P6A_L2_ASK_AIR_POCKET_LONG",
        "P2_C2A_BOOK_VACUUM_PAIN_LONG",
        "B2_C2A_BASE_UNCHOKED_LONG",
    )
    sub = p1_trades_df[p1_trades_df["mech_id"].isin(target_mechs)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    joined = sub.merge(sweep_feats_df, on=["symbol", "signal_ts"], how="left")
    joined["has_agg"] = joined["has_agg"].fillna(0.0)
    for col, dval in (
        ("max_buy_makers_5m", 0.0),
        ("max_sell_makers_5m", 0.0),
        ("max_buy_jump_bps_5m", 0.0),
        ("whale_sweep_buy_share_15m", 0.50),
        ("whale_sweep_buy_share_5m", 0.50),
        ("smart_vs_retail_div_15m", 0.0),
        ("sweep_conc_share_5m", 0.0),
        ("hawkes_buy_sweep_accel", 1.0),
    ):
        joined[col] = joined[col].fillna(dval)

    has_a = joined["has_agg"] > 0.5
    # Pure first-principles physical filters (zero backward optimization):
    #   G1 Anti-Spoof Whale Dominance: Veto when 1ms Whale Sweeps are net-selling into the breakout (`whale_share_15m < 0.52` or `max_sell_makers > 1.5 * max_buy_makers` without buy sweep)
    g1_antispoof = (~has_a) | (
        (joined["whale_sweep_buy_share_15m"] >= 0.52)
        & (joined["whale_sweep_buy_share_5m"] >= 0.50)
        & (joined["max_buy_makers_5m"] >= 6.0)
    )
    #   G2 Kyle Walk-the-Book Jump: Requires an executed 1ms buy sweep jumping >= 4.0 bps + Whale net buying
    g2_kyle_jump = (~has_a) | ((joined["max_buy_jump_bps_5m"] >= 4.0) & (joined["whale_sweep_buy_share_15m"] >= 0.52))
    #   G3 Institutional Sweep Concentration (eliminates dead 2025 retail-chop fakeouts where sweep share < 30%)
    g3_inst_conc = (~has_a) | (
        (joined["sweep_conc_share_5m"] >= 0.32)
        & (joined["whale_sweep_buy_share_15m"] >= 0.53)
        & (joined["smart_vs_retail_div_15m"] >= -0.03)
    )

    variant_frames: list[pd.DataFrame] = []

    for base_m in ("P6A_L2_ASK_AIR_POCKET_LONG", "P2_C2A_BOOK_VACUUM_PAIN_LONG", "B2_C2A_BASE_UNCHOKED_LONG"):
        df_b = joined[joined["mech_id"] == base_m].copy()
        if df_b.empty:
            continue
        short_tag = base_m.split("_")[0] + "_" + base_m.split("_")[1]
        # 0) Raw control
        c0 = df_b.copy()
        c0["mech_id"] = f"{short_tag}_0_RAW_CONTROL"
        variant_frames.append(c0)

        # 1) G1 Anti-Spoof Whale Sweep
        c1 = df_b[g1_antispoof.loc[df_b.index]].copy()
        if not c1.empty:
            c1["mech_id"] = f"{short_tag}_1_ANTISPOOF_WHALE_SWEEP"
            c1["strength"] = c1["strength"] * (0.5 + c1["whale_sweep_buy_share_15m"])
            variant_frames.append(c1)

        # 2) G2 Kyle 1ms Impact Jump
        c2 = df_b[g2_kyle_jump.loc[df_b.index]].copy()
        if not c2.empty:
            c2["mech_id"] = f"{short_tag}_2_KYLE_1MS_JUMP"
            c2["strength"] = c2["strength"] * (1.0 + np.minimum(c2["max_buy_jump_bps_5m"], 30.0) / 20.0)
            variant_frames.append(c2)

        # 3) G3 Institutional Sweep Concentration
        c3 = df_b[g3_inst_conc.loc[df_b.index]].copy()
        if not c3.empty:
            c3["mech_id"] = f"{short_tag}_3_INST_SWEEP_CONC"
            c3["strength"] = c3["strength"] * (0.5 + c3["sweep_conc_share_5m"])
            variant_frames.append(c3)

    # 4) Unified `P6A + P2_C2A` L2-Vacuum Composite (deduplicated per `(symbol, exec_mode, 90m window)`)
    l2_both = joined[joined["mech_id"].isin(["P6A_L2_ASK_AIR_POCKET_LONG", "P2_C2A_BOOK_VACUUM_PAIN_LONG"])].copy()
    if not l2_both.empty:
        for label_u, mask_u in (
            ("UNIFIED_L2_VACUUM_RAW", pd.Series(True, index=l2_both.index)),
            ("UNIFIED_L2_VACUUM_WHALE_SWEEP", g1_antispoof.loc[l2_both.index]),
            ("UNIFIED_L2_VACUUM_INST_CONC", g3_inst_conc.loc[l2_both.index]),
        ):
            u_df = l2_both[mask_u].sort_values(["exec_mode", "symbol", "entry_ts"]).copy()
            if u_df.empty:
                continue
            # Enforce strict non-overlapping per `(exec_mode, symbol)`
            kept_rows = []
            for (_em, _sym), grp in u_df.groupby(["exec_mode", "symbol"], sort=False):
                free_t = -1
                for r_idx, r_val in grp.iterrows():
                    if int(r_val["entry_ts"]) >= free_t:
                        kept_rows.append(r_idx)
                        free_t = int(r_val["exit_ts"]) + 3600
            u_clean = u_df.loc[kept_rows].copy()
            u_clean["mech_id"] = label_u
            variant_frames.append(u_clean)

    all_variants_df = pd.concat(variant_frames, ignore_index=True) if variant_frames else pd.DataFrame()
    if all_variants_df.empty:
        return pd.DataFrame(), joined

    n_trials = int(all_variants_df[["mech_id", "exec_mode"]].drop_duplicates().shape[0])
    lb = evaluate_infold_cpcv_and_treasure_metrics(all_variants_df, total_trials=max(1, n_trials))
    return lb, all_variants_df
