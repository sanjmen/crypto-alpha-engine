"""
Step 2 Laboratory: `1-Millisecond` Matching-Engine Forced-Liquidation Climax, Exhaustion &
Institutional Absorption Physics (`aggTrades`) on `P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG`.

Why raw `1m OHLCV + 5m OI` (`P4_M4`) caught knives in `2023–2025` (`-17.8` / `-8.4 bps`) despite
producing `+70.9 bps/trade` in `2026`:
  - A `15m` OI drop (`<= -1.5%`) + `-35 bps` discount wick + green `1m` candle can occur EITHER:
    (A) **True Exhausted Liquidation Cascade (Edge):** Matching engine fired massive `1-ms`
        multi-maker forced `IOC` sell sweeps (`makers >= 12..50` in `1 ms`) during `[T-15m, T-1m]`,
        those forced sell bursts **decelerated/stopped** in the final `60s` (`[T-60s, T-1ms]`),
        and `1-ms` institutional buy sweeps stepped in to absorb the wick while retail panicked.
    (B) **Mid-Cascade Dead-Cat Bounce (Knife Trap):** Retail limit/click buyers bounced `1m` close
        by a few ticks while institutional `1-ms` sell sweeps were STILL accelerating in `[T-60s, T-1ms]`.

All metrics enforce **strict `0.000 ms` causality**: `transact_time_ms <= signal_ts_sec * 1000 - 1`
(since `entry_ts` in `p3_p4_all_trades` is `(idx + 1)` open = `signal_ts_sec + 60`, we cut off strictly
at `entry_ts * 1000 - 1` = end of signal bar `idx`, `0.000 ms` before trade entry!).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
from typing import Any

import numpy as np
import pandas as pd

from src.data.multimodal_vision_ingestor import PERP_TICKER_ALIASES, VISION_CDN_BASE
from src.features.tick_sweep_physics import _build_session, parse_aggtrades_zip_window
from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics


def compute_liquidation_exhaustion_1ms_metrics(
    ticks_df: pd.DataFrame | None,
    entry_ts_sec: int,
) -> dict[str, float]:
    """
    Computes strictly causal `1-millisecond` liquidation-climax, cascade-exhaustion, and
    whale-absorption metrics over `[entry_ts_sec * 1000 - 900_000, entry_ts_sec * 1000 - 1]`.
    """
    default_out = {
        "has_agg": 0.0,
        "max_sell_makers_15m": 0.0,
        "sell_sweep_conc_prior_14m": 0.0,
        "sell_exhaust_ratio_1m_vs_peak3m": 1.0,
        "whale_buy_share_final_1m": 0.50,
        "whale_buy_share_final_3m": 0.50,
        "retail_buy_share_final_1m": 0.50,
        "smart_vs_retail_final_1m": 0.0,
        "buy_sweep_conc_final_1m": 0.0,
    }
    if ticks_df is None or ticks_df.empty:
        return default_out

    end_ms = int(entry_ts_sec) * 1000 - 1
    start_15m = end_ms - 900_000 + 1
    start_3m = end_ms - 180_000 + 1
    start_1m = end_ms - 60_000 + 1

    w15 = ticks_df[(ticks_df["ts_ms"] >= start_15m) & (ticks_df["ts_ms"] <= end_ms)]
    if len(w15) < 12:
        return default_out

    grp = w15.groupby(["ts_ms", "is_buy"], sort=False)
    b_usd = grp["usd"].sum().to_numpy(dtype=np.float64)
    b_makers = grp["makers"].sum().to_numpy(dtype=np.int64)
    idx_tuples = grp.size().index
    b_ts = idx_tuples.get_level_values(0).to_numpy(dtype=np.int64)
    b_buy = idx_tuples.get_level_values(1).to_numpy(dtype=bool)

    is_whale = (b_makers >= 5) | (b_usd >= 10_000.0)
    is_retail = (b_makers == 1) & (b_usd < 1_000.0)
    in_prior_14m = b_ts < start_1m
    in_final_1m = b_ts >= start_1m
    in_final_3m = b_ts >= start_3m

    # 1. Forced Sell-Liquidation Climax during prior 14m (`[T-15m, T-1m)`)
    sell_prior_mask = in_prior_14m & (~b_buy)
    max_sm_15 = float(b_makers[~b_buy].max()) if bool((~b_buy).any()) else 0.0
    prior_tot_usd = float(b_usd[in_prior_14m].sum())
    prior_wh_sell_usd = float(b_usd[sell_prior_mask & is_whale].sum())
    sell_conc_prior = (prior_wh_sell_usd / prior_tot_usd) if prior_tot_usd > 1.0 else 0.0

    # 2. Cascade Exhaustion Ratio: final 1m whale sell USD vs peak 1m-equivalent whale sell rate in prior 14m
    final_1m_wh_sell = float(b_usd[in_final_1m & (~b_buy) & is_whale].sum())
    # Bucket prior 14m into seven 2-minute windows to capture the liquidation burst rate (`per minute`)
    peak_prior_wh_sell_per_min = prior_wh_sell_usd / 14.0
    for k_b in range(7):
        b_lo = start_15m + k_b * 120_000
        b_hi = min(start_1m, b_lo + 120_000)
        m_b = (b_ts >= b_lo) & (b_ts < b_hi) & (~b_buy) & is_whale
        rate_1m = float(b_usd[m_b].sum()) / 2.0
        if rate_1m > peak_prior_wh_sell_per_min:
            peak_prior_wh_sell_per_min = rate_1m
    sell_exhaust_ratio = (final_1m_wh_sell / peak_prior_wh_sell_per_min) if peak_prior_wh_sell_per_min > 100.0 else 1.0

    # 3. Institutional Whale Absorption in Reversal Window (`[T-60s, T-1ms]` & `[T-3m, T-1ms]`)
    wh_buy_1m = float(b_usd[in_final_1m & is_whale & b_buy].sum())
    wh_sell_1m = final_1m_wh_sell
    wh_tot_1m = wh_buy_1m + wh_sell_1m
    wh_buy_sh_1m = (wh_buy_1m / wh_tot_1m) if wh_tot_1m > 1.0 else 0.50

    wh_buy_3m = float(b_usd[in_final_3m & is_whale & b_buy].sum())
    wh_sell_3m = float(b_usd[in_final_3m & is_whale & (~b_buy)].sum())
    wh_tot_3m = wh_buy_3m + wh_sell_3m
    wh_buy_sh_3m = (wh_buy_3m / wh_tot_3m) if wh_tot_3m > 1.0 else wh_buy_sh_1m

    rt_buy_1m = float(b_usd[in_final_1m & is_retail & b_buy].sum())
    rt_sell_1m = float(b_usd[in_final_1m & is_retail & (~b_buy)].sum())
    rt_tot_1m = rt_buy_1m + rt_sell_1m
    rt_buy_sh_1m = (rt_buy_1m / rt_tot_1m) if rt_tot_1m > 1.0 else 0.50

    tot_1m_usd = float(b_usd[in_final_1m].sum())
    buy_sweep_conc_1m = (wh_buy_1m / tot_1m_usd) if tot_1m_usd > 1.0 else 0.0

    return {
        "has_agg": 1.0,
        "max_sell_makers_15m": round(max_sm_15, 1),
        "sell_sweep_conc_prior_14m": round(sell_conc_prior, 4),
        "sell_exhaust_ratio_1m_vs_peak3m": round(sell_exhaust_ratio, 4),
        "whale_buy_share_final_1m": round(wh_buy_sh_1m, 4),
        "whale_buy_share_final_3m": round(wh_buy_sh_3m, 4),
        "retail_buy_share_final_1m": round(rt_buy_sh_1m, 4),
        "smart_vs_retail_final_1m": round(wh_buy_sh_1m - rt_buy_sh_1m, 4),
        "buy_sweep_conc_final_1m": round(buy_sweep_conc_1m, 4),
    }


def enrich_p4_m4_events_with_1ms_aggtrades(
    p4_m4_trades_df: pd.DataFrame,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Streams daily `aggTrades` archives for every unique `(symbol, entry_ts)` in `P4_M4`
    (`0.000 ms` lookahead: `ts_ms <= entry_ts * 1000 - 1`) in `O(1)` RAM.
    """
    uniq_evs = p4_m4_trades_df[["symbol", "entry_ts"]].drop_duplicates().reset_index(drop=True)
    if uniq_evs.empty:
        return uniq_evs

    date_windows: dict[tuple[str, str], tuple[int, int]] = {}
    ev_to_dates: dict[tuple[str, int], list[str]] = {}
    for row in uniq_evs.itertuples(index=False):
        sym = str(row.symbol)
        en_s = int(row.entry_ts)
        end_ms = en_s * 1000 - 1
        start_ms = end_ms - 900_000 + 1
        d_end = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        d_start = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        ds = sorted({d_start, d_end})
        ev_to_dates[(sym, en_s)] = ds
        for d_str in ds:
            cur = date_windows.get((sym, d_str))
            date_windows[(sym, d_str)] = (
                start_ms if cur is None else min(cur[0], start_ms),
                end_ms if cur is None else max(cur[1], end_ms),
            )

    sess = _build_session(max(16, workers * 2))

    def _fetch_day(item: tuple[tuple[str, str], tuple[int, int]]) -> tuple[tuple[str, str], pd.DataFrame | None]:
        (sym_c, d_str), (lo_ms, hi_ms) = item
        for tk in PERP_TICKER_ALIASES.get(sym_c, (sym_c,)):
            url = f"{VISION_CDN_BASE}/data/futures/um/daily/aggTrades/{tk}/{tk}-aggTrades-{d_str}.zip"
            try:
                r = sess.get(url, timeout=45)
                if r.status_code == 200 and len(r.content) > 128:
                    parsed = parse_aggtrades_zip_window(r.content, lo_ms, hi_ms)
                    if parsed is not None and not parsed.empty:
                        return (sym_c, d_str), parsed
            except Exception:  # noqa: BLE001
                continue
        return (sym_c, d_str), None

    day_items = list(date_windows.items())
    day_ticks: dict[tuple[str, str], pd.DataFrame] = {}
    done = 0
    for b_i in range(0, len(day_items), 32):
        batch = day_items[b_i : b_i + 32]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(_fetch_day, it) for it in batch]
            for fut in as_completed(futs):
                k_d, df_d = fut.result()
                done += 1
                if df_d is not None:
                    day_ticks[k_d] = df_d
        gc.collect()
        if done % 64 == 0 or done == len(day_items):
            print(
                f"  [P4-M4 aggTrades] Streamed {done}/{len(day_items)} daily tick archives (matched={len(day_ticks)})...",
                flush=True,
            )

    out_rows: list[dict[str, Any]] = []
    for row in uniq_evs.itertuples(index=False):
        sym = str(row.symbol)
        en_s = int(row.entry_ts)
        parts = [day_ticks[(sym, d_s)] for d_s in ev_to_dates[(sym, en_s)] if (sym, d_s) in day_ticks]
        cat_df = pd.concat(parts, ignore_index=True) if parts else None
        m = compute_liquidation_exhaustion_1ms_metrics(cat_df, en_s)
        m["symbol"] = sym
        m["entry_ts"] = en_s
        out_rows.append(m)

    del day_ticks
    gc.collect()
    return pd.DataFrame(out_rows)


def evaluate_p4_m4_absorption_variants(
    p4_m4_trades_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates 6 causal `1-ms` liquidation-absorption gate variants on `P4_M4` across
    `MONO_A_CONVEX_TRAIL`, `MONO_B_COUPON_VACUUM`, and `MONO_C_SPLIT_50_50` with True In-Fold CPCV.
    """
    merged = p4_m4_trades_df.merge(metrics_df, on=["symbol", "entry_ts"], how="left")
    has_a = merged["has_agg"].fillna(0.0) > 0.5

    variants: list[tuple[str, pd.Series]] = [
        ("P4_M4_0_RAW_CONTROL", pd.Series(True, index=merged.index)),
        (
            "P4_M4_1_TRUE_LIQ_CLIMAX_1MS",
            has_a & (merged["max_sell_makers_15m"] >= 12.0) & (merged["sell_sweep_conc_prior_14m"] >= 0.28),
        ),
        (
            "P4_M4_2_CASCADE_EXHAUSTION_DECEL",
            has_a & (merged["sell_exhaust_ratio_1m_vs_peak3m"] <= 0.45),
        ),
        (
            "P4_M4_3_WHALE_ABSORPTION_1M",
            has_a & (merged["whale_buy_share_final_1m"] >= 0.56) & (merged["smart_vs_retail_final_1m"] >= 0.02),
        ),
        (
            "P4_M4_4_CLIMAX_EXHAUST_WHALE",
            has_a
            & (merged["max_sell_makers_15m"] >= 12.0)
            & (merged["sell_exhaust_ratio_1m_vs_peak3m"] <= 0.50)
            & (merged["whale_buy_share_final_1m"] >= 0.55),
        ),
        (
            "P4_M4_5_STRICT_INST_SNAPBACK",
            has_a
            & (merged["max_sell_makers_15m"] >= 18.0)
            & (merged["sell_sweep_conc_prior_14m"] >= 0.30)
            & (merged["sell_exhaust_ratio_1m_vs_peak3m"] <= 0.35)
            & (merged["whale_buy_share_final_1m"] >= 0.60),
        ),
    ]

    variant_frames: list[pd.DataFrame] = []
    for v_id, mask in variants:
        sub = merged[mask].copy()
        if sub.empty:
            continue
        sub["mech_id"] = v_id
        variant_frames.append(sub)

    if not variant_frames:
        return pd.DataFrame(), pd.DataFrame()

    all_v_df = pd.concat(variant_frames, ignore_index=True)
    n_trials = int(all_v_df[["mech_id", "exec_mode"]].drop_duplicates().shape[0])
    lb = evaluate_infold_cpcv_and_treasure_metrics(all_v_df, total_trials=max(1, n_trials))
    if lb.empty:
        return pd.DataFrame(), all_v_df

    lb["is_treasure"] = (
        (lb["trades_full"] >= 25)
        & (lb["mean_net_bps_full"] >= 25.0)
        & (lb["net_bps_23_24"] >= 15.0)
        & (lb["net_bps_2025"] >= 15.0)
        & (lb["net_bps_2026"] >= 20.0)
        & (lb["fee_share_of_gross_pct"] <= 30.0)
        & (lb["cpcv_pos_splits_pct"] >= 80.0)
        & (lb["cpcv_infold_oos_bps"] > 0.0)
    )
    lb["min_era_net_bps"] = lb[["net_bps_23_24", "net_bps_2025", "net_bps_2026"]].min(axis=1)
    lb = lb.sort_values(["is_treasure", "min_era_net_bps", "mean_net_bps_full"], ascending=[False, False, False])
    return lb.reset_index(drop=True), all_v_df
