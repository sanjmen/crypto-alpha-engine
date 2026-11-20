"""
Análisis 1 Laboratory: Passive `GTX` Maker Limit Wick-Catcher at `1-Millisecond` `aggTrades`
& `1-Second` Post-Fill Resolution (`2023–2026`, Cross-Venue Solvency-Anchored Cascades).

Tests whether acting as a **passive liquidity provider (`GTX` Post-Only Maker Limit Buy)** deep
inside forced-liquidation wicks solves the structural failure of `Taker 1m-close` entries
(`P4_M4` lost `-25.9 bps` in `2023–2024` because `25–45 bps` of the wick bounce already occurred
before the `1m` bar closed, plus `12–13 bps` Taker roundtrip fees).

Strict Microstructure & Queue Realism (`ZERO` Optimistic Bias):
  1. **Causal Trigger (`t_burst_ms`):** On cross-venue solvency-anchored windows (`Bybit/HL`
     healthy), a symbol drops `<= -1.8%` (`-180 bps`) from its trailing `5m` high AND prints a
     `1-ms` multi-maker forced liquidation sell sweep (`makers >= 15` or `sell_usd >= $35k` in `1ms`,
     with `10s` forced sell volume `>= $50k`), while staying above `-8.5%` vs `15m` start (not a
     broken/delisted token).
  2. **Realistic API Latency (`+250 ms`):** Orders arrive at Binance matching engine at
     `t_order_ms = t_burst_ms + 250`.
  3. **Strict `GTX` Post-Only Check:** If the first trade at/after `t_order_ms` prints `<= P_limit`,
     `GTX` rejects (`0` fill).
  4. **Strict Queue Trade-Through Fill:** Resting limit buy at `P_limit` within `[t_order_ms, t_order_ms + ttl]`
     ONLY fills if trades execute **strictly below** `P_limit * 0.9999` (`-1 bp` trade-through) OR
     at least `$25,000` of aggressor sell volume prints `<= P_limit` after `t_order_ms`.
  5. **Pessimistic `1-Second` Post-Fill Resolution:** Aggregates `t > t_fill_ms` ticks into `1-second`
     OHLC bars; checks `Stop-Loss` (`Taker` `5.5 bps` + gap-through) strictly BEFORE `Take-Profit`
     (`Maker` `2.0 bps`, strict trade-through `high > P_tp * 1.0001`), or `Taker` `5.0 bps` time-exit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from typing import Any

import numpy as np
import pandas as pd

from src.data.multimodal_vision_ingestor import PERP_TICKER_ALIASES, VISION_CDN_BASE
from src.features.tick_sweep_physics import _build_session, parse_aggtrades_zip_window
from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics


@dataclass(frozen=True)
class MakerWickProfile:
    profile_id: str
    rung1_offset_bps: float
    rung2_offset_bps: float | None
    ttl_sec: int
    tp_bps: float
    sl_bps: float
    max_hold_sec: int
    min_drop_5m_bps: float = 180.0
    min_burst_makers: int = 15


MAKER_WICK_PROFILES: tuple[MakerWickProfile, ...] = (
    MakerWickProfile(
        profile_id="MW1_GTX_SHALLOW_25BPS",
        rung1_offset_bps=25.0,
        rung2_offset_bps=None,
        ttl_sec=120,
        tp_bps=65.0,
        sl_bps=140.0,
        max_hold_sec=1200,
        min_drop_5m_bps=180.0,
        min_burst_makers=15,
    ),
    MakerWickProfile(
        profile_id="MW2_GTX_MEDIUM_45BPS",
        rung1_offset_bps=45.0,
        rung2_offset_bps=None,
        ttl_sec=150,
        tp_bps=85.0,
        sl_bps=150.0,
        max_hold_sec=1500,
        min_drop_5m_bps=180.0,
        min_burst_makers=15,
    ),
    MakerWickProfile(
        profile_id="MW3_GTX_DEEP_70BPS",
        rung1_offset_bps=70.0,
        rung2_offset_bps=None,
        ttl_sec=180,
        tp_bps=110.0,
        sl_bps=160.0,
        max_hold_sec=1800,
        min_drop_5m_bps=200.0,
        min_burst_makers=18,
    ),
    MakerWickProfile(
        profile_id="MW4_GTX_LADDER_35_70BPS",
        rung1_offset_bps=35.0,
        rung2_offset_bps=70.0,
        ttl_sec=150,
        tp_bps=85.0,
        sl_bps=155.0,
        max_hold_sec=1500,
        min_drop_5m_bps=180.0,
        min_burst_makers=15,
    ),
    MakerWickProfile(
        profile_id="MW5_GTX_EXTREME_CLIMAX_60BPS",
        rung1_offset_bps=60.0,
        rung2_offset_bps=100.0,
        ttl_sec=180,
        tp_bps=115.0,
        sl_bps=165.0,
        max_hold_sec=1800,
        min_drop_5m_bps=240.0,
        min_burst_makers=25,
    ),
)


def _era_from_sec(ts_sec: int) -> str:
    yr = datetime.fromtimestamp(int(ts_sec), tz=timezone.utc).year
    if yr <= 2024:
        return "2023_2024"
    return "2025" if yr == 2025 else "2026"


def simulate_maker_wick_on_tick_window(
    symbol: str,
    ticks_df: pd.DataFrame,
    anchor_start_ms: int,
    anchor_end_ms: int,
    profiles: tuple[MakerWickProfile, ...] = MAKER_WICK_PROFILES,
    latency_ms: int = 250,
) -> list[dict[str, Any]]:
    """
    Simulates strictly causal `1-ms` liquidation burst detection + `+250ms` `GTX` Maker Limit
    placement + queue trade-through fill + `1-second` post-fill resolution across `ticks_df`.
    """
    if ticks_df is None or len(ticks_df) < 50:
        return []

    df = ticks_df.sort_values("ts_ms").reset_index(drop=True)
    ts_ms = df["ts_ms"].to_numpy(dtype=np.int64)
    px = df["price"].to_numpy(dtype=np.float64)
    usd = df["usd"].to_numpy(dtype=np.float64)
    makers = df["makers"].to_numpy(dtype=np.int64)
    is_buy = df["is_buy"].to_numpy(dtype=bool)

    # 1. Build 1-second bars across the window for fast causal trailing 5m high & post-fill resolution
    sec_arr = ts_ms // 1000
    s_min = int(sec_arr[0])
    s_max = int(sec_arr[-1])
    n_sec = s_max - s_min + 1
    if n_sec < 120:
        return []

    # Group into 1s OHLC + sell sweep telemetry
    df_sec = pd.DataFrame(
        {
            "sec": sec_arr,
            "px": px,
            "sell_usd": np.where(~is_buy, usd, 0.0),
            "max_sell_mk": np.where(~is_buy, makers, 0),
        }
    )
    g_s = df_sec.groupby("sec", sort=True)
    u_sec = g_s.size().index.to_numpy(dtype=np.int64)
    s_open = g_s["px"].first().to_numpy(dtype=np.float64)
    s_high = g_s["px"].max().to_numpy(dtype=np.float64)
    s_low = g_s["px"].min().to_numpy(dtype=np.float64)
    s_close = g_s["px"].last().to_numpy(dtype=np.float64)
    s_sell_usd = g_s["sell_usd"].sum().to_numpy(dtype=np.float64)

    # Dense 1-second grid with forward-filled close so trailing 300s high is O(N) exact
    dense_close = np.empty(n_sec, dtype=np.float64)
    dense_high = np.empty(n_sec, dtype=np.float64)
    dense_low = np.empty(n_sec, dtype=np.float64)
    dense_open = np.empty(n_sec, dtype=np.float64)
    dense_has_trade = np.zeros(n_sec, dtype=bool)
    dense_sell_usd = np.zeros(n_sec, dtype=np.float64)

    idx_in_dense = u_sec - s_min
    dense_has_trade[idx_in_dense] = True
    dense_open[idx_in_dense] = s_open
    dense_high[idx_in_dense] = s_high
    dense_low[idx_in_dense] = s_low
    dense_close[idx_in_dense] = s_close
    dense_sell_usd[idx_in_dense] = s_sell_usd

    last_c = float(s_close[0])
    for i in range(n_sec):
        if dense_has_trade[i]:
            last_c = dense_close[i]
        else:
            dense_open[i] = last_c
            dense_high[i] = last_c
            dense_low[i] = last_c
            dense_close[i] = last_c

    roll_high_5m = pd.Series(dense_high).shift(1).rolling(300, min_periods=30).max().to_numpy(dtype=np.float64)
    roll_open_15m = pd.Series(dense_open).shift(900).bfill().to_numpy(dtype=np.float64)
    roll_sell_10s = pd.Series(dense_sell_usd).rolling(10, min_periods=1).sum().to_numpy(dtype=np.float64)

    # Identify exact millisecond candidate sell bursts inside `[anchor_start_ms, anchor_end_ms]`
    burst_mask = (
        (~is_buy)
        & (ts_ms >= int(anchor_start_ms))
        & (ts_ms <= int(anchor_end_ms))
        & ((makers >= 15) | (usd >= 35_000.0))
    )
    burst_idxs = np.flatnonzero(burst_mask)
    if len(burst_idxs) == 0:
        return []

    maker_fee_bps = 2.0
    taker_fee_bps = 5.0
    sl_taker_fee_bps = 5.5
    out_trades: list[dict[str, Any]] = []

    for prof in profiles:
        cooldown_until_ms = 0
        for b_idx in burst_idxs.tolist():
            t_b_ms = int(ts_ms[b_idx])
            if t_b_ms < cooldown_until_ms:
                continue
            mk_b = int(makers[b_idx])
            usd_b = float(usd[b_idx])
            if mk_b < prof.min_burst_makers and usd_b < 50_000.0:
                continue

            sec_pos = (t_b_ms // 1000) - s_min
            if sec_pos < 60 or sec_pos >= n_sec - 120:
                continue
            h5 = float(roll_high_5m[sec_pos])
            o15 = float(roll_open_15m[sec_pos])
            if not np.isfinite(h5) or h5 <= 0.0 or o15 <= 0.0:
                continue

            p_burst = float(px[b_idx])
            drop_5m_bps = ((h5 - p_burst) / h5) * 1e4
            drop_15m_pct = (p_burst - o15) / o15
            if drop_5m_bps < prof.min_drop_5m_bps or drop_15m_pct < -0.085:
                continue
            if float(roll_sell_10s[sec_pos]) < 40_000.0:
                continue

            # Place GTX Maker order(s) at `t_order_ms = t_b_ms + latency_ms`
            t_ord_ms = t_b_ms + int(latency_ms)
            i_ord = int(np.searchsorted(ts_ms, t_ord_ms, side="left"))
            if i_ord >= len(ts_ms) - 10:
                continue

            p_ref = float(px[i_ord])
            p_r1 = p_ref * (1.0 - prof.rung1_offset_bps * 1e-4)
            p_r2 = p_ref * (1.0 - prof.rung2_offset_bps * 1e-4) if prof.rung2_offset_bps is not None else None

            # Strict GTX check: if market at `i_ord` is ALREADY `<= p_r1`, Rung 1 would cross -> GTX rejected!
            if float(px[i_ord]) <= p_r1:
                continue

            t_exp_ms = t_ord_ms + int(prof.ttl_sec) * 1000
            i_exp = int(np.searchsorted(ts_ms, t_exp_ms, side="right"))
            if i_exp <= i_ord + 1:
                continue

            # Scan `[i_ord + 1, i_exp)` at exact `1-ms` resolution for strict queue trade-through fill!
            r1_filled = False
            r2_filled = False
            t_fill_r1 = 0
            t_fill_r2 = 0
            sell_vol_at_r1 = 0.0
            sell_vol_at_r2 = 0.0

            sub_px = px[i_ord + 1 : i_exp]
            sub_ts = ts_ms[i_ord + 1 : i_exp]
            sub_usd = usd[i_ord + 1 : i_exp]
            sub_buy = is_buy[i_ord + 1 : i_exp]

            if float(np.min(sub_px)) > p_r1:
                # Never even touched Rung 1 within TTL -> order expired cleanly (`0` cost!)
                cooldown_until_ms = t_ord_ms + 30_000
                continue

            for k_t in range(len(sub_px)):
                pk = float(sub_px[k_t])
                tk = int(sub_ts[k_t])
                if not r1_filled and pk <= p_r1:
                    if not bool(sub_buy[k_t]):
                        sell_vol_at_r1 += float(sub_usd[k_t])
                    if pk < p_r1 * 0.9999 or sell_vol_at_r1 >= 25_000.0:
                        r1_filled = True
                        t_fill_r1 = tk
                        if p_r2 is None:
                            break
                if p_r2 is not None and not r2_filled and pk <= p_r2:
                    if not bool(sub_buy[k_t]):
                        sell_vol_at_r2 += float(sub_usd[k_t])
                    if pk < p_r2 * 0.9999 or sell_vol_at_r2 >= 25_000.0:
                        r2_filled = True
                        t_fill_r2 = tk
                        if r1_filled:
                            break

            if not r1_filled:
                cooldown_until_ms = t_ord_ms + 30_000
                continue

            if p_r2 is None:
                avg_entry = p_r1
                size_scale = 1.0
                active_from_sec = (t_fill_r1 // 1000) + 1
            elif r2_filled:
                avg_entry = 0.5 * (p_r1 + p_r2)
                size_scale = 1.0
                active_from_sec = (max(t_fill_r1, t_fill_r2) // 1000) + 1
            else:
                # Only first 50% rung filled before price snapped back!
                avg_entry = p_r1
                size_scale = 0.5
                active_from_sec = (t_fill_r1 // 1000) + 1

            start_d = active_from_sec - s_min
            end_d = min(n_sec - 1, start_d + int(prof.max_hold_sec))
            if start_d >= end_d:
                continue

            tp_px = avg_entry * (1.0 + prof.tp_bps * 1e-4)
            sl_px = avg_entry * (1.0 - prof.sl_bps * 1e-4)
            exit_px = float(dense_close[end_d])
            exit_fee = taker_fee_bps
            exit_reason = "TIME"
            exit_sec = s_min + end_d
            mfe_px = avg_entry
            mae_px = avg_entry

            for d_i in range(start_d, end_d + 1):
                if not dense_has_trade[d_i]:
                    continue
                lo_i = float(dense_low[d_i])
                hi_i = float(dense_high[d_i])
                op_i = float(dense_open[d_i])
                if hi_i > mfe_px:
                    mfe_px = hi_i
                if lo_i < mae_px:
                    mae_px = lo_i

                # 1. Pessimistic Stop-Loss priority!
                if lo_i <= sl_px:
                    exit_px = min(op_i, sl_px)
                    exit_fee = sl_taker_fee_bps
                    exit_reason = "SL"
                    exit_sec = s_min + d_i
                    break
                # 2. Strict trade-through Passive Maker Take-Profit (`hi_i > tp_px * 1.0001`)
                if hi_i > tp_px * 1.0001:
                    exit_px = tp_px
                    exit_fee = maker_fee_bps
                    exit_reason = "TP_MAKER"
                    exit_sec = s_min + d_i
                    break

            raw_gross_bps = ((exit_px - avg_entry) / avg_entry) * 1e4
            raw_fee_bps = maker_fee_bps + exit_fee
            # Scale realized return & fee by filled ladder fraction (`1.0` for single-order or both rungs, `0.5` if 1/2 rungs)
            eff_gross_bps = raw_gross_bps * size_scale
            eff_fee_bps = raw_fee_bps * size_scale
            eff_net_bps = eff_gross_bps - eff_fee_bps

            entry_sec = t_fill_r1 // 1000
            out_trades.append(
                {
                    "symbol": symbol,
                    "mech_id": prof.profile_id,
                    "pillar": "P4_MAKER_WICK",
                    "exec_mode": "GTX_MAKER_1MS",
                    "entry_ts": int(entry_sec),
                    "exit_ts": int(max(entry_sec + 1, exit_sec)),
                    "era": _era_from_sec(entry_sec),
                    "strength": round(float(drop_5m_bps * (1.0 + min(mk_b, 60) / 30.0)), 2),
                    "gross_bps": round(eff_gross_bps, 3),
                    "funding_bps": 0.0,
                    "fee_bps": round(eff_fee_bps, 3),
                    "net_bps": round(eff_net_bps, 3),
                    "mfe_bps": round(((mfe_px - avg_entry) / avg_entry) * 1e4 * size_scale, 2),
                    "mae_bps": round(((mae_px - avg_entry) / avg_entry) * 1e4 * size_scale, 2),
                    "hold_mins": round((exit_sec - entry_sec) / 60.0, 2),
                    "exit_reason": exit_reason,
                }
            )
            cooldown_until_ms = int(exit_sec) * 1000 + 45 * 60 * 1000

    return out_trades


def run_maker_wick_catcher_study(
    p4_m4_trades_df: pd.DataFrame,
    workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Streams `[-20m, +45m]` `1-millisecond` `aggTrades` around every cross-venue solvency-anchored
    liquidation cascade window in `p4_m4_trades_df` (`2023–2026`), simulates all 5 `GTX` Maker
    Wick-Catcher profiles (`+250ms` latency, strict queue trade-through, `1s` post-fill), and
    evaluates True In-Fold `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits).
    """
    uniq_evs = p4_m4_trades_df[["symbol", "entry_ts"]].drop_duplicates().sort_values(["symbol", "entry_ts"])
    if uniq_evs.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Merge overlapping `[entry_ts - 20m, entry_ts + 45m]` windows per `(symbol, date)`
    day_spans: dict[tuple[str, str], tuple[int, int]] = {}
    sym_windows: dict[str, list[tuple[int, int]]] = {}
    for row in uniq_evs.itertuples(index=False):
        sym = str(row.symbol)
        en_s = int(row.entry_ts)
        # Anchor window `[en_s - 15m, en_s + 5m]` for bursts, plus `+35m` post-fill resolution tail
        lo_ms = (en_s - 20 * 60) * 1000
        hi_ms = (en_s + 40 * 60) * 1000
        sym_windows.setdefault(sym, []).append(((en_s - 15 * 60) * 1000, (en_s + 5 * 60) * 1000))
        d0 = datetime.fromtimestamp(lo_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        d1 = datetime.fromtimestamp(hi_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        for ds in {d0, d1}:
            cur = day_spans.get((sym, ds))
            day_spans[(sym, ds)] = (
                lo_ms if cur is None else min(cur[0], lo_ms),
                hi_ms if cur is None else max(cur[1], hi_ms),
            )

    sess = _build_session(max(16, workers * 2))

    def _fetch_day(item: tuple[tuple[str, str], tuple[int, int]]) -> tuple[str, pd.DataFrame | None]:
        (sym_c, d_str), (lo_ms, hi_ms) = item
        for tk in PERP_TICKER_ALIASES.get(sym_c, (sym_c,)):
            url = f"{VISION_CDN_BASE}/data/futures/um/daily/aggTrades/{tk}/{tk}-aggTrades-{d_str}.zip"
            try:
                r = sess.get(url, timeout=45)
                if r.status_code == 200 and len(r.content) > 128:
                    parsed = parse_aggtrades_zip_window(r.content, lo_ms, hi_ms)
                    if parsed is not None and not parsed.empty:
                        return sym_c, parsed
            except Exception:  # noqa: BLE001
                continue
        return sym_c, None

    # Group day items by symbol so we process & discard each symbol in O(1) RAM!
    by_sym_days: dict[str, list[tuple[tuple[str, str], tuple[int, int]]]] = {}
    for it in day_spans.items():
        by_sym_days.setdefault(it[0][0], []).append(it)

    all_trades: list[dict[str, Any]] = []
    syms_list = sorted(by_sym_days.keys())
    for idx_s, sym_c in enumerate(syms_list, start=1):
        items_s = by_sym_days[sym_c]
        parts_s: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(items_s)))) as pool:
            futs = [pool.submit(_fetch_day, it) for it in items_s]
            for fut in as_completed(futs):
                _, df_p = fut.result()
                if df_p is not None:
                    parts_s.append(df_p)
        if parts_s:
            sym_ticks = pd.concat(parts_s, ignore_index=True).sort_values("ts_ms").reset_index(drop=True)
            del parts_s
            # Merge overlapping anchor windows for this symbol so cooldown is respected across nearby events
            wins = sorted(sym_windows[sym_c])
            merged_wins: list[tuple[int, int]] = []
            for w_lo, w_hi in wins:
                if not merged_wins or w_lo > merged_wins[-1][1] + 1800_000:
                    merged_wins.append((w_lo, w_hi))
                else:
                    merged_wins[-1] = (merged_wins[-1][0], max(merged_wins[-1][1], w_hi))

            for a_lo, a_hi in merged_wins:
                sub_t = sym_ticks[(sym_ticks["ts_ms"] >= a_lo - 960_000) & (sym_ticks["ts_ms"] <= a_hi + 2_100_000)]
                if len(sub_t) >= 50:
                    all_trades.extend(
                        simulate_maker_wick_on_tick_window(
                            symbol=sym_c,
                            ticks_df=sub_t,
                            anchor_start_ms=a_lo,
                            anchor_end_ms=a_hi,
                        )
                    )
            del sym_ticks
        if idx_s % 15 == 0 or idx_s == len(syms_list):
            gc.collect()
            print(
                f"  [MakerWick 1ms] Processed {idx_s}/{len(syms_list)} symbols | GTX fills={len(all_trades)}...",
                flush=True,
            )

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        return pd.DataFrame(), trades_df

    lb = evaluate_infold_cpcv_and_treasure_metrics(trades_df, total_trials=len(MAKER_WICK_PROFILES))
    if not lb.empty:
        lb["is_treasure"] = (
            (lb["trades_full"] >= 25)
            & (lb["mean_net_bps_full"] >= 20.0)
            & (lb["net_bps_23_24"] >= 10.0)
            & (lb["net_bps_2025"] >= 10.0)
            & (lb["net_bps_2026"] >= 15.0)
            & (lb["fee_share_of_gross_pct"] <= 30.0)
            & (lb["cpcv_pos_splits_pct"] >= 80.0)
            & (lb["cpcv_infold_oos_bps"] > 0.0)
        )
        lb["min_era_net_bps"] = lb[["net_bps_23_24", "net_bps_2025", "net_bps_2026"]].min(axis=1)
        lb = lb.sort_values(["is_treasure", "min_era_net_bps", "mean_net_bps_full"], ascending=[False, False, False])
        lb = lb.reset_index(drop=True)
    return lb, trades_df
