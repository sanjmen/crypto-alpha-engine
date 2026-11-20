"""
Frente B Laboratory: L2 `bookDepth` Wall-Veto on Live Snipers (`M09`, `M23`, `M04`) +
Creative Inverted Cascade-Breakdown `SHORT` (`1-ms` `aggTrades`) & Structural Trap Snipers (`P8`, `P9`, `P10`).

Executes two complementary investigations with `0.000 ms` lookahead and pessimistic SL-first physics:
  1. **Part B1 — Live Sniper Decomposition & L2 Order-Book Upgrade (`M09_LONG`, `M09_SHORT`, `M23`, `M04`):**
     - Audits `M09_LONG` and `M09_SHORT` separately (exposing hidden leg asymmetry).
     - Attaches strictly causal `L2 bookDepth` (`t_book <= entry_ts`, max `600s` staleness) to every
       historical trigger across `SNIPER_UNIVERSE_82` (`2023–2026`), testing:
         * `RAW`: Current production (blind to L2 order book)
         * `L2_NO_WALL`: Vetoes `LONG` into heavy Ask wall (`book_imb_1pct < -0.15`) and `SHORT` into Bid wall (`> +0.15`)
         * `L2_AIR_POCKET`: Requires `book_imb_1pct >= +0.10` (`LONG`) or `<= -0.10` (`SHORT`)

  2. **Part B2 — Creative First-Principles Mechanisms ("Flipping Analysis 1 Upside Down"):**
     - `P8_CASCADE_WATERFALL_SHORT` (`1-ms` `aggTrades` + `1-s` pessimistic SL-first):
       Flips Analysis 1's `-97 bps` (`t = -17.4`) adverse-selection discovery 180 degrees: when a `1-ms`
       forced sell burst occurs (`T_trig_ms`) AND `+250ms` later price breaks `-30 / -35 / -45 bps` BELOW
       `P_trig` (secondary margin-call cascade ignition!), enters `TAKER SHORT` (`5 bps` fee + `1 bp` slip)
       to ride the secondary liquidation waterfall.
     - `P9_FAILED_BREAKOUT_ASK_WALL_SHORT` (`1m` pessimistic SL-first):
       Altcoin spikes above prior `24h` High on positive funding (`FR >= +1.5 bps`), hits a real `L2 Ask Wall`
       (`book_imb_1pct <= -0.18`), and closes back inside the range with aggressive selling (`tb_share <= 0.46`).
     - `P10_BTC_FLUSH_IMMUNE_LEADER_LONG` (`1m` pessimistic SL-first):
       During a sharp `BTCUSDT` panic hour (`ret_1h <= -1.2%`, `vol >= 1.8x`), an altcoin closes GREEN
       (`ret_1h >= +0.5%`, `+2.0%+` relative alpha) with `taker_buy_share >= 0.54` and `L2 Bid >= Ask`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.unified_1m_simulator import resolve_sniper_1m_path
from src.data.microstructure_p1_ingestor import _build_pooled_session, parse_book_depth_zip_bytes
from src.data.multimodal_vision_ingestor import PERP_TICKER_ALIASES, VISION_CDN_BASE
from src.features.tick_sweep_physics import _build_session, parse_aggtrades_zip_window
from src.models.counterparty_mechanics_lab import evaluate_infold_cpcv_and_treasure_metrics
from src.strategies.event_snipers import (
    SNIPER_BRACKET_PROFILES,
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_UNIVERSE_82,
    EventDrivenSniperEngine,
    SniperBracketSpec,
)

TAKER_FEE_BPS = 5.0
TAKER_SLIP_BPS = 1.0
MAKER_TP_FEE_BPS = 2.0


@dataclass(frozen=True)
class WaterfallShortSpec:
    mech_id: str
    description: str
    min_burst_usd: float
    min_burst_makers: int
    breakdown_trigger_bps: float  # e.g. 35.0 -> enter SHORT when price trades <= P_trig * (1 - 0.0035) after +250ms
    max_wait_sec: float
    stop_loss_bps: float
    take_profit_bps: float
    trail_trigger_bps: float | None = None
    trail_distance_bps: float | None = None
    max_hold_sec: int = 5400


WATERFALL_SHORT_SPECS: tuple[WaterfallShortSpec, ...] = (
    WaterfallShortSpec(
        mech_id="P8A_WATERFALL_SHORT_BREAK_30BPS_CONVEX",
        description="1ms Burst ($35k/8m) -> Breakdown -30bps Taker SHORT, SL +125bps, Convex Trail +150/-75, TP -420bps",
        min_burst_usd=35_000.0,
        min_burst_makers=8,
        breakdown_trigger_bps=30.0,
        max_wait_sec=120.0,
        stop_loss_bps=125.0,
        take_profit_bps=420.0,
        trail_trigger_bps=150.0,
        trail_distance_bps=75.0,
    ),
    WaterfallShortSpec(
        mech_id="P8B_WATERFALL_SHORT_BREAK_45BPS_CONVEX",
        description="1ms Burst ($35k/8m) -> Breakdown -45bps Taker SHORT, SL +135bps, Convex Trail +165/-80, TP -450bps",
        min_burst_usd=35_000.0,
        min_burst_makers=8,
        breakdown_trigger_bps=45.0,
        max_wait_sec=120.0,
        stop_loss_bps=135.0,
        take_profit_bps=450.0,
        trail_trigger_bps=165.0,
        trail_distance_bps=80.0,
    ),
    WaterfallShortSpec(
        mech_id="P8C_WATERFALL_SHORT_MIRROR_MW2_35BPS",
        description="Direct Mirror of MW2: Breakdown -35bps Taker SHORT, SL +115bps, Fixed Maker TP -165bps",
        min_burst_usd=35_000.0,
        min_burst_makers=8,
        breakdown_trigger_bps=35.0,
        max_wait_sec=120.0,
        stop_loss_bps=115.0,
        take_profit_bps=165.0,
        trail_trigger_bps=None,
        trail_distance_bps=None,
    ),
    WaterfallShortSpec(
        mech_id="P8D_WATERFALL_SHORT_HEAVY_CLIMAX_40BPS",
        description="Heavy 1ms Climax ($65k/12m) -> Breakdown -40bps Taker SHORT, SL +135bps, Trail +175/-85, TP -500bps",
        min_burst_usd=65_000.0,
        min_burst_makers=12,
        breakdown_trigger_bps=40.0,
        max_wait_sec=120.0,
        stop_loss_bps=135.0,
        take_profit_bps=500.0,
        trail_trigger_bps=175.0,
        trail_distance_bps=85.0,
    ),
)


def _build_1s_ohlc(
    ts_ms: np.ndarray, px: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 1-second OHLC aggregation from millisecond trade arrays."""
    if len(ts_ms) == 0:
        e_i = np.empty(0, dtype=np.int64)
        e_f = np.empty(0, dtype=np.float64)
        return e_i, e_f, e_f, e_f, e_f
    sec = ts_ms // 1000
    uniq_s, first_idx = np.unique(sec, return_index=True)
    last_idx = np.append(first_idx[1:] - 1, len(sec) - 1)
    o_s = px[first_idx]
    c_s = px[last_idx]
    h_s = np.maximum.reduceat(px, first_idx)
    l_s = np.minimum.reduceat(px, first_idx)
    return uniq_s, o_s, h_s, l_s, c_s


def simulate_waterfall_short_on_window(
    agg_df: pd.DataFrame,
    signal_ts_sec: int,
    spec: WaterfallShortSpec,
    latency_ms: int = 250,
    post_anchor_only: bool = False,
    post_window_sec: int = 1800,
) -> dict[str, Any] | None:
    """
    Simulates a strictly causal (`+250ms` latency, Taker stop-entry, `1s` pessimistic SL-before-TP)
    `SHORT` entry on secondary liquidation cascade breakdown:
      - When `post_anchor_only=False` (legacy window): scans `[signal_ts - 900s, signal_ts - 1ms]`.
      - When `post_anchor_only=True` (`0.000 ms` window-conditioning): scans ONLY strictly AFTER `signal_ts`
        inside `[signal_ts, signal_ts + post_window_sec)`!
      2. Arms stop-sell trigger at `T_arm_ms = T_trig_ms + latency_ms` for `stop_px = P_trig * (1 - break_bps/1e4)`.
      3. Enters `TAKER SHORT` on the first trade at `t_ms >= T_arm_ms` (within `max_wait_sec`) with `px <= stop_px`,
         filling at `fill_px = min(px, stop_px) * (1 - TAKER_SLIP_BPS/1e4)` (pessimistic downward slippage!).
      4. Resolves post-entry `1-second` OHLC bars with strict pessimistic **SL-before-Trail-before-TP** priority!
    """
    if agg_df is None or agg_df.empty:
        return None

    sig_ms = int(signal_ts_sec) * 1000
    if post_anchor_only:
        win_start_ms = sig_ms
        win_end_ms = sig_ms + int(post_window_sec) * 1000
    else:
        win_start_ms = sig_ms - 15 * 60 * 1000
        win_end_ms = sig_ms
    ts_all = agg_df["ts_ms"].to_numpy(dtype=np.int64)
    px_all = agg_df["price"].to_numpy(dtype=np.float64)
    usd_all = agg_df["usd"].to_numpy(dtype=np.float64)
    mk_col = "makers" if "makers" in agg_df.columns else "n_makers"
    mk_all = agg_df[mk_col].to_numpy(dtype=np.int64)
    if "is_buy" in agg_df.columns:
        ibm_all = ~agg_df["is_buy"].to_numpy(dtype=bool)
    else:
        ibm_all = agg_df["is_buyer_maker"].to_numpy(dtype=bool)

    m_pre = (ts_all >= win_start_ms) & (ts_all < win_end_ms) & ibm_all
    if not bool(np.any(m_pre)):
        return None

    pre_df = pd.DataFrame({"ts_ms": ts_all[m_pre], "px": px_all[m_pre], "usd": usd_all[m_pre], "mk": mk_all[m_pre]})
    bursts = (
        pre_df.groupby("ts_ms", sort=True)
        .agg(b_usd=("usd", "sum"), b_mk=("mk", "sum"), last_px=("px", "min"))
        .reset_index()
    )
    hits = bursts[(bursts["b_usd"] >= spec.min_burst_usd) & (bursts["b_mk"] >= spec.min_burst_makers)]
    if hits.empty:
        return None

    first_b = hits.iloc[0]
    trig_ms = int(first_b["ts_ms"])
    trig_px = float(first_b["last_px"])
    if trig_px <= 0.0:
        return None

    arm_ms = trig_ms + int(latency_ms)
    expire_ms = arm_ms + int(spec.max_wait_sec * 1000.0)
    stop_trig_px = trig_px * (1.0 - spec.breakdown_trigger_bps / 10_000.0)

    idx_arm = int(np.searchsorted(ts_all, arm_ms, side="left"))
    idx_exp = int(np.searchsorted(ts_all, expire_ms, side="right"))
    if idx_arm >= len(ts_all) or idx_arm >= idx_exp:
        return None

    sub_px = px_all[idx_arm:idx_exp]
    sub_ts = ts_all[idx_arm:idx_exp]
    brk = np.where(sub_px <= stop_trig_px)[0]
    if brk.size == 0:
        return None

    k_hit = int(brk[0])
    fill_ms = int(sub_ts[k_hit])
    raw_fill_px = min(float(sub_px[k_hit]), stop_trig_px)
    entry_px = raw_fill_px * (1.0 - TAKER_SLIP_BPS / 10_000.0)

    post_idx = idx_arm + k_hit + 1
    if post_idx >= len(ts_all):
        return None
    sec_s, o_s, h_s, l_s, c_s = _build_1s_ohlc(ts_all[post_idx:], px_all[post_idx:])
    if len(sec_s) == 0:
        return None

    fill_sec = fill_ms // 1000
    max_exit_sec = fill_sec + int(spec.max_hold_sec)
    sl_px = entry_px * (1.0 + spec.stop_loss_bps / 10_000.0)
    tp_px = entry_px * (1.0 - spec.take_profit_bps / 10_000.0)
    has_trail = spec.trail_trigger_bps is not None and spec.trail_distance_bps is not None
    tr_trig_px = entry_px * (1.0 - (spec.trail_trigger_bps or 0.0) / 10_000.0)
    tr_dist_f = (spec.trail_distance_bps or 0.0) / 10_000.0

    best_low = entry_px
    trail_armed = False
    exit_px = float(c_s[-1])
    exit_sec = int(sec_s[-1])
    exit_reason = "TIME"
    exit_fee_bps = TAKER_FEE_BPS

    for j in range(len(sec_s)):
        s_t = int(sec_s[j])
        if s_t <= fill_sec:
            continue
        if s_t >= max_exit_sec:
            exit_px = float(o_s[j])
            exit_sec = s_t
            exit_reason = "TIME"
            exit_fee_bps = TAKER_FEE_BPS
            break

        # 1. Pessimistic SL check FIRST (high >= sl_px)
        if float(h_s[j]) >= sl_px:
            exit_px = max(float(o_s[j]), sl_px) * (1.0 + TAKER_SLIP_BPS / 10_000.0)
            exit_sec = s_t
            exit_reason = "SL"
            exit_fee_bps = TAKER_FEE_BPS
            break

        # 2. Trailing stop check (if already armed from prior second!)
        if has_trail and trail_armed:
            dyn_stop = best_low * (1.0 + tr_dist_f)
            if float(h_s[j]) >= dyn_stop:
                exit_px = max(float(o_s[j]), dyn_stop) * (1.0 + TAKER_SLIP_BPS / 10_000.0)
                exit_sec = s_t
                exit_reason = "TRAIL"
                exit_fee_bps = TAKER_FEE_BPS
                break

        # 3. Take-Profit check (Resting Maker BUY at tp_px)
        if float(l_s[j]) <= tp_px:
            exit_px = tp_px
            exit_sec = s_t
            exit_reason = "TP_MAKER"
            exit_fee_bps = MAKER_TP_FEE_BPS
            break

        # Update trailing low at end of second (strictly causal for subsequent seconds!)
        if float(l_s[j]) < best_low:
            best_low = float(l_s[j])
            if has_trail and best_low <= tr_trig_px:
                trail_armed = True

    gross_bps = ((entry_px - exit_px) / entry_px) * 10_000.0
    fee_bps = TAKER_FEE_BPS + exit_fee_bps
    net_bps = gross_bps - fee_bps
    yr = datetime.fromtimestamp(int(fill_sec), tz=timezone.utc).year
    era = "2023_2024" if yr <= 2024 else ("2025" if yr == 2025 else "2026")
    return {
        "mech_id": spec.mech_id,
        "pillar": "P8_Cascade_Waterfall_Short_1ms",
        "exec_mode": "TAKER_STOP_SHORT_1MS",
        "entry_ts": int(fill_sec),
        "exit_ts": int(max(fill_sec + 1, exit_sec)),
        "era": era,
        "strength": round(float(first_b["b_usd"]) / 1000.0, 2),
        "gross_bps": float(gross_bps),
        "funding_bps": 0.0,
        "fee_bps": float(fee_bps),
        "net_bps": float(net_bps),
        "exit_reason": exit_reason,
    }


def run_waterfall_short_1ms_study(
    p4_m4_trades: pd.DataFrame,
    specs: tuple[WaterfallShortSpec, ...] = WATERFALL_SHORT_SPECS,
    workers: int = 8,
    post_anchor_only: bool = False,
    post_window_sec: int = 1800,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs `P8A..P8D` Secondary Cascade Waterfall SHORT across anchor windows (`post_anchor_only=True` -> 0.000ms conditioning)."""
    ev_df = (
        p4_m4_trades[["symbol", "entry_ts"]]
        .drop_duplicates()
        .sort_values(["symbol", "entry_ts"])
        .reset_index(drop=True)
    )
    by_sym: dict[str, list[int]] = {}
    for s, g in ev_df.groupby("symbol", sort=False):
        by_sym[str(s)] = [int(x) for x in g["entry_ts"].tolist()]

    lo_pad = 10 if post_anchor_only else 960
    hi_pad = int(post_window_sec) + 5600 if post_anchor_only else 5600

    def _proc_sym(sym: str, ts_list: list[int]) -> list[dict[str, Any]]:
        sess = _build_session(pool_size=6)
        tickers = PERP_TICKER_ALIASES.get(sym, (sym,))
        by_day: dict[str, list[int]] = {}
        for t_s in ts_list:
            d0 = datetime.fromtimestamp(t_s - lo_pad, tz=timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(t_s + hi_pad, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day.setdefault(d0, []).append(t_s)
            if d1 != d0:
                by_day.setdefault(d1, []).append(t_s)

        day_frames: dict[str, pd.DataFrame] = {}
        for d_str, rel_ts in sorted(by_day.items()):
            min_ms = (min(rel_ts) - lo_pad) * 1000
            max_ms = (max(rel_ts) + hi_pad) * 1000
            for tk in tickers:
                url = f"{VISION_CDN_BASE}/data/futures/um/daily/aggTrades/{tk}/{tk}-aggTrades-{d_str}.zip"
                try:
                    r = sess.get(url, timeout=30)
                    if r.status_code == 200 and r.content:
                        df_w = parse_aggtrades_zip_window(r.content, min_ms=min_ms, max_ms=max_ms)
                        if df_w is not None and not df_w.empty:
                            day_frames[d_str] = df_w
                        break
                    if r.status_code == 404:
                        continue
                except Exception:  # noqa: BLE001
                    continue

        out: list[dict[str, Any]] = []
        for t_s in ts_list:
            d0 = datetime.fromtimestamp(t_s - lo_pad, tz=timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(t_s + hi_pad, tz=timezone.utc).strftime("%Y-%m-%d")
            parts = [day_frames[d] for d in ({d0, d1}) if d in day_frames]
            if not parts:
                continue
            w_df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            w_sub = w_df[(w_df["ts_ms"] >= (t_s - lo_pad) * 1000) & (w_df["ts_ms"] <= (t_s + hi_pad) * 1000)]
            if w_sub.empty:
                continue
            for sp in specs:
                res = simulate_waterfall_short_on_window(
                    w_sub,
                    signal_ts_sec=t_s,
                    spec=sp,
                    post_anchor_only=post_anchor_only,
                    post_window_sec=post_window_sec,
                )
                if res is not None:
                    res["symbol"] = sym
                    out.append(res)
        day_frames.clear()
        sess.close()
        return out

    all_trades: list[dict[str, Any]] = []
    items = sorted(by_sym.items())
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = {pool.submit(_proc_sym, s, tl): s for s, tl in items}
        for fut in as_completed(futs):
            all_trades.extend(fut.result())

    trades_df = pd.DataFrame(all_trades)
    if trades_df.empty:
        return pd.DataFrame(), trades_df
    trades_df.sort_values(["mech_id", "entry_ts"], inplace=True)
    trades_df.reset_index(drop=True, inplace=True)
    lb = evaluate_infold_cpcv_and_treasure_metrics(trades_df, total_trials=max(1, len(specs)))
    return lb, trades_df


def lookup_causal_l2_book_imbalance(
    bd_df: pd.DataFrame | None,
    entry_ts: int,
    max_staleness_sec: int = 600,
) -> tuple[float | None, float | None]:
    """
    Returns `(book_imb_1pct, book_imb_2pct)` strictly at `t_book <= entry_ts` (`0` lookahead)
    within `max_staleness_sec`, or `(None, None)` if no recent L2 snapshot exists.
    """
    if bd_df is None or bd_df.empty:
        return None, None
    ts_b = bd_df["timestamp"].to_numpy(dtype=np.int64)
    idx = int(np.searchsorted(ts_b, int(entry_ts), side="right")) - 1
    if idx < 0 or (int(entry_ts) - int(ts_b[idx])) > max_staleness_sec:
        return None, None
    b1 = float(bd_df["bid_1_usd"].iloc[idx])
    a1 = float(bd_df["ask_1_usd"].iloc[idx])
    b2 = float(bd_df["bid_2_usd"].iloc[idx])
    a2 = float(bd_df["ask_2_usd"].iloc[idx])
    imb1 = (b1 - a1) / (b1 + a1) if (b1 + a1) > 1e-6 else None
    imb2 = (b2 - a2) / (b2 + a2) if (b2 + a2) > 1e-6 else None
    return imb1, imb2


def scan_creative_1h_snipers_p9_p10(
    ts_eval: np.ndarray,
    sniper_syms: list[str],
    c_1h: np.ndarray,
    h_1h: np.ndarray,
    qv_1h: np.ndarray,
    tb_1h: np.ndarray,
    fr_1h: np.ndarray,
    rank_1h: np.ndarray,
) -> list[tuple[int, int, int, str]]:
    """
    Scans strictly causal `1h` triggers (`entry_ts = ts_eval[t] + 3600` = close of bar `t`!) for:
      - `P9_FAILED_BREAKOUT_BULL_TRAP_SHORT`:
        Bar `t - 1` spiked above prior `24h` High (`h_1h[t-1] > max_h24_prev`), positive funding (`fr >= +1.5 bps`),
        AND bar `t` failed back below `max_h24_prev` (`c_1h[t] < max_h24_prev`) with aggressive selling (`tb_share <= 0.46`)
        on liquid perps (`rank < 55`).
      - `P10_BTC_FLUSH_IMMUNE_LEADER_LONG`:
        Bar `t` is a sharp `BTCUSDT` dump (`btc_ret <= -1.2%` on `btc_vol >= 1.8x` 24h median), YET altcoin `j`
        (`rank < 45`) closes GREEN (`alt_ret >= +0.5%`, relative alpha `>= +2.0%`) with `tb_share >= 0.54`.
    Returns `(entry_ts, sym_idx, side, mech_base)`.
    """
    n_b, n_s = c_1h.shape
    if n_b < 55 or "BTCUSDT" not in sniper_syms:
        return []
    btc_j = sniper_syms.index("BTCUSDT")
    btc_c = c_1h[:, btc_j]
    btc_qv = qv_1h[:, btc_j]
    btc_med24 = pd.Series(btc_qv).rolling(24, min_periods=12).median().shift(1).to_numpy(dtype=np.float64)

    h24_prev2 = pd.DataFrame(h_1h).shift(2).rolling(24, min_periods=12).max().to_numpy(dtype=np.float64)
    tb_share = np.where(qv_1h > 1e-8, tb_1h / np.maximum(qv_1h, 1e-8), 0.5)

    evs: list[tuple[int, int, int, str]] = []
    for t in range(30, n_b - 2):
        entry_ts = int(ts_eval[t]) + 3600
        # P10 check (BTC panic hour)
        if np.isfinite(btc_c[t]) and np.isfinite(btc_c[t - 1]) and btc_c[t - 1] > 0:
            btc_r = (btc_c[t] - btc_c[t - 1]) / btc_c[t - 1]
            if btc_r <= -0.012 and np.isfinite(btc_med24[t]) and btc_qv[t] >= 1.8 * btc_med24[t]:
                for j in range(n_s):
                    if j == btc_j or rank_1h[t, j] >= 45:
                        continue
                    if np.isfinite(c_1h[t, j]) and np.isfinite(c_1h[t - 1, j]) and c_1h[t - 1, j] > 0:
                        alt_r = (c_1h[t, j] - c_1h[t - 1, j]) / c_1h[t - 1, j]
                        if alt_r >= 0.005 and (alt_r - btc_r) >= 0.020 and tb_share[t, j] >= 0.54:
                            evs.append((entry_ts, j, 1, "P10_BTC_FLUSH_IMMUNE_LONG"))

        # P9 check (Failed 24h breakout bull-trap short)
        for j in range(n_s):
            if rank_1h[t, j] >= 55:
                continue
            ref_h = h24_prev2[t, j]
            if not np.isfinite(ref_h) or ref_h <= 0:
                continue
            if (
                np.isfinite(h_1h[t - 1, j])
                and h_1h[t - 1, j] > ref_h * 1.002
                and np.isfinite(c_1h[t, j])
                and c_1h[t, j] < ref_h * 0.998
                and fr_1h[t, j] >= 0.00015
                and tb_share[t, j] <= 0.46
            ):
                evs.append((entry_ts, j, -1, "P9_FAILED_BREAKOUT_TRAP_SHORT"))
    return evs


def run_sniper_l2_and_creative_1m_study(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    symbol_1m_loader: Any,
    eval_start: str = "2023-03-01",
    workers: int = 8,
    fetch_live_book_depth: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes Part B1 (`M09_LONG`, `M09_SHORT`, `M23_LONG`, `M04_LONG` x `RAW / L2_NO_WALL / L2_AIR_POCKET`)
    and Part B2 (`P9_FAILED_BREAKOUT_TRAP_SHORT`, `P10_BTC_FLUSH_IMMUNE_LONG` x `RAW / L2_CONFIRMED`)
    on exact `1m` physics across `SNIPER_UNIVERSE_82` (`2023–2026`).
    """
    sniper_syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym and "BTCUSDT" in bars_1h_by_sym]
    if not sniper_syms:
        return pd.DataFrame(), pd.DataFrame()

    btc_df = bars_1h_by_sym["BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    full_idx = pd.DatetimeIndex(pd.to_datetime(btc_df["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
    eval_idx = full_idx[full_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, ns = len(eval_idx), len(sniper_syms)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    c_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    h_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    l_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    qv_1h = np.zeros((n_b, ns), dtype=np.float64)
    tb_1h = np.zeros((n_b, ns), dtype=np.float64)
    fr_1h = np.zeros((n_b, ns), dtype=np.float64)
    f_maps: dict[str, dict[int, float]] = {}
    f_series_map: dict[str, pd.Series] = {}

    for j_s, s_sym in enumerate(sniper_syms):
        bs = bars_1h_by_sym[s_sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        c_1h[:, j_s] = pd.Series(bs["close"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        h_1h[:, j_s] = pd.Series(bs["high"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        l_1h[:, j_s] = pd.Series(bs["low"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        qv_1h[:, j_s] = (
            pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        tb_1h[:, j_s] = (
            pd.Series(bs["taker_buy_quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        if s_sym in funding_by_sym and not funding_by_sym[s_sym].empty:
            fd = funding_by_sym[s_sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            fts = ((fd["timestamp"].to_numpy(dtype=np.int64) + 30) // 60) * 60
            frs = fd["funding_rate"].to_numpy(dtype=np.float64)
            f_maps[s_sym] = {int(k): float(v) for k, v in zip(fts.tolist(), frs.tolist(), strict=False)}
            f_ser = pd.Series(frs, index=pd.DatetimeIndex(pd.to_datetime(fts, unit="s", utc=True)))
            f_ser = f_ser[~f_ser.index.duplicated(keep="last")]
            f_series_map[s_sym] = f_ser
            fr_1h[:, j_s] = f_ser.reindex(eval_idx, method="ffill").fillna(0.0).to_numpy()

    adv30 = pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).mean().shift(1).fillna(0.0)
    rank_1h = adv30.rank(axis=1, ascending=False, method="first").fillna(999.0).to_numpy(dtype=np.int64) - 1
    rk_df_1h = pd.DataFrame(rank_1h, index=eval_idx, columns=sniper_syms)
    idx_15m = pd.date_range(eval_idx[0], eval_idx[-1] + pd.Timedelta(minutes=45), freq="15min", tz="UTC")
    ts_15m = idx_15m.as_unit("s").astype(np.int64).to_numpy()

    sn_eng = EventDrivenSniperEngine()
    by_sym_evs: dict[str, list[tuple[int, int, str, int]]] = {}
    for en_ts, s_idx, side, st_name, _sc_v, rk_v in sn_eng.scan_historical_1h_events(
        ts_eval, sniper_syms, c_1h, h_1h, l_1h, qv_1h, tb_1h, rank_1h
    ):
        base_id = ("M09_LONG" if side > 0 else "M09_SHORT") if st_name == SNIPER_M09_STRAT else "M23_BREADTH_LONG"
        by_sym_evs.setdefault(sniper_syms[s_idx], []).append((en_ts, side, base_id, rk_v))

    for en_ts, s_idx, side, base_id in scan_creative_1h_snipers_p9_p10(
        ts_eval, sniper_syms, c_1h, h_1h, qv_1h, tb_1h, fr_1h, rank_1h
    ):
        rk_v = int(rank_1h[min(n_b - 1, max(0, np.searchsorted(ts_eval, en_ts - 3600))), s_idx])
        by_sym_evs.setdefault(sniper_syms[s_idx], []).append((en_ts, side, base_id, rk_v))

    # Identify M04 candidate symbols so we only stream required 1m symbols
    h22_prev = pd.DataFrame(h_1h).shift(1).rolling(22, min_periods=10).max().to_numpy(dtype=np.float64)
    m04_syms = {
        sniper_syms[int(j)]
        for j in np.where(
            np.any((rank_1h < sn_eng.m04_max_rank) & (fr_1h <= sn_eng.m04_max_funding_rate) & (h_1h > h22_prev), axis=0)
        )[0]
    }
    target_syms = sorted(set(by_sym_evs.keys()) | m04_syms)

    bracket_map: dict[str, SniperBracketSpec] = {
        "M09_LONG": SNIPER_BRACKET_PROFILES[SNIPER_M09_STRAT],
        "M09_SHORT": SNIPER_BRACKET_PROFILES[SNIPER_M09_STRAT],
        "M23_BREADTH_LONG": SNIPER_BRACKET_PROFILES[SNIPER_M23_STRAT],
        "M04_NEGFUND_LONG": SNIPER_BRACKET_PROFILES[SNIPER_M04_STRAT],
        "P9_FAILED_BREAKOUT_TRAP_SHORT": SniperBracketSpec(
            strategy="P9_FAILED_BREAKOUT_TRAP_SHORT",
            stop_loss_pct=0.020,
            take_profit_pct=0.095,
            lock_trigger_pct=0.022,
            lock_level_pct=0.006,
            trail_trigger_pct=0.032,
            trail_distance_pct=0.016,
            max_hold_hours=10.0,
        ),
        "P10_BTC_FLUSH_IMMUNE_LONG": SniperBracketSpec(
            strategy="P10_BTC_FLUSH_IMMUNE_LONG",
            stop_loss_pct=0.021,
            take_profit_pct=0.110,
            lock_trigger_pct=0.024,
            lock_level_pct=0.008,
            trail_trigger_pct=0.035,
            trail_distance_pct=0.017,
            max_hold_hours=12.0,
        ),
    }

    min_eval_s, max_eval_s = int(ts_eval[0]), int(ts_eval[-1]) + 3600

    def _eval_symbol(sym: str) -> list[dict[str, Any]]:
        df1 = symbol_1m_loader(sym)
        if df1 is None or df1.empty:
            return []
        d1 = df1.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        ts_a = d1["timestamp"].to_numpy(dtype=np.int64)
        o_a = d1["open"].to_numpy(dtype=np.float64)
        h_a = d1["high"].to_numpy(dtype=np.float64)
        l_a = d1["low"].to_numpy(dtype=np.float64)
        c_a = d1["close"].to_numpy(dtype=np.float64)

        evs_here = list(by_sym_evs.get(sym, []))
        if sym in m04_syms and sym in f_series_map:
            m_sub = (ts_a >= min_eval_s) & (ts_a < max_eval_s)
            if np.any(m_sub):
                qv_a = d1["quote_volume"].to_numpy(dtype=np.float64)
                tb_a = d1["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
                d_idx = pd.DatetimeIndex(pd.to_datetime(ts_a[m_sub], unit="s", utc=True))
                r15 = (
                    pd.DataFrame(
                        {"high": h_a[m_sub], "close": c_a[m_sub], "qv": qv_a[m_sub], "tb": tb_a[m_sub]}, index=d_idx
                    )
                    .resample("15min", label="left", closed="left")
                    .agg({"high": "max", "close": "last", "qv": "sum", "tb": "sum"})
                    .reindex(idx_15m)
                )
                for en_ts, _z, side, _st, _sc, rk_v in sn_eng.scan_historical_15m_events(
                    ts_15m,
                    r15["close"].to_numpy(dtype=np.float64)[:, None],
                    r15["high"].to_numpy(dtype=np.float64)[:, None],
                    r15["qv"].fillna(0.0).to_numpy(dtype=np.float64)[:, None],
                    r15["tb"].fillna(0.0).to_numpy(dtype=np.float64)[:, None],
                    f_series_map[sym].reindex(idx_15m, method="ffill").fillna(0.0).to_numpy(dtype=np.float64)[:, None],
                    rk_df_1h[sym].reindex(idx_15m, method="ffill").fillna(999).to_numpy(dtype=np.int64)[:, None],
                ):
                    evs_here.append((en_ts, side, "M04_NEGFUND_LONG", rk_v))

        if not evs_here:
            return []

        # Fetch causal daily L2 `bookDepth` archives ONLY for unique event dates of `sym`,
        # parallelized across 16 HTTP threads and immediately sliced to `[-660s, 0s]` around event timestamps (`<10 KB` RAM!).
        bd_merged: pd.DataFrame | None = None
        if fetch_live_book_depth:
            by_date_ts: dict[str, list[int]] = {}
            for en_ts, *_ in evs_here:
                d_s = datetime.fromtimestamp(en_ts - 60, tz=timezone.utc).strftime("%Y-%m-%d")
                by_date_ts.setdefault(d_s, []).append(int(en_ts))
            sess_bd = _build_pooled_session(pool_size=20)
            tks = PERP_TICKER_ALIASES.get(sym, (sym,))

            def _fetch_bd_day(d_str: str, d_ts_list: list[int]) -> pd.DataFrame | None:
                lo_t = min(d_ts_list) - 660
                hi_t = max(d_ts_list)
                for tk in tks:
                    url = f"{VISION_CDN_BASE}/data/futures/um/daily/bookDepth/{tk}/{tk}-bookDepth-{d_str}.zip"
                    try:
                        rb = sess_bd.get(url, timeout=15)
                        if rb.status_code == 200 and rb.content:
                            p_bd = parse_book_depth_zip_bytes(rb.content, sym)
                            if p_bd is not None and not p_bd.empty:
                                sub_bd = p_bd[(p_bd["timestamp"] >= lo_t) & (p_bd["timestamp"] <= hi_t)]
                                return sub_bd.reset_index(drop=True) if not sub_bd.empty else None
                            return None
                        if rb.status_code == 404:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                return None

            bd_parts: list[pd.DataFrame] = []
            with ThreadPoolExecutor(max_workers=min(16, max(1, len(by_date_ts)))) as bd_pool:
                bd_futs = [bd_pool.submit(_fetch_bd_day, ds, tl) for ds, tl in sorted(by_date_ts.items())]
                for bf_i in as_completed(bd_futs):
                    r_bd = bf_i.result()
                    if r_bd is not None and not r_bd.empty:
                        bd_parts.append(r_bd)
            sess_bd.close()
            if bd_parts:
                bd_merged = (
                    pd.concat(bd_parts, ignore_index=True)
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last")
                )
            del bd_parts

        fr_sym = f_maps.get(sym, {})
        rows_out: list[dict[str, Any]] = []
        for en_ts, side, base_id, rk_v in evs_here:
            res = resolve_sniper_1m_path(
                entry_ts=en_ts,
                side=side,
                spec=bracket_map[base_id],
                adv_rank=rk_v,
                ts_1m=ts_a,
                open_1m=o_a,
                high_1m=h_a,
                low_1m=l_a,
                close_1m=c_a,
                fr_map_sym=fr_sym,
            )
            if res is None:
                continue
            ex_ts, hr_ret, _is_loss, _r_str = res
            tot_net_bps = float(sum(hr_ret.values())) * 10_000.0
            est_fee_bps = 12.0 if rk_v < 20 else 15.0
            imb1, _imb2 = lookup_causal_l2_book_imbalance(bd_merged, en_ts)
            yr_e = datetime.fromtimestamp(int(en_ts), tz=timezone.utc).year
            era_e = "2023_2024" if yr_e <= 2024 else ("2025" if yr_e == 2025 else "2026")
            str_v = float(max(1, 100 - rk_v))

            def _mk_row(
                m_name: str,
                _b: str = base_id,
                _en: int = en_ts,
                _ex: int = ex_ts,
                _er: str = era_e,
                _st: float = str_v,
                _nt: float = tot_net_bps,
                _fe: float = est_fee_bps,
            ) -> dict[str, Any]:
                return {
                    "mech_id": m_name,
                    "pillar": _b,
                    "exec_mode": "1M_BRACKET",
                    "symbol": sym,
                    "entry_ts": int(_en),
                    "exit_ts": int(max(_en + 60, _ex)),
                    "era": _er,
                    "strength": _st,
                    "gross_bps": _nt + _fe,
                    "funding_bps": 0.0,
                    "fee_bps": _fe,
                    "net_bps": _nt,
                }

            # 1) RAW variant (always emitted)
            rows_out.append(_mk_row(f"{base_id}__RAW"))
            # 2) L2 Wall-Veto (`no heavy wall blocking our direction`; passes through if pre-2023/missing book)
            no_wall_ok = (imb1 is None) or (side > 0 and imb1 >= -0.15) or (side < 0 and imb1 <= 0.15)
            if no_wall_ok:
                rows_out.append(_mk_row(f"{base_id}__L2_NO_WALL"))
            # 3) Strict L2 Air-Pocket / Wall Confirmation (`requires explicit L2 support`)
            strict_l2_ok = (imb1 is not None) and ((side > 0 and imb1 >= 0.12) or (side < 0 and imb1 <= -0.12))
            if strict_l2_ok:
                rows_out.append(_mk_row(f"{base_id}__L2_STRICT_IMB"))
        del df1, d1, bd_merged
        return rows_out

    all_rows: list[dict[str, Any]] = []
    done = 0
    eff_sym_workers = max(1, min(2, int(workers)))
    for b_i in range(0, len(target_syms), eff_sym_workers):
        b_syms = target_syms[b_i : b_i + eff_sym_workers]
        with ThreadPoolExecutor(max_workers=len(b_syms)) as pool:
            futs = {pool.submit(_eval_symbol, s): s for s in b_syms}
            for fut in as_completed(futs):
                all_rows.extend(fut.result())
                done += 1
                if done % 10 == 0 or done == len(target_syms):
                    print(
                        f"  [FrenteB-L2] Processed {done}/{len(target_syms)} sniper symbols | rows={len(all_rows)}...",
                        flush=True,
                    )
        gc.collect()

    tr_df = pd.DataFrame(all_rows)
    if tr_df.empty:
        return pd.DataFrame(), tr_df
    tr_df.sort_values(["mech_id", "entry_ts"], inplace=True)
    tr_df.reset_index(drop=True, inplace=True)
    lb = evaluate_infold_cpcv_and_treasure_metrics(tr_df, total_trials=max(1, int(tr_df["mech_id"].nunique())))
    return lb, tr_df


def build_causal_1h_panic_flush_anchors(
    bars_1h_by_sym: dict[str, pd.DataFrame],
    eval_start: str = "2023-03-01",
    max_rank: int = 25,
    ret_thresh: float = -0.032,
    vol_mult: float = 2.8,
    tb_max: float = 0.45,
    cooldown_sec: int = 21_600,
) -> pd.DataFrame:
    """
    Constructs `100%` strictly causal (`0.000 ms` lookahead) `1h` panic-flush anchor timestamps
    `entry_ts = ts_1h[t] + 3600` (strictly at the CLOSE of the panic bar!) across liquid perps.
    """
    syms = [s for s in SNIPER_UNIVERSE_82 if s in bars_1h_by_sym]
    if not syms or "BTCUSDT" not in bars_1h_by_sym:
        return pd.DataFrame(columns=["symbol", "entry_ts"])

    btc_df = bars_1h_by_sym["BTCUSDT"].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    full_idx = pd.DatetimeIndex(pd.to_datetime(btc_df["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
    eval_idx = full_idx[full_idx >= pd.Timestamp(eval_start, tz="UTC")]
    n_b, ns = len(eval_idx), len(syms)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    c_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    o_1h = np.full((n_b, ns), np.nan, dtype=np.float64)
    qv_1h = np.zeros((n_b, ns), dtype=np.float64)
    tb_1h = np.zeros((n_b, ns), dtype=np.float64)
    for j_s, s_sym in enumerate(syms):
        bs = bars_1h_by_sym[s_sym].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        b_idx = pd.DatetimeIndex(pd.to_datetime(bs["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        o_1h[:, j_s] = pd.Series(bs["open"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        c_1h[:, j_s] = pd.Series(bs["close"].to_numpy(dtype=np.float64), index=b_idx).reindex(eval_idx).to_numpy()
        qv_1h[:, j_s] = (
            pd.Series(bs["quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )
        tb_1h[:, j_s] = (
            pd.Series(bs["taker_buy_quote_volume"].to_numpy(dtype=np.float64), index=b_idx)
            .reindex(eval_idx)
            .fillna(0.0)
            .to_numpy()
        )

    adv30 = pd.DataFrame(qv_1h).rolling(24 * 30, min_periods=48).mean().shift(1).fillna(0.0)
    rank_1h = adv30.rank(axis=1, ascending=False, method="first").fillna(999.0).to_numpy(dtype=np.int64) - 1
    med24 = pd.DataFrame(qv_1h).rolling(24, min_periods=12).median().shift(1).to_numpy(dtype=np.float64)
    tb_share = np.where(qv_1h > 1e-8, tb_1h / np.maximum(qv_1h, 1e-8), 0.5)

    rows: list[dict[str, Any]] = []
    last_ts_by_sym: dict[int, int] = {}
    for t in range(30, n_b - 2):
        en_ts = int(ts_eval[t]) + 3600
        for j_s in range(ns):
            if rank_1h[t, j_s] >= max_rank:
                continue
            if not (
                np.isfinite(o_1h[t, j_s])
                and np.isfinite(c_1h[t, j_s])
                and o_1h[t, j_s] > 0
                and np.isfinite(med24[t, j_s])
            ):
                continue
            ret_bar = (c_1h[t, j_s] - o_1h[t, j_s]) / o_1h[t, j_s]
            if (
                ret_bar <= ret_thresh
                and qv_1h[t, j_s] >= vol_mult * med24[t, j_s]
                and tb_share[t, j_s] <= tb_max
                and en_ts - last_ts_by_sym.get(j_s, -(10**9)) >= cooldown_sec
            ):
                last_ts_by_sym[j_s] = en_ts
                rows.append({"symbol": syms[j_s], "entry_ts": en_ts})
    return pd.DataFrame(rows)


def evaluate_frente_b_sniper_portfolio_ablation(
    bundle: dict[str, Any],
    m26_trades_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Evaluates the portfolio-level impact (`4Y` + fresh `$2,700` `2026` start against `$2,500` floor)
    of Frente B's `Part B1` Sniper discoveries (`M09_SHORT` decay & `M04` `-43 bps` `2026` decay).
    """
    from src.backtest.core_rebalance_barrier_lab import (
        CoreRebalanceBarrierSpec,
        compute_cpcv_portfolio_sharpe_stability,
        simulate_core_barrier_1m_portfolio,
    )
    from src.backtest.p2_portfolio_integration import (
        compute_portfolio_era_metrics,
        convert_trade_df_to_hourly_sniper_events,
    )
    from src.strategies.event_snipers import SNIPER_M26_STRAT

    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    mats: dict[str, Any] = bundle["mats"]
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]
    p2_w_mat: np.ndarray = bundle["p2_w_mat"]

    base_evs = list(bundle.get("base_sniper_events", []))
    m26_evs: list[dict[str, Any]] = []
    if m26_trades_df is not None and not m26_trades_df.empty:
        sub_26 = m26_trades_df[
            (m26_trades_df["mech_id"] == "P6A_L2_3_INST_SWEEP_CONC")
            & (m26_trades_df["exec_mode"] == "MONO_A_CONVEX_TRAIL")
        ]
        if not sub_26.empty:
            m26_evs = list(convert_trade_df_to_hourly_sniper_events(sub_26, SNIPER_M26_STRAT))

    cb_strict = CoreRebalanceBarrierSpec("STRICT_8H", "Strict 8h", intracycle_mode="off")
    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0

    configs = [
        ("FB0_CURRENT_PROD_ALL_SNIPERS", "Baseline Live Snipers (M09 L+S, M23, M04, M26)"),
        ("FB1_M09_LONG_ONLY_DROP_M09_SHORT", "Drop M09_SHORT (Keep M09_LONG, M23, M04, M26)"),
        ("FB2_DROP_M04_NEGFUND", "Drop M04 (-43bps 2026 decay; Keep M09 L+S, M23, M26)"),
        ("FB3_DROP_M09_SHORT_AND_M04", "Keep ONLY M09_LONG + M23 + M26 (Drop M09_SHORT & M04)"),
        ("FB4_M09_LONG_PRIORITY_OVER_SHORT", "Boost M09_LONG score (+10) over M09_SHORT + Drop M04"),
    ]

    rows: list[dict[str, Any]] = []
    for cfg_id, desc in configs:
        evs_hr: dict[int, list[dict[str, Any]]] = {}
        allowed = {SNIPER_M09_STRAT, SNIPER_M23_STRAT, SNIPER_M04_STRAT, SNIPER_M26_STRAT}
        if cfg_id in ("FB2_DROP_M04_NEGFUND", "FB3_DROP_M09_SHORT_AND_M04", "FB4_M09_LONG_PRIORITY_OVER_SHORT"):
            allowed.discard(SNIPER_M04_STRAT)

        for ev in base_evs:
            st = ev["strat"]
            sd = int(ev.get("side", 1))
            if st not in allowed:
                continue
            if (
                cfg_id in ("FB1_M09_LONG_ONLY_DROP_M09_SHORT", "FB3_DROP_M09_SHORT_AND_M04")
                and st == SNIPER_M09_STRAT
                and sd < 0
            ):
                continue
            ev_use = dict(ev)
            if cfg_id == "FB4_M09_LONG_PRIORITY_OVER_SHORT" and st == SNIPER_M09_STRAT and sd > 0:
                ev_use["score"] = float(ev_use.get("score", 1.0)) + 10.0
            evs_hr.setdefault(int(ev_use["hr_ts"]), []).append(ev_use)
        for ev in m26_evs:
            evs_hr.setdefault(int(ev["hr_ts"]), []).append(ev)

        sim_4y = simulate_core_barrier_1m_portfolio(
            mats=mats,
            o_1m_core=o_1m_core,
            h_1m_core=h_1m_core,
            l_1m_core=l_1m_core,
            eval_idx=eval_idx,
            open_core=open_core,
            p2_w_mat=p2_w_mat,
            spec=cb_strict,
            resolved_sniper_events_by_hour=evs_hr,
            allowed_sniper_strats=allowed,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            evs_26 = {k: v for k, v in evs_hr.items() if k >= ts_2026_start}
            sim_26 = simulate_core_barrier_1m_portfolio(
                mats=mats,
                o_1m_core=o_1m_core[mask_2026],
                h_1m_core=h_1m_core[mask_2026],
                l_1m_core=l_1m_core[mask_2026],
                eval_idx=idx_2026,
                open_core=open_core[mask_2026],
                p2_w_mat=p2_w_mat[mask_2026],
                spec=cb_strict,
                resolved_sniper_events_by_hour=evs_26,
                allowed_sniper_strats=allowed,
            )
            fresh_final = round(float(sim_26["final_eq"]), 2)
            fresh_min = round(float(sim_26["min_eq"]), 2)
            fresh_ok = bool(fresh_min >= 2500.0)

        rows.append(
            {
                "config_id": cfg_id,
                "description": desc,
                "sharpe_4y": m_4y["full_sharpe"],
                "cagr_4y_pct": m_4y["full_cagr_pct"],
                "max_dd_4y_pct": m_4y["full_max_dd_pct"],
                "sharpe_23_24": m_4y["sharpe_23_24"],
                "sharpe_2025": m_4y["sharpe_2025"],
                "sharpe_2026": m_4y["sharpe_2026"],
                "cpcv_mean_sharpe": cpcv_4y["cpcv_mean_sharpe"],
                "cpcv_pos_splits_pct": cpcv_4y["cpcv_pos_splits_pct"],
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_ok,
                "sniper_counts": m_4y["sniper_counts"],
            }
        )
    return pd.DataFrame(rows)
