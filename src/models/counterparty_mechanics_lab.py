"""
1-Minute Linear-Return Counterparty-Mechanics & Forced-Flow Laboratory.

Implements non-consensus, first-principles market mechanics directly on native
`1-minute` Spot + `1-minute` Perp + `+60s`-lagged `5-minute` Derivatives Positioning
(`OI`, `Top-Trader vs Retail L/S`) + exact `:00:00` Dynamic Funding Settlements (`1h/2h/4h/8h`)
across the 220 Point-in-Time USDS-M perpetual universe (`2023–2026`):

Strict Scientific & Anti-Self-Deception Guarantees:
  1. ZERO `np.log` Returns: 100% exact linear perpetual PnL `side * (P_exit - P_entry) / P_entry`,
     eliminating `+0.5 * sigma^2` Jensen convexity inflation on volatile Shorts.
  2. Native `1-Minute` Intrabar Execution & Bracket Physics:
     Compares `TAKER_OPEN` (`open[t+1]` + 6.5 bps fee+slip), `TOUCH_LIMIT` (`0 bps` offset,
     1.5 bps price-through, 2.0 bps Maker fee), and `GTX_1.5BPS` (`+1.5 bps` passive offset,
     2.0 bps price-through) under `100%` pessimistic same-bar Stop-Loss priority.
  3. Exact Realized `:00:00` Funding Cashflows:
     Credits/debits `-side * realized_funding_rate` at every settlement crossed while open.
  4. True Out-of-Sample In-Fold `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits):
     Calibrates physical intensity threshold `theta*` strictly on `train_days` (with `24h`
     embargo around `test_days`) inside EVERY fold before scoring `test_days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.purged_cv import CombinatorialPurgedCV, compute_deflated_sharpe_ratio


@dataclass(frozen=True)
class CounterpartyMechanismSpec:
    """Physical specification and 1m bracket profile for a Counterparty-Mechanics hypothesis."""

    mech_id: str
    pillar: str
    description: str
    tp_pct: float
    sl_pct: float
    lock_trigger_pct: float
    lock_level_pct: float
    trail_trigger_pct: float
    trail_distance_pct: float
    max_hold_minutes: int
    cooldown_minutes: int = 30
    pre_settlement_exit_minute: int | None = None


COUNTERPARTY_MECHANISM_SPECS: dict[str, CounterpartyMechanismSpec] = {
    "M00_NAIVE_M25_CONTROL": CounterpartyMechanismSpec(
        mech_id="M00_NAIVE_M25_CONTROL",
        pillar="Control_M16_6_Baseline",
        description="Unfiltered M25 post-settle SHORT (T+1m, FR <= -20 bps, hold<=54m, :56 guard)",
        tp_pct=0.032,
        sl_pct=0.022,
        lock_trigger_pct=0.010,
        lock_level_pct=0.0045,
        trail_trigger_pct=0.016,
        trail_distance_pct=0.0075,
        max_hold_minutes=54,
        cooldown_minutes=45,
        pre_settlement_exit_minute=56,
    ),
    "C1A_POST_SETTLE_HARVESTER_SHORT": CounterpartyMechanismSpec(
        mech_id="C1A_POST_SETTLE_HARVESTER_SHORT",
        pillar="Pillar_1_PostFunding_Bifurcation_StateA",
        description="State-A Pure Harvester Overhang SHORT (vetoes deep Spot premium, requires Perp basis markup + :00 dump confirm)",
        tp_pct=0.035,
        sl_pct=0.020,
        lock_trigger_pct=0.010,
        lock_level_pct=0.0045,
        trail_trigger_pct=0.016,
        trail_distance_pct=0.0075,
        max_hold_minutes=54,
        cooldown_minutes=45,
        pre_settlement_exit_minute=56,
    ),
    "C1B_PRE_SETTLE_SPOT_SCARCITY_LONG": CounterpartyMechanismSpec(
        mech_id="C1B_PRE_SETTLE_SPOT_SCARCITY_LONG",
        pillar="Pillar_1_PostFunding_Bifurcation_StateB",
        description="State-B Pre-Settle (T-45m) LONG when Spot >> Perp (Basis <= -45 bps) & FR <= -30 bps (collects +funding + squeeze)",
        tp_pct=0.055,
        sl_pct=0.022,
        lock_trigger_pct=0.015,
        lock_level_pct=0.0070,
        trail_trigger_pct=0.025,
        trail_distance_pct=0.0110,
        max_hold_minutes=75,
        cooldown_minutes=60,
        pre_settlement_exit_minute=None,
    ),
    "C1C_POST_SETTLE_SHORT_PANIC_LONG": CounterpartyMechanismSpec(
        mech_id="C1C_POST_SETTLE_SHORT_PANIC_LONG",
        pillar="Pillar_1_PostFunding_Bifurcation_StateB",
        description="State-B Post-Settle (T+1m) LONG when Spot >> Perp (Basis <= -45 bps, FR <= -25 bps) & :00 candle refuses to dump",
        tp_pct=0.060,
        sl_pct=0.020,
        lock_trigger_pct=0.016,
        lock_level_pct=0.0075,
        trail_trigger_pct=0.028,
        trail_distance_pct=0.0120,
        max_hold_minutes=90,
        cooldown_minutes=60,
        pre_settlement_exit_minute=None,
    ),
    "C2A_TRAPPED_SHORTS_PAIN_LONG": CounterpartyMechanismSpec(
        mech_id="C2A_TRAPPED_SHORTS_PAIN_LONG",
        pillar="Pillar_2_Trapped_OI_Pain_Tomography",
        description="Recursive OI-VWAP Trapped Shorts Detonation LONG (OI >= 1.18x 24h, Pain >= +2.2 ATR, FR <= -8 bps, 4h High Break + Spot Flow)",
        tp_pct=0.070,
        sl_pct=0.021,
        lock_trigger_pct=0.018,
        lock_level_pct=0.0080,
        trail_trigger_pct=0.032,
        trail_distance_pct=0.0140,
        max_hold_minutes=180,
        cooldown_minutes=90,
        pre_settlement_exit_minute=None,
    ),
    "C2B_TRAPPED_LONGS_WATERFALL_SHORT": CounterpartyMechanismSpec(
        mech_id="C2B_TRAPPED_LONGS_WATERFALL_SHORT",
        pillar="Pillar_2_Trapped_OI_Pain_Tomography",
        description="Recursive OI-VWAP Trapped Retail Longs Waterfall SHORT (OI >= 1.18x 24h, Pain >= +2.2 ATR below OI-VWAP, FR >= +8 bps, 4h Low Break)",
        tp_pct=0.060,
        sl_pct=0.021,
        lock_trigger_pct=0.018,
        lock_level_pct=0.0080,
        trail_trigger_pct=0.030,
        trail_distance_pct=0.0130,
        max_hold_minutes=180,
        cooldown_minutes=90,
        pre_settlement_exit_minute=None,
    ),
    "C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG": CounterpartyMechanismSpec(
        mech_id="C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG",
        pillar="Pillar_3_Spot_Perp_Liquidation_Elasticity",
        description="1m Post-Liquidation Snap-Back LONG (10m Perp drop <= -2.5%, 15m OI flush <= -2.5%, Perp dislocated <= -25 bps below Spot + 1m reversal)",
        tp_pct=0.038,
        sl_pct=0.017,
        lock_trigger_pct=0.012,
        lock_level_pct=0.0055,
        trail_trigger_pct=0.020,
        trail_distance_pct=0.0090,
        max_hold_minutes=60,
        cooldown_minutes=30,
        pre_settlement_exit_minute=None,
    ),
    "C3B_SPOT_ANCHORED_BLOWOFF_SHORT": CounterpartyMechanismSpec(
        mech_id="C3B_SPOT_ANCHORED_BLOWOFF_SHORT",
        pillar="Pillar_3_Spot_Perp_Liquidation_Elasticity",
        description="1m Post-Short-Liquidation Blow-Off SHORT (10m Perp spike >= +3.0%, 15m OI flush <= -2.5%, Perp dislocated >= +30 bps above Spot + 1m turn)",
        tp_pct=0.038,
        sl_pct=0.017,
        lock_trigger_pct=0.012,
        lock_level_pct=0.0055,
        trail_trigger_pct=0.020,
        trail_distance_pct=0.0090,
        max_hold_minutes=60,
        cooldown_minutes=30,
        pre_settlement_exit_minute=None,
    ),
    "C4_CLOCK_COMPRESSION_SQUEEZE_LONG": CounterpartyMechanismSpec(
        mech_id="C4_CLOCK_COMPRESSION_SQUEEZE_LONG",
        pillar="Pillar_4_Exchange_Clock_Phase_Transition",
        description="Discrete Binance Funding-Clock Compression Transition (8h -> 4h/2h/1h) LONG when FR <= -20 bps & Spot >= Perp",
        tp_pct=0.065,
        sl_pct=0.024,
        lock_trigger_pct=0.018,
        lock_level_pct=0.0080,
        trail_trigger_pct=0.032,
        trail_distance_pct=0.0140,
        max_hold_minutes=180,
        cooldown_minutes=120,
        pre_settlement_exit_minute=None,
    ),
    "C5_SOVEREIGN_CASH_WAVE_IGNITION": CounterpartyMechanismSpec(
        mech_id="C5_SOVEREIGN_CASH_WAVE_IGNITION",
        pillar="Pillar_5_Event_Time_Propagation_Wave",
        description="Sovereign 3m BTC Spot+Perp Impulse (>=0.75% on >=3.5x RVOL & >=62% Spot Taker) into lagging high-beta perps",
        tp_pct=0.032,
        sl_pct=0.014,
        lock_trigger_pct=0.010,
        lock_level_pct=0.0045,
        trail_trigger_pct=0.018,
        trail_distance_pct=0.0080,
        max_hold_minutes=45,
        cooldown_minutes=30,
        pre_settlement_exit_minute=None,
    ),
}


EXECUTION_MODES: tuple[tuple[str, float, float, int, float], ...] = (
    # (mode_name, limit_offset_bps_or_neg_for_taker, queue_penetration_bps, ttl_min, entry_cost_bps)
    ("TAKER_OPEN", -999.0, 0.0, 1, 6.5),
    ("TOUCH_LIMIT", 0.0, 1.5, 3, 2.0),
    ("GTX_1.5BPS", 1.5, 2.0, 3, 2.0),
)


def _fast_rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal O(N) 1D rolling mean using float64 cumulative sum (`out[i]` uses `arr[max(0, i-w+1)..i]`)."""
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    w = max(1, int(window))
    x = np.nan_to_num(arr.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(x, out=cs[1:])
    idx = np.arange(n, dtype=np.int64)
    starts = np.maximum(0, idx - w + 1)
    counts = (idx - starts + 1).astype(np.float64)
    return (cs[idx + 1] - cs[starts]) / counts


def _fast_ema_1d(arr: np.ndarray, span: int) -> np.ndarray:
    """Causal O(N) exponential moving average."""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(max(1, span)) + 1.0)
    one_m = 1.0 - alpha
    acc = float(arr[0]) if np.isfinite(arr[0]) else 0.0
    out[0] = acc
    for i in range(1, n):
        v = float(arr[i])
        if np.isfinite(v):
            acc = alpha * v + one_m * acc
        out[i] = acc
    return out


def compute_recursive_oi_vwap_anchor(
    oi_usd_5m: np.ndarray,
    price_5m: np.ndarray,
    base_decay: float = 0.992,
) -> np.ndarray:
    """
    Computes the recursive OI-Addition-Weighted Average Entry Cost Basis `P_oi_vwap(t)`:
      - When `ΔOI_t > 0`, new leveraged contracts opened at `price_5m[t]` blend into the active
        cost basis with weight `w_add = clip(ΔOI_t / OI_t, 0.0, 1.0)`.
      - Gentle background turnover decay (`1 - base_decay`) accounts for contract churn when OI is flat.
    Strictly causal (`t` depends only on `<= t`).
    """
    n = len(oi_usd_5m)
    anchor = np.empty(n, dtype=np.float64)
    if n == 0:
        return anchor
    cur_p = float(price_5m[0]) if np.isfinite(price_5m[0]) and price_5m[0] > 0 else 1.0
    prev_oi = float(oi_usd_5m[0]) if np.isfinite(oi_usd_5m[0]) and oi_usd_5m[0] > 0 else 0.0
    anchor[0] = cur_p
    bg_alpha = 1.0 - float(base_decay)

    for i in range(1, n):
        px = float(price_5m[i])
        oi = float(oi_usd_5m[i])
        if not np.isfinite(px) or px <= 0.0:
            anchor[i] = cur_p
            continue
        if np.isfinite(oi) and oi > 0.0 and prev_oi > 0.0:
            d_oi = max(0.0, oi - prev_oi)
            w_new = min(1.0, max(bg_alpha, d_oi / oi))
            cur_p = (1.0 - w_new) * cur_p + w_new * px
            prev_oi = oi
        else:
            cur_p = (1.0 - bg_alpha) * cur_p + bg_alpha * px
            if np.isfinite(oi) and oi > 0.0:
                prev_oi = oi
        anchor[i] = cur_p
    return anchor


def extract_sovereign_btc_shocks(
    btc_perp_1m: pd.DataFrame,
    btc_spot_1m: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """
    Pre-computes 1-minute BTC returns and discrete Sovereign Cash+Perp Impulse shocks (`C5`)
    so every altcoin shard can align in `O(log N)` via `np.searchsorted`.
    """
    if btc_perp_1m is None or btc_perp_1m.empty:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return {"ts": empty_i, "ret_1m": empty_f, "ret_3m": empty_f, "shock_dir": empty_i, "shock_strength": empty_f}

    df = btc_perp_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    c = df["close"].to_numpy(dtype=np.float64)
    qv = df["quote_volume"].to_numpy(dtype=np.float64)
    tb = df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)

    n = len(ts)
    ret_1m = np.zeros(n, dtype=np.float64)
    ret_1m[1:] = (c[1:] - c[:-1]) / np.maximum(c[:-1], 1e-12)
    ret_3m = np.zeros(n, dtype=np.float64)
    if n > 3:
        ret_3m[3:] = (c[3:] - c[:-3]) / np.maximum(c[:-3], 1e-12)

    qv_3m = _fast_rolling_mean_1d(qv, 3) * 3.0
    qv_base = np.roll(_fast_rolling_mean_1d(qv, 180) * 3.0, 3)
    qv_base[:4] = qv_3m[:4]
    rvol_3m = qv_3m / np.maximum(qv_base, 1.0)

    if btc_spot_1m is not None and not btc_spot_1m.empty:
        sp = btc_spot_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        sp_ts = sp["timestamp"].to_numpy(dtype=np.int64)
        sp_qv = sp["quote_volume"].to_numpy(dtype=np.float64)
        sp_tb = sp["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        pos = np.searchsorted(sp_ts, ts)
        pos_c = np.minimum(pos, max(0, len(sp_ts) - 1))
        has_sp = (len(sp_ts) > 0) & (pos < len(sp_ts)) & (sp_ts[pos_c] == ts)
        sqv = np.where(has_sp, sp_qv[pos_c], qv)
        stb = np.where(has_sp, sp_tb[pos_c], tb)
    else:
        sqv, stb = qv, tb

    sp_share_3m = (_fast_rolling_mean_1d(stb, 3) * 3.0) / np.maximum(_fast_rolling_mean_1d(sqv, 3) * 3.0, 1.0)
    up_shock = (ret_3m >= 0.0075) & (rvol_3m >= 3.5) & (sp_share_3m >= 0.62)
    dn_shock = (ret_3m <= -0.0075) & (rvol_3m >= 3.5) & (sp_share_3m <= 0.38)

    shock_dir = np.where(up_shock, 1, np.where(dn_shock, -1, 0)).astype(np.int64)
    shock_strength = np.abs(ret_3m) * 1e4 * np.minimum(rvol_3m, 10.0)
    return {
        "ts": ts,
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "shock_dir": shock_dir,
        "shock_strength": shock_strength,
    }


def simulate_1m_event_across_modes(
    ts_1m: np.ndarray,
    open_1m: np.ndarray,
    high_1m: np.ndarray,
    low_1m: np.ndarray,
    close_1m: np.ndarray,
    qvol_1m: np.ndarray,
    funding_map: dict[int, float],
    raw_events: list[tuple[int, int, float, float]],
    spec: CounterpartyMechanismSpec,
    symbol: str,
    order_notional_usd: float = 600.0,
    max_bar_participation: float = 0.025,
) -> list[dict[str, Any]]:
    """
    Simulates a list of `(sig_idx, side, strength, aux_val)` events for `spec` across all 3
    execution modes (`TAKER_OPEN`, `TOUCH_LIMIT`, `GTX_1.5BPS`) with exact linear returns,
    `100%` pessimistic same-bar SL priority, and exact `:00:00` funding settlements.
    """
    if not raw_events or len(ts_1m) < 10:
        return []

    n_bars = len(ts_1m)
    sorted_evs = sorted(raw_events, key=lambda x: x[0])
    min_req_qvol = order_notional_usd / max(1e-6, max_bar_participation)
    taker_exit_cost = 6.5 / 10_000.0  # 5.0 bps VIP0 taker + 1.5 bps stop/time slippage
    maker_tp_cost = 2.0 / 10_000.0
    tp_pen = 1.0 / 10_000.0

    out_records: list[dict[str, Any]] = []

    for mode_name, off_bps, pen_bps, ttl_m, entry_cost_bps in EXECUTION_MODES:
        next_free_idx = 0
        entry_fee_rate = entry_cost_bps / 10_000.0
        q_pen = pen_bps / 10_000.0
        is_taker_mode = off_bps < -100.0

        for sig_i, side_in, strength, aux_v in sorted_evs:
            if sig_i < next_free_idx or sig_i + 2 >= n_bars:
                continue
            sgn = 1.0 if side_in > 0 else -1.0
            ref_c = float(close_1m[sig_i])
            sub_i = sig_i + 1
            # Guard against multi-minute data gaps at signal boundary
            if int(ts_1m[sub_i]) - int(ts_1m[sig_i]) > 180 or ref_c <= 0.0 or not np.isfinite(ref_c):
                continue

            sub_o = float(open_1m[sub_i])
            if not np.isfinite(sub_o) or sub_o <= 0.0:
                continue

            if is_taker_mode:
                if float(qvol_1m[sub_i]) < min_req_qvol:
                    next_free_idx = sub_i + 1
                    continue
                fill_i = sub_i
                entry_px = sub_o
            else:
                off_frac = off_bps / 10_000.0
                raw_lim = ref_c * (1.0 - sgn * off_frac)
                crosses = (sgn > 0 and raw_lim >= sub_o) or (sgn < 0 and raw_lim <= sub_o)
                if crosses:
                    limit_px = sub_o * (1.0 - sgn * 0.0001)
                else:
                    limit_px = raw_lim
                req_px = limit_px * (1.0 - sgn * q_pen)
                ttl_end = min(n_bars, sub_i + ttl_m)
                fill_i = -1
                for m_try in range(sub_i, ttl_end):
                    if float(qvol_1m[m_try]) < min_req_qvol:
                        continue
                    if (sgn > 0 and float(low_1m[m_try]) <= req_px) or (sgn < 0 and float(high_1m[m_try]) >= req_px):
                        fill_i = m_try
                        break
                if fill_i < 0:
                    next_free_idx = ttl_end
                    continue
                entry_px = limit_px

            cur_sl = entry_px * (1.0 - sgn * spec.sl_pct)
            tp_px = entry_px * (1.0 + sgn * spec.tp_pct)
            tp_req = tp_px * (1.0 + sgn * tp_pen)
            peak_fav = entry_px

            hold_end = min(n_bars - 1, fill_i + max(1, spec.max_hold_minutes))
            exit_i = fill_i
            exit_px = float(close_1m[fill_i])
            exit_reason = "TIME_STOP"
            exit_cost_rate = taker_exit_cost
            funding_acc_ret = 0.0

            # Same-bar adverse selection check on `fill_i` (for passive limit fills; for Taker @ open,
            # open->high/low within fill_i can also hit initial SL, checked identically pessimistic!)
            hit_same_sl = (sgn > 0 and float(low_1m[fill_i]) <= cur_sl) or (
                sgn < 0 and float(high_1m[fill_i]) >= cur_sl
            )
            if hit_same_sl:
                exit_i = fill_i
                exit_px = cur_sl
                exit_reason = "STOP_LOSS_SAME_BAR"
                exit_cost_rate = taker_exit_cost
            else:
                settle_at_fill_close = int(ts_1m[fill_i]) + 60
                if settle_at_fill_close % 3600 == 0 and settle_at_fill_close in funding_map:
                    funding_acc_ret += -sgn * float(funding_map[settle_at_fill_close])

                for h in range(fill_i + 1, hold_end + 1):
                    t_h = int(ts_1m[h])
                    oh, hh, lh, ch = float(open_1m[h]), float(high_1m[h]), float(low_1m[h]), float(close_1m[h])

                    # 1. Pessimistic Stop-Loss priority FIRST
                    if (sgn > 0 and lh <= cur_sl) or (sgn < 0 and hh >= cur_sl):
                        exit_i = h
                        exit_px = min(oh, cur_sl) if sgn > 0 else max(oh, cur_sl)
                        is_locked = (sgn > 0 and cur_sl > entry_px) or (sgn < 0 and cur_sl < entry_px)
                        exit_reason = "PROFIT_LOCK_SL" if is_locked else "STOP_LOSS"
                        exit_cost_rate = taker_exit_cost
                        break

                    # 2. Passive Maker Take-Profit with strict price-through
                    if (sgn > 0 and hh >= tp_req) or (sgn < 0 and lh <= tp_req):
                        exit_i = h
                        exit_px = max(oh, tp_px) if sgn > 0 else min(oh, tp_px)
                        exit_reason = "TAKE_PROFIT_MAKER"
                        exit_cost_rate = maker_tp_cost
                        break

                    # 3. Pre-settlement exit guard (e.g., minute >= 56 closes BEFORE `:00:00` settlement)
                    min_of_hr = (t_h % 3600) // 60
                    if (
                        spec.pre_settlement_exit_minute is not None
                        and (h - fill_i) >= 3
                        and min_of_hr >= spec.pre_settlement_exit_minute
                    ):
                        exit_i = h
                        exit_px = ch
                        exit_reason = "PRE_SETTLE_EXIT"
                        exit_cost_rate = taker_exit_cost
                        break

                    # 4. Realized `:00:00` funding settlement at close of minute `h` (`t_h + 60`)
                    t_close = t_h + 60
                    if t_close % 3600 == 0 and t_close in funding_map:
                        funding_acc_ret += -sgn * float(funding_map[t_close])

                    # 5. End-of-bar causal Profit-Lock & Trailing ratchet for subsequent bars `h + 1..`
                    if sgn > 0:
                        if hh > peak_fav:
                            peak_fav = hh
                        pr = (peak_fav - entry_px) / entry_px
                        if spec.trail_trigger_pct > 0 and pr >= spec.trail_trigger_pct:
                            cand_sl = peak_fav * (1.0 - spec.trail_distance_pct)
                            if ch > cand_sl:
                                cur_sl = max(cur_sl, cand_sl)
                        elif spec.lock_trigger_pct > 0 and pr >= spec.lock_trigger_pct:
                            cand_sl = entry_px * (1.0 + spec.lock_level_pct)
                            if ch > cand_sl:
                                cur_sl = max(cur_sl, cand_sl)
                    else:
                        if lh < peak_fav:
                            peak_fav = lh
                        pr = (entry_px - peak_fav) / entry_px
                        if spec.trail_trigger_pct > 0 and pr >= spec.trail_trigger_pct:
                            cand_sl = peak_fav * (1.0 + spec.trail_distance_pct)
                            if ch < cand_sl:
                                cur_sl = min(cur_sl, cand_sl)
                        elif spec.lock_trigger_pct > 0 and pr >= spec.lock_trigger_pct:
                            cand_sl = entry_px * (1.0 - spec.lock_level_pct)
                            if ch < cand_sl:
                                cur_sl = min(cur_sl, cand_sl)

                    if h == hold_end:
                        exit_i = h
                        exit_px = ch
                        exit_reason = "TIME_STOP"
                        exit_cost_rate = taker_exit_cost

            # 100% Exact Linear Return (ZERO `np.log`!)
            gross_linear_ret = sgn * (exit_px - entry_px) / entry_px
            total_fee_ret = entry_fee_rate + exit_cost_rate
            net_linear_ret = gross_linear_ret + funding_acc_ret - total_fee_ret

            out_records.append(
                {
                    "mech_id": spec.mech_id,
                    "exec_mode": mode_name,
                    "symbol": symbol,
                    "signal_ts": int(ts_1m[sig_i]) + 60,
                    "entry_ts": int(ts_1m[fill_i]),
                    "exit_ts": int(ts_1m[exit_i]) + 60,
                    "side": int(side_in),
                    "hold_min": int(exit_i - fill_i + 1),
                    "gross_bps": round(gross_linear_ret * 1e4, 3),
                    "funding_bps": round(funding_acc_ret * 1e4, 3),
                    "fee_bps": round(total_fee_ret * 1e4, 3),
                    "net_bps": round(net_linear_ret * 1e4, 3),
                    "exit_reason": exit_reason,
                    "strength": round(float(strength), 4),
                    "aux_metric": round(float(aux_v), 3),
                }
            )
            next_free_idx = exit_i + max(1, spec.cooldown_minutes)

    return out_records


def scan_symbol_counterparty_mechanisms_1m(
    symbol: str,
    perp_1m: pd.DataFrame,
    spot_1m: pd.DataFrame | None,
    metrics_5m: pd.DataFrame | None,
    funding_df: pd.DataFrame | None,
    btc_shocks: dict[str, np.ndarray] | None = None,
    min_adv_usd: float = 15_000_000.0,
    metrics_safety_lag_sec: int = 60,
) -> list[dict[str, Any]]:
    """
    Processes a single symbol's native `1m` Perp + `1m` Spot + `+60s` lagged `5m` OI/Whale/Retail
    + exact `:00:00` Funding settlements in pure C-vectorized NumPy, detecting and simulating all
    10 Counterparty-Mechanics hypotheses across all 3 execution modes (`TAKER_OPEN`, `TOUCH_LIMIT`, `GTX_1.5BPS`).
    """
    # Note: BTC/ETH are scanned for C1..C4; only C5 (alt lag to BTC) skips BTCUSDT itself.
    if perp_1m is None or len(perp_1m) < 2880:
        return []

    p_df = perp_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    ts = p_df["timestamp"].to_numpy(dtype=np.int64)
    o = p_df["open"].to_numpy(dtype=np.float64)
    h = p_df["high"].to_numpy(dtype=np.float64)
    low = p_df["low"].to_numpy(dtype=np.float64)
    c = p_df["close"].to_numpy(dtype=np.float64)
    qv = p_df["quote_volume"].to_numpy(dtype=np.float64)
    tb = p_df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
    n = len(ts)

    # Causal 7-day trailing ADV (`adv_7d_usd`, lagged by 1m) + 14-day listing maturity (`>= 20,160` 1m bars)
    adv_7d = np.roll(_fast_rolling_mean_1d(qv, 7 * 1440) * 1440.0, 1)
    adv_7d[0] = 0.0
    maturity_ok = np.arange(n) >= 14 * 1440
    liq_gate = maturity_ok & (adv_7d >= float(min_adv_usd)) & np.isfinite(c) & (c > 0.0)
    if not bool(liq_gate.any()):
        return []

    # 1m / 3m / 5m / 10m / 60m / 240m linear returns & ATR_1h
    ret_1m = np.zeros(n, dtype=np.float64)
    ret_1m[1:] = (c[1:] - c[:-1]) / np.maximum(c[:-1], 1e-12)
    ret_3m = np.zeros(n, dtype=np.float64)
    ret_3m[3:] = (c[3:] - c[:-3]) / np.maximum(c[:-3], 1e-12)
    ret_10m = np.zeros(n, dtype=np.float64)
    ret_10m[10:] = (c[10:] - c[:-10]) / np.maximum(c[:-10], 1e-12)
    ret_4h = np.zeros(n, dtype=np.float64)
    ret_4h[240:] = (c[240:] - c[:-240]) / np.maximum(c[:-240], 1e-12)

    perp_tb_share_1m = tb / np.maximum(qv, 1.0)
    perp_tb_share_5m = (_fast_rolling_mean_1d(tb, 5) * 5.0) / np.maximum(_fast_rolling_mean_1d(qv, 5) * 5.0, 1.0)
    qv_5m = _fast_rolling_mean_1d(qv, 5) * 5.0
    qv_5m_base = np.roll(_fast_rolling_mean_1d(qv, 360) * 5.0, 5)
    qv_5m_base[:6] = qv_5m[:6]
    rvol_5m = qv_5m / np.maximum(qv_5m_base, 1.0)

    tr_1m = np.maximum(h - low, np.abs(h - np.roll(c, 1)))
    tr_1m[0] = h[0] - low[0]
    atr_1h = np.maximum(_fast_rolling_mean_1d(tr_1m, 60) * np.sqrt(60.0), c * 0.003)

    # Align 1m Spot bars (exact timestamp match via `searchsorted`)
    has_spot = np.zeros(n, dtype=bool)
    basis_1m_bps = np.zeros(n, dtype=np.float64)
    basis_disloc_bps = np.zeros(n, dtype=np.float64)
    basis_markup_60m_bps = np.zeros(n, dtype=np.float64)
    spot_tb_share_3m = perp_tb_share_5m.copy()
    spot_tb_share_60m = (_fast_rolling_mean_1d(tb, 60) * 60.0) / np.maximum(_fast_rolling_mean_1d(qv, 60) * 60.0, 1.0)

    if spot_1m is not None and not spot_1m.empty:
        s_df = spot_1m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        s_ts = s_df["timestamp"].to_numpy(dtype=np.int64)
        s_c = s_df["close"].to_numpy(dtype=np.float64)
        s_qv = s_df["quote_volume"].to_numpy(dtype=np.float64)
        s_tb = s_df["taker_buy_quote_volume"].to_numpy(dtype=np.float64)
        if len(s_ts) > 100:
            pos_sp = np.searchsorted(s_ts, ts)
            pos_sp_c = np.minimum(pos_sp, len(s_ts) - 1)
            matched_sp = (pos_sp < len(s_ts)) & (s_ts[pos_sp_c] == ts) & (s_c[pos_sp_c] > 0.0)
            # Parity sanity check (`0.5 <= Perp / Spot <= 2.0`)
            sp_px_aligned = np.where(matched_sp, s_c[pos_sp_c], c)
            parity_ok = (
                matched_sp
                & (c / np.maximum(sp_px_aligned, 1e-12) >= 0.5)
                & (c / np.maximum(sp_px_aligned, 1e-12) <= 2.0)
            )
            has_spot = parity_ok
            raw_b = np.where(has_spot, ((c - sp_px_aligned) / np.maximum(sp_px_aligned, 1e-12)) * 1e4, 0.0)
            basis_1m_bps = np.clip(raw_b, -1500.0, 1500.0)
            ema_b60 = _fast_ema_1d(basis_1m_bps, 60)
            basis_disloc_bps = np.where(has_spot, basis_1m_bps - ema_b60, 0.0)
            b_lag60 = np.roll(ema_b60, 60)
            b_lag60[:60] = ema_b60[:60]
            basis_markup_60m_bps = np.where(has_spot, basis_1m_bps - b_lag60, 0.0)

            sp_qv_al = np.where(has_spot, s_qv[pos_sp_c], 0.0)
            sp_tb_al = np.where(has_spot, s_tb[pos_sp_c], 0.0)
            sp_sh3 = (_fast_rolling_mean_1d(sp_tb_al, 3) * 3.0) / np.maximum(
                _fast_rolling_mean_1d(sp_qv_al, 3) * 3.0, 1.0
            )
            sp_sh60 = (_fast_rolling_mean_1d(sp_tb_al, 60) * 60.0) / np.maximum(
                _fast_rolling_mean_1d(sp_qv_al, 60) * 60.0, 1.0
            )
            spot_tb_share_3m = np.where(has_spot & (sp_qv_al > 0.0), sp_sh3, perp_tb_share_5m)
            spot_tb_share_60m = np.where(has_spot & (sp_qv_al > 0.0), sp_sh60, spot_tb_share_60m)

    # Align `+60s` causally lagged `5m` Derivatives Positioning (`OI`, `OI_VWAP`, `Whale vs Retail`)
    has_oi = np.zeros(n, dtype=bool)
    oi_ratio_24h = np.ones(n, dtype=np.float64)
    oi_vel_15m = np.zeros(n, dtype=np.float64)
    oi_vel_1h = np.zeros(n, dtype=np.float64)
    oi_vwap_1m = c.copy()
    smart_money_div = np.zeros(n, dtype=np.float64)

    if (
        metrics_5m is not None
        and not metrics_5m.empty
        and ("sum_open_interest" in metrics_5m.columns or "sum_open_interest_value" in metrics_5m.columns)
    ):
        m_df = metrics_5m.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        m_ts_raw = m_df["timestamp"].to_numpy(dtype=np.int64)
        m_ts_avail = m_ts_raw + int(metrics_safety_lag_sec)
        p_at_m_idx = np.minimum(np.searchsorted(ts, m_ts_raw), n - 1)
        px_5m = np.maximum(o[p_at_m_idx], 1e-12)
        if "sum_open_interest" in m_df.columns:
            m_oi = pd.to_numeric(m_df["sum_open_interest"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        else:
            m_oi_val = (
                pd.to_numeric(m_df["sum_open_interest_value"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            )
            m_oi = m_oi_val / px_5m
        if len(m_ts_avail) > 36 and bool((m_oi > 0).any()):
            # Recursive contract-unit `P_oi_vwap` (`~16.5h` bg half-life, zero price-induced OI drift)
            anchor_5m = compute_recursive_oi_vwap_anchor(m_oi, px_5m, base_decay=0.9965)
            oi_sma24_5m = _fast_rolling_mean_1d(m_oi, 288)
            oi_r24_5m = m_oi / np.maximum(oi_sma24_5m, 1e-9)
            oi_v15_5m = np.zeros(len(m_oi), dtype=np.float64)
            oi_v15_5m[3:] = (m_oi[3:] - m_oi[:-3]) / np.maximum(m_oi[:-3], 1e-9)
            oi_v1h_5m = np.zeros(len(m_oi), dtype=np.float64)
            oi_v1h_5m[12:] = (m_oi[12:] - m_oi[:-12]) / np.maximum(m_oi[:-12], 1e-9)

            top_col = (
                "count_toptrader_long_short_ratio"
                if "count_toptrader_long_short_ratio" in m_df.columns
                else "sum_toptrader_long_short_ratio"
            )
            top_ls = (
                np.clip(
                    pd.to_numeric(m_df[top_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64), 0.05, 20.0
                )
                if top_col in m_df.columns
                else np.ones(len(m_oi))
            )
            ret_ls = (
                np.clip(
                    pd.to_numeric(m_df["count_long_short_ratio"], errors="coerce")
                    .fillna(1.0)
                    .to_numpy(dtype=np.float64),
                    0.05,
                    20.0,
                )
                if "count_long_short_ratio" in m_df.columns
                else np.ones(len(m_oi))
            )
            sm_5m = np.log(top_ls / ret_ls)

            # Backward causal asof join onto 1m bar close `bar_close_ts = ts + 60`
            bar_close_ts = ts + 60
            idx_m = np.searchsorted(m_ts_avail, bar_close_ts, side="right") - 1
            valid_m = (idx_m >= 0) & ((bar_close_ts - m_ts_avail[np.maximum(idx_m, 0)]) <= 1800)
            idx_m_c = np.maximum(idx_m, 0)
            has_oi = valid_m & (m_oi[idx_m_c] > 0.0)
            oi_ratio_24h = np.where(has_oi, oi_r24_5m[idx_m_c], 1.0)
            oi_vel_15m = np.where(has_oi, oi_v15_5m[idx_m_c], 0.0)
            oi_vel_1h = np.where(has_oi, oi_v1h_5m[idx_m_c], 0.0)
            oi_vwap_1m = np.where(has_oi, anchor_5m[idx_m_c], c)
            smart_money_div = np.where(has_oi, sm_5m[idx_m_c], 0.0)

    # Align Funding Settlements (`exact :00:00 epoch sec -> rate` & interval transitions)
    funding_map: dict[int, float] = {}
    last_fr_bps = np.zeros(n, dtype=np.float64)
    f_ts_arr = np.empty(0, dtype=np.int64)
    f_rate_arr = np.empty(0, dtype=np.float64)
    f_iv_arr = np.empty(0, dtype=np.float64)

    if funding_df is not None and not funding_df.empty and "funding_rate" in funding_df.columns:
        f_df = funding_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        raw_fts = f_df["timestamp"].to_numpy(dtype=np.int64)
        # Snap minor 1-3s jitter to exact `:00:00` settlement boundary
        f_ts_arr = ((raw_fts + 30) // 60) * 60
        f_rate_arr = pd.to_numeric(f_df["funding_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "interval_hours" in f_df.columns:
            f_iv_arr = np.clip(
                pd.to_numeric(f_df["interval_hours"], errors="coerce").fillna(8.0).to_numpy(dtype=np.float64),
                1.0,
                8.0,
            )
        else:
            f_iv_arr = np.full(len(f_ts_arr), 8.0, dtype=np.float64)
            if len(f_ts_arr) > 1:
                diffs_h = np.round(np.diff(f_ts_arr) / 3600.0)
                f_iv_arr[1:] = np.clip(diffs_h, 1.0, 8.0)

        for ft_k, fr_k in zip(f_ts_arr.tolist(), f_rate_arr.tolist(), strict=False):
            if abs(fr_k) > 1e-9:
                funding_map[int(ft_k)] = float(fr_k)

        if len(f_ts_arr) > 0:
            bar_close_ts = ts + 60
            f_idx = np.searchsorted(f_ts_arr, bar_close_ts, side="right") - 1
            valid_f = f_idx >= 0
            last_fr_bps = np.where(valid_f, f_rate_arr[np.maximum(f_idx, 0)] * 1e4, 0.0)

    events_by_mech: dict[str, list[tuple[int, int, float, float]]] = {k: [] for k in COUNTERPARTY_MECHANISM_SPECS}

    # ------------------------------------------------------------------
    # PILLAR 1 (`M00`, `C1A`, `C1B`, `C1C`) & PILLAR 4 (`C4`): Funding Settlement & Clock Transitions
    # ------------------------------------------------------------------
    if len(f_ts_arr) >= 2:
        for k_f in range(1, len(f_ts_arr)):
            t_settle = int(f_ts_arr[k_f])
            fr_bps = float(f_rate_arr[k_f]) * 1e4
            prev_fr_bps = float(f_rate_arr[k_f - 1]) * 1e4
            iv_cur = float(f_iv_arr[k_f])
            iv_prev = float(f_iv_arr[k_f - 1])

            # Index of `:00:00` 1m candle (`open_ts == t_settle`, closes at `t_settle + 60` -> submits at `:01:00`, 0 funding paid!)
            idx_00 = int(np.searchsorted(ts, t_settle))
            if 240 <= idx_00 < n - 95 and int(ts[idx_00]) == t_settle and bool(liq_gate[idx_00]):
                b_now = float(basis_1m_bps[idx_00])
                b_mk60 = float(basis_markup_60m_bps[idx_00])
                r_00 = float(ret_1m[idx_00])
                tb_00 = float(perp_tb_share_1m[idx_00])
                oiv15 = float(oi_vel_15m[idx_00])

                # M00 Control: Unfiltered M25 post-settle SHORT (`FR <= -20 bps`)
                if fr_bps <= -20.0:
                    events_by_mech["M00_NAIVE_M25_CONTROL"].append((idx_00, -1, abs(fr_bps), b_now))

                # C1A State-A Pure Harvester Overhang SHORT:
                # Vetoes runaway Spot-scarcity squeezes (`b_now < -160 bps` or `b_now < 3.8 * fr_bps` or `b_mk60 < -15 bps`),
                # vetoes active OI blow-off, and requires `:00` first-minute harvester dump confirmation!
                spot_not_squeezing = (not bool(has_spot[idx_00])) or (
                    b_now >= -160.0 and b_now >= 3.8 * fr_bps and b_mk60 >= -15.0
                )
                if fr_bps <= -20.0 and spot_not_squeezing and oiv15 <= 0.04 and r_00 <= -0.0005 and tb_00 <= 0.47:
                    str_c1a = abs(fr_bps) * (1.0 + max(0.0, -r_00 * 1e4) / 20.0)
                    events_by_mech["C1A_POST_SETTLE_HARVESTER_SHORT"].append((idx_00, -1, str_c1a, b_now))

                # C1C State-B Post-Settle Trapped-Short Panic LONG:
                # Deep Spot premium (`Spot >> Perp`, `b_now <= -80 bps`), `FR <= -25 bps`, AND `:00` candle REFUSES to dump!
                if (
                    bool(has_spot[idx_00])
                    and fr_bps <= -25.0
                    and b_now <= -80.0
                    and r_00 >= 0.0008
                    and tb_00 >= 0.55
                    and float(ret_4h[idx_00]) >= 0.0
                ):
                    str_c1c = abs(b_now) + abs(fr_bps)
                    events_by_mech["C1C_POST_SETTLE_SHORT_PANIC_LONG"].append((idx_00, 1, str_c1c, b_now))

                # C4 Discrete Funding-Clock Compression Transition (`iv_cur < iv_prev`, e.g. `8h -> 4h/2h/1h`)
                if iv_cur < iv_prev - 0.5 and fr_bps <= -20.0 and (not bool(has_spot[idx_00]) or b_now <= -35.0):
                    comp_mult = iv_prev / max(iv_cur, 1.0)
                    events_by_mech["C4_CLOCK_COMPRESSION_SQUEEZE_LONG"].append(
                        (idx_00, 1, abs(fr_bps) * comp_mult, comp_mult)
                    )

            # C1B State-B Pre-Settle (`T_settle - 45m`) Spot-Scarcity Carry + Squeeze LONG:
            # Causal! Uses `prev_fr_bps <= -25 bps` (already known from prior settlement!) + live `Basis <= -80 bps` at `T - 45m`!
            t_pre45 = t_settle - 45 * 60
            idx_pre = int(np.searchsorted(ts, t_pre45))
            if (
                240 <= idx_pre < n - 90
                and int(ts[idx_pre]) == t_pre45
                and bool(liq_gate[idx_pre])
                and bool(has_spot[idx_pre])
                and prev_fr_bps <= -25.0
                and float(basis_1m_bps[idx_pre]) <= -80.0
                and float(spot_tb_share_60m[idx_pre]) >= 0.52
                and float(ret_4h[idx_pre]) >= 0.015
            ):
                b_pre = float(basis_1m_bps[idx_pre])
                events_by_mech["C1B_PRE_SETTLE_SPOT_SCARCITY_LONG"].append(
                    (idx_pre, 1, abs(b_pre) + abs(prev_fr_bps), b_pre)
                )

    # ------------------------------------------------------------------
    # PILLAR 2 (`C2A`, `C2B`): Recursive OI-VWAP Trapped-Margin Pain Tomography (Evaluated on 5m boundaries)
    # ------------------------------------------------------------------
    if bool(has_oi.any()):
        # Rolling 4h Donchian high/low computed on `5m` sub-grid (`48` x `5m` bars) for fast O(N/5) scan
        step5 = np.arange(250, n - 185, 5, dtype=np.int64)
        if len(step5) > 50:
            # Exact causal prior-4h high/low via 5m max/min blocks
            n5 = n // 5
            h5 = h[: n5 * 5].reshape(n5, 5).max(axis=1)
            l5 = low[: n5 * 5].reshape(n5, 5).min(axis=1)
            s_h4 = pd.Series(h5).shift(1).rolling(48, min_periods=24).max().to_numpy(dtype=np.float64)
            s_l4 = pd.Series(l5).shift(1).rolling(48, min_periods=24).min().to_numpy(dtype=np.float64)

            for idx_m in step5.tolist():
                b5 = idx_m // 5
                if b5 >= len(s_h4) or not bool(liq_gate[idx_m]) or not bool(has_oi[idx_m]):
                    continue
                oir = float(oi_ratio_24h[idx_m])
                if oir < 1.12:
                    continue
                pain_signed = (float(c[idx_m]) - float(oi_vwap_1m[idx_m])) / float(atr_1h[idx_m])
                fr_b = float(last_fr_bps[idx_m])
                rv5 = float(rvol_5m[idx_m])

                # C2A: Underwater Trapped Shorts (`pain_signed >= +1.8`, `FR <= -5 bps`, contract OI unflushed, 4h High Break)
                if (
                    pain_signed >= 1.8
                    and fr_b <= -5.0
                    and float(oi_vel_1h[idx_m]) >= 0.0
                    and rv5 >= 1.8
                    and float(c[idx_m]) > float(s_h4[b5])
                    and float(spot_tb_share_3m[idx_m]) >= 0.54
                ):
                    events_by_mech["C2A_TRAPPED_SHORTS_PAIN_LONG"].append((idx_m, 1, pain_signed * oir, fr_b))

                # C2B: Underwater Trapped Retail Longs (`pain_signed <= -1.8`, `FR >= +5 bps`, Retail > Whale, 4h Low Break)
                if (
                    pain_signed <= -1.8
                    and fr_b >= 5.0
                    and float(smart_money_div[idx_m]) <= -0.03
                    and float(oi_vel_1h[idx_m]) >= -0.015
                    and rv5 >= 1.8
                    and float(c[idx_m]) < float(s_l4[b5])
                    and float(perp_tb_share_5m[idx_m]) <= 0.45
                ):
                    events_by_mech["C2B_TRAPPED_LONGS_WATERFALL_SHORT"].append(
                        (idx_m, -1, abs(pain_signed) * oir, fr_b)
                    )

    # ------------------------------------------------------------------
    # PILLAR 3 (`C3A`, `C3B`): 1-Minute Spot-Anchored Perpetual Liquidation Snap-Back
    # ------------------------------------------------------------------
    if bool((has_spot & has_oi).any()):
        base_c3 = liq_gate & has_spot & has_oi & (oi_vel_15m <= -0.020)
        c3a_idx = np.flatnonzero(
            base_c3 & (ret_10m <= -0.022) & (basis_disloc_bps <= -22.0) & (ret_1m > 0.0) & (spot_tb_share_3m >= 0.53)
        )
        for ix in c3a_idx.tolist():
            if 240 <= ix < n - 65:
                st_v = abs(float(basis_disloc_bps[ix])) * (abs(float(oi_vel_15m[ix])) * 100.0)
                events_by_mech["C3A_SPOT_ANCHORED_LIQ_SNAPBACK_LONG"].append((ix, 1, st_v, float(basis_disloc_bps[ix])))

        c3b_idx = np.flatnonzero(
            base_c3 & (ret_10m >= 0.026) & (basis_disloc_bps >= 25.0) & (ret_1m < 0.0) & (spot_tb_share_3m <= 0.46)
        )
        for ix in c3b_idx.tolist():
            if 240 <= ix < n - 65:
                st_v = float(basis_disloc_bps[ix]) * (abs(float(oi_vel_15m[ix])) * 100.0)
                events_by_mech["C3B_SPOT_ANCHORED_BLOWOFF_SHORT"].append((ix, -1, st_v, float(basis_disloc_bps[ix])))

    # ------------------------------------------------------------------
    # PILLAR 5 (`C5`): Sovereign BTC Cash-Shock Event-Time Propagation Wave
    # ------------------------------------------------------------------
    if btc_shocks is not None and symbol != "BTCUSDT" and len(btc_shocks.get("ts", [])) > 100:
        b_ts = btc_shocks["ts"]
        b_dir = btc_shocks["shock_dir"]
        shock_mask_btc = b_dir != 0
        if bool(shock_mask_btc.any()):
            # Map BTC shock timestamps onto this symbol's 1m array
            sh_ts = b_ts[shock_mask_btc]
            sh_dirs = b_dir[shock_mask_btc]
            sh_r3 = btc_shocks["ret_3m"][shock_mask_btc]
            sh_str = btc_shocks["shock_strength"][shock_mask_btc]

            sym_pos = np.searchsorted(ts, sh_ts)
            sym_pos_c = np.minimum(sym_pos, n - 1)
            matched_sh = (sym_pos < n - 50) & (sym_pos >= 120) & (ts[sym_pos_c] == sh_ts)

            # Align BTC 1m returns on `sym_pos_c - 60 .. sym_pos_c - 1` only for matched shocks (`< 0.5%` of bars = ultra fast!)
            b_r1 = btc_shocks["ret_1m"]
            for k_sh in np.flatnonzero(matched_sh).tolist():
                ix = int(sym_pos_c[k_sh])
                if not bool(liq_gate[ix]) or float(adv_7d[ix]) < 25_000_000.0:
                    continue
                # Causal trailing 60m OLS beta to BTC (`t - 60 .. t - 1`)
                b_ix = int(np.searchsorted(b_ts, int(ts[ix])))
                if b_ix < 65 or int(b_ts[b_ix]) != int(ts[ix]):
                    continue
                rb_win = b_r1[b_ix - 60 : b_ix]
                rs_win = ret_1m[ix - 60 : ix]
                var_b = float(np.var(rb_win))
                if var_b <= 1e-10:
                    continue
                beta_60m = float(np.cov(rs_win, rb_win, ddof=0)[0, 1] / var_b)
                if beta_60m < 1.25 or beta_60m > 4.0:
                    continue

                d_sh = int(sh_dirs[k_sh])
                exp_move = beta_60m * float(sh_r3[k_sh])
                act_move = float(ret_3m[ix])
                # Lagging alt (`signed catch-up gap >= +50 bps`, moved `< 35%` of beta-implied move, while Spot/Taker confirms!)
                lag_gap = d_sh * (exp_move - act_move)
                sp_sh = float(spot_tb_share_3m[ix])
                flow_ok = (d_sh > 0 and sp_sh >= 0.55) or (d_sh < 0 and sp_sh <= 0.45)
                if lag_gap >= 0.0050 and (d_sh * act_move) < 0.35 * abs(exp_move) and flow_ok:
                    events_by_mech["C5_SOVEREIGN_CASH_WAVE_IGNITION"].append(
                        (ix, d_sh, float(sh_str[k_sh]) * (lag_gap * 100.0), lag_gap * 1e4)
                    )

    all_trades: list[dict[str, Any]] = []
    for m_id, ev_list in events_by_mech.items():
        if not ev_list:
            continue
        all_trades.extend(
            simulate_1m_event_across_modes(
                ts_1m=ts,
                open_1m=o,
                high_1m=h,
                low_1m=low,
                close_1m=c,
                qvol_1m=qv,
                funding_map=funding_map,
                raw_events=ev_list,
                spec=COUNTERPARTY_MECHANISM_SPECS[m_id],
                symbol=symbol,
            )
        )
    return all_trades


def evaluate_infold_cpcv_and_treasure_metrics(
    trades_df: pd.DataFrame,
    total_trials: int = 30,
    slot_fraction: float = 0.22,
    max_daily_Concurrency_cap: int = 4,
) -> pd.DataFrame:
    """
    Evaluates every `(mech_id, exec_mode)` across:
      1. Pure Trade-Level Linear Metrics (`Mean Net bps`, `Median Net bps`, `P90 bps`, `Worst Trade bps`,
         `Win Rate %`, `Profit Factor`, `Fee Share of Positive Gross %`) in `Full`, `2023–24`, `2025`, and `2026`.
      2. TRUE Out-of-Sample `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits with `24h` embargo)
         where the minimum physical `strength` quantile threshold `q* in {0.0, 0.25, 0.50}` is fit
         STRICTLY on `train_days` inside each fold and applied out-of-sample on `test_days`!
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    df = trades_df.copy()
    df["dt"] = pd.to_datetime(df["entry_ts"], unit="s", utc=True)
    df["day"] = df["dt"].dt.floor("D")
    df["year"] = df["dt"].dt.year

    min_day = df["day"].min()
    max_day = df["day"].max()
    all_days = pd.date_range(min_day, max_day, freq="D", tz="UTC")

    def _slice_stats(sub: pd.DataFrame) -> dict[str, float]:
        if sub.empty:
            return {
                "n": 0,
                "mean_net_bps": 0.0,
                "med_net_bps": 0.0,
                "p90_net_bps": 0.0,
                "worst_bps": 0.0,
                "mean_gross_bps": 0.0,
                "mean_fnd_bps": 0.0,
                "mean_fee_bps": 0.0,
                "win_pct": 0.0,
                "pf": 0.0,
                "fee_share_pct": 100.0,
                "tstat": 0.0,
            }
        net = sub["net_bps"].to_numpy(dtype=np.float64)
        gross = sub["gross_bps"].to_numpy(dtype=np.float64)
        fnd = sub["funding_bps"].to_numpy(dtype=np.float64)
        fee = sub["fee_bps"].to_numpy(dtype=np.float64)
        n_t = len(net)
        mu = float(np.mean(net))
        sd = float(np.std(net, ddof=1)) if n_t > 2 else 1.0
        tstat = (mu / (sd / np.sqrt(n_t))) if (n_t > 2 and sd > 1e-9) else 0.0
        wins = net[net > 0.0].sum()
        losses = abs(net[net < 0.0].sum())
        pos_gross = np.maximum(0.0, gross + fnd).sum()
        fee_sh = (fee.sum() / max(pos_gross, 1e-6)) * 100.0 if pos_gross > 0 else 100.0
        return {
            "n": n_t,
            "mean_net_bps": round(mu, 2),
            "med_net_bps": round(float(np.median(net)), 2),
            "p90_net_bps": round(float(np.percentile(net, 90)), 1),
            "worst_bps": round(float(np.min(net)), 1),
            "mean_gross_bps": round(float(np.mean(gross)), 2),
            "mean_fnd_bps": round(float(np.mean(fnd)), 2),
            "mean_fee_bps": round(float(np.mean(fee)), 2),
            "win_pct": round(float((net > 0.0).mean() * 100.0), 1),
            "pf": round(float(wins / max(losses, 1e-6)), 2) if wins > 0 else 0.0,
            "fee_share_pct": round(min(999.0, float(fee_sh)), 1),
            "tstat": round(float(tstat), 2),
        }

    rows: list[dict[str, Any]] = []
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo_pct=0.01)
    day_frame = pd.DataFrame({"dummy": np.zeros(len(all_days))}, index=all_days)
    t1_days = pd.Series(all_days + pd.Timedelta(days=1), index=all_days)

    for (mech_id, exec_mode), grp in df.groupby(["mech_id", "exec_mode"], sort=False):
        s_full = _slice_stats(grp)
        s_tr = _slice_stats(grp[grp["year"] <= 2024])
        s_25 = _slice_stats(grp[grp["year"] == 2025])
        s_26 = _slice_stats(grp[grp["year"] >= 2026])

        # True In-Fold Refit CPCV: calibrate `strength >= quantile(q)` on `train_days` ONLY, score `test_days`!
        oos_sharpes: list[float] = []
        oos_net_bps_folds: list[float] = []
        if len(all_days) >= 60 and len(grp) >= 25:
            for tr_idx, te_idx in cpcv.split(day_frame, t1=t1_days):
                tr_day_set = set(all_days[tr_idx])
                te_days_index = all_days[te_idx]
                te_day_set = set(te_days_index)

                tr_sub = grp[grp["day"].isin(tr_day_set)]
                best_thresh = -np.inf
                best_obj = -1e18
                if len(tr_sub) >= 12:
                    str_vals = tr_sub["strength"].to_numpy(dtype=np.float64)
                    for q_cand in (0.0, 0.25, 0.50):
                        th_c = float(np.quantile(str_vals, q_cand))
                        cand_t = tr_sub[tr_sub["strength"] >= th_c]
                        if len(cand_t) < 8:
                            continue
                        c_net = cand_t["net_bps"].to_numpy(dtype=np.float64)
                        obj = float(np.mean(c_net)) * np.sqrt(len(cand_t))
                        if obj > best_obj:
                            best_obj = obj
                            best_thresh = th_c

                te_sub = grp[grp["day"].isin(te_day_set) & (grp["strength"] >= best_thresh)]
                if te_sub.empty:
                    oos_sharpes.append(0.0)
                    oos_net_bps_folds.append(0.0)
                    continue

                oos_net_bps_folds.append(float(te_sub["net_bps"].mean()))
                # Daily linear portfolio return capped at `max_daily_Concurrency_cap` top-strength trades per day
                te_capped = (
                    te_sub.sort_values("strength", ascending=False)
                    .groupby("day", sort=False)
                    .head(max_daily_Concurrency_cap)
                )
                d_ret = (te_capped.groupby("day")["net_bps"].sum() * (slot_fraction / 1e4)).reindex(
                    te_days_index, fill_value=0.0
                )
                sd_d = float(d_ret.std(ddof=1))
                sh_d = (float(d_ret.mean()) / sd_d) * np.sqrt(365.25) if sd_d > 1e-12 else 0.0
                oos_sharpes.append(sh_d)

        cpcv_med_sh = round(float(np.median(oos_sharpes)), 2) if oos_sharpes else 0.0
        cpcv_pos_pct = round(float(np.mean(np.asarray(oos_sharpes) > 0.0) * 100.0), 1) if oos_sharpes else 0.0
        cpcv_oos_bps = round(float(np.median(oos_net_bps_folds)), 2) if oos_net_bps_folds else 0.0

        # Full-span daily series for DSR calculation
        grp_cap = (
            grp.sort_values("strength", ascending=False).groupby("day", sort=False).head(max_daily_Concurrency_cap)
        )
        full_d_ret = (
            (grp_cap.groupby("day")["net_bps"].sum() * (slot_fraction / 1e4))
            .reindex(all_days, fill_value=0.0)
            .to_numpy()
        )
        dsr_v = (
            compute_deflated_sharpe_ratio(full_d_ret, num_trials=max(1, total_trials), periods_per_year=365.25)
            if len(full_d_ret) >= 30
            else 0.0
        )

        # Strict "Treasure" Gate:
        #   1) Positive & substantial net linear edge in EVERY era (`2023-24 >= +15 bps`, `2025 >= +15 bps`, `2026 >= +20 bps`)
        #   2) Full `Mean Net >= +25.0 bps/trade` AND `Fee Share <= 30%` of gross gains
        #   3) True In-Fold Refit `CPCV Median OOS bps >= +15.0 bps` & `>= 80%` positive CPCV splits
        #   4) Minimum sample size `>= 60` filled trades
        is_treasure = bool(
            s_full["n"] >= 60
            and s_full["mean_net_bps"] >= 25.0
            and s_tr["mean_net_bps"] >= 15.0
            and s_25["mean_net_bps"] >= 15.0
            and s_26["mean_net_bps"] >= 20.0
            and s_full["fee_share_pct"] <= 30.0
            and cpcv_oos_bps >= 15.0
            and cpcv_pos_pct >= 80.0
        )

        spec_obj = COUNTERPARTY_MECHANISM_SPECS.get(str(mech_id))
        rows.append(
            {
                "mech_id": str(mech_id),
                "pillar": spec_obj.pillar if spec_obj else "",
                "exec_mode": str(exec_mode),
                "trades_full": s_full["n"],
                "trades_2025": s_25["n"],
                "trades_2026": s_26["n"],
                "mean_gross_bps": s_full["mean_gross_bps"],
                "mean_fnd_bps": s_full["mean_fnd_bps"],
                "mean_fee_bps": s_full["mean_fee_bps"],
                "mean_net_bps_full": s_full["mean_net_bps"],
                "med_net_bps_full": s_full["med_net_bps"],
                "p90_net_bps": s_full["p90_net_bps"],
                "worst_trade_bps": s_full["worst_bps"],
                "win_rate_pct": s_full["win_pct"],
                "profit_factor": s_full["pf"],
                "fee_share_of_gross_pct": s_full["fee_share_pct"],
                "trade_tstat": s_full["tstat"],
                "net_bps_23_24": s_tr["mean_net_bps"],
                "net_bps_2025": s_25["mean_net_bps"],
                "net_bps_2026": s_26["mean_net_bps"],
                "cpcv_infold_oos_bps": cpcv_oos_bps,
                "cpcv_infold_sharpe": cpcv_med_sh,
                "cpcv_pos_splits_pct": cpcv_pos_pct,
                "dsr": round(float(dsr_v), 4),
                "is_treasure": is_treasure,
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    out_df["min_era_net_bps"] = out_df[["net_bps_23_24", "net_bps_2025", "net_bps_2026"]].min(axis=1)
    out_df.sort_values(
        ["is_treasure", "min_era_net_bps", "mean_net_bps_full"],
        ascending=[False, False, False],
        inplace=True,
    )
    out_df.reset_index(drop=True, inplace=True)
    return out_df
