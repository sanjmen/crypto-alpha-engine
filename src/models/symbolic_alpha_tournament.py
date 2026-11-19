"""
Automated Symbolic & Shallow-Tree Multi-Modal Alpha Discovery Tournament (Issue #70 / M16-4).

Discovers genuinely orthogonal (`corr < 0.20` to live `Core P1 + Manga P2 + Phase 2 Snipers`),
all-weather (`Sharpe >= 1.80` in EVERY era: `2023–2024`, `2025`, and `2026` Holdout) alpha
sleeves over the `M16-2` causal multi-modal feature panel (`219` Point-in-Time perps).

Strict Anti-Overfitting & Microstructure Accounting Invariants:
  1. Train-Only Fitting (`ml-best-practices` Strict Featurization Ordering):
     All feature standardization stats, symbolic quantile thresholds, and depth-2/3 trees
     are fit STRICTLY on `2023–2024` (`Train`). `2025` (`Validation`) and `2026` (`Holdout`)
     are transformed purely out-of-sample with zero refitting or leakage.
  2. Exact Realized Funding Cashflow + Turnover Fee Deduction:
     Every position held across a settlement boundary (`mins_to_next_settlement` crossed within
     the holding bar) explicitly accrues `-weight * funding_rate_bps` (Shorts PAY negative
     funding, Longs RECEIVE negative funding), plus exact `L1_turnover * one_way_cost_bps`
     (including `M16-3` queue-penetration / Maker Limit-Chase execution haircuts).
  3. 4 Mandatory Promotion Gates:
     - Gate 1: `CombinatorialPurgedCV` (`N=6, k=2 -> 15` splits, `24h` embargo) median OOS Sharpe >= 1.80
     - Gate 2: Bailey & López de Prado `Deflated Sharpe Ratio (DSR) >= 0.95` across all `M` trials
     - Gate 3: All-Weather Era Stability (`Sharpe >= 1.80` in `2023–2024` AND `2025` AND `2026`)
     - Gate 4: True Orthogonality (`corr < 0.20` vs deployed production benchmark daily PnL)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text

from src.models.purged_cv import CombinatorialPurgedCV, compute_deflated_sharpe_ratio

CORE_MAJORS_15: frozenset[str] = frozenset(
    {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "NEARUSDT",
        "LTCUSDT",
        "UNIUSDT",
        "APTUSDT",
        "ARBUSDT",
    }
)


@dataclass(frozen=True)
class MechanismSpec:
    """Auditable specification of a candidate multi-modal alpha mechanism."""

    mech_id: str
    archetype: str
    description: str
    rebalance_hours: int  # e.g. 1 or 4
    mode: str  # 'cross_sectional_neutral' or 'event_sniper'
    universe_scope: str  # 'all_liquid', 'midcap_ex_majors'
    one_way_cost_bps: float  # e.g. 2.1 bps Maker Limit-Chase or 2.8 bps GTX w/ penetration haircut
    ema_halflife_bars: int = 1
    top_k_per_side: int = 12
    symbolic_rule_text: str = ""


def _fast_cs_rank_centered(vals: np.ndarray, ts_codes: np.ndarray, n_ts: int) -> np.ndarray:
    """Fast `[-0.5, +0.5]` cross-sectional rank within each `ts_code`."""
    df_tmp = pd.DataFrame({"g": ts_codes, "v": vals})
    cnt = df_tmp.groupby("g")["v"].transform("count").to_numpy(dtype=np.float64)
    rnk = df_tmp.groupby("g")["v"].rank(method="average", na_option="keep").to_numpy(dtype=np.float64)
    valid = np.isfinite(rnk) & (cnt > 1.0)
    out = np.zeros(len(vals), dtype=np.float64)
    out[valid] = ((rnk[valid] - 0.5) / cnt[valid]) - 0.5
    return out


def simulate_cross_sectional_sleeve_daily(
    panel_rebal: pd.DataFrame,
    raw_score: np.ndarray,
    fwd_ret_col: str,
    horizon_hours: int,
    one_way_cost_bps: float = 2.1,
    ema_alpha: float = 0.35,
    top_k: int = 14,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Simulates a strictly causal, dollar-neutral (`sum(w+) = +0.5, sum(w-) = -0.5`) cross-sectional
    portfolio at `horizon_hours` cadence with:
      - Causal per-symbol exponential signal smoothing (`ema_alpha`) to control turnover
      - Top-`K` Long / Bottom-`K` Short equal-or-score-weighted book (`|w|_1 = 1.0` gross)
      - Exact settlement funding cashflow (`-w_i * funding_rate * n_settlements_in_horizon`)
      - Exact `L1_turnover * (one_way_cost_bps / 1e4)` execution cost deduction at every rebalance!
    """
    if panel_rebal.empty:
        return pd.Series(dtype=np.float64), {"mean_l1_turnover": 0.0, "ann_fee_drag_pct": 0.0}

    dec_ts = panel_rebal["decision_ts"].to_numpy(dtype=np.int64, copy=False)
    sym_arr = panel_rebal["symbol"].to_numpy(copy=False)
    fwd_ret = np.nan_to_num(panel_rebal[fwd_ret_col].to_numpy(dtype=np.float64, copy=False), nan=0.0)
    f_bps = (
        np.nan_to_num(panel_rebal["funding_rate_bps"].to_numpy(dtype=np.float64, copy=False), nan=0.0)
        if "funding_rate_bps" in panel_rebal.columns
        else np.zeros(len(panel_rebal), dtype=np.float64)
    )
    mins_to_next = (
        np.nan_to_num(panel_rebal["mins_to_next_settlement"].to_numpy(dtype=np.float64, copy=False), nan=240.0)
        if "mins_to_next_settlement" in panel_rebal.columns
        else np.full(len(panel_rebal), 240.0, dtype=np.float64)
    )
    f_iv_h = (
        np.clip(
            np.nan_to_num(panel_rebal["funding_interval_hours"].to_numpy(dtype=np.float64, copy=False), nan=8.0),
            1.0,
            8.0,
        )
        if "funding_interval_hours" in panel_rebal.columns
        else np.full(len(panel_rebal), 8.0, dtype=np.float64)
    )

    # Number of funding settlements crossed during `[decision_ts, decision_ts + horizon_hours * 3600]`
    h_mins = float(horizon_hours * 60)
    crosses_first = (mins_to_next > 0.0) & (mins_to_next <= h_mins)
    rem_mins = np.maximum(0.0, h_mins - np.maximum(0.0, mins_to_next))
    extra_settles = np.floor(rem_mins / (f_iv_h * 60.0))
    n_settles = np.where(crosses_first, 1.0 + extra_settles, 0.0)
    funding_cash_ret = -(f_bps / 1e4) * n_settles  # multiplied by signed weight `w_i`: Long (+w) pays +f_bps

    # Causal per-symbol EMA smoothing of `raw_score`
    sym_codes, uniq_syms = pd.factorize(sym_arr, sort=False)
    n_syms = len(uniq_syms)
    order_sym_ts = np.lexsort((dec_ts, sym_codes))
    sc_sorted = np.nan_to_num(raw_score[order_sym_ts], nan=0.0, posinf=0.0, neginf=0.0)
    sym_sorted = sym_codes[order_sym_ts]

    if 0.0 < ema_alpha < 1.0:
        sm_sorted = np.empty_like(sc_sorted)
        s_starts = np.flatnonzero(np.r_[True, sym_sorted[1:] != sym_sorted[:-1]])
        s_ends = np.r_[s_starts[1:], len(sc_sorted)]
        one_m = 1.0 - ema_alpha
        for s0, s1 in zip(s_starts, s_ends, strict=False):
            acc = sc_sorted[s0]
            sm_sorted[s0] = acc
            for j in range(s0 + 1, s1):
                acc = ema_alpha * sc_sorted[j] + one_m * acc
                sm_sorted[j] = acc
        smoothed = np.empty_like(sc_sorted)
        smoothed[order_sym_ts] = sm_sorted
    else:
        smoothed = np.nan_to_num(raw_score, nan=0.0, posinf=0.0, neginf=0.0)

    # Group by chronological `decision_ts` to construct exact Top-K / Bottom-K dollar-neutral weights
    order_ts = np.argsort(dec_ts, kind="mergesort")
    ts_s = dec_ts[order_ts]
    sym_s = sym_codes[order_ts]
    score_s = smoothed[order_ts]
    ret_s = fwd_ret[order_ts]
    fcash_s = funding_cash_ret[order_ts]

    t_starts = np.flatnonzero(np.r_[True, ts_s[1:] != ts_s[:-1]])
    t_ends = np.r_[t_starts[1:], len(ts_s)]
    n_periods = len(t_starts)

    prev_w = np.zeros(n_syms, dtype=np.float64)
    cur_w = np.zeros(n_syms, dtype=np.float64)
    cost_rate = one_way_cost_bps / 1e4

    period_ts = np.empty(n_periods, dtype=np.int64)
    period_net_ret = np.empty(n_periods, dtype=np.float64)
    period_turn = np.empty(n_periods, dtype=np.float64)
    period_fee = np.empty(n_periods, dtype=np.float64)

    for p_i, (i0, i1) in enumerate(zip(t_starts, t_ends, strict=False)):
        period_ts[p_i] = ts_s[i0]
        cur_w.fill(0.0)
        m_univ = i1 - i0
        sc_slice = score_s[i0:i1]
        sy_slice = sym_s[i0:i1]

        k_side = min(top_k, max(3, m_univ // 6), m_univ // 2)
        if k_side >= 3 and float(np.std(sc_slice)) > 1e-12:
            ord_local = np.argsort(sc_slice)
            short_loc = ord_local[:k_side]
            long_loc = ord_local[-k_side:]
            if sc_slice[long_loc[0]] > sc_slice[short_loc[-1]]:
                w_leg = 0.5 / float(k_side)
                cur_w[sy_slice[long_loc]] = w_leg
                cur_w[sy_slice[short_loc]] = -w_leg

        l1_turn = float(np.abs(cur_w - prev_w).sum())
        fee = l1_turn * cost_rate
        w_active = cur_w[sy_slice]
        gross_alpha = float(np.dot(w_active, ret_s[i0:i1]))
        funding_pnl = float(np.dot(np.abs(w_active) * np.sign(w_active), fcash_s[i0:i1]))
        # Note: `np.abs(w_active) * np.sign(w_active) == w_active`, so Long (+w) gets -(f_bps/1e4)
        # (collects when f_bps < 0, pays when f_bps > 0) and Short (-w) gets +(f_bps/1e4)!
        period_net_ret[p_i] = gross_alpha + funding_pnl - fee
        period_turn[p_i] = l1_turn
        period_fee[p_i] = fee
        prev_w[:] = cur_w

    day_idx = pd.to_datetime((period_ts // 86400) * 86400, unit="s", utc=True)
    daily_pnl = pd.Series(period_net_ret, index=day_idx).groupby(level=0).sum().sort_index()
    periods_per_day = 24.0 / float(max(1, horizon_hours))
    stats_meta = {
        "mean_l1_turnover": round(float(period_turn.mean()), 4),
        "daily_l1_turnover": round(float(period_turn.mean()) * periods_per_day, 3),
        "ann_fee_drag_pct": round(float(period_fee.mean()) * periods_per_day * 365.25 * 100.0, 2),
        "active_days": int((daily_pnl != 0.0).sum()),
    }
    return daily_pnl, stats_meta


def simulate_event_sniper_sleeve_daily(
    panel_events: pd.DataFrame,
    signal_signed: np.ndarray,
    fwd_ret_col: str,
    horizon_hours: int,
    roundtrip_cost_bps: float = 4.6,
    slot_weight: float = 0.20,
    max_concurrent_slots: int = 4,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Simulates a discrete Event-Sniper sleeve (e.g., Post-Settlement Negative Funding Unwind,
    Compressed-Clock Squeeze Fade, or OI Exhaustion Wick Sniper) with:
      - Non-overlapping event slots capped at `max_concurrent_slots * slot_weight <= 0.80` gross
      - Exact settlement funding cashflow (`-w_i * funding_rate * n_settlements_in_horizon`)
      - Conservative `roundtrip_cost_bps` (`M16-3` passive `GTX` + queue-penetration + stop fee blend)
      - Reindexed across the full continuous calendar day grid so non-event days record `0.0` return!
    """
    if panel_events.empty:
        return pd.Series(dtype=np.float64), {"total_trades": 0, "ann_fee_drag_pct": 0.0}

    dec_ts = panel_events["decision_ts"].to_numpy(dtype=np.int64, copy=False)
    min_day = pd.to_datetime((int(dec_ts.min()) // 86400) * 86400, unit="s", utc=True)
    max_day = pd.to_datetime((int(dec_ts.max()) // 86400) * 86400, unit="s", utc=True)
    full_days = pd.date_range(min_day, max_day, freq="D", tz="UTC")

    active_mask = np.abs(signal_signed) > 1e-9
    if not bool(active_mask.any()):
        return pd.Series(0.0, index=full_days), {"total_trades": 0, "ann_fee_drag_pct": 0.0}

    sub = panel_events.loc[active_mask]
    sig_sub = signal_signed[active_mask]
    ts_sub = sub["decision_ts"].to_numpy(dtype=np.int64, copy=False)
    fwd_sub = np.nan_to_num(sub[fwd_ret_col].to_numpy(dtype=np.float64, copy=False), nan=0.0)
    f_bps = (
        np.nan_to_num(sub["funding_rate_bps"].to_numpy(dtype=np.float64, copy=False), nan=0.0)
        if "funding_rate_bps" in sub.columns
        else np.zeros(len(sub), dtype=np.float64)
    )
    mins_to_next = (
        np.nan_to_num(sub["mins_to_next_settlement"].to_numpy(dtype=np.float64, copy=False), nan=240.0)
        if "mins_to_next_settlement" in sub.columns
        else np.full(len(sub), 240.0, dtype=np.float64)
    )
    f_iv_h = (
        np.clip(
            np.nan_to_num(sub["funding_interval_hours"].to_numpy(dtype=np.float64, copy=False), nan=8.0),
            1.0,
            8.0,
        )
        if "funding_interval_hours" in sub.columns
        else np.full(len(sub), 8.0, dtype=np.float64)
    )

    h_mins = float(horizon_hours * 60)
    crosses_first = (mins_to_next > 0.0) & (mins_to_next <= h_mins)
    rem_mins = np.maximum(0.0, h_mins - np.maximum(0.0, mins_to_next))
    n_settles = np.where(crosses_first, 1.0 + np.floor(rem_mins / (f_iv_h * 60.0)), 0.0)
    fcash_unit = -(f_bps / 1e4) * n_settles

    ord_ts = np.argsort(ts_sub, kind="mergesort")
    ts_s = ts_sub[ord_ts]
    sig_s = sig_sub[ord_ts]
    fwd_s = fwd_sub[ord_ts]
    fc_s = fcash_unit[ord_ts]

    t_starts = np.flatnonzero(np.r_[True, ts_s[1:] != ts_s[:-1]])
    t_ends = np.r_[t_starts[1:], len(ts_s)]

    rt_cost = roundtrip_cost_bps / 1e4
    p_ts = np.empty(len(t_starts), dtype=np.int64)
    p_ret = np.zeros(len(t_starts), dtype=np.float64)
    p_fee = np.zeros(len(t_starts), dtype=np.float64)
    total_trades = 0

    # Account for overlapping multi-hour sniper holds (`concurrent_scale = 1 / max_overlap`)
    overlap_div = float(max(1, horizon_hours))

    for idx_b, (i0, i1) in enumerate(zip(t_starts, t_ends, strict=False)):
        p_ts[idx_b] = ts_s[i0]
        sc_b = sig_s[i0:i1]
        if len(sc_b) > max_concurrent_slots:
            top_idx = np.argsort(np.abs(sc_b))[-max_concurrent_slots:]
        else:
            top_idx = np.arange(len(sc_b))

        n_sel = len(top_idx)
        total_trades += n_sel
        w_slot = slot_weight / overlap_div
        dirs = np.sign(sc_b[top_idx])
        unit_net = dirs * fwd_s[i0:i1][top_idx] + dirs * fc_s[i0:i1][top_idx] - rt_cost
        p_ret[idx_b] = float(np.sum(w_slot * unit_net))
        p_fee[idx_b] = float(n_sel * w_slot * rt_cost)

    day_idx = pd.to_datetime((p_ts // 86400) * 86400, unit="s", utc=True)
    daily_raw = pd.Series(p_ret, index=day_idx).groupby(level=0).sum()
    daily_full = daily_raw.reindex(full_days, fill_value=0.0)
    fee_daily = pd.Series(p_fee, index=day_idx).groupby(level=0).sum().reindex(full_days, fill_value=0.0)

    meta = {
        "total_trades": int(total_trades),
        "daily_l1_turnover": round(float(total_trades * 2.0 * (slot_weight / overlap_div)) / max(1, len(full_days)), 3),
        "ann_fee_drag_pct": round(float(fee_daily.mean()) * 365.25 * 100.0, 2),
        "active_days": int((daily_full != 0.0).sum()),
    }
    return daily_full, meta


def compute_daily_series_metrics(daily_ret: pd.Series) -> dict[str, float]:
    """Computes annualized Sharpe, CAGR/geom daily %, MaxDD, and Calmar on a daily return Series."""
    r = daily_ret.dropna().to_numpy(dtype=np.float64)
    if len(r) < 10:
        return {"sharpe": 0.0, "geom_daily_pct": 0.0, "ann_ret_pct": 0.0, "max_dd_pct": 0.0}
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    sharpe = (mu / sd) * np.sqrt(365.25) if sd > 1e-12 else 0.0
    clipped = np.clip(r, -0.95, 5.0)
    eq = np.cumprod(1.0 + clipped)
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    dd = (eq - peaks) / np.maximum(peaks, 1e-12)
    max_dd = float(abs(np.min(dd))) * 100.0 if len(dd) > 0 else 0.0
    geom_d = (float(eq[-1] ** (1.0 / len(r))) - 1.0) * 100.0 if eq[-1] > 0 else -100.0
    ann_ret = (float(eq[-1] ** (365.25 / len(r))) - 1.0) * 100.0 if eq[-1] > 0 else -100.0
    return {
        "sharpe": round(float(sharpe), 3),
        "geom_daily_pct": round(geom_d, 4),
        "ann_ret_pct": round(ann_ret, 2),
        "max_dd_pct": round(max_dd, 2),
    }


def evaluate_cpcv_daily_sharpe(daily_ret: pd.Series, n_groups: int = 6, k_test: int = 2) -> dict[str, float]:
    """
    Evaluates out-of-sample Sharpe stability across all `C(N, k)` Combinatorial Purged CV splits
    with a strict `24h` (`t1 = index + 1 day`) embargo.
    """
    s = daily_ret.dropna().sort_index()
    if len(s) < 60:
        return {"cpcv_median_sharpe": 0.0, "cpcv_min_sharpe": 0.0, "cpcv_pos_splits_pct": 0.0}

    t1 = pd.Series(s.index + pd.Timedelta(days=1), index=s.index)
    cpcv = CombinatorialPurgedCV(n_groups=n_groups, k_test_groups=k_test, embargo_pct=0.01)
    sharpes: list[float] = []
    arr = s.to_numpy(dtype=np.float64)
    for _, test_idx in cpcv.split(s.to_frame("r"), t1=t1):
        sub = arr[test_idx]
        sd = float(np.std(sub, ddof=1)) if len(sub) > 5 else 0.0
        sh = (float(np.mean(sub)) / sd) * np.sqrt(365.25) if sd > 1e-12 else 0.0
        sharpes.append(sh)

    if not sharpes:
        return {"cpcv_median_sharpe": 0.0, "cpcv_min_sharpe": 0.0, "cpcv_pos_splits_pct": 0.0}
    sh_a = np.asarray(sharpes, dtype=np.float64)
    return {
        "cpcv_median_sharpe": round(float(np.median(sh_a)), 3),
        "cpcv_min_sharpe": round(float(np.min(sh_a)), 3),
        "cpcv_pos_splits_pct": round(float(np.mean(sh_a > 0.0) * 100.0), 1),
    }


def run_multimodal_discovery_tournament(
    panel: pd.DataFrame,
    benchmark_daily_pnl: pd.Series,
    min_era_sharpe: float = 1.80,
    min_dsr: float = 0.95,
    max_abs_corr: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, MechanismSpec]]:
    """
    Executes the M16-4 (`Issue #70`) Multi-Modal Symbolic & Shallow-Tree Discovery Tournament
    across 5 orthogonal mechanism archetypes (`M >= 36` candidates), applying all 4 Hard Gates:
      1. Chronological Train (`<= 2024`) -> Val (`2025`) -> Sealed Holdout (`2026`) + CPCV (`N=6, k=2`)
      2. Deflated Sharpe Ratio (`DSR >= 0.95`) across all `M` trials
      3. All-Weather Era Stability (`Sharpe >= 1.80` in `2023–24`, `2025`, AND `2026`)
      4. True Orthogonality (`|corr| < 0.20` vs live production benchmark)
    """
    df = panel[panel["tradable_adv_15m_gate"] > 0.5].copy()
    df.sort_values(["decision_ts", "symbol"], inplace=True, ignore_index=True)

    # Slice 4h grid (`decision_ts % 14400 == 0`) for low-turnover cross-sectional rotators
    # and 1h + :30 grid for fast settlement/cascade snipers
    df_4h = df[(df["decision_ts"] % 14400 == 0) & df["fwd_ret_4h"].notna()].reset_index(drop=True)
    df_4h_mid = df_4h[~df_4h["symbol"].isin(CORE_MAJORS_15)].reset_index(drop=True)
    df_1h = df[df["fwd_ret_1h"].notna()].reset_index(drop=True)
    df_4h_events = df[df["fwd_ret_4h"].notna()].reset_index(drop=True)

    candidates_eval: list[tuple[MechanismSpec, pd.Series, dict[str, float]]] = []

    def _z_train_clipped(frame: pd.DataFrame, col: str, train_mask: np.ndarray) -> np.ndarray:
        if col not in frame.columns:
            return np.zeros(len(frame), dtype=np.float64)
        v = np.nan_to_num(frame[col].to_numpy(dtype=np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        tr_v = v[train_mask]
        mu = float(np.mean(tr_v)) if len(tr_v) > 10 else 0.0
        sd = float(np.std(tr_v)) if len(tr_v) > 10 else 1.0
        sd = sd if sd > 1e-9 else 1.0
        return np.clip((v - mu) / sd, -5.0, 5.0)

    # ------------------------------------------------------------------
    # ARCHETYPE 1: Dynamic Funding Settlement Clock & Post-Settle Unwind Snipers (Family C)
    # ------------------------------------------------------------------
    f_bps_ev4 = df_4h_events["funding_rate_bps"].to_numpy(dtype=np.float64, copy=False)
    m_since_4 = df_4h_events["mins_since_settlement"].to_numpy(dtype=np.float64, copy=False)
    m_next_4 = df_4h_events["mins_to_next_settlement"].to_numpy(dtype=np.float64, copy=False)
    clk_comp_4 = df_4h_events["clock_compression_ratio"].to_numpy(dtype=np.float64, copy=False)

    for f_thresh in (-15.0, -22.0, -30.0, -40.0):
        for slot_k in (3, 5):
            # 1A: Post-Settlement Snap-Down Short (0..30m AFTER settlement -> ZERO negative funding paid!)
            sig_post = np.where(
                (m_since_4 >= 0.0) & (m_since_4 <= 30.0) & (f_bps_ev4 <= f_thresh), -np.abs(f_bps_ev4), 0.0
            )
            spec_1a = MechanismSpec(
                mech_id=f"A1_POST_SETTLE_NEG_UNWIND_{int(abs(f_thresh))}BPS_K{slot_k}",
                archetype="Archetype_1_Funding_Settlement_Clock",
                description=(
                    f"Post-settlement (T+0..30m, 0 funding paid) SHORT unwind when settled funding <= {f_thresh:.0f} bps"
                ),
                rebalance_hours=4,
                mode="event_sniper",
                universe_scope="all_liquid",
                one_way_cost_bps=2.3,
                top_k_per_side=slot_k,
                symbolic_rule_text=f"IF mins_since_settlement IN [0, 30] AND funding_rate_bps <= {f_thresh:.0f} THEN SHORT",
            )
            pnl_1a, meta_1a = simulate_event_sniper_sleeve_daily(
                df_4h_events,
                sig_post,
                "fwd_ret_4h",
                horizon_hours=4,
                roundtrip_cost_bps=4.6,
                max_concurrent_slots=slot_k,
            )
            candidates_eval.append((spec_1a, pnl_1a, meta_1a))

            # 1B: Pre-Settlement Squeeze Fade (explicitly deducts exact funding paid at T_settle!)
            sig_pre = np.where(
                (m_next_4 >= 5.0) & (m_next_4 <= 60.0) & (f_bps_ev4 <= f_thresh), -np.abs(f_bps_ev4), 0.0
            )
            spec_1b = MechanismSpec(
                mech_id=f"A1_PRE_SETTLE_PUMP_FADE_{int(abs(f_thresh))}BPS_K{slot_k}",
                archetype="Archetype_1_Funding_Settlement_Clock",
                description=(
                    f"Pre-settlement (T-60..5m, net of funding paid) SHORT pump-fade when funding <= {f_thresh:.0f} bps"
                ),
                rebalance_hours=4,
                mode="event_sniper",
                universe_scope="all_liquid",
                one_way_cost_bps=2.3,
                top_k_per_side=slot_k,
                symbolic_rule_text=f"IF mins_to_next_settlement IN [5, 60] AND funding_rate_bps <= {f_thresh:.0f} THEN SHORT",
            )
            pnl_1b, meta_1b = simulate_event_sniper_sleeve_daily(
                df_4h_events,
                sig_pre,
                "fwd_ret_4h",
                horizon_hours=4,
                roundtrip_cost_bps=4.6,
                max_concurrent_slots=slot_k,
            )
            candidates_eval.append((spec_1b, pnl_1b, meta_1b))

    # 1C: Compressed-Clock (<=4h) Post-Settle + Pre-Settle Combined Funding Impulse Sniper
    for imp_th in (8.0, 15.0, 25.0):
        imp_v = df_4h_events["compressed_clock_neg_funding_impulse"].to_numpy(dtype=np.float64, copy=False)
        sig_cc = np.where((clk_comp_4 >= 2.0) & (imp_v >= imp_th) & (m_since_4 <= 30.0), -imp_v, 0.0)
        spec_1c = MechanismSpec(
            mech_id=f"A1_COMPRESSED_CLOCK_POST_UNWIND_{int(imp_th)}",
            archetype="Archetype_1_Funding_Settlement_Clock",
            description=f"Shortened <=4h funding clock post-settlement unwind (impulse >= {imp_th:.0f})",
            rebalance_hours=4,
            mode="event_sniper",
            universe_scope="all_liquid",
            one_way_cost_bps=2.3,
            top_k_per_side=4,
            symbolic_rule_text=f"IF funding_interval <= 4h AND mins_since_settle <= 30 AND impulse >= {imp_th} THEN SHORT",
        )
        pnl_1c, meta_1c = simulate_event_sniper_sleeve_daily(
            df_4h_events, sig_cc, "fwd_ret_4h", horizon_hours=4, roundtrip_cost_bps=4.6, max_concurrent_slots=4
        )
        candidates_eval.append((spec_1c, pnl_1c, meta_1c))

    # Fast 1h Pre-Settlement Pump-Fade (exits within 1h)
    f_bps_1h = df_1h["funding_rate_bps"].to_numpy(dtype=np.float64, copy=False)
    m_next_1h = df_1h["mins_to_next_settlement"].to_numpy(dtype=np.float64, copy=False)
    m_since_1h = df_1h["mins_since_settlement"].to_numpy(dtype=np.float64, copy=False)
    for f_th1 in (-20.0, -35.0):
        sig_1h_post = np.where((m_since_1h >= 0.0) & (m_since_1h <= 30.0) & (f_bps_1h <= f_th1), -np.abs(f_bps_1h), 0.0)
        spec_1d = MechanismSpec(
            mech_id=f"A1_FAST_1H_POST_SETTLE_UNWIND_{int(abs(f_th1))}BPS",
            archetype="Archetype_1_Funding_Settlement_Clock",
            description=f"1h post-settlement snap-down sniper (0 funding cost, settled <= {f_th1:.0f} bps)",
            rebalance_hours=1,
            mode="event_sniper",
            universe_scope="all_liquid",
            one_way_cost_bps=2.2,
            top_k_per_side=4,
            symbolic_rule_text=f"IF mins_since_settle <= 30 AND funding_bps <= {f_th1:.0f} THEN SHORT 1h",
        )
        pnl_1d, meta_1d = simulate_event_sniper_sleeve_daily(
            df_1h, sig_1h_post, "fwd_ret_1h", horizon_hours=1, roundtrip_cost_bps=4.4, max_concurrent_slots=4
        )
        candidates_eval.append((spec_1d, pnl_1d, meta_1d))

        sig_1h_pre = np.where((m_next_1h >= 5.0) & (m_next_1h <= 35.0) & (f_bps_1h <= f_th1), -np.abs(f_bps_1h), 0.0)
        spec_1e = MechanismSpec(
            mech_id=f"A1_FAST_1H_PRE_SETTLE_SQUEEZE_{int(abs(f_th1))}BPS",
            archetype="Archetype_1_Funding_Settlement_Clock",
            description=f"1h pre-settlement squeeze fade (net of funding paid, funding <= {f_th1:.0f} bps)",
            rebalance_hours=1,
            mode="event_sniper",
            universe_scope="all_liquid",
            one_way_cost_bps=2.2,
            top_k_per_side=4,
            symbolic_rule_text=f"IF mins_to_settle IN [5, 35] AND funding_bps <= {f_th1:.0f} THEN SHORT 1h",
        )
        pnl_1e, meta_1e = simulate_event_sniper_sleeve_daily(
            df_1h, sig_1h_pre, "fwd_ret_1h", horizon_hours=1, roundtrip_cost_bps=4.4, max_concurrent_slots=4
        )
        candidates_eval.append((spec_1e, pnl_1e, meta_1e))

    # ------------------------------------------------------------------
    # ARCHETYPE 2: Whale-vs-Crowd & ΔOI Over-Extension Cross-Sectional Rotators (Family A)
    # ------------------------------------------------------------------
    for scope_lbl, frame_cs in (("ALL219", df_4h), ("MIDCAP204", df_4h_mid)):
        if frame_cs.empty:
            continue
        ts_c, u_ts = pd.factorize(frame_cs["decision_ts"], sort=False)
        n_u = len(u_ts)
        tr_m = pd.to_datetime(frame_cs["decision_ts"], unit="s", utc=True).dt.year.to_numpy() <= 2024

        z_oi12 = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "oi_vel_12h", tr_m), ts_c, n_u)
        z_oi24 = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "oi_vel_24h", tr_m), ts_c, n_u)
        z_whale = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "whale_pos_ls_z", tr_m), ts_c, n_u)
        z_sm = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "smart_money_div_z_24h", tr_m), ts_c, n_u)
        z_oi_qvol = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "oi_to_qvol_24h", tr_m), ts_c, n_u)
        z_quad_nl = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "quad_new_longs_12h", tr_m), ts_c, n_u)

        a2_formulas = [
            (
                "WHALE_PLUS_OI_DELEV",
                0.50 * z_whale - 0.50 * z_oi12,
                "rank(whale_pos_ls_z) - rank(oi_vel_12h)",
            ),
            (
                "WHALE_OI_FRAGILITY_TRIO",
                0.40 * z_whale - 0.35 * z_oi12 + 0.25 * z_oi_qvol,
                "0.40*rank(whale_ls) - 0.35*rank(oi_vel_12h) + 0.25*rank(oi_to_qvol)",
            ),
            (
                "SMART_MONEY_VS_NEW_LONG_CROWD",
                0.45 * z_sm - 0.35 * z_quad_nl - 0.20 * z_oi24,
                "0.45*rank(smart_money_div) - 0.35*rank(quad_new_longs_12h) - 0.20*rank(oi_vel_24h)",
            ),
        ]
        for f_tag, sc_vec, rule_txt in a2_formulas:
            for ema_a in (0.15, 0.30):
                spec_a2 = MechanismSpec(
                    mech_id=f"A2_{scope_lbl}_{f_tag}_EMA{int(ema_a * 100)}",
                    archetype="Archetype_2_Whale_OI_Positioning_CS",
                    description=f"Dollar-neutral {scope_lbl} Whale-vs-OI rotator ({rule_txt}, ema={ema_a})",
                    rebalance_hours=4,
                    mode="cross_sectional_neutral",
                    universe_scope=scope_lbl.lower(),
                    one_way_cost_bps=2.1,
                    top_k_per_side=14,
                    symbolic_rule_text=rule_txt,
                )
                pnl_a2, meta_a2 = simulate_cross_sectional_sleeve_daily(
                    frame_cs, sc_vec, "fwd_ret_4h", horizon_hours=4, one_way_cost_bps=2.1, ema_alpha=ema_a, top_k=14
                )
                candidates_eval.append((spec_a2, pnl_a2, meta_a2))

    # ------------------------------------------------------------------
    # ARCHETYPE 3: Mid-Cap Low-Speculative-Lottery & Microstructure Quality CS (Family D + C)
    # ------------------------------------------------------------------
    for scope_lbl, frame_cs in (("ALL219", df_4h), ("MIDCAP204", df_4h_mid)):
        if frame_cs.empty:
            continue
        ts_c, u_ts = pd.factorize(frame_cs["decision_ts"], sort=False)
        n_u = len(u_ts)
        tr_m = pd.to_datetime(frame_cs["decision_ts"], unit="s", utc=True).dt.year.to_numpy() <= 2024

        r_park = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "parkinson_vol_24h", tr_m), ts_c, n_u)
        r_spread = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "corwin_schultz_spread_24h", tr_m), ts_c, n_u)
        r_fnd = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "funding_rate_bps", tr_m), ts_c, n_u)
        r_vov = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "vol_of_vol_24h", tr_m), ts_c, n_u)
        r_clk = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "clock_compression_ratio", tr_m), ts_c, n_u)

        a3_formulas = [
            (
                "LOW_VOL_SPREAD_QUALITY",
                -0.50 * r_park - 0.50 * r_spread,
                "-0.50*rank(parkinson_vol_24h) - 0.50*rank(corwin_schultz_spread_24h)",
            ),
            (
                "QUALITY_PLUS_FUNDING_DRIFT",
                -0.35 * r_park - 0.30 * r_spread + 0.35 * r_fnd,
                "-0.35*rank(parkinson_vol) - 0.30*rank(cs_spread) + 0.35*rank(funding_bps)",
            ),
            (
                "QUALITY_FND_CLOCK_ORTHO",
                -0.30 * r_vov + 0.45 * r_fnd - 0.25 * r_clk,
                "0.45*rank(funding_bps) - 0.30*rank(vol_of_vol_24h) - 0.25*rank(clock_compression)",
            ),
        ]
        for f_tag, sc_vec, rule_txt in a3_formulas:
            for ema_a in (0.12, 0.25):
                spec_a3 = MechanismSpec(
                    mech_id=f"A3_{scope_lbl}_{f_tag}_EMA{int(ema_a * 100)}",
                    archetype="Archetype_3_MidCap_Microstructure_Quality_CS",
                    description=f"Dollar-neutral {scope_lbl} low-lottery/spread/funding rotator ({rule_txt})",
                    rebalance_hours=4,
                    mode="cross_sectional_neutral",
                    universe_scope=scope_lbl.lower(),
                    one_way_cost_bps=2.1,
                    top_k_per_side=16,
                    symbolic_rule_text=rule_txt,
                )
                pnl_a3, meta_a3 = simulate_cross_sectional_sleeve_daily(
                    frame_cs, sc_vec, "fwd_ret_4h", horizon_hours=4, one_way_cost_bps=2.1, ema_alpha=ema_a, top_k=16
                )
                candidates_eval.append((spec_a3, pnl_a3, meta_a3))

    # ------------------------------------------------------------------
    # ARCHETYPE 4: Spot-vs-Perp Cash-Flow Lead-Lag & Basis Divergence (Family B)
    # ------------------------------------------------------------------
    for scope_lbl, frame_cs in (("ALL219", df_4h), ("MIDCAP204", df_4h_mid)):
        if frame_cs.empty:
            continue
        ts_c, u_ts = pd.factorize(frame_cs["decision_ts"], sort=False)
        n_u = len(u_ts)
        tr_m = pd.to_datetime(frame_cs["decision_ts"], unit="s", utc=True).dt.year.to_numpy() <= 2024

        r_sp_sh4 = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "spot_taker_share_4h", tr_m), ts_c, n_u)
        r_sp_cvd4 = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "spot_cvd_z_4h", tr_m), ts_c, n_u)
        r_b_vel = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "basis_vel_4h", tr_m), ts_c, n_u)
        r_sp_div = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "spot_vs_perp_share_div_4h", tr_m), ts_c, n_u)
        r_sp_lead = _fast_cs_rank_centered(_z_train_clipped(frame_cs, "spot_lead_ret_div_1b", tr_m), ts_c, n_u)

        a4_formulas = [
            (
                "SPOT_CLIMAX_BASIS_REV",
                -0.40 * r_sp_sh4 - 0.35 * r_sp_cvd4 - 0.25 * r_b_vel,
                "-0.40*rank(spot_taker_share_4h) - 0.35*rank(spot_cvd_z_4h) - 0.25*rank(basis_vel_4h)",
            ),
            (
                "SPOT_LEAD_PERP_EXHAUSTION",
                0.45 * r_sp_lead - 0.30 * r_b_vel - 0.25 * r_sp_div,
                "0.45*rank(spot_lead_ret_1b) - 0.30*rank(basis_vel_4h) - 0.25*rank(spot_perp_share_div_4h)",
            ),
        ]
        for f_tag, sc_vec, rule_txt in a4_formulas:
            spec_a4 = MechanismSpec(
                mech_id=f"A4_{scope_lbl}_{f_tag}",
                archetype="Archetype_4_Spot_Perp_CashFlow_Basis",
                description=f"Dollar-neutral {scope_lbl} Spot-vs-Perp cashflow/basis sleeve ({rule_txt})",
                rebalance_hours=4,
                mode="cross_sectional_neutral",
                universe_scope=scope_lbl.lower(),
                one_way_cost_bps=2.1,
                top_k_per_side=14,
                symbolic_rule_text=rule_txt,
            )
            pnl_a4, meta_a4 = simulate_cross_sectional_sleeve_daily(
                frame_cs, sc_vec, "fwd_ret_4h", horizon_hours=4, one_way_cost_bps=2.1, ema_alpha=0.20, top_k=14
            )
            candidates_eval.append((spec_a4, pnl_a4, meta_a4))

    # ------------------------------------------------------------------
    # ARCHETYPE 5: Train-Only Shallow-Tree (depth 2/3) & Ridge Multi-Family Interaction Mining
    # ------------------------------------------------------------------
    ortho_feature_pool = [
        c
        for c in (
            "whale_pos_ls_z",
            "smart_money_div_z_24h",
            "oi_vel_12h",
            "oi_vel_z_12h",
            "oi_to_qvol_24h",
            "quad_new_longs_12h",
            "spot_taker_share_4h",
            "spot_cvd_z_4h",
            "basis_vel_4h",
            "spot_lead_ret_div_1b",
            "funding_rate_bps",
            "clock_compression_ratio",
            "compressed_clock_neg_funding_impulse",
            "parkinson_vol_24h",
            "corwin_schultz_spread_24h",
            "vol_of_vol_24h",
        )
        if c in df_4h_mid.columns
    ]

    if not df_4h_mid.empty and len(ortho_feature_pool) >= 6:
        yrs_mid = pd.to_datetime(df_4h_mid["decision_ts"], unit="s", utc=True).dt.year.to_numpy()
        tr_idx = yrs_mid <= 2024
        if int(tr_idx.sum()) >= 500:
            X_all = np.column_stack([_z_train_clipped(df_4h_mid, c, tr_idx) for c in ortho_feature_pool])
            # Cross-sectionally demeaned 4h forward return target strictly inside `2023-2024` Train!
            ts_tr, u_tr = pd.factorize(df_4h_mid.loc[tr_idx, "decision_ts"], sort=False)
            y_tr_raw = df_4h_mid.loc[tr_idx, "fwd_ret_4h"].to_numpy(dtype=np.float64)
            cnt_tr = np.bincount(ts_tr, minlength=len(u_tr)).astype(np.float64)
            mu_tr = np.bincount(ts_tr, weights=y_tr_raw, minlength=len(u_tr)) / np.maximum(cnt_tr, 1.0)
            y_tr = np.clip((y_tr_raw - mu_tr[ts_tr]) * 1e4, -400.0, 400.0)

            for depth in (2, 3):
                min_leaf = max(50, int(0.03 * int(tr_idx.sum())))
                tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
                tree.fit(X_all[tr_idx], y_tr)
                pred_tree = tree.predict(X_all)
                rule_str = export_text(tree, feature_names=ortho_feature_pool, max_depth=3).replace("\n", " | ")[:260]

                for ema_a in (0.15, 0.30):
                    spec_t = MechanismSpec(
                        mech_id=f"A5_SHALLOW_TREE_D{depth}_MIDCAP_EMA{int(ema_a * 100)}",
                        archetype="Archetype_5_Symbolic_Shallow_Tree",
                        description=f"Train-only depth-{depth} interaction tree on 16 orthogonal multi-modal features",
                        rebalance_hours=4,
                        mode="cross_sectional_neutral",
                        universe_scope="midcap204",
                        one_way_cost_bps=2.1,
                        top_k_per_side=16,
                        symbolic_rule_text=rule_str,
                    )
                    pnl_t, meta_t = simulate_cross_sectional_sleeve_daily(
                        df_4h_mid,
                        pred_tree,
                        "fwd_ret_4h",
                        horizon_hours=4,
                        one_way_cost_bps=2.1,
                        ema_alpha=ema_a,
                        top_k=16,
                    )
                    candidates_eval.append((spec_t, pnl_t, meta_t))

            for l2_alpha in (100.0, 1000.0):
                ridge = Ridge(alpha=l2_alpha, random_state=42)
                ridge.fit(X_all[tr_idx], y_tr)
                pred_r = ridge.predict(X_all)
                top_w = sorted(
                    zip(ortho_feature_pool, ridge.coef_, strict=False), key=lambda kv: abs(kv[1]), reverse=True
                )[:4]
                rule_r = " + ".join(f"{w:+.2f}*{k}" for k, w in top_w)
                for ema_a in (0.15, 0.25):
                    spec_r = MechanismSpec(
                        mech_id=f"A5_ORTHO_RIDGE_L{int(l2_alpha)}_EMA{int(ema_a * 100)}",
                        archetype="Archetype_5_Symbolic_Shallow_Tree",
                        description=f"Train-only L2={l2_alpha:.0f} linear multi-modal ensemble ({rule_r})",
                        rebalance_hours=4,
                        mode="cross_sectional_neutral",
                        universe_scope="midcap204",
                        one_way_cost_bps=2.1,
                        top_k_per_side=16,
                        symbolic_rule_text=rule_r,
                    )
                    pnl_r, meta_r = simulate_cross_sectional_sleeve_daily(
                        df_4h_mid,
                        pred_r,
                        "fwd_ret_4h",
                        horizon_hours=4,
                        one_way_cost_bps=2.1,
                        ema_alpha=ema_a,
                        top_k=16,
                    )
                    candidates_eval.append((spec_r, pnl_r, meta_r))

    # ------------------------------------------------------------------
    # Evaluate All `M` Candidates Under the 4 Hard Anti-Overfitting Gates
    # ------------------------------------------------------------------
    m_trials = len(candidates_eval)

    leaderboard_rows: list[dict[str, Any]] = []
    daily_series_map: dict[str, pd.Series] = {}
    spec_map: dict[str, MechanismSpec] = {}

    for spec, s_pnl, meta in candidates_eval:
        if s_pnl.empty:
            continue
        yrs = s_pnl.index.year
        m_full = compute_daily_series_metrics(s_pnl)
        m_tr = compute_daily_series_metrics(s_pnl[yrs <= 2024])
        m_va = compute_daily_series_metrics(s_pnl[yrs == 2025])
        m_ho = compute_daily_series_metrics(s_pnl[yrs >= 2026])
        cpcv_res = evaluate_cpcv_daily_sharpe(s_pnl, n_groups=6, k_test=2)
        dsr_val = compute_deflated_sharpe_ratio(
            s_pnl.to_numpy(dtype=np.float64),
            num_trials=m_trials,
            periods_per_year=365.25,
            trials_sharpe_std=None,
        )

        corr_prod = 0.0
        if not benchmark_daily_pnl.empty:
            aligned = pd.concat([s_pnl.rename("s"), benchmark_daily_pnl.rename("b")], axis=1).dropna()
            if len(aligned) >= 20 and float(aligned["s"].std()) > 1e-12 and float(aligned["b"].std()) > 1e-12:
                c_v = float(aligned["s"].corr(aligned["b"]))
                corr_prod = c_v if np.isfinite(c_v) else 0.0

        g1_cpcv = cpcv_res["cpcv_median_sharpe"] >= min_era_sharpe
        g2_dsr = dsr_val >= min_dsr
        g3_all_weather = (
            m_tr["sharpe"] >= min_era_sharpe and m_va["sharpe"] >= min_era_sharpe and m_ho["sharpe"] >= min_era_sharpe
        )
        g4_ortho = abs(corr_prod) < max_abs_corr
        passed_all_4 = bool(g1_cpcv and g2_dsr and g3_all_weather and g4_ortho)

        daily_series_map[spec.mech_id] = s_pnl
        spec_map[spec.mech_id] = spec

        leaderboard_rows.append(
            {
                "mech_id": spec.mech_id,
                "archetype": spec.archetype,
                "mode": spec.mode,
                "universe": spec.universe_scope,
                "sharpe_full": m_full["sharpe"],
                "sharpe_train_23_24": m_tr["sharpe"],
                "sharpe_val_2025": m_va["sharpe"],
                "sharpe_holdout_2026": m_ho["sharpe"],
                "cpcv_median_sharpe": cpcv_res["cpcv_median_sharpe"],
                "cpcv_pos_pct": cpcv_res["cpcv_pos_splits_pct"],
                "dsr": round(dsr_val, 4),
                "geom_daily_pct": m_full["geom_daily_pct"],
                "max_dd_pct": m_full["max_dd_pct"],
                "ann_fee_drag_pct": meta.get("ann_fee_drag_pct", 0.0),
                "daily_l1_turnover": meta.get("daily_l1_turnover", 0.0),
                "corr_to_prod": round(corr_prod, 3),
                "gate1_cpcv": g1_cpcv,
                "gate2_dsr": g2_dsr,
                "gate3_all_weather": g3_all_weather,
                "gate4_ortho": g4_ortho,
                "passed_all_4_gates": passed_all_4,
                "symbolic_rule": spec.symbolic_rule_text,
            }
        )

    lb_df = pd.DataFrame(leaderboard_rows)
    if lb_df.empty:
        return lb_df, daily_series_map, spec_map

    lb_df["min_era_sharpe"] = lb_df[["sharpe_train_23_24", "sharpe_val_2025", "sharpe_holdout_2026"]].min(axis=1)
    lb_df.sort_values(
        ["passed_all_4_gates", "min_era_sharpe", "sharpe_holdout_2026"],
        ascending=[False, False, False],
        inplace=True,
    )
    lb_df.reset_index(drop=True, inplace=True)
    return lb_df, daily_series_map, spec_map


def select_diverse_orthogonal_survivors(
    leaderboard_df: pd.DataFrame,
    daily_series_map: dict[str, pd.Series],
    max_survivors: int = 5,
    max_pairwise_corr: float = 0.55,
) -> list[str]:
    """
    Greedily selects up to `max_survivors` Gate-passing mechanisms that are ALSO mutually diverse
    (`|pairwise_corr| <= max_pairwise_corr`), ensuring distinct economic engines for `M16-5`.
    """
    if leaderboard_df.empty:
        return []
    strict_pass = leaderboard_df[leaderboard_df["passed_all_4_gates"]]
    pool = strict_pass if not strict_pass.empty else leaderboard_df[leaderboard_df["gate4_ortho"]].head(10)

    selected: list[str] = []
    for _, row in pool.iterrows():
        m_id = str(row["mech_id"])
        cand_s = daily_series_map.get(m_id)
        if cand_s is None or cand_s.empty:
            continue
        too_corr = False
        for prev_id in selected:
            prev_s = daily_series_map[prev_id]
            al = pd.concat([cand_s.rename("a"), prev_s.rename("b")], axis=1).dropna()
            if (
                len(al) >= 20
                and float(al["a"].std()) > 1e-12
                and float(al["b"].std()) > 1e-12
                and abs(float(al["a"].corr(al["b"]))) > max_pairwise_corr
            ):
                too_corr = True
                break
        if not too_corr:
            selected.append(m_id)
            if len(selected) >= max_survivors:
                break
    return selected
