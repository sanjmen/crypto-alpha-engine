"""
Core P1 Rebalance Barrier, Intracycle Drift & Turnover-Hysteresis Laboratory (`1-Minute` Physics).

Directly addresses the #1 empirical PnL drain observed in `crypto-mainnet` (`live_trading.db`):
`75.8%` (`273 / 360`) of live closed trades occurred at periodic / intracycle rebalance rotations
(`-$652.22`), whereas `Profit-Lock` (`+$117.18`, `64.6%` W) and `Take-Profit` (`+$437.17`, `100%` W)
generated `+$554.35` net.

Evaluates 10 exact `1-minute` One-Way portfolio barrier configurations (`2023–2026` + `2025` + `2026`
fresh `$2,700` Cushion Stress-Test + `15`-split CPCV on daily excess/portfolio returns) alongside
the active production stack (`Core P1 + Manga P2 + M09/M23/M04 + M26_L2SweepAirPocket`, `M25=OFF`):
  1. `intracycle_mode`:
     - `"full_rerank"` (current live `enable_intracycle_rebalance=True`: triggers full mid-cycle
       re-ranking rebalance whenever `TP`/`Profit-Lock` fills push residual net drift `>= 25%`).
     - `"off"` (strict `00:00 / 08:00 / 16:00 UTC` funding-aligned 8h clock).
     - `"collapsed_trim_only"` (never opens new positions mid-cycle; ONLY trims surviving leg
       `reduceOnly` if opposing leg `< 15%` AND unhedged residual `>= 35%` of equity).
  2. `bhs_enter_top_k` / `bhs_hold_top_k` & `deadband_mult`: Novy-Marx rank retention & turnover deadband.
  3. `min_partial_trim_equity_pct`: Execution-layer no-trade buffer suppressing cosmetic `$30–$80`
     same-sign partial micro-trims while executing `100%` of full closes, sign flips, and new entries.
  4. `fast_min_tstat` & `comb_ema_span`: Silencing `F3_Fast_Swing` churn when trailing IC `t-stat`
     lacks statistical edge, and smoothing composite factor transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.p2_portfolio_integration import (
    compute_portfolio_era_metrics,
    convert_trade_df_to_hourly_sniper_events,
)
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.event_snipers import (
    SNIPER_BRACKET_PROFILES,
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_M26_STRAT,
)
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine


@dataclass(frozen=True)
class CoreRebalanceBarrierSpec:
    config_id: str
    description: str
    intracycle_mode: str = "off"  # "full_rerank", "off", "collapsed_trim_only"
    bhs_enter_top_k: int = 3
    bhs_hold_top_k: int = 6
    deadband_mult: float = 1.25
    min_partial_trim_equity_pct: float = 0.009
    comb_ema_span: float = 3.0
    fast_min_tstat: float | None = None


REBALANCE_BARRIER_SPECS: tuple[CoreRebalanceBarrierSpec, ...] = (
    CoreRebalanceBarrierSpec(
        config_id="00_LIVE_INTRACYCLE_FULL_RERANK",
        description="Current Live Mainnet (Intracycle Full Re-Rank >=25% drift, BHS(3,6), DBx1.25, $25 min trim)",
        intracycle_mode="full_rerank",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="01_STRICT_8H_NO_INTRACYCLE",
        description="Strict 8h Funding-Aligned Clock (Intracycle OFF, BHS(3,6), DBx1.25)",
        intracycle_mode="off",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="02_COLLAPSED_TRIM_ONLY_INTRACYCLE",
        description="8h Clock + Emergency ReduceOnly Trim ONLY on One-Side Collapse (>=35% unhedged)",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="03_PARTIAL_TRIM_BUFFER_3.5PCT",
        description="8h Collapsed-Trim + Suppress Partial Same-Sign Micro-Trims < 3.5% Equity (~$95)",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.035,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="04_HYSTERESIS_BHS_3_7_DB_1.75",
        description="8h Collapsed-Trim + Wider Rank Retention BHS(3,7) + Deadband x1.75",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=7,
        deadband_mult=1.75,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="05_HYSTERESIS_BHS_3_8_DB_2.25",
        description="8h Collapsed-Trim + Aggressive Retention BHS(3,8) + Deadband x2.25",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=8,
        deadband_mult=2.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="06_FAST_SWING_TSTAT_GATE_0.50",
        description="8h Collapsed-Trim + Gate F3_Fast_Swing unless Trailing IC t-stat >= +0.50",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=3.0,
        fast_min_tstat=0.50,
    ),
    CoreRebalanceBarrierSpec(
        config_id="07_SCORE_SMOOTH_EMA_6H",
        description="8h Collapsed-Trim + Composite Score Smoothing EMA(6h vs 3h)",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        deadband_mult=1.25,
        min_partial_trim_equity_pct=0.009,
        comb_ema_span=6.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="08_UNIFIED_SHIELD_A",
        description="Collapsed-Trim + BHS(3,7) + DBx1.75 + 3.5% Partial-Trim Buffer",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=7,
        deadband_mult=1.75,
        min_partial_trim_equity_pct=0.035,
        comb_ema_span=3.0,
        fast_min_tstat=None,
    ),
    CoreRebalanceBarrierSpec(
        config_id="09_UNIFIED_SHIELD_B_FULL",
        description="Collapsed-Trim + BHS(3,7) + DBx1.75 + 3.5% TrimBuf + FastGate(0.50) + EMA(6h)",
        intracycle_mode="collapsed_trim_only",
        bhs_enter_top_k=3,
        bhs_hold_top_k=7,
        deadband_mult=1.75,
        min_partial_trim_equity_pct=0.035,
        comb_ema_span=6.0,
        fast_min_tstat=0.50,
    ),
)


def compute_cpcv_portfolio_sharpe_stability(
    daily_rets: pd.Series, n_groups: int = 6, k_test: int = 2
) -> dict[str, float]:
    """
    Evaluates `C(6,2) = 15` purged out-of-sample combinations on daily portfolio returns
    (with 2-day purge boundary around each test block) to verify multi-regime Sharpe stability.
    """
    arr = daily_rets.dropna().to_numpy(dtype=np.float64)
    n = len(arr)
    if n < 60:
        return {"cpcv_mean_sharpe": 0.0, "cpcv_min_sharpe": 0.0, "cpcv_pos_splits_pct": 0.0}

    bounds = np.linspace(0, n, n_groups + 1, dtype=int)
    sharpes: list[float] = []
    for combo in combinations(range(n_groups), k_test):
        mask = np.zeros(n, dtype=bool)
        for g in combo:
            s_i = min(n, bounds[g] + 1)
            e_i = max(s_i, bounds[g + 1] - 1)
            mask[s_i:e_i] = True
        sub = arr[mask]
        if len(sub) >= 10:
            sd = float(np.std(sub, ddof=1))
            sh = (float(np.mean(sub)) / sd) * np.sqrt(365.25) if sd > 1e-12 else 0.0
            sharpes.append(sh)

    if not sharpes:
        return {"cpcv_mean_sharpe": 0.0, "cpcv_min_sharpe": 0.0, "cpcv_pos_splits_pct": 0.0}
    pos_pct = 100.0 * sum(1 for x in sharpes if x > 0.0) / len(sharpes)
    return {
        "cpcv_mean_sharpe": round(float(np.mean(sharpes)), 2),
        "cpcv_min_sharpe": round(float(np.min(sharpes)), 2),
        "cpcv_pos_splits_pct": round(pos_pct, 1),
    }


def simulate_core_barrier_1m_portfolio(
    mats: dict[str, Any],
    o_1m_core: np.ndarray,
    h_1m_core: np.ndarray,
    l_1m_core: np.ndarray,
    eval_idx: pd.DatetimeIndex,
    open_core: np.ndarray,
    p2_w_mat: np.ndarray | None,
    spec: CoreRebalanceBarrierSpec,
    resolved_sniper_events_by_hour: dict[int, list[dict[str, Any]]] | None = None,
    allowed_sniper_strats: set[str] | None = None,
    p1_risk_scale: float = 0.85,
    sniper_slot_lev: float = 0.24,
    sniper_max_pos: int = 3,
    rebal_fee: float = 0.00042,
    initial_capital: float = 2700.0,
    start_leverage: float = 2.5,
    max_leverage: float = 5.0,
    recovery_boost_eq: float | None = 3050.0,
    recovery_boost_lev: float | None = 3.2,
    cushion_floor_usd: float | None = 2500.0,
    cushion_buffer_usd: float = 240.0,
) -> dict[str, Any]:
    """
    Runs the exact 1-minute One-Way portfolio simulator with full barrier-hit accounting:
      - Scheduled 8h rebalances vs Intracycle full re-ranks vs Collapsed-Only ReduceOnly trims
      - Rebalance full exits vs partial micro-trims vs Profit-Lock exits vs Take-Profit exits
      - Total Core transaction fee drag (`%` of equity summed over horizon)
    """
    core_cols = list(mats["close"].columns)
    n_b, n_c = len(eval_idx), len(core_cols)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    c_arr = mats["close"].loc[eval_idx].to_numpy(dtype=np.float64)
    f_arr = mats["funding"].loc[eval_idx].to_numpy(dtype=np.float64)
    w_src = mats["w_masked"] if "w_masked" in mats else mats["w_base_1x"]
    w1x_sh = w_src.shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    vol_sh = mats["vol_scalar"].shift(1).fillna(1.0).loc[eval_idx].to_numpy(dtype=np.float64)
    chop_sh = mats["alpha_chop_overlay"].shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    v24_sh = mats["vol_24h"].shift(1).ffill().fillna(0.02).loc[eval_idx].to_numpy(dtype=np.float64)
    max_tr_sh = (
        mats["t_df"].shift(1).fillna(0.0).loc[eval_idx][["macro", "med", "fast"]].max(axis=1).to_numpy(dtype=np.float64)
    )
    hrs = eval_idx.hour.to_numpy()

    ctrl = SteppedLeverageController(
        initial_capital=initial_capital,
        start_leverage=start_leverage,
        max_leverage=max_leverage,
        floor_leverage=min(2.0, start_leverage),
        dd_bracket=0.06 if max_leverage > start_leverage else 0.50,
        base_deadband_ratio=(0.04 / 2.2) * float(spec.deadband_mult),
        bhs_enter_top_k=int(spec.bhs_enter_top_k),
        bhs_hold_top_k=int(spec.bhs_hold_top_k),
        recovery_boost_equity=recovery_boost_eq,
        recovery_boost_leverage=recovery_boost_lev,
        cushion_floor_usd=cushion_floor_usd,
        cushion_buffer_usd=cushion_buffer_usd,
    )

    equity = float(initial_capital)
    curr_w = np.zeros(n_c, dtype=np.float64)
    prev_p2_w = np.zeros(n_c, dtype=np.float64)
    active_p2_w = np.zeros(n_c, dtype=np.float64)
    entry_px = np.zeros(n_c, dtype=np.float64)
    lock_armed = np.zeros(n_c, dtype=bool)
    cooldown_until = np.full((n_c, 2), -1, dtype=np.int64)

    active_snipers: list[dict[str, Any]] = []
    sniper_cd_until: dict[str, int] = {}
    sniper_exec_counts: dict[str, int] = {k: 0 for k in SNIPER_BRACKET_PROFILES}
    events_map = resolved_sniper_events_by_hour or {}

    net_rets = np.zeros(n_b, dtype=np.float64)
    eq_hourly = np.zeros(n_b, dtype=np.float64)

    # Barrier & Churn Counters
    n_sched_rebal = 0
    n_intracycle_rerank = 0
    n_collapsed_trims = 0
    n_rebal_full_closes = 0
    n_rebal_partial_adjusts = 0
    n_suppressed_micro_trims = 0
    n_profit_lock_exits = 0
    n_tp_exits = 0
    cum_core_turnover_1x = 0.0
    cum_core_fee_pct = 0.0
    pending_bracket_exit_since_rebal = False

    def _run_full_rebalance_at_open(t_idx: int, sniper_syms_now: set[str]) -> float:
        nonlocal curr_w, prev_p2_w, active_p2_w, n_rebal_full_closes, n_rebal_partial_adjusts
        nonlocal n_suppressed_micro_trims, cum_core_turnover_1x
        raw_u = w1x_sh[t_idx].copy()
        for i in range(n_c):
            if (
                core_cols[i] in sniper_syms_now
                or (raw_u[i] > 0 and t_idx < cooldown_until[i, 0])
                or (raw_u[i] < 0 and t_idx < cooldown_until[i, 1])
            ):
                raw_u[i] = 0.0
        pl = float(np.maximum(raw_u, 0.0).sum())
        nl = float(np.maximum(-raw_u, 0.0).sum())
        og = float(np.abs(w1x_sh[t_idx]).sum())
        if pl > 1e-9 and nl > 1e-9 and og > 1e-9:
            raw_u = np.where(raw_u > 0, raw_u * ((og / 2.0) / pl), np.where(raw_u < 0, raw_u * ((og / 2.0) / nl), 0.0))
        else:
            raw_u[:] = 0.0

        u_dict = {core_cols[i]: float(raw_u[i]) for i in range(n_c)}
        if p2_w_mat is not None and np.any(np.abs(prev_p2_w) > 1e-8):
            inv_sc = 1.0 / max(p1_risk_scale, 0.25)
            p1_est = (curr_w - prev_p2_w) * inv_sc
            prev_dict = {core_cols[i]: float(p1_est[i]) for i in range(n_c)} if np.any(np.abs(p1_est) > 1e-6) else None
        else:
            prev_dict = {core_cols[i]: float(curr_w[i]) for i in range(n_c)} if np.any(np.abs(curr_w) > 1e-8) else None

        sc_d, snap_t = ctrl.scale_and_deadband_weights(u_dict, prev_dict, equity, float(vol_sh[t_idx]))
        p1_des = np.array(
            [0.0 if core_cols[i] in sniper_syms_now else sc_d.get(core_cols[i], 0.0) for i in range(n_c)],
            dtype=np.float64,
        )

        if p2_w_mat is not None:
            p1_scaled = p1_des * p1_risk_scale
            ramp_m = min(1.65, max(0.50, snap_t.active_base_leverage / max(start_leverage, 1.0)))
            p2_sc = ramp_m * snap_t.cushion_scalar
            raw_p2 = np.array(
                [0.0 if core_cols[i] in sniper_syms_now else p2_w_mat[t_idx, i] * p2_sc for i in range(n_c)]
            )
            eff_p2 = np.where(
                (np.abs(raw_p2) >= 1e-6) & ((raw_p2 * prev_p2_w) > 0.0) & (np.abs(raw_p2 - prev_p2_w) < 0.030 * p2_sc),
                prev_p2_w,
                raw_p2,
            )
            eff_p2 = np.array([0.0 if core_cols[i] in sniper_syms_now else eff_p2[i] for i in range(n_c)])
            desired_w = p1_scaled + eff_p2
            if float(np.sum(np.abs(desired_w)) * equity) >= 2000.0:
                stub_m = (np.abs(desired_w) > 0.0) & (np.abs(desired_w * equity) < 155.0) & (np.abs(eff_p2) > 0.0)
                eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                desired_w = np.where(stub_m, 0.0, desired_w)
            prev_p2_w = eff_p2.copy()
            active_p2_w = eff_p2.copy()
        else:
            desired_w = p1_des

        # Execution-layer partial-trim hysteresis buffer (suppresses cosmetic same-sign micro-trims)
        min_trim_w = float(spec.min_partial_trim_equity_pct)
        if min_trim_w > 0.010:
            cand_w = desired_w.copy()
            for i in range(n_c):
                if core_cols[i] in sniper_syms_now:
                    continue
                is_same_sign_open = (
                    (abs(curr_w[i]) > 1e-6) and (abs(cand_w[i]) > 1e-6) and ((curr_w[i] * cand_w[i]) > 0)
                )
                if is_same_sign_open and abs(cand_w[i] - curr_w[i]) < min_trim_w:
                    cand_w[i] = curr_w[i]
                    n_suppressed_micro_trims += 1
            # Guard: only adopt buffered weights if residual Core net drift stays within 10% of equity
            p2_net = float(np.sum(active_p2_w))
            if abs(float(np.sum(cand_w)) - p2_net) <= 0.10:
                desired_w = cand_w

        fee_this = 0.0
        cur_o_bar = open_core[t_idx]
        for i in range(n_c):
            if core_cols[i] in sniper_syms_now:
                continue
            dw = desired_w[i] - curr_w[i]
            if abs(dw) < max(1e-6, min(0.009, min_trim_w) if (curr_w[i] * desired_w[i] > 0) else 1e-6):
                desired_w[i] = curr_w[i]
                continue
            fee_this += abs(dw) * rebal_fee
            cum_core_turnover_1x += abs(dw)
            if abs(curr_w[i]) > 1e-6 and (abs(desired_w[i]) < 1e-8 or (curr_w[i] * desired_w[i] < 0)):
                n_rebal_full_closes += 1
            elif abs(curr_w[i]) > 1e-6:
                n_rebal_partial_adjusts += 1

            if abs(desired_w[i]) < 1e-8:
                entry_px[i], lock_armed[i] = 0.0, False
            elif abs(curr_w[i]) < 1e-8 or (curr_w[i] * desired_w[i] < 0):
                entry_px[i], lock_armed[i] = cur_o_bar[i], False
            elif abs(desired_w[i]) > abs(curr_w[i]) + 1e-6:
                add_w = abs(desired_w[i]) - abs(curr_w[i])
                entry_px[i] = (abs(curr_w[i]) * entry_px[i] + add_w * cur_o_bar[i]) / abs(desired_w[i])

        curr_w = desired_w
        return fee_this

    for t in range(n_b):
        hr_ts = int(ts_eval[t])
        active_snipers = [sp for sp in active_snipers if int(sp["exit_ts"]) > hr_ts]
        sniper_open_syms = {str(sp["sym"]) for sp in active_snipers}

        if sniper_slot_lev > 0 and allowed_sniper_strats and hr_ts in events_map:
            snap_s = ctrl.update_state(current_equity=equity, vol_scalar=1.0)
            ramp_s = min(1.65, max(0.50, snap_s.active_base_leverage / max(start_leverage, 1.0)))
            eff_slot_w = sniper_slot_lev * snap_s.cushion_scalar * ramp_s
            core_open_syms = {core_cols[i] for i in range(n_c) if abs(curr_w[i]) > 1e-6}

            evs_hr = sorted(events_map[hr_ts], key=lambda x: (int(x["entry_ts"]), -float(x["score"])))
            for ev in evs_hr:
                st_name = str(ev["strat"])
                if st_name not in allowed_sniper_strats:
                    continue
                entry_ts = int(ev["entry_ts"])
                active_now = [sp for sp in active_snipers if int(sp["exit_ts"]) > entry_ts]
                if len(active_now) >= sniper_max_pos:
                    continue
                sym_name = str(ev["sym"])
                if (
                    sym_name in core_open_syms
                    or any(str(sp["sym"]) == sym_name for sp in active_now)
                    or entry_ts < sniper_cd_until.get(sym_name, 0)
                ):
                    continue
                ex_ts = int(ev["exit_ts"])
                active_snipers.append(
                    {
                        "sym": sym_name,
                        "exit_ts": ex_ts,
                        "weight": eff_slot_w,
                        "hourly_ret": ev["hourly_ret"],
                    }
                )
                sniper_open_syms.add(sym_name)
                sniper_cd_until[sym_name] = ex_ts + (14400 if bool(ev["was_loss"]) else 3600)
                sniper_exec_counts[st_name] = sniper_exec_counts.get(st_name, 0) + 1

        prev_c = c_arr[t - 1] if t > 0 else open_core[0]
        cur_o, cur_c = open_core[t], c_arr[t]
        bar_pnl, bar_fee = 0.0, 0.0
        gap_ret = np.where(prev_c > 0, (cur_o - prev_c) / prev_c, 0.0)
        bar_pnl += float(np.sum(curr_w * gap_ret))

        if hrs[t] % 8 == 0:
            bar_fee += _run_full_rebalance_at_open(t, sniper_open_syms)
            n_sched_rebal += 1
            pending_bracket_exit_since_rebal = False
        elif pending_bracket_exit_since_rebal and spec.intracycle_mode != "off":
            p1_only_w = curr_w - active_p2_w
            long_w = float(np.sum(np.maximum(p1_only_w, 0.0)))
            short_w = float(np.sum(np.maximum(-p1_only_w, 0.0)))
            gross_w = long_w + short_w
            res_drift = abs(long_w - short_w)
            denom_w = max(gross_w, 1.0)
            one_collapsed = min(long_w, short_w) < 0.20 * max(long_w, short_w)

            if spec.intracycle_mode == "full_rerank":
                if (res_drift / denom_w) >= 0.25 or (one_collapsed and res_drift >= 0.12):
                    bar_fee += _run_full_rebalance_at_open(t, sniper_open_syms)
                    n_intracycle_rerank += 1
                    pending_bracket_exit_since_rebal = False
            elif (
                spec.intracycle_mode == "collapsed_trim_only"
                and min(long_w, short_w) < 0.15 * max(long_w, short_w)
                and res_drift >= 0.35
            ):
                # ReduceOnly proportional trim on the unhedged surviving P1 leg — ZERO new opens!
                if long_w > short_w and long_w > 1e-6:
                    target_long = short_w + 0.20
                    scale_f = min(1.0, max(0.0, target_long / long_w))
                    for i in range(n_c):
                        if p1_only_w[i] > 1e-6 and core_cols[i] not in sniper_open_syms:
                            new_p1_i = p1_only_w[i] * scale_f
                            new_tot_i = active_p2_w[i] + new_p1_i
                            dw_i = abs(new_tot_i - curr_w[i])
                            bar_fee += dw_i * rebal_fee
                            cum_core_turnover_1x += dw_i
                            curr_w[i] = new_tot_i
                            n_rebal_partial_adjusts += 1
                elif short_w > long_w and short_w > 1e-6:
                    target_short = long_w + 0.20
                    scale_f = min(1.0, max(0.0, target_short / short_w))
                    for i in range(n_c):
                        if p1_only_w[i] < -1e-6 and core_cols[i] not in sniper_open_syms:
                            new_p1_i = p1_only_w[i] * scale_f
                            new_tot_i = active_p2_w[i] + new_p1_i
                            dw_i = abs(new_tot_i - curr_w[i])
                            bar_fee += dw_i * rebal_fee
                            cum_core_turnover_1x += dw_i
                            curr_w[i] = new_tot_i
                            n_rebal_partial_adjusts += 1
                n_collapsed_trims += 1
                pending_bracket_exit_since_rebal = False

        # 1-minute intrabar Profit-Lock & Take-Profit resolution on Core + Manga P2
        for i in range(n_c):
            w_i = curr_w[i]
            if abs(w_i) < 1e-8 or entry_px[i] <= 0 or not np.isfinite(cur_o[i]) or cur_o[i] <= 0:
                continue
            chop_w, max_tr, sig1h = float(chop_sh[t]), float(max_tr_sh[t]), float(v24_sh[t, i])
            p2_dom = (p2_w_mat is not None) and (abs(active_p2_w[i]) >= 0.15) and ((active_p2_w[i] * w_i) > 0)
            if (chop_w >= 0.25 or max_tr <= 0.0) and not p2_dom:
                sig4 = 2.0 * min(0.035, max(0.004, sig1h))
                cmult = 1.95 - 0.35 * min(1.0, max(0.0, chop_w / 0.75))
                eff_tp = min(0.036, max(0.028, sig4 * cmult))
                eff_ltrig, eff_llvl = 0.024, 0.012
            else:
                eff_tp, eff_ltrig, eff_llvl = 0.050, 0.035, 0.015

            sgn = 1.0 if w_i > 0 else -1.0
            ep = entry_px[i]
            tp_px = ep * (1.0 + sgn * eff_tp)
            lk_px = ep * (1.0 + sgn * eff_llvl)
            trig_px = ep * (1.0 + sgn * eff_ltrig)
            hv, lv, ov = h_1m_core[t, :, i], l_1m_core[t, :, i], o_1m_core[t, :, i]
            if not np.any(np.isfinite(hv)):
                bar_pnl += w_i * ((cur_c[i] - cur_o[i]) / cur_o[i])
                continue
            max_h, min_l = float(np.nanmax(hv)), float(np.nanmin(lv))
            can_hit = (sgn > 0 and (max_h >= min(tp_px, trig_px) or (lock_armed[i] and min_l <= lk_px))) or (
                sgn < 0 and (min_l <= max(tp_px, trig_px) or (lock_armed[i] and max_h >= lk_px))
            )
            if not can_hit:
                bar_pnl += w_i * ((cur_c[i] - cur_o[i]) / cur_o[i])
                continue
            exited, exit_px, exit_fee, hit_kind = False, cur_c[i], 0.00020, ""
            for m in range(60):
                om, hm, lm = ov[m], hv[m], lv[m]
                if not np.isfinite(om):
                    continue
                if lock_armed[i] and ((sgn > 0 and lm <= lk_px) or (sgn < 0 and hm >= lk_px)):
                    exit_px = min(om, lk_px) if sgn > 0 else max(om, lk_px)
                    exit_fee, exited, hit_kind = 0.00055, True, "LOCK"
                    break
                if (sgn > 0 and hm >= tp_px) or (sgn < 0 and lm <= tp_px):
                    exit_px = max(om, tp_px) if sgn > 0 else min(om, tp_px)
                    exit_fee, exited, hit_kind = 0.00020, True, "TP"
                    break
                if not lock_armed[i] and ((sgn > 0 and hm >= trig_px) or (sgn < 0 and lm <= trig_px)):
                    lock_armed[i] = True
            if exited:
                bar_pnl += w_i * ((exit_px - cur_o[i]) / cur_o[i])
                bar_fee += abs(w_i) * exit_fee
                if hit_kind == "LOCK":
                    n_profit_lock_exits += 1
                elif hit_kind == "TP":
                    n_tp_exits += 1
                if sgn * (exit_px - ep) < 0:
                    cooldown_until[i, 0 if sgn > 0 else 1] = t + 5
                curr_w[i], entry_px[i], lock_armed[i], active_p2_w[i] = 0.0, 0.0, False, 0.0
                pending_bracket_exit_since_rebal = True
            else:
                bar_pnl += w_i * ((cur_c[i] - cur_o[i]) / cur_o[i])

        cum_core_fee_pct += bar_fee * 100.0
        sniper_hr_pnl = sum(float(sp["weight"]) * float(sp["hourly_ret"].get(hr_ts, 0.0)) for sp in active_snipers)
        nr = bar_pnl - bar_fee - float(np.sum(curr_w * f_arr[t])) + sniper_hr_pnl
        net_rets[t] = nr
        equity = max(1.0, equity * (1.0 + nr))
        eq_hourly[t] = equity

    s_1h = pd.Series(net_rets, index=eval_idx)
    d_s = s_1h.resample("1D").apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
    eq_s = pd.Series(eq_hourly, index=eval_idx)
    roll24_peak = eq_s.rolling(24, min_periods=1).max()
    worst_24h_dd = abs(float(((eq_s - roll24_peak) / roll24_peak).min())) * 100.0
    return {
        "daily": d_s,
        "eq_1h": eq_s,
        "final_eq": float(eq_s.iloc[-1]),
        "min_eq": float(eq_s.min()),
        "worst_24h_dd": worst_24h_dd,
        "sniper_counts": sniper_exec_counts,
        "barrier_stats": {
            "sched_rebalances": n_sched_rebal,
            "intracycle_reranks": n_intracycle_rerank,
            "collapsed_trims": n_collapsed_trims,
            "rebal_full_closes": n_rebal_full_closes,
            "rebal_partial_adjusts": n_rebal_partial_adjusts,
            "suppressed_micro_trims": n_suppressed_micro_trims,
            "profit_lock_exits": n_profit_lock_exits,
            "tp_exits": n_tp_exits,
            "core_turnover_1x": round(cum_core_turnover_1x, 1),
            "core_fee_drag_pct": round(cum_core_fee_pct, 2),
        },
    }


def run_core_rebalance_barrier_ablation(
    bundle: dict[str, Any],
    bars_1h_top15: dict[str, pd.DataFrame],
    funding_1h_top15: dict[str, pd.Series],
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[CoreRebalanceBarrierSpec, ...] = REBALANCE_BARRIER_SPECS,
) -> pd.DataFrame:
    """
    Evaluates all `CoreRebalanceBarrierSpec` configurations across both:
      1) Full 4-Year Multi-Regime Horizon (`2023–2026`, `23-24`, `2025`, `2026`, `15`-split CPCV)
      2) Fresh `$2,700` `2026-01-01` Live Cushion Stress-Test against the `$2,500` CPPI Hard Floor.
    """
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]

    events_by_hr: dict[int, list[dict[str, Any]]] = {}
    if "base_sniper_events_by_hour" in bundle:
        events_by_hr = {int(k): list(v) for k, v in bundle["base_sniper_events_by_hour"].items()}
    else:
        for ev_b in bundle.get("base_sniper_events", []):
            events_by_hr.setdefault(int(ev_b["hr_ts"]), []).append(ev_b)
    allowed_strats = {SNIPER_M09_STRAT, SNIPER_M23_STRAT, SNIPER_M04_STRAT}
    if m26_trades_df is not None and not m26_trades_df.empty:
        sub_26 = m26_trades_df[
            (m26_trades_df["mech_id"] == "P6A_L2_3_INST_SWEEP_CONC")
            & (m26_trades_df["exec_mode"] == "MONO_A_CONVEX_TRAIL")
        ]
        if not sub_26.empty:
            for ev in convert_trade_df_to_hourly_sniper_events(sub_26, SNIPER_M26_STRAT):
                events_by_hr.setdefault(int(ev["hr_ts"]), []).append(ev)
            allowed_strats.add(SNIPER_M26_STRAT)

    # Cache `SevenFactorOnlineEngine.compute_matrices` by `(comb_ema_span, fast_min_tstat)` so each unique matrix runs once
    mats_cache: dict[tuple[float, float | None], dict[str, Any]] = {}
    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}

    rows: list[dict[str, Any]] = []
    for sp in specs:
        m_key = (float(sp.comb_ema_span), sp.fast_min_tstat)
        if m_key not in mats_cache:
            min_t_map: dict[str, float] = {"liq": 0.75}
            if sp.fast_min_tstat is not None:
                min_t_map["fast"] = float(sp.fast_min_tstat)
            eng = SevenFactorOnlineEngine(
                comb_ema_span=float(sp.comb_ema_span),
                f6_min_tstat=0.75,
                min_tstat_by_factor=min_t_map,
                enable_chop_range_arb=True,
                chop_max_alpha=0.75,
                enable_holistic_regime_shields=True,
            )
            mats_m = eng.compute_matrices(bars_1h_top15, funding_rates=funding_1h_top15)
            raw_close_core = pd.DataFrame({s: bars_1h_top15[s]["close"] for s in mats_m["close"].columns}).reindex(
                mats_m["close"].index
            )
            age_ok = raw_close_core.notna().cumsum() >= 168
            mats_m["w_masked"] = mats_m["w_base_1x"].where(age_ok, 0.0)
            mats_cache[m_key] = mats_m
        mats_cur = mats_cache[m_key]

        sim_4y = simulate_core_barrier_1m_portfolio(
            mats=mats_cur,
            o_1m_core=bundle["o_1m_core"],
            h_1m_core=bundle["h_1m_core"],
            l_1m_core=bundle["l_1m_core"],
            eval_idx=eval_idx,
            open_core=bundle["open_core"],
            p2_w_mat=bundle["p2_w_mat"],
            spec=sp,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])
        bs = sim_4y["barrier_stats"]

        fresh_final = 2700.0
        fresh_min = 2700.0
        fresh_floor_ok = True
        if has_2026:
            sim_26 = simulate_core_barrier_1m_portfolio(
                mats=mats_cur,
                o_1m_core=bundle["o_1m_core"][mask_2026],
                h_1m_core=bundle["h_1m_core"][mask_2026],
                l_1m_core=bundle["l_1m_core"][mask_2026],
                eval_idx=idx_2026,
                open_core=bundle["open_core"][mask_2026],
                p2_w_mat=bundle["p2_w_mat"][mask_2026] if bundle["p2_w_mat"] is not None else None,
                spec=sp,
                resolved_sniper_events_by_hour=evs_2026,
                allowed_sniper_strats=allowed_strats,
            )
            fresh_final = round(float(sim_26["final_eq"]), 2)
            fresh_min = round(float(sim_26["min_eq"]), 2)
            fresh_floor_ok = bool(fresh_min >= 2500.0)

        tot_exits = max(1, bs["rebal_full_closes"] + bs["profit_lock_exits"] + bs["tp_exits"])
        rebal_exit_share_pct = round(100.0 * bs["rebal_full_closes"] / tot_exits, 1)

        rows.append(
            {
                "config_id": sp.config_id,
                "description": sp.description,
                "sharpe_4y": m_4y["full_sharpe"],
                "cagr_4y_pct": m_4y["full_cagr_pct"],
                "max_dd_4y_pct": m_4y["full_max_dd_pct"],
                "sharpe_23_24": m_4y["sharpe_23_24"],
                "sharpe_2025": m_4y["sharpe_2025"],
                "sharpe_2026": m_4y["sharpe_2026"],
                "cpcv_mean_sharpe": cpcv_4y["cpcv_mean_sharpe"],
                "cpcv_min_sharpe": cpcv_4y["cpcv_min_sharpe"],
                "cpcv_pos_splits_pct": cpcv_4y["cpcv_pos_splits_pct"],
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_floor_ok,
                "intracycle_reranks": bs["intracycle_reranks"],
                "collapsed_trims": bs["collapsed_trims"],
                "rebal_full_closes": bs["rebal_full_closes"],
                "rebal_partial_adjusts": bs["rebal_partial_adjusts"],
                "profit_lock_exits": bs["profit_lock_exits"],
                "tp_exits": bs["tp_exits"],
                "rebal_exit_share_pct": rebal_exit_share_pct,
                "core_turnover_1x": bs["core_turnover_1x"],
                "core_fee_drag_pct": bs["core_fee_drag_pct"],
            }
        )

    return pd.DataFrame(rows)
