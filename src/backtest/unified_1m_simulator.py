"""
Unified 1-Minute One-Way Exchange Simulator for Production Portfolio Backtesting.

Consolidates the canonical 1-minute intrabar production simulator (previously prototyped in Colab
Phase 1 & Phase 2 studies) directly into the repository so every backtest and multi-sleeve ablation
uses the EXACT production engines (`SevenFactorOnlineEngine`, `SteppedLeverageController`,
`CausalDollarBarTSMOMEngine`, and `EventDrivenSniperEngine` / new Counterparty-Mechanics Snipers)
with:
  1. Pure linear returns `(P_exit - P_entry) / P_entry * side` (ZERO `np.log` convexity bias!).
  2. Strict One-Way symbol non-collision across Core P1, Manga P2, and isolated Event Snipers.
  3. Tier-realistic Maker & Taker transaction costs (`tier_cost_bps` by causal 30d ADV rank).
  4. Exact 1-minute intrabar Profit-Lock (`2.4%->1.2%` in Chop, `3.5%->1.5%` in Trend), Stop-Loss,
     Take-Profit, Trailing Stop, and exact `:00:00` funding settlements.
  5. Live `$2,700` Two-Phase SafeRamp + Cushion-Safe CPPI Governor (`$2,500` Hard Floor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.event_snipers import SNIPER_BRACKET_PROFILES, SniperBracketSpec


def tier_cost_bps(rank_idx: int, maker: bool = False) -> float:
    """Empirical tier-realistic one-way fee + half-spread/slippage in bps by causal 30d ADV rank."""
    if rank_idx < 5:
        return 1.8 if maker else 4.0
    if rank_idx < 20:
        return 2.2 if maker else 5.0
    if rank_idx < 50:
        return 3.5 if maker else 7.5
    return 6.0 if maker else 13.5


def resolve_sniper_1m_path(
    entry_ts: int,
    side: int,
    spec: SniperBracketSpec,
    adv_rank: int,
    ts_1m: np.ndarray,
    open_1m: np.ndarray,
    high_1m: np.ndarray,
    low_1m: np.ndarray,
    close_1m: np.ndarray,
    fr_map_sym: dict[int, float] | None = None,
    entry_is_maker: bool = False,
    pre_settlement_guard_minute: int | None = None,
) -> tuple[int, dict[int, float], bool, dict[str, float]] | None:
    """
    Walks exact 1-minute OHLC bars for a single discrete event trade using pure linear returns
    and `100%` pessimistic intrabar Stop-Loss priority (`SL` checked strictly before `TP`).

    Returns `(exit_ts_sec, hourly_unlevered_returns_by_hour_ts, is_loss, trade_diagnostics)`.
    """
    i0 = int(np.searchsorted(ts_1m, entry_ts))
    if i0 >= len(ts_1m) - 3 or abs(int(ts_1m[i0]) - int(entry_ts)) > 60:
        return None

    hold_min = max(1, int(round(spec.max_hold_hours * 60.0)))
    i_end = min(len(ts_1m) - 1, i0 + hold_min - 1)
    ep = float(open_1m[i0])
    if not np.isfinite(ep) or ep <= 0.0:
        return None

    tk_c = tier_cost_bps(adv_rank, maker=False) * 1e-4
    mk_c = tier_cost_bps(adv_rank, maker=True) * 1e-4
    entry_fee = mk_c if entry_is_maker else tk_c

    sgn = 1.0 if side > 0 else -1.0
    cur_sl = ep * (1.0 - sgn * spec.stop_loss_pct)
    tp_px = ep * (1.0 + sgn * spec.take_profit_pct)
    peak = ep

    hourly_ret: dict[int, float] = {}
    entry_hr = (int(ts_1m[i0]) // 3600) * 3600
    hourly_ret[entry_hr] = -entry_fee

    prev_mark = ep
    exit_ts = int(ts_1m[i_end]) + 60
    final_px = float(close_1m[i_end])
    total_fee = entry_fee
    total_fnd = 0.0
    fr_lookup = fr_map_sym or {}

    for m in range(i0, i_end + 1):
        t_m = int(ts_1m[m])
        hr_bucket = (t_m // 3600) * 3600
        om, hm, lm, cm = float(open_1m[m]), float(high_1m[m]), float(low_1m[m]), float(close_1m[m])

        # 1. Pessimistic Stop-Loss priority (always checked before Take-Profit on the same 1m bar!)
        if (sgn > 0 and lm <= cur_sl) or (sgn < 0 and hm >= cur_sl):
            final_px = min(om, cur_sl) if sgn > 0 else max(om, cur_sl)
            exit_fee = tk_c * 1.15
            total_fee += exit_fee
            exit_ts = t_m + 60
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + sgn * (final_px - prev_mark) / ep - exit_fee
            prev_mark = final_px
            break

        # 2. Passive Maker Take-Profit
        if (sgn > 0 and hm >= tp_px) or (sgn < 0 and lm <= tp_px):
            final_px = max(om, tp_px) if sgn > 0 else min(om, tp_px)
            exit_fee = mk_c
            total_fee += exit_fee
            exit_ts = t_m + 60
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + sgn * (final_px - prev_mark) / ep - exit_fee
            prev_mark = final_px
            break

        # 3. End-of-minute causal Profit-Lock & Trailing ratchet for subsequent bars `m + 1..`
        if sgn > 0:
            if hm > peak:
                peak = hm
            pr = (peak - ep) / ep
            if spec.trail_trigger_pct > 0 and pr >= spec.trail_trigger_pct:
                cand_sl = peak * (1.0 - spec.trail_distance_pct)
                if cm > cand_sl:
                    cur_sl = max(cur_sl, cand_sl)
            elif spec.lock_trigger_pct > 0 and pr >= spec.lock_trigger_pct:
                cand_sl = ep * (1.0 + spec.lock_level_pct)
                if cm > cand_sl:
                    cur_sl = max(cur_sl, cand_sl)
        else:
            if lm < peak:
                peak = lm
            pr = (ep - peak) / ep
            if spec.trail_trigger_pct > 0 and pr >= spec.trail_trigger_pct:
                cand_sl = peak * (1.0 + spec.trail_distance_pct)
                if cm < cand_sl:
                    cur_sl = min(cur_sl, cand_sl)
            elif spec.lock_trigger_pct > 0 and pr >= spec.lock_trigger_pct:
                cand_sl = ep * (1.0 - spec.lock_level_pct)
                if cm < cand_sl:
                    cur_sl = min(cur_sl, cand_sl)

        # 4. Pre-settlement guard (e.g., minute >= 56 closes BEFORE `:00:00` settlement)
        min_of_hr = (t_m % 3600) // 60
        if pre_settlement_guard_minute is not None and (m - i0) >= 4 and min_of_hr >= pre_settlement_guard_minute:
            exit_fee = tk_c
            total_fee += exit_fee
            final_px = cm
            exit_ts = t_m + 60
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + sgn * (cm - prev_mark) / ep - exit_fee
            prev_mark = cm
            break

        # 5. Exact realized funding settlement at `:00:00` (`t_m + 60`) across 1h/2h/4h/8h schedules
        if (t_m + 60) % 3600 == 0 and (t_m + 60) in fr_lookup:
            fr_v = float(fr_lookup[t_m + 60])
            f_cash = -sgn * fr_v
            total_fnd += f_cash
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + f_cash

        # 6. Mark-to-market at end of hold or hourly boundary
        if m == i_end:
            exit_fee = tk_c
            total_fee += exit_fee
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + sgn * (cm - prev_mark) / ep - exit_fee
            final_px = cm
            prev_mark = cm
        elif ((t_m + 60) // 3600) * 3600 > hr_bucket:
            hourly_ret[hr_bucket] = hourly_ret.get(hr_bucket, 0.0) + sgn * (cm - prev_mark) / ep
            prev_mark = cm

    gross_linear = sgn * (final_px - ep) / ep
    tot_net = sum(hourly_ret.values())
    diag = {
        "gross_bps": gross_linear * 1e4,
        "funding_bps": total_fnd * 1e4,
        "fee_bps": total_fee * 1e4,
        "net_bps": tot_net * 1e4,
    }
    return exit_ts, hourly_ret, (tot_net < 0.0), diag


def run_unified_1m_portfolio_simulation(
    mats: dict[str, Any],
    o_1m_core: np.ndarray,
    h_1m_core: np.ndarray,
    l_1m_core: np.ndarray,
    eval_idx: pd.DatetimeIndex,
    open_core: np.ndarray,
    universe_symbols: list[str],
    p2_w_mat: np.ndarray | None,
    p1_risk_scale: float,
    resolved_sniper_events_by_hour: dict[int, list[dict[str, Any]]] | None = None,
    sniper_slot_lev: float = 0.24,
    sniper_max_pos: int = 3,
    allowed_sniper_strats: set[str] | None = None,
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
    Executes the Unified 1-Minute One-Way Exchange Simulator across `Core P1 + Manga P2 + Snipers`
    using 100% linear returns and exact `SteppedLeverageController` (`SafeRamp + Cushion-Safe CPPI`).
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
        base_deadband_ratio=(0.04 / 2.2) * 1.25,
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
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

        if hrs[t] % 8 == 0:
            raw_u = w1x_sh[t].copy()
            for i in range(n_c):
                if (
                    core_cols[i] in sniper_open_syms
                    or (raw_u[i] > 0 and t < cooldown_until[i, 0])
                    or (raw_u[i] < 0 and t < cooldown_until[i, 1])
                ):
                    raw_u[i] = 0.0
            pl = float(np.maximum(raw_u, 0.0).sum())
            nl = float(np.maximum(-raw_u, 0.0).sum())
            og = float(np.abs(w1x_sh[t]).sum())
            if pl > 1e-9 and nl > 1e-9 and og > 1e-9:
                raw_u = np.where(
                    raw_u > 0, raw_u * ((og / 2.0) / pl), np.where(raw_u < 0, raw_u * ((og / 2.0) / nl), 0.0)
                )
            else:
                raw_u[:] = 0.0

            u_dict = {core_cols[i]: float(raw_u[i]) for i in range(n_c)}
            if p2_w_mat is not None and np.any(np.abs(prev_p2_w) > 1e-8):
                inv_sc = 1.0 / max(p1_risk_scale, 0.25)
                p1_est = (curr_w - prev_p2_w) * inv_sc
                prev_dict = (
                    {core_cols[i]: float(p1_est[i]) for i in range(n_c)} if np.any(np.abs(p1_est) > 1e-6) else None
                )
            else:
                prev_dict = (
                    {core_cols[i]: float(curr_w[i]) for i in range(n_c)} if np.any(np.abs(curr_w) > 1e-8) else None
                )

            sc_d, snap_t = ctrl.scale_and_deadband_weights(u_dict, prev_dict, equity, float(vol_sh[t]))
            p1_des = np.array(
                [0.0 if core_cols[i] in sniper_open_syms else sc_d.get(core_cols[i], 0.0) for i in range(n_c)],
                dtype=np.float64,
            )

            if p2_w_mat is not None:
                p1_scaled = p1_des * p1_risk_scale
                ramp_m = min(1.65, max(0.50, snap_t.active_base_leverage / max(start_leverage, 1.0)))
                p2_sc = ramp_m * snap_t.cushion_scalar
                raw_p2 = np.array(
                    [0.0 if core_cols[i] in sniper_open_syms else p2_w_mat[t, i] * p2_sc for i in range(n_c)]
                )
                eff_p2 = np.where(
                    (np.abs(raw_p2) >= 1e-6)
                    & ((raw_p2 * prev_p2_w) > 0.0)
                    & (np.abs(raw_p2 - prev_p2_w) < 0.030 * p2_sc),
                    prev_p2_w,
                    raw_p2,
                )
                eff_p2 = np.array([0.0 if core_cols[i] in sniper_open_syms else eff_p2[i] for i in range(n_c)])
                desired_w = p1_scaled + eff_p2
                if float(np.sum(np.abs(desired_w)) * equity) >= 2000.0:
                    stub_m = (np.abs(desired_w) > 0.0) & (np.abs(desired_w * equity) < 155.0) & (np.abs(eff_p2) > 0.0)
                    eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                    desired_w = np.where(stub_m, 0.0, desired_w)
                prev_p2_w = eff_p2.copy()
                active_p2_w = eff_p2.copy()
            else:
                desired_w = p1_des

            for i in range(n_c):
                if core_cols[i] in sniper_open_syms:
                    continue
                dw = desired_w[i] - curr_w[i]
                if abs(dw) < 1e-6:
                    continue
                bar_fee += abs(dw) * rebal_fee
                if abs(desired_w[i]) < 1e-8:
                    entry_px[i], lock_armed[i] = 0.0, False
                elif abs(curr_w[i]) < 1e-8 or (curr_w[i] * desired_w[i] < 0):
                    entry_px[i], lock_armed[i] = cur_o[i], False
                elif abs(desired_w[i]) > abs(curr_w[i]) + 1e-6:
                    add_w = abs(desired_w[i]) - abs(curr_w[i])
                    entry_px[i] = (abs(curr_w[i]) * entry_px[i] + add_w * cur_o[i]) / abs(desired_w[i])

            bar_pnl += float(np.sum(curr_w * gap_ret))
            curr_w = desired_w
        else:
            bar_pnl += float(np.sum(curr_w * gap_ret))

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
            exited, exit_px, exit_fee = False, cur_c[i], 0.00020
            for m in range(60):
                om, hm, lm = ov[m], hv[m], lv[m]
                if not np.isfinite(om):
                    continue
                if lock_armed[i] and ((sgn > 0 and lm <= lk_px) or (sgn < 0 and hm >= lk_px)):
                    exit_px = min(om, lk_px) if sgn > 0 else max(om, lk_px)
                    exit_fee, exited = 0.00055, True
                    break
                if (sgn > 0 and hm >= tp_px) or (sgn < 0 and lm <= tp_px):
                    exit_px = max(om, tp_px) if sgn > 0 else min(om, tp_px)
                    exit_fee, exited = 0.00020, True
                    break
                if not lock_armed[i] and ((sgn > 0 and hm >= trig_px) or (sgn < 0 and lm <= trig_px)):
                    lock_armed[i] = True
            if exited:
                bar_pnl += w_i * ((exit_px - cur_o[i]) / cur_o[i])
                bar_fee += abs(w_i) * exit_fee
                if sgn * (exit_px - ep) < 0:
                    cooldown_until[i, 0 if sgn > 0 else 1] = t + 5
                curr_w[i], entry_px[i], lock_armed[i], active_p2_w[i] = 0.0, 0.0, False, 0.0
            else:
                bar_pnl += w_i * ((cur_c[i] - cur_o[i]) / cur_o[i])

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
    }
