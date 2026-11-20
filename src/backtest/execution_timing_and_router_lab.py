"""
Frente A Laboratory: Sub-Cycle Execution Timing (`Pre-Settlement` vs `:00 Storm` vs `Post-Storm`),
Unconditional Stub Suppression & Router Microstructure Physics (`1-Minute` Resolution, `2023–2026`).

Grounded in the forensic audit of `$283,024` real Binance Mainnet fills (`641` orders in `live_trading.db`,
`reports/live_execution_microstructure_audit.json`), which uncovered three concrete microstructure facts:
  1. **Funding-Settlement Storm Tax (`2.22x` execution cost):** Orders routed during `00/08/16 UTC :00-:02`
     suffer `51.7%` Maker fill rate, `+3.46 bps` slippage, and **`6.88 bps` all-in cost**, whereas orders
     routed outside `:00-:02` achieve `69.8%` Maker fill rate, `+0.48 bps` slippage, and **`3.10 bps` all-in cost**.
  2. **REST Quote-Fetch Latency GTX Rejection on Mid-Caps:** `RENDER` (`19.9%` Maker), `FET` (`40.9%` Maker),
     `ADA` (`40.0%`), and `DOGE` (`45.4%`) suffer stale-quote `GTX` rejections from synchronous `fetch_order_book`
     inside `execute_maker_limit_chase` (whereas `XRP/ETH/BTC` achieve `77%–84%` Maker at `~2.51 bps`).
  3. **Low-Gross Stub Leak (`ETH $13.91` live stub):** Stub suppression previously gated on `core_gross >= $2,000`,
     leaking micro-stubs during low-conviction/CPPI de-grossed windows.

This laboratory tests both **equal-fee (`4.20 bps`)** (isolating pure `1m` price + funding coupon timing) and
**empirical Mainnet execution costs (`6.88 bps` in `:00-:02` vs `3.10 bps` off-storm vs `2.55 bps` WS-quote router)**
with strict `0.000s` causality across `2023–2026` (`15`-split CPCV + `$2,700` `2026` Cushion Stress-Test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import compute_cpcv_portfolio_sharpe_stability
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


@dataclass(frozen=True)
class ExecutionTimingRouterSpec:
    exp_id: str
    description: str
    offset_minutes: (
        int  # e.g. 0 (:00:00), +3 (:03:00), +5 (:05:00), +10 (:10:00), -10 (prev hr :50:00), -5 (prev hr :55:00)
    )
    rebal_fee_bps: float  # One-way all-in rebalance execution cost (slippage + fee) in bps
    unconditional_stub_guard: bool = False  # Enforce >= $150 stub filter even when gross < $2,000
    idealized_same_bar_coupon: bool = False  # Control flag matching legacy 1h sim (credits :00 coupon to new weights)


EXECUTION_TIMING_SPECS: tuple[ExecutionTimingRouterSpec, ...] = (
    # --- Part 1: Equal-Fee (`4.20 bps`) Pure Timing & Coupon Physics Isolation ---
    ExecutionTimingRouterSpec(
        exp_id="A00_LEGACY_IDEALIZED_COUPON_4.20BPS",
        description="Reference Control: Legacy Sim Idealized :00 Coupon on New Weights (4.20 bps)",
        offset_minutes=0,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=False,
        idealized_same_bar_coupon=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A00B_REAL_POST_SETTLE_00M_4.20BPS",
        description="True Live :00:05 Physics (Old Weights Pay/Earn :00 Coupon Before Rebalance, 4.20 bps)",
        offset_minutes=0,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=False,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A01_PRE_SETTLE_MINUS_10M_FLAT_4.20BPS",
        description="Pre-Settlement Rebalance at T-10m (23:50/07:50/15:50 UTC, Causal t-2 Signal, Collects Coupon, 4.20 bps)",
        offset_minutes=-10,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A02_PRE_SETTLE_MINUS_05M_FLAT_4.20BPS",
        description="Pre-Settlement Rebalance at T-5m (23:55/07:55/15:55 UTC, Causal t-2 Signal, Collects Coupon, 4.20 bps)",
        offset_minutes=-5,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A03_POST_STORM_PLUS_03M_FLAT_4.20BPS",
        description="Post-Storm Rebalance at T+3m (00:03/08:03/16:03 UTC, Full t-1 Bar Signal, 4.20 bps)",
        offset_minutes=3,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A04_POST_STORM_PLUS_05M_FLAT_4.20BPS",
        description="Post-Storm Rebalance at T+5m (00:05/08:05/16:05 UTC, Full t-1 Bar Signal, 4.20 bps)",
        offset_minutes=5,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A05_POST_STORM_PLUS_10M_FLAT_4.20BPS",
        description="Post-Storm Rebalance at T+10m (00:10/08:10/16:10 UTC, Full t-1 Bar Signal, 4.20 bps)",
        offset_minutes=10,
        rebal_fee_bps=4.20,
        unconditional_stub_guard=True,
    ),
    # --- Part 2: Empirical Mainnet Execution Costs Measured from `$283k` Live Tokyo Fills ---
    ExecutionTimingRouterSpec(
        exp_id="A06_EMPIRICAL_LIVE_AT_00M_6.88BPS",
        description="Empirical Live Mainnet at :00:00 Storm (`6.88 bps` measured in-settlement cost)",
        offset_minutes=0,
        rebal_fee_bps=6.88,
        unconditional_stub_guard=False,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A07_EMPIRICAL_POST_STORM_PLUS_03M_3.10BPS",
        description="Empirical Live Mainnet at T+3m (`3.10 bps` measured off-settlement cost + Stub Guard)",
        offset_minutes=3,
        rebal_fee_bps=3.10,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A08_EMPIRICAL_PRE_SETTLE_MINUS_05M_3.10BPS",
        description="Empirical Live Mainnet at T-5m (`3.10 bps` measured off-settlement cost + Coupon + Stub Guard)",
        offset_minutes=-5,
        rebal_fee_bps=3.10,
        unconditional_stub_guard=True,
    ),
    ExecutionTimingRouterSpec(
        exp_id="A09_WS_QUOTE_OBI_ROUTER_PLUS_03M_2.55BPS",
        description="T+3m Post-Storm + Zero-Lag WS Top-of-Book GTX Router (`2.55 bps` ~80% Maker + Stub Guard)",
        offset_minutes=3,
        rebal_fee_bps=2.55,
        unconditional_stub_guard=True,
    ),
)


def simulate_subminute_timed_1m_portfolio(
    mats: dict[str, Any],
    o_1m_core: np.ndarray,
    h_1m_core: np.ndarray,
    l_1m_core: np.ndarray,
    eval_idx: pd.DatetimeIndex,
    open_core: np.ndarray,
    p2_w_mat: np.ndarray | None,
    spec: ExecutionTimingRouterSpec,
    resolved_sniper_events_by_hour: dict[int, list[dict[str, Any]]] | None = None,
    allowed_sniper_strats: set[str] | None = None,
    p1_risk_scale: float = 0.85,
    sniper_slot_lev: float = 0.24,
    sniper_max_pos: int = 3,
    initial_capital: float = 2700.0,
    start_leverage: float = 2.5,
    max_leverage: float = 5.0,
    recovery_boost_eq: float | None = 3050.0,
    recovery_boost_lev: float | None = 3.2,
    cushion_floor_usd: float | None = 2500.0,
    cushion_buffer_usd: float = 240.0,
) -> dict[str, Any]:
    """
    Exact 1-minute sub-bar timed portfolio simulator:
      - `offset_minutes == 0`: Rebalances at minute `0` of `00/08/16 UTC` AFTER `:00:00` funding settles on old weights.
      - `offset_minutes > 0` (`+3, +5, +10`): Old weights settle `:00:00` funding AND hold minutes `0..m_reb-1`;
        at minute `m_reb` of `00/08/16 UTC`, rebalances at `o_1m[t, m_reb]` using causal `t-1` closed bar signal,
        then new weights hold minutes `m_reb..59` (with full `1m` intrabar TP/Profit-Lock resolution!).
      - `offset_minutes < 0` (`-5, -10`): Rebalances at minute `m_reb = 60 + offset_minutes` (`55` or `50`) of
        `23/07/15 UTC` using STRICTLY CAUSAL `t-1` signal (i.e. bar closed at `23:00/07:00/15:00`, ZERO lookahead
        into the `23:00-00:00` bar close!), executes at `o_1m[t, m_reb]`, holds minutes `m_reb..59` into `:00:00`,
        and the NEW weights collect/pay the `:00:00` funding settlement!
    """
    core_cols = list(mats["close"].columns)
    n_b, n_c = len(eval_idx), len(core_cols)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()

    c_arr = mats["close"].loc[eval_idx].to_numpy(dtype=np.float64)
    f_arr = mats["funding"].loc[eval_idx].to_numpy(dtype=np.float64)
    w_src = mats["w_masked"] if "w_masked" in mats else mats["w_base_1x"]
    # `w1x_sh[t]` is the signal computed at the CLOSE of bar `t - 1` (causal at `t:00:00` and later!).
    # Therefore at minute `50` of bar `t` (`23:50 UTC`), `w1x_sh[t]` IS strictly causal (computed at `23:00:00 UTC`!)!
    w1x_sh = w_src.shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    vol_sh = mats["vol_scalar"].shift(1).fillna(1.0).loc[eval_idx].to_numpy(dtype=np.float64)
    chop_sh = mats["alpha_chop_overlay"].shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    v24_sh = mats["vol_24h"].shift(1).ffill().fillna(0.02).loc[eval_idx].to_numpy(dtype=np.float64)
    max_tr_sh = (
        mats["t_df"].shift(1).fillna(0.0).loc[eval_idx][["macro", "med", "fast"]].max(axis=1).to_numpy(dtype=np.float64)
    )
    hrs = eval_idx.hour.to_numpy()
    rebal_fee = float(spec.rebal_fee_bps) / 10_000.0
    off_m = int(spec.offset_minutes)

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
    cum_core_fee_pct = 0.0
    cum_funding_net_pct = 0.0
    n_stubs_suppressed = 0

    def _exec_rebalance(t_idx: int, exec_prices: np.ndarray, sniper_syms_now: set[str]) -> float:
        nonlocal curr_w, prev_p2_w, active_p2_w, n_stubs_suppressed
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
            gross_des_usd = float(np.sum(np.abs(desired_w)) * equity)
            if gross_des_usd >= 2000.0:
                stub_m = (np.abs(desired_w) > 0.0) & (np.abs(desired_w * equity) < 155.0) & (np.abs(eff_p2) > 0.0)
                n_stubs_suppressed += int(np.sum(stub_m))
                eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                desired_w = np.where(stub_m, 0.0, desired_w)
            elif spec.unconditional_stub_guard and gross_des_usd > 0.0:
                min_dust_usd = max(25.0, 0.05 * gross_des_usd)
                stub_m = (
                    (np.abs(desired_w) > 0.0) & (np.abs(desired_w * equity) < min_dust_usd) & (np.abs(eff_p2) > 0.0)
                )
                n_stubs_suppressed += int(np.sum(stub_m))
                eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                desired_w = np.where(stub_m, 0.0, desired_w)
            prev_p2_w = eff_p2.copy()
            active_p2_w = eff_p2.copy()
        else:
            desired_w = p1_des

        fee_this = 0.0
        for i in range(n_c):
            if core_cols[i] in sniper_syms_now:
                continue
            dw = desired_w[i] - curr_w[i]
            if abs(dw) < (0.009 if (curr_w[i] * desired_w[i] > 0) else 1e-6):
                desired_w[i] = curr_w[i]
                continue
            fee_this += abs(dw) * rebal_fee
            px_i = float(exec_prices[i])
            if abs(desired_w[i]) < 1e-8:
                entry_px[i], lock_armed[i] = 0.0, False
            elif abs(curr_w[i]) < 1e-8 or (curr_w[i] * desired_w[i] < 0):
                entry_px[i], lock_armed[i] = px_i, False
            elif abs(desired_w[i]) > abs(curr_w[i]) + 1e-6:
                add_w = abs(desired_w[i]) - abs(curr_w[i])
                entry_px[i] = (abs(curr_w[i]) * entry_px[i] + add_w * px_i) / abs(desired_w[i])

        curr_w = desired_w
        return fee_this

    def _resolve_minute_slice(
        t_idx: int,
        m_start: int,
        m_end: int,
        start_px_vec: np.ndarray,
        end_px_vec: np.ndarray,
    ) -> tuple[float, float]:
        pnl_seg, fee_seg = 0.0, 0.0
        chop_w, max_tr = float(chop_sh[t_idx]), float(max_tr_sh[t_idx])
        for i in range(n_c):
            w_i = curr_w[i]
            s_px = float(start_px_vec[i])
            e_px = float(end_px_vec[i])
            if abs(w_i) < 1e-8 or entry_px[i] <= 0 or not np.isfinite(s_px) or s_px <= 0 or not np.isfinite(e_px):
                continue
            sig1h = float(v24_sh[t_idx, i])
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

            hv = h_1m_core[t_idx, m_start:m_end, i]
            lv = l_1m_core[t_idx, m_start:m_end, i]
            ov = o_1m_core[t_idx, m_start:m_end, i]
            if hv.size == 0 or not np.any(np.isfinite(hv)):
                pnl_seg += w_i * ((e_px - s_px) / s_px)
                continue
            max_h, min_l = float(np.nanmax(hv)), float(np.nanmin(lv))
            can_hit = (sgn > 0 and (max_h >= min(tp_px, trig_px) or (lock_armed[i] and min_l <= lk_px))) or (
                sgn < 0 and (min_l <= max(tp_px, trig_px) or (lock_armed[i] and max_h >= lk_px))
            )
            if not can_hit:
                pnl_seg += w_i * ((e_px - s_px) / s_px)
                continue

            exited, exit_px, exit_fee = False, e_px, 0.00020
            for m_loc in range(m_end - m_start):
                om, hm, lm = float(ov[m_loc]), float(hv[m_loc]), float(lv[m_loc])
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
                pnl_seg += w_i * ((exit_px - s_px) / s_px)
                fee_seg += abs(w_i) * exit_fee
                if sgn * (exit_px - ep) < 0:
                    cooldown_until[i, 0 if sgn > 0 else 1] = t_idx + 5
                curr_w[i], entry_px[i], lock_armed[i], active_p2_w[i] = 0.0, 0.0, False, 0.0
            else:
                pnl_seg += w_i * ((e_px - s_px) / s_px)
        return pnl_seg, fee_seg

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
                    {"sym": sym_name, "exit_ts": ex_ts, "weight": eff_slot_w, "hourly_ret": ev["hourly_ret"]}
                )
                sniper_open_syms.add(sym_name)
                sniper_cd_until[sym_name] = ex_ts + (14400 if bool(ev["was_loss"]) else 3600)
                sniper_exec_counts[st_name] = sniper_exec_counts.get(st_name, 0) + 1

        prev_c = c_arr[t - 1] if t > 0 else open_core[0]
        cur_o, cur_c = open_core[t], c_arr[t]
        bar_pnl, bar_fee = 0.0, 0.0

        # Overnight / inter-bar open gap (`prev_c -> cur_o` at `t:00:00`)
        gap_ret = np.where(prev_c > 0, (cur_o - prev_c) / prev_c, 0.0)
        bar_pnl += float(np.sum(curr_w * gap_ret))

        is_rebal_bar = False
        reb_min = 0
        if off_m == 0 and (hrs[t] % 8 == 0):
            is_rebal_bar, reb_min = True, 0
        elif off_m > 0 and (hrs[t] % 8 == 0):
            is_rebal_bar, reb_min = True, min(59, off_m)
        elif off_m < 0 and (hrs[t] % 8 == 7):
            is_rebal_bar, reb_min = True, max(1, 60 + off_m)

        # CRITICAL FUNDING SETTLEMENT PHYSICS AT EXACT `t:00:00` (`hrs[t] % 8 == 0`):
        # Unless `idealized_same_bar_coupon=True` (legacy reference), funding at `t:00:00.000` settles on the
        # positions ALREADY open at `t:00:00.000` BEFORE any `:00:05+` orders execute!
        fnd_cashflow_ret = 0.0
        if not spec.idealized_same_bar_coupon:
            fnd_cashflow_ret = -float(np.sum(curr_w * f_arr[t]))
            cum_funding_net_pct += fnd_cashflow_ret * 100.0

        if not is_rebal_bar:
            p_s, f_s = _resolve_minute_slice(t, 0, 60, cur_o, cur_c)
            bar_pnl += p_s
            bar_fee += f_s
        elif reb_min == 0:
            bar_fee += _exec_rebalance(t, cur_o, sniper_open_syms)
            p_s, f_s = _resolve_minute_slice(t, 0, 60, cur_o, cur_c)
            bar_pnl += p_s
            bar_fee += f_s
        else:
            # Split bar at exact minute `reb_min` (`[0 .. reb_min)` on old weights, `[reb_min .. 60)` on new weights!)
            mid_px = o_1m_core[t, reb_min, :].astype(np.float64)
            # Fallback to linear interpolation only for rare missing 1m ticks
            miss = ~np.isfinite(mid_px) | (mid_px <= 0)
            if np.any(miss):
                frac = reb_min / 60.0
                mid_px[miss] = cur_o[miss] * (1.0 - frac) + cur_c[miss] * frac

            p_1, f_1 = _resolve_minute_slice(t, 0, reb_min, cur_o, mid_px)
            bar_pnl += p_1
            bar_fee += f_1

            bar_fee += _exec_rebalance(t, mid_px, sniper_open_syms)

            p_2, f_2 = _resolve_minute_slice(t, reb_min, 60, mid_px, cur_c)
            bar_pnl += p_2
            bar_fee += f_2

        if spec.idealized_same_bar_coupon:
            fnd_cashflow_ret = -float(np.sum(curr_w * f_arr[t]))
            cum_funding_net_pct += fnd_cashflow_ret * 100.0

        cum_core_fee_pct += bar_fee * 100.0
        sniper_hr_pnl = sum(float(sp["weight"]) * float(sp["hourly_ret"].get(hr_ts, 0.0)) for sp in active_snipers)
        nr = bar_pnl - bar_fee + fnd_cashflow_ret + sniper_hr_pnl
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
        "exec_stats": {
            "core_fee_drag_pct": round(cum_core_fee_pct, 2),
            "net_funding_earned_pct": round(cum_funding_net_pct, 2),
            "stubs_suppressed": n_stubs_suppressed,
        },
    }


def evaluate_execution_timing_and_router_specs(
    bundle: dict[str, Any],
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[ExecutionTimingRouterSpec, ...] = EXECUTION_TIMING_SPECS,
) -> pd.DataFrame:
    """Runs all `ExecutionTimingRouterSpec` experiments across 2023–2026 + Fresh `$2,700` `2026` Cushion."""
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    mats: dict[str, Any] = bundle["mats"]
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]
    p2_w_mat: np.ndarray = bundle["p2_w_mat"]

    events_by_hr: dict[int, list[dict[str, Any]]] = {}
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

    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}

    rows: list[dict[str, Any]] = []
    for sp in specs:
        sim_4y = simulate_subminute_timed_1m_portfolio(
            mats=mats,
            o_1m_core=o_1m_core,
            h_1m_core=h_1m_core,
            l_1m_core=l_1m_core,
            eval_idx=eval_idx,
            open_core=open_core,
            p2_w_mat=p2_w_mat,
            spec=sp,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])
        es = sim_4y["exec_stats"]

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            sim_26 = simulate_subminute_timed_1m_portfolio(
                mats=mats,
                o_1m_core=o_1m_core[mask_2026],
                h_1m_core=h_1m_core[mask_2026],
                l_1m_core=l_1m_core[mask_2026],
                eval_idx=idx_2026,
                open_core=open_core[mask_2026],
                p2_w_mat=p2_w_mat[mask_2026],
                spec=sp,
                resolved_sniper_events_by_hour=evs_2026,
                allowed_sniper_strats=allowed_strats,
            )
            fresh_final = round(float(sim_26["final_eq"]), 2)
            fresh_min = round(float(sim_26["min_eq"]), 2)
            fresh_ok = bool(fresh_min >= 2500.0)

        rows.append(
            {
                "exp_id": sp.exp_id,
                "description": sp.description,
                "offset_minutes": sp.offset_minutes,
                "rebal_fee_bps": sp.rebal_fee_bps,
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
                "fresh_2026_floor_2500_ok": fresh_ok,
                "core_fee_drag_pct": es["core_fee_drag_pct"],
                "net_funding_earned_pct": es["net_funding_earned_pct"],
                "stubs_suppressed": es["stubs_suppressed"],
            }
        )
    return pd.DataFrame(rows)
