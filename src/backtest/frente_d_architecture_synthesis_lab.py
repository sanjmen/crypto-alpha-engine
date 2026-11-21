"""
Frente D Laboratory: Core Intrabar Exit Physics (`Disposition Effect`), Causal `Beta_72h`
Neutralization, Conviction-Tiered `Kelly` Sniper Sizing & One-Way Preemption, `M04` Squeeze
Threshold Forensics, and the Cumulative `M17` Synthesis Ladder (`1-Minute` Physics, `2023–2026`).

Tests four structural blind spots uncovered in `Core P1 + Manga P2 + Snipers` that were never
evaluated in Docs `01..20`:
  1. **D1 — `Core P1` Intrabar `PROFIT_LOCK / TP` Disposition Effect Post-`8h` Clock:**
     When `enable_intracycle_rebalance=false`, closing a single winning leg mid-cycle via `PROFIT_LOCK`
     (`+2.4%->+1.2%` / `+3.5%->+1.5%`, `5.5 bps` Taker fee) caps right-tail winners and leaves `Core P1`
     directionally unhedged for up to `7 hours` while losing legs ride uncapped.
  2. **D2 — Hidden `BTC` Beta Leak in Dollar-Neutral `Core P1`:**
     `Core P1` enforces `sum(longs) == -sum(shorts)` in dollars, ignoring causal `beta_72` (`Long SUI/FET/RENDER`
     `beta ~ 1.65` vs `Short BTC/ETH/BNB` `beta ~ 0.90` -> `+0.75x` hidden net `BTC` exposure).
  3. **D3 — Flat `0.24x` Sniper Sizing & One-Way Core Blocking of `M26 / M09_LONG`:**
     Every Sniper currently receives identical `0.24x` weight (`M26` `+236.6 bps` in `2026` == `M04` `-43.0 bps`!),
     `M23` can consume all `3/3` slots in a single hour, and open `Core P1` slices silently block `M26` & `M09_LONG`.
  4. **D4 — Forensic Repair of `M04_NegFundSqueeze` (`-43.0 bps` in `2026`):**
     Decomposes `M04` across causal entry funding buckets (`[-8,-4) bps` harvester noise vs `<= -8 bps` and
     `<= -12 bps` genuine trapped-short pain).
  5. **D5 — Cumulative `M17` Synthesis Stack (`Frentes A + C + D` under Empirical Mainnet Costs):**
     Verifies that the combined upgrades compound additively without interfering with the `$2,500` CPPI floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    compute_cpcv_portfolio_sharpe_stability,
)
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
class FrenteDSynthesisSpec:
    exp_id: str
    description: str
    offset_minutes: int = 0
    rebal_fee_bps: float = 4.20
    idealized_same_bar_coupon: bool = True
    safe_stub_guard: bool = False
    core_exit_mode: str = "default"  # "default", "pure_8h_hold", "tp_only_no_lock", "wide_convex"
    beta_clamp_max: float | None = None  # None, 0.12 (partial beta clamp), 0.0 (strict zero-beta projection)
    sniper_sizing_mode: str = "flat"  # "flat", "kelly_tiered"
    sniper_preempt_core: bool = False  # Let Tier-A Snipers (M26, M09_LONG) cleanly preempt conflicting Core P1 legs
    m04_max_fr: float = -0.0004  # Default -4 bps vs -8 bps (-0.0008) vs -12 bps (-0.0012)
    use_inv_vol_p2: bool = False  # Frente C C02 causal 168h inverse-vol sizing on BTC/ETH/SOL


FRENTE_D_SPECS: tuple[FrenteDSynthesisSpec, ...] = (
    # --- Controls ---
    FrenteDSynthesisSpec(
        exp_id="D00_LIVE_REAL_00M_6.88BPS",
        description="Empirical Live Mainnet at :00:00 Storm (`6.88 bps`, Current Prod Stack)",
        offset_minutes=0,
        rebal_fee_bps=6.88,
        idealized_same_bar_coupon=False,
    ),
    FrenteDSynthesisSpec(
        exp_id="D01_BASELINE_IDEALIZED_4.20BPS",
        description="Reference Baseline (`4.20 bps`, Tight Core Lock+TP, Dollar-Neutral, Flat 0.24x Snipers)",
        offset_minutes=0,
        rebal_fee_bps=4.20,
        idealized_same_bar_coupon=True,
    ),
    # --- Angle D1: Core P1 Intrabar Exit Physics (Disposition Effect & Mid-Cycle Unhedging) ---
    FrenteDSynthesisSpec(
        exp_id="D02_CORE_PURE_8H_HOLD_NO_LOCK_NO_TP",
        description="D1a: Pure 8h Cross-Sectional Hold (`Disable Core Intrabar Lock & TP`, 0 Mid-Cycle Drift)",
        core_exit_mode="pure_8h_hold",
    ),
    FrenteDSynthesisSpec(
        exp_id="D03_CORE_TP_ONLY_NO_PROFIT_LOCK",
        description="D1b: Maker TP Only (`+5.0% / +3.6%`), Disable Taker Profit-Lock Choker (`5.5 bps` saved)",
        core_exit_mode="tp_only_no_lock",
    ),
    FrenteDSynthesisSpec(
        exp_id="D04_CORE_WIDE_CONVEX_BRACKET",
        description="D1c: Wide Convex Core Bracket (`Lock +4.8% -> +2.4%`, `TP +7.5%`)",
        core_exit_mode="wide_convex",
    ),
    # --- Angle D2: Causal 72h BTC Beta Neutralization on Core P1 ---
    FrenteDSynthesisSpec(
        exp_id="D05_CORE_BTC_BETA_OVERLAY_CLAMP_012",
        description="D2a: Pod-Style `BTCUSDT` Beta-Overlay Hedge Clamping `|Net Beta| <= 0.12 * Gross_P1`",
        beta_clamp_max=0.12,
    ),
    FrenteDSynthesisSpec(
        exp_id="D06_CORE_ASYM_LEG_ZERO_BETA_SCALING",
        description="D2b: Asymmetric Leg Scaling (`G_L * beta_L == G_S * beta_S`, Preserves Intra-Leg Alpha)",
        beta_clamp_max=0.0,
    ),
    # --- Angle D3: Conviction-Tiered Kelly Sniper Allocation & One-Way Core Preemption ---
    FrenteDSynthesisSpec(
        exp_id="D07_SNIPER_KELLY_TIERED_SIZING",
        description="D3a: Kelly Tiered Slots (`M26 0.34x, M09_L 0.30x, M23/M09_S 0.20x, M04 0.14x`, Max 2 M23)",
        sniper_sizing_mode="kelly_tiered",
    ),
    FrenteDSynthesisSpec(
        exp_id="D08_SNIPER_KELLY_PLUS_CORE_PREEMPT",
        description="D3b: Kelly Tiered + `M26 / M09_LONG` Preempt Conflicting `Core P1` Single-Name Legs",
        sniper_sizing_mode="kelly_tiered",
        sniper_preempt_core=True,
    ),
    # --- Angle D4: Forensic M04 Negative-Funding Squeeze Threshold Repair ---
    FrenteDSynthesisSpec(
        exp_id="D09_M04_DEEP_SQUEEZE_8BPS",
        description="D4a: Require Deep Negative Funding (`FR <= -8 bps`) on `M04` (Filters Harvester Noise)",
        m04_max_fr=-0.0008,
    ),
    FrenteDSynthesisSpec(
        exp_id="D09B_M04_EXTREME_SQUEEZE_12BPS",
        description="D4b: Require Extreme Negative Funding (`FR <= -12 bps`) on `M04`",
        m04_max_fr=-0.0012,
    ),
    # --- Angle D5: Cumulative M17 Grand Synthesis Stack under Empirical Mainnet Execution Physics ---
    FrenteDSynthesisSpec(
        exp_id="D10_M17_STACK_FRENTE_A_PLUS_C",
        description="M17 Stack (A+C): `T-5m` Clock + `WS BBO Router (2.55 bps)` + Safe Stub Guard + `Inv-Vol P2`",
        offset_minutes=-5,
        rebal_fee_bps=2.55,
        idealized_same_bar_coupon=False,
        safe_stub_guard=True,
        use_inv_vol_p2=True,
    ),
    FrenteDSynthesisSpec(
        exp_id="D11_M17_STACK_A_C_PLUS_KELLY_M04",
        description="M17 Stack (A+C+D3+D4): `D10` + Kelly Tiered Snipers + Core Preemption + `M04 <= -8 bps`",
        offset_minutes=-5,
        rebal_fee_bps=2.55,
        idealized_same_bar_coupon=False,
        safe_stub_guard=True,
        use_inv_vol_p2=True,
        sniper_sizing_mode="kelly_tiered",
        sniper_preempt_core=True,
        m04_max_fr=-0.0008,
    ),
    FrenteDSynthesisSpec(
        exp_id="D12_M17_FULL_SYNTHESIS_ALL_ANGLES",
        description="M17 Full Synthesis: `D11` + Causal `Beta_72` Clamp (`0.12`) + `Core TP-Only (No Lock)`",
        offset_minutes=-5,
        rebal_fee_bps=2.55,
        idealized_same_bar_coupon=False,
        safe_stub_guard=True,
        use_inv_vol_p2=True,
        core_exit_mode="tp_only_no_lock",
        beta_clamp_max=0.12,
        sniper_sizing_mode="kelly_tiered",
        sniper_preempt_core=True,
        m04_max_fr=-0.0008,
    ),
)


def apply_causal_beta_neutralization(
    w_vec: np.ndarray,
    beta_vec: np.ndarray,
    btc_idx: int | None = 0,
    max_beta_ratio: float = 0.0,
) -> np.ndarray:
    """
    Neutralizes hidden `BTC` beta in `Core P1` without distorting intra-leg cross-sectional alpha:
      - When `max_beta_ratio > 0` (`D05` Pod-Style `BTCUSDT` Overlay Clamp): keeps all single-name
        alpha weights intact and adds an offsetting `BTCUSDT` (`beta == 1.0`) hedge for any excess
        `|net_beta| > max_beta_ratio * gross`, rescaling total gross back to `gross`.
      - When `max_beta_ratio == 0.0` (`D06` Exact Asymmetric Leg Beta-Neutrality): scales total Long
        leg `G_L = G * beta_S / (beta_L + beta_S)` and Short leg `G_S = G * beta_L / (beta_L + beta_S)`
        so `sum(w_out * beta_vec) == 0.0` to machine precision while preserving `100%` intra-leg weights!
    """
    pos = np.maximum(w_vec, 0.0)
    neg = np.maximum(-w_vec, 0.0)
    p_sum = float(pos.sum())
    n_sum = float(neg.sum())
    gross = p_sum + n_sum
    if p_sum <= 1e-9 or n_sum <= 1e-9 or gross <= 1e-9:
        return w_vec

    b_clean = np.where(np.isfinite(beta_vec), np.clip(beta_vec, 0.15, 2.80), 1.0)
    if btc_idx is not None and 0 <= btc_idx < len(b_clean):
        b_clean[btc_idx] = 1.0

    if max_beta_ratio <= 1e-9:
        beta_l = float(np.sum(pos * b_clean)) / p_sum
        beta_s = float(np.sum(neg * b_clean)) / n_sum
        if (beta_l + beta_s) <= 1e-8:
            return w_vec
        g_l = gross * (beta_s / (beta_l + beta_s))
        g_s = gross * (beta_l / (beta_l + beta_s))
        return pos * (g_l / p_sum) - neg * (g_s / n_sum)

    net_b = float(np.sum(w_vec * b_clean))
    lim_b = float(max_beta_ratio) * gross
    if abs(net_b) <= lim_b + 1e-8 or btc_idx is None or not (0 <= btc_idx < len(w_vec)):
        return w_vec

    excess_b = net_b - np.sign(net_b) * lim_b
    w_out = w_vec.copy()
    w_out[btc_idx] -= excess_b
    new_gross = float(np.sum(np.abs(w_out)))
    if new_gross > gross * 1.15 and new_gross > 1e-9:
        w_out *= (gross * 1.15) / new_gross
    return w_out


def analyze_m04_funding_buckets(
    base_sniper_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decomposes `M04_NegFundSqueeze` `1m` trades across causal entry funding rate buckets and eras."""
    m04_evs = [e for e in base_sniper_events if e.get("strat") == SNIPER_M04_STRAT]
    if not m04_evs:
        return []

    rows = []
    for e in m04_evs:
        en_ts = int(e["entry_ts"])
        yr = datetime.fromtimestamp(en_ts, tz=timezone.utc).year
        era = "2023_2024" if yr <= 2024 else ("2025" if yr == 2025 else "2026")
        net_bps = float(sum(e["hourly_ret"].values())) * 10_000.0
        fr_bps = float(e.get("entry_fr", -0.0004)) * 10_000.0
        rows.append({"era": era, "net_bps": net_bps, "fr_bps": fr_bps})
    df = pd.DataFrame(rows)

    buckets = [
        ("ALL_M04_LE_MINUS_4BPS", df["fr_bps"] <= -3.99),
        ("MILD_NOISE_MINUS_4_TO_8BPS", (df["fr_bps"] > -8.0) & (df["fr_bps"] <= -3.99)),
        ("DEEP_SQUEEZE_LE_MINUS_8BPS", df["fr_bps"] <= -8.0),
        ("EXTREME_SQUEEZE_LE_MINUS_12BPS", df["fr_bps"] <= -12.0),
    ]
    out: list[dict[str, Any]] = []
    for b_name, mask in buckets:
        sub = df[mask]
        if sub.empty:
            continue
        s24 = sub[sub["era"] == "2023_2024"]["net_bps"]
        s25 = sub[sub["era"] == "2025"]["net_bps"]
        s26 = sub[sub["era"] == "2026"]["net_bps"]
        out.append(
            {
                "bucket": b_name,
                "n_full": int(len(sub)),
                "n_2026": int(len(s26)),
                "mean_net_bps_full": round(float(sub["net_bps"].mean()), 2),
                "med_net_bps_full": round(float(sub["net_bps"].median()), 2),
                "win_rate_pct": round(100.0 * float((sub["net_bps"] > 0).mean()), 1),
                "net_bps_23_24": round(float(s24.mean()), 2) if len(s24) else 0.0,
                "net_bps_2025": round(float(s25.mean()), 2) if len(s25) else 0.0,
                "net_bps_2026": round(float(s26.mean()), 2) if len(s26) else 0.0,
            }
        )
    return out


def simulate_frente_d_1m_portfolio(
    mats: dict[str, Any],
    o_1m_core: np.ndarray,
    h_1m_core: np.ndarray,
    l_1m_core: np.ndarray,
    eval_idx: pd.DatetimeIndex,
    open_core: np.ndarray,
    p2_w_mat: np.ndarray | None,
    spec: FrenteDSynthesisSpec,
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
    """Runs the exact `1-minute` One-Way portfolio simulator with Frente D mechanisms (`0.000 ms` lookahead)."""
    core_cols = list(mats["close"].columns)
    n_b, n_c = len(eval_idx), len(core_cols)
    ts_eval = eval_idx.as_unit("s").astype(np.int64).to_numpy()
    rebal_fee = float(spec.rebal_fee_bps) * 1e-4

    c_arr = mats["close"].loc[eval_idx].to_numpy(dtype=np.float64)
    f_arr = mats["funding"].loc[eval_idx].to_numpy(dtype=np.float64)
    w_src = mats["w_masked"] if "w_masked" in mats else mats["w_base_1x"]
    w1x_sh = w_src.shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    vol_sh = mats["vol_scalar"].shift(1).fillna(1.0).loc[eval_idx].to_numpy(dtype=np.float64)
    chop_sh = mats["alpha_chop_overlay"].shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
    v24_sh = mats["vol_24h"].shift(1).ffill().fillna(0.02).loc[eval_idx].to_numpy(dtype=np.float64)
    beta_df = mats.get("beta_72")
    if beta_df is not None:
        beta_sh = beta_df.shift(1).ffill().fillna(1.0).loc[eval_idx].to_numpy(dtype=np.float64)
    else:
        beta_sh = np.ones((n_b, n_c), dtype=np.float64)
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
    cum_core_fee_pct = 0.0
    n_lock_exits = 0
    n_tp_exits = 0
    n_preemptions = 0
    abs_beta_samples: list[float] = []

    off_m = int(spec.offset_minutes)
    if off_m == 0:
        rebal_hrs = {0, 8, 16}
        rebal_min = 0
    elif off_m > 0:
        rebal_hrs = {0, 8, 16}
        rebal_min = min(59, off_m)
    else:
        rebal_hrs = {7, 15, 23}
        rebal_min = max(1, 60 + off_m)

    def _get_minute_open_vec(t_idx: int, m_idx: int) -> np.ndarray:
        if m_idx <= 0:
            return open_core[t_idx]
        v = o_1m_core[t_idx, m_idx, :].copy()
        bad = ~np.isfinite(v) | (v <= 0.0)
        if np.any(bad):
            v[bad] = open_core[t_idx, bad]
        return v

    def _execute_rebalance(t_idx: int, exec_prices: np.ndarray, sniper_syms_now: set[str]) -> float:
        nonlocal curr_w, prev_p2_w, active_p2_w
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
            raw_u = np.where(
                raw_u > 0,
                raw_u * ((og / 2.0) / pl),
                np.where(raw_u < 0, raw_u * ((og / 2.0) / nl), 0.0),
            )
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
        if spec.beta_clamp_max is not None:
            btc_j = core_cols.index("BTCUSDT") if "BTCUSDT" in core_cols else None
            p1_des = apply_causal_beta_neutralization(
                p1_des,
                beta_sh[t_idx],
                btc_idx=btc_j,
                max_beta_ratio=float(spec.beta_clamp_max),
            )
            p1_des = np.array([0.0 if core_cols[i] in sniper_syms_now else p1_des[i] for i in range(n_c)])

        g_p1_now = float(np.sum(np.abs(p1_des)))
        if g_p1_now > 1e-6:
            abs_beta_samples.append(abs(float(np.sum(p1_des * beta_sh[t_idx]))) / g_p1_now)

        if p2_w_mat is not None:
            p1_scaled = p1_des * p1_risk_scale
            ramp_m = min(1.65, max(0.50, snap_t.active_base_leverage / max(start_leverage, 1.0)))
            p2_sc = ramp_m * snap_t.cushion_scalar
            raw_p2 = np.array(
                [(0.0 if core_cols[i] in sniper_syms_now else p2_w_mat[t_idx, i] * p2_sc) for i in range(n_c)]
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
                eff_p2 = np.where(stub_m, eff_p2 - desired_w, eff_p2)
                desired_w = np.where(stub_m, 0.0, desired_w)
            elif spec.safe_stub_guard and gross_des_usd > 0.0:
                min_dust_usd = max(25.0, 0.05 * gross_des_usd)
                stub_m = (
                    (np.abs(desired_w) > 0.0) & (np.abs(desired_w * equity) < min_dust_usd) & (np.abs(eff_p2) > 0.0)
                )
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
        nonlocal n_lock_exits, n_tp_exits
        pnl_seg, fee_seg = 0.0, 0.0
        chop_w, max_tr = float(chop_sh[t_idx]), float(max_tr_sh[t_idx])
        ex_mode = spec.core_exit_mode
        for i in range(n_c):
            w_i = curr_w[i]
            s_px = float(start_px_vec[i])
            e_px = float(end_px_vec[i])
            if abs(w_i) < 1e-8 or entry_px[i] <= 0 or not np.isfinite(s_px) or s_px <= 0 or not np.isfinite(e_px):
                continue
            if ex_mode == "pure_8h_hold":
                pnl_seg += w_i * ((e_px - s_px) / s_px)
                continue

            sig1h = float(v24_sh[t_idx, i])
            p2_dom = (p2_w_mat is not None) and (abs(active_p2_w[i]) >= 0.15) and ((active_p2_w[i] * w_i) > 0)
            if ex_mode == "wide_convex":
                eff_tp, eff_ltrig, eff_llvl = (0.075, 0.048, 0.024) if not p2_dom else (0.085, 0.055, 0.028)
            elif (chop_w >= 0.25 or max_tr <= 0.0) and not p2_dom:
                sig4 = 2.0 * min(0.035, max(0.004, sig1h))
                cmult = 1.95 - 0.35 * min(1.0, max(0.0, chop_w / 0.75))
                eff_tp = min(0.036, max(0.028, sig4 * cmult))
                eff_ltrig, eff_llvl = 0.024, 0.012
            else:
                eff_tp, eff_ltrig, eff_llvl = 0.050, 0.035, 0.015

            use_lock = ex_mode != "tp_only_no_lock"
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
            min_up = min(tp_px, trig_px) if use_lock else tp_px
            max_dn = max(tp_px, trig_px) if use_lock else tp_px
            can_hit = (sgn > 0 and (max_h >= min_up or (use_lock and lock_armed[i] and min_l <= lk_px))) or (
                sgn < 0 and (min_l <= max_dn or (use_lock and lock_armed[i] and max_h >= lk_px))
            )
            if not can_hit:
                pnl_seg += w_i * ((e_px - s_px) / s_px)
                continue

            exited, exit_px, exit_fee, kind_hit = False, e_px, 0.00020, ""
            for m_loc in range(m_end - m_start):
                om, hm, lm = float(ov[m_loc]), float(hv[m_loc]), float(lv[m_loc])
                if not np.isfinite(om):
                    continue
                if use_lock and lock_armed[i] and ((sgn > 0 and lm <= lk_px) or (sgn < 0 and hm >= lk_px)):
                    exit_px = min(om, lk_px) if sgn > 0 else max(om, lk_px)
                    exit_fee, exited, kind_hit = 0.00055, True, "LOCK"
                    break
                if (sgn > 0 and hm >= tp_px) or (sgn < 0 and lm <= tp_px):
                    exit_px = max(om, tp_px) if sgn > 0 else min(om, tp_px)
                    exit_fee, exited, kind_hit = 0.00020, True, "TP"
                    break
                if use_lock and not lock_armed[i] and ((sgn > 0 and hm >= trig_px) or (sgn < 0 and lm <= trig_px)):
                    lock_armed[i] = True

            if exited:
                pnl_seg += w_i * ((exit_px - s_px) / s_px)
                fee_seg += abs(w_i) * exit_fee
                if kind_hit == "LOCK":
                    n_lock_exits += 1
                else:
                    n_tp_exits += 1
                if sgn * (exit_px - ep) < 0:
                    cooldown_until[i, 0 if sgn > 0 else 1] = t_idx + 5
                curr_w[i], entry_px[i], lock_armed[i], active_p2_w[i] = (
                    0.0,
                    0.0,
                    False,
                    0.0,
                )
            else:
                pnl_seg += w_i * ((e_px - s_px) / s_px)
        return pnl_seg, fee_seg

    for t in range(n_b):
        hr_ts = int(ts_eval[t])
        hr_utc = int(hrs[t])
        active_snipers = [sp for sp in active_snipers if int(sp["exit_ts"]) > hr_ts]
        sniper_open_syms = {str(sp["sym"]) for sp in active_snipers}

        bar_pnl, bar_fee = 0.0, 0.0
        if sniper_slot_lev > 0 and allowed_sniper_strats and hr_ts in events_map:
            snap_s = ctrl.update_state(current_equity=equity, vol_scalar=1.0)
            ramp_s = min(1.65, max(0.50, snap_s.active_base_leverage / max(start_leverage, 1.0)))
            scale_s = snap_s.cushion_scalar * ramp_s
            core_open_syms = {core_cols[i] for i in range(n_c) if abs(curr_w[i]) > 1e-6}

            def _ev_sort_key(ev_x: dict[str, Any]) -> tuple[int, int, float]:
                st_x = str(ev_x["strat"])
                sd_x = int(ev_x.get("side", 1))
                tier_p = 0 if st_x == SNIPER_M26_STRAT else (1 if (st_x == SNIPER_M09_STRAT and sd_x > 0) else 2)
                return (
                    int(ev_x["entry_ts"]),
                    tier_p if spec.sniper_sizing_mode == "kelly_tiered" else 0,
                    -float(ev_x["score"]),
                )

            evs_hr = sorted(events_map[hr_ts], key=_ev_sort_key)
            for ev in evs_hr:
                st_name = str(ev["strat"])
                if st_name not in allowed_sniper_strats:
                    continue
                if st_name == SNIPER_M04_STRAT and float(ev.get("entry_fr", -0.0004)) > float(spec.m04_max_fr) + 1e-9:
                    continue
                entry_ts = int(ev["entry_ts"])
                active_now = [sp for sp in active_snipers if int(sp["exit_ts"]) > entry_ts]
                if len(active_now) >= sniper_max_pos:
                    continue
                if spec.sniper_sizing_mode == "kelly_tiered" and st_name == SNIPER_M23_STRAT:
                    m23_cnt = sum(1 for sp in active_now if sp.get("strat") == SNIPER_M23_STRAT)
                    if m23_cnt >= 2:
                        continue
                sym_name = str(ev["sym"])
                if any(str(sp["sym"]) == sym_name for sp in active_now) or entry_ts < sniper_cd_until.get(sym_name, 0):
                    continue

                sd_ev = int(ev.get("side", 1))
                is_tier_a = st_name == SNIPER_M26_STRAT or (st_name == SNIPER_M09_STRAT and sd_ev > 0)
                if sym_name in core_open_syms:
                    if spec.sniper_preempt_core and is_tier_a and sym_name in core_cols:
                        j_pre = core_cols.index(sym_name)
                        bar_fee += abs(curr_w[j_pre]) * rebal_fee
                        (
                            curr_w[j_pre],
                            entry_px[j_pre],
                            lock_armed[j_pre],
                            active_p2_w[j_pre],
                        ) = (0.0, 0.0, False, 0.0)
                        core_open_syms.discard(sym_name)
                        n_preemptions += 1
                    else:
                        continue

                if spec.sniper_sizing_mode == "kelly_tiered":
                    if st_name == SNIPER_M26_STRAT:
                        base_sl_lev = 0.34
                    elif st_name == SNIPER_M09_STRAT and sd_ev > 0:
                        base_sl_lev = 0.30
                    elif st_name == SNIPER_M04_STRAT:
                        base_sl_lev = 0.14
                    else:
                        base_sl_lev = 0.20
                else:
                    base_sl_lev = sniper_slot_lev

                ex_ts = int(ev["exit_ts"])
                active_snipers.append(
                    {
                        "sym": sym_name,
                        "strat": st_name,
                        "exit_ts": ex_ts,
                        "weight": base_sl_lev * scale_s,
                        "hourly_ret": ev["hourly_ret"],
                    }
                )
                sniper_open_syms.add(sym_name)
                sniper_cd_until[sym_name] = ex_ts + (14400 if bool(ev["was_loss"]) else 3600)
                sniper_exec_counts[st_name] = sniper_exec_counts.get(st_name, 0) + 1

        prev_c = c_arr[t - 1] if t > 0 else open_core[0]
        cur_o, cur_c = open_core[t], c_arr[t]
        gap_ret = np.where(prev_c > 0, (cur_o - prev_c) / prev_c, 0.0)
        bar_pnl += float(np.sum(curr_w * gap_ret))

        fnd_cost = 0.0
        if hr_utc % 8 == 0 and not (off_m == 0 and spec.idealized_same_bar_coupon):
            fnd_cost = float(np.sum(curr_w * f_arr[t]))

        is_rebal_hr = hr_utc in rebal_hrs
        if is_rebal_hr and rebal_min == 0:
            bar_fee += _execute_rebalance(t, cur_o, sniper_open_syms)
            p_s, f_s = _resolve_minute_slice(t, 0, 60, cur_o, cur_c)
            if hr_utc % 8 == 0 and spec.idealized_same_bar_coupon:
                fnd_cost = float(np.sum(curr_w * f_arr[t]))
            bar_pnl += p_s
            bar_fee += f_s
        elif is_rebal_hr and rebal_min > 0:
            mid_px = _get_minute_open_vec(t, rebal_min)
            p_1, f_1 = _resolve_minute_slice(t, 0, rebal_min, cur_o, mid_px)
            bar_pnl += p_1
            bar_fee += f_1
            bar_fee += _execute_rebalance(t, mid_px, sniper_open_syms)
            p_2, f_2 = _resolve_minute_slice(t, rebal_min, 60, mid_px, cur_c)
            bar_pnl += p_2
            bar_fee += f_2
        else:
            p_s, f_s = _resolve_minute_slice(t, 0, 60, cur_o, cur_c)
            bar_pnl += p_s
            bar_fee += f_s

        cum_core_fee_pct += bar_fee * 100.0
        sniper_hr_pnl = sum(float(sp["weight"]) * float(sp["hourly_ret"].get(hr_ts, 0.0)) for sp in active_snipers)
        nr = bar_pnl - bar_fee - fnd_cost + sniper_hr_pnl
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
        "core_fee_drag_pct": round(cum_core_fee_pct, 2),
        "lock_exits": n_lock_exits,
        "tp_exits": n_tp_exits,
        "sniper_preemptions": n_preemptions,
        "mean_abs_beta_ratio": (round(float(np.mean(abs_beta_samples)), 3) if abs_beta_samples else 0.0),
    }


def evaluate_frente_d_suite(
    bundle: dict[str, Any],
    p2_inv_vol_mat: np.ndarray | None = None,
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[FrenteDSynthesisSpec, ...] = FRENTE_D_SPECS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluates all `FrenteDSynthesisSpec` experiments across `2023–2026` + fresh `$2,700` `2026` start."""
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    mats: dict[str, Any] = bundle["mats"]
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]
    p2_base_mat: np.ndarray = bundle["p2_w_mat"]
    p2_iv_mat = p2_inv_vol_mat if p2_inv_vol_mat is not None else p2_base_mat

    base_evs = list(bundle.get("base_sniper_events", []))
    m04_diag = analyze_m04_funding_buckets(base_evs)

    events_by_hr: dict[int, list[dict[str, Any]]] = {}
    for ev_b in base_evs:
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
        use_p2 = p2_iv_mat if sp.use_inv_vol_p2 else p2_base_mat
        sim_4y = simulate_frente_d_1m_portfolio(
            mats=mats,
            o_1m_core=o_1m_core,
            h_1m_core=h_1m_core,
            l_1m_core=l_1m_core,
            eval_idx=eval_idx,
            open_core=open_core,
            p2_w_mat=use_p2,
            spec=sp,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            sim_26 = simulate_frente_d_1m_portfolio(
                mats=mats,
                o_1m_core=o_1m_core[mask_2026],
                h_1m_core=h_1m_core[mask_2026],
                l_1m_core=l_1m_core[mask_2026],
                eval_idx=idx_2026,
                open_core=open_core[mask_2026],
                p2_w_mat=use_p2[mask_2026],
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
                "sharpe_4y": m_4y["full_sharpe"],
                "cagr_4y_pct": m_4y["full_cagr_pct"],
                "max_dd_4y_pct": m_4y["full_max_dd_pct"],
                "sharpe_23_24": m_4y["sharpe_23_24"],
                "sharpe_2025": m_4y["sharpe_2025"],
                "sharpe_2026": m_4y["sharpe_2026"],
                "cpcv_mean_sharpe": cpcv_4y["cpcv_mean_sharpe"],
                "cpcv_pos_splits_pct": cpcv_4y["cpcv_pos_splits_pct"],
                "core_fee_drag_pct": sim_4y["core_fee_drag_pct"],
                "lock_exits": sim_4y["lock_exits"],
                "tp_exits": sim_4y["tp_exits"],
                "sniper_preemptions": sim_4y["sniper_preemptions"],
                "mean_abs_beta_ratio": sim_4y["mean_abs_beta_ratio"],
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_ok,
                "sniper_counts": sim_4y["sniper_counts"],
            }
        )
    return pd.DataFrame(rows), m04_diag
