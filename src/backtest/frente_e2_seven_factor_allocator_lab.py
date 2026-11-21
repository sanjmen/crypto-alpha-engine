"""
Frente E2 Laboratory: `SevenFactorOnlineEngine` `1-Minute` Leave-One-Factor-Out (`LOFO`),
Standalone `8h` Factor Anatomy, Multi-Modal `premiumIndex` Basis Overlay, and `8h`-Aligned
Online Meta-Allocator Audit (`2023–2026`).

Dissects every internal component of Pillar 1 (`Core P1: SevenFactorOnlineEngine`) on exact
`1-minute` physics (`PROD_TOP_15` + `Manga P2` + `Snipers` under the realistic `M17 D11` stack):
  1. **Part E2.1 — Standalone `8h`-Rebalanced Factor Anatomy (`F1..F7` + `F8_CMF`):**
     - Evaluates each individual factor at `1x` Gross Dollar-Neutral on the strict `8h` clock
       (`2.55 bps` execution cost + realized `:00/:08/:16` funding), reporting `4Y Sharpe`,
       `2023–24 / 2025 / 2026 Sharpe`, `15`-split CPCV, turnover, and mean meta-allocator weight share.
  2. **Part E2.2 — Leave-One-Factor-Out (`LOFO`) & Meta-Allocator Architecture (`E2_00..E2_14`):**
     - Drops each factor (`macro, med, fast, rev, carry, liq, size`) one at a time (`0` weight via
       `priors[f]=0.0` & `min_tstat_by_factor[f]=999.0`) to measure its true marginal contribution
       to `4Y Sharpe`, `MaxDD`, `2025` chop resilience, and `$2,700` `2026` equity.
     - Audits `enable_chop_range_arb`, `enable_holistic_regime_shields`, real `1h` `premiumIndex`
       Spot-Perp Basis Alpha (`Issue #49`), `enable_btc_lead_lag`, `Zero-Priors` pure online IC,
       `8h`-aligned slower meta-lookbacks `(72, 168, 360)` with `2.5 bps` turnover penalty, and
       tail concentration (`comb_power=1.65`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import compute_cpcv_portfolio_sharpe_stability
from src.backtest.frente_d_architecture_synthesis_lab import (
    FRENTE_D_SPECS,
    simulate_frente_d_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    compute_portfolio_era_metrics,
    convert_trade_df_to_hourly_sniper_events,
)
from src.strategies.event_snipers import (
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_M26_STRAT,
)
from src.strategies.seven_factor_engine import (
    DEFAULT_PRIORS,
    FACTOR_NAMES,
    SevenFactorOnlineEngine,
    make_continuous_neutral_weights,
)


@dataclass(frozen=True)
class FrenteE2FactorSpec:
    exp_id: str
    description: str
    drop_factor: str | None = None
    zero_priors: bool = False
    custom_priors: dict[str, float] | None = None
    lookbacks: tuple[int, int, int] = (36, 96, 240)
    factor_turnover_penalty: float = 0.0001
    comb_power: float = 1.3
    enable_chop_range_arb: bool = True
    enable_holistic_regime_shields: bool = True
    enable_btc_lead_lag: bool = False
    use_premium_index: bool = False


FRENTE_E2_SPECS: tuple[FrenteE2FactorSpec, ...] = (
    FrenteE2FactorSpec(
        exp_id="E2_00_BASELINE_D11_7FACTOR",
        description="Reference M17 Stack (`D11` + Current Production `SevenFactorOnlineEngine`)",
    ),
    # --- Leave-One-Factor-Out (LOFO) Marginal Attribution ---
    FrenteE2FactorSpec(
        exp_id="E2_01_DROP_F1_MACRO",
        description="LOFO: Drop `F1_Macro` (7d-30d Idio Trend, Prior `0.25`)",
        drop_factor="macro",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_02_DROP_F2_MED",
        description="LOFO: Drop `F2_Med` (3d-7d Idio Momentum, Prior `0.14`)",
        drop_factor="med",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_03_DROP_F3_FAST",
        description="LOFO: Drop `F3_Fast` (12h-48h Swing Momentum, Prior `0.19`)",
        drop_factor="fast",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_04_DROP_F4_REV",
        description="LOFO: Drop `F4_Rev` (24h-72h Mean Reversion, Prior `0.16`)",
        drop_factor="rev",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_05_DROP_F5_CARRY",
        description="LOFO: Drop `F5_Carry` (72h Contrarian Funding, Prior `0.18`)",
        drop_factor="carry",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_06_DROP_F6_LIQ",
        description="LOFO: Drop `F6_Liq_Surge` (168h/720h Volume Expansion, Prior `0.04`)",
        drop_factor="liq",
    ),
    FrenteE2FactorSpec(
        exp_id="E2_07_DROP_F7_SIZE",
        description="LOFO: Drop `F7_Size` (168h Institutional Size Quality, Prior `0.22`)",
        drop_factor="size",
    ),
    # --- Specialist Overlays & Multi-Modal Inputs ---
    FrenteE2FactorSpec(
        exp_id="E2_08_DISABLE_CHOP_RANGE_ARB",
        description="Ablation: Disable `enable_chop_range_arb` Intra-Sector Range-Arb Specialist",
        enable_chop_range_arb=False,
    ),
    FrenteE2FactorSpec(
        exp_id="E2_09_DISABLE_HOLISTIC_SHIELDS",
        description="Ablation: Disable `enable_holistic_regime_shields`",
        enable_holistic_regime_shields=False,
    ),
    FrenteE2FactorSpec(
        exp_id="E2_10_ENABLE_PREMIUM_BASIS_ALPHA",
        description="Multi-Modal: Inject Real `1h` `premiumIndexKlines` (`Issue #49` Spot-Perp Basis Guards)",
        use_premium_index=True,
    ),
    FrenteE2FactorSpec(
        exp_id="E2_11_ENABLE_BTC_LEAD_LAG_6H",
        description="Overlay: Enable `enable_btc_lead_lag` (`6h` BTC -> Alt Beta Catch-Up Spillover)",
        enable_btc_lead_lag=True,
    ),
    # --- Meta-Allocator Mathematics & Priors ---
    FrenteE2FactorSpec(
        exp_id="E2_12_ZERO_PRIORS_PURE_ONLINE_IC",
        description="Meta-Allocator: `Zero Priors` (Pure Causal Online `ReLU(t_stat)^2` + No-Edge De-Gross)",
        zero_priors=True,
    ),
    FrenteE2FactorSpec(
        exp_id="E2_13_8H_ALIGNED_META_LOOKBACKS",
        description="Meta-Allocator: Slower `8h`-Aligned Lookbacks `(72, 168, 360)` + `2.5 bps` Turn Penalty",
        lookbacks=(72, 168, 360),
        factor_turnover_penalty=0.00025,
    ),
    FrenteE2FactorSpec(
        exp_id="E2_14_TAIL_POWER_165_CONCENTRATED",
        description="Portfolio Geometry: Higher Tail Concentration (`comb_power = 1.65` vs `1.30`)",
        comb_power=1.65,
    ),
)


def evaluate_standalone_8h_factors(
    bars_1h_core: dict[str, pd.DataFrame],
    fnd_core: dict[str, pd.Series],
    eval_idx: pd.DatetimeIndex,
    rebal_fee: float = 0.000255,
) -> pd.DataFrame:
    """
    Computes standalone `8h`-rebalanced net performance (`2.55 bps` fee + funding)
    and average production meta-allocator share for each of `F1..F7` + `F8_CMF`.
    """
    eng_base = SevenFactorOnlineEngine(
        enable_chop_range_arb=True,
        enable_holistic_regime_shields=True,
    )
    mats_base = eng_base.compute_matrices(bars_1h=bars_1h_core, funding_rates=fnd_core)
    fw_mean = mats_base["fw_norm"].reindex(eval_idx).mean()

    c_arr = mats_base["close"].loc[eval_idx].to_numpy(dtype=np.float64)
    f_arr = mats_base["funding"].loc[eval_idx].to_numpy(dtype=np.float64)
    r_1h = np.zeros_like(c_arr)
    r_1h[1:] = np.where((c_arr[:-1] > 0) & np.isfinite(c_arr[:-1]), (c_arr[1:] - c_arr[:-1]) / c_arr[:-1], 0.0)
    hrs = eval_idx.hour.to_numpy()
    raw_close_core = pd.DataFrame({s: bars_1h_core[s]["close"] for s in mats_base["close"].columns}).reindex(
        mats_base["close"].index
    )
    age_ok = raw_close_core.notna().cumsum() >= 168

    rows: list[dict[str, Any]] = []
    # Evaluate each single factor by setting ONLY that factor's prior=1.0 and gating all others out
    for f_name in FACTOR_NAMES:
        only_priors = {k: (1.0 if k == f_name else 0.0) for k in FACTOR_NAMES}
        gates_other = {k: 999.0 for k in FACTOR_NAMES if k != f_name}
        eng_f = SevenFactorOnlineEngine(
            priors=only_priors,
            f6_min_tstat=None,
            f5_min_tstat=None,
            min_tstat_by_factor=gates_other,
            enable_chop_range_arb=False,
            enable_holistic_regime_shields=False,
            enable_no_edge_gate=False,
            comb_ema_span=3.0,
        )
        m_f = eng_f.compute_matrices(bars_1h=bars_1h_core, funding_rates=fnd_core)
        w_f = m_f["w_base_1x"].where(age_ok, 0.0).shift(1).fillna(0.0).loc[eval_idx].to_numpy(dtype=np.float64)
        # Normalize active rows to 1x gross so every factor is compared on exact 1x gross leverage!
        g_row = np.sum(np.abs(w_f), axis=1, keepdims=True)
        w_1x = np.where(g_row > 1e-8, w_f / np.maximum(g_row, 1e-8), 0.0)

        n_b, n_c = w_1x.shape
        curr = np.zeros(n_c, dtype=np.float64)
        hr_net = np.zeros(n_b, dtype=np.float64)
        turn_tot = 0.0
        for t in range(n_b):
            fee_t = 0.0
            if hrs[t] % 8 == 0:
                dw = np.abs(w_1x[t] - curr)
                mv = dw >= 0.02
                if np.any(mv):
                    turn_step = float(np.sum(dw[mv]))
                    fee_t = turn_step * rebal_fee
                    turn_tot += turn_step
                    curr = np.where(mv, w_1x[t], curr)
            hr_net[t] = float(np.sum(curr * r_1h[t])) - float(np.sum(curr * f_arr[t])) - fee_t

        d_f = pd.Series(hr_net, index=eval_idx).resample("1D").apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
        eq_f = 2700.0 * np.cumprod(1.0 + d_f.to_numpy())
        m_st = compute_portfolio_era_metrics(
            {
                "daily": d_f,
                "final_eq": float(eq_f[-1]) if len(eq_f) else 2700.0,
                "min_eq": float(np.min(eq_f)) if len(eq_f) else 2700.0,
                "worst_24h_dd": 0.0,
                "sniper_counts": {},
            },
            initial_capital=2700.0,
        )
        cp_f = compute_cpcv_portfolio_sharpe_stability(d_f)
        rows.append(
            {
                "factor": f_name,
                "default_prior": float(DEFAULT_PRIORS.get(f_name, 0.0)),
                "mean_prod_share_pct": round(100.0 * float(fw_mean.get(f_name, 0.0)), 1),
                "solo_8h_sharpe_4y": m_st["full_sharpe"],
                "solo_8h_cagr_pct": m_st["full_cagr_pct"],
                "solo_8h_max_dd_pct": m_st["full_max_dd_pct"],
                "solo_23_24": m_st["sharpe_23_24"],
                "solo_2025": m_st["sharpe_2025"],
                "solo_2026": m_st["sharpe_2026"],
                "cpcv_mean_sharpe": cp_f["cpcv_mean_sharpe"],
                "cpcv_pos_splits_pct": cp_f["cpcv_pos_splits_pct"],
                "turnover_1x_per_day": round(turn_tot / max(1.0, len(d_f)), 2),
            }
        )
    _ = make_continuous_neutral_weights
    return pd.DataFrame(rows)


def evaluate_frente_e2_suite(
    bundle: dict[str, Any],
    bars_1h_by_sym: dict[str, pd.DataFrame],
    funding_by_sym: dict[str, pd.DataFrame],
    prem_1h_by_sym: dict[str, pd.DataFrame] | None = None,
    p2_inv_vol_mat: np.ndarray | None = None,
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[FrenteE2FactorSpec, ...] = FRENTE_E2_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs Part E2.1 (`Standalone 8h Factor Anatomy`) and Part E2.2 (`LOFO & Meta-Allocator Suite`)."""
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]
    p2_mat = p2_inv_vol_mat if p2_inv_vol_mat is not None else bundle["p2_w_mat"]

    present_core = [s for s in PROD_TOP_15 if s in bars_1h_by_sym]
    bars_1h_core: dict[str, pd.DataFrame] = {}
    fnd_core: dict[str, pd.Series] = {}
    prem_core: dict[str, pd.Series] = {}
    for s in present_core:
        b = bars_1h_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
        if "volume" not in b.columns:
            b["volume"] = b["quote_volume"] / np.maximum(b["close"].to_numpy(dtype=np.float64), 1e-12)
        b.index = pd.DatetimeIndex(pd.to_datetime(b["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
        b.drop(columns=["timestamp"], inplace=True)
        bars_1h_core[s] = b
        if s in funding_by_sym and not funding_by_sym[s].empty:
            f = funding_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            f_idx = pd.DatetimeIndex(
                pd.to_datetime(((f["timestamp"].to_numpy(dtype=np.int64) + 30) // 3600) * 3600, unit="s", utc=True)
            )
            f_s = pd.Series(f["funding_rate"].to_numpy(dtype=np.float64), index=f_idx)
            fnd_core[s] = f_s[~f_s.index.duplicated(keep="last")]
        if prem_1h_by_sym and s in prem_1h_by_sym and not prem_1h_by_sym[s].empty:
            p = prem_1h_by_sym[s].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            p_idx = pd.DatetimeIndex(pd.to_datetime(p["timestamp"].to_numpy(dtype=np.int64), unit="s", utc=True))
            p_s = pd.Series(p["p_close_bps"].to_numpy(dtype=np.float64) * 1e-4, index=p_idx)
            prem_core[s] = p_s[~p_s.index.duplicated(keep="last")]

    solo_factors_df = evaluate_standalone_8h_factors(bars_1h_core, fnd_core, eval_idx=eval_idx)

    raw_close_core = pd.DataFrame({s: bars_1h_core[s]["close"] for s in present_core})
    age_ok_base = raw_close_core.notna().cumsum() >= 168

    base_evs = list(bundle.get("base_sniper_events", []))
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

    d11_exec_spec = next(s for s in FRENTE_D_SPECS if s.exp_id == "D11_M17_STACK_A_C_PLUS_KELLY_M04")
    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}

    rows: list[dict[str, Any]] = []
    for sp in specs:
        priors_use: dict[str, float] | None
        if sp.zero_priors:
            priors_use = None
        elif sp.custom_priors is not None:
            priors_use = dict(sp.custom_priors)
        else:
            priors_use = dict(DEFAULT_PRIORS)
            if sp.drop_factor is not None:
                priors_use[sp.drop_factor] = 0.0

        gates_use: dict[str, float] = {"liq": 0.75}
        if sp.enable_chop_range_arb:
            gates_use["carry"] = 1.25
        if sp.drop_factor is not None:
            gates_use[sp.drop_factor] = 999.0

        eng = SevenFactorOnlineEngine(
            lookbacks=sp.lookbacks,
            priors=priors_use,
            comb_power=float(sp.comb_power),
            factor_turnover_penalty=float(sp.factor_turnover_penalty),
            f6_min_tstat=gates_use.get("liq"),
            f5_min_tstat=gates_use.get("carry"),
            min_tstat_by_factor=gates_use,
            enable_chop_range_arb=sp.enable_chop_range_arb,
            enable_holistic_regime_shields=sp.enable_holistic_regime_shields,
            enable_btc_lead_lag=sp.enable_btc_lead_lag,
        )
        mats_sp = eng.compute_matrices(
            bars_1h=bars_1h_core,
            funding_rates=fnd_core,
            premium_index=prem_core if (sp.use_premium_index and prem_core) else None,
        )
        age_ok = age_ok_base.reindex(index=mats_sp["close"].index, columns=mats_sp["close"].columns).fillna(False)
        mats_sp["w_masked"] = mats_sp["w_base_1x"].where(age_ok, 0.0)

        sim_4y = simulate_frente_d_1m_portfolio(
            mats=mats_sp,
            o_1m_core=o_1m_core,
            h_1m_core=h_1m_core,
            l_1m_core=l_1m_core,
            eval_idx=eval_idx,
            open_core=open_core,
            p2_w_mat=p2_mat,
            spec=d11_exec_spec,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            sim_26 = simulate_frente_d_1m_portfolio(
                mats=mats_sp,
                o_1m_core=o_1m_core[mask_2026],
                h_1m_core=h_1m_core[mask_2026],
                l_1m_core=l_1m_core[mask_2026],
                eval_idx=idx_2026,
                open_core=open_core[mask_2026],
                p2_w_mat=p2_mat[mask_2026],
                spec=d11_exec_spec,
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
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_ok,
            }
        )
    return solo_factors_df, pd.DataFrame(rows)
