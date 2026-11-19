"""
Orthogonal Multi-Engine Risk Budgeting & Grand 4-Year Portfolio Stress-Test (Issue #71 / M16-5).

Synthesizes the deployed production baseline (`00_CURRENT_PROD`: Core 7-Factor BHS + Manga P2
Dollar-Bar TSMOM + Phase 2 Event Snipers) alongside each surviving `M16-4` orthogonal mechanism
under:
  1. Causal Inverse-Volatility / Risk-Parity Sleeve Budgeting (`60d` trailing realized vol,
     strictly `shift(1)` lagged -> zero lookahead!)
  2. Live Production `SteppedLeverageController` (`initial_capital = $2,700`, `100% USDT`,
     `BFUSD = OFF`, Two-Phase SafeRamp `$2,700 -> $3,000` boost + Cushion-Safe CPPI Governor
     protecting the `$2,500.00` Hard Equity Floor with `$240.00` smooth buffer)
  3. Dual Execution Cost Regimes:
     - Conservative Regime (`4.2 bps` Core / `2.6 bps` one-way M16 + `M16-3` queue haircut)
     - Maker Limit-Chase Regime (`3.2 bps` Core / `1.8 bps` one-way M16)
  4. Strict Promotion Gate (`Go / No-Go` for `M16-6`):
     - Minimum `+0.35` net Sharpe lift on BOTH `2023–2026` Full and `2026` Sealed Holdout
     - Full 4-year AND `2026` Holdout `MaxDD <= 20.0%`
     - `Min Equity > $2,520.00` (`$2,500` floor `100% SAFE`) starting from `$2,700.00` across
       both `2023–2026` Full and `2026` standalone!
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.risk.stepped_leverage import SteppedLeverageController


@dataclass(frozen=True)
class GrandPortfolioConfig:
    """Configuration for the M16-5 100% USDT Grand Portfolio Stress-Tester."""

    initial_equity_usd: float = 2700.0
    equity_floor_usd: float = 2500.0
    cushion_buffer_usd: float = 240.0
    recovery_boost_equity_usd: float = 3000.0
    start_leverage: float = 2.0
    recovery_boost_leverage: float = 2.8
    max_leverage: float = 4.2
    floor_leverage: float = 1.4
    target_daily_vol: float = 0.0115  # ~22% annualized target vol at 1x reference before SafeRamp
    max_gross_sleeve_sum: float = 2.60


def causal_inverse_vol_combine(
    sleeves_daily: dict[str, pd.Series],
    base_risk_shares: dict[str, float] | None = None,
    vol_lookback_days: int = 45,
    target_daily_vol: float = 0.0115,
    max_leverage_cap: float = 2.60,
) -> pd.Series:
    """
    Combines `K` orthogonal daily return streams using strictly causal (`shift(1)`)
    inverse-volatility risk budgeting and portfolio-level volatility targeting (`MaxDD <= 20%`).
    """
    if not sleeves_daily:
        return pd.Series(dtype=np.float64)

    df = pd.concat(sleeves_daily, axis=1).fillna(0.0).sort_index()
    cols = list(df.columns)
    shares = np.array([float((base_risk_shares or {}).get(c, 1.0)) for c in cols], dtype=np.float64)
    shares = shares / max(1e-12, float(shares.sum()))

    # Strictly causal `shift(1)` trailing volatility per sleeve (floor at 0.25%/day to avoid blow-up on sparse snipers)
    roll_vol = df.rolling(vol_lookback_days, min_periods=10).std(ddof=1).shift(1)
    expanding_vol = df.expanding(min_periods=3).std(ddof=1).shift(1)
    vol_est = roll_vol.fillna(expanding_vol).fillna(0.008).clip(lower=0.0025)

    inv_vol = (1.0 / vol_est).multiply(shares, axis=1)
    norm_w = inv_vol.div(inv_vol.sum(axis=1).replace(0.0, 1.0), axis=0)

    raw_combo = (df * norm_w).sum(axis=1)
    # Strictly causal `shift(1)` portfolio-level volatility targeting so orthogonal sleeves scale safely
    combo_vol = (
        raw_combo.rolling(30, min_periods=10)
        .std(ddof=1)
        .shift(1)
        .fillna(raw_combo.expanding(min_periods=3).std(ddof=1).shift(1))
        .fillna(target_daily_vol)
        .clip(lower=0.0020)
    )
    port_scale = (target_daily_vol / combo_vol).clip(0.35, max_leverage_cap)
    return (raw_combo * port_scale).rename("unit_portfolio_ret")


def simulate_compounding_with_live_shields(
    unit_daily_ret: pd.Series,
    cfg: GrandPortfolioConfig | None = None,
    extra_daily_cost_drag: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Simulates exact `$2,700` live compounding (`100% USDT`, `BFUSD=OFF`) driven by the production
    `SteppedLeverageController` (`Two-Phase SafeRamp + Cushion-Safe CPPI Governor` + `72h/14d` vol scalar
    + `$2,500` Hard Equity Floor).
    """
    c = cfg or GrandPortfolioConfig()
    s = unit_daily_ret.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(), {}

    ctrl = SteppedLeverageController(
        initial_capital=c.initial_equity_usd,
        start_leverage=c.start_leverage,
        max_leverage=c.max_leverage,
        floor_leverage=c.floor_leverage,
        step_leverage=0.35,
        gain_per_step=0.18,
        dd_bracket=0.07,
        recovery_boost_equity=c.recovery_boost_equity_usd,
        recovery_boost_leverage=c.recovery_boost_leverage,
        cushion_floor_usd=c.equity_floor_usd,
        cushion_buffer_usd=c.cushion_buffer_usd,
        min_cushion_scalar=0.15,
    )

    r_arr = s.to_numpy(dtype=np.float64) - float(extra_daily_cost_drag)
    # Causal trailing 10d vol scalar (`[0.55, 1.15]`, shifted by 1 day!)
    vol_10d = pd.Series(r_arr, index=s.index).rolling(10, min_periods=3).std(ddof=1).shift(1).fillna(c.target_daily_vol)
    vol_scalars = np.clip(c.target_daily_vol / np.maximum(vol_10d.to_numpy(dtype=np.float64), 1e-4), 0.55, 1.15)

    eq = float(c.initial_equity_usd)
    hwm = eq
    eq_path = np.empty(len(r_arr), dtype=np.float64)
    net_ret_path = np.empty(len(r_arr), dtype=np.float64)
    eff_lev_path = np.empty(len(r_arr), dtype=np.float64)
    dd_path = np.empty(len(r_arr), dtype=np.float64)

    for i, r_u in enumerate(r_arr):
        snap = ctrl.update_state(current_equity=eq, vol_scalar=float(vol_scalars[i]))
        # Normalize controller's `effective_leverage` relative to `recovery_boost_leverage` (`1.0x` unit reference)
        lev_mult = snap.effective_leverage / max(1e-6, c.recovery_boost_leverage)
        # Smooth pre-floor CPPI taper inside [$2,500, $2,530] (never permanently freezes recovery!)
        if eq <= c.equity_floor_usd + 30.0:
            lev_mult *= max(0.08, (eq - c.equity_floor_usd) / 30.0)
        day_r = float(np.clip(lev_mult * r_u, -0.18, 0.35))
        next_eq = eq * (1.0 + day_r)
        if next_eq < c.equity_floor_usd and eq > c.equity_floor_usd:
            # Intraday WebSocket Hard Equity Floor Kill-Switch (`$2,505` pre-breach flattening buffer)
            next_eq = max(c.equity_floor_usd + 5.0, next_eq)
            day_r = (next_eq / eq) - 1.0

        eq = next_eq
        hwm = max(hwm, eq)
        eq_path[i] = eq
        net_ret_path[i] = day_r
        eff_lev_path[i] = snap.effective_leverage
        dd_path[i] = (hwm - eq) / max(hwm, 1e-9)

    path_df = pd.DataFrame(
        {
            "equity_usd": eq_path,
            "daily_ret": net_ret_path,
            "effective_leverage": eff_lev_path,
            "drawdown": dd_path,
        },
        index=s.index,
    )

    n_days = len(net_ret_path)
    mu = float(np.mean(net_ret_path))
    sd = float(np.std(net_ret_path, ddof=1)) if n_days > 2 else 1.0
    sharpe = (mu / sd) * np.sqrt(365.25) if sd > 1e-12 else 0.0
    total_mult = eq_path[-1] / c.initial_equity_usd
    geom_daily_pct = ((total_mult ** (1.0 / max(1, n_days))) - 1.0) * 100.0
    cagr_pct = ((total_mult ** (365.25 / max(1, n_days))) - 1.0) * 100.0
    max_dd_pct = float(np.max(dd_path)) * 100.0
    calmar = cagr_pct / max(0.5, max_dd_pct)

    monthly_ret = (1.0 + path_df["daily_ret"]).resample("ME").prod() - 1.0
    monthly_win_pct = float((monthly_ret > 0.0).mean() * 100.0) if len(monthly_ret) > 0 else 0.0

    yearly_sharpe: dict[str, float] = {}
    for yr in sorted(path_df.index.year.unique()):
        sub_r = path_df.loc[path_df.index.year == yr, "daily_ret"].to_numpy(dtype=np.float64)
        if len(sub_r) >= 10 and float(np.std(sub_r, ddof=1)) > 1e-12:
            yearly_sharpe[str(yr)] = round(float(np.mean(sub_r) / np.std(sub_r, ddof=1) * np.sqrt(365.25)), 2)

    summary = {
        "initial_equity_usd": round(c.initial_equity_usd, 2),
        "final_equity_usd": round(float(eq_path[-1]), 2),
        "min_equity_usd": round(float(np.min(eq_path)), 2),
        "floor_2500_safe": bool(float(np.min(eq_path)) >= 2520.0),
        "sharpe": round(float(sharpe), 3),
        "geom_daily_pct": round(float(geom_daily_pct), 4),
        "cagr_pct": round(float(cagr_pct), 2),
        "max_dd_pct": round(float(max_dd_pct), 2),
        "calmar": round(float(calmar), 2),
        "worst_24h_pct": round(float(np.min(net_ret_path) * 100.0), 2),
        "monthly_win_rate_pct": round(monthly_win_pct, 1),
        "yearly_sharpe": yearly_sharpe,
    }
    return path_df, summary


def run_grand_portfolio_ablation_matrix(
    prod_baseline_daily: pd.Series,
    survivor_sleeves_daily: dict[str, pd.Series],
    cfg: GrandPortfolioConfig | None = None,
) -> dict[str, Any]:
    """
    Runs the full M16-5 Ablation & Promotion Matrix across:
      - `00_CURRENT_PROD`
      - `+ <Each M16 Survivor Individually>`
      - `M16_FULL_ORTHOGONAL_ENSEMBLE`
    Evaluated on BOTH:
      1. Full 4-Year (`2023–2026`) AND `2026` Standalone Holdout (starting fresh at `$2,700`!)
      2. Conservative (`+0.035%/day` extra fee haircut) vs Maker Limit-Chase (`0.0%` extra drag)
    """
    c = cfg or GrandPortfolioConfig()

    # Align common calendar grid across `2023–2026`
    all_series = {"00_CURRENT_PROD": prod_baseline_daily, **survivor_sleeves_daily}
    aligned_df = pd.concat(all_series, axis=1).fillna(0.0).sort_index()
    prod_s = aligned_df["00_CURRENT_PROD"]

    variants: list[tuple[str, dict[str, pd.Series], dict[str, float]]] = [
        ("00_CURRENT_PROD", {"prod": prod_s}, {"prod": 1.0}),
    ]
    for m_id in survivor_sleeves_daily:
        variants.append(
            (
                f"01_PROD_PLUS_{m_id}",
                {"prod": prod_s, m_id: aligned_df[m_id]},
                {"prod": 0.62, m_id: 0.38},
            )
        )

    if survivor_sleeves_daily:
        ens_dict = {"prod": prod_s}
        ens_weights = {"prod": 0.40}
        per_m_share = 0.60 / float(len(survivor_sleeves_daily))
        for m_id in survivor_sleeves_daily:
            ens_dict[m_id] = aligned_df[m_id]
            ens_weights[m_id] = per_m_share
        variants.append(("99_M16_FULL_ORTHOGONAL_ENSEMBLE", ens_dict, ens_weights))

    ablation_rows: list[dict[str, Any]] = []
    base_full_sh = 0.0
    base_2026_sh = 0.0

    for v_name, sl_map, w_shares in variants:
        unit_ret = (
            prod_s
            if v_name == "00_CURRENT_PROD"
            else causal_inverse_vol_combine(
                sl_map,
                base_risk_shares=w_shares,
                vol_lookback_days=45,
                target_daily_vol=c.target_daily_vol,
                max_leverage_cap=c.max_gross_sleeve_sum,
            )
        )
        # Constant Base Leverage (`1x` reference) metrics
        r_const = unit_ret.to_numpy(dtype=np.float64)
        const_sh = (
            float(np.mean(r_const) / np.std(r_const, ddof=1) * np.sqrt(365.25))
            if len(r_const) > 10 and float(np.std(r_const, ddof=1)) > 1e-12
            else 0.0
        )

        # Conservative Cost Regime (`+3.5 bps/day` additional execution friction stress)
        _, cons_full = simulate_compounding_with_live_shields(unit_ret, cfg=c, extra_daily_cost_drag=0.00035)
        unit_2026 = unit_ret[unit_ret.index.year >= 2026]
        _, cons_2026 = (
            simulate_compounding_with_live_shields(unit_2026, cfg=c, extra_daily_cost_drag=0.00035)
            if len(unit_2026) >= 15
            else (pd.DataFrame(), cons_full)
        )

        # Maker Limit-Chase Cost Regime (`3.2 bps` Maker baseline already embedded in sleeves)
        _, maker_full = simulate_compounding_with_live_shields(unit_ret, cfg=c, extra_daily_cost_drag=0.0)
        _, maker_2026 = (
            simulate_compounding_with_live_shields(unit_2026, cfg=c, extra_daily_cost_drag=0.0)
            if len(unit_2026) >= 15
            else (pd.DataFrame(), maker_full)
        )

        if v_name == "00_CURRENT_PROD":
            base_full_sh = maker_full["sharpe"]
            base_2026_sh = maker_2026["sharpe"]

        lift_full = round(maker_full["sharpe"] - base_full_sh, 3)
        lift_2026 = round(maker_2026["sharpe"] - base_2026_sh, 3)
        go_decision = bool(
            v_name == "00_CURRENT_PROD"
            or (
                lift_full >= 0.35
                and lift_2026 >= 0.35
                and maker_full["max_dd_pct"] <= 20.0
                and maker_2026["max_dd_pct"] <= 20.0
                and maker_full["floor_2500_safe"]
                and maker_2026["floor_2500_safe"]
                and cons_full["floor_2500_safe"]
            )
        )

        ablation_rows.append(
            {
                "variant": v_name,
                "const_1x_sharpe": round(const_sh, 3),
                "maker_full_sharpe": maker_full["sharpe"],
                "maker_sharpe_lift": lift_full,
                "maker_geom_daily_pct": maker_full["geom_daily_pct"],
                "maker_cagr_pct": maker_full["cagr_pct"],
                "maker_max_dd_pct": maker_full["max_dd_pct"],
                "maker_calmar": maker_full["calmar"],
                "maker_min_eq_usd": maker_full["min_equity_usd"],
                "maker_final_eq_usd": maker_full["final_equity_usd"],
                "maker_worst_24h_pct": maker_full["worst_24h_pct"],
                "maker_monthly_win_pct": maker_full["monthly_win_rate_pct"],
                "maker_yearly_sharpe": maker_full["yearly_sharpe"],
                "holdout_2026_sharpe": maker_2026["sharpe"],
                "holdout_2026_lift": lift_2026,
                "holdout_2026_geom_daily_pct": maker_2026["geom_daily_pct"],
                "holdout_2026_max_dd_pct": maker_2026["max_dd_pct"],
                "holdout_2026_min_eq_usd": maker_2026["min_equity_usd"],
                "cons_full_sharpe": cons_full["sharpe"],
                "cons_geom_daily_pct": cons_full["geom_daily_pct"],
                "cons_max_dd_pct": cons_full["max_dd_pct"],
                "cons_min_eq_usd": cons_full["min_equity_usd"],
                "cons_2026_sharpe": cons_2026["sharpe"],
                "go_for_m16_6": go_decision,
            }
        )

    corr_matrix = aligned_df.corr().round(3).to_dict()
    return {
        "ablation_rows": ablation_rows,
        "correlation_matrix": corr_matrix,
    }
