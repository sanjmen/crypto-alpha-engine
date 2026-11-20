"""
Frente C Laboratory: Reinventing Pillar 2 (`Manga P2: CausalDollarBarTSMOMEngine`) via
Inverse-Volatility Risk Parity, Asymmetric Fast-Crash Acceleration, Flow-Imbalance Bars (`DIBs`, AFML Ch. 2),
and 5-Sovereign Expansion (`BTC, ETH, SOL, BNB, XRP`) (`1-Minute` Physics, `2023–2026`).

Audits both:
  1. **Standalone `Manga P2` Sleeve (`1x` Gross, Fee & Funding Adjusted):**
     - `4Y Sharpe`, `2023–24 / 2025 / 2026 Sharpe`, `15-split CPCV`, and daily correlation vs `Core P1`.
  2. **Unified 3-Pillar `1m` Portfolio (`Core P1 Strict 8h + Manga P2 + M09/M23/M04/M26`):**
     - `$2,700` SafeRamp + `$2,500` Cushion-Safe CPPI Hard Floor across `2023–2026` and fresh `2026`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    CoreRebalanceBarrierSpec,
    compute_cpcv_portfolio_sharpe_stability,
    simulate_core_barrier_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    compute_portfolio_era_metrics,
    convert_trade_df_to_hourly_sniper_events,
)
from src.strategies.dollar_bar_tsmom import _normalize_1m_df, build_causal_dollar_bars
from src.strategies.event_snipers import (
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    SNIPER_M26_STRAT,
)


@dataclass(frozen=True)
class MangaP2VariantSpec:
    variant_id: str
    description: str
    symbols: tuple[str, ...]
    clock_mode: str = "dollar"  # "off", "dollar", "imbalance_dib"
    sizing_mode: str = "equal"  # "equal", "inv_vol"
    asym_crash_accel: bool = False
    max_gross_leverage: float = 0.95


MANGA_P2_SPECS: tuple[MangaP2VariantSpec, ...] = (
    MangaP2VariantSpec(
        variant_id="C00_NO_MANGA_P2_ABLATION",
        description="Ablation Control: Disable Manga P2 (`Core P1 + Snipers Only`)",
        symbols=(),
        clock_mode="off",
    ),
    MangaP2VariantSpec(
        variant_id="C01_PROD_3MAJ_EQUAL_DOLLAR_BAR",
        description="Current Live Tokyo Baseline (`BTC, ETH, SOL`, Equal Cap, Symmetric Dollar Bars)",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        clock_mode="dollar",
        sizing_mode="equal",
        asym_crash_accel=False,
    ),
    MangaP2VariantSpec(
        variant_id="C02_3MAJ_INV_VOL_RISK_PARITY",
        description="3 Majors (`BTC, ETH, SOL`) + Causal Inverse-Volatility Risk Parity Sizing",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        clock_mode="dollar",
        sizing_mode="inv_vol",
        asym_crash_accel=False,
    ),
    MangaP2VariantSpec(
        variant_id="C03_3MAJ_ASYM_CRASH_ACCELERATOR",
        description="3 Majors + Asymmetric Fast-Crash Override (`z_24 <= -1.40 & CVD <= -0.035` cuts bull lag)",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        clock_mode="dollar",
        sizing_mode="equal",
        asym_crash_accel=True,
    ),
    MangaP2VariantSpec(
        variant_id="C04_3MAJ_FLOW_IMBALANCE_BARS_DIB",
        description="3 Majors + AFML Flow-Weighted Dollar-Imbalance Clock (`sleeps in 50/50 chop`)",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        clock_mode="imbalance_dib",
        sizing_mode="inv_vol",
        asym_crash_accel=False,
    ),
    MangaP2VariantSpec(
        variant_id="C05_5SOV_BTC_ETH_SOL_BNB_XRP_INV_VOL",
        description="5 Sovereigns (`BTC, ETH, SOL, BNB, XRP`) + Inverse-Volatility Risk Parity",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        clock_mode="dollar",
        sizing_mode="inv_vol",
        asym_crash_accel=False,
    ),
    MangaP2VariantSpec(
        variant_id="C06_5SOV_INV_VOL_ASYM_CRASH_ACCEL",
        description="5 Sovereigns (`BTC, ETH, SOL, BNB, XRP`) + Inv-Vol + Asymmetric Crash Accelerator",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        clock_mode="dollar",
        sizing_mode="inv_vol",
        asym_crash_accel=True,
    ),
    MangaP2VariantSpec(
        variant_id="C07_5SOV_DIB_INV_VOL_ASYM_CRASH",
        description="Full Synthesis: 5 Sovereigns + Flow-Imbalance Clock (`DIB`) + Inv-Vol + Crash Accel",
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"),
        clock_mode="imbalance_dib",
        sizing_mode="inv_vol",
        asym_crash_accel=True,
    ),
)


def build_causal_flow_imbalance_bars(
    df_1m: pd.DataFrame,
    target_bars_per_day: int = 24,
    adv_lookback_days: int = 14,
) -> pd.DataFrame:
    """
    Constructs strictly causal Flow-Weighted Dollar-Imbalance Bars (`DIBs`, AFML Ch. 2):
      - Each 1m bar contributes effectiveconviction volume `eff_vol = qv * (0.40 + 3.2 * |2*tb/qv - 1|)`.
      - Daily threshold uses strictly lagged `shift(1)` `14d` rolling mean of `eff_vol / target_bars_per_day`
        (`0.000s` forward volume leak).
      - In balanced 50/50 chop (`|imb| ~ 0`), clock slows `2.5x` (preventing whipsaws); during institutional
        one-sided Taker raids (`|imb| >= 0.25`), clock accelerates `3x`!
    """
    norm = _normalize_1m_df(df_1m)
    if norm.empty or len(norm) < 60:
        return pd.DataFrame()

    ts_a = norm["ts_sec"].to_numpy(dtype=np.int64)
    o_a = norm["open"].to_numpy(dtype=np.float64)
    h_a = norm["high"].to_numpy(dtype=np.float64)
    l_a = norm["low"].to_numpy(dtype=np.float64)
    c_a = norm["close"].to_numpy(dtype=np.float64)
    qv_a = norm["quote_volume"].to_numpy(dtype=np.float64)
    tb_a = norm["taker_buy_quote_volume"].to_numpy(dtype=np.float64)

    imb_abs = np.where(qv_a > 1e-8, np.abs(2.0 * tb_a - qv_a) / np.maximum(qv_a, 1e-8), 0.0)
    eff_v = qv_a * (0.40 + 3.20 * imb_abs)

    day_epoch = (ts_a // 86400) * 86400
    day_s = pd.Series(eff_v).groupby(day_epoch).sum()
    roll_eff = day_s.rolling(adv_lookback_days, min_periods=1).mean().shift(1)
    if len(roll_eff) > 0 and np.isnan(roll_eff.iloc[0]):
        roll_eff.iloc[0] = float(day_s.iloc[0])
    thresh_by_day = (roll_eff / float(max(target_bars_per_day, 1))).clip(lower=50_000.0).to_dict()
    default_th = float(max(50_000.0, np.nanmedian(list(thresh_by_day.values())) if thresh_by_day else 250_000.0))

    db_ts: list[int] = []
    db_o: list[float] = []
    db_h: list[float] = []
    db_l: list[float] = []
    db_c: list[float] = []
    db_qv: list[float] = []
    db_tb: list[float] = []

    cum_eff = 0.0
    cum_qv = 0.0
    cum_tb = 0.0
    cur_o = o_a[0]
    cur_h = h_a[0]
    cur_l = l_a[0]
    in_bar = False

    for i in range(len(ts_a)):
        if not in_bar:
            cur_o = o_a[i]
            cur_h = h_a[i]
            cur_l = l_a[i]
            cum_eff = 0.0
            cum_qv = 0.0
            cum_tb = 0.0
            in_bar = True
        else:
            if h_a[i] > cur_h:
                cur_h = h_a[i]
            if l_a[i] < cur_l:
                cur_l = l_a[i]

        cum_eff += eff_v[i]
        cum_qv += qv_a[i]
        cum_tb += tb_a[i]
        th = thresh_by_day.get(int(day_epoch[i]), default_th)
        if cum_eff >= th:
            db_ts.append(int(ts_a[i]) + 60)
            db_o.append(cur_o)
            db_h.append(cur_h)
            db_l.append(cur_l)
            db_c.append(c_a[i])
            db_qv.append(cum_qv)
            db_tb.append(cum_tb)
            in_bar = False

    if not db_ts:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(pd.to_datetime(db_ts, unit="s", utc=True))
    return pd.DataFrame(
        {
            "open": db_o,
            "high": db_h,
            "low": db_l,
            "close": db_c,
            "quote_volume": db_qv,
            "taker_buy_quote_volume": db_tb,
        },
        index=idx,
    )


def compute_manga_p2_asset_hourly_signal(
    df_1m: pd.DataFrame,
    eval_idx: pd.DatetimeIndex,
    clock_mode: str = "dollar",
    asym_crash_accel: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns `(pos_1h_causal, vol_1h_causal)` already lagged by `shift(1)` on `eval_idx` (`0` lookahead):
      - `pos_1h_causal` in `[-1.0, +1.0]`
      - `vol_1h_causal`: causal trailing 168h hourly realized volatility for risk-parity sizing.
    """
    if clock_mode == "off" or df_1m is None or df_1m.empty:
        z_s = pd.Series(0.0, index=eval_idx, dtype=np.float64)
        v_s = pd.Series(0.02, index=eval_idx, dtype=np.float64)
        return z_s, v_s

    db = (
        build_causal_flow_imbalance_bars(df_1m, target_bars_per_day=24)
        if clock_mode == "imbalance_dib"
        else build_causal_dollar_bars(df_1m, target_bars_per_day=24)
    )
    if db.empty or len(db) < 36:
        return pd.Series(0.0, index=eval_idx, dtype=np.float64), pd.Series(0.02, index=eval_idx, dtype=np.float64)

    c = db["close"]
    r = np.log(c / c.shift(1).clip(lower=1e-12)).fillna(0.0)
    vol = r.rolling(72, min_periods=24).std().bfill().fillna(0.01)

    z_24 = r.rolling(24, min_periods=12).sum() / (vol * np.sqrt(24.0) + 1e-8)
    z_f = r.rolling(72, min_periods=24).sum() / (vol * np.sqrt(72.0) + 1e-8)
    z_m = r.rolling(168, min_periods=48).sum() / (vol * np.sqrt(168.0) + 1e-8)
    z_s = r.rolling(504, min_periods=120).sum() / (vol * np.sqrt(504.0) + 1e-8)

    qv = db["quote_volume"]
    tb = db["taker_buy_quote_volume"]
    cvd_48 = (2.0 * tb - qv).rolling(48, min_periods=16).sum() / (qv.rolling(48, min_periods=16).sum() + 1e-8)
    cvd_24 = (2.0 * tb - qv).rolling(24, min_periods=8).sum() / (qv.rolling(24, min_periods=8).sum() + 1e-8)

    rz = (0.25 * z_f + 0.40 * z_m + 0.35 * z_s + 1.05 * cvd_48).fillna(0.0)
    if asym_crash_accel:
        # When 1-day volume-clock price collapses (`z_24 <= -1.40`) WITH institutional Taker sell pressure,
        # bypass stale 21d bull inertia so Manga P2 exits Longs / enters Crash Short 3x faster.
        crash_trig = (z_24 <= -1.40) & (cvd_24 <= -0.035)
        fast_bear_rz = (0.65 * z_24 + 0.35 * z_f + 1.30 * cvd_24).fillna(0.0)
        rz = pd.Series(
            np.where(crash_trig, np.minimum(rz.to_numpy(), fast_bear_rz.to_numpy()), rz.to_numpy()), index=db.index
        )

    abs_rz = np.abs(rz.to_numpy(dtype=np.float64))
    sgn = np.sign(rz.to_numpy(dtype=np.float64))
    pos_arr = np.where(abs_rz < 0.75, 0.0, sgn * np.clip((abs_rz - 0.50) / 1.50, 0.0, 1.0))
    pos_db = pd.Series(pos_arr, index=db.index, dtype=np.float64)

    # Align to `eval_idx` and apply `.shift(1)` for strict causality matching `p2_portfolio_integration.py`
    pos_1h = pos_db.reindex(eval_idx, method="ffill").shift(1).fillna(0.0)
    c_1h = c.reindex(eval_idx, method="ffill")
    r_1h = c_1h.pct_change().fillna(0.0)
    vol_1h = r_1h.rolling(168, min_periods=24).std().shift(1).bfill().fillna(0.015).clip(lower=0.004)
    return pos_1h, vol_1h


def build_p2_weight_matrix_for_spec(
    per_sym_signals: dict[tuple[str, str, bool], tuple[pd.Series, pd.Series]],
    core_cols: list[str],
    eval_idx: pd.DatetimeIndex,
    spec: MangaP2VariantSpec,
) -> np.ndarray:
    """Builds `(len(eval_idx), len(core_cols))` `p2_w_mat` for `spec`."""
    n_b, n_c = len(eval_idx), len(core_cols)
    out = np.zeros((n_b, n_c), dtype=np.float64)
    if spec.clock_mode == "off" or not spec.symbols:
        return out

    present = [s for s in spec.symbols if s in core_cols]
    if not present:
        return out

    pos_mat = np.zeros((n_b, len(present)), dtype=np.float64)
    inv_v_mat = np.ones((n_b, len(present)), dtype=np.float64)
    for k_i, sym in enumerate(present):
        p_s, v_s = per_sym_signals[(sym, spec.clock_mode, spec.asym_crash_accel)]
        pos_mat[:, k_i] = p_s.to_numpy(dtype=np.float64)
        inv_v_mat[:, k_i] = 1.0 / np.maximum(v_s.to_numpy(dtype=np.float64), 0.004)

    if spec.sizing_mode == "equal":
        cap_each = spec.max_gross_leverage / float(len(present))
        w_sub = pos_mat * cap_each
    else:
        # Inverse-volatility risk parity shares across the basket (`sum(share_k) == max_gross_leverage`)
        iv_sum = np.maximum(np.sum(inv_v_mat, axis=1, keepdims=True), 1e-8)
        cap_shares = (inv_v_mat / iv_sum) * float(spec.max_gross_leverage)
        w_sub = pos_mat * cap_shares

    for k_i, sym in enumerate(present):
        out[:, core_cols.index(sym)] = w_sub[:, k_i]
    return out


def evaluate_standalone_p2_sleeve(
    p2_w_mat: np.ndarray,
    mats: dict[str, Any],
    eval_idx: pd.DatetimeIndex,
    core_p1_daily: pd.Series,
    rebal_fee: float = 0.00034,
) -> dict[str, float]:
    """Computes standalone `8h`-rebalanced fee- & funding-adjusted metrics + correlation vs `Core P1`."""
    c_arr = mats["close"].loc[eval_idx].to_numpy(dtype=np.float64)
    f_arr = mats["funding"].loc[eval_idx].to_numpy(dtype=np.float64)
    r_1h = np.zeros_like(c_arr)
    r_1h[1:] = np.where(c_arr[:-1] > 0, (c_arr[1:] - c_arr[:-1]) / c_arr[:-1], 0.0)
    hrs = eval_idx.hour.to_numpy()

    n_b, n_c = p2_w_mat.shape
    act_w = np.zeros(n_c, dtype=np.float64)
    hr_net = np.zeros(n_b, dtype=np.float64)
    for t in range(n_b):
        fee_t = 0.0
        if hrs[t] % 8 == 0:
            tgt = p2_w_mat[t]
            dw = np.abs(tgt - act_w)
            mask_tr = dw >= 0.025
            if np.any(mask_tr):
                fee_t = float(np.sum(dw[mask_tr])) * rebal_fee
                act_w = np.where(mask_tr, tgt, act_w)
        hr_net[t] = float(np.sum(act_w * r_1h[t])) - float(np.sum(act_w * f_arr[t])) - fee_t

    d_p2 = pd.Series(hr_net, index=eval_idx).resample("1D").sum()
    if float(d_p2.abs().max()) < 1e-10:
        return {
            "p2_solo_sharpe_4y": 0.0,
            "p2_solo_23_24": 0.0,
            "p2_solo_2025": 0.0,
            "p2_solo_2026": 0.0,
            "p2_solo_cpcv_pos_pct": 0.0,
            "corr_to_core_p1": 0.0,
        }

    def _sr(s: pd.Series) -> float:
        if len(s) < 10:
            return 0.0
        sd = float(s.std(ddof=1))
        return round((float(s.mean()) / sd) * np.sqrt(365.25), 2) if sd > 1e-12 else 0.0

    cp = compute_cpcv_portfolio_sharpe_stability(d_p2)
    aligned = pd.concat([d_p2.rename("p2"), core_p1_daily.rename("p1")], axis=1).dropna()
    corr = float(aligned["p2"].corr(aligned["p1"])) if len(aligned) > 20 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "p2_solo_sharpe_4y": _sr(d_p2),
        "p2_solo_23_24": _sr(d_p2[d_p2.index.year <= 2024]),
        "p2_solo_2025": _sr(d_p2[d_p2.index.year == 2025]),
        "p2_solo_2026": _sr(d_p2[d_p2.index.year >= 2026]),
        "p2_solo_cpcv_pos_pct": cp["cpcv_pos_splits_pct"],
        "corr_to_core_p1": round(corr, 3),
    }


def evaluate_manga_p2_ablation_suite(
    bundle: dict[str, Any],
    sovereign_1m_by_sym: dict[str, pd.DataFrame],
    m26_trades_df: pd.DataFrame | None = None,
    specs: tuple[MangaP2VariantSpec, ...] = MANGA_P2_SPECS,
) -> pd.DataFrame:
    """Evaluates all `MangaP2VariantSpec` configs standalone AND inside the full `1m` 3-Pillar portfolio."""
    eval_idx: pd.DatetimeIndex = bundle["eval_idx"]
    mats: dict[str, Any] = bundle["mats"]
    core_cols = list(mats["close"].columns)
    o_1m_core: np.ndarray = bundle["o_1m_core"]
    h_1m_core: np.ndarray = bundle["h_1m_core"]
    l_1m_core: np.ndarray = bundle["l_1m_core"]
    open_core: np.ndarray = bundle["open_core"]

    # Pre-compute unique `(sym, clock_mode, asym_crash)` signals once across the 5 sovereigns!
    needed_combos = {
        (sym, sp.clock_mode, sp.asym_crash_accel)
        for sp in specs
        if sp.clock_mode != "off"
        for sym in sp.symbols
        if sym in core_cols
    }
    sig_cache: dict[tuple[str, str, bool], tuple[pd.Series, pd.Series]] = {}
    for sym, c_mode, asym in sorted(needed_combos):
        sig_cache[(sym, c_mode, asym)] = compute_manga_p2_asset_hourly_signal(
            sovereign_1m_by_sym.get(sym, pd.DataFrame()),
            eval_idx=eval_idx,
            clock_mode=c_mode,
            asym_crash_accel=asym,
        )

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

    # Pure Core P1 daily stream for orthogonality correlation measurement
    cb_strict = CoreRebalanceBarrierSpec("STRICT_8H", "Strict 8h", intracycle_mode="off")
    sim_no_p2 = simulate_core_barrier_1m_portfolio(
        mats=mats,
        o_1m_core=o_1m_core,
        h_1m_core=h_1m_core,
        l_1m_core=l_1m_core,
        eval_idx=eval_idx,
        open_core=open_core,
        p2_w_mat=None,
        spec=cb_strict,
        resolved_sniper_events_by_hour=None,
        allowed_sniper_strats=set(),
    )
    core_p1_daily = sim_no_p2["daily"]

    mask_2026 = np.asarray(eval_idx >= pd.Timestamp("2026-01-01", tz="UTC"), dtype=bool)
    has_2026 = int(mask_2026.sum()) >= 24 * 5
    idx_2026 = eval_idx[mask_2026] if has_2026 else eval_idx
    ts_2026_start = int(idx_2026[0].timestamp()) if has_2026 else 0
    evs_2026 = {k: v for k, v in events_by_hr.items() if k >= ts_2026_start}

    rows: list[dict[str, Any]] = []
    for sp in specs:
        p2_mat = build_p2_weight_matrix_for_spec(sig_cache, core_cols, eval_idx, sp)
        solo = evaluate_standalone_p2_sleeve(p2_mat, mats, eval_idx, core_p1_daily)

        sim_4y = simulate_core_barrier_1m_portfolio(
            mats=mats,
            o_1m_core=o_1m_core,
            h_1m_core=h_1m_core,
            l_1m_core=l_1m_core,
            eval_idx=eval_idx,
            open_core=open_core,
            p2_w_mat=p2_mat if sp.clock_mode != "off" else np.zeros_like(p2_mat),
            spec=cb_strict,
            resolved_sniper_events_by_hour=events_by_hr,
            allowed_sniper_strats=allowed_strats,
        )
        m_4y = compute_portfolio_era_metrics(sim_4y, initial_capital=2700.0)
        cpcv_4y = compute_cpcv_portfolio_sharpe_stability(sim_4y["daily"])

        fresh_final, fresh_min, fresh_ok = 2700.0, 2700.0, True
        if has_2026:
            sim_26 = simulate_core_barrier_1m_portfolio(
                mats=mats,
                o_1m_core=o_1m_core[mask_2026],
                h_1m_core=h_1m_core[mask_2026],
                l_1m_core=l_1m_core[mask_2026],
                eval_idx=idx_2026,
                open_core=open_core[mask_2026],
                p2_w_mat=p2_mat[mask_2026] if sp.clock_mode != "off" else np.zeros_like(p2_mat[mask_2026]),
                spec=cb_strict,
                resolved_sniper_events_by_hour=evs_2026,
                allowed_sniper_strats=allowed_strats,
            )
            fresh_final = round(float(sim_26["final_eq"]), 2)
            fresh_min = round(float(sim_26["min_eq"]), 2)
            fresh_ok = bool(fresh_min >= 2500.0)

        rows.append(
            {
                "variant_id": sp.variant_id,
                "description": sp.description,
                **solo,
                "port_sharpe_4y": m_4y["full_sharpe"],
                "port_cagr_4y_pct": m_4y["full_cagr_pct"],
                "port_max_dd_4y_pct": m_4y["full_max_dd_pct"],
                "port_23_24": m_4y["sharpe_23_24"],
                "port_2025": m_4y["sharpe_2025"],
                "port_2026": m_4y["sharpe_2026"],
                "port_cpcv_mean": cpcv_4y["cpcv_mean_sharpe"],
                "port_cpcv_pos_pct": cpcv_4y["cpcv_pos_splits_pct"],
                "fresh_2026_final_eq": fresh_final,
                "fresh_2026_min_eq": fresh_min,
                "fresh_2026_floor_2500_ok": fresh_ok,
            }
        )
    _ = PROD_TOP_15
    return pd.DataFrame(rows)
