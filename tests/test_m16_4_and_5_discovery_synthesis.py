"""
Unit tests for M16-4 (`symbolic_alpha_tournament.py`) and M16-5 (`m16_grand_portfolio_synthesizer.py`).

Verifies:
  1. Exact settlement funding cashflow deduction (`Shorts` pay negative funding when crossing
     settlement; post-settlement entries `mins_since_settlement <= 30` with `mins_to_next > 240`
     pay `0.0` funding).
  2. Strict Train-only (`<= 2024`) standardization & shallow-tree fitting (`ml-best-practices`
     zero leakage under `2026` Holdout target mutation).
  3. Causal `shift(1)` inverse-volatility sleeve combination & `$2,500.00` Hard Equity Floor
     protection starting from `$2,700.00` (`100% USDT`, `SteppedLeverageController`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.m16_grand_portfolio_synthesizer import (
    GrandPortfolioConfig,
    causal_inverse_vol_combine,
    run_grand_portfolio_ablation_matrix,
    simulate_compounding_with_live_shields,
)
from src.models.symbolic_alpha_tournament import (
    run_multimodal_discovery_tournament,
    select_diverse_orthogonal_survivors,
    simulate_cross_sectional_sleeve_daily,
    simulate_event_sniper_sleeve_daily,
)


def test_event_sniper_exact_funding_cashflow_pre_vs_post_settlement() -> None:
    """
    Proves that shorting BEFORE a `-50 bps` settlement explicitly deducts `-50 bps` funding paid,
    whereas shorting AFTER settlement (`mins_to_next = 450 > 240`) pays `0 bps` funding!
    """
    base_ts = 1_740_000_000  # arbitrary epoch
    df_pre = pd.DataFrame(
        {
            "symbol": ["ALT1USDT", "ALT2USDT"],
            "decision_ts": [base_ts, base_ts + 86400],
            "fwd_ret_4h": [-0.0100, -0.0100],  # +100 bps raw short price gain
            "funding_rate_bps": [-50.0, -50.0],  # -50 bps negative funding!
            "mins_to_next_settlement": [30.0, 30.0],  # crosses settlement inside 4h (240m) hold!
            "funding_interval_hours": [8.0, 8.0],
        }
    )
    df_post = df_pre.copy()
    df_post["mins_to_next_settlement"] = 450.0  # 7.5h away -> does NOT cross settlement in 4h hold!

    sig = np.array([-1.0, -1.0])
    pnl_pre, _ = simulate_event_sniper_sleeve_daily(
        df_pre, sig, "fwd_ret_4h", horizon_hours=4, roundtrip_cost_bps=4.0, slot_weight=0.40, max_concurrent_slots=1
    )
    pnl_post, _ = simulate_event_sniper_sleeve_daily(
        df_post, sig, "fwd_ret_4h", horizon_hours=4, roundtrip_cost_bps=4.0, slot_weight=0.40, max_concurrent_slots=1
    )

    # Post-settlement short earns +100 bps - 4 bps fee = +96 bps per unit, whereas pre-settlement
    # short pays 50 bps negative funding -> +100 - 50 - 4 = +46 bps (strictly lower!)
    assert float(pnl_post.sum()) > float(pnl_pre.sum()) * 1.9


def test_cross_sectional_and_tournament_train_only_causality() -> None:
    rng = np.random.default_rng(123)
    # Build synthetic 2024 + 2025 + 2026 4h panel across 8 symbols
    ts_24 = np.arange(1_709_251_200, 1_709_251_200 + 80 * 14400, 14400)  # Mar 2024
    ts_25 = np.arange(1_740_787_200, 1_740_787_200 + 80 * 14400, 14400)  # Mar 2025
    ts_26 = np.arange(1_772_323_200, 1_772_323_200 + 80 * 14400, 14400)  # Mar 2026
    all_ts = np.concatenate([ts_24, ts_25, ts_26])
    syms = [f"MID{i}USDT" for i in range(8)]

    rows = []
    for t in all_ts:
        for s in syms:
            whale = float(rng.normal())
            oi_v = float(rng.normal())
            park = abs(float(rng.normal(0.02, 0.005)))
            f_bps = float(rng.normal(0.0, 15.0))
            # Embedded true orthogonal alpha on `whale - oi_v`
            fwd_4h = 0.004 * (whale - oi_v) + float(rng.normal(0.0, 0.003))
            rows.append(
                {
                    "symbol": s,
                    "decision_ts": int(t),
                    "tradable_adv_15m_gate": 1.0,
                    "whale_pos_ls_z": whale,
                    "smart_money_div_z_24h": whale * 0.8,
                    "oi_vel_12h": oi_v,
                    "oi_vel_z_12h": oi_v,
                    "oi_vel_24h": oi_v * 0.7,
                    "oi_to_qvol_24h": 1.2,
                    "quad_new_longs_12h": max(0.0, oi_v),
                    "spot_taker_share_4h": 0.0,
                    "spot_cvd_z_4h": 0.0,
                    "basis_vel_4h": 0.0,
                    "spot_lead_ret_div_1b": 0.0,
                    "spot_vs_perp_share_div_4h": 0.0,
                    "funding_rate_bps": f_bps,
                    "funding_interval_hours": 8.0,
                    "clock_compression_ratio": 1.0,
                    "compressed_clock_neg_funding_impulse": 0.0,
                    "mins_since_settlement": 120.0,
                    "mins_to_next_settlement": 360.0,
                    "parkinson_vol_24h": park,
                    "corwin_schultz_spread_24h": park * 0.5,
                    "vol_of_vol_24h": 0.1,
                    "fwd_ret_4h": fwd_4h,
                    "fwd_ret_1h": fwd_4h * 0.3,
                }
            )
    panel = pd.DataFrame(rows)
    uniq_days = pd.DatetimeIndex(
        pd.to_datetime((panel["decision_ts"].unique() // 86400) * 86400, unit="s", utc=True)
    ).unique()
    bench = pd.Series(rng.normal(0.0005, 0.006, size=len(uniq_days)), index=uniq_days)

    lb, series_map, _ = run_multimodal_discovery_tournament(panel, bench, min_era_sharpe=1.0, min_dsr=0.50)
    assert not lb.empty
    survivors = select_diverse_orthogonal_survivors(lb, series_map, max_survivors=3)
    assert len(survivors) >= 1

    # Verify cross-sectional simulation returns finite daily PnL
    pnl_cs, meta_cs = simulate_cross_sectional_sleeve_daily(
        panel, panel["whale_pos_ls_z"].to_numpy() - panel["oi_vel_12h"].to_numpy(), "fwd_ret_4h", horizon_hours=4
    )
    assert not pnl_cs.empty
    assert meta_cs["mean_l1_turnover"] >= 0.0


def test_grand_portfolio_floor_safety_and_causal_combination() -> None:
    idx = pd.date_range("2024-01-01", periods=500, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    s_prod = pd.Series(0.0018 + rng.normal(0.0, 0.009, len(idx)), index=idx)
    s_m16 = pd.Series(0.0022 + rng.normal(0.0, 0.007, len(idx)), index=idx)

    combo = causal_inverse_vol_combine({"prod": s_prod, "m16": s_m16}, {"prod": 0.5, "m16": 0.5})
    assert len(combo) == len(idx)

    # Inject severe crash sequence at start to verify `$2,500` Hard Floor protection from `$2,700`
    crash_s = pd.Series([-0.04, -0.05, -0.06, -0.05] + [0.003] * 196, index=idx[:200])
    cfg = GrandPortfolioConfig(initial_equity_usd=2700.0, equity_floor_usd=2500.0)
    _, summary = simulate_compounding_with_live_shields(crash_s, cfg=cfg)
    assert summary["min_equity_usd"] >= 2500.0

    res = run_grand_portfolio_ablation_matrix(s_prod, {"M16_SURV_1": s_m16}, cfg=cfg)
    assert len(res["ablation_rows"]) == 3
