import numpy as np
import pandas as pd

from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)
from src.models.basis_and_carry_harvester_lab import (
    build_and_evaluate_continuous_carry_sleeves,
    compute_causal_predicted_funding_bps,
    evaluate_portfolio_carry_integrations,
    run_issue64_coupon_harvester_1m_study,
)


def test_causal_funding_prediction_and_carry_harvester_suite():
    # 1. Causal funding prediction test
    est = compute_causal_predicted_funding_bps(np.array([-25.0, 20.0]), np.array([-30.0, 28.0]))
    assert est[0] < -20.0 and est[1] > 18.0

    # 2. End-to-end synthetic universe test for Parts E1.1, E1.2, and E1.3
    rng = np.random.default_rng(77)
    n_1h = 2800  # ~116 days across late 2025 -> early 2026 so mask_2026 is non-empty!
    t0 = int(pd.Timestamp("2025-10-15", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym, fnd_by_sym, prem_1h_by_sym, map_1m = {}, {}, {}, {}
    syms = list(PROD_TOP_15[:6])

    for i_s, sym in enumerate(syms):
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.006, size=n_1h))
        vol_usd = 120_000_000.0 / 24.0
        df_h = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "open": np.roll(px, 1),
                "high": px * 1.008,
                "low": px * 0.992,
                "close": px,
                "volume": vol_usd / px,
                "quote_volume": np.full(n_1h, vol_usd),
                "taker_buy_quote_volume": np.full(n_1h, 0.54 * vol_usd),
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h

        # Give sym 1 extreme negative funding (-28 bps) & Spot > Perp (-35 bps), sym 2 frothy positive (+26 bps)
        fr_v = -0.0028 if i_s == 1 else (0.0026 if i_s == 2 else 0.0002)
        pr_v = -35.0 if i_s == 1 else (25.0 if i_s == 2 else 3.0)
        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), fr_v)})
        prem_1h_by_sym[sym] = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "symbol": sym,
                "p_open_bps": pr_v,
                "p_high_bps": pr_v + 2.0,
                "p_low_bps": pr_v - 2.0,
                "p_close_bps": pr_v,
            }
        )

        # Compact 1m series for test speed: last 450 hours in 1m + 1h fallback
        sub_h = 450
        ts_1m = np.arange(ts_1h[-sub_h], ts_1h[-1] + 3600, 60, dtype=np.int64)
        px_1m = np.repeat(px[-sub_h:], 60)
        map_1m[sym] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": px_1m,
                "high": px_1m * 1.002,
                "low": px_1m * 0.998,
                "close": px_1m,
                "quote_volume": np.full(len(ts_1m), vol_usd / 60.0),
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.54 * vol_usd / 60.0),
            }
        )

    lb_h, tr_h = run_issue64_coupon_harvester_1m_study(
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        prem_1h_by_sym=prem_1h_by_sym,
        symbol_1m_loader=lambda s: map_1m.get(s),
        eval_start="2025-12-25",
    )
    assert not tr_h.empty and not lb_h.empty

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-12-20",
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        loader_workers=1,
    )
    d_zero = pd.Series(0.0, index=bundle["eval_idx"]).resample("1D").sum()
    lb_c, sleeves = build_and_evaluate_continuous_carry_sleeves(
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        prem_1h_by_sym=prem_1h_by_sym,
        eval_idx=bundle["eval_idx"],
        core_p1_daily=d_zero,
        manga_p2_daily=d_zero,
    )
    assert len(lb_c) == 4

    lb_port = evaluate_portfolio_carry_integrations(
        bundle=bundle,
        p2_inv_vol_mat=bundle["p2_w_mat"],
        sleeves_out=sleeves,
        coupon_trades_df=tr_h,
    )
    assert len(lb_port) >= 6
