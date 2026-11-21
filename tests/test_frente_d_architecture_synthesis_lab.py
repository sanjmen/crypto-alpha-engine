import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    CoreRebalanceBarrierSpec,
    simulate_core_barrier_1m_portfolio,
)
from src.backtest.frente_d_architecture_synthesis_lab import (
    FRENTE_D_SPECS,
    apply_causal_beta_neutralization,
    evaluate_frente_d_suite,
    simulate_frente_d_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)


def test_beta_projection_and_frente_d_parity_and_suite():
    # 1. Exact Asymmetric Zero-Beta Scaling (`D06`) + Pod `BTCUSDT` Beta-Overlay Clamp (`D05`)
    w_in = np.array([0.0, 0.30, 0.20, -0.25, -0.25], dtype=np.float64)
    beta_in = np.array([1.00, 1.80, 1.50, 0.85, 0.95], dtype=np.float64)
    w_zero_b = apply_causal_beta_neutralization(w_in, beta_in, btc_idx=0, max_beta_ratio=0.0)
    assert abs(float(np.sum(np.abs(w_zero_b))) - 1.0) < 1e-9
    assert abs(float(np.sum(w_zero_b * beta_in))) < 1e-9  # Exact 0.000000000 beta!

    w_clamp_b = apply_causal_beta_neutralization(w_in, beta_in, btc_idx=0, max_beta_ratio=0.12)
    assert abs(float(np.sum(w_clamp_b * beta_in))) <= 0.12 * float(np.sum(np.abs(w_clamp_b))) + 1e-6

    # 2. Exact `rtol=1e-10` parity against `simulate_core_barrier_1m_portfolio` on `D01` + full suite
    rng = np.random.default_rng(41)
    n_1h = 340
    t0 = int(pd.Timestamp("2025-11-01", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)
    bars_1h_by_sym, fnd_by_sym, map_1m = {}, {}, {}
    syms = list(PROD_TOP_15)

    for i_s, sym in enumerate(syms):
        px = 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.008, size=n_1h))
        vol_mult = (20 - i_s) * 2e6
        df_h = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "open": np.roll(px, 1),
                "high": px * 1.012,
                "low": px * 0.988,
                "close": px,
                "volume": vol_mult / px,
                "quote_volume": np.full(n_1h, vol_mult),
                "taker_buy_quote_volume": np.full(n_1h, 0.53 * vol_mult),
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h
        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), -0.0006)})

        ts_1m = np.arange(t0, t0 + n_1h * 3600, 60, dtype=np.int64)
        px_1m = np.repeat(px, 60)
        map_1m[sym] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": px_1m,
                "high": px_1m * 1.004,
                "low": px_1m * 0.996,
                "close": px_1m,
                "quote_volume": np.full(len(ts_1m), vol_mult / 60.0),
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.56 * vol_mult / 60.0),
            }
        )

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-11-09",
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        loader_workers=1,
    )

    ref_sim = simulate_core_barrier_1m_portfolio(
        mats=bundle["mats"],
        o_1m_core=bundle["o_1m_core"],
        h_1m_core=bundle["h_1m_core"],
        l_1m_core=bundle["l_1m_core"],
        eval_idx=bundle["eval_idx"],
        open_core=bundle["open_core"],
        p2_w_mat=bundle["p2_w_mat"],
        spec=CoreRebalanceBarrierSpec("STRICT_8H", "Strict 8h", intracycle_mode="off"),
        resolved_sniper_events_by_hour={},
        allowed_sniper_strats=set(),
    )
    d01_sim = simulate_frente_d_1m_portfolio(
        mats=bundle["mats"],
        o_1m_core=bundle["o_1m_core"],
        h_1m_core=bundle["h_1m_core"],
        l_1m_core=bundle["l_1m_core"],
        eval_idx=bundle["eval_idx"],
        open_core=bundle["open_core"],
        p2_w_mat=bundle["p2_w_mat"],
        spec=FRENTE_D_SPECS[1],
        resolved_sniper_events_by_hour={},
        allowed_sniper_strats=set(),
    )
    assert np.allclose(ref_sim["eq_1h"].to_numpy(), d01_sim["eq_1h"].to_numpy(), rtol=1e-10, atol=1e-8)

    lb_df, _m04_d = evaluate_frente_d_suite(bundle)
    assert len(lb_df) == len(FRENTE_D_SPECS)
