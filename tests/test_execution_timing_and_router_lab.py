import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    CoreRebalanceBarrierSpec,
    simulate_core_barrier_1m_portfolio,
)
from src.backtest.execution_timing_and_router_lab import (
    ExecutionTimingRouterSpec,
    evaluate_execution_timing_and_router_specs,
    simulate_subminute_timed_1m_portfolio,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)


def test_execution_timing_parity_and_pre_post_settlement_splits():
    rng = np.random.default_rng(19)
    n_1h = 520
    t0 = int(pd.Timestamp("2025-11-01", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym = {}
    fnd_by_sym = {}
    map_1m = {}
    syms = list(PROD_TOP_15)

    for i_s, sym in enumerate(syms):
        px = 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.006, size=n_1h))
        vol_mult = (20 - i_s) * 2e6
        df_h = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "open": np.roll(px, 1),
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": vol_mult / px,
                "quote_volume": np.full(n_1h, vol_mult),
                "taker_buy_quote_volume": np.full(n_1h, 0.51 * vol_mult),
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h
        fr_sign = -0.0003 if i_s < 5 else 0.0003
        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), fr_sign)})

        ts_1m = np.arange(t0, t0 + n_1h * 3600, 60, dtype=np.int64)
        px_1m = np.repeat(px, 60)
        map_1m[sym] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": px_1m,
                "high": px_1m * 1.003,
                "low": px_1m * 0.997,
                "close": px_1m,
                "quote_volume": np.full(len(ts_1m), vol_mult / 60.0),
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.5 * vol_mult / 60.0),
            }
        )

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-11-12",
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        loader_workers=1,
    )

    ref_cb = simulate_core_barrier_1m_portfolio(
        mats=bundle["mats"],
        o_1m_core=bundle["o_1m_core"],
        h_1m_core=bundle["h_1m_core"],
        l_1m_core=bundle["l_1m_core"],
        eval_idx=bundle["eval_idx"],
        open_core=bundle["open_core"],
        p2_w_mat=bundle["p2_w_mat"],
        spec=CoreRebalanceBarrierSpec("REF", "Ref", intracycle_mode="off"),
    )
    ref_tm = simulate_subminute_timed_1m_portfolio(
        mats=bundle["mats"],
        o_1m_core=bundle["o_1m_core"],
        h_1m_core=bundle["h_1m_core"],
        l_1m_core=bundle["l_1m_core"],
        eval_idx=bundle["eval_idx"],
        open_core=bundle["open_core"],
        p2_w_mat=bundle["p2_w_mat"],
        spec=ExecutionTimingRouterSpec(
            "A00", "Ref", offset_minutes=0, rebal_fee_bps=4.20, idealized_same_bar_coupon=True
        ),
    )
    assert np.allclose(ref_cb["eq_1h"].to_numpy(), ref_tm["eq_1h"].to_numpy(), rtol=1e-10, atol=1e-8)

    lb = evaluate_execution_timing_and_router_specs(bundle)
    assert len(lb) == 11
    assert set(lb["offset_minutes"].unique()) == {-10, -5, 0, 3, 5, 10}
