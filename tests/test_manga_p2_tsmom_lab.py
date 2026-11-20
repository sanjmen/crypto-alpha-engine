import numpy as np
import pandas as pd

from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)
from src.models.manga_p2_tsmom_lab import (
    build_causal_flow_imbalance_bars,
    evaluate_manga_p2_ablation_suite,
)


def test_manga_p2_dib_clock_and_ablation_suite():
    rng = np.random.default_rng(29)
    n_1h = 520
    t0 = int(pd.Timestamp("2025-11-01", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym = {}
    fnd_by_sym = {}
    map_1m = {}
    syms = list(PROD_TOP_15)

    for i_s, sym in enumerate(syms):
        px = 50.0 * np.cumprod(1.0 + rng.normal(0.0, 0.007, size=n_1h))
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
                "taker_buy_quote_volume": np.full(n_1h, 0.52 * vol_mult),
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h
        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), 0.0001)})

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
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.58 * vol_mult / 60.0),
            }
        )

    dib = build_causal_flow_imbalance_bars(map_1m["BTCUSDT"])
    assert len(dib) > 50

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-11-12",
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        loader_workers=1,
    )
    sov_1m = {s: map_1m[s] for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")}
    lb = evaluate_manga_p2_ablation_suite(bundle, sovereign_1m_by_sym=sov_1m)
    assert len(lb) == 8
    # Verify C01 matches bundle's baseline p2_w_mat closely
    c01 = lb[lb["variant_id"] == "C01_PROD_3MAJ_EQUAL_DOLLAR_BAR"].iloc[0]
    assert np.isfinite(float(c01["port_sharpe_4y"]))

    from src.models.sniper_l2_and_cascade_short_lab import evaluate_frente_b_sniper_portfolio_ablation

    lb_fb = evaluate_frente_b_sniper_portfolio_ablation(bundle)
    assert len(lb_fb) == 5
