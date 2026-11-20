import numpy as np
import pandas as pd

from src.backtest.core_universe_factor_lab import (
    CANDIDATE_POOL_35,
    UniverseFactorExperimentSpec,
    build_pool35_1m_bundle,
    compute_causal_pit_universe_mask,
    compute_dynamic_pit_stitched_matrices,
    evaluate_universe_and_factor_experiments,
)
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine


def test_causal_pit_universe_mask_and_experiment_pipeline():
    rng = np.random.default_rng(11)
    n_1h = 520
    t0 = int(pd.Timestamp("2025-11-01", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym = {}
    fnd_by_sym = {}
    map_1m = {}
    syms = list(CANDIDATE_POOL_35[:18])

    for i_s, sym in enumerate(syms):
        px = 40.0 * np.cumprod(1.0 + rng.normal(0.0, 0.007, size=n_1h))
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
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.5 * vol_mult / 60.0),
            }
        )

    idx_h = pd.DatetimeIndex(pd.to_datetime(ts_1h, unit="s", utc=True))
    dv_df = pd.DataFrame({s: bars_1h_by_sym[s]["quote_volume"].to_numpy() for s in syms}, index=idx_h)
    cl_df = pd.DataFrame({s: bars_1h_by_sym[s]["close"].to_numpy() for s in syms}, index=idx_h)
    pit_m = compute_causal_pit_universe_mask(dv_df, cl_df, top_k=10, hold_buffer=2, min_age_hours=168)
    assert int(pit_m.iloc[-1].sum()) == 10
    assert int(pit_m.iloc[50].sum()) == 0  # Strictly masked before 168h warmup!

    bundle = build_pool35_1m_bundle(
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        eval_start="2025-11-12",
        loader_workers=1,
    )

    # Verify zero contamination when unlisted NaN coins exist in pool outside active mask
    active_10 = syms[:10]
    static_mask = pd.DataFrame(False, index=idx_h, columns=syms)
    static_mask[active_10] = True
    b_pool_corrupt = dict(bundle["bars_1h_pool"])
    b_pool_corrupt[syms[-1]] = b_pool_corrupt[syms[-1]].iloc[-50:].copy()  # Late-listed coin (NaN earlier!)
    eng = SevenFactorOnlineEngine(enable_chop_range_arb=True, enable_holistic_regime_shields=True)
    m_direct = eng.compute_matrices({s: b_pool_corrupt[s] for s in active_10}, funding_rates=bundle["fnd_pool"])
    m_stitched = compute_dynamic_pit_stitched_matrices(
        bars_1h_pool=b_pool_corrupt,
        fnd_pool=bundle["fnd_pool"],
        pit_mask=static_mask,
        eng=eng,
    )
    assert np.allclose(
        m_direct["w_base_1x"].to_numpy(),
        m_stitched["w_masked"][active_10].to_numpy(),
        atol=1e-12,
    )

    specs = (
        UniverseFactorExperimentSpec("STATIC", "Static 15", universe_mode="static_15"),
        UniverseFactorExperimentSpec(
            "DYN_15",
            "Dynamic 15",
            universe_mode="dynamic_pit_15",
            fast_min_tstat=0.50,
            pit_min_age_hours=168,
        ),
    )
    lb = evaluate_universe_and_factor_experiments(bundle, specs=specs)
    assert len(lb) == 2
