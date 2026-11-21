import numpy as np
import pandas as pd

from src.backtest.frente_e2_seven_factor_allocator_lab import (
    FRENTE_E2_SPECS,
    evaluate_frente_e2_suite,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)


def test_frente_e2_standalone_factors_and_lofo_suite():
    rng = np.random.default_rng(91)
    n_1h = 900
    t0 = int(pd.Timestamp("2025-12-05", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym, fnd_by_sym, prem_1h_by_sym, map_1m = {}, {}, {}, {}
    syms = list(PROD_TOP_15[:8])

    for i_s, sym in enumerate(syms):
        px = 80.0 * np.cumprod(1.0 + rng.normal(0.0001, 0.007, size=n_1h))
        vol_usd = (20 - i_s) * 2e6
        df_h = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "open": np.roll(px, 1),
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "volume": vol_usd / px,
                "quote_volume": np.full(n_1h, vol_usd),
                "taker_buy_quote_volume": np.full(n_1h, 0.52 * vol_usd),
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h
        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), 0.00015)})
        prem_1h_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h, "p_close_bps": np.full(n_1h, 4.0)})

        sub_h = 300
        ts_1m = np.arange(ts_1h[-sub_h], ts_1h[-1] + 3600, 60, dtype=np.int64)
        px_1m = np.repeat(px[-sub_h:], 60)
        map_1m[sym] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": px_1m,
                "high": px_1m * 1.003,
                "low": px_1m * 0.997,
                "close": px_1m,
                "quote_volume": np.full(len(ts_1m), vol_usd / 60.0),
                "taker_buy_quote_volume": np.full(len(ts_1m), 0.53 * vol_usd / 60.0),
            }
        )

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-12-30",
        symbol_1m_loader=lambda s: map_1m.get(s),
        symbols_to_stream=syms,
        loader_workers=1,
    )
    solo_df, lofo_df = evaluate_frente_e2_suite(
        bundle=bundle,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        prem_1h_by_sym=prem_1h_by_sym,
    )
    assert len(solo_df) == 7
    assert len(lofo_df) == len(FRENTE_E2_SPECS)
