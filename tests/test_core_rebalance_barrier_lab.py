import numpy as np
import pandas as pd

from src.backtest.core_rebalance_barrier_lab import (
    CoreRebalanceBarrierSpec,
    run_core_rebalance_barrier_ablation,
)
from src.backtest.p2_portfolio_integration import (
    PROD_TOP_15,
    build_core_1m_tensors_and_base_snipers,
)


def test_core_rebalance_barrier_ablation_pipeline():
    rng = np.random.default_rng(7)
    n_1h = 520
    t0 = int(pd.Timestamp("2025-11-01", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h_by_sym = {}
    bars_1h_top15 = {}
    fnd_by_sym = {}
    fnd_top15 = {}
    core_1m_by_sym = {}

    for i_s, sym in enumerate(PROD_TOP_15[:8]):
        drift = (i_s - 3.5) * 0.00018
        rets = drift + rng.normal(0.0, 0.008, size=n_1h)
        px = 50.0 * np.cumprod(1.0 + rets)
        df_h = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "open": np.roll(px, 1),
                "high": px * 1.012,
                "low": px * 0.988,
                "close": px,
                "volume": np.full(n_1h, 1e5),
                "quote_volume": px * 1e5,
                "taker_buy_quote_volume": px * 5.2e4,
            }
        )
        df_h.loc[0, "open"] = px[0]
        bars_1h_by_sym[sym] = df_h

        b_idx = pd.DatetimeIndex(pd.to_datetime(ts_1h, unit="s", utc=True))
        d_c = df_h.drop(columns=["timestamp"]).copy()
        d_c.index = b_idx
        bars_1h_top15[sym] = d_c

        fnd_by_sym[sym] = pd.DataFrame({"timestamp": ts_1h[::8], "funding_rate": np.full(len(ts_1h[::8]), 0.0001)})
        fnd_top15[sym] = pd.Series(0.0001, index=b_idx[::8])

        ts_1m = np.arange(t0, t0 + n_1h * 3600, 60, dtype=np.int64)
        px_1m = np.repeat(px, 60)
        core_1m_by_sym[sym] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": px_1m,
                "high": px_1m * 1.004,
                "low": px_1m * 0.996,
                "close": px_1m,
                "quote_volume": np.full(len(ts_1m), 2e4),
                "taker_buy_quote_volume": np.full(len(ts_1m), 1e4),
            }
        )

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=core_1m_by_sym,
        bars_1h_by_sym=bars_1h_by_sym,
        funding_by_sym=fnd_by_sym,
        eval_start="2025-11-12",
    )

    specs = (
        CoreRebalanceBarrierSpec("LIVE_RERANK", "Live full rerank", intracycle_mode="full_rerank"),
        CoreRebalanceBarrierSpec(
            "SHIELD",
            "Collapsed trim + buffer",
            intracycle_mode="collapsed_trim_only",
            bhs_hold_top_k=7,
            deadband_mult=1.75,
            min_partial_trim_equity_pct=0.035,
            fast_min_tstat=0.50,
        ),
    )
    lb = run_core_rebalance_barrier_ablation(
        bundle=bundle,
        bars_1h_top15=bars_1h_top15,
        funding_1h_top15=fnd_top15,
        specs=specs,
    )
    assert len(lb) == 2
    assert lb["core_turnover_1x"].iloc[1] <= lb["core_turnover_1x"].iloc[0] + 1e-6


def test_mainnet_feature_flag_disables_intracycle_rebalance_by_default():
    from src.execution.feature_flags import FeatureFlags

    ff_mainnet = FeatureFlags.from_env_and_args(args=None, mode="mainnet", environ={})
    assert ff_mainnet.enable_intracycle_rebalance is False

    ff_shadow = FeatureFlags.from_env_and_args(args=None, mode="shadow", environ={})
    assert ff_shadow.enable_intracycle_rebalance is True
