"""
Unit Tests for Unified 1-Minute Portfolio Integration (`src/backtest/p2_portfolio_integration.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.p2_portfolio_integration import (
    build_core_1m_tensors_and_base_snipers,
    convert_trade_df_to_hourly_sniper_events,
    identify_required_1m_symbols_for_portfolio,
    run_p2_portfolio_ablation_1m,
)


def test_convert_trade_df_to_hourly_sniper_events_exact_linear_sum() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "ENAUSDT",
                "entry_ts": 1_740_000_300,
                "exit_ts": 1_740_007_800,  # Spans 3 hourly buckets
                "net_bps": 145.6,
                "fee_bps": 12.7,
                "strength": 88.0,
            }
        ]
    )
    evs = convert_trade_df_to_hourly_sniper_events(df, "Sniper:P6A")
    assert len(evs) == 1
    ev = evs[0]
    assert ev["was_loss"] is False
    assert abs(sum(ev["hourly_ret"].values()) - (145.6 * 1e-4)) < 1e-10


def test_end_to_end_portfolio_integration_pipeline() -> None:
    rng = np.random.default_rng(42)
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    n_1h = 520
    t0 = int(pd.Timestamp("2025-12-15", tz="UTC").timestamp())
    ts_1h = np.arange(t0, t0 + n_1h * 3600, 3600, dtype=np.int64)

    bars_1h: dict[str, pd.DataFrame] = {}
    fnd_map: dict[str, pd.DataFrame] = {}
    core_1m: dict[str, pd.DataFrame] = {}

    for idx_s, s in enumerate(syms):
        px = 100.0 * np.exp(np.cumsum(rng.normal(0.0002 * (idx_s + 1), 0.006, size=n_1h)))
        bars_1h[s] = pd.DataFrame(
            {
                "timestamp": ts_1h,
                "symbol": s,
                "open": px * 0.999,
                "high": px * 1.004,
                "low": px * 0.996,
                "close": px,
                "volume": np.full(n_1h, 1000.0),
                "quote_volume": np.full(n_1h, 25_000_000.0),
                "taker_buy_quote_volume": np.full(n_1h, 13_500_000.0),
            }
        )
        fnd_map[s] = pd.DataFrame(
            {
                "timestamp": ts_1h[::8],
                "symbol": s,
                "funding_rate": np.full(len(ts_1h[::8]), -0.0005 if s == "DOGEUSDT" else 0.0001),
            }
        )
        # Compact 1m grid over last 400 hours
        ts_1m = np.arange(ts_1h[100], ts_1h[-1] + 3600, 60, dtype=np.int64)
        p1m = np.interp(ts_1m, ts_1h, px)
        core_1m[s] = pd.DataFrame(
            {
                "timestamp": ts_1m,
                "open": p1m,
                "high": p1m * 1.001,
                "low": p1m * 0.999,
                "close": p1m,
                "quote_volume": np.full(len(ts_1m), 400_000.0),
                "taker_buy_quote_volume": np.full(len(ts_1m), 220_000.0),
            }
        )

    req_syms = identify_required_1m_symbols_for_portfolio(bars_1h, fnd_map, eval_start="2025-12-20")
    assert set(syms).issubset(set(req_syms))

    bundle = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=core_1m,
        bars_1h_by_sym=bars_1h,
        funding_by_sym=fnd_map,
        eval_start="2025-12-20",
    )
    bundle_stream = build_core_1m_tensors_and_base_snipers(
        core_1m_by_sym=None,
        bars_1h_by_sym=bars_1h,
        funding_by_sym=fnd_map,
        eval_start="2025-12-20",
        symbol_1m_loader=lambda s_k: core_1m.get(s_k),
        symbols_to_stream=req_syms,
        loader_workers=2,
    )
    assert np.allclose(bundle["o_1m_core"], bundle_stream["o_1m_core"], equal_nan=True)
    assert np.allclose(bundle["p2_w_mat"], bundle_stream["p2_w_mat"], equal_nan=True)
    assert len(bundle["base_sniper_events"]) == len(bundle_stream["base_sniper_events"])
    cand_df = pd.DataFrame(
        [
            {
                "symbol": "ENAUSDT",
                "entry_ts": int(ts_1h[300]) + 300,
                "exit_ts": int(ts_1h[302]) + 600,
                "net_bps": 220.0,
                "fee_bps": 12.7,
                "strength": 95.0,
            }
        ]
    )
    rows = run_p2_portfolio_ablation_1m(
        bundle=bundle,
        candidate_sleeves={"P6A_TEST": cand_df},
        initial_capital=2700.0,
    )
    assert len(rows) == 2
    assert rows[1]["new_sleeve_trades_executed"] == 1
    assert rows[1]["final_eq_usd"] > rows[0]["final_eq_usd"]
