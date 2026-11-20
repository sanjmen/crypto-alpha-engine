import numpy as np
import pandas as pd

from src.models.liquidation_absorption_p4_lab import (
    compute_liquidation_exhaustion_1ms_metrics,
    evaluate_p4_m4_absorption_variants,
)


def test_liquidation_exhaustion_1ms_metrics_strict_causality():
    entry_ts = 1_750_000_200
    end_ms = entry_ts * 1000 - 1

    # Prior 14m: massive 25-maker forced sell cascade ($80k at 1ms)
    # Final 1m: sell sweeps exhausted ($0), whale buy sweep ($45k, 14 makers) while retail sells ($400, 1 maker)
    # Future post-entry tick (`entry_ts * 1000 + 50`): massive $500k sell that MUST be ignored (`0.000 ms` lookahead!)
    ticks = pd.DataFrame(
        [
            {"ts_ms": end_ms - 300_000, "price": 10.2, "usd": 80_000.0, "makers": 25, "is_buy": False},
            {"ts_ms": end_ms - 240_000, "price": 10.1, "usd": 30_000.0, "makers": 14, "is_buy": False},
            *[
                {"ts_ms": end_ms - 180_000 + i * 500, "price": 10.0, "usd": 500.0, "makers": 1, "is_buy": False}
                for i in range(10)
            ],
            {"ts_ms": end_ms - 20_000, "price": 9.95, "usd": 45_000.0, "makers": 14, "is_buy": True},
            {"ts_ms": end_ms - 10_000, "price": 9.96, "usd": 400.0, "makers": 1, "is_buy": False},
            {"ts_ms": entry_ts * 1000 + 50, "price": 9.50, "usd": 500_000.0, "makers": 90, "is_buy": False},
        ]
    )

    m = compute_liquidation_exhaustion_1ms_metrics(ticks, entry_ts)
    assert m["has_agg"] == 1.0
    assert m["max_sell_makers_15m"] == 25.0  # Post-entry 90-maker sell strictly excluded!
    assert m["sell_exhaust_ratio_1m_vs_peak3m"] == 0.0
    assert m["whale_buy_share_final_1m"] > 0.95
    assert m["smart_vs_retail_final_1m"] > 0.80


def test_evaluate_p4_m4_absorption_variants_filters_knives():
    rng = np.random.default_rng(19)
    rows_tr = []
    rows_m = []
    base_ts = int(pd.Timestamp("2024-02-01", tz="UTC").timestamp())

    for k in range(60):
        en_ts = base_ts + k * 14 * 86400
        dt_str = pd.Timestamp(en_ts, unit="s", tz="UTC")
        era = "2023_2024" if dt_str.year <= 2024 else ("2025" if dt_str.year == 2025 else "2026")
        is_true_exhaust = (k % 2) == 0
        net = float(rng.normal(95.0, 25.0)) if is_true_exhaust else float(rng.normal(-110.0, 25.0))
        rows_tr.append(
            {
                "symbol": "SOLUSDT",
                "mech_id": "P4_M4_ONCHAIN_ANCHORED_LIQ_SNAP_LONG",
                "pillar": "",
                "exec_mode": "MONO_C_SPLIT_50_50",
                "entry_ts": en_ts,
                "exit_ts": en_ts + 1800,
                "era": era,
                "strength": 40.0 + k * 0.2,
                "gross_bps": net + 12.0,
                "funding_bps": 0.0,
                "fee_bps": 12.0,
                "net_bps": net,
                "mfe_bps": max(20.0, net + 30.0),
                "mae_bps": -40.0,
                "hold_mins": 30,
                "exit_reason": "TP",
            }
        )
        rows_m.append(
            {
                "symbol": "SOLUSDT",
                "entry_ts": en_ts,
                "has_agg": 1.0,
                "max_sell_makers_15m": 24.0 if is_true_exhaust else 4.0,
                "sell_sweep_conc_prior_14m": 0.42 if is_true_exhaust else 0.10,
                "sell_exhaust_ratio_1m_vs_peak3m": 0.15 if is_true_exhaust else 1.40,
                "whale_buy_share_final_1m": 0.72 if is_true_exhaust else 0.30,
                "whale_buy_share_final_3m": 0.68 if is_true_exhaust else 0.35,
                "retail_buy_share_final_1m": 0.40 if is_true_exhaust else 0.65,
                "smart_vs_retail_final_1m": 0.32 if is_true_exhaust else -0.35,
                "buy_sweep_conc_final_1m": 0.45 if is_true_exhaust else 0.08,
            }
        )

    lb, _ = evaluate_p4_m4_absorption_variants(pd.DataFrame(rows_tr), pd.DataFrame(rows_m))
    assert not lb.empty
    best = lb.iloc[0]
    assert best["mech_id"] != "P4_M4_0_RAW_CONTROL"
    assert bool(best["is_treasure"]) is True
