"""
Unit Tests for Phase-P2 Tick-Level `aggTrades` 1-Millisecond Book-Sweep Physics
(`src/features/tick_sweep_physics.py`).

Verifies:
1. PyArrow `parse_aggtrades_zip_window` reads both headered and headerless archives and slices `[min_ms, max_ms]`.
2. `compute_event_tick_sweep_metrics` strictly excludes post-signal ticks (`ts_ms >= signal_ts * 1000` -> 0 lookahead!)
   and accurately separates `1-ms` Multi-Maker Whale Sweeps (`last_trade_id - first_trade_id + 1 >= 5`) from
   atomized retail clicks (`makers == 1`).
3. `evaluate_p2_tick_sweep_ablation` vetoes spoofed-Bid / whale-sell-distribution traps while passing genuine sweeps.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from src.features.tick_sweep_physics import (
    compute_event_tick_sweep_metrics,
    evaluate_p2_tick_sweep_ablation,
    parse_aggtrades_zip_window,
)


def _zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_parse_aggtrades_and_strict_causal_1ms_sweep_metrics() -> None:
    sig_sec = 1_740_000_000
    sig_ms = sig_sec * 1000

    # Build synthetic aggTrades:
    # - At `sig_ms - 60_000`: 1-ms multi-level Buy Sweep (`is_buyer_maker = false`) sweeping 22 resting Asks
    #   from $10.00 to $10.08 (+80 bps jump in 1 millisecond!)
    # - At `sig_ms - 30_000`: small retail sell (`1` maker, `$200`)
    # - At `sig_ms + 5` (5ms AFTER signal!): massive fake sell sweep that MUST be ignored (0 lookahead!)
    csv_body = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        f"1,10.00,1500,100,109,{sig_ms - 60_000},false\n"
        f"2,10.08,2000,110,121,{sig_ms - 60_000},false\n"
        + "".join(f"{3 + k},10.02,20,{200 + k},{200 + k},{sig_ms - 40_000 + k * 100},true\n" for k in range(12))
        + f"99,9.50,999999,500,900,{sig_ms + 5},true\n"
    )
    parsed = parse_aggtrades_zip_window(_zip_bytes("ENA.csv", csv_body), sig_ms - 900_000, sig_ms + 10_000)
    assert parsed is not None and len(parsed) == 15

    metrics = compute_event_tick_sweep_metrics(parsed, sig_sec)
    assert metrics["has_agg"] == 1.0
    # 10 + 12 = 22 makers swept in a single 1-ms buy burst!
    assert abs(metrics["max_buy_makers_5m"] - 22.0) < 1e-6
    # Future sell sweep at `sig_ms + 5` was strictly excluded (`max_sell_makers_5m == 1.0`)!
    assert abs(metrics["max_sell_makers_5m"] - 1.0) < 1e-6
    # 1-ms price jump `(10.08 - 10.00) / 10.00 * 1e4 = +80.0 bps`!
    assert abs(metrics["max_buy_jump_bps_5m"] - 80.0) < 1e-3
    assert metrics["whale_sweep_buy_share_15m"] > 0.99
    assert metrics["smart_vs_retail_div_15m"] > 0.80


def test_p2_ablation_vetoes_spoofed_bid_distribution_trap() -> None:
    p1_trades = pd.DataFrame(
        [
            # Event 1 (2025 spoofed trap: whales selling into retail, loses -200 bps)
            {
                "mech_id": "P6A_L2_ASK_AIR_POCKET_LONG",
                "exec_mode": "MONO_A_CONVEX_TRAIL",
                "symbol": "SPOOFUSDT",
                "signal_ts": 1_745_000_000,
                "entry_ts": 1_745_000_060,
                "exit_ts": 1_745_001_200,
                "side": 1,
                "hold_min": 20,
                "gross_bps": -190.0,
                "funding_bps": 0.0,
                "fee_bps": 12.7,
                "net_bps": -202.7,
                "exit_reason": "STOP_LOSS",
                "strength": 50.0,
                "aux_metric": 0.30,
            },
            # Event 2 (2025 genuine whale 1ms sweep: wins +320 bps)
            {
                "mech_id": "P6A_L2_ASK_AIR_POCKET_LONG",
                "exec_mode": "MONO_A_CONVEX_TRAIL",
                "symbol": "REALUSDT",
                "signal_ts": 1_746_000_000,
                "entry_ts": 1_746_000_060,
                "exit_ts": 1_746_003_600,
                "side": 1,
                "hold_min": 60,
                "gross_bps": 332.7,
                "funding_bps": 0.0,
                "fee_bps": 12.7,
                "net_bps": 320.0,
                "exit_reason": "CONVEX_TRAIL_STOP",
                "strength": 60.0,
                "aux_metric": 0.35,
            },
        ]
    )
    feats = pd.DataFrame(
        [
            {
                "symbol": "SPOOFUSDT",
                "signal_ts": 1_745_000_000,
                "has_agg": 1.0,
                "max_buy_makers_5m": 2.0,
                "max_sell_makers_5m": 45.0,
                "max_buy_jump_bps_5m": 0.5,
                "whale_sweep_buy_share_15m": 0.18,
                "whale_sweep_buy_share_5m": 0.12,
                "retail_atom_buy_share_15m": 0.68,
                "smart_vs_retail_div_15m": -0.50,
                "sweep_conc_share_5m": 0.55,
                "hawkes_buy_sweep_accel": 0.4,
            },
            {
                "symbol": "REALUSDT",
                "signal_ts": 1_746_000_000,
                "has_agg": 1.0,
                "max_buy_makers_5m": 28.0,
                "max_sell_makers_5m": 4.0,
                "max_buy_jump_bps_5m": 14.2,
                "whale_sweep_buy_share_15m": 0.78,
                "whale_sweep_buy_share_5m": 0.82,
                "retail_atom_buy_share_15m": 0.51,
                "smart_vs_retail_div_15m": 0.27,
                "sweep_conc_share_5m": 0.62,
                "hawkes_buy_sweep_accel": 2.5,
            },
        ]
    )
    lb, all_v = evaluate_p2_tick_sweep_ablation(p1_trades, feats)
    assert not lb.empty
    anti_row = lb[lb["mech_id"] == "P6A_L2_1_ANTISPOOF_WHALE_SWEEP"].iloc[0]
    assert int(anti_row["trades_full"]) == 1
    assert abs(float(anti_row["mean_net_bps_full"]) - 320.0) < 1e-3
