import numpy as np
import pandas as pd

from src.models.maker_wick_catcher_lab import (
    MakerWickProfile,
    simulate_maker_wick_on_tick_window,
)


def test_maker_wick_catcher_strict_gtx_and_queue_tradethrough():
    t0_sec = 1_750_010_000
    rows = []
    # 1. Calm pre-cascade baseline around $100.00 for 120 seconds
    for s in range(120):
        rows.append({"ts_ms": (t0_sec + s) * 1000, "price": 100.0, "usd": 5_000.0, "makers": 2, "is_buy": True})

    # 2. Sudden -2.2% forced sell cascade at t = t0 + 125s (`$97.80`, 22 makers, $60k sell!)
    t_burst_ms = (t0_sec + 125) * 1000
    rows.append({"ts_ms": t_burst_ms, "price": 97.80, "usd": 60_000.0, "makers": 22, "is_buy": False})
    # Price at +250ms (`t_order_ms`) is still $97.80 -> GTX limit at -45 bps = $97.3599
    rows.append({"ts_ms": t_burst_ms + 260, "price": 97.80, "usd": 2_000.0, "makers": 1, "is_buy": True})

    # 3. Exhaustion tail dips strictly through $97.3599 down to $97.25 (`+4s` later), filling our resting GTX Maker!
    rows.append({"ts_ms": t_burst_ms + 4_000, "price": 97.25, "usd": 35_000.0, "makers": 12, "is_buy": False})

    # 4. Elastic snap-back to $98.35 (`+1.0%` above $97.36 > +85 bps Maker TP!) 30 seconds later
    for s_post in range(6, 60):
        px_s = 97.40 + (s_post - 6) * 0.025
        rows.append(
            {
                "ts_ms": t_burst_ms + s_post * 1000,
                "price": min(px_s, 98.50),
                "usd": 8_000.0,
                "makers": 3,
                "is_buy": True,
            }
        )
    # Pad trailing 150s so window bounds are satisfied
    for s_tail in range(60, 220):
        rows.append({"ts_ms": t_burst_ms + s_tail * 1000, "price": 98.50, "usd": 3_000.0, "makers": 1, "is_buy": True})

    prof = (
        MakerWickProfile(
            profile_id="TEST_MW_45",
            rung1_offset_bps=45.0,
            rung2_offset_bps=None,
            ttl_sec=120,
            tp_bps=85.0,
            sl_bps=150.0,
            max_hold_sec=600,
        ),
    )
    trades = simulate_maker_wick_on_tick_window(
        symbol="SOLUSDT",
        ticks_df=pd.DataFrame(rows),
        anchor_start_ms=t_burst_ms - 5_000,
        anchor_end_ms=t_burst_ms + 5_000,
        profiles=prof,
        latency_ms=250,
    )
    assert len(trades) == 1
    tr = trades[0]
    assert tr["exit_reason"] == "TP_MAKER"
    assert np.isclose(tr["fee_bps"], 4.0)  # 2.0 bps Maker Entry + 2.0 bps Maker TP!
    assert np.isclose(tr["net_bps"], 81.0, atol=0.2)
