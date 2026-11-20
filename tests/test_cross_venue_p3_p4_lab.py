"""
Unit Tests for Phase P3/P4 Cross-Venue & On-Chain Microstructure Lab (`src/models/cross_venue_p3_p4_lab.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cross_venue_p3_ingestor import _normalize_settlement_pts_to_bps_per_h
from src.models.cross_venue_p3_p4_lab import (
    evaluate_symbol_p3_p4_mechanisms,
    prepare_symbol_p3_p4_state,
)


def test_normalize_settlement_pts_to_hourly_carry_bps() -> None:
    t0 = 1_750_000_000
    pts = [(t0 + k * 4 * 3600, -0.0020) for k in range(8)]  # -20 bps every 4h -> -5.0 bps/h
    df = _normalize_settlement_pts_to_bps_per_h(pts, "ENAUSDT", "BYBIT")
    assert df is not None
    assert abs(float(df["peer_fr_bps_per_h"].iloc[-1]) - (-5.0)) < 1e-4


def test_p4_m3_onchain_hyperliquid_oracle_lead_triggers_cleanly() -> None:
    n = 22_000
    t0 = 1_750_000_000
    ts = np.arange(t0, t0 + n * 60, 60, dtype=np.int64)
    c = np.full(n, 10.0, dtype=float)
    h = np.full(n, 10.02, dtype=float)
    low = np.full(n, 9.98, dtype=float)
    qv = np.full(n, 30_000.0, dtype=float)  # $43.2M/day >= $15M & >= $24k/min participation guard

    ix_trig = 21_500
    c[ix_trig] = 10.12
    h[ix_trig] = 10.14
    qv[ix_trig - 4 : ix_trig + 1] = 120_000.0  # RVOL_5m = 4.0x
    c[ix_trig + 1 : ix_trig + 160] = np.linspace(10.13, 10.65, 159)
    h[ix_trig + 1 : ix_trig + 160] = c[ix_trig + 1 : ix_trig + 160] * 1.003
    low[ix_trig + 1 : ix_trig + 160] = c[ix_trig + 1 : ix_trig + 160] * 0.998

    perp_1m = pd.DataFrame(
        {
            "timestamp": ts,
            "open": c,
            "high": h,
            "low": low,
            "close": c,
            "quote_volume": qv,
            "taker_buy_quote_volume": qv * 0.60,
        }
    )
    fnd_df = pd.DataFrame({"timestamp": ts[::480], "funding_rate": np.full(len(ts[::480]), 0.0001)})
    ts_5m = ts[::5]
    oi_vals = np.full(len(ts_5m), 1_000_000.0)
    oi_vals[(ix_trig // 5) - 2 : (ix_trig // 5) + 5] = 1_250_000.0
    met_5m = pd.DataFrame({"timestamp": ts_5m, "sum_open_interest": oi_vals})
    hl_df = pd.DataFrame(
        {
            "timestamp": ts[::60],
            "symbol": "ENAUSDT",
            "hl_fr_bps_per_h": np.full(len(ts[::60]), 2.6),
            "hl_premium_bps": np.full(len(ts[::60]), 16.0),
        }
    )
    book_df = pd.DataFrame(
        {
            "timestamp": [int(ts[ix_trig])],
            "symbol": ["ENAUSDT"],
            "bid_1_usd": [400_000.0],
            "ask_1_usd": [220_000.0],
            "bid_2_usd": [700_000.0],
            "ask_2_usd": [400_000.0],
        }
    )

    st = prepare_symbol_p3_p4_state(
        symbol="ENAUSDT",
        perp_1m=perp_1m,
        prem_1m=None,
        metrics_5m=met_5m,
        funding_df=fnd_df,
        hl_funding_df=hl_df,
        peer_cex_df=None,
    )
    assert st is not None
    trades = evaluate_symbol_p3_p4_mechanisms(st, book_df)
    m3_trades = [t for t in trades if t["mech_id"] == "P4_M3_ONCHAIN_HL_ORACLE_LEAD_LONG"]
    assert len(m3_trades) >= 1
    assert any(float(t["net_bps"]) > 100.0 for t in m3_trades)
