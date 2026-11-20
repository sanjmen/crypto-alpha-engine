"""
Unit Tests for Phase-P1 Microstructure Data Ingestor (`src/data/microstructure_p1_ingestor.py`)
and Convex Monetization Laboratory (`src/models/microstructure_monetization_lab.py`).

Verifies:
1. `premiumIndexKlines` and `bookDepth` ZIP parsers (`±0.2%..±5%` Bid/Ask USD wide pivot) + `Hyperliquid` coin mapping.
2. Exact linear accounting (`gross + funding - fee == net`) across all 3 Monetization Geometries:
   - `MONO_A_CONVEX_TRAIL` (rides `+5.0%` squeeze without premature `+70 bps` choke)
   - `MONO_B_COUPON_VACUUM` (banks `:00:00` coupon and exits cleanly at `:01:00` when post-settle stalls)
   - `MONO_C_SPLIT_50_50` (`50%` Maker TP1 @ `+1.8%` + `50%` risk-free runner @ `+9.0%`)
3. Causal L2 `bookDepth` alignment (`t_book <= bar_close_ts`) vetoing heavy Ask walls while passing Ask vacuums.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from src.data.microstructure_p1_ingestor import (
    map_binance_perp_to_hl_coin,
    map_binance_perp_to_okx_inst,
    parse_book_depth_zip_bytes,
    parse_premium_index_zip_bytes,
)
from src.models.microstructure_monetization_lab import (
    P1_MECHANISM_PROFILES,
    evaluate_symbol_p1_with_book_depth,
    prepare_symbol_p1_base_state,
    simulate_1m_monetization_geometries,
)


def _make_zip_csv(filename: str, csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, csv_text)
    return buf.getvalue()


def test_parsers_and_cross_venue_symbol_mapping() -> None:
    assert map_binance_perp_to_hl_coin("1000PEPEUSDT") == "kPEPE"
    assert map_binance_perp_to_hl_coin("ENAUSDT") == "ENA"
    assert map_binance_perp_to_okx_inst("1000BONKUSDT") == "BONK-USDT-SWAP"

    prem_csv = (
        "open_time,open,high,low,close,volume\n"
        "1704067200000,-0.0020,-0.0010,-0.0045,-0.0030,0\n"
        "1704067260000,-0.0030,-0.0015,-0.0050,-0.0025,0\n"
    )
    prem_df = parse_premium_index_zip_bytes(_make_zip_csv("ENAUSDT-1m.csv", prem_csv), "ENAUSDT")
    assert prem_df is not None and len(prem_df) == 2
    assert abs(float(prem_df.loc[0, "p_low_bps"]) - (-45.0)) < 1e-3
    assert abs(float(prem_df.loc[0, "p_close_bps"]) - (-30.0)) < 1e-3

    bd_csv = (
        "timestamp,percentage,depth,notional\n"
        "2025-01-01 08:00:10,-2,10000,250000.0\n"
        "2025-01-01 08:00:10,-1,6000,150000.0\n"
        "2025-01-01 08:00:10,-0.2,2000,50000.0\n"
        "2025-01-01 08:00:10,0.2,1000,25000.0\n"
        "2025-01-01 08:00:10,1,2000,50000.0\n"
        "2025-01-01 08:00:10,2,3000,75000.0\n"
    )
    bd_df = parse_book_depth_zip_bytes(_make_zip_csv("ENAUSDT-bookDepth.csv", bd_csv), "ENAUSDT")
    assert bd_df is not None and len(bd_df) == 1
    assert abs(float(bd_df.loc[0, "bid_1_usd"]) - 150_000.0) < 1.0
    assert abs(float(bd_df.loc[0, "ask_1_usd"]) - 50_000.0) < 1.0
    assert abs(float(bd_df.loc[0, "bid_2_usd"]) - 250_000.0) < 1.0
    assert abs(float(bd_df.loc[0, "ask_2_usd"]) - 75_000.0) < 1.0


def test_three_monetization_geometries_unchoked_convex_and_coupon_bank() -> None:
    # Start 5 minutes before 08:00:00 UTC settlement (`t_0800 = 1_704_096_000`)
    t_0800 = 1_704_096_000
    ts = np.arange(t_0800 - 5 * 60, t_0800 + 25 * 60, 60, dtype=np.int64)
    n = len(ts)
    o = np.full(n, 100.0, dtype=np.float64)
    h = np.full(n, 100.2, dtype=np.float64)
    low = np.full(n, 99.8, dtype=np.float64)
    c = np.full(n, 100.0, dtype=np.float64)
    qv = np.full(n, 500_000.0, dtype=np.float64)
    tb_sh = np.full(n, 0.45, dtype=np.float64)
    bk_imb = np.zeros(n, dtype=np.float64)

    # Bar 2 rises to +2.0% ($102.0) then pulls back to +0.9% ($100.9) on Bar 3 (which USED to stop out at +70 bps!),
    # and then after 08:00:00 (Bar 5 = 08:00:00 stalls flat, Bars 8..15 rip up to +9.5% = $109.50!)
    h[2], c[2] = 102.0, 101.8
    low[3], c[3] = 100.85, 100.95
    for k in range(8, 20):
        o[k] = 101.0 + (k - 7) * 0.8
        c[k] = 101.0 + (k - 6) * 0.8
        h[k] = c[k] + 0.3
        low[k] = o[k] - 0.1

    f_map = {t_0800: -0.0050}  # -50 bps negative funding -> LONG receives +50.0 bps at 08:00:00!
    prof = P1_MECHANISM_PROFILES["P1_C1B_GLOBAL_SCARCITY_LONG"]
    recs = simulate_1m_monetization_geometries(
        ts_1m=ts,
        open_1m=o,
        high_1m=h,
        low_1m=low,
        close_1m=c,
        qvol_1m=qv,
        tb_share_1m=tb_sh,
        book_imb_1pct_1m=bk_imb,
        funding_map=f_map,
        raw_events=[(0, 1, 80.0, -90.0)],
        profile=prof,
        symbol="ENAUSDT",
    )
    by_m = {r["exec_mode"]: r for r in recs}

    # 1) MONO_A_CONVEX_TRAIL was NOT choked on Bar 3's dip to $100.85 and hit full +8.5% (+850 bps) Maker TP + 50 bps funding!
    r_a = by_m["MONO_A_CONVEX_TRAIL"]
    assert r_a["exit_reason"] == "TAKE_PROFIT_MAKER"
    assert r_a["gross_bps"] >= 849.0
    assert abs(r_a["funding_bps"] - 50.0) < 1e-3
    assert abs(r_a["gross_bps"] + r_a["funding_bps"] - r_a["fee_bps"] - r_a["net_bps"]) < 1e-2

    # 2) MONO_B_COUPON_VACUUM banked the +50.0 bps funding at 08:00:00 and exited cleanly at 08:01:00 (`COUPON_BANKED_EXIT`)!
    r_b = by_m["MONO_B_COUPON_VACUUM"]
    assert r_b["exit_reason"] == "COUPON_BANKED_EXIT"
    assert abs(r_b["funding_bps"] - 50.0) < 1e-3

    # 3) MONO_C_SPLIT_50_50 banked 50% @ +1.8% (+180 bps) on Bar 2 AND rode 50% runner to +9.0% (+900 bps) = +540 bps avg gross!
    r_c = by_m["MONO_C_SPLIT_50_50"]
    assert r_c["exit_reason"] == "SPLIT_FULL_TP2_RUNNER"
    assert abs(r_c["gross_bps"] - 540.0) < 1.0
    # Funding at 08:00:00 occurred AFTER Leg 1 closed on Bar 2 -> exact 50% * +50 bps = +25.0 bps!
    assert abs(r_c["funding_bps"] - 25.0) < 1e-3
    assert abs(r_c["gross_bps"] + r_c["funding_bps"] - r_c["fee_bps"] - r_c["net_bps"]) < 1e-2


def test_l2_book_depth_ask_wall_veto_vs_ask_vacuum_pass() -> None:
    n = 16 * 1440
    t0 = 1_735_689_600  # 2025-01-01 00:00:00 UTC
    ts = np.arange(t0, t0 + n * 60, 60, dtype=np.int64)
    c = np.full(n, 10.0, dtype=np.float64)
    o = c.copy()
    h = c * 1.001
    low = c * 0.999
    qv = np.full(n, 35_000.0, dtype=np.float64)
    tb = qv * 0.50

    # Create a 5m OI expansion + underwater trapped shorts breakout at day 15 10:00 UTC (`idx_b = 15*1440 + 600`)
    idx_b = 15 * 1440 + 600
    c[idx_b - 30 : idx_b + 1] = np.linspace(10.0, 10.8, 31)
    o[idx_b - 30 : idx_b + 1] = c[idx_b - 30 : idx_b + 1] - 0.01
    h[idx_b - 30 : idx_b + 1] = c[idx_b - 30 : idx_b + 1] + 0.01
    low[idx_b - 30 : idx_b + 1] = o[idx_b - 30 : idx_b + 1] - 0.01
    qv[idx_b - 4 : idx_b + 5] = 150_000.0
    tb[idx_b - 4 : idx_b + 5] = 100_000.0

    perp_df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": low,
            "close": c,
            "quote_volume": qv,
            "taker_buy_quote_volume": tb,
        }
    )
    prem_df = pd.DataFrame(
        {"timestamp": ts, "p_open_bps": -40.0, "p_high_bps": -20.0, "p_low_bps": -55.0, "p_close_bps": -40.0}
    )
    m_ts = np.arange(t0, t0 + n * 60, 300, dtype=np.int64)
    m_oi = np.full(len(m_ts), 1_000_000.0, dtype=np.float64)
    # OI jumped to 1.4M contracts while price was still $10.0 (`OI-VWAP` stays near $10.0 -> Pain > +2 ATR at $10.8!)
    b5_trig = idx_b // 5
    m_oi[b5_trig - 20 : b5_trig + 5] = 1_400_000.0
    met_df = pd.DataFrame({"timestamp": m_ts, "sum_open_interest": m_oi, "sum_open_interest_value": m_oi * 10.0})
    fnd_df = pd.DataFrame(
        {
            "timestamp": [int(ts[idx_b]) - 7200, int(ts[idx_b]) + 3600],
            "funding_rate": [-0.0020, -0.0020],
            "interval_hours": [8.0, 8.0],
        }
    )

    st = prepare_symbol_p1_base_state("ENAUSDT", perp_df, None, prem_df, met_df, fnd_df, None)
    assert st is not None
    assert len(st["needed_dates"]) >= 1

    # Case 1: Heavy L2 Ask Wall (`bid_2 = $50k` vs `ask_2 = $450k` -> `imb_2 = -0.80`) MUST VETO `P2_C2A` and `P6A`!
    bd_wall = pd.DataFrame(
        {
            "timestamp": [int(ts[idx_b])],
            "symbol": ["ENAUSDT"],
            "bid_02_usd": [10_000.0],
            "ask_02_usd": [90_000.0],
            "bid_1_usd": [25_000.0],
            "ask_1_usd": [225_000.0],
            "bid_2_usd": [50_000.0],
            "ask_2_usd": [450_000.0],
            "bid_5_usd": [100_000.0],
            "ask_5_usd": [900_000.0],
        }
    )
    recs_wall = evaluate_symbol_p1_with_book_depth(st, bd_wall)
    mechs_wall = {r["mech_id"] for r in recs_wall}
    assert "B2_C2A_BASE_UNCHOKED_LONG" in mechs_wall
    assert "P2_C2A_BOOK_VACUUM_PAIN_LONG" not in mechs_wall
    assert "P6A_L2_ASK_AIR_POCKET_LONG" not in mechs_wall

    # Case 2: Strong L2 Ask Vacuum (`bid_2 = $400k` vs `ask_2 = $100k` -> `imb_2 = +0.60`) PASSES BOTH `P2_C2A` and `P6A`!
    bd_vac = pd.DataFrame(
        {
            "timestamp": [int(ts[idx_b])],
            "symbol": ["ENAUSDT"],
            "bid_02_usd": [80_000.0],
            "ask_02_usd": [20_000.0],
            "bid_1_usd": [200_000.0],
            "ask_1_usd": [50_000.0],
            "bid_2_usd": [400_000.0],
            "ask_2_usd": [100_000.0],
            "bid_5_usd": [800_000.0],
            "ask_5_usd": [200_000.0],
        }
    )
    recs_vac = evaluate_symbol_p1_with_book_depth(st, bd_vac)
    mechs_vac = {r["mech_id"] for r in recs_vac}
    assert "P2_C2A_BOOK_VACUUM_PAIN_LONG" in mechs_vac
    assert "P6A_L2_ASK_AIR_POCKET_LONG" in mechs_vac
