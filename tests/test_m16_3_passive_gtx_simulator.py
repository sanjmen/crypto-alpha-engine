"""
Unit Tests for Issue #69 (M16-3):
Conservative 1m Passive Post-Only (`GTX`) Wick-Sniper & Queue-Penetration Simulator.

Covers all 4 anti-optimism microstructure invariants:
  1. Exact touch (`low == limit_price` / `high == limit_price`) NEVER fills; requires strict
     `queue_penetration_bps` price-through + sufficient bar quote volume.
  2. `GTX` Post-Only book-crossing check rejects (`GTX_REJECTED`) or passively re-prices when
     `limit_price` crosses `open[t + 1]`.
  3. Deterministic 100% worst-case same-bar Stop-Loss priority (`STOP_LOSS_SAME_BAR` on entry bar
     even if `high >= take_profit`, and `STOP_LOSS` before `TAKE_PROFIT` on holding bars).
  4. Short TTL expiration (`TTL_EXPIRED`) and sensitivity matrix across `1.0 - 3.0 bps` penetration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.passive_gtx_simulator import (
    GTXSignalEvent,
    PassiveGTXConfig,
    PassiveGTXExecutionSimulator,
)


def _make_bars(rows: list[tuple[int, float, float, float, float, float]], symbol: str = "SOLUSDT") -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "quote_volume"],
    ).assign(symbol=symbol)


def test_exact_touch_does_not_fill_requires_strict_queue_penetration() -> None:
    # Signal at idx=0 close=100.0 with 10 bps offset -> BUY LIMIT at 99.90.
    # With queue_penetration_bps=2.0, fill requires low <= 99.90 * (1 - 0.0002) = 99.88002!
    bars = _make_bars(
        [
            (1000, 100.0, 100.1, 99.95, 100.00, 500_000.0),  # idx 0: signal bar
            (1060, 99.95, 100.0, 99.90, 99.94, 500_000.0),  # idx 1: EXACT touch 99.90 -> MUST NOT FILL!
            (1120, 99.94, 100.0, 99.89, 99.95, 500_000.0),  # idx 2: 1 bps through (99.89 > 99.88002) -> MUST NOT FILL!
            (1180, 99.95, 100.2, 99.93, 100.15, 500_000.0),  # idx 3: runs away -> TTL expires!
        ]
    )
    sim = PassiveGTXExecutionSimulator(PassiveGTXConfig(queue_penetration_bps=2.0, ttl_minutes=3))
    ev = GTXSignalEvent(signal_idx=0, side=1, limit_offset_bps=10.0, take_profit_pct=0.01, stop_loss_pct=0.01)
    res = sim.simulate_signals_on_1m_bars(bars, [ev])
    assert len(res) == 1
    assert bool(res["filled"].iloc[0]) is False
    assert res["status"].iloc[0] == "TTL_EXPIRED"

    # Now penetrate to 99.87 on idx 2 and hit Maker TP (99.90 * 1.01 = 100.899, +1bp pen = 100.909) on idx 3!
    bars_fill = _make_bars(
        [
            (1000, 100.0, 100.1, 99.95, 100.00, 500_000.0),
            (1060, 99.95, 100.0, 99.90, 99.94, 500_000.0),
            (1120, 99.94, 100.0, 99.87, 99.98, 500_000.0),  # fills @ 99.90!
            (
                1180,
                99.98,
                100.95,
                99.96,
                100.92,
                500_000.0,
            ),  # hits Maker TP (+100 bps gross - 4 bps roundtrip Maker = +96 bps net!)
        ]
    )
    res2 = sim.simulate_signals_on_1m_bars(bars_fill, [ev])
    assert bool(res2["filled"].iloc[0]) is True
    assert res2["exit_reason"].iloc[0] == "TAKE_PROFIT_MAKER"
    assert res2["fee_bps"].iloc[0] == pytest.approx(4.0)  # 2.0 Maker entry + 2.0 Maker TP exit
    assert res2["net_ret_bps"].iloc[0] == pytest.approx(96.0, abs=0.05)


def test_gtx_post_only_rejects_or_reprices_when_submission_crosses_open() -> None:
    # Signal at idx=0 close=100.0, offset=5 bps -> BUY LIMIT = 99.95.
    # Next minute gaps down to open=99.80 -> BUY LIMIT 99.95 >= 99.80 crosses book!
    bars = _make_bars(
        [
            (1000, 100.0, 100.1, 99.95, 100.00, 500_000.0),
            (1060, 99.80, 100.5, 99.60, 100.40, 500_000.0),
            (1120, 100.40, 101.2, 100.30, 101.10, 500_000.0),
        ]
    )
    ev = GTXSignalEvent(signal_idx=0, side=1, limit_offset_bps=5.0, take_profit_pct=0.01, stop_loss_pct=0.01)

    # Policy 1: 'reject' -> strictly rejected, 0 fill!
    sim_rej = PassiveGTXExecutionSimulator(PassiveGTXConfig(gtx_cross_policy="reject"))
    r_rej = sim_rej.simulate_signals_on_1m_bars(bars, [ev])
    assert r_rej["status"].iloc[0] == "GTX_REJECTED"
    assert bool(r_rej["filled"].iloc[0]) is False

    # Policy 2: 'reprice' -> reprices passive below open[1]=99.80 -> 99.80 * (1 - 0.00015) = 99.78503, fills via low=99.60!
    sim_rep = PassiveGTXExecutionSimulator(
        PassiveGTXConfig(gtx_cross_policy="reprice", gtx_reprice_offset_bps=1.5, queue_penetration_bps=2.0)
    )
    r_rep = sim_rep.simulate_signals_on_1m_bars(bars, [ev])
    assert bool(r_rep["filled"].iloc[0]) is True
    assert bool(r_rep["gtx_repriced"].iloc[0]) is True
    assert r_rep["entry_price"].iloc[0] == pytest.approx(99.80 * (1.0 - 0.00015), rel=1e-6)


def test_same_bar_fill_and_stop_loss_enforces_100pct_worst_case_adverse_ordering() -> None:
    # BUY LIMIT @ 99.90, TP=+1.0% (100.899), SL=-0.8% (99.1008).
    # On idx=1, bar has BOTH low=99.00 (breaches SL) AND high=101.50 (breaches TP)!
    # Simulator MUST enforce 100% adverse selection: STOP_LOSS_SAME_BAR (-80 bps gross - 8 bps Maker+TakerStop = -88 bps net)!
    bars = _make_bars(
        [
            (1000, 100.0, 100.1, 99.95, 100.00, 500_000.0),
            (1060, 99.95, 101.50, 99.00, 101.20, 500_000.0),
            (1120, 101.20, 102.00, 101.00, 101.80, 500_000.0),
        ]
    )
    sim = PassiveGTXExecutionSimulator(
        PassiveGTXConfig(
            queue_penetration_bps=2.0,
            maker_fee_bps=2.0,
            taker_stop_fee_bps=5.0,
            stop_slippage_bps=1.0,
        )
    )
    ev = GTXSignalEvent(signal_idx=0, side=1, limit_offset_bps=10.0, take_profit_pct=0.010, stop_loss_pct=0.008)
    res = sim.simulate_signals_on_1m_bars(bars, [ev])
    assert res["exit_reason"].iloc[0] == "STOP_LOSS_SAME_BAR"
    assert res["gross_ret_bps"].iloc[0] == pytest.approx(-80.0, abs=0.05)
    assert res["fee_bps"].iloc[0] == pytest.approx(8.0, abs=0.05)  # 2.0 Maker entry + 6.0 Taker SL exit
    assert res["net_ret_bps"].iloc[0] == pytest.approx(-88.0, abs=0.05)


def test_penetration_sensitivity_matrix_monotonically_tightens_fills() -> None:
    rng = np.random.default_rng(99)
    n = 600
    ts = np.arange(1_760_000_000, 1_760_000_000 + n * 60, 60, dtype=np.int64)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.0025, size=n)))
    o = np.roll(c, 1)
    o[0] = c[0]
    h = np.maximum(o, c) * (1.0 + rng.uniform(0.0004, 0.0030, size=n))
    l_arr = np.minimum(o, c) * (1.0 - rng.uniform(0.0004, 0.0030, size=n))
    bars = pd.DataFrame(
        {"timestamp": ts, "open": o, "high": h, "low": l_arr, "close": c, "quote_volume": np.full(n, 1e6)}
    )
    evs = [
        GTXSignalEvent(
            signal_idx=i, side=1, limit_offset_bps=8.0, take_profit_pct=0.006, stop_loss_pct=0.006, symbol="SOLUSDT"
        )
        for i in range(10, 520, 25)
    ]
    mat = PassiveGTXExecutionSimulator.run_penetration_sensitivity_matrix(
        bars_by_symbol={"SOLUSDT": bars},
        events_by_symbol={"SOLUSDT": evs},
        penetration_bps_grid=(0.5, 1.5, 3.0),
        ttl_grid=(3,),
    )
    assert len(mat) == 3
    fills = list(mat["filled_trades"])
    assert fills[0] >= fills[1] >= fills[2]
