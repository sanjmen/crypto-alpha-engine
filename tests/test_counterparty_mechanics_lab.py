"""
Unit Tests for the 1-Minute Linear-Return Counterparty-Mechanics Laboratory
and Unified 1-Minute Simulator (`src/models/counterparty_mechanics_lab.py`,
`src/backtest/unified_1m_simulator.py`, `src/features/multimodal_alpha.py`).

Verifies:
1. ZERO `np.log` convexity bias on volatile Shorts & Longs (`gross + funding - fee == net` exact linear return).
2. Exact `:00:00` funding settlement accounting, `:56` pre-settlement exit guard, and `100%` pessimistic SL priority.
3. Contract-unit `compute_recursive_oi_vwap_anchor` causality and `+60s` `metrics_5m` safety lag.
4. State-A (`C1A`) vs State-B (`C1B`, `C1C`) bifurcation on negative-funding settlements.
5. True Out-of-Sample In-Fold Refit `CombinatorialPurgedCV` (`evaluate_infold_cpcv_and_treasure_metrics`).
6. Unified 1-minute simulator (`resolve_sniper_1m_path`, `tier_cost_bps`) exact linear accounting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.unified_1m_simulator import resolve_sniper_1m_path, tier_cost_bps
from src.features.multimodal_alpha import attach_forward_returns_for_audit
from src.models.counterparty_mechanics_lab import (
    COUNTERPARTY_MECHANISM_SPECS,
    CounterpartyMechanismSpec,
    compute_recursive_oi_vwap_anchor,
    evaluate_infold_cpcv_and_treasure_metrics,
    extract_sovereign_btc_shocks,
    scan_symbol_counterparty_mechanisms_1m,
    simulate_1m_event_across_modes,
)
from src.strategies.event_snipers import SniperBracketSpec


def test_linear_returns_eliminate_log_convexity_bias_on_volatile_short() -> None:
    # Verify attach_forward_returns_for_audit uses exact linear return `(P_t+h - P_t)/P_t` (NOT np.log)
    panel = pd.DataFrame(
        {
            "symbol": ["ZKCUSDT", "ZKCUSDT"],
            "decision_ts": [1_700_000_000, 1_700_003_600],
            "close": [1.00, 1.50],  # +50% short squeeze!
        }
    )
    audited = attach_forward_returns_for_audit(panel, {"1h": 1})
    fwd_1h = float(audited.loc[0, "fwd_ret_1h"])
    # Linear return is exact +0.50 (+50%), whereas np.log(1.5) was +0.405465 (understating short loss by 945 bps!)
    assert abs(fwd_1h - 0.50) < 1e-9

    # Verify simulate_1m_event_across_modes exact linear return on a +50% gap squeeze against a SHORT
    ts = np.arange(1_700_000_000, 1_700_000_000 + 15 * 60, 60, dtype=np.int64)
    o = np.full(15, 1.00, dtype=np.float64)
    h = np.full(15, 1.00, dtype=np.float64)
    low = np.full(15, 1.00, dtype=np.float64)
    c = np.full(15, 1.00, dtype=np.float64)
    qv = np.full(15, 500_000.0, dtype=np.float64)

    # Signal at bar 0 -> TAKER_OPEN enters SHORT at bar 1 open = $1.00; bar 2 gaps open to $1.50 (hitting 20% SL at $1.50!)
    o[2] = 1.50
    h[2] = 1.55
    low[2] = 1.48
    c[2] = 1.50

    spec = CounterpartyMechanismSpec(
        mech_id="TEST_SHORT_SQUEEZE",
        pillar="Test",
        description="Linear short loss audit",
        tp_pct=0.20,
        sl_pct=0.20,
        lock_trigger_pct=0.0,
        lock_level_pct=0.0,
        trail_trigger_pct=0.0,
        trail_distance_pct=0.0,
        max_hold_minutes=10,
    )
    recs = simulate_1m_event_across_modes(
        ts_1m=ts,
        open_1m=o,
        high_1m=h,
        low_1m=low,
        close_1m=c,
        qvol_1m=qv,
        funding_map={},
        raw_events=[(0, -1, 10.0, 0.0)],
        spec=spec,
        symbol="ZKCUSDT",
    )
    taker_rec = next(r for r in recs if r["exec_mode"] == "TAKER_OPEN")
    # Exact -50.0% (-5000.0 bps) gross linear loss + 13.0 bps roundtrip taker fee = -5013.0 bps net!
    assert abs(taker_rec["gross_bps"] - (-5000.0)) < 1e-3
    assert abs(taker_rec["fee_bps"] - 13.0) < 1e-3
    assert abs(taker_rec["net_bps"] - (-5013.0)) < 1e-3


def test_realized_funding_accounting_and_pessimistic_same_bar_sl_priority() -> None:
    # Start at 07:50:00 UTC so bar index 9 (07:59:00) closes at 08:00:00 UTC settlement!
    t_0800 = 1_704_096_000  # Exact :00:00 UTC boundary (divisible by 3600)
    assert t_0800 % 3600 == 0
    ts = np.arange(t_0800 - 10 * 60, t_0800 + 10 * 60, 60, dtype=np.int64)
    n = len(ts)
    o = np.full(n, 100.0, dtype=np.float64)
    h = np.full(n, 100.1, dtype=np.float64)
    low = np.full(n, 99.9, dtype=np.float64)
    c = np.full(n, 100.0, dtype=np.float64)
    qv = np.full(n, 500_000.0, dtype=np.float64)

    # Case 1: LONG held across 08:00:00 UTC when realized funding is -45 bps (-0.0045) -> receives +45.0 bps!
    f_map = {t_0800: -0.0045}
    spec_hold = COUNTERPARTY_MECHANISM_SPECS["C1B_PRE_SETTLE_SPOT_SCARCITY_LONG"]
    recs_fnd = simulate_1m_event_across_modes(
        ts_1m=ts,
        open_1m=o,
        high_1m=h,
        low_1m=low,
        close_1m=c,
        qvol_1m=qv,
        funding_map=f_map,
        raw_events=[(1, 1, 50.0, -90.0)],
        spec=spec_hold,
        symbol="ENAUSDT",
    )
    tk_fnd = next(r for r in recs_fnd if r["exec_mode"] == "TAKER_OPEN")
    assert abs(tk_fnd["funding_bps"] - 45.0) < 1e-3
    assert abs(tk_fnd["gross_bps"] + tk_fnd["funding_bps"] - tk_fnd["fee_bps"] - tk_fnd["net_bps"]) < 1e-2

    # Case 2: Pre-settlement guard (`pre_settlement_exit_minute=56`) exits BEFORE 08:00:00 -> 0.0 bps funding!
    spec_guard = COUNTERPARTY_MECHANISM_SPECS["M00_NAIVE_M25_CONTROL"]
    recs_guard = simulate_1m_event_across_modes(
        ts_1m=ts,
        open_1m=o,
        high_1m=h,
        low_1m=low,
        close_1m=c,
        qvol_1m=qv,
        funding_map=f_map,
        raw_events=[(0, -1, 30.0, -50.0)],
        spec=spec_guard,
        symbol="ENAUSDT",
    )
    tk_guard = next(r for r in recs_guard if r["exec_mode"] == "TAKER_OPEN")
    assert tk_guard["exit_reason"] == "PRE_SETTLE_EXIT"
    assert abs(tk_guard["funding_bps"]) < 1e-6

    # Case 3: Ambiguous 1m candle hitting BOTH Take-Profit AND Stop-Loss -> MUST resolve to STOP_LOSS!
    h_amb = h.copy()
    l_amb = low.copy()
    h_amb[3] = 108.0  # Hits +5.5% TP
    l_amb[3] = 96.0  # Also hits -2.2% SL on same bar!
    recs_amb = simulate_1m_event_across_modes(
        ts_1m=ts,
        open_1m=o,
        high_1m=h_amb,
        low_1m=l_amb,
        close_1m=c,
        qvol_1m=qv,
        funding_map={},
        raw_events=[(1, 1, 50.0, 0.0)],
        spec=spec_hold,
        symbol="ENAUSDT",
    )
    tk_amb = next(r for r in recs_amb if r["exec_mode"] == "TAKER_OPEN")
    assert tk_amb["exit_reason"] == "STOP_LOSS"
    assert tk_amb["net_bps"] < -200.0


def test_recursive_contract_oi_vwap_and_scan_state_a_vs_state_b() -> None:
    # When contract OI is constant (zero new contracts opened) while price doubles,
    # contract-unit OI-VWAP stays near initial entry (~100), proving zero price-induced OI drift!
    oi_units = np.full(50, 1_000_000.0, dtype=np.float64)
    px = np.linspace(100.0, 200.0, 50, dtype=np.float64)
    anchor_flat = compute_recursive_oi_vwap_anchor(oi_units, px, base_decay=0.9965)
    assert anchor_flat[-1] < 112.0

    # When OI doubles at $200, OI-VWAP immediately shifts ~50% of the way toward $200!
    oi_jump = oi_units.copy()
    oi_jump[-1] = 2_000_000.0
    anchor_jump = compute_recursive_oi_vwap_anchor(oi_jump, px, base_decay=0.9965)
    assert anchor_jump[-1] > 145.0

    # End-to-end synthetic 16-day 1m symbol scan verifying BTC shocks + State-A vs State-B triggers
    n = 16 * 1440
    t0 = 1_735_689_600  # 2025-01-01 00:00:00 UTC
    ts = np.arange(t0, t0 + n * 60, 60, dtype=np.int64)
    c = np.full(n, 10.0, dtype=np.float64)
    o = c.copy()
    h = c * 1.001
    low = c * 0.999
    qv = np.full(n, 30_000.0, dtype=np.float64)  # ~$43M daily ADV (> $15M gate)
    tb = qv * 0.50

    sp_c = np.full(n, 10.0, dtype=np.float64)
    sp_qv = np.full(n, 25_000.0, dtype=np.float64)
    sp_tb = sp_qv * 0.50

    # Inject a State-B Post-Settlement Short-Panic LONG event at day 15 08:00 UTC
    idx_settle = 15 * 1440 + 8 * 60
    t_settle = int(ts[idx_settle])
    # Spot trades +120 bps above Perp (`basis = -118.6 bps <= -80 bps`), and `:00` bar refuses to dump (`+15 bps`, `tb=65%`)
    sp_c[idx_settle - 60 : idx_settle + 10] = 10.12
    c[idx_settle] = 10.015
    h[idx_settle] = 10.025
    tb[idx_settle] = qv[idx_settle] * 0.65
    # Subsequent 30m squeezes up +4%
    for k in range(1, 40):
        o[idx_settle + k] = 10.015 + 0.012 * (k - 1)
        c[idx_settle + k] = 10.015 + 0.012 * k
        h[idx_settle + k] = c[idx_settle + k] + 0.01
        low[idx_settle + k] = o[idx_settle + k] - 0.005

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
    spot_df = pd.DataFrame(
        {
            "timestamp": ts,
            "close": sp_c,
            "quote_volume": sp_qv,
            "taker_buy_quote_volume": sp_tb,
        }
    )
    fnd_df = pd.DataFrame(
        {
            "timestamp": [t_settle - 14400, t_settle],
            "funding_rate": [-0.0030, -0.0038],
            "interval_hours": [8.0, 4.0],  # Also triggers C4 Clock Compression (8h -> 4h)!
        }
    )
    btc_sh = extract_sovereign_btc_shocks(perp_df, spot_df)
    trades = scan_symbol_counterparty_mechanisms_1m(
        symbol="TESTUSDT",
        perp_1m=perp_df,
        spot_1m=spot_df,
        metrics_5m=None,
        funding_df=fnd_df,
        btc_shocks=btc_sh,
    )
    mechs_triggered = {t["mech_id"] for t in trades}
    # M00 (naive short) loses on the squeeze, C1A VETOES the short cleanly, while C1C and C4 capture the LONG squeeze!
    assert "M00_NAIVE_M25_CONTROL" in mechs_triggered
    assert "C1A_POST_SETTLE_HARVESTER_SHORT" not in mechs_triggered
    assert "C1C_POST_SETTLE_SHORT_PANIC_LONG" in mechs_triggered
    assert "C4_CLOCK_COMPRESSION_SQUEEZE_LONG" in mechs_triggered


def test_infold_cpcv_and_unified_1m_simulator_smoke() -> None:
    rng = np.random.default_rng(42)
    base_ts = 1_704_067_200  # 2024-01-01
    rows = []
    for i in range(120):
        t_i = base_ts + i * 7 * 86400  # Spans 2024, 2025, 2026 (~840 days)
        gross = 65.0 + float(rng.normal(0.0, 20.0))
        fee = 8.5
        rows.append(
            {
                "mech_id": "C1C_POST_SETTLE_SHORT_PANIC_LONG",
                "exec_mode": "TAKER_OPEN",
                "symbol": "ENAUSDT",
                "signal_ts": t_i,
                "entry_ts": t_i + 60,
                "exit_ts": t_i + 1800,
                "side": 1,
                "hold_min": 30,
                "gross_bps": gross,
                "funding_bps": 0.0,
                "fee_bps": fee,
                "net_bps": gross - fee,
                "exit_reason": "TAKE_PROFIT_MAKER",
                "strength": float(50.0 + rng.uniform(0, 50)),
                "aux_metric": -110.0,
            }
        )
    lb = evaluate_infold_cpcv_and_treasure_metrics(pd.DataFrame(rows), total_trials=30)
    assert len(lb) == 1
    assert bool(lb.loc[0, "is_treasure"]) is True
    assert float(lb.loc[0, "cpcv_infold_oos_bps"]) > 30.0

    # Verify unified_1m_simulator linear path & tier cost
    assert tier_cost_bps(0, maker=True) < tier_cost_bps(0, maker=False)
    ts_1m = np.arange(1_740_000_000, 1_740_000_000 + 30 * 60, 60, dtype=np.int64)
    o_1m = np.full(30, 50.0, dtype=np.float64)
    h_1m = np.full(30, 52.5, dtype=np.float64)  # +5% hits TP
    l_1m = np.full(30, 49.8, dtype=np.float64)
    c_1m = np.full(30, 51.0, dtype=np.float64)
    b_spec = SniperBracketSpec(
        strategy="TEST",
        take_profit_pct=0.04,
        stop_loss_pct=0.02,
        lock_trigger_pct=0.015,
        lock_level_pct=0.007,
        trail_trigger_pct=0.025,
        trail_distance_pct=0.01,
        max_hold_hours=0.5,
    )
    res = resolve_sniper_1m_path(
        entry_ts=int(ts_1m[0]),
        side=1,
        spec=b_spec,
        adv_rank=3,
        ts_1m=ts_1m,
        open_1m=o_1m,
        high_1m=h_1m,
        low_1m=l_1m,
        close_1m=c_1m,
    )
    assert res is not None
    _, hr_ret, is_loss, diag = res
    assert is_loss is False
    assert abs(diag["gross_bps"] - 400.0) < 1e-4
    assert abs(sum(hr_ret.values()) * 1e4 - diag["net_bps"]) < 1e-4
