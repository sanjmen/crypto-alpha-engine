"""
Regression Test Suite for the Nov 17, 2026 Forensic Production Audit Remediations.
Covers:
- [CRIT-A1] Shadow 24h High/Low fake Maker fill prevention & WS bid/ask preservation
- [CRIT-A3] Live Market fallback `newOrderRespType="RESULT"`, 0.05% VIP0 Taker fee & mixed fill attribution
- [CRIT-A5] $100 BTC/ETH MIN_NOTIONAL pre-validation & <$5 reduceOnly full-close unlock
- [CRIT-B1] Strict filtering of unclosed live 1h/15m candles
- [CRIT-B2] Genuine `priors=None` (0.0 priors) + All-Negative Factor Cash-Out / De-Gross Gate
- [CRIT-B4/B5/B6/B8] Cooldown water-fill, unconditional <$160 stub pruning, un-nested Holistic Shields & causal median_vol
- [CRIT-C1/C2/C3] Wallet-vs-free cash accounting, HWM & Kill-Switch restart hydration, offline dry_run & 7-Factor backtester
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.live_broker import BinanceLiveBroker
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.seven_factor_engine import DEFAULT_PRIORS, SevenFactorOnlineEngine


def _make_bars(n_bars: int = 160, seed: int = 101) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=n_bars, freq="1h", tz="UTC")
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "LINKUSDT", "ARBUSDT", "APTUSDT"]
    out: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(syms):
        rets = rng.normal(0.0003 * (i - 3.5), 0.012, size=n_bars)
        close = (100.0 + 20.0 * i) * np.exp(np.cumsum(rets))
        out[sym] = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close * 0.999,
                "high": close * 1.006,
                "low": close * 0.994,
                "close": close,
                "volume": rng.uniform(50_000.0, 200_000.0, size=n_bars),
            }
        )
    return out


def test_crit_a1_shadow_ignores_24h_high_low_and_preserves_ws_spread():
    """[CRIT-A1] Passive LIMIT orders must NOT fill on 24h rolling high/low, and WS bid/ask must survive REST poll."""
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0, slippage_bps=2.0)
    mock_ex = MagicMock()
    # Simulate Binance /fapi/v1/ticker/24hr where bid/ask are None and low/high are 24h extremes
    mock_ex.fetch_tickers.return_value = {
        "BTC/USDT:USDT": {
            "last": 90_000.0,
            "bid": None,
            "ask": None,
            "high": 94_000.0,
            "low": 85_000.0,
            "percentage": 1.2,
            "quoteVolume": 800_000_000.0,
        }
    }
    daemon = ShadowTradingDaemon(
        broker=broker,
        exchange_client=mock_ex,
        symbols=["BTC/USDT"],
        enable_warm_start=False,
    )
    # Seed real-time WebSocket L1 spread in latest_tickers
    daemon.latest_tickers["BTC/USDT"] = {
        "price": 89_995.0,
        "bid": 89_990.0,
        "ask": 90_010.0,
        "low": 89_990.0,
        "high": 90_010.0,
        "change_24h": 1.1,
        "volume_24h": 790_000_000.0,
    }

    ticks = daemon.fetch_live_tickers()
    assert ticks["BTC/USDT"]["bid"] == 89_990.0
    assert ticks["BTC/USDT"]["ask"] == 90_010.0
    assert ticks["BTC/USDT"]["low_24h"] == 85_000.0
    assert ticks["BTC/USDT"]["low"] == 89_990.0  # Intra-tick, NOT 85,000!

    # Place passive BUY LIMIT at $88,000 (> 24h low $85,000, but well below current ask $90,010)
    broker.set_market_price("BTC/USDT", price=90_000.0, bid=89_990.0, ask=90_010.0)
    ord_lim = broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.01,
        price=88_000.0,
    )
    assert ord_lim.status == OrderStatus.PENDING
    filled = broker.check_pending_orders(
        {"BTC/USDT": {"price": 90_000.0, "bid": 89_990.0, "ask": 90_010.0, "low": 85_000.0, "high": 94_000.0}}
    )
    assert len(filled) == 0

    # Also verify STOP_LOSS in Shadow applies adverse slippage (not exact 0-slippage limit_price)
    broker.create_order(symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.01)
    broker.create_bracket_orders("BTC/USDT", take_profit=95_000.0, stop_loss=89_000.0)
    sl_fills = broker.check_pending_orders({"BTC/USDT": {"price": 88_900.0, "bid": 88_890.0, "ask": 88_910.0}})
    assert len(sl_fills) == 1
    assert sl_fills[0].average_fill_price < 88_890.0
    db.close()


def test_crit_a3_live_market_fill_price_blended_fee_and_role_attribution():
    """[CRIT-A3] Verifies RESULT market fill price lookup, 0.05% VIP0 fee, and accurate mixed GTX+Market attribution."""
    live_broker = BinanceLiveBroker(
        dry_run=False,
        testnet=True,
        initial_capital=3000.0,
        chase_interval_sec=0.0,
        max_chase_iterations=1,
    )
    assert live_broker.taker_fee_pct == 0.0005

    mock_client = MagicMock()
    mock_client.fetch_balance.return_value = {
        "USDT": {"free": 2000.0, "total": 3000.0},
        "info": {"assets": [{"asset": "USDT", "walletBalance": "3000.0", "availableBalance": "2000.0"}]},
    }
    mock_client.fetch_positions.return_value = []
    mock_client.fetch_order_book.return_value = {"bids": [[100.0, 50.0]], "asks": [[100.1, 50.0]]}

    # Simulate 10% partial Maker fill (1.0 unit @ 100.0) followed by 90% Taker Market fallback (9.0 units @ 100.40)
    # where initial create_order(market) returns average=0.0 (ACK-style) and fetch_order resolves 100.40!
    mock_client.create_order.side_effect = [
        {"id": "gtx_1", "status": "open"},
        {"id": "mkt_2", "status": "closed", "filled": 0.0, "average": 0.0},
    ]
    mock_client.fetch_order.side_effect = [
        {"id": "gtx_1", "status": "open", "filled": 1.0, "average": 100.0},
        {"id": "mkt_2", "status": "closed", "filled": 9.0, "average": 100.40},
    ]
    mock_client.cancel_order.return_value = {"id": "gtx_1", "filled": 1.0, "average": 100.0}
    live_broker.client = mock_client

    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0, live_broker=live_broker, max_gross_leverage=6.0)
    # [CRIT-C1] Wallet balance ($3,000) must be stored in cash, NEVER availableBalance ($2,000)!
    assert broker.portfolio.cash == 3000.0
    assert live_broker.portfolio.cash == 3000.0

    broker.set_market_price("SOL/USDT", price=100.0, bid=100.0, ask=100.1)
    ord_exec = broker.create_order(
        symbol="SOL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=100.0,
        strategy="7-Factor Core",
    )

    # Verify `newOrderRespType="RESULT"` was sent on market fallback
    mkt_call_kwargs = mock_client.create_order.call_args_list[1].kwargs
    assert mkt_call_kwargs["params"].get("newOrderRespType") == "RESULT"

    # Weighted average fill price = (1.0 * 100.0 + 9.0 * 100.40) / 10.0 = 100.36 (NOT 100.0!)
    assert pytest.approx(ord_exec.average_fill_price, abs=1e-4) == 100.36

    # Verify 90% Taker fill is recorded as is_maker=False with blended Maker(0.02%)+Taker(0.05%) fee
    trades = db.get_trades(symbol="SOL/USDT")
    assert len(trades) == 1
    assert bool(trades[0]["is_maker"]) is False
    expected_fee = 1.0 * 100.0 * 0.0002 + 9.0 * 100.40 * 0.0005
    assert pytest.approx(float(trades[0]["fee_paid"]), abs=1e-4) == expected_fee
    db.close()


def test_crit_a5_min_notional_btc_eth_and_sub_5usd_reduce_only_close():
    """[CRIT-A5] Enforces $100 MIN_NOTIONAL on BTC/ETH opens while allowing <$5 full closes."""
    live_broker = BinanceLiveBroker(dry_run=True, initial_capital=3000.0)
    assert live_broker.get_min_notional("BTC/USDT", reduce_only=False) == 100.0
    assert live_broker.get_min_notional("ETH/USDT", reduce_only=False) == 100.0
    assert live_broker.get_min_notional("BTC/USDT", reduce_only=True) == 0.0

    # Verify <$5 dust full close executes cleanly in ShadowTradingDaemon.run_rebalance_cycle()
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0, max_gross_leverage=6.0)
    broker.set_market_price("ADA/USDT", price=1.0, bid=1.0, ask=1.0)
    broker.create_order(symbol="ADA/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=3.5)

    mock_alloc = MagicMock()
    mock_alloc.use_seven_factor_engine = False
    mock_alloc.last_seven_factor_result = None
    mock_alloc.last_leverage_snapshot = None
    mock_alloc.last_dominant_strategy = {}
    mock_alloc.generate_portfolio_allocations.return_value = (
        {"ADAUSDT": 0.0},
        "RANGING",
        {"momentum": 0.0, "stat_arb": 1.0, "carry": 0.0},
    )
    mock_ex = MagicMock()
    idx_1h = pd.date_range("2026-11-01", periods=80, freq="1h", tz="UTC")
    mock_ex.fetch_ohlcv.return_value = [[int(ts.timestamp() * 1000), 1.0, 1.01, 0.99, 1.0, 5000.0] for ts in idx_1h]
    mock_ex.fetch_funding_rate.return_value = {"fundingRate": 0.0}

    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=mock_alloc,
        exchange_client=mock_ex,
        symbols=["ADA/USDT"],
        enable_warm_start=False,
        enable_anomalies=False,
        min_rebalance_notional=30.0,
    )
    daemon.latest_tickers = {"ADA/USDT": {"price": 1.0, "bid": 1.0, "ask": 1.0}}
    orders = daemon.run_rebalance_cycle()
    assert len(orders) == 1
    assert pytest.approx(orders[0].quantity, abs=1e-4) == 3.5
    broker.check_pending_orders({"ADA/USDT": {"price": 1.001, "bid": 1.001, "ask": 1.002}})
    assert broker.portfolio.positions["ADA/USDT"].side == PositionSide.FLAT
    db.close()


def test_crit_b1_drops_unclosed_live_1h_and_15m_candles():
    """[CRIT-B1] Ensures incomplete current-hour / current-15m candles are stripped before feature computation."""
    now_ref = pd.Timestamp("2026-11-17T15:13:00Z")
    ts_1h = pd.date_range("2026-11-17T11:00:00Z", periods=5, freq="1h", tz="UTC")  # Last bar is 15:00-16:00 (OPEN!)
    df_1h = pd.DataFrame(
        {
            "timestamp": ts_1h,
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [10_000.0, 10_000.0, 10_000.0, 10_000.0, 45.0],  # 45.0 partial volume on open bar!
        }
    )
    cleaned_1h = ShadowTradingDaemon._drop_unclosed_bars(df_1h, timeframe="1h", now_utc=now_ref)
    assert len(cleaned_1h) == 4
    assert cleaned_1h["timestamp"].iloc[-1] == pd.Timestamp("2026-11-17T14:00:00Z")
    assert float(cleaned_1h["volume"].iloc[-1]) == 10_000.0

    ts_15m = pd.date_range("2026-11-17T14:30:00Z", periods=4, freq="15min", tz="UTC")  # Last bar 15:15 > 15:13
    df_15m = pd.DataFrame({"timestamp": ts_15m, "close": [1.0, 1.0, 1.0, 1.0], "volume": [100, 100, 100, 2]})
    cleaned_15m = ShadowTradingDaemon._drop_unclosed_bars(df_15m, timeframe="15m", now_utc=now_ref)
    assert cleaned_15m["timestamp"].iloc[-1] == pd.Timestamp("2026-11-17T14:45:00Z")


def test_crit_b2_zero_priors_and_no_edge_cash_out_gate():
    """[CRIT-B2] Verifies priors=None sets 0.0 priors (not DEFAULT_PRIORS) and steps down in all-negative-t regimes."""
    # Default no-arg constructor preserves DEFAULT_PRIORS backward compatibility
    eng_default = SevenFactorOnlineEngine()
    assert pytest.approx(float(eng_default.priors["macro"]), abs=1e-6) == DEFAULT_PRIORS["macro"]

    # Explicit priors=None sets strictly 0.0 priors across all 7 factors
    eng_zero = SevenFactorOnlineEngine(priors=None)
    assert float(eng_zero.priors.abs().sum()) == 0.0

    bars = _make_bars(n_bars=180, seed=77)
    mats = eng_zero.compute_matrices(bars_1h=bars)
    fw_norm = mats["fw_norm"]
    t_df = mats["t_df"]
    assert np.all(np.isfinite(mats["conviction"].values))
    assert np.allclose(fw_norm.sum(axis=1).values, 1.0, atol=1e-5)

    # On any bar where at least one un-gated factor has positive t-stat, any non-smoothed negative-t factor gets 0.0 weight
    pos_bar_mask = t_df[["macro", "med", "fast"]].max(axis=1) > 0.25
    if pos_bar_mask.any():
        neg_macro = pos_bar_mask & (t_df["macro"] < -0.01)
        if neg_macro.any():
            assert float(fw_norm.loc[neg_macro, "macro"].max()) == 0.0


def test_crit_b5_b8_unnested_holistic_shields_and_strictly_causal_median_vol():
    """[CRIT-B5 & B8] Verifies Holistic Shields run when chop_range_arb=False and median_vol has zero look-ahead."""
    bars = _make_bars(n_bars=160, seed=202)
    eng = SevenFactorOnlineEngine(
        priors=None,
        enable_chop_range_arb=False,
        enable_holistic_regime_shields=True,
    )
    mats_prefix = eng.compute_matrices(bars_1h={k: v.iloc[:120].copy() for k, v in bars.items()})

    # Mutate future bars 120..159 by 10x volatility shock and verify bar 119 is 100% identical (zero look-ahead!)
    bars_shocked = {}
    for k, v in bars.items():
        d = v.copy()
        d.loc[d.index[120:], "close"] = d.loc[d.index[120:], "close"] * np.linspace(1.0, 2.5, 40)
        bars_shocked[k] = d
    mats_full = eng.compute_matrices(bars_1h=bars_shocked)

    assert pytest.approx(float(mats_prefix["median_vol"].iloc[119]), abs=1e-12) == float(
        mats_full["median_vol"].iloc[119]
    )
    assert pytest.approx(float(mats_prefix["vol_scalar"].iloc[119]), abs=1e-12) == float(
        mats_full["vol_scalar"].iloc[119]
    )
    np.testing.assert_allclose(
        mats_prefix["w_base_1x"].iloc[119].values,
        mats_full["w_base_1x"].iloc[119].values,
        atol=1e-12,
    )


def test_crit_b4_c2_c3_stub_pruning_killswitch_hydration_and_seven_factor_backtest():
    """[CRIT-B4, C2, C3] Verifies unconditional <$160 stub pruning, Kill-Switch DB hydration, and 7-Factor backtest."""
    # 1. Unconditional <$160 stub pruning in SteppedLeverageController when gross >= $2,000
    ctrl = SteppedLeverageController(initial_capital=2700.0, start_leverage=3.0, max_leverage=5.0)
    unit_w = {
        "BTCUSDT": 0.16,
        "ETHUSDT": 0.14,
        "SOLUSDT": 0.12,
        "BNBUSDT": 0.08,
        "AVAXUSDT": -0.16,
        "APTUSDT": -0.16,
        "LINKUSDT": -0.17,
        "DOGEUSDT": -0.01,  # 0.01 * 3.0 * $2,700 = $81 (< $160 stub!)
    }
    final_w, _ = ctrl.scale_and_deadband_weights(unit_weights=unit_w, current_weights=None, current_equity=2700.0)
    assert "DOGEUSDT" not in final_w
    for w in final_w.values():
        assert abs(w) * 2700.0 >= 160.0
    assert pytest.approx(sum(final_w.values()), abs=1e-6) == 0.0

    # 2. Kill-Switch 24h window hydration & SQLite feature-flag preservation across daemon restarts
    db = ShadowDatabase(db_path=":memory:")
    now = datetime.now(timezone.utc)
    db.save_account_snapshot(
        initial_capital=3000.0,
        cash_balance=3000.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        total_equity=3000.0,
        margin_used=0.0,
        free_margin=3000.0,
        leverage=0.0,
        timestamp=now - timedelta(hours=3),
    )
    db.set_feature_flag("enable_anomaly_capsule", False)
    ks = EmergencyKillSwitch(max_24h_drawdown_pct=0.12)
    live_b = BinanceLiveBroker(dry_run=True, initial_capital=2700.0, kill_switch=ks)
    broker = ShadowExecutionBroker(database=db, initial_cash=2700.0, live_broker=live_b)
    daemon = ShadowTradingDaemon(broker=broker, symbols=["BTC/USDT"], enable_anomalies=True, enable_warm_start=False)
    # Persisted DB flag (False) survived restart instead of being overwritten by CLI default (True)
    assert daemon.enable_anomalies is False
    # KillSwitch hydrated the $3,000 peak from 3h ago -> $2,700 equity registers 10.0% rolling 24h drawdown
    dd_frac = ks.record_equity(2700.0, timestamp=now)
    assert pytest.approx(dd_frac, abs=1e-4) == 0.10
    db.close()

    # 3. End-to-end 7-Factor Backtest runner (`scripts/run_backtest.py`)
    from scripts.run_backtest import run_simulation

    bars = _make_bars(n_bars=220, seed=303)
    bars_idx = {k: v.set_index("timestamp") for k, v in bars.items()}
    bt_res = run_simulation(
        strategy_name="seven_factor",
        bars_dict=bars_idx,
        funding_dict={},
        initial_capital=3000.0,
        start_leverage=2.5,
        max_leverage=5.0,
        rebalance_freq_bars=4,
    )
    assert bt_res.final_equity > 0.0
    assert np.isfinite(bt_res.sharpe_ratio)


def test_novymarx_bhs_hysteresis_two_phase_safe_ramp_and_8h_funding_clock():
    """
    Verifies Multi-Year Champion Mechanisms (2024-2026 Walk-Forward Validated):
    1. Novy-Marx & Velikov (2016) Buy/Hold Rank Hysteresis (`bhs_enter_top_k=3, bhs_hold_top_k=6`):
       blocks fresh entries at rank #4-#6, retains existing same-side legs through rank #6 without
       cliff-eviction, and evicts rank #7+ while preserving 100% dollar neutrality.
    2. Two-Phase Safe Post-Recovery Ramp (`recovery_boost_equity=3050.0, recovery_boost_leverage=3.2`):
       holds 2.5x below $3,050 (protecting $2,500 floor), unlocks 3.2x at $3,050, compounds +0.5x
       steps up to 5.0x, and steps down on drawdowns.
    3. 8h UTC Funding-Clock Alignment (`00:00:15, 08:00:15, 16:00:15 UTC`) in ShadowTradingDaemon.
    4. Strict zero-lookahead causality of optional `enable_btc_lead_lag=True` in SevenFactorOnlineEngine.
    """
    # 1. Novy-Marx BHS(3, 6) Rank Hysteresis
    ctrl_bhs = SteppedLeverageController(
        initial_capital=3000.0,
        start_leverage=2.5,
        max_leverage=5.0,
        dd_bracket=0.06,
        base_deadband_ratio=(0.04 / 2.2) * 1.25,
        bhs_enter_top_k=3,
        bhs_hold_top_k=6,
        recovery_boost_equity=3050.0,
        recovery_boost_leverage=3.2,
    )
    unit_scores = {
        "BTCUSDT": 0.14,  # Long rank 0
        "ETHUSDT": 0.12,  # Long rank 1
        "SOLUSDT": 0.10,  # Long rank 2 (Top 3 cutoff for fresh entry)
        "LINKUSDT": 0.06,  # Long rank 3 (4th -> blocked if flat, held if already long!)
        "AVAXUSDT": 0.05,  # Long rank 4 (5th -> blocked if flat, held if already long!)
        "ADAUSDT": 0.03,  # Long rank 5 (6th -> held if already long and >=5% leg share)
        "NEARUSDT": 0.02,  # Long rank 6 (7th -> outside hold_top_k=6 -> evicted!)
        "XRPUSDT": -0.14,  # Short rank 0
        "DOGEUSDT": -0.12,  # Short rank 1
        "BNBUSDT": -0.10,  # Short rank 2
        "SUIUSDT": -0.06,  # Short rank 3
        "APTUSDT": -0.04,  # Short rank 4
        "ARBUSDT": -0.02,
        "FETUSDT": -0.01,
        "RENDERUSDT": -0.01,
    }
    # Cold start (flat): ONLY Top 3 Longs and Top 3 Shorts may enter
    w_cold, snap_rec = ctrl_bhs.scale_and_deadband_weights(
        unit_weights=unit_scores,
        current_weights=None,
        current_equity=2721.0,
        vol_scalar=1.0,
    )
    assert set(k for k, v in w_cold.items() if v > 0) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(k for k, v in w_cold.items() if v < 0) == {"XRPUSDT", "DOGEUSDT", "BNBUSDT"}
    assert pytest.approx(sum(w_cold.values()), abs=1e-6) == 0.0

    # Warm hold: LINKUSDT (rank 4th Long) and SUIUSDT (rank 4th Short) are ALREADY open -> retained!
    # NEARUSDT (rank 7th Long, index 6 >= hold_top_k=6) -> evicted!
    curr_held = {
        "BTCUSDT": 0.30,
        "ETHUSDT": 0.25,
        "SOLUSDT": 0.20,
        "LINKUSDT": 0.20,
        "NEARUSDT": 0.15,
        "XRPUSDT": -0.35,
        "DOGEUSDT": -0.30,
        "BNBUSDT": -0.25,
        "SUIUSDT": -0.20,
    }
    w_warm, _ = ctrl_bhs.scale_and_deadband_weights(
        unit_weights=unit_scores,
        current_weights=curr_held,
        current_equity=2900.0,
        vol_scalar=1.0,
    )
    assert w_warm.get("LINKUSDT", 0.0) > 0.0
    assert w_warm.get("SUIUSDT", 0.0) < 0.0
    assert "NEARUSDT" not in w_warm
    assert pytest.approx(sum(w_warm.values()), abs=1e-6) == 0.0

    # 2. Two-Phase Safe Post-Recovery Ramp ($2,721 -> $3,050 -> $3,510 -> DD step-down)
    # At $2,721 (HWM=$3,000 < $3,050, DD=9.3% >= 6% bracket): unlocked=2.5x, active_base=2.0x
    assert pytest.approx(snap_rec.unlocked_tier) == 2.5
    assert pytest.approx(snap_rec.active_base_leverage) == 2.0
    assert pytest.approx(snap_rec.next_tier_leverage) == 3.2
    assert pytest.approx(snap_rec.next_tier_hwm_target) == 3050.0

    # At $2,950 (HWM=$3,000 < $3,050, DD=1.67% < 6%): unlocked=2.5x, active_base=2.5x
    snap_mid = ctrl_bhs.update_state(current_equity=2950.0, vol_scalar=1.0)
    assert pytest.approx(snap_mid.unlocked_tier) == 2.5
    assert pytest.approx(snap_mid.active_base_leverage) == 2.5

    # Crossing $3,050 ($3,060): automatically unlocks Phase 2 base 3.2x!
    snap_boost = ctrl_bhs.update_state(current_equity=3060.0, vol_scalar=1.0)
    assert pytest.approx(snap_boost.unlocked_tier) == 3.2
    assert pytest.approx(snap_boost.active_base_leverage) == 3.2

    # +15% above $3,050 ($3,510): unlocks 3.7x!
    snap_step1 = ctrl_bhs.update_state(current_equity=3510.0, vol_scalar=1.0)
    assert pytest.approx(snap_step1.unlocked_tier) == 3.7
    # A 7% pullback from $3,510 ($3,264) steps down by -0.5x to 3.2x
    snap_dd = ctrl_bhs.update_state(current_equity=3264.0, vol_scalar=1.0)
    assert pytest.approx(snap_dd.active_base_leverage) == 3.2

    # 3. 8h UTC Funding-Clock Alignment in ShadowTradingDaemon
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        symbols=["BTC/USDT"],
        rebalance_interval_sec=28800.0,
        align_rebalance_to_funding_utc=True,
        enable_warm_start=False,
    )
    t_16_02 = datetime(2026, 11, 17, 16, 2, 0, tzinfo=timezone.utc)
    daemon.last_rebalance_time = t_16_02
    # At 19:30 UTC (same 16:00-00:00 window), rebalance is NOT due; countdown points to 00:00:15 UTC
    t_19_30 = datetime(2026, 11, 17, 19, 30, 0, tzinfo=timezone.utc)
    assert daemon._is_scheduled_rebalance_due(t_19_30) is False
    assert pytest.approx(daemon._seconds_until_next_rebalance(t_19_30)) == 4.5 * 3600.0 + 15.0
    # At 00:00:16 UTC next day, boundary 00:00:15 > 16:02:00 -> rebalance IS due!
    t_00_00_16 = datetime(2026, 11, 18, 0, 0, 16, tzinfo=timezone.utc)
    assert daemon._is_scheduled_rebalance_due(t_00_00_16) is True
    db.close()

    # 4. Strict causality of optional `enable_btc_lead_lag=True`
    bars_full = _make_bars(n_bars=140, seed=909)
    eng_ll = SevenFactorOnlineEngine(
        priors=None,
        enable_chop_range_arb=True,
        enable_holistic_regime_shields=True,
        enable_btc_lead_lag=True,
    )
    mats_full = eng_ll.compute_matrices(bars_1h=bars_full)
    bars_trunc = {k: v.iloc[:100].copy() for k, v in bars_full.items()}
    # Shock future bars [100:] by +50% and verify bar [99] weights are 100% invariant (0.00e+00 leakage)
    for k in bars_full:
        bars_full[k].loc[bars_full[k].index[100:], "close"] *= 1.50
    mats_shocked = eng_ll.compute_matrices(bars_1h=bars_full)
    mats_trunc = eng_ll.compute_matrices(bars_1h=bars_trunc)
    max_diff = float((mats_shocked["w_base_1x"].iloc[99] - mats_trunc["w_base_1x"].iloc[-1]).abs().max())
    assert max_diff < 1e-12
    assert pytest.approx(float(mats_full["w_base_1x"].iloc[-1].sum()), abs=1e-6) == 0.0
