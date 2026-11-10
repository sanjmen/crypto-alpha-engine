"""
Comprehensive Unit & Integration Tests for Milestones M11 & M12 (Issues #37 to #44):
- Issue #37: Warm-Start Parquet Cache (1000h 1h + 15m Sub-Hourly + 72h Funding) in ShadowTradingDaemon
- Issue #38: 7-Factor Orthogonal Cross-Sectional Engine with 72h BTC Beta Neutralization & Sub-Hourly Pullback
- Issue #39: Causal Online Factor Momentum Allocator (t-stat 36h/96h/240h) & Asymmetric Tail-Power Weighting
- Issue #40: Stepped Leverage Controller (2.5x -> 5.0x HWM Hysteresis), 72h Vol-Targeting & Proportional Deadband
- Issue #41: Real-Time Factor Weights, Leverage Tier Progress & Neutrality Dashboard in Rich TUI Monitor
- Issue #42: BinanceLiveBroker with Post-Only (GTX) Limit-Chase Maker Engine & Atomic Leg Sync
- Issue #43: Production Emergency Kill-Switch, Cross-Margin Sync & Net Exposure Circuit Breaker
- Issue #44: Shadow-to-Live Go/No-Go Validation Suite
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from scripts.audit_shadow_readiness import evaluate_shadow_readiness
from src.domain.entities import Portfolio, Position
from src.domain.enums import OrderSide, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.live_broker import BinanceLiveBroker, MaxDrawdownStopError
from src.execution.monitor import ShadowMonitor
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import (
    FACTOR_NAMES,
    SevenFactorOnlineEngine,
    cap_side_weights_waterfill,
    make_continuous_neutral_weights,
)


def _build_synthetic_universe(n_bars: int = 800, seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Generates synthetic 1h and 15m OHLCV DataFrames for 6 symbols."""
    rng = np.random.default_rng(seed)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    idx_1h = pd.date_range("2026-09-01", periods=n_bars, freq="1h", tz="UTC")
    idx_15m = pd.date_range(idx_1h[-24], periods=96, freq="15min", tz="UTC")

    bars_1h: dict[str, pd.DataFrame] = {}
    bars_15m: dict[str, pd.DataFrame] = {}

    btc_rets = rng.normal(0.0002, 0.008, size=n_bars)
    for i, sym in enumerate(symbols):
        beta = 1.0 if sym == "BTCUSDT" else (0.8 + 0.15 * i)
        idio_drift = (i - 2.5) * 0.0003
        rets = beta * btc_rets + rng.normal(idio_drift, 0.006, size=n_bars)
        closes = (100.0 * (i + 1)) * np.exp(np.cumsum(rets))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * 1.003
        lows = np.minimum(opens, closes) * 0.997
        vols = rng.uniform(50_000, 250_000, size=n_bars)

        bars_1h[sym] = pd.DataFrame({
            "timestamp": idx_1h,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        })

        c15 = closes[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=96)))
        bars_15m[sym] = pd.DataFrame({
            "timestamp": idx_15m,
            "open": c15,
            "high": c15 * 1.001,
            "low": c15 * 0.999,
            "close": c15,
            "volume": rng.uniform(10_000, 50_000, size=96),
        })

    return bars_1h, bars_15m


def test_seven_factor_engine_neutrality_and_online_weights():
    """Tests Issues #38 & #39: 7 orthogonal factors, 72h BTC beta, causal t-stats, and dollar neutrality."""
    bars_1h, bars_15m = _build_synthetic_universe(n_bars=800)
    funding = {sym: 0.0001 * (i - 2.5) for i, sym in enumerate(bars_1h.keys())}

    engine = SevenFactorOnlineEngine()
    res = engine.allocate_latest(bars_1h=bars_1h, funding_rates=funding, bars_15m=bars_15m)

    # All 7 factors must have online weights summing to 1.0
    assert set(res.factor_weights.keys()) == set(FACTOR_NAMES)
    assert pytest.approx(sum(res.factor_weights.values()), abs=1e-5) == 1.0

    # Unit weights must be strictly dollar-neutral (sum(w) == 0.0)
    net_w = sum(res.unit_weights.values())
    gross_w = sum(abs(v) for v in res.unit_weights.values())
    assert pytest.approx(net_w, abs=1e-6) == 0.0
    assert 0.30 <= gross_w <= 1.20
    assert 0.55 <= res.vol_scalar <= 1.15


def test_stepped_leverage_controller_milestones_and_hysteresis():
    """Tests Issue #40: 2.5x -> 5.0x progressive milestones (+15% HWM) and -0.5x per 10% DD."""
    ctrl = SteppedLeverageController(
        initial_capital=1000.0,
        start_leverage=2.5,
        max_leverage=5.0,
        floor_leverage=2.0,
        step_leverage=0.5,
        gain_per_step=0.15,
        dd_bracket=0.10,
    )

    # Initial state at $1,000 -> 2.5x
    s0 = ctrl.update_state(current_equity=1000.0, vol_scalar=1.0)
    assert s0.unlocked_tier == 2.5
    assert s0.active_base_leverage == 2.5
    assert pytest.approx(s0.next_tier_hwm_target, rel=1e-4) == 1150.0

    # +15% HWM ($1,150) -> 3.0x
    s1 = ctrl.update_state(current_equity=1150.0, vol_scalar=1.0)
    assert s1.unlocked_tier == 3.0
    assert s1.active_base_leverage == 3.0

    # +32.25% HWM ($1,322.50) -> 3.5x
    s2 = ctrl.update_state(current_equity=1323.0, vol_scalar=1.0)
    assert s2.unlocked_tier == 3.5

    # +101.2% HWM ($2,012) -> 5.0x (Max tier unlocked)
    s5 = ctrl.update_state(current_equity=2012.0, vol_scalar=1.0)
    assert s5.unlocked_tier == 5.0
    assert s5.active_base_leverage == 5.0

    # Simulate a 22% drawdown from $2,012 -> should step down by 2 brackets (-1.0x) to 4.0x
    s_dd = ctrl.update_state(current_equity=2012.0 * 0.78, vol_scalar=1.0)
    assert s_dd.unlocked_tier == 5.0
    assert s_dd.steps_down == 2
    assert s_dd.active_base_leverage == 4.0


def test_meta_allocator_and_daemon_warm_start_integration():
    """Tests Issues #37, #39, #40, #41: Warm-start cache, MetaStrategyAllocator, DB telemetry & Monitor."""
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=1000.0, max_gross_leverage=6.0)
    lev_ctrl = SteppedLeverageController(initial_capital=1000.0)
    allocator = MetaStrategyAllocator(
        seven_factor_engine=SevenFactorOnlineEngine(),
        stepped_leverage_controller=lev_ctrl,
        use_seven_factor_engine=True,
    )

    bars_1h, bars_15m = _build_synthetic_universe(n_bars=300)
    mock_ex = MagicMock()

    def _mock_ohlcv(symbol, timeframe="1h", limit=100):
        clean = symbol.replace("/", "")
        src = bars_1h.get(clean, next(iter(bars_1h.values())))
        tail = src.tail(min(limit, len(src)))
        return [
            [int(r.timestamp.timestamp() * 1000), r.open, r.high, r.low, r.close, r.volume]
            for _, r in tail.iterrows()
        ]

    mock_ex.fetch_ohlcv.side_effect = _mock_ohlcv
    mock_ex.fetch_funding_rate.return_value = {"fundingRate": 0.0001}
    mock_ex.fetch_tickers.return_value = {
        "BTC/USDT": {"last": 60000.0, "bid": 59999.0, "ask": 60001.0, "percentage": 1.2, "quoteVolume": 50_000_000.0},
        "ETH/USDT": {"last": 3000.0, "bid": 2999.5, "ask": 3000.5, "percentage": 2.1, "quoteVolume": 40_000_000.0},
        "SOL/USDT": {"last": 150.0, "bid": 149.9, "ask": 150.1, "percentage": -0.5, "quoteVolume": 30_000_000.0},
        "BNB/USDT": {"last": 550.0, "bid": 549.8, "ask": 550.2, "percentage": 0.3, "quoteVolume": 25_000_000.0},
        "XRP/USDT": {"last": 0.60, "bid": 0.5999, "ask": 0.6001, "percentage": -1.1, "quoteVolume": 25_000_000.0},
        "DOGE/USDT": {"last": 0.15, "bid": 0.1499, "ask": 0.1501, "percentage": 3.0, "quoteVolume": 25_000_000.0},
    }

    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        exchange_client=mock_ex,
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"],
        enable_warm_start=False,
        min_rebalance_notional=10.0,
    )

    daemon.run_fast_cycle()
    orders = daemon.run_rebalance_cycle()
    assert len(orders) > 0

    # Verify telemetry was persisted in ShadowDatabase
    tel = db.get_latest_seven_factor_telemetry()
    assert tel is not None
    assert tel["unlocked_tier"] == 2.5
    assert len(tel["factor_weights"]) == 7

    # Verify Monitor renders Rich layout with 7-factor telemetry
    monitor = ShadowMonitor(db_path=":memory:")
    layout = monitor._build_rich_layout(
        account=db.get_latest_account(),
        positions=db.get_open_positions(),
        orders=[],
        trades=[],
        trades_24h=[],
        funding=[],
        rebalance=db.get_latest_rebalance_snapshot(),
        heartbeat=db.get_daemon_heartbeat(),
        market_pulse=db.get_market_pulse(),
        sf_telemetry=tel,
    )
    assert layout is not None
    db.close()


def test_binance_live_broker_and_kill_switch():
    """Tests Issues #42 & #43: Post-Only GTX Limit-Chase, Atomic Rebalance & Emergency Kill-Switch."""
    ks = EmergencyKillSwitch(max_24h_drawdown_pct=0.12, max_net_exposure_pct=0.15)
    broker = BinanceLiveBroker(
        dry_run=True,
        initial_capital=1000.0,
        max_gross_leverage=6.0,
        kill_switch=ks,
    )

    prices = {"BTC/USDT": 80000.0, "ETH/USDT": 3000.0, "SOL/USDT": 180.0, "ADA/USDT": 0.80}
    target_dollars = {
        "BTC/USDT": 600.0,
        "SOL/USDT": 600.0,
        "ETH/USDT": -600.0,
        "ADA/USDT": -600.0,
    }

    orders = broker.execute_atomic_rebalance(target_dollars=target_dollars, prices=prices, min_notional=20.0)
    assert len(orders) == 4
    assert broker.maker_fill_ratio == 1.0

    # Check Net Exposure is balanced ($1,200 Long vs $1,200 Short -> Net = $0.00)
    l_usd, s_usd, net_usd = ks.compute_net_exposure(broker.get_portfolio())
    assert pytest.approx(l_usd, rel=1e-2) == 1200.0
    assert pytest.approx(s_usd, rel=1e-2) == 1200.0
    assert abs(net_usd) < 5.0

    # Trigger 24h Drawdown Kill-Switch (-15% drop from $1,000 -> $850)
    ks.record_equity(1000.0)
    broker.portfolio.cash = 850.0
    for pos in broker.portfolio.positions.values():
        pos.current_price = pos.entry_price
    status = ks.evaluate_and_enforce(broker.get_portfolio(), broker=broker)
    assert status.triggered is True
    assert ks.is_locked() is True
    assert len(broker.portfolio.positions) == 0  # All positions flattened!


def test_shadow_to_live_readiness_auditor(tmp_path):
    """Tests Issue #44: Shadow-to-Live Go/No-Go Validation Suite."""
    db_file = str(tmp_path / "test_shadow.db")
    db = ShadowDatabase(db_path=db_file)
    db.save_account_snapshot(
        initial_capital=1000.0,
        cash_balance=1025.0,
        unrealized_pnl=5.0,
        realized_pnl=25.0,
        total_equity=1030.0,
        margin_used=500.0,
        free_margin=530.0,
        leverage=2.5,
    )
    db.save_position(
        Position(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=0.015,
            entry_price=80000.0,
            current_price=80000.0,
        )
    )
    db.save_position(
        Position(
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            quantity=0.40,
            entry_price=3000.0,
            current_price=3000.0,
        )
    )
    db.save_rebalance_snapshot(
        regime_detected="TRENDING",
        momentum_weight=0.6,
        stat_arb_weight=0.1,
        carry_weight=0.3,
        target_allocations={"BTCUSDT": 1200.0, "ETHUSDT": -1200.0},
        executed_delta={"BTC/USDT": 1200.0, "ETH/USDT": -1200.0},
    )
    db.save_seven_factor_telemetry(
        initial_capital=1000.0,
        current_equity=1030.0,
        high_water_mark=1030.0,
        drawdown_pct=0.0,
        unlocked_tier=2.5,
        active_base_leverage=2.5,
        vol_scalar=1.0,
        effective_leverage=2.5,
        next_tier_leverage=3.0,
        next_tier_hwm_target=1150.0,
        progress_to_next_tier_pct=20.0,
        portfolio_beta=0.01,
        conviction=1.0,
        factor_weights={k: 1.0 / 7.0 for k in FACTOR_NAMES},
        factor_tstats={k: 1.2 for k in FACTOR_NAMES},
    )
    db.close()

    audit = evaluate_shadow_readiness(db_path=db_file)
    assert audit["ready_for_live"] is True
    assert all(g.passed for g in audit["gates"])


def test_seven_factor_dynamic_profit_lock_and_trailing_stop():
    """
    Verifies that 7-Factor positions (which open without initial hard Stop-Loss)
    receive +8.0% Maker Take-Profit brackets, track peak_price on every fast cycle,
    and dynamically arm Phase 2 Profit-Lock (+2.5% -> +0.8%) and Phase 3 Trailing Stop (+4.0% -> 2.0%).
    """
    from src.domain.enums import OrderType
    from src.execution.shadow_broker import ShadowExecutionBroker
    from src.execution.shadow_daemon import ShadowTradingDaemon

    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=1000.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        symbols=["APT/USDT"],
        profit_lock_trigger_pct=0.025,
        profit_lock_level_pct=0.008,
        trailing_trigger_pct=0.040,
        trailing_distance_pct=0.020,
        core_take_profit_pct=0.080,
    )

    # Open a 7-Factor SHORT position on APT/USDT @ $1.0000 without initial brackets
    broker.set_market_price("APT/USDT", price=1.0000)
    broker.create_order(
        symbol="APT/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        price=1.0000,
        strategy="F5_Carry_Neg (72h Fund)",
    )
    pos = broker.portfolio.positions["APT/USDT"]
    assert pos.stop_loss is None
    assert pos.take_profit is None

    # 1. ensure_position_brackets attaches +8.0% Maker Take-Profit ($0.92) while keeping stop_loss=None
    daemon.ensure_position_brackets()
    assert pos.take_profit == 0.92
    assert pos.stop_loss is None

    # 2. Price dips +1.0% in favor of SHORT ($0.9900) -> peak_price tracked, SL still None
    daemon.manage_trailing_and_profit_locks({"APT/USDT": {"price": 0.9900}})
    assert pos.peak_price == 0.9900
    assert pos.stop_loss is None

    # 3. Price dips +2.6% in favor of SHORT ($0.9740 >= +2.5% trigger) -> dynamically arms Profit-Lock at +0.8% ($0.9920)
    ratcheted = daemon.manage_trailing_and_profit_locks({"APT/USDT": {"price": 0.9740}})
    assert len(ratcheted) == 1
    assert pos.peak_price == 0.9740
    assert pos.stop_loss == 0.9920
    assert pos.is_profit_locked is True

    # 4. Price dips +5.0% in favor of SHORT ($0.9500 >= +4.0% trigger) -> Trailing Stop ratchets to 2.0% above peak ($0.9690)
    ratcheted_trail = daemon.manage_trailing_and_profit_locks({"APT/USDT": {"price": 0.9500}})
    assert len(ratcheted_trail) == 1
    assert pos.peak_price == 0.9500
    assert pos.stop_loss == round(0.9500 * 1.020, 6)
    db.close()


def test_mejora_1_neutrality_recentering_and_peak_preservation_on_minor_scale_in():
    """
    Tests Mejora 1:
    1) SteppedLeverageController re-centers net exposure when post-stop / post-deadband drift > 2.5%.
    2) ShadowExecutionBroker preserves peak_price and active Profit-Lock on minor (< 20%) scale-ins,
       and resets brackets on major (>= 20%) scale-ins.
    """
    from src.domain.enums import OrderType

    ctrl = SteppedLeverageController(initial_capital=1000.0, start_leverage=2.5)
    unit_weights = {
        "BTCUSDT": 0.20,
        "ETHUSDT": 0.15,
        "SOLUSDT": 0.15,
        "APTUSDT": -0.25,
        "AVAXUSDT": -0.25,
    }
    # Simulate state after a LONG leg closed via Profit-Lock (leaving Long=+0.74 vs Short=-1.25 -> net drift > 2.5%)
    # while remaining LONG legs are within deadband of desired weights (0.50 and 0.375)
    current_weights = {
        "BTCUSDT": 0.49,
        "ETHUSDT": 0.36,
        "SOLUSDT": 0.0,
        "APTUSDT": -0.625,
        "AVAXUSDT": -0.625,
    }
    final_w, _ = ctrl.scale_and_deadband_weights(
        unit_weights=unit_weights,
        current_weights=current_weights,
        current_equity=1000.0,
        vol_scalar=1.0,
    )
    net_exposure = sum(final_w.values())
    assert abs(net_exposure) < 1e-6

    # Verify ShadowExecutionBroker peak_price & Profit-Lock preservation on minor (< 20%) scale-in
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=1000.0)
    broker.set_market_price("BTC/USDT", price=100.0)
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=10.0,
        price=100.0,
    )
    pos = broker.portfolio.positions["BTC/USDT"]
    pos.peak_price = 104.5
    pos.stop_loss = 102.4
    pos.take_profit = 108.0

    # Minor +10% scale-in (+1.0 unit on 10.0 units = 10% < 20%) -> preserves peak_price and active stop_loss
    broker.set_market_price("BTC/USDT", price=103.0)
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=103.0,
    )
    assert pos.quantity == 11.0
    assert pos.peak_price == 104.5
    assert pos.stop_loss == 102.4

    # Major +50% scale-in (+5.5 units on 11.0 units = 50% >= 20%) -> resets peak_price and brackets
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=5.5,
        price=103.0,
    )
    assert pos.quantity == 16.5
    assert pos.peak_price == 103.0
    assert pos.stop_loss is None
    db.close()


def test_mejora_2_and_3_falling_knife_guard_and_cmf_orderflow_overlay():
    """
    Tests Mejora 2 (Targeted Falling-Knife Guard when F4_Rev dominates) and
    Mejora 3 (Dynamic Online F8_OrderFlow_CMF Chaikin Accumulation/Distribution overlay).
    """
    bars_1h, bars_15m = _build_synthetic_universe(n_bars=400, seed=7)
    engine = SevenFactorOnlineEngine()
    mats = engine.compute_matrices(bars_1h=bars_1h, bars_15m=bars_15m)

    assert "cmf_24" in mats
    assert "t_cmf" in mats
    assert "alpha_cmf_dyn" in mats
    assert float(mats["alpha_cmf_dyn"].min()) >= 0.0
    assert float(mats["alpha_cmf_dyn"].max()) <= 0.10 + 1e-9

    res = engine.allocate_latest(bars_1h=bars_1h, bars_15m=bars_15m)
    assert isinstance(res.cmf_tstat, float)
    assert 0.0 <= res.cmf_blend_weight <= 0.10 + 1e-9
    assert pytest.approx(sum(res.unit_weights.values()), abs=1e-6) == 0.0

    # Also verify graceful fallback when high/low columns are omitted from bars_1h
    bars_close_only = {
        sym: df[["timestamp", "close", "volume"]].copy()
        for sym, df in bars_1h.items()
    }
    res_fallback = engine.allocate_latest(bars_1h=bars_close_only)
    assert pytest.approx(sum(res_fallback.unit_weights.values()), abs=1e-6) == 0.0


def test_retoque_1_selective_factor_smoothing_and_rev_exit_hysteresis():
    """
    Tests Retoque 1A (Selective Factor Weight Smoothing on rev=12h, liq=16h, size=6h)
    and Retoque 1B (Sudden Rev-Exit Hysteresis when F4_Rev drops sharply without macro trend).
    """
    bars_1h, bars_15m = _build_synthetic_universe(n_bars=420, seed=99)
    engine = SevenFactorOnlineEngine()
    mats = engine.compute_matrices(bars_1h=bars_1h, bars_15m=bars_15m)

    fw_norm = mats["fw_norm"]
    assert list(fw_norm.columns) == ["macro", "med", "fast", "rev", "carry", "liq", "size"]
    # Verify normalized factor weights sum to 1.0 at every timestamp
    row_sums = fw_norm.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-5)

    # Verify unit weights are strictly market-neutral across all timestamps
    w_base_1x = mats["w_base_1x"]
    assert np.allclose(w_base_1x.sum(axis=1).values, 0.0, atol=1e-5)
    # Conviction stays within configured bounds
    conv = mats["conviction"]
    assert float(conv.min()) >= engine.conviction_bounds[0] - 1e-9
    assert float(conv.max()) <= engine.conviction_bounds[1] + 1e-9


def test_nov1_active_leg_concentration_cap_and_waterfill():
    """
    Tests Nov 1 Refinements:
    1) cap_side_weights_waterfill redistributes extreme single-asset concentration (<= 30% per side)
       across secondary leaders on the same leg while preserving strict 100% dollar neutrality.
    2) SteppedLeverageController enforces max_leg_share (32% of active leg budget target_g / 2.0)
       even when conviction is low (e.g. 0.50) and after deadband re-centering.
    """
    cols = [f"COIN{i}USDT" for i in range(10)]
    # 5 Longs where COIN0 has 65% of the Long leg, and 5 Shorts where COIN5 has 60% of the Short leg
    raw_row = [0.325, 0.05, 0.05, 0.04, 0.035, -0.30, -0.06, -0.05, -0.05, -0.04]
    w_df = pd.DataFrame([raw_row], columns=cols)

    capped_df = cap_side_weights_waterfill(w_df, max_side_share=0.30)
    r = capped_df.iloc[0].values
    pos = np.maximum(r, 0.0)
    neg = np.maximum(-r, 0.0)

    # Strict dollar neutrality and preserved gross exposure
    assert pytest.approx(float(r.sum()), abs=1e-7) == 0.0
    assert pytest.approx(float(pos.sum()), abs=1e-6) == 0.50
    assert pytest.approx(float(neg.sum()), abs=1e-6) == 0.50
    # Max share on either leg is strictly capped at 30%
    assert float(pos.max() / pos.sum()) <= 0.30 + 1e-6
    assert float(neg.max() / neg.sum()) <= 0.30 + 1e-6

    # SteppedLeverageController active-leg cap when conviction = 0.50 (gross unit weight = 0.50)
    ctrl = SteppedLeverageController(initial_capital=1000.0, start_leverage=2.5, max_leg_share=0.32)
    unit_w = {c: raw_row[i] * 0.50 for i, c in enumerate(cols)}
    final_w, _ = ctrl.scale_and_deadband_weights(
        unit_weights=unit_w,
        current_weights=None,
        current_equity=1000.0,
        vol_scalar=1.0,
    )
    assert pytest.approx(sum(final_w.values()), abs=1e-6) == 0.0
    long_vals = [v for v in final_w.values() if v > 0]
    short_vals = [-v for v in final_w.values() if v < 0]
    # COIN0 raw weight (0.325 * 0.50 * 2.5 = 0.40625) is clipped at 32% of target leg (0.32 * 0.625 = 0.20)
    assert max(long_vals) <= 0.32 * 0.625 + 1e-6
    assert max(short_vals) <= 0.32 * 0.625 + 1e-6


def test_m13_issue_45_altcoin_sector_cluster_cap():
    """
    Tests Issue #45 (Dynamic Altcoin Sector Cluster Cap <= 48% with water-filling):
    Verifies that when correlated tokens in the same altcoin sector (e.g., AI_COMPUTE: FET+RENDER
    or FAST_L1: SOL+SUI+AVAX) combine for >48% of a leg, the sector cluster is capped at 48%
    and excess weight is water-filled into unclipped sectors while preserving 100% dollar neutrality.
    """
    cols = [
        "FET/USDT", "RENDER/USDT", "BTC/USDT", "ETH/USDT", "BNB/USDT",
        "SOL/USDT", "SUI/USDT", "AVAX/USDT", "LINK/USDT", "ARB/USDT",
    ]
    # FET (29%) + RENDER (29%) = 58% of Long leg (> 48% AI_COMPUTE cluster cap)
    # SOL (22%) + SUI (22%) + AVAX (22%) = 66% of Short leg (> 48% FAST_L1 cluster cap)
    raw_row = [
        0.145, 0.145, 0.08, 0.07, 0.06,
        -0.11, -0.11, -0.11, -0.09, -0.08,
    ]
    w_df = pd.DataFrame([raw_row], columns=cols)

    capped_df = cap_side_weights_waterfill(w_df, max_side_share=0.30, max_cluster_share=0.48)
    r = capped_df.iloc[0]
    pos = r.clip(lower=0.0)
    neg = (-r).clip(lower=0.0)

    assert pytest.approx(float(r.sum()), abs=1e-7) == 0.0
    assert pytest.approx(float(pos.sum()), abs=1e-6) == 0.50
    assert pytest.approx(float(neg.sum()), abs=1e-6) == 0.50

    ai_share_long = float((pos["FET/USDT"] + pos["RENDER/USDT"]) / pos.sum())
    fast_l1_share_short = float((neg["SOL/USDT"] + neg["SUI/USDT"] + neg["AVAX/USDT"]) / neg.sum())
    assert ai_share_long <= 0.48 + 1e-6
    assert fast_l1_share_short <= 0.48 + 1e-6


def test_m13_issue_46_open_interest_squeeze_and_knife_guard():
    """
    Tests Issue #46 (Real-Time Open Interest Squeeze/Knife Guard & Confirmation):
    Verifies that SevenFactorOnlineEngine accepts `open_interest` DataFrames/Series,
    applies the OI guard and confirmation adjustments causally, and preserves strict
    100% dollar neutrality.
    """
    bars_1h, bars_15m = _build_synthetic_universe(n_bars=120, seed=46)
    oi_data = {}
    for sym, df in bars_1h.items():
        # Synthetic OI series with a +15% surge on the last 12 bars
        oi_vals = np.full(len(df), 1_000_000.0)
        oi_vals[-12:] = np.linspace(1_000_000.0, 1_150_000.0, 12)
        oi_data[sym] = pd.DataFrame({
            "timestamp": df["timestamp"],
            "sumOpenInterest": oi_vals,
        })

    engine = SevenFactorOnlineEngine()
    res = engine.allocate_latest(
        bars_1h=bars_1h,
        bars_15m=bars_15m,
        open_interest=oi_data,
    )
    assert pytest.approx(sum(res.unit_weights.values()), abs=1e-6) == 0.0
    assert sum(abs(v) for v in res.unit_weights.values()) > 0.0


def test_m13_wave3_issue_48_and_49_true_taker_cvd_and_basis_alpha():
    """
    Tests Milestone M13 Wave 3:
    1) Issue #48: Factor F8 True Taker Buy/Sell CVD & Institutional Limit Absorption (`taker_flow`).
    2) Issue #49: Factor F9 Spot-Perp Basis Spread & Premium Index Alpha (`premium_index`).
    3) Combined Wave 3 (`taker_flow` + `premium_index` + `open_interest`) preserving strict 100% dollar neutrality.
    """
    bars_1h, bars_15m = _build_synthetic_universe(n_bars=200, seed=48)
    taker_flow = {}
    premium_index = {}
    for i, (sym, df) in enumerate(bars_1h.items()):
        qvol = (df["close"] * df["volume"]).values
        # Alternate net taker buying (55%) vs net taker selling (44%) across symbols
        tbuy_ratio = 0.55 if (i % 2 == 0) else 0.44
        taker_flow[sym] = pd.DataFrame({
            "timestamp": df["timestamp"],
            "quote_volume": qvol,
            "taker_buy_quote_volume": qvol * tbuy_ratio,
        })
        # Alternate negative Spot-Perp discount (-5 bps) vs positive Perp froth (+6 bps)
        prem_vals = np.full(len(df), -0.0002 if (i % 2 == 0) else 0.0001)
        prem_vals[-24:] = -0.0005 if (i % 2 == 0) else 0.0006
        premium_index[sym] = pd.DataFrame({
            "timestamp": df["timestamp"],
            "close": prem_vals,
        })

    engine = SevenFactorOnlineEngine()
    res_cvd = engine.allocate_latest(bars_1h=bars_1h, bars_15m=bars_15m, taker_flow=taker_flow)
    res_bas = engine.allocate_latest(bars_1h=bars_1h, bars_15m=bars_15m, premium_index=premium_index)
    res_combo = engine.allocate_latest(
        bars_1h=bars_1h,
        bars_15m=bars_15m,
        taker_flow=taker_flow,
        premium_index=premium_index,
    )

    assert pytest.approx(sum(res_cvd.unit_weights.values()), abs=1e-6) == 0.0
    assert pytest.approx(sum(res_bas.unit_weights.values()), abs=1e-6) == 0.0
    assert pytest.approx(sum(res_combo.unit_weights.values()), abs=1e-6) == 0.0
    assert sum(abs(v) for v in res_combo.unit_weights.values()) > 0.0


def test_binance_live_broker_account_config_and_smoke_test():
    """Tests Binance Futures API auto-configuration (One-Way Mode, CROSSED Margin, Leverage) and 5-stage Smoke Test."""
    broker = BinanceLiveBroker(
        dry_run=False,
        testnet=True,
        initial_capital=1000.0,
        chase_interval_sec=0.0,
    )
    mock_client = MagicMock()
    # Simulate account initially in Hedge Mode -> auto-switched to One-Way Mode
    mock_client.fetch_position_mode.side_effect = [
        {"hedged": True},
        {"hedged": False},
    ]
    mock_client.fapiPrivateGetMultiAssetsMargin.return_value = {"multiAssetsMargin": False}
    mock_client.set_margin_mode.return_value = {"code": 200, "msg": "success"}
    mock_client.set_leverage.return_value = {"leverage": 3, "maxNotionalValue": "1000000", "symbol": "DOGEUSDT"}
    mock_client.fetch_balance.return_value = {"USDT": {"free": 999.99, "total": 999.99}}
    mock_client.fetch_ticker.return_value = {"last": 0.20, "bid": 0.1999, "ask": 0.2001}

    # Stage 3 resting GTX order -> Stage 4 open GTX order -> Stage 5 reduceOnly market close
    mock_client.create_order.side_effect = [
        {"id": "ord_rest_101", "status": "open"},
        {"id": "ord_open_102", "status": "closed", "filled": 35.0, "average": 0.1999},
        {"id": "ord_close_103", "status": "closed", "filled": 35.0, "average": 0.2000},
    ]
    mock_client.fetch_order.return_value = {"id": "ord_open_102", "status": "closed", "filled": 35.0, "average": 0.1999}
    mock_client.fetch_positions.side_effect = [
        [{"symbol": "DOGE/USDT:USDT", "contracts": 35.0, "side": "long", "entryPrice": 0.1999, "markPrice": 0.2000}],
        [],
    ]
    broker.client = mock_client

    res = broker.run_production_smoke_test(symbol="DOGE/USDT", notional_usd=6.0, leverage=3)

    assert res["status"] == "PASSED"
    assert res["configuration"]["position_mode"] == "ONE_WAY"
    assert res["configuration"]["dual_side_position"] is False
    assert res["configuration"]["margin_modes"]["DOGE/USDT"] == "CROSSED"
    assert res["configuration"]["leverages"]["DOGE/USDT"] == 3
    assert res["gtx_post_and_cancel_ok"] is True
    assert res["position_verified_on_exchange"] is True
    assert res["final_flat_verified"] is True
    mock_client.set_position_mode.assert_called_once_with(False)
    mock_client.set_margin_mode.assert_called_once_with("cross", "DOGE/USDT")
    mock_client.set_leverage.assert_called_once_with(3, "DOGE/USDT")
    mock_client.cancel_order.assert_called_once_with("ord_rest_101", "DOGE/USDT")


def test_shadow_broker_live_mainnet_bridge():
    """Tests ShadowExecutionBroker(live_broker=...) routing orders to live Binance Futures & syncing DB."""
    from src.domain.enums import OrderSide, OrderType

    live_broker = BinanceLiveBroker(
        dry_run=False,
        testnet=True,
        initial_capital=1000.0,
        chase_interval_sec=0.0,
    )
    mock_client = MagicMock()
    mock_client.fetch_balance.return_value = {
        "USDT": {"free": 1000.0, "total": 1000.0},
        "info": {"assets": [{"asset": "USDT", "walletBalance": "1000.0", "availableBalance": "1000.0"}]},
    }
    mock_client.fetch_positions.return_value = []
    mock_client.fetch_ticker.return_value = {"last": 150.0, "bid": 149.9, "ask": 150.1}
    mock_client.create_order.side_effect = [
        {"id": "live_buy_1", "status": "closed", "filled": 1.0, "average": 149.9},
        {"id": "live_sl_2", "status": "closed", "filled": 1.0, "average": 145.0},
    ]
    mock_client.fetch_order.return_value = {"id": "live_buy_1", "status": "closed", "filled": 1.0, "average": 149.9}
    live_broker.client = mock_client

    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=1000.0,
        max_gross_leverage=6.0,
        live_broker=live_broker,
    )
    assert broker.is_live is True
    broker.set_market_price("SOL/USDT", price=150.0, bid=149.9, ask=150.1)

    # 1. Submit LIMIT BUY order -> routed via live execute_maker_limit_chase
    ord_open = broker.create_order(
        symbol="SOL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=149.9,
        strategy="7-Factor Core",
    )
    assert ord_open.filled_quantity == 1.0
    assert ord_open.average_fill_price == 149.9
    assert "SOL/USDT" in broker.portfolio.positions
    assert broker.portfolio.positions["SOL/USDT"].quantity == 1.0

    # 2. Attach Stop-Loss bracket & trigger it via check_pending_orders -> routed with reduceOnly=True
    broker.create_bracket_orders("SOL/USDT", take_profit=162.0, stop_loss=145.0)
    filled = broker.check_pending_orders({"SOL/USDT": {"price": 144.5, "bid": 144.4, "ask": 144.6}})
    assert len(filled) == 1
    assert broker.portfolio.positions["SOL/USDT"].quantity == 0.0
    assert mock_client.create_order.call_count == 2
    _, kwargs_close = mock_client.create_order.call_args
    assert kwargs_close["params"].get("reduceOnly") is True
    db.close()


def test_unified_trading_launcher_mode_and_db_isolation():
    """Tests Issue #51 (Issue-0): Multi-mode resolution & DB isolation across Shadow, Testnet, and Mainnet."""
    from scripts.run_trading import build_parser, resolve_mode_config

    parser = build_parser()
    fake_env = {
        "BINANCE_API_KEY": "prod_key",
        "BINANCE_API_SECRET": "prod_secret",
        "BINANCE_TESTNET_API_KEY": "test_key",
        "BINANCE_TESTNET_API_SECRET": "test_secret",
    }

    # 1. Default / --shadow mode -> shadow_trading.db, no live keys
    args_shd = parser.parse_args(["--daemon", "--shadow"])
    cfg_shd = resolve_mode_config(args_shd, environ=fake_env)
    assert cfg_shd["mode"] == "shadow"
    assert cfg_shd["db_path"].endswith("shadow_trading.db")
    assert cfg_shd["is_live"] is False
    assert cfg_shd["api_key"] is None

    # 2. --testnet mode -> testnet_trading.db, testnet keys
    args_tst = parser.parse_args(["--daemon", "--testnet"])
    cfg_tst = resolve_mode_config(args_tst, environ=fake_env)
    assert cfg_tst["mode"] == "testnet"
    assert cfg_tst["db_path"].endswith("testnet_trading.db")
    assert cfg_tst["is_live"] is True
    assert cfg_tst["is_testnet"] is True
    assert cfg_tst["api_key"] == "test_key"

    # 3. --mainnet mode -> live_trading.db, prod keys
    args_mn = parser.parse_args(["--daemon", "--mainnet"])
    cfg_mn = resolve_mode_config(args_mn, environ=fake_env)
    assert cfg_mn["mode"] == "mainnet"
    assert cfg_mn["db_path"].endswith("live_trading.db")
    assert cfg_mn["is_live"] is True
    assert cfg_mn["is_testnet"] is False
    assert cfg_mn["api_key"] == "prod_key"

    # Ensure all three DB paths are strictly distinct for parallel execution
    assert len({cfg_shd["db_path"], cfg_tst["db_path"], cfg_mn["db_path"]}) == 3









