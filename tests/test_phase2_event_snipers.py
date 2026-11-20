"""
Unit & Integration Tests for Phase 2 Event-Driven Microstructure Sniper Engine (`M09`, `M23`, `M04`):
1. `Sniper:M09_CoilBreakout` (Dispersion-Compression Coil + 48h Donchian + Volume + Taker-CVD)
2. `Sniper:M23_BreadthThrust` (82-Symbol Simultaneous 24h Breakout Breadth >= 65% + 2x Agg Vol -> Majors)
3. `Sniper:M04_NegFundSqueeze` (15m Trapped-Short Negative Funding <= -4bps + 24h High Breakout)
4. Full `ShadowTradingDaemon` Lifecycle: Cushion-Safe Sizing, One-Way Non-Collision during 8h Core Rebalance,
   Profit-Lock & Trailing Stop Ratchet, Shield Invariants Isolation, Kill-Switch Sync, and Runtime Hot-Disable.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.domain.enums import PositionSide
from src.execution.database import ShadowDatabase
from src.execution.feature_flags import FeatureFlags
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.event_snipers import (
    SNIPER_M04_STRAT,
    SNIPER_M09_STRAT,
    SNIPER_M23_STRAT,
    EventDrivenSniperEngine,
)
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine


def _build_1h_universe(n_hours: int = 240, n_syms: int = 20, seed: int = 99) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")
    syms = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "SUIUSDT",
        "NEARUSDT",
        "APTUSDT",
        "ARBUSDT",
        "RENDERUSDT",
        "FETUSDT",
        "INJUSDT",
        "TIAUSDT",
        "OPUSDT",
        "PENDLEUSDT",
        "TAOUSDT",
    ][:n_syms]
    out: dict[str, pd.DataFrame] = {}
    for j, s in enumerate(syms):
        # First 200h: normal dispersion (~1.5% hourly noise); last 39h before final bar: ultra-tight coil (~0.05% noise)!
        rets = np.concatenate(
            [
                rng.normal(0.0, 0.012, size=n_hours - 40),
                rng.normal(0.0, 0.0004, size=39),
                [0.0],
            ]
        )
        close = (100.0 + j * 10.0) * np.exp(np.cumsum(rets))
        open_p = np.concatenate([[close[0]], close[:-1]])
        high_p = np.maximum(open_p, close) * 1.001
        low_p = np.minimum(open_p, close) * 0.999
        qvol = np.full(n_hours, (30 - j) * 2_000_000.0)
        tb_qvol = qvol * 0.50
        out[s] = pd.DataFrame(
            {
                "timestamp": idx,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close,
                "quote_volume": qvol,
                "taker_buy_quote_volume": tb_qvol,
            }
        )
    return out


def test_m09_dispersion_coil_breakout_fires_only_when_coiled() -> None:
    eng = EventDrivenSniperEngine()
    u_map = _build_1h_universe(n_hours=240, n_syms=20, seed=101)

    # Inject a 48h Donchian upside breakout + 3.0x volume + 65% taker buy on INJUSDT on final bar
    inj = u_map["INJUSDT"].copy()
    prev_48_high = float(inj["high"].iloc[-49:-1].max())
    inj.iloc[-1, inj.columns.get_loc("close")] = prev_48_high * 1.025
    inj.iloc[-1, inj.columns.get_loc("high")] = prev_48_high * 1.028
    inj.iloc[-1, inj.columns.get_loc("quote_volume")] *= 3.0
    inj.iloc[-1, inj.columns.get_loc("taker_buy_quote_volume")] = (
        inj.iloc[-1, inj.columns.get_loc("quote_volume")] * 0.65
    )
    u_map["INJUSDT"] = inj

    sigs = eng.evaluate_1h_snipers(u_map)
    m09_sigs = [s for s in sigs if s.strategy == SNIPER_M09_STRAT and s.symbol == "INJUSDT"]
    assert len(m09_sigs) == 1
    assert m09_sigs[0].side == 1
    assert abs(m09_sigs[0].bracket.stop_loss_pct - 0.020) < 1e-6

    # Now expand prior-hour dispersion wildly across the market -> M09 MUST NOT fire!
    for idx_k, sym_k in enumerate(list(u_map.keys())[:12]):
        df_k = u_map[sym_k].copy()
        df_k.iloc[-2, df_k.columns.get_loc("close")] *= 1.25 if (idx_k % 2 == 0) else 0.80
        u_map[sym_k] = df_k
    sigs_expanded = eng.evaluate_1h_snipers(u_map)
    assert not any(s.strategy == SNIPER_M09_STRAT for s in sigs_expanded)


def test_m23_breadth_thrust_fires_on_unblocked_majors() -> None:
    eng = EventDrivenSniperEngine()
    u_map = _build_1h_universe(n_hours=220, n_syms=20, seed=202)

    # Make 16 out of 20 coins (80% >= 65%) break above their 24h high simultaneously on 2.8x volume!
    for sym_k in list(u_map.keys())[:16]:
        df_k = u_map[sym_k].copy()
        h24 = float(df_k["high"].iloc[-25:-1].max())
        df_k.iloc[-1, df_k.columns.get_loc("close")] = h24 * 1.03
        df_k.iloc[-1, df_k.columns.get_loc("high")] = h24 * 1.032
        df_k.iloc[-1, df_k.columns.get_loc("quote_volume")] *= 2.8
        u_map[sym_k] = df_k

    # Block BTCUSDT & ETHUSDT (simulating open Core legs) -> M23 routes cleanly to SOLUSDT / BNBUSDT / XRPUSDT!
    sigs = eng.evaluate_1h_snipers(u_map, blocked_symbols={"BTCUSDT", "ETHUSDT"})
    m23_syms = [s.symbol for s in sigs if s.strategy == SNIPER_M23_STRAT]
    assert "BTCUSDT" not in m23_syms
    assert "ETHUSDT" not in m23_syms
    assert "SOLUSDT" in m23_syms


def test_m04_neg_funding_squeeze_and_full_daemon_lifecycle() -> None:
    idx_15m = pd.date_range("2026-08-01 00:00:00", periods=110, freq="15min", tz="UTC")
    close = np.full(110, 20.0) + np.sin(np.arange(110) * 0.2) * 0.15
    high = close + 0.05
    low = close - 0.05
    qvol = np.full(110, 500_000.0)
    tb = qvol * 0.50

    # Final 15m bar on INJUSDT breaks 24h high on 3.0x volume + 62% taker buy!
    close[-1] = 20.65
    high[-1] = 20.70
    qvol[-1] = 1_500_000.0
    tb[-1] = 1_500_000.0 * 0.62
    df_inj_15m = pd.DataFrame(
        {
            "timestamp": idx_15m,
            "open": close - 0.02,
            "high": high,
            "low": low,
            "close": close,
            "quote_volume": qvol,
            "taker_buy_quote_volume": tb,
        }
    )

    # Build Daemon at $2,630 equity (halfway in [$2,510, $2,750] Cushion-Safe zone -> cushion_scalar = 0.50!)
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=2630.0)
    lev_ctrl = SteppedLeverageController(
        initial_capital=2630.0,
        start_leverage=2.5,
        max_leverage=5.0,
        cushion_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
    )
    alloc = MetaStrategyAllocator(
        seven_factor_engine=SevenFactorOnlineEngine(),
        stepped_leverage_controller=lev_ctrl,
        use_seven_factor_engine=True,
    )
    ff = FeatureFlags.from_env_and_args(mode="mainnet", environ={})
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=alloc,
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "INJ/USDT"],
        equity_floor_usd=2500.0,
        enable_hard_equity_floor=True,
        enable_warm_start=False,
        feature_flags=ff,
    )
    daemon.latest_tickers["INJ/USDT"] = {"price": 20.65, "bid": 20.64, "ask": 20.66, "volume_24h": 80_000_000.0}

    # Execute Phase 2 Sniper scan with INJUSDT at -6.5 bps funding (`-0.00065 <= -0.0004`)
    now_t = datetime(2026, 8, 2, 3, 15, 10, tzinfo=timezone.utc)
    orders = daemon.run_event_sniper_scan(
        now=now_t,
        bars_15m_override={"INJUSDT": df_inj_15m},
        funding_override={"INJUSDT": -0.00065},
    )
    assert len(orders) == 1
    pos_inj = broker.get_portfolio().positions["INJ/USDT"]
    assert pos_inj.side == PositionSide.LONG
    assert pos_inj.strategy == SNIPER_M04_STRAT

    # Verify Cushion-Safe 0.50x scaling: 0.24 * $2,630 * 0.50 = ~$315.60 notional!
    notional_opened = pos_inj.quantity * pos_inj.entry_price
    assert 300.0 <= notional_opened <= 330.0
    assert pos_inj.stop_loss is not None and pos_inj.take_profit is not None

    # Verify Shield Invariants remain ALL_OK (Sniper isolated from Core neutrality AND from legacy anomalies!)
    inv = daemon.evaluate_shield_invariants()
    assert inv["all_invariants_ok"] is True
    assert inv["inv5_anomaly_capsule_isolated"] is True
    assert len(inv["open_snipers"]) == 1

    # Verify Profit-Lock (+2.8% -> +1.2% above entry) and Trailing Stop (+4.5% -> 2.2% behind peak) ratchet!
    daemon.manage_trailing_and_profit_locks({"INJ/USDT": {"price": 20.65 * 1.032}})
    pos_after_lock = broker.get_portfolio().positions["INJ/USDT"]
    assert pos_after_lock.stop_loss > 20.65 * 1.010

    daemon.manage_trailing_and_profit_locks({"INJ/USDT": {"price": 20.65 * 1.060}})
    pos_after_trail = broker.get_portfolio().positions["INJ/USDT"]
    assert pos_after_trail.stop_loss > 20.65 * 1.035

    # Verify EmergencyKillSwitch authorizes the active Sniper exposure when synced
    ks = EmergencyKillSwitch(equity_floor_usd=2500.0)
    broker.live_broker = type("DummyLive", (), {"kill_switch": ks})()
    daemon._sync_kill_switch_directional_target()
    ks_status = ks.evaluate_and_enforce(broker.get_portfolio())
    assert ks_status.triggered is False

    # Verify runtime hot-disable of `enable_event_snipers=False` cleanly auto-closes open Sniper slot!
    db.set_feature_flag("enable_event_snipers", False)
    daemon.sync_feature_flags_from_db()
    assert daemon.enable_event_snipers is False
    daemon.manage_trailing_and_profit_locks({"INJ/USDT": {"price": 20.65 * 1.05}})
    assert broker.get_portfolio().positions["INJ/USDT"].side == PositionSide.FLAT


def test_usdt_only_default_and_container_restart_p2_hydration(tmp_path):
    """
    Verifies:
    1) `ShadowExecutionBroker` defaults to 100% USDT (`enable_yield_collateral=False`, ZERO BFUSD).
    2) Mid-cycle container restart hydrates `allocator.last_p2_weights`, `allocator.last_directional_net_usd`,
       and `EmergencyKillSwitch.target_net_exposure_usd` from SQLite without triggering spurious asymmetry
       warnings or losing Trend-Aware Chop Bracket Protection (`5.0% TP / 3.5%->1.5%`).
    3) Fast-cycle `save_seven_factor_telemetry` updates preserve persisted `p2_weights` and `directional_net_usd`.
    """
    from src.domain.entities import OrderSide, OrderType
    from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine

    db_file = str(tmp_path / "restart_hydration.db")
    db = ShadowDatabase(db_path=db_file)
    broker = ShadowExecutionBroker(database=db, initial_cash=2700.0)
    # 1. 100% USDT by default — no simulated BFUSD collateral yield
    assert broker.enable_yield_collateral is False
    assert broker.settle_collateral_yield() == 0.0

    # Open Core + P2 directional short portfolio:
    # Core P1: +$900 ETH/USDT, -$900 BTC/USDT; P2 directional sleeve: -$420 BTC/USDT -> Net -$420.00 (-15.56% of $2,700!)
    broker.set_market_price("ETH/USDT", 3000.0, 2999.5, 3000.5)
    broker.set_market_price("BTC/USDT", 60000.0, 59995.0, 60005.0)
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.30,
        strategy="7-Factor Core",
    )
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.022,  # -$1,320 notional -> net portfolio = +$900 - $1,320 = -$420.00 (-15.56% > 15% limit!)
        strategy="7-Factor Core + DollarBar P2",
    )

    reb_ts = datetime(2026, 11, 18, 8, 0, 20, tzinfo=timezone.utc)
    db.save_rebalance_snapshot(
        regime_detected="CHOP",
        momentum_weight=0.35,
        stat_arb_weight=0.45,
        carry_weight=0.20,
        target_allocations={"ETHUSDT": 900.0, "BTCUSDT": -1320.0},
        executed_delta={"ETH/USDT": 900.0, "BTC/USDT": -1320.0},
        timestamp=reb_ts,
    )
    db.save_seven_factor_telemetry(
        initial_capital=2700.0,
        current_equity=2700.0,
        high_water_mark=2750.0,
        drawdown_pct=0.018,
        unlocked_tier=2.5,
        active_base_leverage=2.5,
        vol_scalar=1.0,
        effective_leverage=2.5,
        next_tier_leverage=3.0,
        next_tier_hwm_target=3162.5,
        progress_to_next_tier_pct=0.10,
        portfolio_beta=0.0,
        conviction=1.0,
        factor_weights={"macro": 0.0, "med": 0.12, "fast": 0.0, "rev": 0.58, "carry": 0.30, "liq": 0.0, "size": 0.0},
        factor_tstats={"macro": -0.8, "med": -0.3, "fast": -0.5, "rev": 1.8, "carry": 1.2, "liq": 0.2, "size": -0.4},
        p2_weights={"BTCUSDT": -0.22, "ETHUSDT": 0.064},
        directional_net_usd=-420.0,
        engine_mode="dynamic_online_ic",
        timestamp=reb_ts,
    )
    # Simulate a fast-cycle telemetry update (passing None for p2_weights / directional_net_usd)
    db.save_seven_factor_telemetry(
        initial_capital=2700.0,
        current_equity=2702.0,
        high_water_mark=2750.0,
        drawdown_pct=0.017,
        unlocked_tier=2.5,
        active_base_leverage=2.5,
        vol_scalar=1.0,
        effective_leverage=2.5,
        next_tier_leverage=3.0,
        next_tier_hwm_target=3162.5,
        progress_to_next_tier_pct=0.11,
        portfolio_beta=0.0,
        conviction=1.0,
        factor_weights={},
        factor_tstats={},
    )
    db.close()

    # 2. Simulate Mid-Cycle Container Restart (e.g. 12:52 UTC — between 08:00 and 16:00 UTC)
    db2 = ShadowDatabase(db_path=db_file)
    ks2 = EmergencyKillSwitch(equity_floor_usd=2500.0)
    dummy_live = type("DummyLive", (), {"kill_switch": ks2, "dry_run": True})()
    broker2 = ShadowExecutionBroker(database=db2, initial_cash=2700.0, live_broker=dummy_live)
    broker2.set_market_price("ETH/USDT", 3000.0, 2999.5, 3000.5)
    broker2.set_market_price("BTC/USDT", 60000.0, 59995.0, 60005.0)

    ff2 = FeatureFlags.from_env_and_args(mode="mainnet", environ={})
    lev2 = SteppedLeverageController(
        initial_capital=2700.0,
        start_leverage=2.5,
        max_leverage=5.0,
        cushion_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
    )
    alloc2 = MetaStrategyAllocator(
        seven_factor_engine=SevenFactorOnlineEngine(),
        stepped_leverage_controller=lev2,
        use_seven_factor_engine=True,
        dollar_bar_engine=CausalDollarBarTSMOMEngine(max_gross_leverage=0.95),
        enable_dollar_bar_tsmom=True,
        core_p1_risk_scale=0.85,
    )
    daemon2 = ShadowTradingDaemon(
        broker=broker2,
        allocator=alloc2,
        symbols=["BTC/USDT", "ETH/USDT"],
        core_take_profit_pct=0.050,
        profit_lock_trigger_pct=0.035,
        profit_lock_level_pct=0.015,
        equity_floor_usd=2500.0,
        enable_hard_equity_floor=True,
        enable_warm_start=False,
        feature_flags=ff2,
    )

    # Verify hydration BEFORE any rebalance runs!
    assert pytest.approx(daemon2.allocator.last_p2_weights.get("BTCUSDT", 0.0), abs=1e-6) == -0.22
    assert pytest.approx(daemon2.allocator.last_directional_net_usd, abs=1e-4) == -420.0
    assert pytest.approx(ks2.target_net_exposure_usd, abs=2.0) == -420.0
    assert daemon2.last_rebalance_time == reb_ts

    # Verify EmergencyKillSwitch & Shield Invariants emit 0 asymmetry errors (-15.56% raw net, 0.0% unintended drift!)
    halted = daemon2.enforce_hard_equity_floor_and_kill_switch(broker2.get_portfolio())
    assert halted is False
    inv2 = daemon2.evaluate_shield_invariants()
    assert inv2["all_invariants_ok"] is True
    assert inv2["inv4_dollar_neutrality"] is True

    # Verify Trend-Aware Chop Bracket Protection survives restart (`BTC/USDT` short aligned with `p2_w = -0.22`)
    daemon2.allocator.last_seven_factor_result = type("DummySF", (), {"chop_overlay_weight": 0.65})()
    tp_btc, lk_tr_btc, lk_lv_btc = daemon2.compute_dynamic_barrier_profile("BTC/USDT")
    tp_eth, lk_tr_eth, lk_lv_eth = daemon2.compute_dynamic_barrier_profile("ETH/USDT")
    assert (tp_btc, lk_tr_btc, lk_lv_btc) == (0.050, 0.035, 0.015)
    assert tp_eth <= 0.036
    assert (lk_tr_eth, lk_lv_eth) == (0.024, 0.012)
    db2.close()


def test_backtest_cli_supports_phase2_snipers():
    """Verifies `scripts/run_backtest.py` natively supports `--enable-dollar-bar-tsmom` + `--enable-event-snipers`."""
    from scripts.run_backtest import generate_synthetic_universe, run_simulation

    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "INJUSDT"]
    bars, funding = generate_synthetic_universe(symbols=syms, n_bars=600)
    res = run_simulation(
        strategy_name="seven_factor",
        bars_dict=bars,
        funding_dict=funding,
        initial_capital=2700.0,
        enable_dollar_bar_tsmom=True,
        enable_event_snipers=True,
        cushion_floor_usd=2500.0,
    )
    assert len(res.equity_curve) > 100
    assert res.equity_curve.iloc[-1] > 0.0


def test_progressive_profit_lock_ws_peak_and_ranging_barriers():
    """
    Verifies the 4 live forensic production fixes:
    1) Real-time WebSocket `set_market_price` updates `pos.peak_price` on sub-second wicks & triggers `_ws_barrier_event`.
    2) Low-conviction `RANGING` (`conviction=0.38 < 0.50`, `max_tr=0.57 < 1.0`) activates Chop barriers (`TP <= 3.6%`)
       on pure Core legs while keeping `5.0%` Trend TP on `|p2_w| >= 0.15` aligned legs.
    3) 3-Stage Progressive Profit-Lock Ratchet (`+2.2%->+0.9%`, `+3.3%->+1.8%`, `+4.3%->+3.2%` in Trend;
       `+1.8%->+0.8%`, `+2.4%->+1.2%`, `~(TP-0.4%)->~(TP-1.2%)` in Chop) protects open winners from reversing.
    4) Mid-cycle `TAKE_PROFIT` / `Profit-Lock` exits dynamically update authorized net USD (`0` false Kill-Switch asymmetry).
    """
    from src.domain.entities import OrderSide, OrderType
    from src.execution.ws_manager import PriceTick
    from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine

    db = ShadowDatabase(db_path=":memory:")
    ks = EmergencyKillSwitch(equity_floor_usd=2500.0)
    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=2700.0,
        live_broker=type("DummyLive", (), {"kill_switch": ks, "dry_run": True})(),
    )
    ff = FeatureFlags.from_env_and_args(mode="mainnet", environ={})
    assert ff.enable_progressive_profit_lock is True

    sf_eng = SevenFactorOnlineEngine(priors=None)
    assert sf_eng._zero_priors is True
    alloc = MetaStrategyAllocator(
        seven_factor_engine=sf_eng,
        stepped_leverage_controller=SteppedLeverageController(initial_capital=2700.0),
        use_seven_factor_engine=True,
        dollar_bar_engine=CausalDollarBarTSMOMEngine(max_gross_leverage=0.95),
        enable_dollar_bar_tsmom=True,
        core_p1_risk_scale=0.85,
    )
    alloc.last_p2_weights = {"BTCUSDT": -0.20}
    alloc.last_directional_net_usd = -540.0
    # Simulate Tokyo live telemetry: weak RANGING (`conviction = 0.38`, `med = +0.57`, `chop_overlay_weight = 0.0`)
    alloc.last_seven_factor_result = type(
        "LiveRangingSF",
        (),
        {
            "factor_tstats": {"macro": -0.04, "med": 0.57, "fast": 0.04, "rev": -0.37, "carry": -1.05},
            "chop_overlay_weight": 0.0,
            "conviction": 0.38,
            "asset_vol_24h": {"ARBUSDT": 0.010, "BTCUSDT": 0.008},
        },
    )()

    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=alloc,
        symbols=["BTC/USDT", "ARB/USDT", "SOL/USDT"],
        core_take_profit_pct=0.050,
        profit_lock_trigger_pct=0.035,
        profit_lock_level_pct=0.015,
        equity_floor_usd=2500.0,
        enable_hard_equity_floor=True,
        enable_warm_start=False,
        feature_flags=ff,
    )

    # Open balanced Core + P2 portfolio: SOL LONG +$500, ARB SHORT -$500, BTC SHORT -$540 (P2)
    broker.set_market_price("SOL/USDT", 100.0)
    broker.set_market_price("ARB/USDT", 1.0000)
    broker.set_market_price("BTC/USDT", 60000.0)
    broker.create_order("SOL/USDT", OrderSide.BUY, OrderType.MARKET, 5.0, strategy="F2_Med_Idio_PB")
    broker.create_order("ARB/USDT", OrderSide.SELL, OrderType.MARKET, 500.0, strategy="F2_Med_Idio_PB")
    broker.create_order("BTC/USDT", OrderSide.SELL, OrderType.MARKET, 0.009, strategy="P2:DollarBar_TSMOM")

    # 1. Verify Low-Conviction Ranging compresses ARB to Chop TP (~3.5%), while P2-aligned BTC keeps 5.0% Trend TP!
    tp_arb, _, _ = daemon.compute_dynamic_barrier_profile("ARB/USDT")
    tp_btc, _, _ = daemon.compute_dynamic_barrier_profile("BTC/USDT")
    assert 0.028 <= tp_arb <= 0.036
    assert pytest.approx(tp_btc, abs=1e-6) == 0.050
    daemon.ensure_position_brackets()

    # 2. Verify WebSocket sub-second wick updates `peak_price` immediately AND sets `_ws_barrier_event`!
    daemon._ws_barrier_event.clear()
    daemon._on_ws_tick(
        PriceTick(
            symbol="BTCUSDT",
            price=58500.0,
            bid=58499.0,
            ask=58501.0,
            timestamp=datetime.now(timezone.utc),
        )
    )
    pos_btc = broker.portfolio.positions["BTC/USDT"]
    assert pos_btc.peak_price == pytest.approx(58500.0, rel=1e-4)  # +2.5% short gain captured on WS tick!
    assert daemon._ws_barrier_event.is_set() is True

    # 3. Verify 3-Stage Progressive Profit-Lock on BTC SHORT (Trend profile):
    ep_btc = pos_btc.entry_price
    # Stage 1 (+2.5% >= +2.2%): locks +0.9%
    daemon.manage_trailing_and_profit_locks({"BTC/USDT": {"price": 58500.0}})
    assert pos_btc.stop_loss == pytest.approx(ep_btc * 0.991, abs=0.5)
    # Stage 2 (+3.5% >= +3.3%): ratchets to +1.8%
    daemon.manage_trailing_and_profit_locks({"BTC/USDT": {"price": 57900.0}})
    assert pos_btc.stop_loss == pytest.approx(ep_btc * 0.982, abs=0.5)
    # Stage 3 (+4.6% >= +4.3%): ratchets to +3.2%
    daemon.manage_trailing_and_profit_locks({"BTC/USDT": {"price": 57240.0}})
    assert pos_btc.stop_loss == pytest.approx(ep_btc * 0.968, abs=0.5)

    # 4. Simulate BOTH BTC (-$540 P2) and ARB (-$500 Core P1) hitting Take-Profit mid-cycle -> 0 Kill-Switch false alarm!
    filled = broker.check_pending_orders(
        {
            "BTC/USDT": {"price": 56900.0, "bid": 56899.0, "ask": 56901.0},
            "ARB/USDT": {"price": 0.9600, "bid": 0.9599, "ask": 0.9601},
        }
    )
    assert len(filled) == 2
    daemon._refresh_anomaly_cooldowns(filled)
    assert daemon.enforce_hard_equity_floor_and_kill_switch(broker.get_portfolio()) is False
    inv = daemon.evaluate_shield_invariants()
    assert inv["inv4_dollar_neutrality"] is True

    # 5. Verify Mainnet defaults to Strict 8h Clock (intracycle OFF), and still supports opt-in intracycle check
    from datetime import timedelta

    assert daemon.enable_intracycle_rebalance is False
    now_fill = daemon.last_core_bracket_fill_time
    assert now_fill is not None
    daemon.last_rebalance_time = now_fill - timedelta(hours=1)
    assert daemon.is_intracycle_rebalance_needed(now_fill + timedelta(minutes=16)) is False

    daemon.enable_intracycle_rebalance = True
    assert daemon.is_intracycle_rebalance_needed(now_fill + timedelta(minutes=10)) is False
    assert daemon.is_intracycle_rebalance_needed(now_fill + timedelta(minutes=16)) is True
    assert daemon.evaluate_shield_invariants()["inv3_no_spurious_intracycle"] is True
    db.close()
