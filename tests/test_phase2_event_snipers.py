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
