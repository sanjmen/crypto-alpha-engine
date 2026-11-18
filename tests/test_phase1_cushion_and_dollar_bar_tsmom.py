"""
Unit & Integration Tests for Phase 1 Production Upgrade:
1. Cushion-Safe CPPI Equity Floor Governor (`SteppedLeverageController`)
2. Causal 1-Minute Dollar-Bar Multi-Horizon TSMOM + Taker-CVD (`CausalDollarBarTSMOMEngine`)
3. Sub-Ledger Netting, Zombie-Stub Pruning, Shield Invariants (#2 & #4), and Kill-Switch Directional Target Sync
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.domain.entities import Portfolio
from src.domain.enums import OrderSide, OrderType
from src.execution.database import ShadowDatabase
from src.execution.feature_flags import FeatureFlags
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine, build_causal_dollar_bars
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine


def _make_synthetic_1m_bars(
    start: str = "2026-01-01 00:00:00",
    n_minutes: int = 36 * 1440,
    drift_per_min: float = 0.000035,
    taker_buy_share: float = 0.62,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_minutes, freq="1min", tz="UTC")
    rets = drift_per_min + rng.normal(0.0, 0.00045, size=n_minutes)
    close = 50_000.0 * np.exp(np.cumsum(rets))
    open_px = np.concatenate([[close[0]], close[:-1]])
    high_px = np.maximum(open_px, close) * 1.0003
    low_px = np.minimum(open_px, close) * 0.9997
    qvol = rng.uniform(800_000.0, 1_600_000.0, size=n_minutes)
    tb_qvol = qvol * np.clip(taker_buy_share + rng.normal(0.0, 0.03, size=n_minutes), 0.05, 0.95)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close,
            "quote_volume": qvol,
            "taker_buy_quote_volume": tb_qvol,
        }
    )


def test_cushion_safe_cppi_governor_scales_smoothly_above_floor() -> None:
    ctrl = SteppedLeverageController(
        initial_capital=2700.0,
        start_leverage=2.5,
        max_leverage=5.0,
        recovery_boost_equity=3050.0,
        recovery_boost_leverage=3.2,
        cushion_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
        min_cushion_scalar=0.15,
    )

    # Above $2,750 ($2,510 + $240): full 1.0 cushion scalar -> 2.5x effective leverage
    s_high = ctrl.update_state(current_equity=2800.0, vol_scalar=1.0)
    assert abs(s_high.cushion_scalar - 1.0) < 1e-9
    assert abs(s_high.effective_leverage - 2.5) < 1e-9

    # Mid-buffer ($2,630 -> halfway between $2,510 and $2,750): 0.50 cushion scalar -> 1.25x leverage
    s_mid = ctrl.update_state(current_equity=2630.0, vol_scalar=1.0)
    assert abs(s_mid.cushion_scalar - 0.50) < 1e-6
    assert abs(s_mid.effective_leverage - 1.25) < 1e-6

    # Right above $2,500 hard floor ($2,508 = 10.43% DD from $2,800 HWM -> base de-steps 2.5x->2.0x AND cushion=0.15 -> 0.30x!)
    s_near_floor = ctrl.update_state(current_equity=2508.0, vol_scalar=1.0)
    assert abs(s_near_floor.active_base_leverage - 2.0) < 1e-6
    assert abs(s_near_floor.cushion_scalar - 0.15) < 1e-6
    assert abs(s_near_floor.effective_leverage - 2.0 * 0.15) < 1e-6

    # Once HWM crosses $3,050 recovery boost, base leverage steps up to 3.2x with cushion_scalar=1.0
    s_boost = ctrl.update_state(current_equity=3100.0, vol_scalar=1.0)
    assert abs(s_boost.cushion_scalar - 1.0) < 1e-9
    assert abs(s_boost.active_base_leverage - 3.2) < 1e-6


def test_causal_dollar_bars_strict_point_in_time_no_lookahead() -> None:
    df_full = _make_synthetic_1m_bars(n_minutes=28 * 1440, drift_per_min=0.00004, taker_buy_share=0.63, seed=7)
    # Inject a massive 50x volume shock in the last 3 days and verify earlier Dollar Bars & signals are 100% unchanged!
    cutoff_min = 24 * 1440
    df_trunc = df_full.iloc[:cutoff_min].copy()
    df_shocked = df_full.copy()
    df_shocked.loc[df_shocked.index[cutoff_min:], "quote_volume"] *= 50.0
    df_shocked.loc[df_shocked.index[cutoff_min:], "close"] *= 0.50

    db_trunc = build_causal_dollar_bars(df_trunc, target_bars_per_day=24)
    db_shock = build_causal_dollar_bars(df_shocked, target_bars_per_day=24)
    cutoff_ts = df_full["timestamp"].iloc[cutoff_min - 1] + pd.Timedelta(minutes=1)
    db_shock_before = db_shock.loc[db_shock.index <= cutoff_ts]

    pd.testing.assert_frame_equal(db_trunc, db_shock_before)

    eng = CausalDollarBarTSMOMEngine(symbols=("BTCUSDT",), max_gross_leverage=0.90)
    s_trunc = eng.compute_asset_signal_series(df_trunc)["position"]
    s_shock = eng.compute_asset_signal_series(df_shocked)["position"].loc[:cutoff_ts]
    pd.testing.assert_series_equal(s_trunc, s_shock)
    assert float(s_trunc.iloc[-1]) > 0.5  # Strong uptrend + 63% taker buy -> strong long signal!


def test_dollar_bar_tsmom_short_signal_and_deadzone() -> None:
    eng = CausalDollarBarTSMOMEngine(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), max_gross_leverage=0.90)
    df_bear = _make_synthetic_1m_bars(n_minutes=30 * 1440, drift_per_min=-0.000045, taker_buy_share=0.36, seed=11)
    df_flat = _make_synthetic_1m_bars(n_minutes=30 * 1440, drift_per_min=0.0, taker_buy_share=0.50, seed=19)
    # Force ultra-flat zero drift on df_flat
    df_flat["close"] = 3000.0 + np.sin(np.arange(len(df_flat)) * 0.05) * 0.5
    df_flat["taker_buy_quote_volume"] = 0.50 * df_flat["quote_volume"]

    res = eng.allocate_latest({"BTCUSDT": df_bear, "ETHUSDT": df_flat})
    assert res.unit_weights["BTCUSDT"] < -0.15
    assert res.unit_weights["ETHUSDT"] == 0.0
    assert res.net_unit_exposure < -0.15


def test_meta_allocator_p1_p2_netting_invariants_and_kill_switch() -> None:
    # Build 12 symbols x 360h of 1h data for P1 + 30d of 1m data for BTC/ETH/SOL P2
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
    ]
    idx_1h = pd.date_range("2026-01-15 00:00:00", periods=360, freq="1h", tz="UTC")
    rng = np.random.default_rng(123)
    market_1h = {}
    for i, s in enumerate(syms):
        drift = (i - 5.5) * 0.0003
        c = 100.0 * np.exp(np.cumsum(drift + rng.normal(0.0, 0.008, size=len(idx_1h))))
        market_1h[s] = pd.DataFrame(
            {
                "timestamp": idx_1h,
                "open": c * 0.999,
                "high": c * 1.003,
                "low": c * 0.997,
                "close": c,
                "volume": rng.uniform(1e4, 5e4, size=len(idx_1h)),
            }
        )

    bars_1m = {
        "BTCUSDT": _make_synthetic_1m_bars(n_minutes=32 * 1440, drift_per_min=0.000045, taker_buy_share=0.64, seed=1),
        "ETHUSDT": _make_synthetic_1m_bars(n_minutes=32 * 1440, drift_per_min=0.000040, taker_buy_share=0.62, seed=2),
        "SOLUSDT": _make_synthetic_1m_bars(n_minutes=32 * 1440, drift_per_min=0.000050, taker_buy_share=0.65, seed=3),
    }

    sf_eng = SevenFactorOnlineEngine()
    lev_ctrl = SteppedLeverageController(
        initial_capital=2700.0,
        start_leverage=2.5,
        max_leverage=5.0,
        cushion_floor_usd=2500.0,
        cushion_buffer_usd=240.0,
    )
    db_eng = CausalDollarBarTSMOMEngine(max_gross_leverage=0.95)
    alloc = MetaStrategyAllocator(
        seven_factor_engine=sf_eng,
        stepped_leverage_controller=lev_ctrl,
        use_seven_factor_engine=True,
        dollar_bar_engine=db_eng,
        enable_dollar_bar_tsmom=True,
        core_p1_risk_scale=0.88,
    )

    pf = Portfolio(cash=2700.0, initial_cash=2700.0)
    target_dollars, _, _ = alloc.generate_portfolio_allocations(
        market_data=market_1h,
        portfolio=pf,
        volatilities={s: 0.02 for s in syms},
        bars_1m=bars_1m,
    )

    # Verify intentional positive directional net exposure from P2 (`BTC/ETH/SOL` all strongly bullish)
    assert alloc.last_directional_net_usd > 400.0
    assert abs(sum(target_dollars.values()) - alloc.last_directional_net_usd) < 5.0
    # Verify 0 zombie stubs < $150!
    for sym, usd_val in target_dollars.items():
        assert abs(usd_val) >= 150.0, f"Found unexpected zombie stub {sym}: ${usd_val:.2f}"

    # Populate broker & daemon and verify Shield Invariants (#2 & #4) and EmergencyKillSwitch pass!
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=2700.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=alloc,
        symbols=[f"{s[:-4]}/USDT" for s in syms],
        core_take_profit_pct=0.050,
        equity_floor_usd=2500.0,
        enable_hard_equity_floor=True,
        enable_warm_start=False,
        feature_flags=FeatureFlags.from_env_and_args(mode="mainnet", environ={}),
    )
    for sym, usd_val in target_dollars.items():
        slash_sym = f"{sym[:-4]}/USDT"
        broker.set_market_price(slash_sym, 100.0)
        broker.create_order(
            symbol=slash_sym,
            side=OrderSide.BUY if usd_val > 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=abs(usd_val) / 100.0,
            strategy=alloc.last_dominant_strategy.get(sym, "7Factor:F1"),
        )

    inv = daemon.evaluate_shield_invariants()
    assert inv["inv2_no_zombie_stubs"] is True
    assert inv["inv4_dollar_neutrality"] is True
    assert inv["all_invariants_ok"] is True

    # Verify EmergencyKillSwitch allows intentional P2 directional target, while catching rogue unintended asymmetry!
    ks = EmergencyKillSwitch(
        max_24h_drawdown_pct=0.12,
        max_net_exposure_pct=0.15,
        equity_floor_usd=2500.0,
        target_net_exposure_usd=alloc.last_directional_net_usd,
    )
    status_ok = ks.evaluate_and_enforce(broker.get_portfolio())
    assert status_ok.triggered is False

    # Now simulate a rogue $1,500 unhedged leg on ADA/USDT without P2 authorization -> KillSwitch MUST trigger!
    broker.create_order(
        symbol="ADA/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=15.0,
        strategy="7Factor:F1",
    )
    status_rogue = ks.evaluate_and_enforce(broker.get_portfolio(), broker=MagicMock())
    assert status_rogue.triggered is True
    assert "Net Exposure Asymmetry" in status_rogue.reason

    # Verify Trend-Aware Bracket Protection:
    # - When P2 (+long) aligns with a net LONG leg (`|p2_w| >= 0.15`), preserve 5.0% Trend TP!
    # - On a pure P1 leg (`APT/USDT`, `p2_w == 0`), compress to dynamic Chop TP (`<= 3.6%`)!
    alloc.last_seven_factor_result.chop_overlay_weight = 0.65
    p2_long_sym = next(
        s
        for s in ("SOLUSDT", "ETHUSDT", "BTCUSDT")
        if target_dollars.get(s, 0.0) > 0 and alloc.last_p2_weights.get(s, 0.0) >= 0.15
    )
    tp_tr, ltrig_tr, llvl_tr = daemon.compute_dynamic_barrier_profile(f"{p2_long_sym[:-4]}/USDT")
    assert abs(tp_tr - 0.050) < 1e-6
    assert abs(ltrig_tr - 0.035) < 1e-6
    assert abs(llvl_tr - 0.015) < 1e-6

    tp_chop, ltrig_chop, llvl_chop = daemon.compute_dynamic_barrier_profile("APT/USDT")
    assert tp_chop <= 0.036
    assert abs(ltrig_chop - 0.024) < 1e-6
    assert abs(llvl_chop - 0.012) < 1e-6
