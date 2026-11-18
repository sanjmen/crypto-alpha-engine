"""
Unit Tests for ShadowTradingDaemon.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Order
from src.domain.enums import OrderSide, OrderStatus, OrderType, PositionSide
from src.execution.database import ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import ShadowTradingDaemon
from src.strategies.meta_allocator import MetaStrategyAllocator


def generate_mock_klines(n_bars: int = 100, base_price: float = 90_000.0) -> pd.DataFrame:
    """Generates synthetic OHLCV data for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.01, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    opens = prices * (1 - 0.001)
    highs = prices * (1 + 0.005)
    lows = prices * (1 - 0.005)
    closes = prices
    volumes = np.random.uniform(50, 200, n_bars)

    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="1h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def mock_exchange():
    """Provides a mock ccxt exchange returning deterministic data."""
    exchange = MagicMock()
    exchange.fetch_tickers.return_value = {
        "BTC/USDT": {"last": 90_000.0, "bid": 89_990.0, "ask": 90_010.0, "high": 91_000.0, "low": 89_500.0},
        "ETH/USDT": {"last": 3_500.0, "bid": 3_499.0, "ask": 3_501.0, "high": 3_550.0, "low": 3_480.0},
    }
    mock_df = generate_mock_klines(60, 90_000.0)
    ohlcv_list = [
        [int(row.timestamp.timestamp() * 1000), row.open, row.high, row.low, row.close, row.volume]
        for row in mock_df.itertuples()
    ]
    exchange.fetch_ohlcv.return_value = ohlcv_list
    exchange.fetch_funding_rate.return_value = {"symbol": "BTC/USDT", "fundingRate": 0.0001}
    return exchange


@pytest.fixture
def daemon(mock_exchange):
    """Initializes daemon with in-memory SQLite broker and mock exchange."""
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=100_000.0)
    router = SmartExecutionRouter(broker=broker)

    d = ShadowTradingDaemon(
        broker=broker,
        router=router,
        exchange_client=mock_exchange,
        symbols=["BTC/USDT", "ETH/USDT"],
        fast_interval_sec=1.0,
        rebalance_interval_sec=28_800.0,
        min_rebalance_notional=50.0,
        turnover_deadband=0.04,
    )
    yield d
    db.close()


def test_daemon_fetch_tickers(daemon: ShadowTradingDaemon):
    """Verifies parsing of live ticker data."""
    ticks = daemon.fetch_live_tickers()
    assert "BTC/USDT" in ticks
    assert ticks["BTC/USDT"]["price"] == 90_000.0
    assert ticks["BTC/USDT"]["bid"] == 89_990.0
    assert ticks["BTC/USDT"]["ask"] == 90_010.0


def test_daemon_fast_cycle(daemon: ShadowTradingDaemon):
    """Verifies fast cycle prices update and mark-to-market snapshot."""
    # Place a pending Maker limit order on ETH
    daemon.broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=3_450.0,  # Below market (pending)
    )
    assert len(daemon.broker.db.get_pending_orders()) == 1

    portfolio, filled_orders = daemon.run_fast_cycle()
    assert portfolio.total_equity == 100_000.0
    # ETH ask is 3,501.0 > 3,450.0 (and 24h low alone must NOT falsely fill passive limit orders)
    assert len(filled_orders) == 0
    daemon.exchange.fetch_tickers.return_value["ETH/USDT"]["low"] = 3_440.0
    _, fake_low_fills = daemon.run_fast_cycle()
    assert len(fake_low_fills) == 0

    # Now simulate real top-of-book market dipping to 3,440.0 (ask <= 3,450.0)
    daemon.exchange.fetch_tickers.return_value["ETH/USDT"].update({"last": 3_440.0, "bid": 3_439.0, "ask": 3_441.0})
    portfolio, filled_orders = daemon.run_fast_cycle()
    assert len(filled_orders) == 1
    assert filled_orders[0].symbol == "ETH/USDT"
    assert filled_orders[0].status == OrderStatus.FILLED


def test_daemon_rebalance_cycle(daemon: ShadowTradingDaemon):
    """Verifies multi-strategy portfolio rebalance signal execution and deadband logic."""
    daemon.run_rebalance_cycle()
    assert daemon.last_rebalance_time is not None

    # Verify that rebalance snapshot was saved in database
    latest_snap = daemon.broker.db.get_latest_rebalance_snapshot()
    assert latest_snap is not None
    assert latest_snap["regime_detected"] in ("TRENDING", "RANGING", "VOLATILE")


def test_daemon_funding_cycle(daemon: ShadowTradingDaemon):
    """Verifies 8h funding settlement execution in daemon."""
    # Open an inventory position in BTC
    daemon.broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )
    cashflows = daemon.run_funding_cycle()
    assert "BTC/USDT" in cashflows
    # Long pays positive funding
    assert cashflows["BTC/USDT"] < 0.0


def test_daemon_single_step(daemon: ShadowTradingDaemon):
    """Verifies synchronous step execution."""
    res = daemon.step()
    assert "total_equity" in res
    assert "rebalance_orders" in res
    assert res["total_equity"] > 0.0


def test_daemon_anomaly_scan(daemon: ShadowTradingDaemon):
    """Verifies event-driven real-time anomaly scanning and bracket order creation."""
    from src.strategies.extreme_anomaly import OversoldBounceStrategy

    daemon.enable_anomalies = True
    daemon.anomaly_size = 15.0
    daemon.min_volume_24h_usd = 20_000_000.0

    bounce = OversoldBounceStrategy(dump_threshold_24h=-0.15, rsi_threshold=28.0, symbols_filter=None)
    daemon.allocator.oversold_bounce_strategy = bounce

    daemon.latest_tickers = {
        "EPIC/USDT": {
            "price": 0.50,
            "bid": 0.499,
            "ask": 0.501,
            "change_24h": -25.0,
            "volume_24h": 40_000_000.0,
        }
    }

    # Generate synthetic flush klines (price dropping from 1.0 to 0.50)
    import numpy as np

    closes = np.linspace(1.0, 0.50, 30)
    synthetic_klines = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
            "open": closes * 1.01,
            "high": closes * 1.02,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(30, 100_000.0),
        }
    )
    daemon.fetch_symbol_klines = MagicMock(return_value=synthetic_klines)

    orders = daemon.run_anomaly_scan()
    assert len(orders) == 1
    assert orders[0].symbol == "EPIC/USDT"
    assert orders[0].strategy == "Extreme Drop Bounce"

    # Verify position and brackets
    pos = daemon.broker.portfolio.positions["EPIC/USDT"]
    assert pos.quantity > 0
    assert pos.take_profit is not None
    assert pos.stop_loss is not None
    assert pos.take_profit > pos.entry_price
    assert pos.stop_loss < pos.entry_price

    # Verify that having >3 core 7-Factor positions open does NOT block run_anomaly_scan()
    for i, core_sym in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]):
        daemon.broker.set_market_price(core_sym, price=100.0)
        daemon.broker.create_order(
            symbol=core_sym,
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            strategy="Cross-Sectional Momentum",
        )
    daemon.latest_tickers["ZRO/USDT"] = {
        "price": 0.50,
        "bid": 0.499,
        "ask": 0.501,
        "change_24h": -25.0,
        "volume_24h": 40_000_000.0,
    }
    orders2 = daemon.run_anomaly_scan()
    assert len(orders2) == 1
    assert orders2[0].symbol == "ZRO/USDT"
    assert orders2[0].strategy == "Extreme Drop Bounce"


def test_daemon_trailing_and_profit_locks(daemon: ShadowTradingDaemon):
    """Verifies Phase 2 profit-lock (+3.5% -> +1.5%) and Phase 3 continuous trailing stop ratchet (+4.5% -> 2.0%)."""
    daemon.broker.set_market_price("WAL/USDT", price=0.0600)
    daemon.broker.create_order(
        symbol="WAL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000.0,
    )
    daemon.broker.create_bracket_orders(
        symbol="WAL/USDT",
        take_profit=0.0750,
        stop_loss=0.0570,
    )

    pos = daemon.broker.portfolio.positions["WAL/USDT"]
    assert pos.stop_loss == 0.0570
    assert pos.is_profit_locked is False

    # 1. Price rises to 0.0622 (+3.67% gain >= +3.5% trigger -> triggers Phase 2 Profit-Lock at +1.5%)
    ticks_p2 = {"WAL/USDT": {"price": 0.0622}}
    ratcheted_p2 = daemon.manage_trailing_and_profit_locks(ticks_p2)
    assert len(ratcheted_p2) == 1
    assert pos.is_profit_locked is True
    expected_lock_sl = round(pos.entry_price * 1.015, 6)
    assert pos.stop_loss == expected_lock_sl

    # 2. Price surges to 0.0660 (+10.0% gain >= +4.5% trigger -> triggers Phase 3 Trailing Stop at -2.0% from peak)
    ticks_p3 = {"WAL/USDT": {"price": 0.0660}}
    ratcheted_p3 = daemon.manage_trailing_and_profit_locks(ticks_p3)
    assert len(ratcheted_p3) == 1
    expected_trail_sl = round(0.0660 * (1.0 - 0.020), 6)
    assert pos.stop_loss == expected_trail_sl
    assert pos.stop_loss > expected_lock_sl
    assert pos.is_profit_locked is True
    assert pos.sl_pnl_pct > 7.0


def test_retoque_2_anomaly_asymmetric_cooldown_and_profit_lock(daemon: ShadowTradingDaemon):
    """
    Verifies Retoque 2 on the Satellite Anomaly Capsule:
    1) max_anomaly_positions == 3 by default.
    2) Calibrated Anomaly Profit-Lock: +3.0% peak does NOT prematurely choke an anomaly trade,
       whereas +4.2% peak locks +2.0% profit (+5.8% peak trails at 2.0% distance).
    3) Asymmetric per-symbol cooldown: 12h after Win, 24h after Stop-Loss.
    """
    from datetime import datetime, timedelta, timezone

    assert daemon.max_anomaly_positions == 3

    # Open an Early Breakout anomaly position at $1.00 on native crypto token HAEDAL/USDT
    daemon.broker.set_market_price("HAEDAL/USDT", price=1.00)
    daemon.broker.create_order(
        symbol="HAEDAL/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=15.0,
        strategy="Early Breakout",
    )
    daemon.broker.create_bracket_orders(
        symbol="HAEDAL/USDT",
        take_profit=1.15,
        stop_loss=0.96,
    )
    pos = daemon.broker.portfolio.positions["HAEDAL/USDT"]

    # At +4.0% peak ($1.04), anomaly waits for +4.5% trigger so it doesn't choke early
    r1 = daemon.manage_trailing_and_profit_locks({"HAEDAL/USDT": {"price": 1.04}})
    assert len(r1) == 0
    assert pos.stop_loss == 0.96

    # At +4.6% peak ($1.046), anomaly locks +2.5% above entry with 2.0%+ breathing room
    r2 = daemon.manage_trailing_and_profit_locks({"HAEDAL/USDT": {"price": 1.046}})
    assert len(r2) == 1
    assert pos.stop_loss == round(pos.entry_price * 1.025, 2)

    # At +6.5% peak ($1.065), anomaly trails 2.0% behind peak ($1.065 * 0.980 = $1.04)
    r3 = daemon.manage_trailing_and_profit_locks({"HAEDAL/USDT": {"price": 1.065}})
    assert len(r3) == 1
    assert pos.stop_loss == round(1.065 * 0.980, 2)

    # Close position and simulate Stop-Loss exit 15h ago (< 24h loss cooldown -> blocked)
    from src.strategies.extreme_anomaly import OversoldBounceStrategy

    daemon.allocator.oversold_bounce_strategy = OversoldBounceStrategy(
        dump_threshold_24h=-0.15, rsi_threshold=28.0, symbols_filter=None
    )
    daemon.broker.create_order(
        symbol="HAEDAL/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=15.0,
        strategy="Early Breakout",
    )
    now = datetime.now(timezone.utc)
    daemon.anomaly_last_exit["HAEDALUSDT"] = (now - timedelta(hours=15), True)
    daemon.latest_tickers = {
        "HAEDAL/USDT": {
            "price": 0.50,
            "bid": 0.499,
            "ask": 0.501,
            "change_24h": -25.0,
            "volume_24h": 40_000_000.0,
        }
    }
    closes = np.linspace(1.0, 0.50, 30)
    synthetic_klines = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
            "open": closes * 1.01,
            "high": closes * 1.02,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(30, 100_000.0),
        }
    )
    daemon.fetch_symbol_klines = MagicMock(return_value=synthetic_klines)

    # Replace DB get_trades with empty list so manual cooldown timestamp is tested directly
    daemon.broker.db.get_trades = MagicMock(return_value=[])
    blocked_orders = daemon.run_anomaly_scan()
    assert len(blocked_orders) == 0

    # If the exit 15h ago was a WIN (> 12h win cooldown), re-entry is allowed!
    daemon.anomaly_last_exit["HAEDALUSDT"] = (now - timedelta(hours=15), False)
    allowed_orders = daemon.run_anomaly_scan()
    assert len(allowed_orders) == 1
    assert allowed_orders[0].symbol == "HAEDAL/USDT"


def test_three_shields_offline_rebalance_intracycle_drift_and_crypto_only_filter(daemon: ShadowTradingDaemon):
    """
    Tests the 3 Operational Shields:
    A) Offline Rebalance Guard: Defers rebalance without advancing last_rebalance_time when live kline fetches fail.
    B) Intra-Cycle Neutrality Guard: Triggers automatic neutrality rebalance when Core net drift >= 25% of Equity.
    C) Crypto-Only Anomaly Filter: Blocks TradFi stocks/ETFs (INTC, INTW, MVLL, SOXS) and gapped/flat-bar symbols.
    """
    from datetime import datetime, timedelta, timezone

    from src.strategies.extreme_anomaly import OversoldBounceStrategy

    # --- Shield A: Offline Rebalance Guard ---
    daemon.last_rebalance_time = None
    daemon.enable_warm_start = False
    daemon.exchange.fetch_ohlcv.side_effect = RuntimeError("Network unreachable")
    orders_offline = daemon.run_rebalance_cycle()
    assert len(orders_offline) == 0
    assert daemon.last_rebalance_time is None  # Did NOT advance last_rebalance_time while offline!

    # Restore network connectivity -> rebalance succeeds and advances last_rebalance_time
    daemon.exchange.fetch_ohlcv.side_effect = None
    daemon.run_rebalance_cycle()
    assert daemon.last_rebalance_time is not None

    # --- Shield B: Intra-Cycle Neutrality Guard ---
    daemon.allocator.use_seven_factor_engine = True
    now = datetime.now(timezone.utc)
    daemon.last_rebalance_time = now - timedelta(hours=1)
    daemon.last_core_bracket_fill_time = now - timedelta(minutes=20)
    # Create unbalanced Core exposure (e.g., 2 x $16,000 = $32,000 Long on $100,000 equity = 32% >= 25% drift)
    daemon.broker.set_market_price("BTC/USDT", price=80_000.0)
    for _ in range(2):
        daemon.broker.create_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.20,
            strategy="F7_HighLiq_Size (Quality)",
        )
    assert daemon.is_intracycle_rebalance_needed(now) is True
    # If a Core bracket filled only 2 minutes ago (< 15m settle window), wait for bounce to settle
    daemon.last_core_bracket_fill_time = now - timedelta(minutes=2)
    assert daemon.is_intracycle_rebalance_needed(now) is False

    # --- Shield C: Crypto-Only Anomaly Filter ---
    daemon.allocator.oversold_bounce_strategy = OversoldBounceStrategy(
        dump_threshold_24h=-0.15, rsi_threshold=28.0, symbols_filter=None
    )
    daemon.latest_tickers = {
        "INTC/USDT": {"price": 50.0, "bid": 49.9, "ask": 50.1, "change_24h": -28.0, "volume_24h": 80_000_000.0},
        "INTW/USDT": {"price": 7.5, "bid": 7.49, "ask": 7.51, "change_24h": -55.0, "volume_24h": 90_000_000.0},
        "MVLL/USDT": {"price": 19.0, "bid": 18.9, "ask": 19.1, "change_24h": -20.0, "volume_24h": 50_000_000.0},
    }
    closes = np.linspace(100.0, 50.0, 30)
    synthetic_klines = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
            "open": closes * 1.01,
            "high": closes * 1.02,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(30, 100_000.0),
        }
    )
    daemon.fetch_symbol_klines = MagicMock(return_value=synthetic_klines)
    tradfi_orders = daemon.run_anomaly_scan()
    assert len(tradfi_orders) == 0


def test_nov3_anomaly_open_pos_tracking_and_48h_timestop(daemon: ShadowTradingDaemon):
    """
    Tests Nov 3 Anomaly Capsule Refinements:
    1) Open anomaly positions outside Top 10 Gainers/Losers still receive ticker updates in fetch_live_tickers().
    2) Anomaly positions older than 48h are automatically closed via Time-Stop to prevent zombie slots.
    3) Early Breakout requires >= 12.0% 24h change in run_anomaly_scan().
    """
    from datetime import datetime, timedelta, timezone

    # 1. Open an anomaly position on DOS/USDT that is NOT in top 10 gainers/losers
    daemon.broker.set_market_price("DOS/USDT", price=0.1877)
    daemon.broker.create_order(
        symbol="DOS/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=80.0,
        strategy="Extreme Drop Bounce",
    )
    daemon.exchange.fetch_tickers = MagicMock(
        return_value={
            "DOS/USDT:USDT": {
                "last": 0.1950,
                "bid": 0.1949,
                "ask": 0.1951,
                "high": 0.2000,
                "low": 0.1850,
                "percentage": -2.5,
                "quoteVolume": 5_000_000.0,  # Below 20M scan threshold and not top 10 loser
            }
        }
    )
    tickers = daemon.fetch_live_tickers()
    assert "DOS/USDT" in tickers
    assert tickers["DOS/USDT"]["price"] == 0.1950

    # 2. Age DOS/USDT beyond 48h -> manage_trailing_and_profit_locks closes it via Time-Stop
    pos = daemon.broker.portfolio.positions["DOS/USDT"]
    pos.opened_at = datetime.now(timezone.utc) - timedelta(hours=50)
    daemon.manage_trailing_and_profit_locks({"DOS/USDT": {"price": 0.1950}})
    assert pos.side == PositionSide.FLAT
    assert pos.quantity == 0.0
    assert "DOSUSDT" in daemon.anomaly_last_exit


def test_dynamic_confluence_anomaly_sizing(daemon: ShadowTradingDaemon):
    """
    Tests Issue #50 (Dynamic Confluence Position Sizing $10.00 - $25.00):
    1) Marginal low-dispersion setups receive near-minimum sizing (~$10-$13).
    2) High-confluence extreme capitulation / volume surge setups in high-dispersion regimes
       scale up toward $22-$25.
    """
    from src.strategies.extreme_anomaly import compute_anomaly_confluence_size

    # Marginal bounce: -18.1% drop, RSI 24.8, low volume ratio 1.25x, low market dispersion 1.8%
    sz_low, conf_low = compute_anomaly_confluence_size(
        "Extreme Drop Bounce",
        {"ret_24h": -0.181, "rsi": 24.8, "vol_ratio": 1.25},
        mkt_disp_24h=0.018,
        min_notional=10.0,
        max_notional=25.0,
    )
    # High-confluence bounce: -28% drop, RSI 12.0, volume surge 4.5x, high market dispersion 5.5%
    sz_high, conf_high = compute_anomaly_confluence_size(
        "Extreme Drop Bounce",
        {"ret_24h": -0.28, "rsi": 12.0, "vol_ratio": 4.5},
        mkt_disp_24h=0.055,
        min_notional=10.0,
        max_notional=25.0,
    )
    assert 10.0 <= sz_low <= 11.5
    assert conf_low < 0.10
    assert 24.0 <= sz_high <= 25.0
    assert conf_high > 0.90


def test_nov5_core_bracket_cooldown_and_neutrality_recentering(daemon: ShadowTradingDaemon):
    """
    Tests Nov 5 Core Bracket Cooldown & Neutrality Re-Centering:
    1) When a Core position (e.g. RENDER/USDT SHORT or ETH/USDT LONG) hits its Trailing-Stop
       or Take-Profit mid-cycle, `_refresh_anomaly_cooldowns` records `core_bracket_last_exit`
       with the exited PositionSide (and also hydrates it from SQLite trades on restart).
    2) During the 4h cooldown window (`core_bracket_cooldown_sec = 14400`), `_apply_core_bracket_cooldowns`
       blocks same-direction re-entry on that symbol ($0.00) and scales down the opposite leg
       so strict Dollar Neutrality (`sum(Longs) == sum(Shorts)`) is preserved.
    3) Opposite-direction signals on that symbol are NOT blocked, and once >= 4h have elapsed,
       normal same-direction allocation is permitted again.
    """
    from datetime import datetime, timedelta, timezone

    from src.domain.enums import OrderStatus, OrderType

    now = datetime.now(timezone.utc)

    # 1. Simulate a filled Core Trailing-Stop BUY order closing a RENDER/USDT SHORT position
    filled_sl_order = Order(
        id="shd_sl_1bc0e515d2",
        symbol="RENDER/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.STOP_LOSS,
        quantity=261.26,
        price=1.47,
        status=OrderStatus.FILLED,
        strategy="F5_Carry_Neg (72h Fund)",
    )
    daemon.broker.db.get_trades = MagicMock(return_value=[])
    daemon._refresh_anomaly_cooldowns([filled_sl_order])

    assert "RENDERUSDT" in daemon.core_bracket_last_exit
    exit_ts, exited_side = daemon.core_bracket_last_exit["RENDERUSDT"]
    assert exited_side == PositionSide.SHORT

    # 2. Simulate intra-cycle rebalance 20 minutes later where allocator still wants RENDERUSDT = -$410.0
    raw_targets = {
        "BNBUSDT": 460.8,
        "ETHUSDT": 460.8,
        "BTCUSDT": 281.6,
        "FETUSDT": 228.0,
        "RENDERUSDT": -410.0,
        "NEARUSDT": -326.6,
        "APTUSDT": -290.4,
        "ARBUSDT": -404.2,
    }
    # Raw targets are dollar-neutral: Longs = $1,431.2, Shorts = -$1,431.2
    adjusted = daemon._apply_core_bracket_cooldowns(raw_targets, now=exit_ts + timedelta(minutes=20))

    # RENDERUSDT same-direction SHORT re-entry is blocked ($0.0)
    assert adjusted["RENDERUSDT"] == 0.0
    # Remaining Shorts = -(326.6 + 290.4 + 404.2) = -$1,021.2
    # Longs are scaled down proportionally so Longs == Shorts == $1,021.2 (Net = $0.00!)
    long_sum = sum(v for v in adjusted.values() if v > 0)
    short_sum = sum(abs(v) for v in adjusted.values() if v < 0)
    assert abs(long_sum - 1021.2) < 1e-6
    assert abs(short_sum - 1021.2) < 1e-6
    assert abs(sum(adjusted.values())) < 1e-6

    # 3. If allocator flips RENDERUSDT to LONG (+$200) while SHORT is on cooldown, it is NOT blocked
    flipped_targets = {
        "BNBUSDT": 300.0,
        "RENDERUSDT": 200.0,
        "NEARUSDT": -250.0,
        "APTUSDT": -250.0,
    }
    adj_flipped = daemon._apply_core_bracket_cooldowns(flipped_targets, now=exit_ts + timedelta(minutes=20))
    assert adj_flipped["RENDERUSDT"] == 200.0

    # 4. After 4 hours + 1 minute (> 14,400s), same-direction SHORT on RENDERUSDT is allowed again
    adj_expired = daemon._apply_core_bracket_cooldowns(raw_targets, now=exit_ts + timedelta(hours=4, minutes=1))
    assert adj_expired["RENDERUSDT"] == -410.0

    # 5. Verify SQLite hydration of `core_bracket_last_exit` across daemon restarts:
    #    - Winning Take-Profit (`shd_tp_`) or positive Profit-Lock (`realized_pnl > 0`) must NOT block re-entry.
    #    - Loss-making Stop-Loss (`shd_sl_` with `realized_pnl <= 0`) MUST hydrate cooldown cleanly.
    daemon.core_bracket_last_exit.clear()
    daemon.broker.db.get_trades = MagicMock(
        return_value=[
            {
                "timestamp": (now - timedelta(minutes=25)).isoformat(),
                "symbol": "SOL/USDT",
                "side": "SELL",
                "order_id": "shd_tp_1111111111",
                "realized_pnl": 28.40,
                "strategy": "F3_Fast_Swing_PB (12h-48h)",
            },
            {
                "timestamp": (now - timedelta(minutes=20)).isoformat(),
                "symbol": "BNB/USDT",
                "side": "SELL",
                "order_id": "shd_sl_2222222222",
                "realized_pnl": 14.10,
                "strategy": "F4_Rev_24_72h (MeanRev)",
            },
            {
                "timestamp": (now - timedelta(minutes=15)).isoformat(),
                "symbol": "ETH/USDT",
                "side": "SELL",
                "order_id": "shd_sl_9dbfd234fa",
                "realized_pnl": -23.20,
                "strategy": "F7_HighLiq_Size (Quality)",
            },
        ]
    )
    daemon._refresh_anomaly_cooldowns()
    assert "SOLUSDT" not in daemon.core_bracket_last_exit
    assert "BNBUSDT" not in daemon.core_bracket_last_exit
    assert "ETHUSDT" in daemon.core_bracket_last_exit
    assert daemon.core_bracket_last_exit["ETHUSDT"][1] == PositionSide.LONG
    assert daemon.enable_subhourly_pullback is False


def test_nov9_zombie_ticker_eviction_kline_price_sync_and_tradfi_filter(daemon: ShadowTradingDaemon):
    """
    Tests Nov 9 Satellite Capsule Refinements:
    1) Zombie Ticker Eviction: Non-Core, non-open-position symbols that drop out of the current
       cycle's Top-10 Gainers/Losers are evicted from `self.latest_tickers` on refresh.
    2) Kline Price & Broker Quote Synchronization: If `latest_tickers` or `broker.market_prices`
       has a stale price ($0.2326) while fresh 1h kline close is $0.1800, `run_anomaly_scan()`
       synchronizes execution price and SL/TP brackets to $0.1800 before order creation.
    3) Expanded TradFi Filter: SNDK, UVXY, DELL, QCOM, RKLB, SAMSUNG, ANTHROPIC are blocked.
    """
    from src.strategies.extreme_anomaly import OversoldBounceStrategy

    # 1. Verify expanded TradFi equity/ETF/commodity bases are blocked
    for tradfi_sym in [
        "SNDK/USDT",
        "UVXY/USDT",
        "DELL/USDT",
        "QCOM/USDT",
        "RKLB/USDT",
        "SAMSUNG/USDT",
        "ANTHROPIC/USDT",
    ]:
        assert daemon._is_tradfi_or_gapped_symbol(tradfi_sym) is True

    # 2. Seed a stale zombie ticker ("ZOMBIE/USDT") that is NOT in Core symbols and NOT an open position
    daemon.latest_tickers["ZOMBIE/USDT"] = {
        "price": 0.50,
        "bid": 0.499,
        "ask": 0.501,
        "change_24h": -20.0,
        "volume_24h": 30_000_000.0,
    }
    daemon.exchange.fetch_tickers = MagicMock(
        return_value={
            "BTC/USDT:USDT": {
                "last": 75000.0,
                "bid": 74999.0,
                "ask": 75001.0,
                "high": 76000.0,
                "low": 74000.0,
                "percentage": -1.5,
                "quoteVolume": 500_000_000.0,
            }
        }
    )
    refreshed = daemon.fetch_live_tickers()
    assert "ZOMBIE/USDT" not in refreshed
    assert "BTC/USDT" in refreshed

    # 3. Verify Kline Price Synchronization in run_anomaly_scan() when broker/ticker had a stale price ($0.2326 vs $0.1800)
    daemon.allocator.oversold_bounce_strategy = OversoldBounceStrategy(
        dump_threshold_24h=-0.15, rsi_threshold=28.0, symbols_filter=None
    )
    daemon.broker.set_market_price("ARX/USDT", price=0.2326, bid=0.2325, ask=0.2327)
    daemon.latest_tickers = {
        "ARX/USDT": {
            "price": 0.2326,
            "bid": 0.2325,
            "ask": 0.2327,
            "change_24h": -22.0,
            "volume_24h": 35_000_000.0,
        }
    }
    closes = np.linspace(0.2800, 0.1800, 30)
    fresh_klines = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-11-08", periods=30, freq="1h", tz="UTC"),
            "open": closes * 1.01,
            "high": closes * 1.02,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(30, 150_000.0),
        }
    )
    daemon.fetch_symbol_klines = MagicMock(return_value=fresh_klines)
    daemon.broker.db.get_trades = MagicMock(return_value=[])

    orders = daemon.run_anomaly_scan()
    assert len(orders) == 1
    pos = daemon.broker.portfolio.positions["ARX/USDT"]
    # Entry price and SL must be anchored to fresh kline close ($0.1800), NOT stale $0.2326!
    assert abs(pos.entry_price - 0.1800) < 0.001
    assert abs(pos.stop_loss - round(pos.entry_price * 0.95, 6)) < 1e-5


def test_nov15_funding_cache_200_settlements_and_5x_intracycle_drift_denom(tmp_path):
    """
    Tests Nov 15 Production Audit Fixes:
    1) fetch_funding_history_72h fetches up to 200 8h settlements (1,600h > 1,000h matrix),
       persists settled records to `FUNDING/{SYM}.parquet`, and SevenFactorOnlineEngine only
       charges `funding_per_bar` on true 8h settlement hours (`hour % 8 == 0`), eliminating
       artificial 8h Carry/Fast factor flips.
    2) is_intracycle_rebalance_needed normalizes net drift by `max(core_gross_usd, equity)`
       so a single TP/Profit-Lock at 5x leverage (~10-12% of gross book) does NOT falsely
       trigger premature mid-bar book churn, while >=25% gross book drift DOES trigger.
    """
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0, max_gross_leverage=6.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        cache_dir=str(tmp_path),
        enable_warm_start=True,
    )

    # 1. Verify 200-settlement funding fetch & Parquet persistence
    base_ts = pd.Timestamp("2026-11-01T00:00:00Z")
    synthetic_hist = [
        {
            "timestamp": int((base_ts + pd.Timedelta(hours=8 * i)).timestamp() * 1000),
            "fundingRate": 0.0001,
        }
        for i in range(150)
    ]

    class DummyFundingExchange:
        @staticmethod
        def fetch_funding_rate_history(symbol, limit=200):
            assert limit >= 150
            return synthetic_hist

        @staticmethod
        def fetch_funding_rate(symbol):
            return {"fundingRate": 0.0001}

    daemon.exchange = DummyFundingExchange()
    df_f = daemon.fetch_funding_history_72h("BTC/USDT")
    assert len(df_f) >= 150
    parquet_file = tmp_path / "FUNDING" / "BTCUSDT.parquet"
    assert parquet_file.exists()
    cached_parquet = pd.read_parquet(parquet_file)
    assert len(cached_parquet) == 150

    # 2. Verify 5x leverage intracycle drift denominator (core_gross_usd vs equity)
    from datetime import timedelta

    daemon.allocator.use_seven_factor_engine = True
    now = datetime.now(timezone.utc)
    daemon.last_rebalance_time = now - timedelta(hours=1)
    daemon.last_core_bracket_fill_time = now - timedelta(minutes=20)

    # Simulate 5x leverage ($3,000 equity, $7,000 Long vs $5,600 Short after one $1,400 Short hit TP):
    # Drift is $1,400 / $3,000 = 46.7% of Equity, but only $1,400 / $12,600 = 11.1% of Gross (< 25%)!
    broker.set_market_price("BTC/USDT", price=1000.0)
    broker.set_market_price("ETH/USDT", price=1000.0)
    broker.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=7.0,
        strategy="F3_Fast_Swing_PB (12h-48h)",
    )
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=5.6,
        strategy="F7_HighLiq_Size (Quality)",
    )
    assert daemon.is_intracycle_rebalance_needed(now) is False

    # Now simulate multiple shorts closing so Short leg drops to $3,600 ($7,000 Long vs $3,600 Short):
    # Gross = $10,600, Net Drift = $3,400 / $10,600 = 32.1% >= 25.0% -> MUST trigger!
    broker.create_order(
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=2.0,
        strategy="F7_HighLiq_Size (Quality)",
    )
    assert daemon.is_intracycle_rebalance_needed(now) is True
    db.close()


def test_nov15_zombie_stub_pruning_signflip_deadband_and_anomaly_guards():
    """
    Tests Nov 15 Deep Audit Fixes:
    1) SteppedLeverageController.scale_and_deadband_weights() prunes <5% leg-share tail stubs
       on the 15-coin universe and releases sign-flipped (e.g. SUIUSDT) and sub-deadband stubs
       to 0.0 instead of trapping them inside the 9.09% ($251) deadband.
    2) Anomaly strategies block upper-wick blow-offs (EarlyBreakout), vertical short-squeeze candles
       (PumpFundingFade), vertical crash bars (OversoldBounce), and negative-return freefalls
       (ExtremeFundingHarvester).
    3) run_anomaly_scan() blocks Core 7-Factor universe symbols from being hijacked by Anomaly
       strategies, and manage_trailing_and_profit_locks() auto-closes orphan Funding Harvester Hedges.
    """
    from src.risk.stepped_leverage import SteppedLeverageController
    from src.strategies.extreme_anomaly import (
        EarlyBreakoutStrategy,
        ExtremeFundingHarvesterStrategy,
        OversoldBounceStrategy,
        PumpFundingFadeStrategy,
    )

    # 1. Verify Zombie Stub Pruning & Sign-Flip Release on the 15-coin live snapshot
    ctrl = SteppedLeverageController(initial_capital=2767.0, start_leverage=5.0, max_leverage=5.0)
    unit_w = {
        "BTCUSDT": 0.0811,
        "ARBUSDT": 0.0588,
        "LINKUSDT": 0.0536,
        "ETHUSDT": 0.0465,
        "NEARUSDT": 0.0111,  # 4.1% of Long leg (< 5% -> pruned!)
        "XRPUSDT": 0.0084,  # 3.2% of Long leg (< 5% -> pruned!)
        "SOLUSDT": 0.0066,
        "BNBUSDT": 0.0031,
        "ADAUSDT": 0.0010,
        "FETUSDT": 0.0001,
        "SUIUSDT": -0.0021,  # 0.8% of Short leg + sign-flipped vs +$29.84 current -> pruned!
        "DOGEUSDT": -0.0150,
        "AVAXUSDT": -0.0845,
        "APTUSDT": -0.0845,
        "RENDERUSDT": -0.0845,
    }
    eq = 2767.09
    curr_w = {
        "BTCUSDT": 1422.49 / eq,
        "ARBUSDT": 275.36 / eq,
        "LINKUSDT": 1001.97 / eq,
        "ETHUSDT": 932.54 / eq,
        "NEARUSDT": 127.96 / eq,
        "XRPUSDT": 201.27 / eq,
        "BNBUSDT": 159.65 / eq,
        "ADAUSDT": 13.63 / eq,
        "SUIUSDT": 29.84 / eq,  # Previously trapped LONG despite negative score!
        "DOGEUSDT": -333.20 / eq,
        "AVAXUSDT": -1351.20 / eq,
        "APTUSDT": -1352.51 / eq,
        "RENDERUSDT": -1152.80 / eq,
    }
    final_w, _ = ctrl.scale_and_deadband_weights(
        unit_weights=unit_w,
        current_equity=eq,
        vol_scalar=1.0,
        current_weights=curr_w,
    )
    # All 5 zombie Long stubs (NEAR, XRP, BNB, ADA, SUI) must be closed (absent from final_w)
    for stub_sym in ("NEARUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT", "SUIUSDT", "SOLUSDT", "FETUSDT"):
        assert stub_sym not in final_w
    # Top 4 Longs and Top 4 Shorts remain active and strictly dollar-neutral
    for leader_sym in ("BTCUSDT", "ARBUSDT", "LINKUSDT", "ETHUSDT"):
        assert final_w.get(leader_sym, 0.0) > 0.0
    for short_sym in ("AVAXUSDT", "APTUSDT", "RENDERUSDT", "DOGEUSDT"):
        assert final_w.get(short_sym, 0.0) < 0.0
    assert pytest.approx(sum(final_w.values()), abs=1e-6) == 0.0

    # 2. Verify Anomaly Candle-Structure Guards on today's exact blow-off/crash patterns
    idx = pd.date_range("2026-11-15", periods=30, freq="1h", tz="UTC")
    # 2a. JCT/USDT upper-wick blow-off (spiked to 0.00284, closed at 0.00225 = 20.8% below 4h high)
    c_jct = np.full(30, 0.00200)
    c_jct[-4:] = [0.00201, 0.00210, 0.00225, 0.00228]
    h_jct = c_jct * 1.01
    h_jct[-2] = 0.00284  # Massive upper blow-off wick
    v_jct = np.full(30, 1000.0)
    v_jct[-1] = 5000.0
    df_jct = pd.DataFrame(
        {"open": c_jct, "high": h_jct, "low": c_jct * 0.99, "close": c_jct, "volume": v_jct}, index=idx
    )
    eb = EarlyBreakoutStrategy(pump_threshold_4h=0.08, volume_multiplier=2.5)
    assert len(eb.generate_signals({"JCT/USDT": df_jct})) == 0

    # 2b. EVAA/USDT vertical +16% 1h short-squeeze bar -> PumpFundingFade must wait for deceleration
    c_evaa = np.linspace(1.80, 2.41, 30)
    c_evaa[-1] = 2.80  # +16.2% in the final 1h bar
    df_evaa = pd.DataFrame(
        {"open": c_evaa, "high": c_evaa * 1.01, "low": c_evaa * 0.99, "close": c_evaa, "volume": np.full(30, 1000.0)},
        index=idx,
    )
    pf = PumpFundingFadeStrategy(pump_threshold_24h=0.25, min_funding_rate=0.0005)
    assert (
        len(
            pf.generate_signals(
                {"EVAA/USDT": df_evaa}, funding_data={"EVAA/USDT": pd.DataFrame([{"funding_rate": 0.0010}])}
            )
        )
        == 0
    )

    # 2c. BTW/USDT vertical -6.6% crash bar closing at low -> OversoldBounce must not catch falling knife
    c_btw = np.linspace(0.80, 0.605, 30)
    c_btw[-1] = 0.565  # -6.6% 1h crash bar
    df_btw = pd.DataFrame(
        {"open": c_btw * 1.01, "high": c_btw * 1.01, "low": c_btw, "close": c_btw, "volume": np.full(30, 1000.0)},
        index=idx,
    )
    ob = OversoldBounceStrategy(dump_threshold_24h=-0.15, rsi_threshold=28.0)
    assert len(ob.generate_signals({"BTW/USDT": df_btw})) == 0

    # 2d. SOPH/USDT negative 4h/1h freefall -> ExtremeFundingHarvester must block LONG entry
    c_soph = np.linspace(0.00160, 0.00138, 30)
    df_soph = pd.DataFrame(
        {"open": c_soph, "high": c_soph * 1.01, "low": c_soph * 0.99, "close": c_soph, "volume": np.full(30, 1000.0)},
        index=idx,
    )
    fh = ExtremeFundingHarvesterStrategy()
    assert (
        len(
            fh.generate_signals(
                {"SOPH/USDT": df_soph}, funding_data={"SOPH/USDT": pd.DataFrame([{"funding_rate": -0.0030}])}
            )
        )
        == 0
    )

    # 3. Verify Core symbol exclusion in run_anomaly_scan and Orphan Harvester Hedge cleanup
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0)
    d = ShadowTradingDaemon(broker=broker, symbols=["APT/USDT", "SOL/USDT"], enable_warm_start=False)
    d.allocator.oversold_bounce_strategy = OversoldBounceStrategy(dump_threshold_24h=-0.15, rsi_threshold=28.0)
    d.latest_tickers = {
        "APT/USDT": {"price": 0.44, "bid": 0.439, "ask": 0.441, "change_24h": -22.0, "volume_24h": 50_000_000.0}
    }
    assert len(d.run_anomaly_scan()) == 0  # APT/USDT is a Core symbol -> never hijacked by Anomaly scan

    # Open an orphan Funding Harvester Hedge on SOL/USDT with no active Extreme Funding Harvester position
    broker.set_market_price("SOL/USDT", price=72.0)
    broker.create_order(
        symbol="SOL/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.0,
        strategy="Funding Harvester Hedge",
    )
    assert broker.portfolio.positions["SOL/USDT"].side == PositionSide.SHORT
    d.manage_trailing_and_profit_locks({"SOL/USDT": {"price": 71.5}})
    assert broker.portfolio.positions["SOL/USDT"].side == PositionSide.FLAT
    db.close()


def test_three_shield_rules_feature_flags_and_hard_equity_floor():
    """
    Tests the 3 Reglas de Blindaje & Runtime Feature Flag system:
    1) Regla #1: Feature flag `enable_anomaly_capsule` defaults to False in Mainnet and, when
       disabled (at startup or via live SQLite hot-reload), automatically closes any open
       satellite anomaly position (`TAC/USDT`) while leaving Core 7-Factor legs intact.
    2) Regla #2: Hard Equity Floor (`equity_floor_usd = 2600.0`) flattens 100% of positions,
       cancels open orders, and halts the daemon (`status='HALTED_EQUITY_FLOOR'`) when equity <= $2,600.
    3) Regla #3: `evaluate_shield_invariants()` audits and persists the 4 production invariants
       into SQLite `shield_invariants_telemetry`.
    """
    from src.execution.feature_flags import FeatureFlags

    # 1. Verify FeatureFlags defaults for Mainnet vs Shadow
    ff_mn = FeatureFlags.from_env_and_args(mode="mainnet", environ={})
    assert ff_mn.enable_anomaly_capsule is False
    assert ff_mn.auto_close_anomalies_when_disabled is True
    assert ff_mn.enable_hard_equity_floor is True
    assert ff_mn.equity_floor_usd == 2500.0

    ff_shd = FeatureFlags.from_env_and_args(mode="shadow", environ={})
    assert ff_shd.enable_anomaly_capsule is True
    assert ff_shd.equity_floor_usd == 0.0

    # 2. Verify Regla #1: Live feature flag hot-reload & automatic closure of open anomaly position
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=2750.0)
    daemon = ShadowTradingDaemon(
        broker=broker,
        symbols=["LINK/USDT", "APT/USDT"],
        enable_anomalies=True,
        equity_floor_usd=2600.0,
        enable_hard_equity_floor=True,
        enable_warm_start=False,
    )
    # Open 1 Core LONG (LINK/USDT), 1 Core SHORT (APT/USDT), and 1 Satellite Anomaly SHORT (TAC/USDT)
    broker.set_market_price("LINK/USDT", price=14.0)
    broker.set_market_price("APT/USDT", price=2.0)
    broker.set_market_price("TAC/USDT", price=0.0013)
    broker.create_order(
        symbol="LINK/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        strategy="7Factor:F3_Fast_Swing_PB",
    )
    broker.create_order(
        symbol="APT/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=350.0,
        strategy="7Factor:F3_Fast_Swing_PB",
    )
    broker.create_order(
        symbol="TAC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=12000.0,
        strategy="Pump & Funding Fade",
    )
    assert broker.portfolio.positions["TAC/USDT"].side == PositionSide.SHORT
    assert broker.portfolio.positions["LINK/USDT"].side == PositionSide.LONG

    # Toggle `enable_anomaly_capsule = False` in SQLite at runtime (zero-downtime hot reload)
    db.set_feature_flag("enable_anomaly_capsule", False)
    daemon.sync_feature_flags_from_db()
    assert daemon.enable_anomalies is False

    # Next fast cycle's trailing/bracket manager auto-closes TAC/USDT while keeping LINK & APT open
    daemon.manage_trailing_and_profit_locks(
        {
            "LINK/USDT": {"price": 14.0},
            "APT/USDT": {"price": 2.0},
            "TAC/USDT": {"price": 0.0013},
        }
    )
    assert broker.portfolio.positions["TAC/USDT"].side == PositionSide.FLAT
    assert broker.portfolio.positions["LINK/USDT"].side == PositionSide.LONG
    assert broker.portfolio.positions["APT/USDT"].side == PositionSide.SHORT

    # 3. Verify Regla #3: Shield Invariants Audit passes when book is clean & dollar-neutral
    inv_report = daemon.evaluate_shield_invariants()
    assert inv_report["all_invariants_ok"] is True
    assert inv_report["inv1_no_factor_flip_flop"] is True
    assert inv_report["inv2_no_zombie_stubs"] is True
    assert inv_report["inv3_no_spurious_intracycle"] is True
    assert inv_report["inv4_dollar_neutrality"] is True
    assert inv_report["inv5_anomaly_capsule_isolated"] is True
    assert inv_report["inv6_above_equity_floor"] is True
    saved_tel = db.get_latest_shield_invariants_telemetry()
    assert saved_tel is not None and saved_tel["status"] == "ALL_OK"

    # 4. Verify Regla #2: Hard Equity Floor ($2,600.00) flattens all positions and halts daemon
    # Simulate an adverse price move that brings total equity to $2,590.00 (<= $2,600 floor)
    broker.mark_to_market({"LINK/USDT": 10.5, "APT/USDT": 2.0})
    assert broker.portfolio.total_equity < 2600.0
    triggered = daemon.enforce_hard_equity_floor_and_kill_switch(broker.portfolio)
    assert triggered is True
    assert daemon.is_halted is True
    assert "HARD_EQUITY_FLOOR_BREACH" in daemon.halt_reason
    assert broker.portfolio.positions["LINK/USDT"].side == PositionSide.FLAT
    assert broker.portfolio.positions["APT/USDT"].side == PositionSide.FLAT
    db.close()


def test_chop_range_arb_specialist_and_adaptive_chop_tp():
    """
    Tests the Unified Chop & Whipsaw Intra-Sector Range-Arb Specialist:
    1) `SevenFactorOnlineEngine(enable_chop_range_arb=True, f5_min_tstat=1.25)` computes
       causal `chop_specialist_tstat` and `chop_overlay_weight` while preserving strict
       100% dollar neutrality (`sum(unit_weights) == 0.0`).
    2) When the engine is in Chop Regime (`max(t_macro, t_med, t_fast) <= 0` or `chop_overlay_weight >= 0.25`),
       `ShadowTradingDaemon.ensure_position_brackets()` tightens Core passive Maker Take-Profit
       from `+5.0%` (`core_take_profit_pct`) to `+3.5%` (`chop_take_profit_pct`).
    """
    from src.strategies.seven_factor_engine import SevenFactorAllocationResult, SevenFactorOnlineEngine

    idx = pd.date_range("2026-10-01", periods=400, freq="1h", tz="UTC")
    rng = np.random.default_rng(116)
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "FETUSDT", "RENDERUSDT", "APTUSDT", "NEARUSDT"]
    bars_1h = {}
    for i, s in enumerate(syms):
        # Generate mean-reverting range-bound prices (Ornstein-Uhlenbeck chop)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.85 * x[t - 1] + rng.normal(0.0, 0.012)
        c = (100.0 + i * 10.0) * np.exp(x)
        bars_1h[s] = pd.DataFrame(
            {
                "open": c * 0.999,
                "high": c * 1.005,
                "low": c * 0.995,
                "close": c,
                "volume": np.full(400, 10_000.0),
            },
            index=idx,
        )

    eng = SevenFactorOnlineEngine(
        f6_min_tstat=0.75,
        f5_min_tstat=1.25,
        enable_chop_range_arb=True,
    )
    res = eng.allocate_latest(bars_1h=bars_1h)
    assert pytest.approx(sum(res.unit_weights.values()), abs=1e-5) == 0.0
    assert isinstance(res.chop_specialist_tstat, float)
    assert 0.0 <= res.chop_overlay_weight <= 0.75

    # Verify adaptive Chop Take-Profit (+3.5%) in ShadowTradingDaemon.ensure_position_brackets()
    db = ShadowDatabase(db_path=":memory:")
    broker = ShadowExecutionBroker(database=db, initial_cash=3000.0)
    allocator = MetaStrategyAllocator(seven_factor_engine=eng, use_seven_factor_engine=True)
    allocator.last_seven_factor_result = SevenFactorAllocationResult(
        unit_weights={"BTCUSDT": 0.25, "APTUSDT": -0.25},
        vol_scalar=1.0,
        port_vol_72h=0.003,
        median_vol=0.003,
        conviction=0.60,
        factor_weights={"macro": 0.0, "med": 0.0, "fast": 0.0, "rev": 0.65, "carry": 0.0, "liq": 0.0, "size": 0.35},
        factor_tstats={
            "macro": -0.60,
            "med": -0.55,
            "fast": -0.40,
            "rev": 0.50,
            "carry": -0.20,
            "liq": -0.30,
            "size": 0.10,
        },
        asset_betas={"BTCUSDT": 1.0, "APTUSDT": 1.1},
        portfolio_beta=0.0,
        chop_specialist_tstat=1.45,
        chop_overlay_weight=0.55,
    )
    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        symbols=["BTC/USDT", "APT/USDT"],
        core_take_profit_pct=0.050,
        enable_warm_start=False,
    )
    broker.set_market_price("APT/USDT", price=1.0000)
    broker.create_order(
        symbol="APT/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=500.0,
        strategy="F4_Rev_24_72h (MeanRev)",
    )
    daemon.ensure_position_brackets()
    # In Chop Regime without symbol vol override, SHORT @ ~1.0000 gets +3.5% TP (~0.9649) instead of +5.0% TP (~0.9499)
    assert pytest.approx(broker.portfolio.positions["APT/USDT"].take_profit, abs=1e-3) == 0.965

    # Now supply per-asset 24h 1h-return volatility (`asset_vol_24h`) and verify dynamic scaling:
    # Low-vol symbol (sigma_1h = 0.0020 -> clipped to chop_min_tp_pct = 2.80%)
    allocator.last_seven_factor_result.asset_vol_24h = {"BTCUSDT": 0.0020, "APTUSDT": 0.0100}
    btc_tp, btc_trig, btc_lock = daemon.compute_dynamic_barrier_profile("BTC/USDT")
    apt_tp, apt_trig, apt_lock = daemon.compute_dynamic_barrier_profile("APT/USDT")
    assert pytest.approx(btc_tp, abs=1e-4) == 0.0280
    assert pytest.approx(btc_trig, abs=1e-4) == 0.0240
    assert pytest.approx(btc_lock, abs=1e-4) == 0.0120
    # Higher-vol symbol (sigma_1h = 0.0100 -> 2*0.0100*1.6933 = 3.39%, within [2.8%, 3.6%])
    assert pytest.approx(apt_tp, abs=1e-4) == 0.0339
    assert pytest.approx(apt_trig, abs=1e-4) == 0.0240
    assert pytest.approx(apt_lock, abs=1e-4) == 0.0120

    # In Active Trend Regime, TP stays at +5.0% so winning trends are never cut early
    allocator.last_seven_factor_result.factor_tstats["macro"] = 1.20
    allocator.last_seven_factor_result.chop_overlay_weight = 0.0
    trend_tp, trend_trig, trend_lock = daemon.compute_dynamic_barrier_profile("BTC/USDT")
    assert pytest.approx(trend_tp, abs=1e-4) == 0.0500
    assert pytest.approx(trend_trig, abs=1e-4) == 0.0350
    assert pytest.approx(trend_lock, abs=1e-4) == 0.0150

    # Verify Holistic Regime Shields (`enable_holistic_regime_shields=True`) & Cooldown Water-Fill / Stub Pruning
    eng_hol = SevenFactorOnlineEngine(
        f6_min_tstat=0.75,
        f5_min_tstat=1.25,
        enable_chop_range_arb=True,
        enable_holistic_regime_shields=True,
    )
    res_hol = eng_hol.allocate_latest(bars_1h=bars_1h)
    assert pytest.approx(sum(res_hol.unit_weights.values()), abs=1e-5) == 0.0

    # Verify `_apply_core_bracket_cooldowns` water-fills remaining Longs and prunes <$160 Short stubs
    now_dt = datetime.now(timezone.utc)
    daemon.core_bracket_last_exit = {
        "ADAUSDT": (now_dt, PositionSide.LONG),
        "ARBUSDT": (now_dt, PositionSide.LONG),
        "AVAXUSDT": (now_dt, PositionSide.LONG),
    }
    raw_targets = {
        "ADA/USDT": 400.0,
        "ARB/USDT": 1200.0,
        "AVAX/USDT": 1200.0,
        "SOL/USDT": 500.0,
        "BNB/USDT": 500.0,
        "ETH/USDT": -1100.0,
        "NEAR/USDT": -1000.0,
        "LINK/USDT": -1000.0,
        "DOGE/USDT": -500.0,
        "BTC/USDT": -200.0,
    }
    adj_targets = daemon._apply_core_bracket_cooldowns(raw_targets, now=now_dt)
    long_after = sum(v for v in adj_targets.values() if v > 0)
    short_after = sum(abs(v) for v in adj_targets.values() if v < 0)
    assert pytest.approx(long_after, abs=1e-2) == short_after
    # Remaining 2 Longs (SOL, BNB) must be water-filled to 2 * 35% * $3,800 = $2,660 (not left at $1,000!)
    assert pytest.approx(long_after, abs=1.0) == 2660.0
    # And zero active positions < $150 (BTC/USDT stub must be pruned and redistributed!)
    for v in adj_targets.values():
        if abs(v) > 1e-6:
            assert abs(v) >= 160.0

    db.close()
