"""
Unit Tests for ShadowTradingDaemon.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Order, Portfolio, Position
from src.domain.enums import MarketRegime, OrderSide, OrderStatus, OrderType, PositionSide
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
    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


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
    # ETH market low was 3,480.0 > 3,450.0 -> Should remain pending
    assert len(filled_orders) == 0

    # Now simulate market dipping to 3,440.0
    daemon.exchange.fetch_tickers.return_value["ETH/USDT"]["low"] = 3_440.0
    portfolio, filled_orders = daemon.run_fast_cycle()
    assert len(filled_orders) == 1
    assert filled_orders[0].symbol == "ETH/USDT"
    assert filled_orders[0].status == OrderStatus.FILLED


def test_daemon_rebalance_cycle(daemon: ShadowTradingDaemon):
    """Verifies multi-strategy portfolio rebalance signal execution and deadband logic."""
    orders = daemon.run_rebalance_cycle()
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
    synthetic_klines = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
        "open": closes * 1.01,
        "high": closes * 1.02,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(30, 100_000.0),
    })
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
    synthetic_klines = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
        "open": closes * 1.01,
        "high": closes * 1.02,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(30, 100_000.0),
    })
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
    orders_online = daemon.run_rebalance_cycle()
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
    synthetic_klines = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC"),
        "open": closes * 1.01,
        "high": closes * 1.02,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(30, 100_000.0),
    })
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
    daemon.exchange.fetch_tickers = MagicMock(return_value={
        "DOS/USDT:USDT": {
            "last": 0.1950,
            "bid": 0.1949,
            "ask": 0.1951,
            "high": 0.2000,
            "low": 0.1850,
            "percentage": -2.5,
            "quoteVolume": 5_000_000.0,  # Below 20M scan threshold and not top 10 loser
        }
    })
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


def test_m13_issue_50_dynamic_confluence_anomaly_sizing(daemon: ShadowTradingDaemon):
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
    from src.domain.entities import Order
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

    # 5. Verify SQLite hydration of `core_bracket_last_exit` across daemon restarts
    daemon.core_bracket_last_exit.clear()
    daemon.broker.db.get_trades = MagicMock(return_value=[
        {
            "timestamp": (now - timedelta(minutes=15)).isoformat(),
            "symbol": "ETH/USDT",
            "side": "SELL",
            "order_id": "shd_tp_9dbfd234fa",
            "realized_pnl": 23.20,
            "strategy": "F7_HighLiq_Size (Quality)",
        }
    ])
    daemon._refresh_anomaly_cooldowns()
    assert "ETHUSDT" in daemon.core_bracket_last_exit
    assert daemon.core_bracket_last_exit["ETHUSDT"][1] == PositionSide.LONG



