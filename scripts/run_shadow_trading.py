"""
Master CLI Launcher for Live Shadow Trading Engine.
Orchestrates ShadowTradingDaemon and ShadowMonitor against Binance Mainnet.

Usage:
    # Run a single step (poll Binance Mainnet, evaluate orders, record snapshot):
    python scripts/run_shadow_trading.py --once

    # Run the continuous daemon loop in background/terminal:
    python scripts/run_shadow_trading.py --daemon

    # Run the real-time terminal telemetry monitor:
    python scripts/run_shadow_trading.py --monitor
"""

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_dotenv_file() -> None:
    """Loads key=value or 'export key=value' pairs from .env into os.environ."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv_file()

from src.execution.database import DEFAULT_DB_PATH, ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.live_broker import BinanceLiveBroker
from src.execution.monitor import ShadowMonitor
from src.execution.rate_limiter import ExchangeRateLimiter
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import DEFAULT_SHADOW_SYMBOLS, ShadowTradingDaemon
from src.execution.ws_manager import ResilientWebSocketClient
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.cointegrated_pairs import CointegratedPairsStrategy
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("shadow_trading_runner")


def main():
    parser = argparse.ArgumentParser(description="Live Mainnet Shadow Trading Engine")
    parser.add_argument("--daemon", action="store_true", help="Run the autonomous shadow trading daemon loop")
    parser.add_argument("--monitor", action="store_true", help="Launch the real-time Rich terminal telemetry dashboard")
    parser.add_argument("--once", action="store_true", help="Execute a single step across all cycles and print status")
    parser.add_argument("--shadow", action="store_true", help="Run in local Shadow Lab mode (data/shadow_trading.db, 0 capital risk)")
    parser.add_argument("--mainnet", action="store_true", help="Enable LIVE Mainnet execution on Binance USDS-M Futures (data/live_trading.db)")
    parser.add_argument("--testnet", action="store_true", help="Enable Binance Futures Testnet execution (data/testnet_trading.db)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial shadow portfolio capital in USDT (default: 1000.0)")
    parser.add_argument("--reset", action="store_true", help="Reset all previous database records before starting")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. 'BTC/USDT,ETH/USDT')")
    parser.add_argument("--fast-interval", type=float, default=10.0, help="Fast loop interval in seconds (default: 10.0)")
    parser.add_argument("--rebalance-interval", type=float, default=14400.0, help="Rebalance interval in seconds (default: 14400 = 4h)")
    parser.add_argument("--start-leverage", type=float, default=2.5, help="Initial gross leverage tier (default: 2.5x)")
    parser.add_argument("--max-leverage", type=float, default=5.0, help="Maximum stepped gross leverage tier (default: 5.0x)")
    parser.add_argument("--step-leverage", type=float, default=0.5, help="Leverage increment per HWM milestone (default: 0.5x)")
    parser.add_argument("--milestone-gain", type=float, default=0.15, help="HWM compound gain required per +0.5x step (default: 0.15 = +15%%)")
    parser.add_argument("--legacy-allocator", action="store_true", help="Use legacy 3-strategy allocator instead of the 7-Factor Causal Online Engine")
    parser.add_argument(
        "--enable-anomalies",
        dest="enable_anomalies",
        action="store_true",
        default=True,
        help="Enable extreme anomaly & illiquid tail strategies (ENABLED by default)",
    )
    parser.add_argument(
        "--disable-anomalies",
        dest="enable_anomalies",
        action="store_false",
        help="Disable extreme anomaly satellite strategies",
    )
    parser.add_argument("--enable-pairs", action="store_true", help="Enable Cointegrated Pairs strategy alongside the 3 core Meta-Strategy pillars")
    parser.add_argument("--anomaly-size", type=float, default=15.0, help="Max dollar allocation per anomaly event (default: 15.0 USDT test mode, 100.0 USDT prod)")
    parser.add_argument("--min-volume", type=float, default=20000000.0, help="Minimum 24h volume filter in USDT for anomaly scanning (default: 20M)")
    parser.add_argument("--profit-lock-trigger", type=float, default=0.035, help="Peak gain required to arm Core Profit-Lock Stop-Profit (default: 0.035 = +3.5%%)")
    parser.add_argument("--profit-lock-level", type=float, default=0.015, help="Net profit floor locked when Core Profit-Lock triggers (default: 0.015 = +1.5%%)")
    parser.add_argument("--trailing-trigger", type=float, default=0.045, help="Peak gain required to arm Core continuous Trailing Stop (default: 0.045 = +4.5%%)")
    parser.add_argument("--trailing-distance", type=float, default=0.020, help="Distance behind peak price for Core Trailing Stop (default: 0.020 = 2.0%%)")
    parser.add_argument("--core-tp", type=float, default=0.080, help="Passive Maker Take-Profit distance for core/7-Factor legs (default: 0.080 = +8.0%%)")

    args = parser.parse_args()

    if args.mainnet and args.db == DEFAULT_DB_PATH:
        args.db = str(Path(DEFAULT_DB_PATH).parent / "live_trading.db")
    elif args.testnet and args.db == DEFAULT_DB_PATH:
        args.db = str(Path(DEFAULT_DB_PATH).parent / "testnet_trading.db")

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SHADOW_SYMBOLS.copy()
    if args.enable_anomalies and args.legacy_allocator and not args.symbols:
        for extra in ["CTSI/USDT", "MAGIC/USDT", "OGN/USDT", "RLC/USDT"]:
            if extra not in symbols:
                symbols.append(extra)

    # 1. Monitor Mode
    if args.monitor:
        logger.info(f"Launching Shadow Monitor connected to {args.db}...")
        monitor = ShadowMonitor(db_path=args.db)
        monitor.run_live()
        return

    # 2. Daemon Initialization
    logger.info(f"Initializing Trading Database at {args.db}...")
    db = ShadowDatabase(db_path=args.db)
    if args.reset:
        logger.info("Resetting all previous database records...")
        db.reset_all_data()

    live_broker = None
    if args.mainnet or args.testnet:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if args.mainnet and (not api_key or not api_secret):
            logger.error("BINANCE_API_KEY and BINANCE_API_SECRET are required in .env or environment for --mainnet.")
            sys.exit(1)
        kill_switch = EmergencyKillSwitch(
            max_24h_drawdown_pct=0.12,
            max_net_exposure_pct=0.15,
            max_margin_ratio=0.50,
        )
        live_broker = BinanceLiveBroker(
            api_key=api_key,
            api_secret=api_secret,
            testnet=args.testnet,
            dry_run=False,
            initial_capital=args.capital,
            max_gross_leverage=max(6.0, args.max_leverage * 1.25),
            chase_interval_sec=3.0,
            max_chase_iterations=4,
            allow_taker_fallback=True,
            kill_switch=kill_switch,
        )
        cfg = live_broker.ensure_account_configuration(
            symbols=symbols,
            target_leverage=max(3.0, args.start_leverage),
        )
        logger.info(
            f"🟢 LIVE BINANCE FUTURES CONNECTED | Mode={cfg.get('position_mode')} | "
            f"USDT Balance=${cfg.get('usdt_total_balance', 0.0):,.2f}"
        )

    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=args.capital,
        max_gross_leverage=max(6.0, args.max_leverage * 1.25),
        live_broker=live_broker,
    )
    router = SmartExecutionRouter(broker=broker)
    pairs_strategy = CointegratedPairsStrategy() if args.enable_pairs else None
    use_seven_factor = not args.legacy_allocator
    seven_factor_engine = SevenFactorOnlineEngine() if use_seven_factor else None
    stepped_leverage_ctrl = (
        SteppedLeverageController(
            initial_capital=args.capital,
            start_leverage=args.start_leverage,
            max_leverage=args.max_leverage,
            step_leverage=args.step_leverage,
            gain_per_step=args.milestone_gain,
            dd_bracket=0.10,
        )
        if use_seven_factor
        else None
    )

    if use_seven_factor:
        logger.info(
            f"7-Factor Causal Online Engine ENABLED | Stepped Leverage: "
            f"{args.start_leverage:.1f}x -> {args.max_leverage:.1f}x "
            f"(+{args.step_leverage:.1f}x per +{args.milestone_gain*100:.0f}% HWM)"
        )

    if args.enable_anomalies:
        from src.strategies.extreme_anomaly import (
            create_illiquid_early_breakout,
            create_illiquid_funding_fade,
            create_illiquid_oversold_bounce,
        )
        bounce_strat = create_illiquid_oversold_bounce(test_mode=(args.anomaly_size <= 25.0))
        bounce_strat.symbols_filter = None
        bounce_strat.max_position_usd = args.anomaly_size
        breakout_strat = create_illiquid_early_breakout(test_mode=(args.anomaly_size <= 25.0))
        breakout_strat.symbols_filter = None
        breakout_strat.max_position_usd = args.anomaly_size
        fade_strat = create_illiquid_funding_fade(test_mode=(args.anomaly_size <= 25.0))
        fade_strat.symbols_filter = None
        fade_strat.max_position_usd = args.anomaly_size

        logger.info(f"Extreme Anomaly Strategies ENABLED on Illiquid Tail (Max Notional: ${args.anomaly_size:.2f} USDT)")
        allocator = MetaStrategyAllocator(
            pairs_strategy=pairs_strategy,
            oversold_bounce_strategy=bounce_strat,
            early_breakout_strategy=breakout_strat,
            funding_fade_strategy=fade_strat,
            seven_factor_engine=seven_factor_engine,
            stepped_leverage_controller=stepped_leverage_ctrl,
            use_seven_factor_engine=use_seven_factor,
        )
    else:
        allocator = MetaStrategyAllocator(
            pairs_strategy=pairs_strategy,
            seven_factor_engine=seven_factor_engine,
            stepped_leverage_controller=stepped_leverage_ctrl,
            use_seven_factor_engine=use_seven_factor,
        )
    rate_limiter = ExchangeRateLimiter()
    ws_client = ResilientWebSocketClient(symbols=symbols)

    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        router=router,
        symbols=symbols,
        fast_interval_sec=args.fast_interval,
        rebalance_interval_sec=args.rebalance_interval,
        rate_limiter=rate_limiter,
        ws_client=ws_client,
        min_volume_24h_usd=args.min_volume,
        enable_market_scan=True,
        enable_anomalies=args.enable_anomalies,
        anomaly_size=args.anomaly_size,
        profit_lock_trigger_pct=args.profit_lock_trigger,
        profit_lock_level_pct=args.profit_lock_level,
        trailing_trigger_pct=args.trailing_trigger,
        trailing_distance_pct=args.trailing_distance,
        core_take_profit_pct=args.core_tp,
        min_rebalance_notional=min(10.0, args.anomaly_size) if args.enable_anomalies else max(10.0, min(50.0, args.capital * 0.01)),
    )

    # 3. Single-step execution mode
    if args.once:
        logger.info("Executing single synchronous step on Binance Mainnet...")
        step_result = daemon.step()
        logger.info(f"Step completed: {step_result}")

        # Render monitor snapshot
        monitor = ShadowMonitor(db_path=args.db)
        layout = monitor.render_snapshot()
        if monitor.console and layout:
            monitor.console.print(layout)
        db.close()
        return

    # 4. Continuous Daemon Mode
    if args.daemon:
        logger.info("Starting continuous ShadowTradingDaemon loop (Press Ctrl+C to stop)...")

        def handle_sigint(sig, frame):
            logger.info("Termination signal received. Shutting down daemon...")
            daemon.stop()
            db.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        daemon.start()

        while daemon.is_running:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                handle_sigint(None, None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
