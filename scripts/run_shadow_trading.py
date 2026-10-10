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
import signal
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.database import DEFAULT_DB_PATH, ShadowDatabase
from src.execution.execution_router import SmartExecutionRouter
from src.execution.monitor import ShadowMonitor
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.shadow_daemon import DEFAULT_SHADOW_SYMBOLS, ShadowTradingDaemon
from src.strategies.meta_allocator import MetaStrategyAllocator

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
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. 'BTC/USDT,ETH/USDT')")
    parser.add_argument("--fast-interval", type=float, default=10.0, help="Fast loop interval in seconds (default: 10.0)")
    parser.add_argument("--rebalance-interval", type=float, default=28800.0, help="Rebalance interval in seconds (default: 28800 = 8h)")

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SHADOW_SYMBOLS

    # 1. Monitor Mode
    if args.monitor:
        logger.info(f"Launching Shadow Monitor connected to {args.db}...")
        monitor = ShadowMonitor(db_path=args.db)
        monitor.run_live()
        return

    # 2. Daemon Initialization
    logger.info(f"Initializing Shadow Trading Database at {args.db}...")
    db = ShadowDatabase(db_path=args.db)
    broker = ShadowExecutionBroker(database=db, initial_cash=100_000.0)
    router = SmartExecutionRouter(broker=broker)
    allocator = MetaStrategyAllocator()

    daemon = ShadowTradingDaemon(
        broker=broker,
        allocator=allocator,
        router=router,
        symbols=symbols,
        fast_interval_sec=args.fast_interval,
        rebalance_interval_sec=args.rebalance_interval,
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
