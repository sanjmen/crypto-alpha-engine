"""
Unified Multi-Mode CLI Launcher for Crypto-Alpha-Engine (Issue #51 / Milestone M13).
Supports running Shadow Lab (--shadow), Binance Futures Testnet (--testnet), and
Live Mainnet Production (--mainnet) simultaneously in isolated processes and SQLite databases.

Usage:
    # 1. Production Live Mainnet (uses data/live_trading.db):
    python scripts/run_trading.py --daemon --mainnet
    python scripts/run_trading.py --monitor --mainnet

    # 2. Local Shadow Lab in parallel (0 capital risk, uses data/shadow_trading.db):
    python scripts/run_trading.py --daemon --shadow
    python scripts/run_trading.py --monitor --shadow

    # 3. Binance Futures Testnet in parallel (uses data/testnet_trading.db):
    python scripts/run_trading.py --daemon --testnet
    python scripts/run_trading.py --monitor --testnet
"""

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

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
logger = logging.getLogger("trading_runner")


def resolve_mode_config(
    args: argparse.Namespace,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Resolves execution mode ('shadow', 'testnet', 'mainnet'), isolated SQLite DB path,
    and appropriate API credentials so multiple modes can run in parallel without conflict.
    """
    env = environ if environ is not None else os.environ
    explicit_mode = getattr(args, "mode", None)
    if getattr(args, "mainnet", False) or explicit_mode == "mainnet":
        mode = "mainnet"
    elif getattr(args, "testnet", False) or explicit_mode == "testnet":
        mode = "testnet"
    else:
        mode = "shadow"

    db_arg = getattr(args, "db", DEFAULT_DB_PATH)
    data_dir = Path(DEFAULT_DB_PATH).parent
    if db_arg == DEFAULT_DB_PATH:
        if mode == "mainnet":
            db_path = str(data_dir / "live_trading.db")
        elif mode == "testnet":
            db_path = str(data_dir / "testnet_trading.db")
        else:
            db_path = str(data_dir / "shadow_trading.db")
    else:
        db_path = str(db_arg)

    if mode == "mainnet":
        api_key = env.get("BINANCE_API_KEY")
        api_secret = env.get("BINANCE_API_SECRET")
    elif mode == "testnet":
        api_key = env.get("BINANCE_TESTNET_API_KEY") or env.get("BINANCE_API_KEY")
        api_secret = env.get("BINANCE_TESTNET_API_SECRET") or env.get("BINANCE_API_SECRET")
    else:
        api_key = None
        api_secret = None

    raw_cap = getattr(args, "capital", None)
    default_cap = 3000.0 if mode == "shadow" else 1000.0
    effective_capital = float(raw_cap) if raw_cap is not None else default_cap

    return {
        "mode": mode,
        "db_path": db_path,
        "api_key": api_key,
        "api_secret": api_secret,
        "is_live": mode in ("mainnet", "testnet"),
        "is_testnet": mode == "testnet",
        "capital": effective_capital,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Multi-Mode Trading Engine (Shadow / Testnet / Mainnet)"
    )
    parser.add_argument("--daemon", action="store_true", help="Run the autonomous trading daemon loop")
    parser.add_argument("--monitor", action="store_true", help="Launch the real-time Rich terminal telemetry dashboard")
    parser.add_argument("--once", action="store_true", help="Execute a single synchronous step and print status")
    parser.add_argument(
        "--mode",
        choices=["shadow", "testnet", "mainnet"],
        default=None,
        help="Execution mode: shadow (data/shadow_trading.db), testnet (data/testnet_trading.db), or mainnet (data/live_trading.db)",
    )
    parser.add_argument("--shadow", action="store_true", help="Run in local Shadow Lab mode (read-only Mainnet feeds, 0 capital risk)")
    parser.add_argument("--testnet", action="store_true", help="Run on Binance Futures Testnet (data/testnet_trading.db)")
    parser.add_argument("--mainnet", action="store_true", help="Run LIVE on Binance USDS-M Futures Mainnet (data/live_trading.db)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5-stage exchange connectivity & order lifecycle smoke test")
    parser.add_argument("--smoke-symbol", type=str, default="DOGE/USDT", help="Symbol for --smoke-test (default: DOGE/USDT)")
    parser.add_argument("--smoke-notional", type=float, default=6.0, help="Target USD notional for --smoke-test (default: 6.0)")
    parser.add_argument("--configure-only", action="store_true", help="Configure One-Way mode, Cross Margin & Leverage on exchange and exit")
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial portfolio capital in USDT (default: 3000.0 in --shadow, 1000.0 in --mainnet)",
    )
    parser.add_argument("--rebalance-now", action="store_true", help="Force an immediate 7-Factor rebalance cycle on daemon startup")
    parser.add_argument("--reset", action="store_true", help="Reset all previous database records in the selected mode DB before starting")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Override path to SQLite database")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. 'BTC/USDT,ETH/USDT')")
    parser.add_argument("--fast-interval", type=float, default=10.0, help="Fast loop interval in seconds (default: 10.0)")
    parser.add_argument("--rebalance-interval", type=float, default=14400.0, help="Rebalance interval in seconds (default: 14400 = 4h)")
    parser.add_argument("--start-leverage", type=float, default=2.5, help="Initial gross leverage tier (default: 2.5x)")
    parser.add_argument("--max-leverage", type=float, default=5.0, help="Maximum stepped gross leverage tier (default: 5.0x)")
    parser.add_argument("--step-leverage", type=float, default=0.5, help="Leverage increment per HWM milestone (default: 0.5x)")
    parser.add_argument("--milestone-gain", type=float, default=0.15, help="HWM compound gain required per +0.5x step (default: 0.15 = +15%%)")
    parser.add_argument("--dd-bracket", type=float, default=0.10, help="Drawdown bracket for SteppedLeverageController de-stepping (default: 0.10)")
    parser.add_argument(
        "--market-neutral-3x",
        action="store_true",
        help="Lever #3 preset: Start at 3.0x Market-Neutral base leverage with tighter 6%% DD de-step bracket",
    )
    parser.add_argument("--f6-min-tstat", type=float, default=0.75, help="Lever #1: Minimum t-stat required to activate F6_Liq_Surge (default: 0.75)")
    parser.add_argument("--anomaly-pct-equity", type=float, default=0.045, help="Lever #2: Max anomaly notional as fraction of total equity (default: 0.045 = 4.5%%)")
    parser.add_argument(
        "--anomaly-prefer-maker",
        dest="anomaly_prefer_maker",
        action="store_true",
        default=True,
        help="Lever #2: Execute Anomaly entries via Post-Only (GTX) Maker Limit-Chase (ENABLED by default)",
    )
    parser.add_argument(
        "--anomaly-taker-only",
        dest="anomaly_prefer_maker",
        action="store_false",
        help="Disable Maker Limit-Chase on Anomaly entries and use Market orders",
    )
    parser.add_argument("--max-single-clip-usd", type=float, default=10000.0, help="Lever #4: Max single child clip USD before automatic order chunking (default: 10000.0)")
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
    parser.add_argument("--enable-pairs", action="store_true", help="Enable Cointegrated Pairs strategy")
    parser.add_argument("--anomaly-size", type=float, default=15.0, help="Max dollar allocation per anomaly event (default: 15.0 USDT)")
    parser.add_argument("--min-volume", type=float, default=20000000.0, help="Minimum 24h volume filter in USDT for anomaly scanning (default: 20M)")
    parser.add_argument("--profit-lock-trigger", type=float, default=0.035, help="Peak gain required to arm Core Profit-Lock Stop-Profit (default: 0.035)")
    parser.add_argument("--profit-lock-level", type=float, default=0.015, help="Net profit floor locked when Core Profit-Lock triggers (default: 0.015)")
    parser.add_argument("--trailing-trigger", type=float, default=0.045, help="Peak gain required to arm Core continuous Trailing Stop (default: 0.045)")
    parser.add_argument("--trailing-distance", type=float, default=0.020, help="Distance behind peak price for Core Trailing Stop (default: 0.020)")
    parser.add_argument("--core-tp", type=float, default=0.080, help="Passive Maker Take-Profit distance for core/7-Factor legs (default: 0.080)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_mode_config(args)
    mode = cfg["mode"]
    db_path = cfg["db_path"]
    effective_capital = float(cfg["capital"])

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SHADOW_SYMBOLS.copy()
    if args.enable_anomalies and args.legacy_allocator and not args.symbols:
        for extra in ["CTSI/USDT", "MAGIC/USDT", "OGN/USDT", "RLC/USDT"]:
            if extra not in symbols:
                symbols.append(extra)

    # 1. Monitor Mode
    if args.monitor:
        logger.info(f"Launching Monitor [{mode.upper()}] connected to {db_path}...")
        monitor = ShadowMonitor(db_path=db_path)
        monitor.run_live()
        return

    # 2. Live / Testnet Broker Initialization (if applicable)
    live_broker = None
    if cfg["is_live"]:
        api_key = cfg["api_key"]
        api_secret = cfg["api_secret"]
        if not api_key or not api_secret:
            logger.error(f"API credentials required in .env or environment for --{mode} mode.")
            sys.exit(1)
        kill_switch = EmergencyKillSwitch(
            max_24h_drawdown_pct=0.12,
            max_net_exposure_pct=0.15,
            max_margin_ratio=0.50,
        )
        live_broker = BinanceLiveBroker(
            api_key=api_key,
            api_secret=api_secret,
            testnet=cfg["is_testnet"],
            dry_run=False,
            initial_capital=effective_capital,
            max_gross_leverage=max(6.0, args.max_leverage * 1.25),
            chase_interval_sec=1.5,
            max_chase_iterations=2,
            allow_taker_fallback=True,
            kill_switch=kill_switch,
        )

        if args.configure_only:
            ex_cfg = live_broker.ensure_account_configuration(
                symbols=symbols,
                target_leverage=max(3.0, args.start_leverage),
            )
            print(json.dumps(ex_cfg, indent=2))
            return

        if args.smoke_test:
            report = live_broker.run_production_smoke_test(
                symbol=args.smoke_symbol,
                notional_usd=args.smoke_notional,
                leverage=int(round(max(2.0, args.start_leverage))),
            )
            print(json.dumps(report, indent=2))
            return

        ex_cfg = live_broker.ensure_account_configuration(
            symbols=symbols,
            target_leverage=max(3.0, args.start_leverage),
        )
        logger.info(
            f"🟢 [{mode.upper()}] BINANCE FUTURES CONNECTED | Mode={ex_cfg.get('position_mode')} | "
            f"USDT Balance=${ex_cfg.get('usdt_total_balance', 0.0):,.2f}"
        )

    # 3. Database & Daemon Initialization
    logger.info(f"Initializing [{mode.upper()}] Trading Database at {db_path}...")
    db = ShadowDatabase(db_path=db_path)
    if args.reset:
        logger.info(f"Resetting database records in {db_path}...")
        db.reset_all_data()

    start_lev = 3.0 if getattr(args, "market_neutral_3x", False) and args.start_leverage == 2.5 else args.start_leverage
    dd_bracket_val = 0.06 if getattr(args, "market_neutral_3x", False) and args.dd_bracket == 0.10 else args.dd_bracket

    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=effective_capital,
        max_gross_leverage=max(6.0, args.max_leverage * 1.25),
        live_broker=live_broker,
    )
    if not cfg["is_live"]:
        broker.scale_capital(effective_capital)

    router = SmartExecutionRouter(
        broker=broker,
        enable_auto_chunking=True,
        max_single_clip_usd=args.max_single_clip_usd,
    )
    pairs_strategy = CointegratedPairsStrategy() if args.enable_pairs else None
    use_seven_factor = not args.legacy_allocator
    seven_factor_engine = (
        SevenFactorOnlineEngine(f6_min_tstat=args.f6_min_tstat)
        if use_seven_factor
        else None
    )
    stepped_leverage_ctrl = (
        SteppedLeverageController(
            initial_capital=effective_capital,
            start_leverage=start_lev,
            max_leverage=args.max_leverage,
            step_leverage=args.step_leverage,
            gain_per_step=args.milestone_gain,
            dd_bracket=dd_bracket_val,
        )
        if use_seven_factor
        else None
    )

    if use_seven_factor:
        logger.info(
            f"7-Factor Causal Online Engine ENABLED [{mode.upper()}] | Capital: ${effective_capital:,.2f} | "
            f"Stepped Leverage: {start_lev:.1f}x -> {args.max_leverage:.1f}x "
            f"(+{args.step_leverage:.1f}x per +{args.milestone_gain*100:.0f}% HWM | "
            f"F6 Gate t>={args.f6_min_tstat:.2f})"
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
        anomaly_pct_equity=args.anomaly_pct_equity,
        anomaly_prefer_maker=args.anomaly_prefer_maker,
        profit_lock_trigger_pct=args.profit_lock_trigger,
        profit_lock_level_pct=args.profit_lock_level,
        trailing_trigger_pct=args.trailing_trigger,
        trailing_distance_pct=args.trailing_distance,
        core_take_profit_pct=args.core_tp,
        min_rebalance_notional=(
            min(10.0, args.anomaly_size)
            if args.enable_anomalies
            else max(10.0, min(50.0, effective_capital * 0.01))
        ),
    )
    if getattr(args, "rebalance_now", False):
        daemon.last_rebalance_time = None

    if args.once:
        logger.info(f"Executing single synchronous step [{mode.upper()}]...")
        step_result = daemon.step()
        logger.info(f"Step completed: {step_result}")
        monitor = ShadowMonitor(db_path=db_path)
        layout = monitor.render_snapshot()
        if monitor.console and layout:
            monitor.console.print(layout)
        db.close()
        return

    if args.daemon:
        logger.info(f"Starting continuous [{mode.upper()}] Daemon loop (Press Ctrl+C to stop)...")

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
