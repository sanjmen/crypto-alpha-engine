"""
Production Live Execution Launcher for the 7-Factor Causal Online Alpha Engine
with Stepped Leverage (2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x) and Emergency Kill-Switch.

Supports 3 execution stages:
  1. --dry-run (Default): Connects to Binance Futures Mainnet data, runs full live broker logic locally.
  2. --testnet: Routes Post-Only (GTX) Limit-Chase orders to Binance Futures Testnet.
  3. --mainnet: Routes Post-Only (GTX) Limit-Chase orders to Binance Futures Mainnet (requires API keys).
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.live_broker import BinanceLiveBroker
from src.execution.shadow_daemon import DEFAULT_SHADOW_SYMBOLS, ShadowTradingDaemon
from src.execution.shadow_broker import ShadowExecutionBroker
from src.execution.database import DEFAULT_DB_PATH, ShadowDatabase
from src.risk.kill_switch import EmergencyKillSwitch
from src.risk.stepped_leverage import SteppedLeverageController
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.seven_factor_engine import SevenFactorOnlineEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("live_trading_runner")


def run_live_cycle(
    live_broker: BinanceLiveBroker,
    data_daemon: ShadowTradingDaemon,
    allocator: MetaStrategyAllocator,
    symbols: list[str],
) -> dict:
    """
    Executes one complete production cycle:
    1. Reconciles positions & cash from Binance (`GET /fapi/v2/positionRisk`)
    2. Evaluates Emergency Kill-Switch (-12% 24h DD / >15% Net Exposure Asymmetry)
    3. Warm-starts 1,000h `1h` bars + `15m` sub-hourly bars + `72h` funding
    4. Computes 7-Factor Causal Online weights + Stepped Leverage (2.5x -> 5.0x)
    5. Synchronizes Cross-Margin & per-symbol leverage on Binance
    6. Executes atomic Long/Short leg rebalance via Post-Only (GTX) Maker Limit-Chase
    """
    live_broker.reconcile_positions_from_exchange()
    portfolio = live_broker.get_portfolio()

    ks_status = live_broker.kill_switch.evaluate_and_enforce(portfolio, broker=live_broker)
    if ks_status.triggered and live_broker.kill_switch.is_locked():
        logger.critical(f"Execution halted by Kill-Switch: {ks_status.reason}")
        return {"status": "HALTED", "reason": ks_status.reason}

    market_data = {}
    bars_15m = {}
    funding_data = {}
    volatilities = {}
    prices = {}

    for sym in symbols:
        clean_sym = sym.replace("/", "")
        df_1h = data_daemon.fetch_warm_klines(sym, timeframe="1h", target_bars=1000)
        if not df_1h.empty and len(df_1h) >= 30:
            market_data[clean_sym] = df_1h
            prices[sym] = float(df_1h["close"].iloc[-1])
            prices[clean_sym] = prices[sym]
            volatilities[clean_sym] = 0.02
            df_15m = data_daemon.fetch_warm_klines(sym, timeframe="15m", target_bars=24)
            if not df_15m.empty:
                bars_15m[clean_sym] = df_15m
            df_f = data_daemon.fetch_funding_history_72h(sym)
            if not df_f.empty:
                funding_data[clean_sym] = df_f

    target_dollars, regime, factor_weights = allocator.generate_portfolio_allocations(
        market_data=market_data,
        portfolio=portfolio,
        volatilities=volatilities,
        funding_data=funding_data,
        bars_15m=bars_15m,
    )

    lev_snap = allocator.last_leverage_snapshot
    eff_lev = lev_snap.effective_leverage if lev_snap else 2.5

    live_broker.kill_switch.sync_margin_and_leverage(
        exchange_client=live_broker.client,
        symbols=symbols,
        effective_leverage=eff_lev,
        dry_run=live_broker.dry_run,
    )

    # Map clean symbols back to slash symbols for execution
    sym_map = {s.replace("/", ""): s for s in symbols}
    mapped_targets = {sym_map.get(k, k): v for k, v in target_dollars.items()}

    orders = live_broker.execute_atomic_rebalance(
        target_dollars=mapped_targets,
        prices=prices,
        min_notional=max(15.0, portfolio.total_equity * 0.015),
    )

    return {
        "status": "OK",
        "regime": regime.value if hasattr(regime, "value") else str(regime),
        "effective_leverage": eff_lev,
        "unlocked_tier": lev_snap.unlocked_tier if lev_snap else 2.5,
        "orders_executed": len(orders),
        "maker_fill_ratio": live_broker.maker_fill_ratio,
        "factor_weights": factor_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="7-Factor Live Production Trading Engine (Binance Futures)")
    parser.add_argument("--mainnet", action="store_true", help="Enable LIVE Mainnet order execution")
    parser.add_argument("--testnet", action="store_true", help="Enable Binance Futures Testnet execution")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Run in safe dry-run mode (default)")
    parser.add_argument("--once", action="store_true", help="Execute a single rebalance cycle and exit")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital in USDT (default: 1000.0)")
    parser.add_argument("--start-leverage", type=float, default=2.5, help="Initial leverage tier (default: 2.5x)")
    parser.add_argument("--max-leverage", type=float, default=5.0, help="Max stepped leverage tier (default: 5.0x)")
    parser.add_argument("--rebalance-interval", type=float, default=14400.0, help="Rebalance interval in seconds (default: 14400 = 4h)")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to SQLite state DB")
    args = parser.parse_args()

    is_dry_run = not (args.mainnet or args.testnet)
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")

    if args.mainnet and (not api_key or not api_secret):
        logger.error("BINANCE_API_KEY and BINANCE_API_SECRET environment variables are required for --mainnet.")
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
        dry_run=is_dry_run,
        initial_capital=args.capital,
        max_gross_leverage=max(6.0, args.max_leverage * 1.25),
        kill_switch=kill_switch,
    )

    db = ShadowDatabase(db_path=args.db)
    shadow_helper = ShadowExecutionBroker(database=db, initial_cash=args.capital)
    lev_ctrl = SteppedLeverageController(
        initial_capital=args.capital,
        start_leverage=args.start_leverage,
        max_leverage=args.max_leverage,
        step_leverage=0.5,
        gain_per_step=0.15,
        dd_bracket=0.10,
    )
    allocator = MetaStrategyAllocator(
        seven_factor_engine=SevenFactorOnlineEngine(),
        stepped_leverage_controller=lev_ctrl,
        use_seven_factor_engine=True,
    )
    data_daemon = ShadowTradingDaemon(
        broker=shadow_helper,
        allocator=allocator,
        symbols=DEFAULT_SHADOW_SYMBOLS.copy(),
    )

    mode_label = "MAINNET LIVE" if args.mainnet else ("TESTNET" if args.testnet else "DRY-RUN")
    logger.info(f"🚀 Starting 7-Factor Production Engine [{mode_label}] | Tier: {args.start_leverage:.1f}x -> {args.max_leverage:.1f}x")

    if args.once:
        res = run_live_cycle(live_broker, data_daemon, allocator, DEFAULT_SHADOW_SYMBOLS.copy())
        logger.info(f"Live Cycle Result: {res}")
        db.close()
        return

    while True:
        try:
            res = run_live_cycle(live_broker, data_daemon, allocator, DEFAULT_SHADOW_SYMBOLS.copy())
            logger.info(f"Live Cycle Result: {res}")
            time.sleep(args.rebalance_interval)
        except KeyboardInterrupt:
            logger.info("Shutting down Live Trading Engine...")
            db.close()
            break


if __name__ == "__main__":
    main()
