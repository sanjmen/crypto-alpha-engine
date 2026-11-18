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
from dataclasses import asdict, dataclass
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
from src.execution.feature_flags import FeatureFlags
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
    parser = argparse.ArgumentParser(description="Unified Multi-Mode Trading Engine (Shadow / Testnet / Mainnet)")
    parser.add_argument("--daemon", action="store_true", help="Run the autonomous trading daemon loop")
    parser.add_argument("--monitor", action="store_true", help="Launch the real-time Rich terminal telemetry dashboard")
    parser.add_argument("--once", action="store_true", help="Execute a single synchronous step and print status")
    parser.add_argument(
        "--mode",
        choices=["shadow", "testnet", "mainnet"],
        default=None,
        help="Execution mode: shadow (data/shadow_trading.db), testnet (data/testnet_trading.db), or mainnet (data/live_trading.db)",
    )
    parser.add_argument(
        "--shadow", action="store_true", help="Run in local Shadow Lab mode (read-only Mainnet feeds, 0 capital risk)"
    )
    parser.add_argument(
        "--testnet", action="store_true", help="Run on Binance Futures Testnet (data/testnet_trading.db)"
    )
    parser.add_argument(
        "--mainnet", action="store_true", help="Run LIVE on Binance USDS-M Futures Mainnet (data/live_trading.db)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 5-stage exchange connectivity & order lifecycle smoke test"
    )
    parser.add_argument(
        "--smoke-symbol", type=str, default="DOGE/USDT", help="Symbol for --smoke-test (default: DOGE/USDT)"
    )
    parser.add_argument(
        "--smoke-notional", type=float, default=6.0, help="Target USD notional for --smoke-test (default: 6.0)"
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Configure One-Way mode, Cross Margin & Leverage on exchange and exit",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial portfolio capital in USDT (default: 3000.0 in --shadow, 1000.0 in --mainnet)",
    )
    parser.add_argument(
        "--rebalance-now", action="store_true", help="Force an immediate 7-Factor rebalance cycle on daemon startup"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all previous database records in the selected mode DB before starting",
    )
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Override path to SQLite database")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. 'BTC/USDT,ETH/USDT')")
    parser.add_argument(
        "--fast-interval", type=float, default=10.0, help="Fast loop interval in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--rebalance-interval",
        type=float,
        default=28800.0,
        help="Rebalance interval in seconds (default: 28800 = 8h funding-aligned cadence)",
    )
    parser.add_argument(
        "--align-funding-clock",
        dest="align_funding_clock",
        action="store_true",
        default=True,
        help="Align scheduled rebalances to Binance 8h funding boundaries (00:00, 08:00, 16:00 UTC)",
    )
    parser.add_argument(
        "--no-align-funding-clock",
        dest="align_funding_clock",
        action="store_false",
        help="Use elapsed-seconds rebalance interval instead of UTC funding clock alignment",
    )
    parser.add_argument(
        "--bhs-enter-top-k",
        type=int,
        default=3,
        help="Novy-Marx & Velikov Buy/Hold Spread: top rank required to open a new leg (default: 3)",
    )
    parser.add_argument(
        "--bhs-hold-top-k",
        type=int,
        default=6,
        help="Novy-Marx & Velikov Buy/Hold Spread: rank buffer to retain an open leg without cliff-eviction (default: 6)",
    )
    parser.add_argument(
        "--deadband-mult",
        type=float,
        default=1.25,
        help="Multiplier on proportional turnover deadband (default: 1.25)",
    )
    parser.add_argument(
        "--recovery-boost-equity",
        type=float,
        default=3050.0,
        help="Two-Phase Safe Ramp: HWM equity threshold to unlock post-recovery base leverage (0 disables, default: 3050.0)",
    )
    parser.add_argument(
        "--recovery-boost-leverage",
        type=float,
        default=3.2,
        help="Two-Phase Safe Ramp: base leverage unlocked once HWM >= recovery-boost-equity (default: 3.2x)",
    )
    parser.add_argument(
        "--enable-btc-lead-lag",
        action="store_true",
        default=False,
        help="Enable optional Makarov & Schoar 6h BTC -> Alt lead-lag catch-up tilt inside 7-Factor engine",
    )
    parser.add_argument("--start-leverage", type=float, default=2.5, help="Initial gross leverage tier (default: 2.5x)")
    parser.add_argument(
        "--max-leverage", type=float, default=5.0, help="Maximum stepped gross leverage tier (default: 5.0x)"
    )
    parser.add_argument(
        "--step-leverage", type=float, default=0.5, help="Leverage increment per HWM milestone (default: 0.5x)"
    )
    parser.add_argument(
        "--milestone-gain",
        type=float,
        default=0.15,
        help="HWM compound gain required per +0.5x step (default: 0.15 = +15%%)",
    )
    parser.add_argument(
        "--dd-bracket",
        type=float,
        default=0.10,
        help="Drawdown bracket for SteppedLeverageController de-stepping (default: 0.10)",
    )
    parser.add_argument(
        "--market-neutral-3x",
        action="store_true",
        help="Lever #3 preset: Start at 3.0x Market-Neutral base leverage with tighter 6%% DD de-step bracket",
    )
    parser.add_argument(
        "--enable-experimental-satellites",
        "--enable-m13",
        dest="enable_experimental_satellites",
        action="store_true",
        help="Explicitly enable quarantined experimental satellite extensions (Harvester & 4.5%% Anomaly Sizing) in --mainnet/--testnet",
    )
    parser.add_argument(
        "--legacy-baseline-priors",
        action="store_true",
        help="Revert to legacy static BASELINE_20D_PRIORS instead of Dynamic Online IC Softmax + Liquidity Gate",
    )
    parser.add_argument(
        "--f6-min-tstat",
        type=float,
        default=0.75,
        help="Lever #1: Minimum t-stat required to activate F6_Liq_Surge (default: 0.75)",
    )
    parser.add_argument(
        "--anomaly-pct-equity",
        type=float,
        default=0.045,
        help="Lever #2: Max anomaly notional as fraction of total equity (default: 0.045 = 4.5%%)",
    )
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
    parser.add_argument(
        "--max-single-clip-usd",
        type=float,
        default=10000.0,
        help="Lever #4: Max single child clip USD before automatic order chunking (default: 10000.0)",
    )
    parser.add_argument(
        "--legacy-allocator",
        action="store_true",
        help="Use legacy 3-strategy allocator instead of the 7-Factor Causal Online Engine",
    )
    parser.add_argument(
        "--enable-anomalies",
        dest="enable_anomalies",
        action="store_true",
        default=None,
        help="Enable extreme anomaly & illiquid tail strategies (default: OFF in --mainnet, ON in --shadow)",
    )
    parser.add_argument(
        "--disable-anomalies",
        dest="enable_anomalies",
        action="store_false",
        help="Disable extreme anomaly satellite strategies",
    )
    parser.add_argument(
        "--equity-floor-usd",
        type=float,
        default=None,
        help="Regla #2: Hard capital floor in USDT that flattens all positions and halts trading (default: 2500.0 in --mainnet)",
    )
    parser.add_argument(
        "--cushion-buffer-usd",
        type=float,
        default=None,
        help="Phase 1 CPPI Cushion-Safe smooth de-risking buffer above equity_floor_usd (default: 240.0 USDT)",
    )
    parser.add_argument(
        "--enable-dollar-bar-tsmom",
        dest="enable_dollar_bar_tsmom",
        action="store_true",
        default=None,
        help="Enable Manga P2 Causal 1m Dollar-Bar Multi-Horizon TSMOM + Taker-CVD (default: True)",
    )
    parser.add_argument(
        "--disable-dollar-bar-tsmom",
        dest="enable_dollar_bar_tsmom",
        action="store_false",
        help="Disable Manga P2 Causal 1m Dollar-Bar TSMOM sleeve",
    )
    parser.add_argument(
        "--dollar-bar-max-gross",
        type=float,
        default=None,
        help="Maximum gross leverage for Manga P2 Dollar-Bar TSMOM sleeve (default: 0.95)",
    )
    parser.add_argument(
        "--core-p1-risk-scale",
        type=float,
        default=None,
        help="Risk budget scale on Core P1 when Manga P2 / Snipers are active (default: 0.85)",
    )
    parser.add_argument(
        "--enable-event-snipers",
        dest="enable_event_snipers",
        action="store_true",
        default=None,
        help="Enable Phase 2 Event-Driven Microstructure Snipers M09, M23 & M04 (default: True)",
    )
    parser.add_argument(
        "--disable-event-snipers",
        dest="enable_event_snipers",
        action="store_false",
        help="Disable Phase 2 Event-Driven Microstructure Snipers",
    )
    parser.add_argument(
        "--sniper-slot-leverage",
        type=float,
        default=None,
        help="Base equity fraction per Phase 2 Sniper slot before Cushion/SafeRamp scaling (default: 0.24)",
    )
    parser.add_argument(
        "--enable-yield-collateral",
        action="store_true",
        default=False,
        help="Opt-in passive BFUSD/LDUSDT margin yield simulation (default: False — 100%% USDT Single-Asset Cross-Margin)",
    )
    parser.add_argument(
        "--sniper-max-positions",
        type=int,
        default=None,
        help="Maximum concurrent Phase 2 Event Sniper positions (default: 3)",
    )
    parser.add_argument(
        "--max-24h-dd",
        type=float,
        default=0.065,
        help="Emergency 24h drawdown circuit breaker fraction (default: 0.065 = 6.5%%)",
    )
    parser.add_argument(
        "--show-flags",
        action="store_true",
        help="Print active runtime Feature Flags and 3 Reglas de Blindaje invariants from SQLite and exit",
    )
    parser.add_argument(
        "--set-flag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Live-toggle a runtime feature flag in SQLite without restarting daemon (e.g. --set-flag enable_anomaly_capsule=false)",
    )
    parser.add_argument(
        "--audit-readiness",
        action="store_true",
        help="Audit the selected mode SQLite database against institutional Go/No-Go readiness checkgates and exit",
    )
    parser.add_argument(
        "--audit-json",
        action="store_true",
        help="Emit --audit-readiness scorecard as machine-readable JSON",
    )
    parser.add_argument("--enable-pairs", action="store_true", help="Enable Cointegrated Pairs strategy")
    parser.add_argument(
        "--anomaly-size", type=float, default=15.0, help="Max dollar allocation per anomaly event (default: 15.0 USDT)"
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=20000000.0,
        help="Minimum 24h volume filter in USDT for anomaly scanning (default: 20M)",
    )
    parser.add_argument(
        "--profit-lock-trigger",
        type=float,
        default=0.035,
        help="Peak gain required to arm Core Profit-Lock Stop-Profit (default: 0.035)",
    )
    parser.add_argument(
        "--profit-lock-level",
        type=float,
        default=0.015,
        help="Net profit floor locked when Core Profit-Lock triggers (default: 0.015)",
    )
    parser.add_argument(
        "--trailing-trigger",
        type=float,
        default=0.0,
        help="Peak gain required to arm Core continuous Trailing Stop (default: 0.0 [disabled])",
    )
    parser.add_argument(
        "--trailing-distance",
        type=float,
        default=0.020,
        help="Distance behind peak price for Core Trailing Stop (default: 0.020)",
    )
    parser.add_argument(
        "--core-tp",
        type=float,
        default=0.050,
        help="Passive Maker Take-Profit distance for core/7-Factor legs (default: 0.050 = +5.0%%)",
    )
    parser.add_argument(
        "--core-sl",
        type=float,
        default=None,
        help="Initial protective Stop-Loss distance for core/7-Factor legs (default: None [disabled])",
    )
    return parser


@dataclass
class CheckgateResult:
    name: str
    passed: bool
    observed: str
    threshold: str
    details: str


def evaluate_shadow_readiness(db_path: str = DEFAULT_DB_PATH) -> dict[str, Any]:
    """
    Evaluates all quantitative Go/No-Go checkgates from `db_path`:
    1. Dollar Neutrality Check: |Long_USD - Short_USD| / Equity <= 5.0%
    2. BTC Beta Neutrality Check: |Portfolio_Beta| <= 0.20
    3. Maker Execution Efficiency: Maker fill ratio >= 80%
    4. Drawdown & Stepped Leverage Hygiene: Max drawdown < 12.0% & Tier <= 5.0x
    5. 7-Factor Rebalance & Telemetry Active
    """
    if not os.path.exists(db_path):
        return {
            "ready_for_live": False,
            "reason": f"Database not found at {db_path}",
            "gates": [],
        }

    db = ShadowDatabase(db_path=db_path)
    latest_acc = db.get_latest_account()
    acc_hist = db.get_account_history(limit=1000)
    positions = db.get_open_positions()
    trades = db.get_trades(limit=500)
    sf_tel = db.get_latest_seven_factor_telemetry()
    rebalances = db.get_latest_rebalance_snapshot()
    db.close()

    if not latest_acc:
        return {
            "ready_for_live": False,
            "reason": "No account snapshots recorded yet.",
            "gates": [],
        }

    equity = max(float(latest_acc["total_equity"]), 1.0)
    init_cap = max(float(latest_acc["initial_capital"]), 1.0)

    # 1. Dollar Neutrality Check
    long_usd = sum(
        p.quantity * (p.current_price or p.entry_price) for p in positions.values() if p.side.value == "LONG"
    )
    short_usd = sum(
        p.quantity * (p.current_price or p.entry_price) for p in positions.values() if p.side.value == "SHORT"
    )
    net_usd = long_usd - short_usd
    net_pct = abs(net_usd) / equity * 100.0
    gate_neutral = CheckgateResult(
        name="1. Dollar Neutrality (|Long - Short| / Equity)",
        passed=(net_pct <= 5.0),
        observed=f"{net_pct:.2f}% (${net_usd:+,.2f})",
        threshold="<= 5.00%",
        details=f"Long: ${long_usd:,.2f} | Short: ${short_usd:,.2f}",
    )

    # 2. BTC Beta Neutrality Check
    port_beta = float(sf_tel.get("portfolio_beta", 0.0)) if sf_tel else 0.0
    gate_beta = CheckgateResult(
        name="2. BTC Beta Neutrality (72h Rolling OLS)",
        passed=(abs(port_beta) <= 0.20),
        observed=f"{port_beta:+.3f}",
        threshold="|Beta| <= 0.200",
        details="7-Factor Idiosyncratic Residual Neutralization active" if sf_tel else "Awaiting 7-Factor snapshot",
    )

    # 3. Maker Fill Ratio & Fee Drag Hygiene
    total_trades = len(trades)
    maker_trades = sum(1 for t in trades if int(t.get("is_maker", 0)) == 1)
    maker_ratio_pct = (maker_trades / total_trades * 100.0) if total_trades > 0 else 100.0
    total_fees_usd = sum(float(t.get("fee_paid", 0.0)) for t in trades)
    fee_pct_of_cap = (total_fees_usd / init_cap) * 100.0

    gate_maker = CheckgateResult(
        name="3. Execution Alpha (Maker Fill Ratio)",
        passed=(maker_ratio_pct >= 80.0),
        observed=f"{maker_ratio_pct:.1f}% ({maker_trades}/{total_trades} trades)",
        threshold=">= 80.0%",
        details=f"Cumulative Fees Paid: ${total_fees_usd:,.2f} ({fee_pct_of_cap:.2f}% of initial cap)",
    )

    # 4. Drawdown & Stepped Leverage Integrity
    equities = [float(a["total_equity"]) for a in acc_hist] if acc_hist else [equity]
    peak = init_cap
    max_dd_pct = 0.0
    for eq_val in equities:
        if eq_val > peak:
            peak = eq_val
        dd = (peak - eq_val) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

    u_tier = float(sf_tel.get("unlocked_tier", 2.5)) if sf_tel else 2.5
    eff_lev = float(sf_tel.get("effective_leverage", 2.5)) if sf_tel else float(latest_acc.get("leverage", 0.0))
    gate_dd = CheckgateResult(
        name="4. Drawdown & Stepped Leverage Integrity",
        passed=(max_dd_pct < 12.0 and u_tier <= 5.0),
        observed=f"MaxDD: {max_dd_pct:.2f}% | Tier: {u_tier:.1f}x (Eff: {eff_lev:.2f}x)",
        threshold="MaxDD < 12.0% & Tier in [2.5x, 5.0x]",
        details=f"HWM: ${peak:,.2f} | Current Equity: ${equity:,.2f}",
    )

    # 5. Rebalance Cycle & Telemetry Completeness
    has_rebalance = rebalances is not None and sf_tel is not None
    gate_cycle = CheckgateResult(
        name="5. 7-Factor Rebalance & Telemetry Active",
        passed=has_rebalance,
        observed="ACTIVE" if has_rebalance else "PENDING",
        threshold="ACTIVE",
        details=(
            f"Last Regime: {rebalances['regime_detected']} | Factors: {len(sf_tel.get('factor_weights', {}))}/7"
            if has_rebalance
            else "Run at least 1 rebalance cycle"
        ),
    )

    gates: list[CheckgateResult] = [gate_neutral, gate_beta, gate_maker, gate_dd, gate_cycle]
    all_passed = all(g.passed for g in gates)

    return {
        "ready_for_live": all_passed,
        "equity": equity,
        "initial_capital": init_cap,
        "return_pct": (equity / init_cap - 1.0) * 100.0,
        "max_dd_pct": max_dd_pct,
        "unlocked_tier": u_tier,
        "effective_leverage": eff_lev,
        "gates": gates,
    }


def print_readiness_report(db_path: str = DEFAULT_DB_PATH, as_json: bool = False) -> int:
    """Prints formatted Go/No-Go readiness scorecard or JSON and returns exit code."""
    res = evaluate_shadow_readiness(db_path=db_path)
    if as_json:
        serializable = dict(res)
        serializable["gates"] = [asdict(g) for g in res.get("gates", [])]
        print(json.dumps(serializable, indent=2))
        return 0 if res.get("ready_for_live") else 1

    print("=" * 110)
    print(" SHADOW-TO-LIVE GO/NO-GO READINESS SCORECARD (7-FACTOR CAUSAL ENGINE | 2.5x -> 5.0x)")
    print("=" * 110)

    if not res.get("gates"):
        print(f" [NO-GO] {res.get('reason')}")
        return 1

    for g in res["gates"]:
        status = "✅ PASS" if g.passed else "❌ FAIL"
        print(f" {status} | {g.name:<44} | Obs: {g.observed:<28} | Req: {g.threshold}")
        print(f"         └─ {g.details}")

    print("-" * 110)
    verdict = "🟢 GO FOR MAINNET TIER 1 (2.5x)" if res["ready_for_live"] else "🟡 HOLD IN SHADOW TRADING"
    print(
        f" VERDICT: {verdict} | Equity: ${res['equity']:,.2f} ({res['return_pct']:+.2f}%) | "
        f"MaxDD: {res['max_dd_pct']:.2f}% | Active Tier: {res['unlocked_tier']:.1f}x"
    )
    print("=" * 110)
    return 0 if res.get("ready_for_live") else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_mode_config(args)
    mode = cfg["mode"]
    db_path = cfg["db_path"]
    effective_capital = float(cfg["capital"])

    enable_experimental_satellites = (
        (mode == "shadow")
        or getattr(args, "enable_experimental_satellites", False)
        or getattr(args, "enable_m13", False)
    )
    enable_dynamic_factor_ic = enable_experimental_satellites or not getattr(args, "legacy_baseline_priors", False)
    feature_flags = FeatureFlags.from_env_and_args(
        args=args,
        mode=mode,
        enable_institutional_data_feeds=enable_dynamic_factor_ic,
    )

    if getattr(args, "show_flags", False) or getattr(args, "set_flag", None):
        db_flag = ShadowDatabase(db_path=db_path)
        stored = db_flag.get_feature_flags()
        active_ff = FeatureFlags.from_dict(stored, fallback=feature_flags)
        if getattr(args, "set_flag", None):
            for raw_item in args.set_flag:
                if "=" not in raw_item:
                    logger.error(f"Invalid --set-flag format '{raw_item}', expected KEY=VALUE")
                    db_flag.close()
                    sys.exit(1)
                k_raw, v_raw = raw_item.split("=", 1)
                c_key, p_val = active_ff.apply_override(k_raw, v_raw)
                logger.info(f"🚩 Updated Feature Flag: {c_key} = {p_val}")
            db_flag.save_feature_flags(active_ff.to_dict())
        invariants_tel = db_flag.get_latest_shield_invariants_telemetry()
        print(
            json.dumps(
                {
                    "mode": mode,
                    "db_path": db_path,
                    "feature_flags": active_ff.to_dict(),
                    "shield_invariants": invariants_tel,
                },
                indent=2,
            )
        )
        db_flag.close()
        return

    if getattr(args, "audit_readiness", False):
        exit_code = print_readiness_report(
            db_path=db_path,
            as_json=bool(getattr(args, "audit_json", False)),
        )
        sys.exit(exit_code)

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SHADOW_SYMBOLS.copy()
    if feature_flags.enable_anomaly_capsule and args.legacy_allocator and not args.symbols:
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
            max_24h_drawdown_pct=float(getattr(args, "max_24h_dd", 0.065) or 0.065),
            max_net_exposure_pct=0.15,
            max_margin_ratio=0.50,
            equity_floor_usd=(feature_flags.equity_floor_usd if feature_flags.enable_hard_equity_floor else 0.0),
        )
        live_broker = BinanceLiveBroker(
            api_key=api_key,
            api_secret=api_secret,
            testnet=cfg["is_testnet"],
            dry_run=False,
            initial_capital=effective_capital,
            max_gross_leverage=max(6.0, args.max_leverage * 1.25),
            chase_interval_sec=3.0,
            max_chase_iterations=4,
            allow_taker_fallback=True,
            kill_switch=kill_switch,
        )

        if args.configure_only:
            ex_cfg = live_broker.ensure_account_configuration(
                symbols=symbols,
                target_leverage=max(5.0, args.max_leverage, args.start_leverage),
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
            target_leverage=max(5.0, args.max_leverage, args.start_leverage),
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

    # Selective Core & Experimental Satellites Promotion (Issues #61, #62, #63, #64, #65):
    # - Proven Core (Dynamic Online IC Softmax, F6 Gate t>=0.75, F8 Smart Money Flow,
    #   Quarterly Delivery Basis Scalar, 100% USDT Single-Asset Margin) is enabled in ALL modes (--mainnet & --shadow).
    # - Quarantined Satellite Extensions (Extreme Funding Harvester [Issue #64] & 4.5% Dynamic
    #   Anomaly Sizing [Issue #65]) remain restricted to --shadow (or explicit --enable-experimental-satellites).
    enable_experimental_satellites = (mode == "shadow") or getattr(args, "enable_experimental_satellites", False)
    enable_dynamic_factor_ic = enable_experimental_satellites or not getattr(args, "legacy_baseline_priors", False)
    use_mn_3x = getattr(args, "market_neutral_3x", False) or enable_experimental_satellites
    start_lev = 3.0 if use_mn_3x and args.start_leverage == 2.5 else args.start_leverage
    dd_bracket_val = 0.06 if use_mn_3x and args.dd_bracket == 0.10 else args.dd_bracket

    BASELINE_20D_PRIORS = {
        "macro": 0.28,
        "med": 0.16,
        "fast": 0.16,
        "rev": 0.14,
        "carry": 0.15,
        "liq": 0.11,
        "size": 0.18,
    }
    priors_cfg = None if enable_dynamic_factor_ic else BASELINE_20D_PRIORS
    f6_gate_cfg = args.f6_min_tstat if enable_dynamic_factor_ic else None

    broker = ShadowExecutionBroker(
        database=db,
        initial_cash=effective_capital,
        max_gross_leverage=max(6.0, args.max_leverage * 1.25),
        live_broker=live_broker,
        enable_yield_collateral=bool(getattr(args, "enable_yield_collateral", False)),
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
        SevenFactorOnlineEngine(
            priors=priors_cfg,
            f6_min_tstat=f6_gate_cfg,
            f5_min_tstat=1.25 if feature_flags.enable_chop_range_arb else None,
            enable_chop_range_arb=feature_flags.enable_chop_range_arb,
            enable_holistic_regime_shields=feature_flags.enable_holistic_regime_shields,
            enable_btc_lead_lag=getattr(args, "enable_btc_lead_lag", False),
        )
        if use_seven_factor
        else None
    )
    eff_rec_eq = getattr(args, "recovery_boost_equity", 0.0)
    eff_rec_lev = getattr(args, "recovery_boost_leverage", 0.0)
    use_cushion_gov = bool(
        feature_flags.enable_cushion_governor
        and feature_flags.enable_hard_equity_floor
        and feature_flags.equity_floor_usd > 0.0
    )
    stepped_leverage_ctrl = (
        SteppedLeverageController(
            initial_capital=effective_capital,
            start_leverage=start_lev,
            max_leverage=args.max_leverage,
            step_leverage=args.step_leverage,
            gain_per_step=args.milestone_gain,
            dd_bracket=dd_bracket_val,
            base_deadband_ratio=(0.04 / 2.2) * max(0.5, float(getattr(args, "deadband_mult", 1.25))),
            bhs_enter_top_k=getattr(args, "bhs_enter_top_k", 3),
            bhs_hold_top_k=getattr(args, "bhs_hold_top_k", 6),
            recovery_boost_equity=eff_rec_eq if (eff_rec_eq and eff_rec_eq > 0) else None,
            recovery_boost_leverage=(
                eff_rec_lev if (eff_rec_eq and eff_rec_eq > 0 and eff_rec_lev and eff_rec_lev > start_lev) else None
            ),
            cushion_floor_usd=feature_flags.equity_floor_usd if use_cushion_gov else None,
            cushion_buffer_usd=feature_flags.cushion_buffer_usd,
        )
        if use_seven_factor
        else None
    )

    if use_seven_factor:
        if enable_experimental_satellites:
            suite_tag = "Experimental Full Suite"
        elif enable_dynamic_factor_ic:
            suite_tag = "Dynamic Online IC + 8h Funding-Clock + Novy-Marx BHS(3,6)"
        else:
            suite_tag = "Baseline 20d + 100% Maker Priority"
        f6_tag = f"t>={f6_gate_cfg:.2f}" if f6_gate_cfg is not None else "Baseline"
        ramp_tag = (
            f" | SafeRamp: {start_lev:.1f}x (<${eff_rec_eq:,.0f}) -> {eff_rec_lev:.1f}x"
            if (eff_rec_eq and eff_rec_eq > 0 and eff_rec_lev and eff_rec_lev > start_lev)
            else ""
        )
        logger.info(
            f"7-Factor Causal Online Engine ENABLED [{mode.upper()} | {suite_tag}] | Capital: ${effective_capital:,.2f} | "
            f"Stepped Leverage: {start_lev:.1f}x -> {args.max_leverage:.1f}x{ramp_tag} "
            f"(+{args.step_leverage:.1f}x per +{args.milestone_gain * 100:.0f}% HWM | "
            f"BHS({getattr(args, 'bhs_enter_top_k', 3)},{getattr(args, 'bhs_hold_top_k', 6)}) | "
            f"DBx{getattr(args, 'deadband_mult', 1.25):.2f} | F6 Gate: {f6_tag})"
        )

    from src.strategies.extreme_anomaly import (
        create_extreme_funding_harvester,
        create_illiquid_early_breakout,
        create_illiquid_funding_fade,
        create_illiquid_oversold_bounce,
    )

    is_small_test = args.anomaly_size <= 25.0 and effective_capital <= 1000.0
    bounce_strat = create_illiquid_oversold_bounce(test_mode=is_small_test)
    bounce_strat.symbols_filter = None
    bounce_strat.max_position_usd = args.anomaly_size
    breakout_strat = create_illiquid_early_breakout(test_mode=is_small_test)
    breakout_strat.symbols_filter = None
    breakout_strat.max_position_usd = args.anomaly_size
    fade_strat = create_illiquid_funding_fade(test_mode=is_small_test)
    fade_strat.symbols_filter = None
    fade_strat.max_position_usd = args.anomaly_size

    # Issue #64: Quarantine Extreme Funding Harvester from Mainnet until recalibrated
    harvester_strat = None
    if enable_experimental_satellites:
        harvester_strat = create_extreme_funding_harvester(test_mode=is_small_test)
        harvester_strat.symbols_filter = None
        harvester_strat.max_position_usd = args.anomaly_size

    from src.strategies.dollar_bar_tsmom import CausalDollarBarTSMOMEngine

    dollar_bar_eng = (
        CausalDollarBarTSMOMEngine(max_gross_leverage=feature_flags.dollar_bar_max_gross)
        if (use_seven_factor and feature_flags.enable_dollar_bar_tsmom)
        else None
    )

    allocator = MetaStrategyAllocator(
        pairs_strategy=pairs_strategy,
        oversold_bounce_strategy=bounce_strat,
        early_breakout_strategy=breakout_strat,
        funding_fade_strategy=fade_strat,
        funding_harvester_strategy=harvester_strat,
        seven_factor_engine=seven_factor_engine,
        stepped_leverage_controller=stepped_leverage_ctrl,
        use_seven_factor_engine=use_seven_factor,
        dollar_bar_engine=dollar_bar_eng,
        enable_dollar_bar_tsmom=bool(use_seven_factor and feature_flags.enable_dollar_bar_tsmom),
        core_p1_risk_scale=feature_flags.core_p1_risk_scale,
    )

    rate_limiter = ExchangeRateLimiter()
    ws_client = ResilientWebSocketClient(symbols=symbols)

    # Unified Core bracket profile across both Shadow Lab (--shadow) and Production (--mainnet):
    # TP +5.0% Maker + Profit-Lock (+3.5% -> +1.5%), Trailing Stop disabled (0.0), initial negative SL disabled (None)
    eff_core_tp = args.core_tp if args.core_tp is not None else 0.050
    eff_trailing_trigger = args.trailing_trigger if args.trailing_trigger is not None else 0.0
    eff_core_sl = args.core_sl if (args.core_sl is not None and args.core_sl > 0) else None
    logger.info(
        f"🛡️ Core Bracket Profile [{mode.upper()}]: TP={eff_core_tp * 100:.1f}% (Maker), "
        f"Profit-Lock={args.profit_lock_trigger * 100:.1f}%->{args.profit_lock_level * 100:.1f}%, "
        f"Trailing={'DISABLED' if not eff_trailing_trigger or eff_trailing_trigger <= 0 else f'{eff_trailing_trigger * 100:.1f}%/{args.trailing_distance * 100:.1f}%'}, "
        f"Initial SL={'DISABLED' if not eff_core_sl or eff_core_sl <= 0 else f'-{eff_core_sl * 100:.1f}%'}"
    )
    logger.info(
        f"🚩 Feature Flags & 3 Reglas de Blindaje [{mode.upper()}]: "
        f"AnomalyCapsule={'ENABLED' if feature_flags.enable_anomaly_capsule else 'DISABLED (100% Pure Core 7-Factor)'} | "
        f"ChopRangeArb={'ENABLED (TP=' + format(feature_flags.chop_take_profit_pct * 100, '.1f') + '%, Lock=' + format(feature_flags.chop_profit_lock_trigger_pct * 100, '.1f') + '%->' + format(feature_flags.chop_profit_lock_level_pct * 100, '.1f') + '%)' if feature_flags.enable_chop_range_arb else 'DISABLED'} | "
        f"SubhourlyPB={'ENABLED' if feature_flags.enable_subhourly_pullback else 'DISABLED'} | "
        f"DollarBarP2={'ENABLED (Max=' + format(feature_flags.dollar_bar_max_gross, '.2f') + 'x, P1Scale=' + format(feature_flags.core_p1_risk_scale, '.2f') + 'x)' if feature_flags.enable_dollar_bar_tsmom else 'DISABLED'} | "
        f"EventSnipers={'ENABLED (M09/M23/M04, Max=' + str(feature_flags.sniper_max_positions) + 'x' + format(feature_flags.sniper_slot_leverage, '.2f') + ')' if feature_flags.enable_event_snipers else 'DISABLED'} | "
        f"CushionGov={'ENABLED (Buf=$' + format(feature_flags.cushion_buffer_usd, '.0f') + ')' if use_cushion_gov else 'DISABLED'} | "
        f"HolisticShields={'ENABLED' if feature_flags.enable_holistic_regime_shields else 'DISABLED'} | "
        f"HardEquityFloor={'$' + format(feature_flags.equity_floor_usd, ',.2f') if feature_flags.enable_hard_equity_floor else 'DISABLED'} | "
        f"InvariantAudit={'ENABLED' if feature_flags.enable_invariant_audit else 'DISABLED'}"
    )

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
        enable_anomalies=feature_flags.enable_anomaly_capsule,
        anomaly_size=args.anomaly_size,
        # Issue #65: Quarantine 4.5% dynamic anomaly equity sizing from Mainnet (keep $45 cap)
        anomaly_pct_equity=args.anomaly_pct_equity if enable_experimental_satellites else None,
        anomaly_prefer_maker=args.anomaly_prefer_maker,
        enable_subhourly_pullback=feature_flags.enable_subhourly_pullback,
        enable_smart_money=feature_flags.enable_smart_money,
        enable_quarterly_basis=feature_flags.enable_quarterly_basis,
        profit_lock_trigger_pct=args.profit_lock_trigger,
        profit_lock_level_pct=args.profit_lock_level,
        trailing_trigger_pct=eff_trailing_trigger,
        trailing_distance_pct=args.trailing_distance,
        core_take_profit_pct=eff_core_tp,
        core_stop_loss_pct=eff_core_sl,
        equity_floor_usd=feature_flags.equity_floor_usd,
        enable_hard_equity_floor=feature_flags.enable_hard_equity_floor,
        auto_close_anomalies_when_disabled=feature_flags.auto_close_anomalies_when_disabled,
        enable_intracycle_rebalance=feature_flags.enable_intracycle_rebalance,
        enable_invariant_audit=feature_flags.enable_invariant_audit,
        align_rebalance_to_funding_utc=bool(use_seven_factor and getattr(args, "align_funding_clock", True)),
        feature_flags=feature_flags,
        min_rebalance_notional=(
            max(21.0, min(50.0, effective_capital * 0.01))
            if effective_capital >= 500.0
            else (
                min(10.0, args.anomaly_size)
                if feature_flags.enable_anomaly_capsule
                else max(10.0, min(50.0, effective_capital * 0.01))
            )
        ),
    )
    # In --shadow mode (or when --rebalance-now is explicitly requested), trigger an immediate rebalance on startup.
    # In --mainnet mode, preserve the scheduled 8h funding-aligned cadence (`daemon.last_rebalance_time` from SQLite)
    # so container restarts / deployments never induce spurious mid-bar portfolio churn, EXCEPT a one-time migration
    # when upgrading a DB explicitly recorded under `static_priors` mode.
    if getattr(args, "rebalance_now", False) or (mode == "shadow"):
        daemon.last_rebalance_time = None
    elif mode == "mainnet" and priors_cfg is None and hasattr(db, "get_latest_seven_factor_telemetry"):
        sf_prev = db.get_latest_seven_factor_telemetry() or {}
        saved_mode = str(sf_prev.get("engine_mode") or "")
        prev_w = sf_prev.get("factor_weights") or {}
        is_all_seven_static = len(prev_w) == 7 and all(
            float(prev_w.get(k, 0.0)) >= 0.08 for k in ("macro", "med", "fast", "rev", "carry", "liq", "size")
        )
        if saved_mode == "static_priors" or (not saved_mode and is_all_seven_static):
            logger.info(
                "🚀 [Dynamic Online IC Migration] Legacy static-prior factor weights detected — scheduling one-time rebalance."
            )
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
            if not (daemon._thread and daemon._thread.is_alive()):
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
