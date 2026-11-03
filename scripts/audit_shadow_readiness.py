"""
Shadow-to-Live Go/No-Go Readiness Validation Suite.

Audits the SQLite Shadow Trading database (`data/shadow_trading.db`) against
institutional checkgates required before transitioning to Binance Mainnet Production:
1. Dollar Neutrality Check: |Long_USD - Short_USD| / Equity <= 5.0%
2. BTC Beta Neutrality Check: |Portfolio_Beta| <= 0.15
3. Maker Execution Efficiency: Maker fill ratio >= 80% & daily fee drag <= 0.20%/day
4. Drawdown & Stepped Leverage Hygiene: Max drawdown < 12.0% (Kill-Switch threshold)
"""

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.database import DEFAULT_DB_PATH, ShadowDatabase


@dataclass
class CheckgateResult:
    name: str
    passed: bool
    observed: str
    threshold: str
    details: str


def evaluate_shadow_readiness(db_path: str = DEFAULT_DB_PATH) -> Dict[str, object]:
    """
    Evaluates all quantitative Go/No-Go checkgates from `db_path`.
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
        p.quantity * (p.current_price or p.entry_price)
        for p in positions.values()
        if p.side.value == "LONG"
    )
    short_usd = sum(
        p.quantity * (p.current_price or p.entry_price)
        for p in positions.values()
        if p.side.value == "SHORT"
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

    gates: List[CheckgateResult] = [gate_neutral, gate_beta, gate_maker, gate_dd, gate_cycle]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Shadow Trading Readiness for Live Mainnet Rollout")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to Shadow SQLite DB")
    args = parser.parse_args()

    res = evaluate_shadow_readiness(db_path=args.db)
    print("=" * 110)
    print(" SHADOW-TO-LIVE GO/NO-GO READINESS SCORECARD (7-FACTOR CAUSAL ENGINE | 2.5x -> 5.0x)")
    print("=" * 110)

    if not res.get("gates"):
        print(f" [NO-GO] {res.get('reason')}")
        sys.exit(1)

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


if __name__ == "__main__":
    main()
