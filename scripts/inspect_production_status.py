#!/usr/bin/env python3
"""
Read-only Production Telemetry, Strategy Performance & Barrier Audit Utility.

Inspects `data/live_trading.db` (or `--db-path`), `logs/m26_l2_sweep_telemetry.jsonl`,
and `crypto-mainnet` container logs/state without acquiring write locks (`mode=ro`).
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any


def run_production_audit(base_dir: Path, container_name: str = "crypto-mainnet") -> dict[str, Any]:
    db_path = base_dir / "data" / "live_trading.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    out: dict[str, Any] = {}

    # 1. Account & Equity Progression
    cur.execute("SELECT * FROM accounts ORDER BY id DESC LIMIT 1")
    r = cur.fetchone()
    out["latest_account"] = dict(r) if r else {}

    cur.execute(
        "SELECT MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts, "
        "MIN(total_equity) AS min_eq, MAX(total_equity) AS max_eq, COUNT(*) AS n_snaps "
        "FROM accounts"
    )
    out["account_span"] = dict(cur.fetchone())

    for key, lookback in [
        ("account_24h_ago", "-24 hours"),
        ("account_7d_ago", "-7 days"),
        ("account_30d_ago", "-30 days"),
    ]:
        cur.execute(
            "SELECT timestamp, total_equity, cash_balance, unrealized_pnl, realized_pnl, leverage "
            f"FROM accounts WHERE timestamp >= datetime('now', '{lookback}') ORDER BY id ASC LIMIT 1"
        )
        row = cur.fetchone()
        out[key] = dict(row) if row else {}

    # 2. Telemetry, Heartbeat, Shield Invariants, Reconciliations & Feature Flags
    for tbl in ("daemon_heartbeat", "seven_factor_telemetry", "shield_invariants_telemetry"):
        try:
            cur.execute(f"SELECT * FROM {tbl} LIMIT 1")
            row = cur.fetchone()
            out[tbl] = dict(row) if row else {}
        except Exception as exc:  # noqa: BLE001
            out[tbl] = {"error": str(exc)}

    cur.execute(
        "SELECT status, COUNT(*) AS cnt, MAX(timestamp) AS last_ts FROM reconciliation_audit_logs GROUP BY status"
    )
    out["reconciliation_summary"] = [dict(x) for x in cur.fetchall()]
    cur.execute("SELECT * FROM reconciliation_audit_logs ORDER BY id DESC LIMIT 3")
    out["recent_reconciliations"] = [dict(x) for x in cur.fetchall()]

    cur.execute("SELECT * FROM feature_flags")
    out["feature_flags"] = [dict(x) for x in cur.fetchall()]

    # 3. Open Positions (Live Mark-to-Market + Active SL/TP/Peak Brackets)
    cur.execute("SELECT * FROM positions WHERE quantity > 0 AND side != 'FLAT' ORDER BY notional DESC")
    open_pos = [dict(x) for x in cur.fetchall()]
    out["open_positions"] = open_pos

    open_by_strat: dict[str, dict[str, float]] = {}
    for p in open_pos:
        st = str(p.get("strategy") or "Core_or_Unlabeled")
        agg = open_by_strat.setdefault(st, {"count": 0, "gross_notional": 0.0, "net_notional": 0.0, "upnl": 0.0})
        agg["count"] += 1
        notional = float(p.get("notional") or 0.0)
        sgn = 1.0 if p.get("side") == "LONG" else -1.0
        agg["gross_notional"] = round(agg["gross_notional"] + notional, 2)
        agg["net_notional"] = round(agg["net_notional"] + sgn * notional, 2)
        agg["upnl"] = round(agg["upnl"] + float(p.get("unrealized_pnl") or 0.0), 4)
    out["open_positions_by_strategy"] = open_by_strat

    # 4. Realized Trades by Strategy (All-Time, 7d, 24h)
    for label, clause in [
        ("trades_by_strategy_all", ""),
        ("trades_by_strategy_7d", "WHERE t.timestamp >= datetime('now', '-7 days')"),
        ("trades_by_strategy_24h", "WHERE t.timestamp >= datetime('now', '-24 hours')"),
    ]:
        cur.execute(
            f"""
            SELECT
                COALESCE(NULLIF(t.strategy, ''), NULLIF(o.strategy, ''), 'UNKNOWN') AS strategy,
                COUNT(*) AS n_fills,
                SUM(CASE WHEN abs(t.realized_pnl) > 1e-8 THEN 1 ELSE 0 END) AS n_closing_fills,
                SUM(CASE WHEN t.realized_pnl > 1e-8 THEN 1 ELSE 0 END) AS n_win_closes,
                SUM(CASE WHEN t.realized_pnl < -1e-8 THEN 1 ELSE 0 END) AS n_loss_closes,
                ROUND(SUM(t.notional), 2) AS total_notional,
                ROUND(SUM(t.realized_pnl), 4) AS gross_realized_pnl,
                ROUND(SUM(t.fee_paid), 4) AS total_fees,
                ROUND(SUM(t.realized_pnl - t.fee_paid), 4) AS net_realized_pnl,
                ROUND(AVG(CASE WHEN t.is_maker = 1 THEN 100.0 ELSE 0.0 END), 1) AS maker_pct,
                MIN(t.timestamp) AS first_trade,
                MAX(t.timestamp) AS last_trade
            FROM trades t
            LEFT JOIN orders o ON t.order_id = o.id
            {clause}
            GROUP BY COALESCE(NULLIF(t.strategy, ''), NULLIF(o.strategy, ''), 'UNKNOWN')
            ORDER BY net_realized_pnl DESC
            """
        )
        out[label] = [dict(x) for x in cur.fetchall()]

    # 5. Funding Settlements
    cur.execute(
        """
        SELECT
            COUNT(*) AS n_events,
            ROUND(SUM(cashflow_credited), 4) AS total_funding_usd,
            ROUND(SUM(CASE WHEN cashflow_credited > 0 THEN cashflow_credited ELSE 0 END), 4) AS pos_funding_usd,
            ROUND(SUM(CASE WHEN cashflow_credited < 0 THEN cashflow_credited ELSE 0 END), 4) AS neg_funding_usd,
            MIN(timestamp) AS first_ts,
            MAX(timestamp) AS last_ts
        FROM funding_settlements
        """
    )
    out["funding_summary"] = dict(cur.fetchone())

    cur.execute(
        """
        SELECT symbol, position_side, COUNT(*) AS n, ROUND(SUM(cashflow_credited), 4) AS net_funding
        FROM funding_settlements
        GROUP BY symbol, position_side
        ORDER BY abs(SUM(cashflow_credited)) DESC
        LIMIT 10
        """
    )
    out["top_funding_symbols"] = [dict(x) for x in cur.fetchall()]

    # 6. Order Types & Exit Barriers Matrix
    cur.execute(
        """
        SELECT
            COALESCE(NULLIF(strategy, ''), 'UNKNOWN') AS strategy,
            order_type,
            status,
            COALESCE(role, 'NA') AS role,
            COUNT(*) AS cnt
        FROM orders
        GROUP BY COALESCE(NULLIF(strategy, ''), 'UNKNOWN'), order_type, status, COALESCE(role, 'NA')
        ORDER BY cnt DESC
        """
    )
    out["orders_matrix"] = [dict(x) for x in cur.fetchall()]

    # Closing trades joined with order_type so we see exact PnL per exit barrier (STOP_LOSS vs TAKE_PROFIT vs LIMIT vs MARKET)
    cur.execute(
        """
        SELECT
            COALESCE(NULLIF(t.strategy, ''), NULLIF(o.strategy, ''), 'UNKNOWN') AS strategy,
            COALESCE(o.order_type, 'EXCHANGE_OR_BRACKET') AS exit_order_type,
            COUNT(*) AS n_closes,
            SUM(CASE WHEN t.realized_pnl > 1e-8 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN t.realized_pnl < -1e-8 THEN 1 ELSE 0 END) AS losses,
            ROUND(SUM(t.realized_pnl), 4) AS gross_pnl,
            ROUND(SUM(t.fee_paid), 4) AS fees,
            ROUND(SUM(t.realized_pnl - t.fee_paid), 4) AS net_pnl
        FROM trades t
        LEFT JOIN orders o ON t.order_id = o.id
        WHERE abs(t.realized_pnl) > 1e-8
        GROUP BY COALESCE(NULLIF(t.strategy, ''), NULLIF(o.strategy, ''), 'UNKNOWN'),
                 COALESCE(o.order_type, 'EXCHANGE_OR_BRACKET')
        ORDER BY n_closes DESC
        """
    )
    out["closing_trades_by_barrier_type"] = [dict(x) for x in cur.fetchall()]

    cur.execute(
        """
        SELECT t.timestamp, t.symbol, t.side, t.price, ROUND(t.notional, 2) AS notional,
               ROUND(t.fee_paid, 4) AS fee, ROUND(t.realized_pnl, 4) AS rpnl, t.is_maker,
               COALESCE(NULLIF(t.strategy, ''), NULLIF(o.strategy, ''), 'UNKNOWN') AS strat,
               o.order_type, o.client_order_id
        FROM trades t
        LEFT JOIN orders o ON t.order_id = o.id
        WHERE abs(t.realized_pnl) > 1e-8
        ORDER BY t.timestamp DESC
        LIMIT 25
        """
    )
    out["recent_closing_trades"] = [dict(x) for x in cur.fetchall()]

    cur.execute(
        "SELECT id, timestamp, regime_detected, momentum_weight, stat_arb_weight, carry_weight "
        "FROM rebalance_snapshots ORDER BY id DESC LIMIT 6"
    )
    out["recent_rebalances"] = [dict(x) for x in cur.fetchall()]

    # 7. Log Barrier & Gate Frequency Scan
    barrier_counts: Counter[str] = Counter()
    log_warnings: list[str] = []
    log_events: list[str] = []

    log_lines: list[str] = []
    for lf in sorted(glob.glob(str(base_dir / "logs" / "*.log*")))[-4:]:
        try:
            with open(lf, encoding="utf-8", errors="ignore") as f:
                log_lines.extend(f.readlines()[-5000:])
        except OSError:
            pass

    try:
        dlog = subprocess.check_output(
            ["docker", "logs", "--tail", "3000", container_name],
            stderr=subprocess.STDOUT,
            text=True,
            errors="ignore",
        )
        log_lines.extend(dlog.splitlines())
    except Exception as exc:  # noqa: BLE001
        log_warnings.append(f"docker_logs_note: {exc}")

    patterns = {
        "STOP_LOSS_TRIGGER_OR_BRACKET": re.compile(r"Stop[- ]?Loss|STOP_LOSS|SL triggered|SL hit", re.I),
        "TAKE_PROFIT_TRIGGER_OR_BRACKET": re.compile(r"Take[- ]?Profit|TAKE_PROFIT|TP triggered|TP hit", re.I),
        "PROFIT_LOCK_RATCHET": re.compile(r"Profit[- ]?Lock|locked stop|lock_level", re.I),
        "TRAILING_STOP_RATCHET": re.compile(r"Trailing stop|trail_distance|ratcheted", re.I),
        "TIME_BARRIER_MAX_HOLD": re.compile(r"max[_ ]hold|time[- ]?stop|Time barrier|expired", re.I),
        "MIN_DELTA_TURNOVER_BARRIER": re.compile(r"min_trade|below threshold|skip.*small|turnover", re.I),
        "CPPI_CUSHION_OR_FLOOR_BARRIER": re.compile(r"Cushion|HardEquityFloor|SafeRamp|Drawdown Brake", re.I),
        "ONE_WAY_COLLISION_BARRIER": re.compile(r"collision|already has an open|blocked_clean", re.I),
        "RATE_LIMIT_429_OR_THROTTLE": re.compile(r"429|Too Many Requests|rate_limit.*wait", re.I),
        "SNIPER_SCAN_OR_SIGNAL": re.compile(r"Sniper|M09|M23|M04|M25|M26|L2Sweep", re.I),
    }

    seen: set[str] = set()
    for raw_ln in log_lines:
        s = raw_ln.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        for k, pat in patterns.items():
            if pat.search(s):
                barrier_counts[k] += 1
        if any(lvl in s for lvl in ("ERROR", "CRITICAL", "Traceback", "WARNING")):
            log_warnings.append(s[-220:])
        if any(
            tag in s for tag in ("Sniper", "M26", "Trailing", "Profit", "Rebalance", "Executed", "Closed", "MaxHold")
        ):
            log_events.append(s[-220:])

    out["log_barrier_counts"] = dict(barrier_counts)
    out["recent_warnings_errors"] = log_warnings[-25:]
    out["recent_key_events"] = log_events[-30:]

    # 8. M26 Prospective L2 + 1ms Sweep Gate Telemetry
    m26_path = base_dir / "logs" / "m26_l2_sweep_telemetry.jsonl"
    if m26_path.exists():
        rows_m26 = [json.loads(x) for x in m26_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        gate_fail_counts: Counter[str] = Counter()
        for r26 in rows_m26:
            if float(r26.get("oi_ratio_24h") or 0.0) < 1.16:
                gate_fail_counts["GATE_1_OI_EXPANSION_LT_1.16X"] += 1
            elif (
                r26.get("book_imb_1pct") is None
                or float(r26.get("book_imb_1pct") or 0.0) < 0.24
                or float(r26.get("book_imb_2pct") or 0.0) < 0.20
            ):
                gate_fail_counts["GATE_2_L2_BOOK_AIR_POCKET_FAIL"] += 1
            elif float(r26.get("spot_tb_3m") or 0.0) < 0.53:
                gate_fail_counts["GATE_3_SPOT_TAKER_BUY_LT_53PCT"] += 1
            elif float(r26.get("sweep_conc_5m") or 0.0) < 0.32 or float(r26.get("whale_buy_15m") or 0.0) < 0.53:
                gate_fail_counts["GATE_4_1MS_INST_SWEEP_FAIL"] += 1
            else:
                gate_fail_counts["PASSED_ALL_GATES_EXECUTED"] += 1
        out["m26_telemetry_summary"] = {
            "total_4h_breakout_candidates": len(rows_m26),
            "gate_breakdown": dict(gate_fail_counts),
            "recent_candidates": rows_m26[-8:],
        }
    else:
        out["m26_telemetry_summary"] = {
            "total_4h_breakout_candidates": 0,
            "gate_breakdown": {},
            "recent_candidates": [],
        }

    # 9. Docker Container State & Restarts
    try:
        insp = subprocess.check_output(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "Status={{.State.Status}} OOMKilled={{.State.OOMKilled}} Restarts={{.RestartCount}} StartedAt={{.State.StartedAt}}",
            ],
            text=True,
        ).strip()
        out["container_inspect"] = insp
    except Exception as exc:  # noqa: BLE001
        out["container_inspect"] = str(exc)

    conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Production Audit")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(os.path.expanduser("~/crypto-alpha-engine")),
        help="Root directory of crypto-alpha-engine on host",
    )
    parser.add_argument("--container", type=str, default="crypto-mainnet")
    args = parser.parse_args()
    report = run_production_audit(args.base_dir, container_name=args.container)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
