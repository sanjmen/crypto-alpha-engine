#!/usr/bin/env python3
"""
Frente A — Read-Only Forensic Audit of Live Execution Microstructure (`live_trading.db`).

Analyzes real Mainnet execution quality in strictly read-only SQLite mode (`?mode=ro`):
  1. Maker vs. Taker-Fallback split on `execute_maker_limit_chase` (`order_type='LIMIT'`, `role='MAKER'` vs `'TAKER'`).
  2. Implementation Shortfall (`slippage_bps` signed relative to arrival `orders.price`) + Exchange Fee (`fee_bps`) = All-In Cost (`bps`).
  3. Fill Latency (`updated_at - created_at` seconds) proving the cost of failed `3 x 5s` GTX chases.
  4. Funding Settlement Storm Collision (`00/08/16 UTC` minutes `:00-:02` vs off-settlement minutes).
  5. Symbol-by-Symbol & Order-Size Bucket Execution Cost Scorecard.
  6. Realized Funding Cashflows (`funding_settlements`) & Current Live Net Exposure Attribution.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
from typing import Any


def _parse_iso_sec(ts_str: str | None) -> float | None:
    if not ts_str:
        return None
    s = str(ts_str).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_orders": 0,
            "notional_usd": 0.0,
            "maker_order_pct": 0.0,
            "maker_notional_pct": 0.0,
            "mean_slippage_bps": 0.0,
            "notional_wtd_slippage_bps": 0.0,
            "median_slippage_bps": 0.0,
            "p90_slippage_bps": 0.0,
            "notional_wtd_fee_bps": 0.0,
            "notional_wtd_all_in_bps": 0.0,
            "mean_latency_sec": 0.0,
            "median_latency_sec": 0.0,
            "p90_latency_sec": 0.0,
        }

    tot_notional = sum(r["notional"] for r in rows)
    maker_notional = sum(r["notional"] for r in rows if r["is_maker"])
    maker_cnt = sum(1 for r in rows if r["is_maker"])
    slips = sorted(r["slippage_bps"] for r in rows)
    lats = sorted(r["latency_sec"] for r in rows if r["latency_sec"] is not None)

    w_slip = sum(r["slippage_bps"] * r["notional"] for r in rows) / tot_notional if tot_notional > 1e-9 else 0.0
    w_fee = sum(r["fee_bps"] * r["notional"] for r in rows) / tot_notional if tot_notional > 1e-9 else 0.0
    return {
        "n_orders": len(rows),
        "notional_usd": round(tot_notional, 2),
        "maker_order_pct": round(100.0 * maker_cnt / len(rows), 1),
        "maker_notional_pct": round(100.0 * maker_notional / max(tot_notional, 1e-9), 1),
        "mean_slippage_bps": round(sum(slips) / len(slips), 2),
        "notional_wtd_slippage_bps": round(w_slip, 2),
        "median_slippage_bps": round(_percentile(slips, 0.50), 2),
        "p90_slippage_bps": round(_percentile(slips, 0.90), 2),
        "notional_wtd_fee_bps": round(w_fee, 2),
        "notional_wtd_all_in_bps": round(w_slip + w_fee, 2),
        "mean_latency_sec": round(sum(lats) / len(lats), 2) if lats else 0.0,
        "median_latency_sec": round(_percentile(lats, 0.50), 2) if lats else 0.0,
        "p90_latency_sec": round(_percentile(lats, 0.90), 2) if lats else 0.0,
    }


def audit_live_execution_db(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            o.id AS order_id,
            o.symbol,
            o.side,
            o.order_type,
            o.price AS arrival_limit_px,
            o.quantity AS req_qty,
            o.filled_quantity AS fill_qty,
            o.average_fill_price AS avg_fill_px,
            o.status,
            o.role,
            o.fee AS order_fee,
            o.strategy AS order_strategy,
            o.created_at,
            o.updated_at,
            t.price AS trade_px,
            t.notional AS trade_notional,
            t.fee_paid AS trade_fee,
            t.is_maker AS trade_is_maker,
            t.realized_pnl AS trade_rpnl,
            t.strategy AS trade_strategy,
            t.timestamp AS trade_ts
        FROM orders o
        LEFT JOIN trades t ON t.order_id = o.id
        WHERE o.status = 'FILLED'
        ORDER BY o.created_at ASC
        """
    )
    raw_rows = [dict(r) for r in cur.fetchall()]

    enriched: list[dict[str, Any]] = []
    for r in raw_rows:
        fill_px = float(r["avg_fill_px"] or r["trade_px"] or 0.0)
        arr_px = float(r["arrival_limit_px"] or fill_px)
        qty = float(r["fill_qty"] or r["req_qty"] or 0.0)
        if fill_px <= 0.0 or qty <= 0.0:
            continue
        notional = float(r["trade_notional"] or (fill_px * qty))
        fee_usd = float(r["trade_fee"] if r["trade_fee"] is not None else (r["order_fee"] or 0.0))
        side = str(r["side"]).upper()
        sgn = 1.0 if side == "BUY" else -1.0
        # Positive slippage_bps = filled at a WORSE price than `arrival_limit_px`; negative = price improvement
        slippage_bps = sgn * ((fill_px - arr_px) / max(arr_px, 1e-12)) * 10_000.0 if arr_px > 0 else 0.0
        fee_bps = (fee_usd / max(notional, 1e-12)) * 10_000.0
        is_maker = (
            bool(int(r["trade_is_maker"]) == 1)
            if r["trade_is_maker"] is not None
            else (str(r["role"] or "").upper() == "MAKER")
        )
        t_c = _parse_iso_sec(r["created_at"])
        t_u = _parse_iso_sec(r["trade_ts"] or r["updated_at"])
        lat = max(0.0, t_u - t_c) if (t_c is not None and t_u is not None) else None

        dt_utc = datetime.fromtimestamp(t_c, tz=datetime.now().astimezone().tzinfo) if t_c else None
        is_settle_window = False
        utc_hr, utc_min = -1, -1
        if t_c is not None:
            from datetime import timezone

            d_u = datetime.fromtimestamp(t_c, tz=timezone.utc)
            utc_hr, utc_min = d_u.hour, d_u.minute
            is_settle_window = (utc_hr in (0, 8, 16)) and (utc_min in (0, 1, 2))
        _ = dt_utc

        strat = str(r["trade_strategy"] or r["order_strategy"] or "UNKNOWN")
        enriched.append(
            {
                "order_id": r["order_id"],
                "symbol": str(r["symbol"]).split(":")[0].replace("/", ""),
                "side": side,
                "order_type": str(r["order_type"]),
                "role": "MAKER" if is_maker else "TAKER",
                "is_maker": is_maker,
                "notional": notional,
                "fee_usd": fee_usd,
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "all_in_bps": slippage_bps + fee_bps,
                "latency_sec": lat,
                "is_settle_window": is_settle_window,
                "utc_hour": utc_hr,
                "utc_minute": utc_min,
                "strategy": strat,
                "realized_pnl": float(r["trade_rpnl"] or 0.0),
                "created_at": r["created_at"],
            }
        )

    limit_orders = [x for x in enriched if x["order_type"] == "LIMIT"]
    limit_maker = [x for x in limit_orders if x["is_maker"]]
    limit_taker_fb = [x for x in limit_orders if not x["is_maker"]]
    market_orders = [x for x in enriched if x["order_type"] == "MARKET"]

    # Post-fill short-horizon adverse selection proxy on LIMIT fills:
    # Compare fill price to next observed account/position or next trade on same symbol within 30m
    by_sym: dict[str, list[dict[str, Any]]] = {}
    for x in enriched:
        by_sym.setdefault(x["symbol"], []).append(x)

    sym_scorecard: dict[str, Any] = {}
    for sym, s_rows in sorted(by_sym.items(), key=lambda kv: -sum(z["notional"] for z in kv[1])):
        s_lim = [z for z in s_rows if z["order_type"] == "LIMIT"]
        sym_scorecard[sym] = {
            "all_orders": _summarize_group(s_rows),
            "limit_chase_only": _summarize_group(s_lim),
        }

    # Notional size buckets on LIMIT orders
    size_buckets = {
        "small_lt_250": _summarize_group([x for x in limit_orders if x["notional"] < 250.0]),
        "mid_250_to_600": _summarize_group([x for x in limit_orders if 250.0 <= x["notional"] < 600.0]),
        "large_ge_600": _summarize_group([x for x in limit_orders if x["notional"] >= 600.0]),
    }

    # Funding settlement window (`00/08/16 UTC :00-:02`) vs off-settlement on LIMIT orders
    settle_collision = {
        "in_funding_settlement_00_02m": _summarize_group([x for x in limit_orders if x["is_settle_window"]]),
        "off_settlement_minutes": _summarize_group([x for x in limit_orders if not x["is_settle_window"]]),
    }

    # Status distribution (including CANCELED / REJECTED)
    cur.execute("SELECT order_type, status, role, COUNT(*) AS cnt FROM orders GROUP BY order_type, status, role")
    status_counts = [dict(r) for r in cur.fetchall()]

    # Funding settlements table summary
    cur.execute(
        """
        SELECT
            COUNT(*) AS n_events,
            ROUND(SUM(cashflow_credited), 4) AS net_funding_usd,
            ROUND(SUM(CASE WHEN cashflow_credited > 0 THEN cashflow_credited ELSE 0 END), 4) AS pos_funding_usd,
            ROUND(SUM(CASE WHEN cashflow_credited < 0 THEN cashflow_credited ELSE 0 END), 4) AS neg_funding_usd
        FROM funding_settlements
        """
    )
    fnd_tot = dict(cur.fetchone() or {})
    cur.execute(
        """
        SELECT symbol, position_side, COUNT(*) AS n, ROUND(SUM(cashflow_credited), 4) AS net_usd,
               ROUND(AVG(funding_rate) * 10000.0, 2) AS avg_fr_bps
        FROM funding_settlements
        GROUP BY symbol, position_side
        ORDER BY net_usd ASC
        """
    )
    fnd_by_sym = [dict(r) for r in cur.fetchall()]

    # Live Net $ exact breakdown (`Core P1` vs `Manga P2` vs `Snipers`)
    cur.execute("SELECT * FROM seven_factor_telemetry LIMIT 1")
    sf_row = dict(cur.fetchone() or {})
    p2_weights: dict[str, float] = {}
    if sf_row.get("p2_weights_json"):
        try:
            p2_weights = json.loads(sf_row["p2_weights_json"])
        except Exception:  # noqa: BLE001
            p2_weights = {}

    cur.execute(
        "SELECT symbol, side, quantity, entry_price, current_price, notional, unrealized_pnl, strategy FROM positions WHERE quantity > 0 AND side != 'FLAT'"
    )
    open_pos = [dict(r) for r in cur.fetchall()]
    tot_long = sum(float(p["notional"]) for p in open_pos if p["side"] == "LONG")
    tot_short = sum(float(p["notional"]) for p in open_pos if p["side"] == "SHORT")

    conn.close()
    return {
        "total_filled_orders": len(enriched),
        "overall_summary": _summarize_group(enriched),
        "by_execution_path": {
            "LIMIT_ALL_CHASE": _summarize_group(limit_orders),
            "LIMIT_FILLED_AS_MAKER": _summarize_group(limit_maker),
            "LIMIT_FELL_BACK_TO_TAKER_15S": _summarize_group(limit_taker_fb),
            "DIRECT_MARKET_TAKER": _summarize_group(market_orders),
        },
        "funding_settlement_window_collision": settle_collision,
        "limit_size_buckets": size_buckets,
        "order_status_distribution": status_counts,
        "symbol_scorecard": sym_scorecard,
        "funding_settlements_summary": fnd_tot,
        "funding_by_symbol_side": fnd_by_sym[:20],
        "live_exposure_snapshot": {
            "open_positions_count": len(open_pos),
            "gross_long_usd": round(tot_long, 2),
            "gross_short_usd": round(tot_short, 2),
            "net_usd": round(tot_long - tot_short, 2),
            "p2_weights": p2_weights,
            "open_positions": open_pos,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only forensic audit of live Mainnet execution microstructure")
    parser.add_argument("--db-path", type=Path, default=Path("data/live_trading.db"))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    res = audit_live_execution_db(args.db_path)
    txt = json.dumps(res, indent=2, default=str)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
