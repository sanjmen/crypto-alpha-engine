#!/usr/bin/env python3
"""
6-Month Backtest Runner for Extreme Anomaly & Illiquid Tail Strategies.
Evaluates:
- Baseline: Meta-Strategy Ensemble (Momentum + Stat-Arb + Carry + Cointegrated Pairs RENDER/FET)
- Variant A: Meta-Strategy + Anomaly Strategies (Test Sizing: $15 USDT max notional)
- Variant B: Meta-Strategy + Anomaly Strategies (Full Sizing: $100 USDT max notional)

Period: 2026-04-14 to 2026-10-14 UTC (Exact 6 months prior to today)
Universe: 23 assets (Core + Illiquid Tail + Pumping/Dumping High-Volume Anomaly Movers)
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain.entities import Portfolio
from src.strategies.cointegrated_pairs import CointegratedPairsStrategy
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
)
from src.strategies.meta_allocator import MetaStrategyAllocator

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_UNIVERSE = [
    # Core & Cointegrated benchmark
    "BTCUSDT",
    "ETHUSDT",
    "RENDERUSDT",
    "FETUSDT",
    # Illiquid tail altcoins
    "CTSIUSDT",
    "MAGICUSDT",
    "OGNUSDT",
    "RLCUSDT",
    "GTCUSDT",
    "COMPUSDT",
    "CRVUSDT",
    "DYDXUSDT",
    # Extreme Anomaly Movers (>= $20M daily volume)
    "WALUSDT",
    "BSPUSDT",
    "TRUTHUSDT",
    "ZEREBROUSDT",
    "BTWUSDT",
    "QUSDT",
    "AVAUSDT",
    "EPICUSDT",
    "MUBARAKUSDT",
    "CRWVUSDT",
    "CAPUSDT",
]


def load_market_data(
    symbols: List[str],
    cache_dir: Path = Path("/tmp/crypto_alpha_backtest_cache"),
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Loads 1h bars and funding rates from parquet cache."""
    bars_dir = cache_dir / "bars" / "1h"
    funding_dir = cache_dir / "funding"

    bars_dict: Dict[str, pd.DataFrame] = {}
    funding_dict: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        bar_path = bars_dir / f"{sym}.parquet"
        if bar_path.exists():
            df = pd.read_parquet(bar_path)
            if "timestamp" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            bars_dict[sym] = df

        funding_path = funding_dir / f"{sym}.parquet"
        if funding_path.exists():
            f_df = pd.read_parquet(funding_path)
            if "timestamp" in f_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(f_df["timestamp"]):
                    f_df["timestamp"] = pd.to_datetime(f_df["fundingTime" if "fundingTime" in f_df.columns else "timestamp"], unit="s" if f_df["timestamp"].iloc[0] < 1e11 else "ms", utc=True)
                f_df.set_index("timestamp", inplace=True)
            f_df.sort_index(inplace=True)
            funding_dict[sym] = f_df

    return bars_dict, funding_dict


def run_simulation(
    symbols: List[str],
    bars_dict: Dict[str, pd.DataFrame],
    funding_dict: Dict[str, pd.DataFrame],
    start_date: str = "2026-04-14",
    end_date: str = "2026-10-14",
    initial_capital: float = 1000.0,
    enable_anomalies: bool = True,
    anomaly_size: float = 15.0,
    rebalance_freq_bars: int = 8,
    turnover_deadband: float = 0.04,
    maker_fee: float = 0.0002,
    warmup_bars: int = 240,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """Runs vectorized simulation and captures granular metrics."""
    valid_symbols = [s for s in symbols if s in bars_dict and not bars_dict[s].empty]
    anchor_sym = "BTCUSDT" if "BTCUSDT" in valid_symbols else valid_symbols[0]
    full_idx = bars_dict[anchor_sym].index

    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)

    eval_idx = full_idx[(full_idx >= start_ts) & (full_idx <= end_ts)]
    first_eval_pos = full_idx.get_loc(eval_idx[0])
    warmup_start_pos = max(0, first_eval_pos - warmup_bars)
    sim_idx = full_idx[warmup_start_pos : full_idx.get_loc(eval_idx[-1]) + 1]

    # Close prices matrix
    close_dfs = {}
    for sym in valid_symbols:
        s = bars_dict[sym]["close"].reindex(sim_idx)
        close_dfs[sym] = s.ffill().bfill()
    close_df = pd.DataFrame(close_dfs)
    returns_df = close_df.pct_change().fillna(0.0)

    # Funding settlements matrix (hours 0, 8, 16)
    funding_dfs = {}
    for sym in valid_symbols:
        if sym in funding_dict and not funding_dict[sym].empty:
            f_series = funding_dict[sym]["funding_rate"]
            f_aligned = f_series.reindex(sim_idx, method="ffill").fillna(0.0)
            is_settlement = sim_idx.hour.isin([0, 8, 16])
            funding_dfs[sym] = f_aligned * is_settlement
        else:
            funding_dfs[sym] = pd.Series(0.0, index=sim_idx)
    funding_df = pd.DataFrame(funding_dfs)

    # Instantiate strategies
    pairs_strat = CointegratedPairsStrategy(lookback_bars=240)
    oversold_strat = OversoldBounceStrategy(max_position_usd=anomaly_size) if enable_anomalies else None
    breakout_strat = EarlyBreakoutStrategy(max_position_usd=anomaly_size) if enable_anomalies else None
    fade_strat = PumpFundingFadeStrategy(max_position_usd=anomaly_size) if enable_anomalies else None

    allocator = MetaStrategyAllocator(
        pairs_strategy=pairs_strat,
        oversold_bounce_strategy=oversold_strat,
        early_breakout_strategy=breakout_strat,
        funding_fade_strategy=fade_strat,
    )

    portfolio = Portfolio(cash=initial_capital, initial_cash=initial_capital)
    raw_weights_list = []
    current_weights = {sym: 0.0 for sym in valid_symbols}
    total_steps = len(sim_idx)

    # Anomaly trigger counters
    anomaly_counts = {"oversold_bounce": 0, "early_breakout": 0, "pump_funding_fade": 0}

    for pos in range(warmup_bars, total_steps):
        ts = sim_idx[pos]

        if (pos - warmup_bars) % rebalance_freq_bars == 0:
            window_start = max(0, pos - 250)
            sub_idx = sim_idx[window_start : pos + 1]

            sub_market = {}
            for sym in valid_symbols:
                c_idx = bars_dict[sym].index.intersection(sub_idx)
                if len(c_idx) >= 26:
                    sub_market[sym] = bars_dict[sym].loc[c_idx]

            sub_funding = {
                sym: funding_dict[sym].loc[funding_dict[sym].index <= ts]
                for sym in valid_symbols
                if sym in funding_dict and not funding_dict[sym].empty
            }

            volatilities = {
                sym: float(returns_df[sym].iloc[max(0, pos - 24) : pos].std()) or 0.02
                for sym in sub_market
            }

            # Count anomaly triggers
            if enable_anomalies and oversold_strat and breakout_strat and fade_strat:
                anomaly_counts["oversold_bounce"] += len(oversold_strat.generate_signals(sub_market))
                anomaly_counts["early_breakout"] += len(breakout_strat.generate_signals(sub_market))
                anomaly_counts["pump_funding_fade"] += len(fade_strat.generate_signals(sub_market, funding_data=sub_funding))

            target_dollars, regime, strat_w = allocator.generate_portfolio_allocations(
                market_data=sub_market,
                portfolio=portfolio,
                volatilities=volatilities,
                funding_data=sub_funding,
            )

            total_eq = portfolio.total_equity
            current_weights = {
                sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0)
                for sym in valid_symbols
            }

        raw_weights_list.append((ts, current_weights.copy()))

    w_records = {ts: w for ts, w in raw_weights_list}
    raw_weights_df = pd.DataFrame.from_dict(w_records, orient="index").reindex(eval_idx).fillna(0.0)
    rets_eval = returns_df.reindex(eval_idx).fillna(0.0)
    funding_eval = funding_df.reindex(eval_idx).fillna(0.0)

    # Turnover deadband
    effective_weights_arr = np.zeros_like(raw_weights_df.values)
    prev_w = np.zeros(len(valid_symbols))
    raw_arr = raw_weights_df.values

    for t in range(len(raw_arr)):
        desired = raw_arr[t]
        delta = np.abs(desired - prev_w)
        rebalance_mask = delta > turnover_deadband
        new_w = np.where(rebalance_mask, desired, prev_w)
        effective_weights_arr[t] = new_w
        prev_w = new_w

    weights_df = pd.DataFrame(effective_weights_arr, index=eval_idx, columns=valid_symbols)
    shifted_weights = weights_df.shift(1).fillna(0.0)

    # Financial mechanics
    gross_returns = (shifted_weights * rets_eval).sum(axis=1)
    weight_turnover = (weights_df - shifted_weights).abs().sum(axis=1)
    transaction_costs = weight_turnover * maker_fee
    funding_pnl = -(shifted_weights * funding_eval).sum(axis=1)
    net_returns = gross_returns - transaction_costs + funding_pnl

    equity_curve = initial_capital * (1.0 + net_returns).cumprod()
    prev_equity = equity_curve.shift(1).fillna(initial_capital)

    hourly_df = pd.DataFrame({
        "gross_return": gross_returns,
        "weight_turnover": weight_turnover,
        "fee_cost_pct": transaction_costs,
        "funding_pnl_pct": funding_pnl,
        "net_return": net_returns,
        "equity": equity_curve,
        "gross_exposure": weights_df.abs().sum(axis=1),
        "gross_pnl_usd": gross_returns * prev_equity,
        "fees_usd": transaction_costs * prev_equity,
        "funding_usd": funding_pnl * prev_equity,
        "net_pnl_usd": net_returns * prev_equity,
    }, index=eval_idx)

    # Daily aggregation
    daily_groups = hourly_df.groupby(hourly_df.index.date)
    daily_records = []
    running_max_equity = initial_capital

    for dt, group in daily_groups:
        start_eq = float(group["equity"].iloc[0] - group["net_pnl_usd"].iloc[0])
        end_eq = float(group["equity"].iloc[-1])
        day_pnl = end_eq - start_eq
        day_ret_pct = (end_eq / start_eq - 1.0) * 100.0 if start_eq > 0 else 0.0

        cum_pnl = end_eq - initial_capital
        cum_ret_pct = (end_eq / initial_capital - 1.0) * 100.0

        running_max_equity = max(running_max_equity, end_eq)
        drawdown_pct = (end_eq - running_max_equity) / running_max_equity * 100.0

        gross_pnl_day = float(group["gross_pnl_usd"].sum())
        fees_day = float(group["fees_usd"].sum())
        funding_day = float(group["funding_usd"].sum())
        avg_leverage = float(group["gross_exposure"].mean())
        turnover_day = float(group["weight_turnover"].sum())

        daily_records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "start_equity": round(start_eq, 2),
            "end_equity": round(end_eq, 2),
            "daily_pnl": round(day_pnl, 2),
            "daily_return_pct": round(day_ret_pct, 2),
            "cum_pnl": round(cum_pnl, 2),
            "cum_return_pct": round(cum_ret_pct, 2),
            "gross_pnl": round(gross_pnl_day, 2),
            "fees_paid": round(fees_day, 2),
            "funding_pnl": round(funding_day, 2),
            "avg_leverage": round(avg_leverage, 2),
            "turnover": round(turnover_day, 2),
            "drawdown_pct": round(drawdown_pct, 2),
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df.set_index("date", inplace=True)

    # Compute KPIs
    total_return_pct = float((equity_curve.iloc[-1] / initial_capital - 1.0) * 100.0)
    n_days = len(daily_df)
    cagr_pct = float(((equity_curve.iloc[-1] / initial_capital) ** (365.0 / n_days) - 1.0) * 100.0)

    daily_rets = daily_df["daily_return_pct"] / 100.0
    mean_d = float(daily_rets.mean())
    std_d = float(daily_rets.std()) or 1e-8
    sharpe = float((mean_d / std_d) * np.sqrt(365))

    downside_d = float(daily_rets[daily_rets < 0].std()) or 1e-8
    sortino = float((mean_d / downside_d) * np.sqrt(365))

    running_max_series = equity_curve.cummax()
    drawdown_series = (equity_curve - running_max_series) / running_max_series
    max_dd_pct = float(abs(drawdown_series.min()) * 100.0)
    calmar = float(cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

    win_days = int((daily_df["daily_pnl"] > 0).sum())
    loss_days = int((daily_df["daily_pnl"] < 0).sum())
    win_rate_pct = float((win_days / n_days) * 100.0)

    gross_profit = float(daily_df[daily_df["daily_pnl"] > 0]["daily_pnl"].sum())
    gross_loss = float(abs(daily_df[daily_df["daily_pnl"] < 0]["daily_pnl"].sum())) or 1e-8
    profit_factor = float(gross_profit / gross_loss)

    total_fees = float(daily_df["fees_paid"].sum())
    total_funding = float(daily_df["funding_pnl"].sum())
    total_turnover = float(daily_df["turnover"].sum())

    kpis = {
        "initial_capital": initial_capital,
        "final_equity": float(equity_curve.iloc[-1]),
        "total_pnl": float(equity_curve.iloc[-1] - initial_capital),
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "calmar_ratio": calmar,
        "win_rate_days_pct": win_rate_pct,
        "win_days": win_days,
        "loss_days": loss_days,
        "profit_factor": profit_factor,
        "total_fees_paid": total_fees,
        "total_funding_pnl": total_funding,
        "total_turnover": total_turnover,
        "n_days": n_days,
    }

    return hourly_df, daily_df, kpis, anomaly_counts


def main():
    parser = argparse.ArgumentParser(description="Run 6-Month Backtest for Extreme Anomaly & Illiquid Tail Strategies")
    parser.add_argument("--start", default="2026-04-14", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-10-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital in USDT")
    parser.add_argument("--rebalance-freq", type=int, default=8, help="Rebalance interval in hours")
    args = parser.parse_args()

    print("=" * 80)
    print(" CRYPTO ALPHA ENGINE: 6-MONTH INSTITUTIONAL BACKTEST")
    print(" ANOMALY STRATEGIES & ILLIQUID TAIL BENCHMARK")
    print("=" * 80)
    print(f" Horizon: {args.start} to {args.end} (6 months prior to today 2026-10-14)")
    print(f" Capital: ${args.capital:,.2f} USDT | VIP Maker Execution (0.02% fee)")
    print(f" Universe: {len(DEFAULT_UNIVERSE)} assets (Core + Illiquid Tail + High-Vol Anomaly Movers)")
    print("-" * 80)

    bars_dict, funding_dict = load_market_data(DEFAULT_UNIVERSE)
    print(f" Loaded {len(bars_dict)} / {len(DEFAULT_UNIVERSE)} assets with complete market data.")

    # 1. Run Baseline (Meta-Strategy without Anomaly Overlays)
    print("\n[1/3] Running Meta-Strategy BASELINE (No Anomaly Overlays)...")
    _, daily_base, kpi_base, _ = run_simulation(
        symbols=DEFAULT_UNIVERSE,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        enable_anomalies=False,
        rebalance_freq_bars=args.rebalance_freq,
    )
    print(f"   Baseline: Equity ${kpi_base['final_equity']:,.2f} | Return: {kpi_base['total_return_pct']:+.2f}% | Sharpe: {kpi_base['sharpe_ratio']:.2f} | MDD: {kpi_base['max_drawdown_pct']:.2f}%")

    # 2. Run Test Mode ($15 Anomaly Sizing)
    print("\n[2/3] Running Meta-Strategy + ANOMALIES (Test Mode: $15 USDT max notional)...")
    _, daily_test, kpi_test, counts_test = run_simulation(
        symbols=DEFAULT_UNIVERSE,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        enable_anomalies=True,
        anomaly_size=15.0,
        rebalance_freq_bars=args.rebalance_freq,
    )
    print(f"   Test Mode ($15): Equity ${kpi_test['final_equity']:,.2f} | Return: {kpi_test['total_return_pct']:+.2f}% | Sharpe: {kpi_test['sharpe_ratio']:.2f} | MDD: {kpi_test['max_drawdown_pct']:.2f}%")
    print(f"   Signals Triggered: Bounce={counts_test['oversold_bounce']}, Breakout={counts_test['early_breakout']}, Fade={counts_test['pump_funding_fade']}")

    # 3. Run Production Mode ($100 Anomaly Sizing)
    print("\n[3/3] Running Meta-Strategy + ANOMALIES (Production Mode: $100 USDT max notional)...")
    _, daily_prod, kpi_prod, counts_prod = run_simulation(
        symbols=DEFAULT_UNIVERSE,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        enable_anomalies=True,
        anomaly_size=100.0,
        rebalance_freq_bars=args.rebalance_freq,
    )
    print(f"   Prod Mode ($100): Equity ${kpi_prod['final_equity']:,.2f} | Return: {kpi_prod['total_return_pct']:+.2f}% | Sharpe: {kpi_prod['sharpe_ratio']:.2f} | MDD: {kpi_prod['max_drawdown_pct']:.2f}%")

    # Export daily CSV
    daily_csv_path = REPORTS_DIR / "backtest_anomaly_illiquid_daily.csv"
    daily_prod.to_csv(daily_csv_path)
    print(f"\n✅ Exported daily simulation breakdown to {daily_csv_path}")

    # Generate Markdown Report
    report_path = REPORTS_DIR / "backtest_anomaly_illiquid_report.md"
    with open(report_path, "w") as f:
        f.write("# 6-Month Backtest Report: Extreme Anomaly & Illiquid Tail Strategies\n\n")
        f.write(f"- **Horizon**: {args.start} to {args.end} (Last 183 days up to today 2026-10-14)\n")
        f.write(f"- **Starting Capital**: ${args.capital:,.2f} USDT\n")
        f.write(f"- **Universe Size**: {len(DEFAULT_UNIVERSE)} assets (Core pairs + Illiquid tail + Anomaly pump/dump movers)\n")
        f.write("- **Execution Model**: Maker Limit Orders (0.02% fee, 0.0% slippage) + 4.0% Turnover Deadband + 8h Funding Settlements\n\n")

        f.write("## 1. Executive Performance Comparison\n\n")
        f.write("| KPI Metric | Baseline (Core Meta) | Anomaly Test Mode ($15 Sizing) | Anomaly Prod Mode ($100 Sizing) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Final Equity** | ${kpi_base['final_equity']:,.2f} | ${kpi_test['final_equity']:,.2f} | **${kpi_prod['final_equity']:,.2f}** |\n")
        f.write(f"| **Total Net Return** | {kpi_base['total_return_pct']:+.2f}% | {kpi_test['total_return_pct']:+.2f}% | **{kpi_prod['total_return_pct']:+.2f}%** |\n")
        f.write(f"| **Annualized Return (CAGR)** | {kpi_base['cagr_pct']:+.2f}% | {kpi_test['cagr_pct']:+.2f}% | **{kpi_prod['cagr_pct']:+.2f}%** |\n")
        f.write(f"| **Sharpe Ratio** | {kpi_base['sharpe_ratio']:.2f} | {kpi_test['sharpe_ratio']:.2f} | **{kpi_prod['sharpe_ratio']:.2f}** |\n")
        f.write(f"| **Sortino Ratio** | {kpi_base['sortino_ratio']:.2f} | {kpi_test['sortino_ratio']:.2f} | **{kpi_prod['sortino_ratio']:.2f}** |\n")
        f.write(f"| **Max Drawdown (MDD)** | -{kpi_base['max_drawdown_pct']:.2f}% | -{kpi_test['max_drawdown_pct']:.2f}% | -{kpi_prod['max_drawdown_pct']:.2f}% |\n")
        f.write(f"| **Daily Win Rate** | {kpi_base['win_rate_days_pct']:.1f}% ({kpi_base['win_days']}W / {kpi_base['loss_days']}L) | {kpi_test['win_rate_days_pct']:.1f}% ({kpi_test['win_days']}W / {kpi_test['loss_days']}L) | **{kpi_prod['win_rate_days_pct']:.1f}%** ({kpi_prod['win_days']}W / {kpi_prod['loss_days']}L) |\n")
        f.write(f"| **Profit Factor** | {kpi_base['profit_factor']:.2f} | {kpi_test['profit_factor']:.2f} | **{kpi_prod['profit_factor']:.2f}** |\n")
        f.write(f"| **Total Maker Fees Paid** | ${kpi_base['total_fees_paid']:,.2f} | ${kpi_test['total_fees_paid']:,.2f} | ${kpi_prod['total_fees_paid']:,.2f} |\n")
        f.write(f"| **Perpetual Funding PnL** | ${kpi_base['total_funding_pnl']:+,.2f} | ${kpi_test['total_funding_pnl']:+,.2f} | **${kpi_prod['total_funding_pnl']:+,.2f}** |\n\n")

        f.write("## 2. Event-Driven Anomaly Signals Breakdown\n\n")
        f.write(f"- **❄️ Oversold Bounce Signals**: {counts_prod['oversold_bounce']} events captured (violent dumps >15% + RSI < 28)\n")
        f.write(f"- **⚡ Early Breakout Signals**: {counts_prod['early_breakout']} events captured (+8% 4h impulse + 2.5x volume)\n")
        f.write(f"- **🔥 Pump Funding Fade Signals**: {counts_prod['pump_funding_fade']} events captured (+25% 24h pump + funding rate > +0.05%)\n\n")

        f.write("## 3. Daily Capital Curve (Last 15 Days)\n\n")
        f.write("| Date | Start Equity | End Equity | Daily PnL | Daily Return | Cumulative Return | Drawdown |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for dt_idx, r in daily_prod.tail(15).iterrows():
            f.write(f"| {dt_idx} | ${r['start_equity']:,.2f} | ${r['end_equity']:,.2f} | ${r['daily_pnl']:+,.2f} | {r['daily_return_pct']:+.2f}% | {r['cum_return_pct']:+.2f}% | {r['drawdown_pct']:.2f}% |\n")

    print(f"✅ Generated comprehensive Markdown report at {report_path}")


if __name__ == "__main__":
    main()
