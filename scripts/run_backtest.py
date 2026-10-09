#!/usr/bin/env python3
"""
Institutional Backtest Runner CLI.
Executes multi-asset simulations for Momentum, Stat-Arb, Funding Carry, or Meta-Strategy Allocator.
Reads data directly from Google Drive (or local cache) and produces performance metrics & Plotly tear sheets.

Usage:
    python scripts/run_backtest.py --strategy meta --preset top5 --source synthetic
    python scripts/run_backtest.py --strategy meta --preset top5 --source drive
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.engine import BacktestResult, VectorizedBacktester
from src.domain.entities import Portfolio
from src.domain.enums import MarketRegime
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy

PRESETS = {
    "top5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    "top10": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"],
}


def load_drive_market_data(
    symbols: List[str],
    timeframe: str = "1h",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Downloads requested symbol Parquet files from Google Drive to /tmp in a single batch.
    Ensures 0 MB of persistent local disk space is consumed.
    """
    temp_dir = Path("/tmp/crypto_alpha_backtest_cache")
    bars_dir = temp_dir / "bars" / timeframe
    funding_dir = temp_dir / "funding"
    bars_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)

    include_pattern = "{" + ",".join([f"{s}.parquet" for s in symbols]) + "}"
    print(f"📥 Streaming {len(symbols)} assets from Google Drive (rclone parallel batch)...")

    # 1. Batch sync klines bars
    subprocess.run(
        [
            "rclone", "copy",
            f"gdrive:trading/crypto-alpha-engine/data/bars/{timeframe}/",
            str(bars_dir),
            "--include", include_pattern,
            "--transfers", "8",
            "--checkers", "8",
        ],
        check=False,
    )

    # 2. Batch sync funding
    subprocess.run(
        [
            "rclone", "copy",
            "gdrive:trading/crypto-alpha-engine/data/funding/",
            str(funding_dir),
            "--include", include_pattern,
            "--transfers", "8",
            "--checkers", "8",
        ],
        check=False,
    )

    bars_dict: Dict[str, pd.DataFrame] = {}
    funding_dict: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        bar_path = bars_dir / f"{sym}.parquet"
        if bar_path.exists():
            df = pd.read_parquet(bar_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df.set_index("timestamp", inplace=True)
            bars_dict[sym] = df

        funding_path = funding_dir / f"{sym}.parquet"
        if funding_path.exists():
            f_df = pd.read_parquet(funding_path)
            if "timestamp" in f_df.columns:
                f_df["timestamp"] = pd.to_datetime(f_df["timestamp"], unit="s", utc=True)
                f_df.set_index("timestamp", inplace=True)
            funding_dict[sym] = f_df

    return bars_dict, funding_dict


def generate_synthetic_universe(
    symbols: List[str],
    n_bars: int = 2000,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Generates synthetic multi-asset price and funding data for offline backtesting."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    bars_dict: Dict[str, pd.DataFrame] = {}
    funding_dict: Dict[str, pd.DataFrame] = {}

    base_prices = {"BTCUSDT": 45000.0, "ETHUSDT": 2500.0, "SOLUSDT": 100.0, "BNBUSDT": 300.0, "DOGEUSDT": 0.12}

    for sym in symbols:
        p0 = base_prices.get(sym, 50.0)
        daily_drift = np.random.uniform(-0.0002, 0.0005)
        vol = np.random.uniform(0.01, 0.03)
        returns = np.random.normal(daily_drift, vol, n_bars)
        price_series = p0 * np.exp(np.cumsum(returns))

        bars_dict[sym] = pd.DataFrame(
            {
                "open": price_series * (1.0 - np.random.uniform(0, 0.002, n_bars)),
                "high": price_series * (1.0 + np.random.uniform(0.001, 0.005, n_bars)),
                "low": price_series * (1.0 - np.random.uniform(0.001, 0.005, n_bars)),
                "close": price_series,
                "volume": np.random.uniform(500, 5000, n_bars) * (p0 / 100.0),
            },
            index=timestamps,
        )

        funding_dict[sym] = pd.DataFrame(
            {"funding_rate": np.random.normal(0.0001, 0.0002, n_bars)},
            index=timestamps,
        )

    return bars_dict, funding_dict


def run_simulation(
    strategy_name: str,
    bars_dict: Dict[str, pd.DataFrame],
    funding_dict: Dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
    rebalance_freq_bars: int = 1,
) -> BacktestResult:
    """
    Simulates portfolio strategy signals and returns.
    """
    symbols = list(bars_dict.keys())
    common_idx = bars_dict[symbols[0]].index
    for sym in symbols[1:]:
        common_idx = common_idx.intersection(bars_dict[sym].index)

    # Price matrix & returns
    close_df = pd.DataFrame({sym: bars_dict[sym].loc[common_idx, "close"] for sym in symbols})
    returns_df = close_df.pct_change().fillna(0.0)

    # Funding settlements matrix
    funding_df = pd.DataFrame({
        sym: funding_dict[sym].loc[common_idx.intersection(funding_dict[sym].index), "funding_rate"]
        for sym in symbols if sym in funding_dict
    }).reindex(index=common_idx).fillna(0.0)

    # Initialize strategy
    if strategy_name == "momentum":
        strat = CrossSectionalMomentumStrategy()
    elif strategy_name == "stat_arb":
        strat = StatArbMeanReversionStrategy()
    elif strategy_name == "carry":
        strat = FundingCarryArbitrageStrategy()
    else:
        strat = MetaStrategyAllocator()

    weights_history = []
    portfolio = Portfolio(cash=initial_capital, initial_cash=initial_capital)

    lookback = 48
    print(f"⚙️  Executing {strategy_name.upper()} simulation across {len(common_idx)} bars...")

    for i in range(lookback, len(common_idx)):
        # Sub-window of data
        sub_idx = common_idx[:i + 1]
        sub_market = {sym: bars_dict[sym].loc[sub_idx] for sym in symbols}
        sub_funding = {sym: funding_dict[sym].loc[funding_dict[sym].index <= sub_idx[-1]] for sym in symbols if sym in funding_dict}

        # Volatilities
        volatilities = {sym: float(returns_df[sym].iloc[max(0, i - 24):i].std()) or 0.02 for sym in symbols}

        if strategy_name == "meta":
            target_dollars, _, _ = strat.generate_portfolio_allocations(
                market_data=sub_market,
                portfolio=portfolio,
                volatilities=volatilities,
                funding_data=sub_funding,
            )
            total_eq = portfolio.total_equity
            target_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in symbols}
        elif strategy_name == "carry":
            signals = strat.generate_signals(sub_market, funding_data=sub_funding)
            target_dollars = strat.allocate_weights(signals, portfolio, volatilities)
            total_eq = portfolio.total_equity
            target_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in symbols}
        else:
            signals = strat.generate_signals(sub_market)
            target_dollars = strat.allocate_weights(signals, portfolio, volatilities)
            total_eq = portfolio.total_equity
            target_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in symbols}

        weights_history.append((common_idx[i], target_weights))

    # Build weights DataFrame
    weight_records = {ts: w for ts, w in weights_history}
    weights_df = pd.DataFrame.from_dict(weight_records, orient="index").reindex(common_idx).fillna(0.0)

    # Run backtester
    backtester = VectorizedBacktester(
        initial_capital=initial_capital,
        taker_fee=0.0004,
        maker_fee=0.0002,
        slippage_bps=1.0,
    )
    return backtester.run(returns_df=returns_df, weights_df=weights_df, funding_rates_df=funding_df)


def generate_plotly_tearsheet(result: BacktestResult, strategy_name: str, output_path: Path) -> None:
    """Generates an institutional interactive HTML report with Plotly."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"Institutional Equity Curve ({strategy_name.upper()})",
            "Underwater Drawdown Profile (%)",
        ],
    )

    # 1. Equity curve
    fig.add_trace(
        go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name="Portfolio Equity ($)",
            line=dict(color="#00C853", width=2),
            hovertemplate="Time: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 2. Drawdown curve
    fig.add_trace(
        go.Scatter(
            x=result.drawdown_curve.index,
            y=result.drawdown_curve.values * 100.0,
            mode="lines",
            fill="tozeroy",
            name="Drawdown (%)",
            line=dict(color="#D50000", width=1.5),
            fillcolor="rgba(213, 0, 0, 0.2)",
            hovertemplate="Time: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title_text=f"<b>Crypto Alpha Engine: {strategy_name.upper()} Tearsheet</b> | Sharpe: {result.sharpe_ratio:.2f} | Max DD: {result.max_drawdown_pct:.2f}%",
        height=750,
        showlegend=True,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"📊 Interactive Plotly tear sheet exported to: {output_path}")


def print_performance_table(result: BacktestResult, strategy_name: str) -> None:
    """Prints institutional summary table."""
    print("\n" + "=" * 70)
    print(f"  INSTITUTIONAL PERFORMANCE REPORT: {strategy_name.upper()}")
    print("=" * 70)
    print(f"  Total Return:        {result.total_return_pct:+8.2f}%")
    print(f"  CAGR (Annualized):   {result.cagr_pct:+8.2f}%")
    print(f"  Annualized Vol:      {result.annualized_vol_pct:8.2f}%")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:8.2f}")
    print(f"  Sortino Ratio:       {result.sortino_ratio:8.2f}")
    print(f"  Calmar Ratio:        {result.calmar_ratio:8.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:8.2f}%")
    print(f"  Profit Factor:       {result.profit_factor:8.2f}")
    print(f"  Win Rate:            {result.win_rate_pct:8.2f}%")
    print(f"  Total Trades:        {result.total_trades:8d}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Institutional Crypto Strategy Backtester")
    parser.add_argument("--strategy", choices=["momentum", "stat_arb", "carry", "meta"], default="meta", help="Strategy to evaluate.")
    parser.add_argument("--preset", choices=["top5", "top10"], default="top5", help="Asset universe preset.")
    parser.add_argument("--source", choices=["drive", "synthetic"], default="synthetic", help="Data source.")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial portfolio capital.")
    parser.add_argument("--report", default="reports/backtest_tearsheet.html", help="Output path for HTML report.")
    args = parser.parse_args()

    symbols = PRESETS[args.preset]

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: INSTITUTIONAL BACKTEST RUNNER")
    print("=" * 70)
    print(f"  Strategy:  {args.strategy.upper()}")
    print(f"  Universe:  {args.preset.upper()} ({', '.join(symbols)})")
    print(f"  Source:    {args.source.upper()}")
    print(f"  Capital:   ${args.capital:,.2f}")
    print("-" * 70)

    if args.source == "drive":
        bars_dict, funding_dict = load_drive_market_data(symbols=symbols)
        if not bars_dict:
            print("⚠️  No data retrieved from Drive. Falling back to synthetic simulation.")
            bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)
    else:
        bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)

    result = run_simulation(
        strategy_name=args.strategy,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        initial_capital=args.capital,
    )

    print_performance_table(result, args.strategy)
    generate_plotly_tearsheet(result, args.strategy, Path(args.report))


if __name__ == "__main__":
    main()
