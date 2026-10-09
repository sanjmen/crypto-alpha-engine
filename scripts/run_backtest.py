#!/usr/bin/env python3
"""
Institutional Multi-Asset Backtest Runner CLI.
Simulates Momentum, Stat-Arb, Funding Carry, and Meta-Strategy Allocator
over the historical dataset stored on Google Drive (2023–2026).
Enforces realistic taker/maker fees, exchange slippage, and 8h funding cashflows.

Usage:
    python scripts/run_backtest.py --strategy meta --preset top10 --source drive --rebalance-freq 8
    python scripts/run_backtest.py --strategy carry --preset top10 --source drive --rebalance-freq 8
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
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
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy
from src.strategies.meta_allocator import MetaStrategyAllocator
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy

PRESETS = {
    "top5": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    "top10": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT"],
    "top20": [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT",
        "SUIUSDT", "APTUSDT", "DOTUSDT", "ATOMUSDT", "FTMUSDT", "ALGOUSDT", "EGLDUSDT", "KAVAUSDT", "ICPUSDT", "ARBUSDT"
    ],
}


def load_drive_market_data(
    symbols: List[str],
    timeframe: str = "1h",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Downloads requested symbol Parquet files from Google Drive to /tmp in parallel.
    Uses local cache in /tmp to avoid re-downloading existing files.
    Ensures 0 MB of persistent local disk space is consumed in the repo.
    """
    temp_dir = Path("/tmp/crypto_alpha_backtest_cache")
    bars_dir = temp_dir / "bars" / timeframe
    funding_dir = temp_dir / "funding"
    bars_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)

    missing_bars = [s for s in symbols if not (bars_dir / f"{s}.parquet").exists()]
    missing_funding = [s for s in symbols if not (funding_dir / f"{s}.parquet").exists()]

    if missing_bars:
        print(f"📥 Downloading {len(missing_bars)} missing bar files from Google Drive via rclone...")
        pattern = "{" + ",".join([f"{s}.parquet" for s in missing_bars]) + "}"
        subprocess.run(
            [
                "rclone", "copy",
                f"gdrive:trading/crypto-alpha-engine/data/bars/{timeframe}/",
                str(bars_dir),
                "--include", pattern,
                "--transfers", "16",
                "--checkers", "16",
            ],
            check=False,
        )

    if missing_funding:
        print(f"📥 Downloading {len(missing_funding)} missing funding files from Google Drive via rclone...")
        pattern = "{" + ",".join([f"{s}.parquet" for s in missing_funding]) + "}"
        subprocess.run(
            [
                "rclone", "copy",
                "gdrive:trading/crypto-alpha-engine/data/funding/",
                str(funding_dir),
                "--include", pattern,
                "--transfers", "16",
                "--checkers", "16",
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
                # Versatile timestamp parser for float, int, or datetime
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
                    f_df["timestamp"] = pd.to_datetime(f_df["timestamp"], unit="s", utc=True)
                f_df.set_index("timestamp", inplace=True)
            f_df.sort_index(inplace=True)
            funding_dict[sym] = f_df

    return bars_dict, funding_dict


def generate_synthetic_universe(
    symbols: List[str],
    n_bars: int = 2000,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Generates synthetic multi-asset price and funding data for testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    bars_dict: Dict[str, pd.DataFrame] = {}
    funding_dict: Dict[str, pd.DataFrame] = {}

    base_prices = {"BTCUSDT": 45000.0, "ETHUSDT": 2500.0, "SOLUSDT": 100.0, "BNBUSDT": 300.0, "DOGEUSDT": 0.12}

    for sym in symbols:
        p0 = base_prices.get(sym, 50.0)
        daily_drift = np.random.uniform(-0.0001, 0.0004)
        vol = np.random.uniform(0.015, 0.035)
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
    rebalance_freq_bars: int = 8,
    execution_mode: str = "maker",
    turnover_deadband: float = 0.04,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestResult:
    """
    Executes historical portfolio simulation with frequency gating and window slicing.
    """
    valid_symbols = [s for s in bars_dict if not bars_dict[s].empty]
    if not valid_symbols:
        raise ValueError("No valid price data available for backtest.")

    # Find common date index
    common_idx = bars_dict[valid_symbols[0]].index
    for sym in valid_symbols[1:]:
        common_idx = common_idx.intersection(bars_dict[sym].index)

    if start_date:
        common_idx = common_idx[common_idx >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        common_idx = common_idx[common_idx <= pd.Timestamp(end_date, tz="UTC")]

    if len(common_idx) < 100:
        raise ValueError(f"Insufficient aligned bars ({len(common_idx)}) to evaluate backtest.")

    # Price matrix & returns
    close_df = pd.DataFrame({sym: bars_dict[sym].loc[common_idx, "close"] for sym in valid_symbols})
    returns_df = close_df.pct_change().fillna(0.0)

    # Funding settlements matrix (forward-fill 8h settlements across 1h bars)
    funding_dfs = {}
    for sym in valid_symbols:
        if sym in funding_dict and not funding_dict[sym].empty:
            f_series = funding_dict[sym]["funding_rate"]
            f_aligned = f_series.reindex(common_idx, method="ffill").fillna(0.0)
            # Funding is only paid/received on the 8h settlement bar (00:00, 08:00, 16:00)
            is_settlement = common_idx.hour.isin([0, 8, 16])
            funding_dfs[sym] = f_aligned * is_settlement
        else:
            funding_dfs[sym] = pd.Series(0.0, index=common_idx)
    funding_df = pd.DataFrame(funding_dfs)

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

    lookback = 72
    print(f"⚙️  Simulating {strategy_name.upper()} across {len(common_idx)} bars ({common_idx[0].strftime('%Y-%m-%d')} to {common_idx[-1].strftime('%Y-%m-%d')})...")
    print(f"⏱️  Rebalance Frequency: Every {rebalance_freq_bars} bars | Execution: {execution_mode.upper()} | Deadband: {turnover_deadband*100:.1f}%")

    current_weights = {sym: 0.0 for sym in valid_symbols}

    for i in range(lookback, len(common_idx)):
        ts = common_idx[i]

        # Rebalance only at specified bar cadence
        if (i - lookback) % rebalance_freq_bars == 0:
            window_start = max(0, i - 120)
            sub_idx = common_idx[window_start:i + 1]
            sub_market = {sym: bars_dict[sym].loc[sub_idx] for sym in valid_symbols}
            sub_funding = {sym: funding_dict[sym].loc[funding_dict[sym].index <= ts] for sym in valid_symbols if sym in funding_dict}

            # Intraday rolling volatility
            volatilities = {sym: float(returns_df[sym].iloc[max(0, i - 24):i].std()) or 0.02 for sym in valid_symbols}

            if strategy_name == "meta":
                target_dollars, _, _ = strat.generate_portfolio_allocations(
                    market_data=sub_market,
                    portfolio=portfolio,
                    volatilities=volatilities,
                    funding_data=sub_funding,
                )
                total_eq = portfolio.total_equity
                current_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in valid_symbols}
            elif strategy_name == "carry":
                signals = strat.generate_signals(sub_market, funding_data=sub_funding)
                target_dollars = strat.allocate_weights(signals, portfolio, volatilities)
                total_eq = portfolio.total_equity
                current_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in valid_symbols}
            else:
                signals = strat.generate_signals(sub_market)
                target_dollars = strat.allocate_weights(signals, portfolio, volatilities)
                total_eq = portfolio.total_equity
                current_weights = {sym: target_dollars.get(sym, 0.0) / max(total_eq, 1.0) for sym in valid_symbols}

        weights_history.append((ts, current_weights))

    # Build weights DataFrame
    weight_records = {ts: w for ts, w in weights_history}
    weights_df = pd.DataFrame.from_dict(weight_records, orient="index").reindex(common_idx).fillna(0.0)

    # Run backtester with Execution Alpha
    backtester = VectorizedBacktester(
        initial_capital=initial_capital,
        taker_fee=0.0004,
        maker_fee=0.0002,
        slippage_bps=1.0,
        execution_mode=execution_mode,
        turnover_deadband=turnover_deadband,
    )
    return backtester.run(returns_df=returns_df, weights_df=weights_df, funding_rates_df=funding_df)


def generate_plotly_tearsheet(result: BacktestResult, strategy_name: str, preset: str, output_path: Path) -> None:
    """Generates an institutional interactive HTML report with Plotly."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"Institutional Cumulative Equity Curve — {strategy_name.upper()} ({preset.upper()})",
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
            line=dict(color="#00E676", width=2),
            hovertemplate="<b>Date:</b> %{x}<br><b>Equity:</b> $%{y:,.2f}<extra></extra>",
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
            line=dict(color="#FF1744", width=1.5),
            fillcolor="rgba(255, 23, 68, 0.2)",
            hovertemplate="<b>Date:</b> %{x}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title_text=(
            f"<b>Crypto Alpha Engine: {strategy_name.upper()} Tearsheet</b> | "
            f"Sharpe: {result.sharpe_ratio:.2f} | "
            f"Net Return: {result.total_return_pct:+.2f}% | "
            f"Max DD: {result.max_drawdown_pct:.2f}%"
        ),
        height=800,
        showlegend=True,
        margin=dict(l=60, r=40, t=90, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"📊 Interactive Plotly tear sheet exported to: {output_path}")


def print_performance_table(result: BacktestResult, strategy_name: str, preset: str, execution_mode: str, deadband: float) -> None:
    """Prints institutional summary table with fee breakdown."""
    print("\n" + "=" * 70)
    print(f"  INSTITUTIONAL PERFORMANCE REPORT: {strategy_name.upper()} ({preset.upper()})")
    print(f"  Execution: {execution_mode.upper()} | Turnover Deadband Buffer: {deadband*100:.1f}%")
    print("=" * 70)
    print(f"  Gross Alpha Return:  {result.gross_return_pct:+8.2f}%")
    print(f"  Total Turnover:      {result.total_turnover:8.2f}x portfolio")
    print(f"  Total Fees/Slippage: {result.total_fees_pct:8.2f}%")
    print(f"  Funding Cashflow PnL:{result.total_funding_pnl_pct:+8.2f}%")
    print("-" * 70)
    print(f"  NET TOTAL RETURN:    {result.total_return_pct:+8.2f}%")
    print(f"  CAGR (Annualized):   {result.cagr_pct:+8.2f}%")
    print(f"  Annualized Vol:      {result.annualized_vol_pct:8.2f}%")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:8.2f}")
    print(f"  Sortino Ratio:       {result.sortino_ratio:8.2f}")
    print(f"  Calmar Ratio:        {result.calmar_ratio:8.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:8.2f}%")
    print(f"  Profit Factor:       {result.profit_factor:8.2f}")
    print(f"  Win Rate:            {result.win_rate_pct:8.2f}%")
    print(f"  Active Trades:       {result.total_trades:8d}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Institutional Crypto Strategy Backtester")
    parser.add_argument("--strategy", choices=["momentum", "stat_arb", "carry", "meta"], default="meta", help="Strategy to evaluate.")
    parser.add_argument("--preset", choices=["top5", "top10", "top20"], default="top10", help="Asset universe preset.")
    parser.add_argument("--source", choices=["drive", "synthetic"], default="drive", help="Data source.")
    parser.add_argument("--rebalance-freq", type=int, default=8, help="Rebalancing frequency in bars (e.g. 8 for 8-hour settlements).")
    parser.add_argument("--execution-mode", choices=["taker", "maker", "hybrid"], default="maker", help="Execution mode (maker=0.02%% limit, taker=0.04%% market+slip, hybrid).")
    parser.add_argument("--turnover-deadband", type=float, default=0.04, help="Turnover deadband threshold (e.g. 0.04 ignores weight changes < 4%%).")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial portfolio capital.")
    parser.add_argument("--start", type=str, default=None, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--report", default=None, help="Output path for HTML report.")
    args = parser.parse_args()

    symbols = PRESETS[args.preset]
    report_file = Path(args.report) if args.report else Path(f"reports/tearsheet_{args.strategy}_{args.preset}_{args.execution_mode}.html")

    print("=" * 70)
    print("  CRYPTO ALPHA ENGINE: INSTITUTIONAL BACKTEST RUNNER")
    print("=" * 70)
    print(f"  Strategy:         {args.strategy.upper()}")
    print(f"  Universe:         {args.preset.upper()} ({len(symbols)} symbols: {', '.join(symbols[:5])}...)")
    print(f"  Source:           {args.source.upper()}")
    print(f"  Execution Mode:   {args.execution_mode.upper()} ({'0.02% passive' if args.execution_mode == 'maker' else '0.04% + 1bp taker'})")
    print(f"  Turnover Deadband:{args.turnover_deadband*100:.1f}%")
    print(f"  Initial Capital:  ${args.capital:,.2f}")
    print(f"  Rebalance Freq:   Every {args.rebalance_freq} bars")
    print("-" * 70)

    if args.source == "drive":
        bars_dict, funding_dict = load_drive_market_data(symbols=symbols)
        if not bars_dict or len(bars_dict) < len(symbols):
            print("⚠️  Missing data from Drive. Supplementing with synthetic simulation.")
            bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)
    else:
        bars_dict, funding_dict = generate_synthetic_universe(symbols=symbols)

    result = run_simulation(
        strategy_name=args.strategy,
        bars_dict=bars_dict,
        funding_dict=funding_dict,
        initial_capital=args.capital,
        rebalance_freq_bars=args.rebalance_freq,
        execution_mode=args.execution_mode,
        turnover_deadband=args.turnover_deadband,
        start_date=args.start,
        end_date=args.end,
    )

    print_performance_table(result, args.strategy, args.preset, args.execution_mode, args.turnover_deadband)
    generate_plotly_tearsheet(result, args.strategy, args.preset, report_file)


if __name__ == "__main__":
    main()
