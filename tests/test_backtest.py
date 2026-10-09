import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import VectorizedBacktester, BacktestResult


def test_vectorized_backtester_metrics():
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    # 3 crypto assets
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Returns with positive drift
    returns_df = pd.DataFrame(
        np.random.normal(loc=0.0005, scale=0.01, size=(n_bars, 3)),
        index=dates,
        columns=symbols,
    )

    # Dollar-neutral weights
    weights_df = pd.DataFrame(
        {"BTCUSDT": 0.5, "ETHUSDT": -0.5, "SOLUSDT": 0.0},
        index=dates,
    )

    # 8h funding rates (every 8 bars)
    funding_df = pd.DataFrame(0.0, index=dates, columns=symbols)
    funding_df.iloc[::8] = 0.0001  # 0.01% every 8h

    engine = VectorizedBacktester(
        initial_capital=100_000.0,
        taker_fee=0.0004,
        slippage_bps=1.0,
    )

    result = engine.run(returns_df, weights_df, funding_rates_df=funding_df)

    assert isinstance(result, BacktestResult)
    assert len(result.equity_curve) == n_bars
    assert len(result.drawdown_curve) == n_bars
    assert result.max_drawdown_pct >= 0.0
    assert not np.isnan(result.sharpe_ratio)
    assert not np.isnan(result.sortino_ratio)
    assert not np.isnan(result.profit_factor)
    assert 0.0 <= result.win_rate_pct <= 100.0


def test_maker_vs_taker_execution_fee_savings():
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    symbols = ["BTCUSDT", "ETHUSDT"]

    returns_df = pd.DataFrame(0.0001, index=dates, columns=symbols)
    # Frequent rebalance weights
    w_btc = np.sin(np.linspace(0, 10, n_bars)) * 0.4
    weights_df = pd.DataFrame({"BTCUSDT": w_btc, "ETHUSDT": -w_btc}, index=dates)

    taker_engine = VectorizedBacktester(execution_mode="taker")
    maker_engine = VectorizedBacktester(execution_mode="maker")

    res_taker = taker_engine.run(returns_df, weights_df)
    res_maker = maker_engine.run(returns_df, weights_df)

    # Maker execution pays 0.02% vs 0.05% taker -> higher final equity and lower fees
    assert res_maker.total_fees_pct < res_taker.total_fees_pct
    assert res_maker.total_return_pct > res_taker.total_return_pct


def test_turnover_deadband_reduces_trades():
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    symbols = ["BTCUSDT"]

    returns_df = pd.DataFrame(0.0001, index=dates, columns=symbols)
    # Noisy weights wiggling by 1%
    w = 0.5 + np.random.normal(0, 0.015, n_bars)
    weights_df = pd.DataFrame({"BTCUSDT": w}, index=dates)

    no_deadband = VectorizedBacktester(turnover_deadband=0.0)
    with_deadband = VectorizedBacktester(turnover_deadband=0.03)  # 3% buffer

    res_no_db = no_deadband.run(returns_df, weights_df)
    res_with_db = with_deadband.run(returns_df, weights_df)

    # Deadband eliminates micro-trades and cuts turnover significantly
    assert res_with_db.total_trades < res_no_db.total_trades
    assert res_with_db.total_turnover < res_no_db.total_turnover
