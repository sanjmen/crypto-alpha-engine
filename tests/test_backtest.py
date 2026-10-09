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
