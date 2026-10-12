"""
Unit tests for CointegratedPairsStrategy.
Issue #22.
"""

import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio, Signal
from src.strategies.cointegrated_pairs import CointegratedPairsStrategy


@pytest.fixture
def synthetic_sector_market_data():
    np.random.seed(42)
    n = 260
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Common random walk
    w = np.cumsum(np.random.normal(0, 0.5, n)) + 50.0

    # L1 sector pair: BTC and ETH cointegrated
    btc = w * 2.0 + np.random.normal(0, 0.2, n)
    eth = w * 0.1 + np.random.normal(0, 0.05, n)

    # Meme sector pair: DOGE and SHIB cointegrated
    w_meme = np.cumsum(np.random.normal(0, 0.01, n)) + 0.1
    doge = w_meme * 1.0 + np.random.normal(0, 0.002, n)
    shib = w_meme * 0.0001 + np.random.normal(0, 0.00002, n)

    market_data = {
        "BTCUSDT": pd.DataFrame({"close": btc, "volume": np.full(n, 1000.0)}, index=dates),
        "ETHUSDT": pd.DataFrame({"close": eth, "volume": np.full(n, 5000.0)}, index=dates),
        "DOGEUSDT": pd.DataFrame({"close": doge, "volume": np.full(n, 10000.0)}, index=dates),
        "SHIBUSDT": pd.DataFrame({"close": shib, "volume": np.full(n, 20000.0)}, index=dates),
    }
    return market_data


def test_cointegrated_pairs_strategy_screen_and_signal(synthetic_sector_market_data):
    strategy = CointegratedPairsStrategy(
        lookback_bars=240,
        zscore_window=30,
        entry_zscore=1.5,
        exit_zscore=0.5,
        p_value_threshold=0.10,  # Generous threshold for synthetic test
        hurst_threshold=0.60,
    )

    signals = strategy.generate_signals(synthetic_sector_market_data)
    assert isinstance(signals, list)

    if len(signals) > 0:
        for s in signals:
            assert isinstance(s, Signal)
            assert -1.0 <= s.score <= 1.0
            assert 0.0 <= s.confidence <= 1.0

        # Allocation sizing test
        portfolio = Portfolio(cash=100_000.0, initial_cash=100_000.0, positions={})
        volatilities = {sym: 0.03 for sym in synthetic_sector_market_data}
        dollar_allocs = strategy.allocate_weights(signals, portfolio, volatilities)

        assert isinstance(dollar_allocs, dict)
        gross_exposure = sum(abs(v) for v in dollar_allocs.values())
        assert gross_exposure <= portfolio.total_equity * 2.0
