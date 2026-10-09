import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio, Signal
from src.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy


def test_cross_sectional_momentum_strategy():
    np.random.seed(42)
    strategy = CrossSectionalMomentumStrategy()

    # Generate synthetic 100-bar OHLCV DataFrames for 5 crypto pairs
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]
    market_data = {}
    volatilities = {}

    for i, sym in enumerate(symbols):
        # Base drift different per asset
        drift = (i - 2) * 0.001
        rets = np.random.normal(loc=drift, scale=0.01, size=100)
        closes = 100.0 * np.exp(np.cumsum(rets))
        highs = closes * (1.0 + np.abs(np.random.normal(0, 0.005, size=100)))
        lows = closes * (1.0 - np.abs(np.random.normal(0, 0.005, size=100)))
        opens = (highs + lows) / 2.0
        volumes = np.random.uniform(1000, 5000, size=100)

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        })
        market_data[sym] = df
        volatilities[sym] = 0.50

    signals = strategy.generate_signals(market_data)
    assert len(signals) == len(symbols)
    assert all(isinstance(s, Signal) for s in signals)

    # Dollar-neutral test: sum of signal scores ~ 0
    sum_scores = sum(s.score for s in signals)
    assert pytest.approx(sum_scores, abs=1e-5) == 0.0

    # Test weight allocation via VolTargetingRiskManager
    portfolio = Portfolio(cash=100_000.0)
    weights = strategy.allocate_weights(signals, portfolio, volatilities)
    assert len(weights) == len(symbols)

    # Dollar neutrality check on target dollar allocations
    net_exposure = sum(weights.values())
    assert pytest.approx(net_exposure, abs=1e-4) == 0.0
