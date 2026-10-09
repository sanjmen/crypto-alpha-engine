import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio, Signal
from src.strategies.stat_arb_mean_reversion import StatArbMeanReversionStrategy


def test_stat_arb_mean_reversion_strategy():
    np.random.seed(42)
    strategy = StatArbMeanReversionStrategy(lookback_window=48, hurst_threshold=0.60)

    # Generate synthetic Ornstein-Uhlenbeck mean-reverting series for 3 tokens
    # dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    symbols = ["TOKEN_A", "TOKEN_B", "TOKEN_C"]
    market_data = {}
    volatilities = {}

    for sym in symbols:
        prices = [100.0]
        theta = 0.15
        mu = 100.0
        sigma = 1.0
        for _ in range(60):
            dp = theta * (mu - prices[-1]) + np.random.normal(0, sigma)
            prices.append(prices[-1] + dp)

        # Force last price to be an outlier to trigger entry
        if sym == "TOKEN_A":
            prices[-1] = 110.0  # +10 deviation above mean -> should short
        elif sym == "TOKEN_B":
            prices[-1] = 90.0   # -10 deviation below mean -> should long

        p_arr = np.array(prices)
        df = pd.DataFrame({
            "open": p_arr,
            "high": p_arr * 1.01,
            "low": p_arr * 0.99,
            "close": p_arr,
            "volume": np.ones_like(p_arr) * 1000.0,
        })
        market_data[sym] = df
        volatilities[sym] = 0.40

    signals = strategy.generate_signals(market_data)
    assert len(signals) > 0
    assert all(isinstance(s, Signal) for s in signals)

    # TOKEN_A should have negative score (shorting overbought)
    sig_a = next((s for s in signals if s.symbol == "TOKEN_A"), None)
    sig_b = next((s for s in signals if s.symbol == "TOKEN_B"), None)

    if sig_a:
        assert sig_a.score < 0
    if sig_b:
        assert sig_b.score > 0

    # Allocate positions
    portfolio = Portfolio(cash=100_000.0)
    alloc = strategy.allocate_weights(signals, portfolio, volatilities)
    assert len(alloc) == len(signals)
    assert pytest.approx(sum(alloc.values()), abs=1e-4) == 0.0
