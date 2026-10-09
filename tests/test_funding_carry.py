import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio, Signal
from src.strategies.funding_carry_arbitrage import FundingCarryArbitrageStrategy


def test_funding_carry_arbitrage_strategy():
    strategy = FundingCarryArbitrageStrategy(min_annualized_carry=0.10)

    # 4 crypto assets with diverse funding rates
    # TOKEN_POS: +0.05% / 8h = ~54.7% ann -> should SHORT
    # TOKEN_NEG: -0.03% / 8h = ~-32.8% ann -> should LONG
    # TOKEN_NEUT: +0.001% / 8h = ~1.0% ann -> below threshold
    market_data = {
        "BTCUSDT": pd.DataFrame({"close": [60000.0] * 35, "funding_rate": [0.0001] * 35}),
        "TOKEN_POS": pd.DataFrame({"close": [100.0] * 35, "funding_rate": [0.0005] * 35}),
        "TOKEN_NEG": pd.DataFrame({"close": [50.0] * 35, "funding_rate": [-0.0003] * 35}),
        "TOKEN_NEUT": pd.DataFrame({"close": [10.0] * 35, "funding_rate": [0.00001] * 35}),
    }
    volatilities = {k: 0.40 for k in market_data}

    signals = strategy.generate_signals(market_data)
    assert len(signals) > 0
    assert all(isinstance(s, Signal) for s in signals)

    # TOKEN_POS should be short (negative score)
    sig_pos = next((s for s in signals if s.symbol == "TOKEN_POS"), None)
    # TOKEN_NEG should be long (positive score)
    sig_neg = next((s for s in signals if s.symbol == "TOKEN_NEG"), None)

    assert sig_pos is not None and sig_pos.score < 0
    assert sig_neg is not None and sig_neg.score > 0

    # Dollar neutrality check on allocated positions
    portfolio = Portfolio(cash=100_000.0)
    alloc = strategy.allocate_weights(signals, portfolio, volatilities)
    assert pytest.approx(sum(alloc.values()), abs=1e-4) == 0.0
