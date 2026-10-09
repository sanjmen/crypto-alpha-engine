import numpy as np
import pytest

from src.features.volatility import (
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    OnlineEWMAVolatility,
    OnlineGARCH11,
)


def test_volatility_estimators_basic():
    np.random.seed(42)
    n = 100
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.005, n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.005, n)))
    opens = prices * (1.0 + np.random.normal(0, 0.002, n))
    closes = prices

    rv = realized_volatility(closes)
    pv = parkinson_volatility(highs, lows)
    gk = garman_klass_volatility(opens, highs, lows, closes)
    rs = rogers_satchell_volatility(opens, highs, lows, closes)

    assert rv > 0.0
    assert pv > 0.0
    assert gk > 0.0
    assert rs > 0.0


def test_online_ewma_and_garch():
    ewma = OnlineEWMAVolatility(decay=0.94, initial_vol=0.02)
    v1 = ewma.update(0.01)
    v2 = ewma.update(0.03)
    assert v2 > 0.0

    garch = OnlineGARCH11(omega=1e-6, alpha=0.08, beta=0.90, initial_vol=0.02)
    g1 = garch.update(0.01)
    assert g1 > 0.0
    cum_var = garch.forecast_variance(horizon_steps=12)
    assert cum_var > 0.0
