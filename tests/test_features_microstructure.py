import numpy as np
import pytest

from src.features.microstructure import corwin_schultz_spread, roll_effective_spread, amihud_illiquidity


def test_corwin_schultz_spread():
    np.random.seed(42)
    n = 50
    highs = 100.0 + np.random.uniform(1.0, 3.0, n)
    lows = 100.0 - np.random.uniform(1.0, 3.0, n)

    spreads = corwin_schultz_spread(highs, lows)
    assert len(spreads) == n
    assert np.all(spreads >= 0.0)


def test_roll_and_amihud():
    np.random.seed(42)
    prices = [100.0, 99.5, 100.2, 99.8, 100.1, 99.9]
    roll_s = roll_effective_spread(prices)
    assert roll_s >= 0.0

    rets = np.array([0.01, -0.02, 0.015, -0.005])
    volumes = np.array([1000.0, 1500.0, 800.0, 1200.0])
    illiq = amihud_illiquidity(rets, volumes)
    assert illiq > 0.0
