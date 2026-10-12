"""
Unit tests for Engle-Granger and Johansen Cointegration Tests & OU Calibration.
Issue #22.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.cointegration import (
    CointegrationResult,
    EngleGrangerTester,
    JohansenBasketTester,
    OrnsteinUhlenbeckCalibrator,
    SpreadZScoreEngine,
)


@pytest.fixture
def synthetic_cointegrated_data():
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Common random walk
    w = np.cumsum(np.random.normal(0, 1.0, n)) + 100.0

    # Asset X and Asset Y cointegrated: Y = 10.0 + 1.8 * X + stationary noise
    x = w + np.random.normal(0, 0.5, n)
    # Mean-reverting noise (AR(1) with phi = 0.5)
    eta = np.zeros(n)
    for i in range(1, n):
        eta[i] = 0.5 * eta[i - 1] + np.random.normal(0, 0.4)
    y = 10.0 + 1.8 * x + eta

    # Asset Z: independent random walk (NOT cointegrated with X or Y)
    z = np.cumsum(np.random.normal(0, 1.0, n)) + 50.0

    df = pd.DataFrame({"Y": y, "X": x, "Z": z}, index=dates)
    return df


def test_ornstein_uhlenbeck_calibration():
    np.random.seed(42)
    n = 300
    # Simulate known OU process with theta = 0.1, mu = 5.0, sigma = 0.5
    spread = np.zeros(n)
    spread[0] = 5.0
    theta_true = 0.1
    mu_true = 5.0
    sigma_true = 0.5

    for t in range(1, n):
        spread[t] = spread[t - 1] + theta_true * (mu_true - spread[t - 1]) + np.random.normal(0, sigma_true)

    theta, mu, _sigma, half_life = OrnsteinUhlenbeckCalibrator.calibrate(spread)

    assert theta > 0.0
    assert pytest.approx(theta, abs=0.05) == theta_true
    assert pytest.approx(mu, abs=1.0) == mu_true
    assert half_life > 0.0
    assert half_life < float("inf")


def test_engle_granger_tester_cointegrated_vs_independent(synthetic_cointegrated_data):
    df = synthetic_cointegrated_data
    tester = EngleGrangerTester(p_value_threshold=0.05, hurst_threshold=0.55)

    # 1. Cointegrated pair (Y, X)
    res_coint = tester.test_pair(df["Y"], df["X"], y_symbol="Y", x_symbol="X")
    assert isinstance(res_coint, CointegrationResult)
    assert res_coint.is_cointegrated
    assert res_coint.p_value < 0.05
    assert pytest.approx(abs(res_coint.hedge_ratios["X"]), abs=0.2) == 1.8
    assert res_coint.half_life_bars > 0.0
    assert res_coint.half_life_bars < 72.0

    # 2. Independent pair (Y, Z)
    res_indep = tester.test_pair(df["Y"], df["Z"], y_symbol="Y", x_symbol="Z")
    assert not res_indep.is_tradable
    assert res_indep.p_value > 0.05 or res_indep.hurst_exponent >= 0.50


def test_johansen_basket_tester(synthetic_cointegrated_data):
    df = synthetic_cointegrated_data
    tester = JohansenBasketTester(significance_level=1)

    # Basket of cointegrated assets Y and X
    res_basket = tester.test_basket(df[["Y", "X"]])
    assert isinstance(res_basket, CointegrationResult)
    assert res_basket.is_cointegrated
    assert res_basket.test_statistic > res_basket.critical_value_95
    assert len(res_basket.hedge_ratios) == 2


def test_spread_zscore_engine():
    engine = SpreadZScoreEngine(lookback_window=30, entry_zscore=2.0, exit_zscore=0.5)

    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    prices = pd.DataFrame(
        {"Y": np.linspace(100, 110, 100), "X": np.linspace(50, 55, 100)},
        index=dates,
    )
    hedge_ratios = {"Y": 1.0, "X": -2.0}

    spread = engine.compute_spread(prices, hedge_ratios, intercept=0.0)
    zscore, _r_mean, _r_std = engine.compute_rolling_zscore(spread)

    assert len(spread) == 100
    assert len(zscore) == 100
    assert not zscore.isna().any()
