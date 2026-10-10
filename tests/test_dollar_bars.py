"""
Unit Tests for Dollar Bars, Dollar Imbalance Bars, and Statistical Validation.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.features.dollar_bars import (
    DollarBarSynthesizer,
    DollarImbalanceBarSynthesizer,
    StatisticalValidator,
)


@pytest.fixture
def sample_ticks():
    """Generates synthetic trade ticks."""
    np.random.seed(42)
    n_ticks = 1000
    base_price = 90_000.0
    returns = np.random.normal(0.00001, 0.0005, n_ticks)
    prices = base_price * np.exp(np.cumsum(returns))
    quantities = np.random.uniform(0.01, 0.2, n_ticks)
    is_buyer_maker = np.random.choice([True, False], size=n_ticks, p=[0.48, 0.52])

    timestamps = pd.date_range("2024-01-01 12:00:00", periods=n_ticks, freq="1s", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": quantities,
        "notional": prices * quantities,
        "is_buyer_maker": is_buyer_maker,
    })


def test_dollar_bar_synthesizer(sample_ticks):
    """Verifies that Dollar Bars close when cumulative notional breaches threshold."""
    # Total ticks notional is ~1000 * 90000 * 0.1 = ~$9,000,000
    threshold = 500_000.0  # Should generate ~18 bars
    synth = DollarBarSynthesizer(dollar_threshold=threshold, carry_remainder=True)

    bars_df = synth.process_ticks(sample_ticks)
    assert not bars_df.empty
    assert len(bars_df) >= 15

    # Check OHLC properties
    for idx, row in bars_df.iterrows():
        assert row["high"] >= row["low"]
        assert row["high"] >= row["open"]
        assert row["high"] >= row["close"]
        assert row["low"] <= row["open"]
        assert row["low"] <= row["close"]
        assert row["notional"] >= threshold
        assert row["volume"] > 0
        assert row["ticks_count"] > 0
        assert -1.0 <= row["order_flow_imbalance"] <= 1.0


def test_dollar_imbalance_bars(sample_ticks):
    """Verifies adaptive Dollar Imbalance Bar closing and sign accumulation."""
    threshold = 100_000.0
    dib_synth = DollarImbalanceBarSynthesizer(initial_threshold=threshold, ewma_alpha=0.1)

    dib_bars = dib_synth.process_ticks(sample_ticks)
    assert not dib_bars.empty
    for idx, row in dib_bars.iterrows():
        assert abs(row["signed_imbalance"]) >= threshold * 0.5
        assert row["imbalance_side"] in ("BUY", "SELL")
        assert row["high"] >= row["low"]


def test_statistical_validator_normality():
    """Verifies Jarque-Bera test and comparison utilities."""
    np.random.seed(42)
    # 1. Perfectly Gaussian returns
    gaussian_returns = pd.Series(np.random.normal(0, 0.01, 1000))
    stats_gauss = StatisticalValidator.test_normality_jarque_bera(gaussian_returns)
    assert stats_gauss["jb_stat"] < 20.0
    assert abs(stats_gauss["skew"]) < 0.2
    assert abs(stats_gauss["excess_kurtosis"]) < 0.4

    # 2. Heavy-tailed Student-t returns
    heavy_tail_returns = pd.Series(np.random.standard_t(df=3, size=1000) * 0.01)
    stats_heavy = StatisticalValidator.test_normality_jarque_bera(heavy_tail_returns)
    # Should reject normality with massive JB stat
    assert stats_heavy["jb_stat"] > 50.0
    assert stats_heavy["excess_kurtosis"] > 1.0


def test_compare_time_vs_dollar_bars():
    """Verifies comparative benchmarking function."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    time_df = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.02, 100)))}, index=dates)
    dollar_df = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))}, index=dates)

    comp = StatisticalValidator.compare_time_vs_dollar_bars(time_df, dollar_df)
    assert "Time Bars" in comp
    assert "Dollar Bars" in comp
    assert "jb_stat" in comp["Time Bars"]
    assert "excess_kurtosis" in comp["Dollar Bars"]
