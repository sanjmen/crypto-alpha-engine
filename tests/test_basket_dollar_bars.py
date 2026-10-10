"""
Unit Tests for MarketWideBasketDollarBarSynthesizer (Issue #33).
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.features.basket_dollar_bars import MarketWideBasketDollarBarSynthesizer


@pytest.fixture
def multi_asset_bars():
    """Generates synthetic multi-asset OHLCV bars across BTC, ETH, and SOL."""
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    base_prices = {"BTCUSDT": 90_000.0, "ETHUSDT": 3_500.0, "SOLUSDT": 200.0}
    volumes = {"BTCUSDT": 50.0, "ETHUSDT": 500.0, "SOLUSDT": 5000.0}

    bars = {}
    for sym in symbols:
        p0 = base_prices[sym]
        rets = np.random.normal(0, 0.005, n_bars)
        closes = p0 * np.exp(np.cumsum(rets))
        vol = volumes[sym] * np.random.uniform(0.8, 1.5, n_bars)
        notional = closes * vol

        bars[sym] = pd.DataFrame(
            {
                "open": closes * (1 - 0.001),
                "high": closes * (1 + 0.004),
                "low": closes * (1 - 0.004),
                "close": closes,
                "volume": vol,
                "notional": notional,
            },
            index=dates,
        )
    return bars


def test_basket_dollar_bar_synchronization(multi_asset_bars):
    """
    Verifies that Basket Dollar Bars emit identical timestamps for all assets,
    resolving the cross-sectional desynchronization paradox.
    """
    threshold = 100_000_000.0  # $100M market-wide pulse
    synthesizer = MarketWideBasketDollarBarSynthesizer(market_dollar_threshold=threshold)

    market_summary, synced_asset_bars = synthesizer.process_synchronized_bars(multi_asset_bars)

    assert not market_summary.empty
    assert len(market_summary) >= 5
    assert "BTCUSDT" in synced_asset_bars
    assert "ETHUSDT" in synced_asset_bars
    assert "SOLUSDT" in synced_asset_bars

    # Verify that all 3 assets share the exact same timestamp index
    btc_idx = synced_asset_bars["BTCUSDT"].index
    eth_idx = synced_asset_bars["ETHUSDT"].index
    sol_idx = synced_asset_bars["SOLUSDT"].index

    assert len(btc_idx) == len(eth_idx) == len(sol_idx)
    assert (btc_idx == eth_idx).all()
    assert (eth_idx == sol_idx).all()


def test_cross_sectional_matrix_extraction(multi_asset_bars):
    """Verifies wide matrix extraction for factor modeling."""
    synthesizer = MarketWideBasketDollarBarSynthesizer(market_dollar_threshold=50_000_000.0)
    market_summary, synced_asset_bars = synthesizer.process_synchronized_bars(multi_asset_bars)

    close_matrix = synthesizer.extract_cross_sectional_matrix(synced_asset_bars, metric="close")
    return_matrix = synthesizer.extract_cross_sectional_matrix(synced_asset_bars, metric="return")

    assert close_matrix.shape[1] == 3
    assert not close_matrix.isna().any().any()
    assert return_matrix.shape[1] == 3
    assert not return_matrix.isna().any().any()
