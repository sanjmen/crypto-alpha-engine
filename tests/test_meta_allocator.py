"""
Unit tests for MetaStrategyAllocator.
"""

import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio
from src.domain.enums import MarketRegime
from src.strategies.meta_allocator import MetaStrategyAllocator


def create_synthetic_data(n_bars: int = 150) -> dict[str, pd.DataFrame]:
    """Generates synthetic price data for multi-strategy allocation tests."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz="UTC")

    # BTC trending upwards
    btc_close = 40000 + np.cumsum(np.random.randn(n_bars) * 100 + 30)
    btc_df = pd.DataFrame(
        {
            "open": btc_close - 50,
            "high": btc_close + 100,
            "low": btc_close - 100,
            "close": btc_close,
            "volume": np.random.uniform(500, 2000, n_bars),
        },
        index=timestamps,
    )

    # ETH ranging
    eth_close = 2500 + np.sin(np.linspace(0, 10, n_bars)) * 100 + np.random.randn(n_bars) * 10
    eth_df = pd.DataFrame(
        {
            "open": eth_close - 10,
            "high": eth_close + 20,
            "low": eth_close - 20,
            "close": eth_close,
            "volume": np.random.uniform(1000, 5000, n_bars),
        },
        index=timestamps,
    )

    # SOL oscillating with some volatility
    sol_close = 100 + np.cumsum(np.random.randn(n_bars) * 2)
    sol_df = pd.DataFrame(
        {
            "open": sol_close - 1,
            "high": sol_close + 3,
            "low": sol_close - 3,
            "close": sol_close,
            "volume": np.random.uniform(2000, 8000, n_bars),
        },
        index=timestamps,
    )

    return {"BTCUSDT": btc_df, "ETHUSDT": eth_df, "SOLUSDT": sol_df}


def test_compute_regime_weights():
    allocator = MetaStrategyAllocator()

    w_trending = allocator.compute_regime_weights(MarketRegime.TRENDING)
    assert w_trending["momentum"] > w_trending["stat_arb"]
    assert pytest.approx(w_trending["momentum"], 0.01) == 0.60
    assert pytest.approx(w_trending["carry"], 0.01) == 0.30

    w_ranging = allocator.compute_regime_weights(MarketRegime.RANGING)
    assert w_ranging["stat_arb"] > w_ranging["momentum"]
    assert pytest.approx(w_ranging["stat_arb"], 0.01) == 0.55

    w_volatile = allocator.compute_regime_weights(MarketRegime.VOLATILE)
    assert w_volatile["momentum"] <= 0.15
    assert w_volatile["stat_arb"] <= 0.15


def test_detect_market_regime():
    allocator = MetaStrategyAllocator()
    data = create_synthetic_data(100)
    regime = allocator.detect_market_regime(data)
    assert isinstance(regime, MarketRegime)


def test_generate_portfolio_allocations():
    allocator = MetaStrategyAllocator()
    data = create_synthetic_data(120)

    portfolio = Portfolio(
        cash=100_000.0,
        initial_cash=100_000.0,
        positions={},
    )
    volatilities = {"BTCUSDT": 0.02, "ETHUSDT": 0.03, "SOLUSDT": 0.04}

    # Synthetic funding data
    funding_data = {
        sym: pd.DataFrame(
            {"funding_rate": [0.0001] * 20},
            index=pd.date_range("2024-01-01", periods=20, freq="8h", tz="UTC"),
        )
        for sym in data
    }

    net_positions, regime, strat_weights = allocator.generate_portfolio_allocations(
        market_data=data,
        portfolio=portfolio,
        volatilities=volatilities,
        funding_data=funding_data,
    )

    assert isinstance(regime, MarketRegime)
    assert isinstance(strat_weights, dict)
    assert isinstance(net_positions, dict)

    # Net positions should not exceed portfolio total equity in gross notional
    gross_exposure = sum(abs(pos) for pos in net_positions.values())
    assert gross_exposure <= portfolio.total_equity * 1.5


def test_meta_allocator_hrp_strategy_weights():
    allocator = MetaStrategyAllocator()
    np.random.seed(42)
    n_periods = 100

    # 3 strategy return series: momentum and carry have low correlation, stat_arb has high volatility
    strategy_returns = pd.DataFrame({
        "momentum": np.random.normal(0.0005, 0.015, n_periods),
        "stat_arb": np.random.normal(0.0003, 0.030, n_periods),
        "carry": np.random.normal(0.0002, 0.005, n_periods),
    })

    # Pure HRP weights
    hrp_weights = allocator.compute_hrp_strategy_weights(strategy_returns, regime=None)
    assert len(hrp_weights) == 3
    assert pytest.approx(sum(hrp_weights.values()), abs=1e-5) == 1.0
    # Lower volatility carry strategy should receive higher risk-balanced weight than high-vol stat_arb
    assert hrp_weights["carry"] > hrp_weights["stat_arb"]

    # Blended HRP + Regime weights
    blended_weights = allocator.compute_hrp_strategy_weights(
        strategy_returns, regime=MarketRegime.TRENDING, regime_blend_ratio=0.5
    )
    assert pytest.approx(sum(blended_weights.values()), abs=1e-5) == 1.0
    # Trending regime boosts momentum weight
    assert blended_weights["momentum"] > hrp_weights["momentum"]


def test_meta_allocator_generate_portfolio_with_hrp():
    allocator = MetaStrategyAllocator()
    data = create_synthetic_data(120)
    portfolio = Portfolio(cash=100_000.0, initial_cash=100_000.0, positions={})
    volatilities = {"BTCUSDT": 0.02, "ETHUSDT": 0.03, "SOLUSDT": 0.04}

    strategy_returns = pd.DataFrame({
        "momentum": np.random.normal(0.0005, 0.015, 50),
        "stat_arb": np.random.normal(0.0003, 0.025, 50),
        "carry": np.random.normal(0.0002, 0.008, 50),
    })

    net_positions, _regime, strat_weights = allocator.generate_portfolio_allocations(
        market_data=data,
        portfolio=portfolio,
        volatilities=volatilities,
        strategy_returns=strategy_returns,
        use_hrp=True,
    )

    assert isinstance(net_positions, dict)
    assert isinstance(strat_weights, dict)
    assert pytest.approx(sum(strat_weights.values()), abs=1e-4) == 1.0
    assert len(net_positions) > 0
