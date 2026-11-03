"""
Unit & Integration Tests for Extreme Anomaly & Illiquid Tail Strategies.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Portfolio
from src.domain.enums import MarketRegime
from src.strategies.extreme_anomaly import (
    EarlyBreakoutStrategy,
    OversoldBounceStrategy,
    PumpFundingFadeStrategy,
    calculate_rsi,
    create_illiquid_early_breakout,
    create_illiquid_funding_fade,
    create_illiquid_oversold_bounce,
    create_liquid_early_breakout,
    create_liquid_funding_fade,
    create_liquid_oversold_bounce,
)
from src.strategies.meta_allocator import MetaStrategyAllocator


@pytest.fixture
def sample_market_data():
    """Generates synthetic 72-hour market data for 3 assets."""
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now, periods=72, freq="1h")

    # 1. Normal asset
    p_normal = 100.0 + np.sin(np.linspace(0, 3, 72)) * 2.0
    v_normal = np.full(72, 1000.0)

    # 2. Oversold dump asset (-20% in last 24h)
    p_dump = np.full(72, 100.0)
    p_dump[-24:] = np.linspace(100.0, 78.0, 24)
    v_dump = np.full(72, 2000.0)

    # 3. Super pump asset (+35% in last 24h)
    p_pump = np.full(72, 10.0)
    p_pump[-24:] = np.linspace(10.0, 13.5, 24)
    v_pump = np.full(72, 5000.0)

    # 4. Breakout asset (+10% in last 4h with 4x volume)
    p_breakout = np.full(72, 50.0)
    p_breakout[-4:] = np.linspace(50.0, 55.5, 4)
    v_breakout = np.full(72, 1000.0)
    v_breakout[-4:] = 4000.0

    market = {
        "BTCUSDT": pd.DataFrame({"open": p_normal, "high": p_normal + 1, "low": p_normal - 1, "close": p_normal, "volume": v_normal}, index=idx),
        "CTSIUSDT": pd.DataFrame({"open": p_dump, "high": p_dump + 0.5, "low": p_dump - 0.5, "close": p_dump, "volume": v_dump}, index=idx),
        "MAGICUSDT": pd.DataFrame({"open": p_pump, "high": p_pump + 0.5, "low": p_pump - 0.5, "close": p_pump, "volume": v_pump}, index=idx),
        "OGNUSDT": pd.DataFrame({"open": p_breakout, "high": p_breakout + 0.5, "low": p_breakout - 0.5, "close": p_breakout, "volume": v_breakout}, index=idx),
    }

    funding = {
        "MAGICUSDT": pd.DataFrame({"funding_rate": np.full(72, 0.0008)}, index=idx),  # +0.08% extreme funding
        "BTCUSDT": pd.DataFrame({"funding_rate": np.full(72, 0.0001)}, index=idx),
    }

    return market, funding


def test_rsi_calculation():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
    rsi = calculate_rsi(prices, period=14)
    assert len(rsi) == len(prices)
    assert rsi.iloc[-1] > 70.0  # Strong uptrend -> High RSI


def test_oversold_bounce_strategy(sample_market_data):
    market, _ = sample_market_data
    strategy = OversoldBounceStrategy(dump_threshold_24h=-0.15, max_position_usd=15.0)

    signals = strategy.generate_signals(market)
    assert len(signals) >= 1
    dump_sig = next(s for s in signals if s.symbol == "CTSIUSDT")
    assert dump_sig.score > 0  # LONG
    assert dump_sig.metadata["stop_loss"] < dump_sig.metadata["entry_price"]
    assert dump_sig.metadata["take_profit"] > dump_sig.metadata["entry_price"]

    portfolio = Portfolio(cash=1000.0, initial_cash=1000.0)
    alloc = strategy.allocate_weights(signals, portfolio, {})
    assert "CTSIUSDT" in alloc
    assert 0 < alloc["CTSIUSDT"] <= 15.0  # Capped at $15 in test mode


def test_early_breakout_strategy(sample_market_data):
    market, _ = sample_market_data
    strategy = EarlyBreakoutStrategy(pump_threshold_4h=0.08, volume_multiplier=2.5, max_position_usd=15.0)

    signals = strategy.generate_signals(market)
    assert any(s.symbol == "OGNUSDT" for s in signals)
    sig = next(s for s in signals if s.symbol == "OGNUSDT")
    assert sig.score == 1.0  # LONG

    portfolio = Portfolio(cash=1000.0, initial_cash=1000.0)
    alloc = strategy.allocate_weights(signals, portfolio, {})
    assert alloc["OGNUSDT"] == 15.0


def test_pump_funding_fade_strategy(sample_market_data):
    market, funding = sample_market_data
    strategy = PumpFundingFadeStrategy(pump_threshold_24h=0.25, min_funding_rate=0.0005, max_position_usd=15.0)

    signals = strategy.generate_signals(market, funding_data=funding)
    assert any(s.symbol == "MAGICUSDT" for s in signals)
    sig = next(s for s in signals if s.symbol == "MAGICUSDT")
    assert sig.score < 0  # SHORT

    portfolio = Portfolio(cash=1000.0, initial_cash=1000.0)
    alloc = strategy.allocate_weights(signals, portfolio, {})
    assert alloc["MAGICUSDT"] == -15.0  # Negative dollar allocation for short


def test_factory_presets():
    # Test mode ($15 USDT)
    s1 = create_liquid_oversold_bounce(test_mode=True)
    s2 = create_illiquid_oversold_bounce(test_mode=True)
    s3 = create_liquid_early_breakout(test_mode=True)
    s4 = create_illiquid_early_breakout(test_mode=True)
    s5 = create_liquid_funding_fade(test_mode=True)
    s6 = create_illiquid_funding_fade(test_mode=True)

    assert s1.max_position_usd == 15.0
    assert s2.max_position_usd == 15.0
    assert s3.max_position_usd == 15.0
    assert s4.max_position_usd == 15.0
    assert s5.max_position_usd == 15.0
    assert s6.max_position_usd == 15.0

    # Production mode ($100 USDT)
    s1_prod = create_liquid_oversold_bounce(test_mode=False)
    assert s1_prod.max_position_usd == 100.0


def test_meta_allocator_integration(sample_market_data):
    market, funding = sample_market_data
    portfolio = Portfolio(cash=1000.0, initial_cash=1000.0)
    volatilities = {sym: 0.03 for sym in market}

    bounce_strat = create_illiquid_oversold_bounce(test_mode=True)
    fade_strat = create_illiquid_funding_fade(test_mode=True)

    allocator = MetaStrategyAllocator(
        oversold_bounce_strategy=bounce_strat,
        funding_fade_strategy=fade_strat,
    )

    net_positions, regime, weights = allocator.generate_portfolio_allocations(
        market_data=market,
        portfolio=portfolio,
        volatilities=volatilities,
        funding_data=funding,
    )

    assert isinstance(net_positions, dict)
    assert isinstance(regime, MarketRegime)
    # The anomaly positions should be netted into the output
    assert "CTSIUSDT" in net_positions or "MAGICUSDT" in net_positions
