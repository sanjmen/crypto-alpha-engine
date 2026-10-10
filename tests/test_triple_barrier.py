"""
Unit tests for Marcos López de Prado's Triple Barrier Method & Sample Weighting Engine.
AFML Chapters 3 & 4.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.triple_barrier import TripleBarrierConfig, TripleBarrierLabeler


@pytest.fixture
def synthetic_ohlcv():
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    # Base price path with known movements
    close_vals = np.full(100, 100.0)
    # Event 1 (idx 10): rallies to 110 at idx 15
    close_vals[11:16] = [102.0, 104.0, 106.0, 108.0, 110.0]
    close_vals[16:30] = 105.0
    # Event 2 (idx 30): crashes to 90 at idx 35
    close_vals[31:36] = [98.0, 96.0, 94.0, 92.0, 90.0]
    close_vals[36:50] = 95.0
    # Event 3 (idx 60): stays flat at 100 until idx 90
    close_vals[60:90] = 100.0

    close = pd.Series(close_vals, index=dates)
    high = close + 0.5
    low = close - 0.5
    vol = pd.Series(0.04, index=dates)  # 4% fixed volatility

    return close, high, low, vol, dates


def test_triple_barrier_long_take_profit(synthetic_ohlcv):
    close, high, low, vol, dates = synthetic_ohlcv
    # Event at index 10: Long position. pt = 1.5 * 0.04 = +6% (price 106.0)
    events = pd.DataFrame({"side": [1.0]}, index=[dates[10]])

    config = TripleBarrierConfig(pt=1.5, sl=1.0, holding_period_bars=20)
    labeler = TripleBarrierLabeler(config)

    barriers = labeler.apply_barriers(close=close, events=events, high=high, low=low, volatility=vol)

    assert len(barriers) == 1
    row = barriers.iloc[0]
    assert row["touch_type"] == "pt"
    assert row["label"] == 1
    assert row["ret"] > 0
    # High touches 106+ at idx 13 (price 106.0, high 106.5)
    assert row["t1"] <= dates[15]


def test_triple_barrier_long_stop_loss(synthetic_ohlcv):
    close, high, low, vol, dates = synthetic_ohlcv
    # Event at index 30: Long position. sl = 1.0 * 0.04 = -4% (price 96.0)
    # Price crashes to 90 at idx 35, hitting SL
    events = pd.DataFrame({"side": [1.0]}, index=[dates[30]])

    config = TripleBarrierConfig(pt=1.5, sl=1.0, holding_period_bars=20)
    labeler = TripleBarrierLabeler(config)

    barriers = labeler.apply_barriers(close=close, events=events, high=high, low=low, volatility=vol)

    assert len(barriers) == 1
    row = barriers.iloc[0]
    assert row["touch_type"] == "sl"
    assert row["label"] == 0
    assert row["ret"] < 0


def test_triple_barrier_short_profitable(synthetic_ohlcv):
    close, high, low, vol, dates = synthetic_ohlcv
    # Event at index 30: Short position. Price drops from 105 to 90 -> Profit for short!
    events = pd.DataFrame({"side": [-1.0]}, index=[dates[30]])

    config = TripleBarrierConfig(pt=1.5, sl=1.0, holding_period_bars=20)
    labeler = TripleBarrierLabeler(config)

    barriers = labeler.apply_barriers(close=close, events=events, high=high, low=low, volatility=vol)

    assert len(barriers) == 1
    row = barriers.iloc[0]
    assert row["touch_type"] == "pt"
    assert row["label"] == 1
    assert row["ret"] > 0


def test_triple_barrier_vertical_timeout(synthetic_ohlcv):
    close, high, low, vol, dates = synthetic_ohlcv
    # Event at index 60: Flat market for 15 bars. Holding period = 10 bars
    events = pd.DataFrame({"side": [1.0]}, index=[dates[60]])

    config = TripleBarrierConfig(pt=2.0, sl=2.0, holding_period_bars=10, min_ret=0.05)
    labeler = TripleBarrierLabeler(config)

    barriers = labeler.apply_barriers(close=close, events=events, high=high, low=low, volatility=vol)

    assert len(barriers) == 1
    row = barriers.iloc[0]
    assert row["touch_type"] == "vertical"
    assert row["t1"] == dates[70]


def test_sample_uniqueness_and_weights(synthetic_ohlcv):
    _, _, _, _, dates = synthetic_ohlcv
    # 3 overlapping events
    events = pd.DataFrame(
        {
            "side": [1.0, 1.0, 1.0],
            "t1": [dates[15], dates[18], dates[20]],
            "ret": [0.05, 0.03, 0.08],
        },
        index=[dates[10], dates[12], dates[14]],
    )

    uniqueness = TripleBarrierLabeler.compute_sample_uniqueness(events, dates)
    assert len(uniqueness) == 3
    # Uniqueness should be between 0 and 1
    for u in uniqueness:
        assert 0.0 < u <= 1.0

    # Earlier event starting at idx 10 has some non-overlapping bars (idx 10, 11)
    # Event starting at idx 14 is heavily overlapped by both earlier events
    assert uniqueness.loc[dates[10]] > uniqueness.loc[dates[14]]

    # Sample weights
    weights = TripleBarrierLabeler.compute_sample_weights(events, dates, attribute_returns=True)
    assert len(weights) == 3
    # Mean weight should be normalized to 1.0
    assert pytest.approx(weights.mean(), abs=1e-5) == 1.0
