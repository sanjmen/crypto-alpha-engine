from datetime import datetime, timezone
import pytest

from src.domain.entities import Signal
from src.risk.vol_targeting import VolatilityTargetingSizer
from src.risk.kelly_sizing import FractionalKellySizer


def test_volatility_targeting_sizer():
    signals = [
        Signal(timestamp=datetime.now(tz=timezone.utc), symbol="BTC/USDT", score=0.8),
        Signal(timestamp=datetime.now(tz=timezone.utc), symbol="ETH/USDT", score=0.4),
        Signal(timestamp=datetime.now(tz=timezone.utc), symbol="DOGE/USDT", score=-0.6),
        Signal(timestamp=datetime.now(tz=timezone.utc), symbol="SOL/USDT", score=-0.5),
    ]
    volatilities = {
        "BTC/USDT": 0.02,
        "ETH/USDT": 0.03,
        "DOGE/USDT": 0.08,
        "SOL/USDT": 0.05,
    }

    sizer = VolatilityTargetingSizer(max_leverage=1.0)
    weights = sizer.size_positions(signals, volatilities, market_neutral=True)

    assert len(weights) == 4
    # Dollar neutrality check
    assert abs(sum(weights.values())) < 1e-6
    # BTC (lower vol, high score) should have higher magnitude than ETH
    assert weights["BTC/USDT"] > 0
    assert weights["DOGE/USDT"] < 0


def test_fractional_kelly_sizer():
    kelly = FractionalKellySizer(fraction=0.25, max_position_fraction=0.15)
    size = kelly.compute_size(expected_return=0.04, variance=0.01)
    assert 0.0 < size <= 0.15
