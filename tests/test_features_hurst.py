import numpy as np
import pytest

from src.features.hurst import estimate_hurst_variance_time, estimate_hurst_rs, AssetHurstCalibrator


def test_hurst_variance_time_and_rs():
    np.random.seed(42)
    # Random walk: H should be near 0.5
    rw = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 200)))

    h_vt = estimate_hurst_variance_time(rw)
    assert 0.05 <= h_vt <= 0.95

    rets = np.diff(np.log(rw))
    h_rs = estimate_hurst_rs(rets)
    assert 0.05 <= h_rs <= 0.95


def test_asset_hurst_calibrator():
    calibrator = AssetHurstCalibrator(default_hurst=0.50)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 150)))
    h = calibrator.calibrate("BTC/USDT", prices)
    assert 0.05 <= h <= 0.95
    assert calibrator.get_hurst("BTC/USDT") == h

    vol_scaled = calibrator.scale_volatility(base_vol=0.02, horizon_ratio=4.0, symbol="BTC/USDT")
    assert vol_scaled > 0.02
