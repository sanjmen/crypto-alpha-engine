import numpy as np
import pandas as pd
import pytest

from src.features.derivatives import DerivativesFeatures


def test_open_interest_velocity_and_accel():
    oi = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
    vel = DerivativesFeatures.open_interest_velocity(oi, window=1)
    assert np.isnan(vel.iloc[0])
    assert pytest.approx(vel.iloc[1], 1e-4) == 0.05
    assert pytest.approx(vel.iloc[2], 1e-4) == 0.047619

    acc = DerivativesFeatures.open_interest_acceleration(oi, window=1)
    assert np.isnan(acc.iloc[1])
    assert not np.isnan(acc.iloc[2])


def test_smart_money_divergence():
    top = pd.Series([1.5, 1.8, 1.2])
    retail = pd.Series([1.2, 1.1, 1.4])
    div = DerivativesFeatures.smart_money_divergence(top, retail)
    assert pytest.approx(div.iloc[0], 1e-4) == 0.3
    assert pytest.approx(div.iloc[1], 1e-4) == 0.7
    assert pytest.approx(div.iloc[2], 1e-4) == -0.2


def test_taker_flow_imbalance():
    # Ratio = 1.0 -> 0.0 imbalance
    ratio = pd.Series([1.0, 3.0, 0.333333])
    imb = DerivativesFeatures.taker_flow_imbalance(ratio)
    assert pytest.approx(imb.iloc[0], 1e-4) == 0.0
    assert pytest.approx(imb.iloc[1], 1e-4) == 0.5  # (3 - 1) / (3 + 1) = 2/4 = 0.5
    assert pytest.approx(imb.iloc[2], 1e-2) == -0.5  # (0.333 - 1) / (0.333 + 1) = -0.666 / 1.333 = -0.5


def test_annualized_funding_rate():
    rate_8h = pd.Series([0.0001, 0.0005])  # 0.01%, 0.05%
    ann = DerivativesFeatures.annualized_funding_rate(rate_8h)
    assert pytest.approx(ann.iloc[0], 1e-4) == 0.0001 * 3 * 365  # 0.1095 (10.95%)
    assert pytest.approx(ann.iloc[1], 1e-4) == 0.0005 * 3 * 365  # 0.5475 (54.75%)


def test_compute_all_derivatives_features():
    df = pd.DataFrame({
        "open_interest": [100.0, 110.0, 120.0, 130.0],
        "top_trader_ratio": [1.5, 1.6, 1.7, 1.8],
        "retail_ratio": [1.2, 1.2, 1.1, 1.0],
        "taker_long_short_ratio": [1.2, 1.5, 1.8, 2.0],
        "funding_rate": [0.0001, 0.0001, 0.0002, 0.0003]
    })
    res = DerivativesFeatures.compute_all_derivatives_features(df, window=1)
    assert "oi_velocity" in res.columns
    assert "oi_accel" in res.columns
    assert "smart_money_div" in res.columns
    assert "taker_imbalance" in res.columns
    assert "funding_ann" in res.columns
    assert len(res) == 4
