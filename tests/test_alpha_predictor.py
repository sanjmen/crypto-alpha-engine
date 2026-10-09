import numpy as np
import pandas as pd
import pytest

from src.models.alpha_predictor import CrossSectionalAlphaPredictor
from src.domain.entities import Signal


def test_alpha_predictor_fit_and_predict():
    np.random.seed(42)
    n_samples = 200
    n_assets = 10

    # Synthetic features: 3 quantitative factors
    features = pd.DataFrame({
        "mom_24h": np.random.randn(n_samples),
        "vol_parkinson": np.abs(np.random.randn(n_samples)),
        "smart_money_div": np.random.randn(n_samples),
    })

    # Forward returns correlated with factors + noise
    targets = (
        0.4 * features["mom_24h"]
        - 0.2 * features["vol_parkinson"]
        + 0.3 * features["smart_money_div"]
        + 0.5 * np.random.randn(n_samples)
    )

    predictor = CrossSectionalAlphaPredictor(ridge_alpha=50.0)
    predictor.fit(features, targets)

    assert predictor.ridge_model is not None
    assert predictor.lgb_model is not None

    # Predict on test universe
    test_features = pd.DataFrame({
        "symbol": [f"CRYPTO_{i}USDT" for i in range(n_assets)],
        "mom_24h": np.random.randn(n_assets),
        "vol_parkinson": np.abs(np.random.randn(n_assets)),
        "smart_money_div": np.random.randn(n_assets),
    })

    signals = predictor.predict(test_features)
    assert len(signals) == n_assets
    assert all(isinstance(s, Signal) for s in signals)

    # Dollar neutrality check: sum of signal scores should be approximately 0
    sum_signals = sum(s.score for s in signals)
    assert pytest.approx(sum_signals, abs=1e-5) == 0.0

    # Bounds check: signals in [-1.0, 1.0]
    for s in signals:
        assert -1.0001 <= s.score <= 1.0001
        assert 0.0 <= s.confidence <= 1.0
