"""
Unit tests for Secondary Precision Meta-Classifier & Kelly Position Sizing.
AFML Chapter 3.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.domain.entities import Signal
from src.models.meta_labeling import (
    KellyPositionSizer,
    MetaLabelingReport,
    SecondaryMetaClassifier,
)


def test_kelly_position_sizer_filtering_and_scaling():
    sizer = KellyPositionSizer(kelly_fraction=0.5, max_leverage=1.0)

    # 1. Below or equal to 0.5 -> Trade rejected (size = 0.0)
    assert sizer.compute_size(0.50, side=1.0) == 0.0
    assert sizer.compute_size(0.35, side=1.0) == 0.0
    assert sizer.compute_size(0.10, side=-1.0) == 0.0

    # 2. Probability = 0.75 -> raw_kelly = 2*0.75 - 1 = 0.5. Half-Kelly = 0.25
    assert pytest.approx(sizer.compute_size(0.75, side=1.0), abs=1e-5) == 0.25
    assert pytest.approx(sizer.compute_size(0.75, side=-1.0), abs=1e-5) == -0.25

    # 3. Probability = 1.0 -> raw_kelly = 1.0. Half-Kelly = 0.50
    assert pytest.approx(sizer.compute_size(1.0, side=1.0), abs=1e-5) == 0.50


def test_kelly_position_sizer_vectorized():
    sizer = KellyPositionSizer(kelly_fraction=0.5, max_leverage=1.0)
    probs = np.array([0.4, 0.5, 0.6, 0.8, 1.0])
    sides = np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    sizes = sizer.size_batch(probs, sides)
    expected = np.array([0.0, 0.0, 0.1, -0.3, 0.5])
    np.testing.assert_allclose(sizes, expected, atol=1e-5)


def test_secondary_meta_classifier_fit_evaluate_and_size():
    np.random.seed(42)
    n_samples = 300

    # Meta-features: regime, volatility, spread, signal strength
    features = pd.DataFrame({
        "vol_ratio": np.random.uniform(0.5, 2.5, n_samples),
        "hurts_h": np.random.uniform(0.3, 0.7, n_samples),
        "amihud_illiq": np.random.exponential(1.0, n_samples),
        "primary_abs_score": np.random.uniform(0.1, 1.0, n_samples),
    })

    # True success probability increases with signal strength and Hurst persistence
    true_logit = 2.0 * features["primary_abs_score"] + 1.5 * (features["hurts_h"] - 0.5) - 0.5
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    targets = pd.Series(np.random.binomial(1, true_prob, n_samples))
    weights = pd.Series(np.random.uniform(0.8, 1.2, n_samples))

    meta_clf = SecondaryMetaClassifier(kelly_fraction=0.5)
    meta_clf.fit(features.iloc[:200], targets.iloc[:200], sample_weights=weights.iloc[:200])

    assert meta_clf.model is not None
    assert len(meta_clf.feature_names) == 4

    # Evaluate on holdout
    report = meta_clf.evaluate_filter(features.iloc[200:], targets.iloc[200:])
    assert isinstance(report, MetaLabelingReport)
    assert 0.0 <= report.auc_roc <= 1.0
    assert 0.0 <= report.brier_score <= 1.0
    assert 0.0 <= report.trades_executed_ratio <= 1.0

    # Test signal sizing
    now = datetime.now(timezone.utc)
    test_signals = [
        Signal(timestamp=now, symbol="BTCUSDT", score=0.8, confidence=0.8, horizon_minutes=60),
        Signal(timestamp=now, symbol="ETHUSDT", score=-0.6, confidence=0.6, horizon_minutes=60),
    ]

    test_meta_features = pd.DataFrame(
        [
            {"vol_ratio": 1.0, "hurts_h": 0.65, "amihud_illiq": 0.2, "primary_abs_score": 0.8},
            {"vol_ratio": 2.5, "hurts_h": 0.35, "amihud_illiq": 2.0, "primary_abs_score": 0.1},
        ],
        index=["BTCUSDT", "ETHUSDT"],
    )

    sized = meta_clf.size_signals(test_signals, test_meta_features)
    assert len(sized) == 2
    assert sized[0].symbol == "BTCUSDT"
    assert sized[1].symbol == "ETHUSDT"
    # Strong signal in persistent regime should get higher allocation than weak signal in noisy regime
    assert abs(sized[0].score) >= abs(sized[1].score)
