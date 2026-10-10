"""
Unit tests for Purged and Embargoed Cross-Validation Engine.
AFML Chapter 7.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.purged_cv import (
    PurgedKFoldCV,
    PurgedWalkForwardCV,
    compute_cv_metrics,
    compute_rank_ic,
)


@pytest.fixture
def sample_events_and_data():
    n_samples = 100
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1h")
    X = pd.DataFrame(
        {
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
        },
        index=dates,
    )
    y = pd.Series(np.random.randn(n_samples), index=dates)

    # Let each event have a holding period of 5 bars: t1 = t0 + 5 hours
    t1_list = [dates[min(i + 5, n_samples - 1)] for i in range(n_samples)]
    t1 = pd.Series(t1_list, index=dates)

    return X, y, t1, dates


def test_purged_walk_forward_cv_leakage_prevention(sample_events_and_data):
    X, y, t1, dates = sample_events_and_data

    cv = PurgedWalkForwardCV(n_splits=3, min_train_pct=0.4, embargo_pct=0.02)
    splits = list(cv.split(X, y, t1=t1))

    assert len(splits) == 3

    for fold, (train_idx, test_idx) in enumerate(splits):
        # 1. All train indices must be strictly before test_idx
        assert np.max(train_idx) < np.min(test_idx)

        # 2. Check PURGING: no train sample's t1 can touch or enter test_idx
        test_start_ts = dates[np.min(test_idx)]
        for tr_i in train_idx:
            sample_t1 = t1.iloc[tr_i]
            # Must strictly terminate before the test start timestamp!
            assert sample_t1 < test_start_ts


def test_purged_kfold_cv_purging_and_embargo(sample_events_and_data):
    X, y, t1, dates = sample_events_and_data

    cv = PurgedKFoldCV(n_splits=4, embargo_pct=0.05)  # 5% embargo = 5 bars
    splits = list(cv.split(X, y, t1=t1))

    assert len(splits) == 4

    for fold, (train_idx, test_idx) in enumerate(splits):
        test_start_idx = np.min(test_idx)
        test_end_idx = np.max(test_idx)

        # Check train samples before test
        train_before = train_idx[train_idx < test_start_idx]
        test_start_ts = dates[test_start_idx]
        for tr_i in train_before:
            assert t1.iloc[tr_i] < test_start_ts

        # Check train samples after test (EMBARGO)
        train_after = train_idx[train_idx > test_end_idx]
        if len(train_after) > 0:
            # First sample after test must respect embargo of 5 bars
            min_train_after = np.min(train_after)
            assert min_train_after >= test_end_idx + 5


def test_compute_rank_ic():
    # Perfect monotonic relationship
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert pytest.approx(compute_rank_ic(y_pred, y_true), abs=1e-5) == 1.0

    # Inverted relationship
    y_inv = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    assert pytest.approx(compute_rank_ic(y_inv, y_true), abs=1e-5) == -1.0


def test_compute_cv_metrics():
    fold_preds = [
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 2.2, 2.9, 4.1])),
        (np.array([5.0, 6.0, 7.0, 8.0]), np.array([4.9, 6.1, 6.8, 8.2])),
    ]

    metrics = compute_cv_metrics(fold_preds)
    assert metrics["mean_rank_ic"] > 0.95
    assert metrics["mean_pearson_ic"] > 0.95
    assert metrics["directional_accuracy"] == 1.0
    assert metrics["information_ratio"] > 0.0
