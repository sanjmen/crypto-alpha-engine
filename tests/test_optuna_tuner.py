"""
Unit tests for Optuna Hyperparameter Optimization Pipeline with Purged CV.
Issue #21.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.alpha_predictor import CrossSectionalAlphaPredictor
from src.models.meta_labeling import SecondaryMetaClassifier
from src.models.optuna_tuner import OptunaHyperparameterTuner, TuningResult


@pytest.fixture
def synthetic_tuning_data():
    np.random.seed(42)
    n_samples = 120
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="1h")

    features = pd.DataFrame(
        {
            "mom": np.random.randn(n_samples),
            "vol": np.abs(np.random.randn(n_samples)),
            "smart": np.random.randn(n_samples),
        },
        index=dates,
    )

    # Continuous targets correlated with factors
    targets_cont = (
        0.5 * features["mom"]
        - 0.3 * features["vol"]
        + 0.4 * features["smart"]
        + 0.2 * np.random.randn(n_samples)
    )

    # Binary targets for meta-classification
    targets_bin = pd.Series(
        (targets_cont > targets_cont.median()).astype(int),
        index=dates,
    )

    t1 = pd.Series(
        [dates[min(i + 3, n_samples - 1)] for i in range(n_samples)],
        index=dates,
    )

    return features, targets_cont, targets_bin, t1


def test_optuna_tuner_alpha_predictor(synthetic_tuning_data):
    features, targets_cont, _, t1 = synthetic_tuning_data

    tuner = OptunaHyperparameterTuner(
        n_splits=3,
        min_train_pct=0.4,
        embargo_pct=0.01,
        random_state=42,
    )

    result = tuner.optimize_alpha_predictor(
        features=features,
        targets=targets_cont,
        t1=t1,
        metric="rank_ic",
        n_trials=3,
    )

    assert isinstance(result, TuningResult)
    assert result.n_trials == 3
    assert "learning_rate" in result.best_params
    assert "ridge_alpha" in result.best_params
    assert isinstance(result.best_score, float)
    assert len(result.trials_df) == 3

    # Build tuned model and ensure it fits/predicts
    tuned_model = tuner.build_tuned_alpha_predictor(result)
    assert isinstance(tuned_model, CrossSectionalAlphaPredictor)
    tuned_model.fit(features, targets_cont)
    preds = tuned_model.predict_raw(features.iloc[:10])
    assert len(preds) == 10


def test_optuna_tuner_meta_classifier(synthetic_tuning_data):
    features, _, targets_bin, t1 = synthetic_tuning_data

    tuner = OptunaHyperparameterTuner(
        n_splits=3,
        min_train_pct=0.4,
        embargo_pct=0.01,
        random_state=42,
    )

    result = tuner.optimize_meta_classifier(
        features=features,
        targets=targets_bin,
        t1=t1,
        n_trials=3,
    )

    assert isinstance(result, TuningResult)
    assert result.metric_name == "auc_roc"
    assert result.n_trials == 3
    assert 0.0 <= result.best_score <= 1.0

    # Build tuned meta classifier
    tuned_meta = tuner.build_tuned_meta_classifier(result)
    assert isinstance(tuned_meta, SecondaryMetaClassifier)
    tuned_meta.fit(features, targets_bin)
    probs = tuned_meta.predict_proba(features.iloc[:10])
    assert len(probs) == 10
    assert all(0.0 <= p <= 1.0 for p in probs)
