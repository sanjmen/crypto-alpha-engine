"""
Secondary Precision Meta-Classifier and Kelly Position Sizing Engine.
Reference: Marcos López de Prado, Advances in Financial Machine Learning (AFML), Chapter 3.

Decouples the directional trade decision (Primary Model) from the trade execution
and sizing decision (Secondary Meta-Model).
Transforms low-precision directional alphas into high-precision trading strategies
by filtering false positives and sizing positions via the Kelly Criterion.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)

from src.domain.entities import Signal


@dataclass
class MetaLabelingReport:
    """Performance telemetry for the Secondary Meta-Classifier."""
    auc_roc: float
    brier_score: float
    precision_filtered: float
    precision_unfiltered: float
    trades_executed_ratio: float
    expected_sharpe_uplift: float


class KellyPositionSizer:
    """
    Translates secondary model success probability P(Y=1) and primary direction
    into optimal Kelly or Fractional Kelly position sizing.
    """

    def __init__(self, kelly_fraction: float = 0.5, max_leverage: float = 1.0):
        """
        Args:
            kelly_fraction: Fraction of full Kelly to deploy (default 0.5 = half-Kelly for tail protection).
            max_leverage: Maximum absolute leverage per position (default 1.0 = 100% notional).
        """
        if not (0.0 < kelly_fraction <= 1.0):
            raise ValueError("kelly_fraction must be in (0, 1]")
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage

    def compute_size(self, prob_success: float, side: float) -> float:
        """
        Computes signed position size w_t in [-max_leverage, max_leverage].

        size = side * max(0.0, 2 * P - 1) * kelly_fraction
        When P <= 0.5, size is 0.0 (trade is rejected).
        """
        if prob_success <= 0.5:
            return 0.0

        raw_kelly = 2.0 * prob_success - 1.0  # Scales from 0.0 to 1.0 for P in [0.5, 1.0]
        allocated = raw_kelly * self.kelly_fraction * self.max_leverage
        allocated = min(allocated, self.max_leverage)

        sign = 1.0 if side >= 0 else -1.0
        return sign * allocated

    def size_batch(self, probs: np.ndarray, sides: np.ndarray) -> np.ndarray:
        """Vectorized computation of signed positions."""
        probs = np.asarray(probs, dtype=np.float64)
        sides = np.asarray(sides, dtype=np.float64)
        active_mask = probs > 0.5
        raw_kelly = np.where(active_mask, 2.0 * probs - 1.0, 0.0)
        allocated = np.clip(raw_kelly * self.kelly_fraction * self.max_leverage, 0.0, self.max_leverage)
        signed_sizes = np.sign(sides) * allocated
        return signed_sizes


class SecondaryMetaClassifier:
    """
    LightGBM Gradient Boosted Decision Tree Meta-Classifier.
    Learns P(Y = 1 | Market State, Regime, Volatility, Liquidity, Signal Strength).
    Trained with AFML Chapter 4 sample uniqueness weights to prevent overfitting.
    """

    def __init__(
        self,
        lgb_params: dict | None = None,
        kelly_fraction: float = 0.5,
        max_leverage: float = 1.0,
    ):
        self.lgb_params = lgb_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []
        self.sizer = KellyPositionSizer(kelly_fraction=kelly_fraction, max_leverage=max_leverage)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
    ) -> "SecondaryMetaClassifier":
        """
        Fits the secondary classifier on meta-features and Triple Barrier labels.

        Args:
            X: DataFrame of meta-features at event inception (t0).
            y: Binary target from Triple Barrier (1 = profit-take, 0 = stopped out).
            sample_weights: Optional Series of sample uniqueness/return-attribution weights.
        """
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        valid_arr = valid_mask.values if hasattr(valid_mask, "values") else np.asarray(valid_mask)
        X_clean = X.iloc[valid_arr].copy()
        y_clean = y.iloc[valid_arr].astype(int).copy()

        if len(X_clean) == 0:
            raise ValueError("No valid samples for training MetaClassifier.")

        self.feature_names = list(X_clean.columns)
        weights_clean = None
        if sample_weights is not None:
            weights_clean = sample_weights.iloc[valid_arr].values

        self.model = lgb.LGBMClassifier(**self.lgb_params)
        self.model.fit(X_clean, y_clean, sample_weight=weights_clean)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns success probability P(Y=1 | X)."""
        if self.model is None:
            raise RuntimeError("MetaClassifier must be fitted before predict_proba.")

        X_aligned = X[self.feature_names].fillna(0.0)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return probs

    def evaluate_filter(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> MetaLabelingReport:
        """
        Evaluates the precision uplift and filtering efficiency of the meta-model.
        """
        probs = self.predict_proba(X_test)
        y_true = y_test.values.astype(int)

        # Baseline unfiltered precision (win rate of primary model)
        baseline_precision = float(np.mean(y_true))

        # Filtered trades: only trades where P > 0.5 are taken
        filtered_mask = probs > 0.5
        if np.sum(filtered_mask) > 0:
            filtered_precision = float(np.mean(y_true[filtered_mask]))
            trades_ratio = float(np.mean(filtered_mask))
        else:
            filtered_precision = baseline_precision
            trades_ratio = 0.0

        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
        brier = float(brier_score_loss(y_true, probs))

        # Sharpe ratio uplift heuristic: Precision improvement / sqrt(Trades ratio)
        if trades_ratio > 0 and baseline_precision > 0:
            sharpe_uplift = (filtered_precision / baseline_precision) * np.sqrt(trades_ratio)
        else:
            sharpe_uplift = 1.0

        return MetaLabelingReport(
            auc_roc=auc,
            brier_score=brier,
            precision_filtered=filtered_precision,
            precision_unfiltered=baseline_precision,
            trades_executed_ratio=trades_ratio,
            expected_sharpe_uplift=sharpe_uplift,
        )

    def size_signals(
        self,
        primary_signals: list[Signal],
        meta_features: pd.DataFrame,
    ) -> list[Signal]:
        """
        Filters and sizes primary alpha signals using the secondary meta-classifier.

        Args:
            primary_signals: Signals emitted by primary model (score in [-1, 1]).
            meta_features: DataFrame indexed by symbol containing current meta-factors.

        Returns:
            List of sized Signal objects. Unprofitable signals are vetoed (score = 0.0).
        """
        if not primary_signals:
            return []

        if self.model is None:
            return primary_signals

        sized_signals = []
        for sig in primary_signals:
            sym = sig.symbol
            if sym not in meta_features.index:
                # If symbol not in meta-features, retain original signal
                sized_signals.append(sig)
                continue

            row_feat = meta_features.loc[[sym]]
            prob_success = float(self.predict_proba(row_feat)[0])
            primary_side = 1.0 if sig.score >= 0 else -1.0

            # Compute Kelly size
            allocated_weight = self.sizer.compute_size(prob_success, primary_side)

            # Updated signal
            sized_signals.append(Signal(
                timestamp=sig.timestamp,
                symbol=sig.symbol,
                score=float(allocated_weight),
                confidence=float(prob_success),
                horizon_minutes=sig.horizon_minutes,
            ))

        return sized_signals
