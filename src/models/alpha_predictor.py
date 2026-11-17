"""
Cross-Sectional Alpha Predictor.
Combines Ridge Regression and LightGBM gradient boosted decision trees with
cross-sectional rank blending, Gaussian rank transforms, and dollar-neutral signal generation.
Ported and adapted from the institutional architecture in datacrunch-2.
"""

from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

from src.domain.entities import Signal
from src.ports.model_ports import IAlphaModel


class CrossSectionalAlphaPredictor(IAlphaModel):
    """
    Ensemble model combining L2-regularized linear Ridge and LightGBM GBDT.
    Predicts cross-sectional relative performance across the asset universe.
    Enforces dollar neutrality and cross-sectional rank normalization.
    """

    def __init__(
        self,
        ridge_alpha: float = 100.0,
        lgb_params: dict[str, Any] | None = None,
        ridge_weight: float = 0.5,
        lgb_weight: float = 0.5,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.ridge_weight = ridge_weight
        self.lgb_weight = lgb_weight

        self.lgb_params = lgb_params or {
            "n_estimators": 100,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }

        self.ridge_model: Ridge | None = None
        self.lgb_model: lgb.LGBMRegressor | None = None
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "CrossSectionalAlphaPredictor":
        """
        Fits Ridge and LightGBM models on feature matrix and cross-sectional targets.
        features: DataFrame where rows are (timestamp, symbol) observations and columns are factors.
        targets: Series of forward returns or cross-sectional ranks.
        """
        valid_mask = ~(
            features.isna().any(axis=1) | np.isinf(features).any(axis=1) | targets.isna() | np.isinf(targets)
        )
        features_clean = features.loc[valid_mask].copy()
        targets_clean = targets.loc[valid_mask].copy()

        if len(features_clean) == 0:
            raise ValueError("No valid training samples after dropping NaNs.")

        self.feature_names = list(features_clean.columns)

        self.ridge_model = Ridge(alpha=self.ridge_alpha)
        self.ridge_model.fit(features_clean, targets_clean)

        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        self.lgb_model.fit(features_clean, targets_clean)

        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generates blended ensemble continuous predictions.
        """
        if self.ridge_model is None or self.lgb_model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        features_aligned = features[self.feature_names].fillna(0.0)

        preds_ridge = self.ridge_model.predict(features_aligned)
        preds_lgb = self.lgb_model.predict(features_aligned)

        std_ridge = float(np.std(preds_ridge))
        std_lgb = float(np.std(preds_lgb))

        norm_ridge = (
            (preds_ridge - float(np.mean(preds_ridge))) / std_ridge
            if (np.isfinite(std_ridge) and std_ridge > 1e-8)
            else np.zeros_like(preds_ridge)
        )
        norm_lgb = (
            (preds_lgb - float(np.mean(preds_lgb))) / std_lgb
            if (np.isfinite(std_lgb) and std_lgb > 1e-8)
            else np.zeros_like(preds_lgb)
        )

        blended = (self.ridge_weight * norm_ridge) + (self.lgb_weight * norm_lgb)
        return np.asarray(blended, dtype=np.float64)

    def predict(
        self,
        features: pd.DataFrame,
        timestamp: datetime | None = None,
    ) -> list[Signal]:
        """
        Generates normalized, dollar-neutral Signal objects for each asset.
        features: DataFrame where index contains symbols or has a 'symbol' column.
        """
        blended_preds = self.predict_raw(features)

        if "symbol" in features.columns:
            symbols = features["symbol"].tolist()
        else:
            symbols = [str(idx) for idx in features.index]

        n_assets = len(blended_preds)
        if n_assets == 0:
            return []

        if n_assets == 1:
            ranks = np.array([0.0])
        else:
            ranks = stats.rankdata(blended_preds) / (n_assets + 1.0)
            ranks = (ranks - 0.5) * 2.0

        mean_val = float(np.mean(ranks))
        neutral_signals = ranks - mean_val
        raw_max_abs = float(np.max(np.abs(neutral_signals)))
        max_abs = raw_max_abs if (np.isfinite(raw_max_abs) and raw_max_abs > 1e-8) else 1.0
        normalized = neutral_signals / max_abs

        signal_time = timestamp or datetime.now(timezone.utc)
        return [
            Signal(
                timestamp=signal_time,
                symbol=sym,
                score=float(val),
                confidence=float(min(1.0, abs(val))),
                horizon_minutes=60,
            )
            for sym, val in zip(symbols, normalized, strict=False)
        ]
