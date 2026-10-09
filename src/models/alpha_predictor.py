"""
Cross-Sectional Alpha Predictor.
Combines Ridge Regression and LightGBM gradient boosted decision trees with
cross-sectional rank blending, Gaussian rank transforms, and dollar-neutral signal generation.
Ported and adapted from the institutional architecture in datacrunch-2.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
import lightgbm as lgb

from src.domain.entities import Signal
from src.ports.model_ports import IAlphaModel


class CrossSectionalAlphaPredictor(IAlphaModel):
    """
    Ensemble model combining L2-regularized linear Ridge and LightGBM GBDT.
    Predicts cross-sectional relative performance across the asset universe.
    Enforces dollar neutrality and Gaussian rank normalization.
    """

    def __init__(
        self,
        ridge_alpha: float = 100.0,
        lgb_params: Optional[Dict] = None,
        ridge_weight: float = 0.5,
        lgb_weight: float = 0.5,
    ):
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

        self.ridge_model: Optional[Ridge] = None
        self.lgb_model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> "CrossSectionalAlphaPredictor":
        """
        Fits Ridge and LightGBM models on feature matrix and cross-sectional targets.
        features: DataFrame where rows are (timestamp, symbol) observations and columns are factors.
        targets: Series of forward returns or cross-sectional ranks.
        """
        # Clean NaNs and Infinities
        valid_mask = ~(features.isna().any(axis=1) | targets.isna() | np.isinf(targets))
        X = features.loc[valid_mask].copy()
        y = targets.loc[valid_mask].copy()

        if len(X) == 0:
            raise ValueError("No valid training samples after dropping NaNs.")

        self.feature_names = list(X.columns)

        # 1. Fit Ridge Regression
        self.ridge_model = Ridge(alpha=self.ridge_alpha)
        self.ridge_model.fit(X, y)

        # 2. Fit LightGBM Regressor
        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        self.lgb_model.fit(X, y)

        return self

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generates blended ensemble continuous predictions.
        """
        if self.ridge_model is None or self.lgb_model is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X = features[self.feature_names].fillna(0.0)

        preds_ridge = self.ridge_model.predict(X)
        preds_lgb = self.lgb_model.predict(X)

        # Standardize individual model predictions
        std_ridge = np.std(preds_ridge) or 1.0
        std_lgb = np.std(preds_lgb) or 1.0

        norm_ridge = (preds_ridge - np.mean(preds_ridge)) / std_ridge
        norm_lgb = (preds_lgb - np.mean(preds_lgb)) / std_lgb

        blended = (self.ridge_weight * norm_ridge) + (self.lgb_weight * norm_lgb)
        return blended

    def predict(self, features: pd.DataFrame) -> List[Signal]:
        """
        Generates normalized, dollar-neutral Signal objects for each asset.
        features: DataFrame where index contains symbols or has a 'symbol' column.
        """
        blended_preds = self.predict_raw(features)

        if "symbol" in features.columns:
            symbols = features["symbol"].tolist()
        else:
            symbols = [str(idx) for idx in features.index]

        # Cross-sectional rank transform to [-1.0, 1.0]
        n_assets = len(blended_preds)
        if n_assets == 0:
            return []

        if n_assets == 1:
            ranks = np.array([0.0])
        else:
            ranks = stats.rankdata(blended_preds) / (n_assets + 1.0)  # Uniform (0, 1)
            ranks = (ranks - 0.5) * 2.0  # Zero-mean centered [-1.0, 1.0]

        # Dollar neutralization: sum of weights = 0
        mean_val = np.mean(ranks)
        neutral_signals = ranks - mean_val
        max_abs = np.max(np.abs(neutral_signals)) or 1.0
        normalized = neutral_signals / max_abs

        signals = []
        now = datetime.now(timezone.utc)
        for sym, val in zip(symbols, normalized):
            signals.append(Signal(
                timestamp=now,
                symbol=sym,
                score=float(val),
                confidence=float(min(1.0, abs(val))),
                horizon_minutes=60,
            ))

        return signals
