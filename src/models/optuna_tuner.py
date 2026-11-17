"""
Optuna Hyperparameter Optimization Pipeline with Purged Walk-Forward Cross-Validation.
Machine Learning Alpha & Meta-Classifier Tuning.

Optimizes LightGBM and Ridge hyperparameters against Out-of-Sample Rank IC,
Information Ratio (IR), and Meta-Labeling AUC-ROC without temporal leakage.
Supports hardware acceleration (Colab GPU / multithreaded CPU).
"""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from src.models.alpha_predictor import CrossSectionalAlphaPredictor
from src.models.meta_labeling import SecondaryMetaClassifier
from src.models.purged_cv import PurgedWalkForwardCV, compute_rank_ic

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    """Encapsulates the optimal parameters and validation metrics from an Optuna study."""

    best_params: dict[str, Any]
    best_score: float
    metric_name: str
    n_trials: int
    study: optuna.Study
    trials_df: pd.DataFrame


class OptunaHyperparameterTuner:
    """
    Automated hyperparameter optimization using Tree-structured Parzen Estimators (TPE).
    Integrates PurgedWalkForwardCV to prevent look-ahead bias and autocorrelation leakage.
    """

    def __init__(
        self,
        n_splits: int = 4,
        min_train_pct: float = 0.40,
        embargo_pct: float = 0.01,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.embargo_pct = embargo_pct
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.cv = PurgedWalkForwardCV(
            n_splits=self.n_splits,
            min_train_pct=self.min_train_pct,
            embargo_pct=self.embargo_pct,
            expanding=True,
        )

    def optimize_alpha_predictor(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        t1: pd.Series | None = None,
        metric: str = "rank_ic",
        n_trials: int = 30,
        timeout: int | None = None,
    ) -> TuningResult:
        """
        Optimizes CrossSectionalAlphaPredictor (Ridge + LightGBM ensemble) on out-of-sample Rank IC.
        """
        valid_mask = ~(features.isna().any(axis=1) | targets.isna() | np.isinf(targets))
        valid_arr = valid_mask.values if hasattr(valid_mask, "values") else np.asarray(valid_mask)
        features_clean = features.iloc[valid_arr].copy()
        targets_clean = targets.iloc[valid_arr].copy()
        t1_clean = t1.iloc[valid_arr].copy() if t1 is not None else None

        splits = list(self.cv.split(features_clean, targets_clean, t1=t1_clean))
        if not splits:
            raise ValueError("Cross-validation generated 0 splits. Check feature size and CV parameters.")

        def objective(trial: optuna.Trial) -> float:
            ridge_alpha = trial.suggest_float("ridge_alpha", 1e-1, 1e4, log=True)
            ridge_weight = trial.suggest_float("ridge_weight", 0.1, 0.9)
            lgb_weight = 1.0 - ridge_weight

            lgb_params: dict[str, Any] = {
                "n_estimators": 80,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": -1,
                "n_jobs": -1 if not self.use_gpu else 4,
            }
            if self.use_gpu:
                lgb_params["device"] = "gpu"

            fold_ics: list[float] = []
            for train_idx, test_idx in splits:
                train_features, train_targets = features_clean.iloc[train_idx], targets_clean.iloc[train_idx]
                test_features, test_targets = features_clean.iloc[test_idx], targets_clean.iloc[test_idx]

                ridge = Ridge(alpha=ridge_alpha)
                ridge.fit(train_features, train_targets)
                pred_ridge = ridge.predict(test_features)

                model_lgb = lgb.LGBMRegressor(**lgb_params)
                model_lgb.fit(train_features, train_targets)
                pred_lgb = model_lgb.predict(test_features)

                std_ridge = float(np.std(pred_ridge))
                std_lgb = float(np.std(pred_lgb))
                norm_ridge = (
                    (pred_ridge - float(np.mean(pred_ridge))) / std_ridge
                    if (np.isfinite(std_ridge) and std_ridge > 1e-8)
                    else np.zeros_like(pred_ridge)
                )
                norm_lgb = (
                    (pred_lgb - float(np.mean(pred_lgb))) / std_lgb
                    if (np.isfinite(std_lgb) and std_lgb > 1e-8)
                    else np.zeros_like(pred_lgb)
                )

                pred_blend = (ridge_weight * norm_ridge) + (lgb_weight * norm_lgb)
                fold_ics.append(compute_rank_ic(pred_blend, test_targets.values))

            mean_ic = float(np.mean(fold_ics))
            std_ic = float(np.std(fold_ics)) if len(fold_ics) > 1 else 1e-4

            if metric == "information_ratio":
                return mean_ic / max(std_ic, 1e-4)
            return mean_ic

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_trial = study.best_trial
        return TuningResult(
            best_params=best_trial.params,
            best_score=float(best_trial.value),
            metric_name=metric,
            n_trials=len(study.trials),
            study=study,
            trials_df=study.trials_dataframe(),
        )

    def optimize_meta_classifier(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        sample_weights: pd.Series | None = None,
        t1: pd.Series | None = None,
        n_trials: int = 30,
        timeout: int | None = None,
    ) -> TuningResult:
        """
        Optimizes SecondaryMetaClassifier (LightGBM binary classifier) on out-of-sample AUC-ROC.
        """
        valid_mask = ~(features.isna().any(axis=1) | targets.isna())
        valid_arr = valid_mask.values if hasattr(valid_mask, "values") else np.asarray(valid_mask)
        features_clean = features.iloc[valid_arr].copy()
        targets_clean = targets.iloc[valid_arr].astype(int).copy()
        weights_clean = sample_weights.iloc[valid_arr].copy() if sample_weights is not None else None
        t1_clean = t1.iloc[valid_arr].copy() if t1 is not None else None

        splits = list(self.cv.split(features_clean, targets_clean, t1=t1_clean))
        if not splits:
            raise ValueError("Cross-validation generated 0 splits.")

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": -1,
                "n_jobs": -1 if not self.use_gpu else 4,
            }
            if self.use_gpu:
                params["device"] = "gpu"

            fold_aucs: list[float] = []
            for train_idx, test_idx in splits:
                train_features, train_targets = features_clean.iloc[train_idx], targets_clean.iloc[train_idx]
                test_features, test_targets = features_clean.iloc[test_idx], targets_clean.iloc[test_idx]
                train_weights = weights_clean.iloc[train_idx].values if weights_clean is not None else None

                clf = lgb.LGBMClassifier(**params)
                clf.fit(train_features, train_targets, sample_weight=train_weights)
                probs = clf.predict_proba(test_features)[:, 1]

                if len(np.unique(test_targets)) > 1:
                    fold_aucs.append(float(roc_auc_score(test_targets, probs)))

            return float(np.mean(fold_aucs)) if fold_aucs else 0.5

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_trial = study.best_trial
        return TuningResult(
            best_params=best_trial.params,
            best_score=float(best_trial.value),
            metric_name="auc_roc",
            n_trials=len(study.trials),
            study=study,
            trials_df=study.trials_dataframe(),
        )

    @staticmethod
    def build_tuned_alpha_predictor(result: TuningResult) -> CrossSectionalAlphaPredictor:
        """Constructs a fully configured CrossSectionalAlphaPredictor with tuned parameters."""
        best = result.best_params
        lgb_params = {
            "n_estimators": 80,
            "learning_rate": best.get("learning_rate", 0.03),
            "num_leaves": best.get("num_leaves", 15),
            "max_depth": best.get("max_depth", 4),
            "min_child_samples": best.get("min_child_samples", 20),
            "subsample": best.get("subsample", 0.8),
            "colsample_bytree": best.get("colsample_bytree", 0.8),
            "reg_alpha": best.get("reg_alpha", 0.1),
            "reg_lambda": best.get("reg_lambda", 1.0),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        ridge_w = best.get("ridge_weight", 0.5)
        return CrossSectionalAlphaPredictor(
            ridge_alpha=best.get("ridge_alpha", 100.0),
            lgb_params=lgb_params,
            ridge_weight=ridge_w,
            lgb_weight=1.0 - ridge_w,
        )

    @staticmethod
    def build_tuned_meta_classifier(
        result: TuningResult,
        kelly_fraction: float = 0.5,
    ) -> SecondaryMetaClassifier:
        """Constructs a fully configured SecondaryMetaClassifier with tuned parameters."""
        best = result.best_params
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": best.get("learning_rate", 0.03),
            "num_leaves": best.get("num_leaves", 15),
            "max_depth": best.get("max_depth", 4),
            "min_child_samples": best.get("min_child_samples", 20),
            "subsample": best.get("subsample", 0.8),
            "colsample_bytree": best.get("colsample_bytree", 0.8),
            "reg_alpha": best.get("reg_alpha", 0.1),
            "reg_lambda": best.get("reg_lambda", 1.0),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        return SecondaryMetaClassifier(
            lgb_params=lgb_params,
            kelly_fraction=kelly_fraction,
        )
