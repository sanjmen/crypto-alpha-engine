"""
Purged and Embargoed Cross-Validation Engine.
Reference: Marcos López de Prado, Advances in Financial Machine Learning (AFML), Chapter 7.

Eliminates temporal data leakage caused by:
1. Overlapping label horizons (Purging)
2. Serial correlation and autoregressive memory leakage (Embargoing)
"""

from collections.abc import Generator

import numpy as np
import pandas as pd
from scipy import stats


class PurgedWalkForwardCV:
    """
    Purged and Embargoed Walk-Forward Cross-Validation.
    Respects strict chronological ordering. For each fold:
      - Train set precedes Test set.
      - Any training sample whose label horizon (t1) extends into or past the test start is PURGED.
      - Optionally expands (anchored) or rolls (fixed window) the training window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_pct: float = 0.40,
        test_pct: float | None = None,
        embargo_pct: float = 0.01,
        expanding: bool = True,
    ):
        """
        Args:
            n_splits: Number of walk-forward folds.
            min_train_pct: Minimum fraction of data required for the first training set.
            test_pct: Fraction of data per test fold (if None, splits remaining data equally).
            embargo_pct: Fraction of total dataset length used as embargo after test period.
            expanding: If True, train window expands; if False, rolling fixed-size window.
        """
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        self.n_splits = n_splits
        self.min_train_pct = min_train_pct
        self.test_pct = test_pct
        self.embargo_pct = embargo_pct
        self.expanding = expanding

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        t1: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates (train_indices, test_indices) tuples.

        Args:
            X: Feature matrix. If DataFrame with DatetimeIndex or RangeIndex,
               timestamps/indices are extracted.
            y: Optional target series.
            t1: Series of label end times (touch time) indexed by event start time (t0).
                If None, assumes 1-step non-overlapping labels (t1 = t0).
        """
        n_samples = len(X)
        if n_samples == 0:
            return

        indices = np.arange(n_samples)

        # Build t1 index mapping
        if t1 is not None and isinstance(X, (pd.DataFrame, pd.Series)):
            # Map index positions to exit index positions
            t0_times = list(X.index)
            time_to_pos = {t: pos for pos, t in enumerate(t0_times)}
            t1_positions = np.array([
                time_to_pos.get(t1.get(t, t), pos)
                for pos, t in enumerate(t0_times)
            ])
        else:
            # Fallback: assume each sample horizon ends at its own index (no overlap)
            t1_positions = np.arange(n_samples)

        # Calculate fold sizes
        min_train_size = int(np.floor(n_samples * self.min_train_pct))
        remaining = n_samples - min_train_size

        if self.test_pct is not None:
            test_size = int(np.floor(n_samples * self.test_pct))
        else:
            test_size = int(np.floor(remaining / self.n_splits))

        test_size = max(1, test_size)

        for fold in range(self.n_splits):
            test_start = min_train_size + fold * test_size
            test_end = min(n_samples, test_start + test_size)

            if test_start >= n_samples:
                break

            if self.expanding:
                train_start = 0
            else:
                train_start = fold * test_size

            train_end = test_start

            # Candidate train indices
            train_candidates = indices[train_start:train_end]

            # PURGING: remove train observations whose label horizon t1 extends into test
            # i.e., t1_position >= test_start
            valid_train = [
                i for i in train_candidates
                if t1_positions[i] < test_start
            ]

            test_idx = indices[test_start:test_end]
            train_idx = np.array(valid_train, dtype=int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class PurgedKFoldCV:
    """
    Marcos López de Prado's Purged and Embargoed K-Fold Cross-Validation (AFML Ch. 7).
    Partitions the dataset into K contiguous sequential blocks.
    In turn, each block serves as the Test set, while remaining blocks form the Train set.
    - Purging eliminates training samples before the test block that overlap with it.
    - Embargo eliminates training samples immediately following the test block to suppress serial correlation.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 for K-Fold CV")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        t1: pd.Series | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        if n_samples == 0:
            return

        indices = np.arange(n_samples)

        # Build t1 index mapping
        if t1 is not None and isinstance(X, (pd.DataFrame, pd.Series)):
            t0_times = list(X.index)
            time_to_pos = {t: pos for pos, t in enumerate(t0_times)}
            t1_positions = np.array([
                time_to_pos.get(t1.get(t, t), pos)
                for pos, t in enumerate(t0_times)
            ])
        else:
            t1_positions = np.arange(n_samples)

        embargo_samples = int(np.ceil(n_samples * self.embargo_pct))
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for fold in range(self.n_splits):
            test_start = fold_bounds[fold]
            test_end = fold_bounds[fold + 1]
            test_idx = indices[test_start:test_end]

            # 1. Train candidates before test: indices in [0, test_start)
            # PURGE: Drop observations whose horizon reaches or passes test_start
            train_before = [
                i for i in indices[:test_start]
                if t1_positions[i] < test_start
            ]

            # 2. Train candidates after test: indices in [test_end, n_samples)
            # EMBARGO: Drop observations starting within embargo_samples after test_end
            embargo_limit = min(n_samples, test_end + embargo_samples)
            train_after = [
                i for i in indices[embargo_limit:]
            ]

            train_idx = np.array(train_before + train_after, dtype=int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def compute_rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes Spearman's Rank Information Coefficient (Rank IC).
    Rank correlation measures monotonic relationship between alpha predictions and realized returns,
    invariant to linear scaling or outliers.
    """
    valid = ~(np.isnan(y_pred) | np.isnan(y_true) | np.isinf(y_pred) | np.isinf(y_true))
    if np.sum(valid) < 3:
        return 0.0

    rho, _ = stats.spearmanr(y_pred[valid], y_true[valid])
    return 0.0 if np.isnan(rho) else float(rho)


def compute_cv_metrics(
    fold_predictions: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    """
    Computes comprehensive out-of-sample Cross-Validation performance statistics.

    Args:
        fold_predictions: List of tuples (y_test_true, y_test_pred) per CV fold.

    Returns:
        Dict containing:
            - 'mean_rank_ic': Average out-of-sample Rank IC
            - 'std_rank_ic': Standard deviation of Rank IC across folds
            - 'information_ratio': Rank IC / Std(Rank IC) (prediction stability)
            - 'mean_pearson_ic': Average linear Pearson correlation
            - 'directional_accuracy': Fraction of predictions with matching directional sign
    """
    if not fold_predictions:
        return {
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "information_ratio": 0.0,
            "mean_pearson_ic": 0.0,
            "directional_accuracy": 0.0,
        }

    rank_ics = []
    pearson_ics = []
    dir_accs = []

    for y_true, y_pred in fold_predictions:
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        yt = y_true[valid]
        yp = y_pred[valid]

        if len(yt) < 3:
            continue

        # Rank IC
        ric = compute_rank_ic(yp, yt)
        rank_ics.append(ric)

        # Pearson IC
        pic, _ = stats.pearsonr(yp, yt)
        pearson_ics.append(0.0 if np.isnan(pic) else float(pic))

        # Directional Accuracy (sign agreement)
        correct_dir = np.mean(np.sign(yp) == np.sign(yt))
        dir_accs.append(float(correct_dir))

    if not rank_ics:
        return {
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "information_ratio": 0.0,
            "mean_pearson_ic": 0.0,
            "directional_accuracy": 0.0,
        }

    mean_ric = float(np.mean(rank_ics))
    std_ric = float(np.std(rank_ics)) if len(rank_ics) > 1 else 1e-4
    ir = mean_ric / max(std_ric, 1e-4)

    return {
        "mean_rank_ic": mean_ric,
        "std_rank_ic": std_ric,
        "information_ratio": ir,
        "mean_pearson_ic": float(np.mean(pearson_ics)),
        "directional_accuracy": float(np.mean(dir_accs)),
    }
