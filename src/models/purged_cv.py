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


def _resolve_t1_positions(
    X: pd.DataFrame | pd.Series | np.ndarray,
    t1: pd.Series | None,
) -> np.ndarray:
    """
    Maps each observation index `i` to the positional end index of its label horizon `t1`.
    Handles duplicate panel timestamps and out-of-range future `t1` timestamps safely.
    """
    n_samples = len(X)
    if t1 is None or not isinstance(X, (pd.DataFrame, pd.Series)):
        return np.arange(n_samples, dtype=int)

    index_vals = X.index
    if len(t1) == n_samples and t1.index.equals(index_vals):
        t1_targets = t1.values
    else:
        time_to_pos = {ts: pos for pos, ts in enumerate(index_vals)}
        t1_positions = np.empty(n_samples, dtype=int)
        for pos, ts in enumerate(index_vals):
            raw_end = t1.get(ts, ts)
            if isinstance(raw_end, pd.Series):
                raw_end = raw_end.iloc[-1]
            if raw_end in time_to_pos:
                t1_positions[pos] = time_to_pos[raw_end]
            else:
                try:
                    t1_positions[pos] = int(np.searchsorted(index_vals, raw_end, side="right")) - 1
                except TypeError:
                    t1_positions[pos] = n_samples
        return np.maximum(t1_positions, np.arange(n_samples, dtype=int))

    try:
        resolved = np.searchsorted(index_vals.values, t1_targets, side="right") - 1
        return np.maximum(resolved.astype(int), np.arange(n_samples, dtype=int))
    except TypeError:
        time_to_pos = {ts: pos for pos, ts in enumerate(index_vals)}
        return np.array(
            [max(pos, time_to_pos.get(val, n_samples)) for pos, val in enumerate(t1_targets)],
            dtype=int,
        )


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
    ) -> None:
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
        """
        n_samples = len(X)
        if n_samples == 0:
            return

        indices = np.arange(n_samples)
        t1_positions = _resolve_t1_positions(X, t1)

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

            train_start = 0 if self.expanding else fold * test_size
            train_end = test_start

            train_candidates = indices[train_start:train_end]
            valid_train = [idx for idx in train_candidates if t1_positions[idx] < test_start]

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

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01) -> None:
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
        t1_positions = _resolve_t1_positions(X, t1)

        embargo_samples = int(np.ceil(n_samples * self.embargo_pct))
        fold_bounds = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for fold in range(self.n_splits):
            test_start = fold_bounds[fold]
            test_end = fold_bounds[fold + 1]
            test_idx = indices[test_start:test_end]

            train_before = [idx for idx in indices[:test_start] if t1_positions[idx] < test_start]
            embargo_limit = min(n_samples, test_end + embargo_samples)
            train_after = list(indices[embargo_limit:])

            train_idx = np.array(train_before + train_after, dtype=int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def compute_rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes Spearman's Rank Information Coefficient (Rank IC).
    """
    valid = ~(np.isnan(y_pred) | np.isnan(y_true) | np.isinf(y_pred) | np.isinf(y_true))
    if np.sum(valid) < 3:
        return 0.0

    pred_valid = y_pred[valid]
    true_valid = y_true[valid]
    if float(np.std(pred_valid)) <= 1e-12 or float(np.std(true_valid)) <= 1e-12:
        return 0.0

    rho, _ = stats.spearmanr(pred_valid, true_valid)
    return 0.0 if np.isnan(rho) else float(rho)


def compute_cv_metrics(
    fold_predictions: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    """
    Computes comprehensive out-of-sample Cross-Validation performance statistics.
    """
    if not fold_predictions:
        return {
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "information_ratio": 0.0,
            "mean_pearson_ic": 0.0,
            "directional_accuracy": 0.0,
        }

    rank_ics: list[float] = []
    pearson_ics: list[float] = []
    dir_accs: list[float] = []

    for y_true, y_pred in fold_predictions:
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        true_valid = y_true[valid]
        pred_valid = y_pred[valid]

        if len(true_valid) < 3:
            continue

        rank_ic = compute_rank_ic(pred_valid, true_valid)
        rank_ics.append(rank_ic)

        if float(np.std(pred_valid)) > 1e-12 and float(np.std(true_valid)) > 1e-12:
            pearson_ic, _ = stats.pearsonr(pred_valid, true_valid)
            pearson_ics.append(0.0 if np.isnan(pearson_ic) else float(pearson_ic))
        else:
            pearson_ics.append(0.0)

        correct_dir = float(np.mean(np.sign(pred_valid) == np.sign(true_valid)))
        dir_accs.append(correct_dir)

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
    info_ratio = mean_ric / max(std_ric, 1e-4)

    return {
        "mean_rank_ic": mean_ric,
        "std_rank_ic": std_ric,
        "information_ratio": info_ratio,
        "mean_pearson_ic": float(np.mean(pearson_ics)),
        "directional_accuracy": float(np.mean(dir_accs)),
    }
