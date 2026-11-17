"""
Cross-Sectional Normalization, Gaussian Rank Transformation, and Beta Neutralization.
Ported and adapted from DataCrunch-2 for multi-asset crypto portfolios.

Implements:
1. Cross-Sectional Ranking and Gaussian Inverse Normal CDF mapping.
2. Cross-Sectional Z-Score standardizer.
3. Market Beta Neutralization via Gram-Schmidt orthogonal projection.
4. Dollar Neutralization (Zero-Net-Investment constraint).
"""

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import rankdata


def gaussian_rank_transform(
    values: np.ndarray | pd.Series | list[float],
    clip_eps: float = 1e-4,
) -> np.ndarray:
    """
    Transforms cross-sectional values to a standard Gaussian N(0, 1) distribution:
        u_i = (Rank(x_i) - 0.5) / N
        tilde{x}_i = sqrt(2) * erfinv(2 * u_i - 1)
    Handles non-finite values safely by setting their output score to 0.0.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    num_items = len(values_arr)
    if num_items == 0:
        return np.array([], dtype=np.float64)
    if num_items == 1:
        return np.zeros(1, dtype=np.float64)

    finite_mask = np.isfinite(values_arr)
    num_finite = int(np.sum(finite_mask))
    if num_finite <= 1:
        return np.zeros(num_items, dtype=np.float64)

    ranks = rankdata(values_arr[finite_mask], method="average")
    uniform_ranks = (ranks - 0.5) / float(num_finite)
    uniform_clipped = np.clip(uniform_ranks, clip_eps, 1.0 - clip_eps)

    normal_scores = np.zeros(num_items, dtype=np.float64)
    normal_scores[finite_mask] = np.sqrt(2.0) * erfinv(2.0 * uniform_clipped - 1.0)
    return normal_scores


def cross_sectional_zscore(
    values: np.ndarray | pd.Series | list[float],
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes cross-sectional z-score: (x - mean) / std.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    if len(values_arr) == 0:
        return np.array([], dtype=np.float64)

    finite_mask = np.isfinite(values_arr)
    if not np.any(finite_mask):
        return np.zeros_like(values_arr)

    mean_val = float(np.mean(values_arr[finite_mask]))
    std_val = float(np.std(values_arr[finite_mask]))
    if std_val < eps:
        return np.zeros_like(values_arr)

    result = np.zeros_like(values_arr)
    result[finite_mask] = (values_arr[finite_mask] - mean_val) / std_val
    return result


def dollar_neutralize(weights: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    """
    Enforces dollar neutrality such that sum(w_i) = 0.
    Separates positive and negative weights and balances gross exposure:
    w_long_sum = 0.5, w_short_sum = -0.5.
    """
    weights_arr = np.asarray(weights, dtype=np.float64)
    if len(weights_arr) == 0:
        return np.array([], dtype=np.float64)

    weights_clean = np.where(np.isfinite(weights_arr), weights_arr, 0.0)
    pos_mask = weights_clean > 0
    neg_mask = weights_clean < 0

    # If all weights have the same sign, demean first so both legs exist
    if (np.any(pos_mask) and not np.any(neg_mask)) or (np.any(neg_mask) and not np.any(pos_mask)):
        weights_clean = weights_clean - float(np.mean(weights_clean))
        pos_mask = weights_clean > 0
        neg_mask = weights_clean < 0

    neutral_weights = np.zeros_like(weights_clean)
    pos_sum = float(np.sum(weights_clean[pos_mask]))
    neg_sum = float(np.sum(np.abs(weights_clean[neg_mask])))

    if pos_sum > 0 and neg_sum > 0:
        neutral_weights[pos_mask] = 0.5 * (weights_clean[pos_mask] / pos_sum)
        neutral_weights[neg_mask] = -0.5 * (np.abs(weights_clean[neg_mask]) / neg_sum)

    return neutral_weights


def neutralize_beta(
    weights: np.ndarray | pd.Series | list[float],
    betas: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    """
    Orthogonalizes portfolio weights against the market beta vector:
    w_neutral = w - (w^T beta / ||beta||^2) * beta
    """
    weights_arr = np.asarray(weights, dtype=np.float64)
    betas_arr = np.asarray(betas, dtype=np.float64)

    if len(weights_arr) == 0 or len(weights_arr) != len(betas_arr):
        return weights_arr

    weights_clean = np.where(np.isfinite(weights_arr), weights_arr, 0.0)
    betas_clean = np.where(np.isfinite(betas_arr), betas_arr, 0.0)

    beta_norm_sq = float(np.dot(betas_clean, betas_clean))
    if beta_norm_sq < 1e-12:
        return weights_clean

    projection = (float(np.dot(weights_clean, betas_clean)) / beta_norm_sq) * betas_clean
    return weights_clean - projection
