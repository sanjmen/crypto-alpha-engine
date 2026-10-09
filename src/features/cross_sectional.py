"""
Cross-Sectional Normalization, Gaussian Rank Transformation, and Beta Neutralization.
Ported and adapted from DataCrunch-2 for multi-asset crypto portfolios.

Implements:
1. Cross-Sectional Ranking and Gaussian Inverse Normal CDF mapping.
2. Cross-Sectional Z-Score standardizer.
3. Market Beta Neutralization via Gram-Schmidt orthogonal projection.
4. Dollar Neutralization (Zero-Net-Investment constraint).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.special import erfinv


def gaussian_rank_transform(values: Union[np.ndarray, pd.Series, List[float]], clip_eps: float = 1e-4) -> np.ndarray:
    """
    Transforms cross-sectional values to a standard Gaussian N(0, 1) distribution.
    u_i = (Rank(x_i) - 0.5) / N
    tilde{x}_i = sqrt(2) * erfinv(2 * u_i - 1)
    """
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    # Compute ranks with average tie-breaking
    temp = x.argsort()
    ranks = np.empty_like(temp, dtype=np.float64)
    ranks[temp] = np.arange(n, dtype=np.float64) + 1.0

    # Handle duplicates by averaging
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(counts) < n:
        sum_ranks = np.bincount(inv, weights=ranks)
        ranks = (sum_ranks / counts)[inv]

    # Map ranks to uniform [eps, 1 - eps]
    u = (ranks - 0.5) / float(n)
    u_clipped = np.clip(u, clip_eps, 1.0 - clip_eps)

    # Inverse normal CDF
    normal_scores = np.sqrt(2.0) * erfinv(2.0 * u_clipped - 1.0)
    return np.asarray(normal_scores, dtype=np.float64)


def cross_sectional_zscore(values: Union[np.ndarray, pd.Series, List[float]], eps: float = 1e-8) -> np.ndarray:
    """
    Computes cross-sectional z-score: (x - mean) / std.
    """
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    mean = np.mean(x)
    std = np.std(x)
    if std < eps:
        return np.zeros_like(x)
    return (x - mean) / std


def dollar_neutralize(weights: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """
    Enforces dollar neutrality such that sum(w_i) = 0.
    Separates positive and negative weights and balances gross exposure:
    w_long_sum = 0.5, w_short_sum = -0.5
    """
    w = np.asarray(weights, dtype=np.float64)
    if len(w) == 0:
        return np.array([], dtype=np.float64)

    pos_mask = w > 0
    neg_mask = w < 0

    w_out = np.zeros_like(w)
    pos_sum = np.sum(w[pos_mask])
    neg_sum = np.sum(np.abs(w[neg_mask]))

    if pos_sum > 0:
        w_out[pos_mask] = 0.5 * (w[pos_mask] / pos_sum)
    if neg_sum > 0:
        w_out[neg_mask] = -0.5 * (np.abs(w[neg_mask]) / neg_sum)

    return w_out


def neutralize_beta(
    weights: Union[np.ndarray, pd.Series, List[float]],
    betas: Union[np.ndarray, pd.Series, List[float]],
) -> np.ndarray:
    """
    Orthogonalizes portfolio weights against the market beta vector:
    w_neutral = w - (w^T beta / ||beta||^2) * beta
    """
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(betas, dtype=np.float64)

    if len(w) == 0 or len(w) != len(b):
        return w

    b_norm_sq = np.dot(b, b)
    if b_norm_sq < 1e-12:
        return w

    proj = (np.dot(w, b) / b_norm_sq) * b
    w_neutral = w - proj
    return w_neutral
