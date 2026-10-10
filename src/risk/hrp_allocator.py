"""
Marcos López de Prado's Hierarchical Risk Parity (HRP) Portfolio Allocator.
Reference: Advances in Financial Machine Learning (AFML), Chapter 16.
Building Diversified Portfolios that Outperform Out-of-Sample (JPM 2016).

Solves the Markowitz curse of dimensionality and matrix singularity:
1. Tree Clustering (Ward / Single linkage on correlation distance metric d_ij = sqrt(0.5*(1 - rho_ij))).
2. Quasi-Diagonalization (reordering covariance matrix along dendrogram leaves).
3. Recursive Bisection (inverse-variance top-down capital allocation without matrix inversion).
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) Portfolio Optimization.
    Computes strictly positive, diversified risk weights without inverting the covariance matrix.
    """

    def __init__(self, linkage_method: str = "single"):
        """
        Args:
            linkage_method: Hierarchical clustering linkage method ('single', 'ward', 'complete', 'average').
        """
        valid_methods = {"single", "ward", "complete", "average"}
        if linkage_method not in valid_methods:
            raise ValueError(f"linkage_method must be one of {valid_methods}, got {linkage_method}")
        self.linkage_method = linkage_method
        self.weights_: pd.Series | None = None
        self.ordered_indices_: list[int] | None = None
        self.linkage_matrix_: np.ndarray | None = None

    @staticmethod
    def correlation_to_distance(corr: np.ndarray) -> np.ndarray:
        """
        Computes the correlation distance metric:
        d_ij = sqrt(0.5 * (1 - rho_ij))
        Satisfies metric space properties (d_ii = 0, d_ij >= 0, triangle inequality).
        """
        corr_clipped = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_clipped), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def get_cluster_distances(dist_matrix: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Euclidean distances between distance vectors:
        D_ij = sqrt( sum_k (d_ik - d_jk)^2 )
        Measures similarity across the entire correlation structure.
        """
        n = dist_matrix.shape[0]
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = dist_matrix - dist_matrix[i, :]
            D[i, :] = np.sqrt(np.sum(diff**2, axis=1))
        np.fill_diagonal(D, 0.0)
        return D

    def compute_linkage(self, corr: np.ndarray) -> np.ndarray:
        """
        Performs hierarchical agglomerative tree clustering on the correlation structure.
        """
        dist_mat = self.correlation_to_distance(corr)
        cluster_dists = self.get_cluster_distances(dist_mat)
        # Convert square distance matrix to condensed form for scipy linkage
        condensed_dist = squareform(cluster_dists, checks=False)
        link = sch.linkage(condensed_dist, method=self.linkage_method)
        return link

    @staticmethod
    def get_quasi_diag(link: np.ndarray) -> list[int]:
        """
        Traverses the hierarchical dendrogram to produce quasi-diagonal leaf order.
        Contiguous leaves belong to the same cluster.
        """
        link = link.astype(float)
        root_node = sch.to_tree(link, rd=False)

        def traverse(node) -> list[int]:
            if node.is_leaf():
                return [node.id]
            left_leaves = traverse(node.left)
            right_leaves = traverse(node.right)
            return left_leaves + right_leaves

        return traverse(root_node)

    @staticmethod
    def get_cluster_variance(cov: np.ndarray, cluster_indices: list[int]) -> float:
        """
        Computes the variance of a cluster under Inverse-Variance Portfolio (IVP) weighting:
        w_i = (1 / sigma_i^2) / sum_j (1 / sigma_j^2)
        V_cluster = w^T * Cov_sub * w
        """
        cov_sub = cov[np.ix_(cluster_indices, cluster_indices)]
        diag = np.diag(cov_sub)
        # Avoid division by zero
        inv_vars = 1.0 / np.maximum(diag, 1e-12)
        weights = inv_vars / np.sum(inv_vars)
        cluster_var = float(weights @ cov_sub @ weights)
        return max(cluster_var, 1e-12)

    def recursive_bisection(self, cov: np.ndarray, sort_order: list[int]) -> np.ndarray:
        """
        Top-down recursive bisection allocating capital between bifurcating clusters.
        alpha_1 = 1 - V_1 / (V_1 + V_2) = V_2 / (V_1 + V_2)
        alpha_2 = 1 - alpha_1
        """
        n = cov.shape[0]
        weights = np.ones(n, dtype=np.float64)
        clusters = [sort_order]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    # Bisect cluster into left and right halves
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    var_c1 = self.get_cluster_variance(cov, c1)
                    var_c2 = self.get_cluster_variance(cov, c2)

                    # Allocation split factor
                    alpha_1 = var_c2 / (var_c1 + var_c2)
                    alpha_2 = 1.0 - alpha_1

                    # Multiply weights recursively
                    weights[c1] *= alpha_1
                    weights[c2] *= alpha_2

                    if len(c1) > 1:
                        new_clusters.append(c1)
                    if len(c2) > 1:
                        new_clusters.append(c2)

            clusters = new_clusters

        return weights

    def allocate(self, cov: pd.DataFrame | np.ndarray) -> pd.Series:
        """
        Computes optimal HRP portfolio weights from a covariance matrix.

        Args:
            cov: Covariance matrix (DataFrame or numpy array).

        Returns:
            pd.Series of portfolio weights summing to 1.0.
        """
        if isinstance(cov, pd.DataFrame):
            asset_names = list(cov.columns)
            cov_mat = cov.values.astype(np.float64)
        else:
            asset_names = [f"asset_{i}" for i in range(cov.shape[0])]
            cov_mat = np.asarray(cov, dtype=np.float64)

        n = cov_mat.shape[0]
        if n == 0:
            return pd.Series(dtype=np.float64)
        if n == 1:
            return pd.Series([1.0], index=asset_names)

        # 1. Compute correlation matrix from covariance
        std = np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
        corr = cov_mat / np.outer(std, std)
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Hierarchical Tree Clustering
        self.linkage_matrix_ = self.compute_linkage(corr)

        # 3. Quasi-Diagonalization
        self.ordered_indices_ = self.get_quasi_diag(self.linkage_matrix_)

        # 4. Recursive Bisection
        w_arr = self.recursive_bisection(cov_mat, self.ordered_indices_)

        # Normalize to strictly sum to 1.0
        w_arr = w_arr / np.sum(w_arr)
        self.weights_ = pd.Series(w_arr, index=asset_names, name="hrp_weight")

        return self.weights_

    def allocate_from_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Computes HRP portfolio weights directly from a DataFrame of asset returns.
        """
        clean_rets = returns.dropna(how="all").fillna(0.0)
        cov = clean_rets.cov()
        return self.allocate(cov)

    @staticmethod
    def compute_equal_weight(n: int, names: list[str] | None = None) -> pd.Series:
        """Benchmark: 1/N Equal Weight portfolio."""
        w = np.full(n, 1.0 / max(n, 1))
        idx = names or [f"asset_{i}" for i in range(n)]
        return pd.Series(w, index=idx, name="equal_weight")

    @staticmethod
    def compute_inverse_variance(cov: pd.DataFrame | np.ndarray) -> pd.Series:
        """Benchmark: Standard Inverse-Variance Portfolio (IVP / Naive Risk Parity)."""
        if isinstance(cov, pd.DataFrame):
            names = list(cov.columns)
            cov_mat = cov.values
        else:
            names = [f"asset_{i}" for i in range(cov.shape[0])]
            cov_mat = np.asarray(cov)

        variances = np.diag(cov_mat)
        inv_var = 1.0 / np.maximum(variances, 1e-12)
        weights = inv_var / np.sum(inv_var)
        return pd.Series(weights, index=names, name="inverse_variance_weight")

    @staticmethod
    def evaluate_portfolio_metrics(
        weights: pd.Series | np.ndarray,
        returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Evaluates portfolio return, volatility, Sharpe ratio, and Maximum Drawdown.
        """
        if isinstance(weights, pd.Series):
            w = weights.reindex(returns.columns).fillna(0.0).values
        else:
            w = np.asarray(weights)

        port_returns = returns.values @ w
        mean_ret = float(np.mean(port_returns)) * 24 * 365
        vol = float(np.std(port_returns)) * np.sqrt(24 * 365)
        sharpe = mean_ret / max(vol, 1e-6)

        # Max Drawdown
        cum_ret = np.cumprod(1.0 + port_returns)
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        return {
            "annualized_return": mean_ret,
            "annualized_volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }
