"""
Unit tests for Hierarchical Risk Parity (HRP) Portfolio Allocator.
Reference: Marcos López de Prado, AFML Chapter 16.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.hrp_allocator import HierarchicalRiskParity


@pytest.fixture
def synthetic_covariance():
    # 5 assets with known block correlation structure
    # Assets 0 and 1 belong to Cluster A (high correlation 0.8)
    # Assets 2 and 3 belong to Cluster B (high correlation 0.7)
    # Asset 4 is uncorrelated idiosyncratic asset
    corr = np.array([
        [1.0, 0.8, 0.1, 0.1, 0.0],
        [0.8, 1.0, 0.1, 0.1, 0.0],
        [0.1, 0.1, 1.0, 0.7, 0.0],
        [0.1, 0.1, 0.7, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    # Volatilities: Asset 0 has higher vol than Asset 1
    vols = np.array([0.05, 0.02, 0.04, 0.03, 0.01])
    cov = np.outer(vols, vols) * corr
    names = ["BTC", "ETH", "SOL", "AVAX", "USDC"]
    return pd.DataFrame(cov, index=names, columns=names)


def test_correlation_to_distance():
    corr = np.array([
        [1.0, 0.5, -0.5],
        [0.5, 1.0, 0.0],
        [-0.5, 0.0, 1.0],
    ])
    dist = HierarchicalRiskParity.correlation_to_distance(corr)

    # Diagonal must be 0
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-7)
    # Perfectly correlated rho=1 -> dist=0
    # Positively correlated rho=0.5 -> dist = sqrt(0.5*(1-0.5)) = 0.5
    assert pytest.approx(dist[0, 1], abs=1e-5) == 0.5
    # Negatively correlated rho=-0.5 -> dist = sqrt(0.5*(1+0.5)) = sqrt(0.75) ~ 0.866
    assert pytest.approx(dist[0, 2], abs=1e-5) == np.sqrt(0.75)
    # Symmetry
    np.testing.assert_allclose(dist, dist.T, atol=1e-7)


def test_quasi_diagonalization(synthetic_covariance):
    hrp = HierarchicalRiskParity(linkage_method="single")
    corr = synthetic_covariance.values / np.outer(
        np.sqrt(np.diag(synthetic_covariance.values)),
        np.sqrt(np.diag(synthetic_covariance.values)),
    )
    link = hrp.compute_linkage(corr)
    order = hrp.get_quasi_diag(link)

    assert len(order) == 5
    assert set(order) == {0, 1, 2, 3, 4}


def test_hrp_allocation_properties(synthetic_covariance):
    hrp = HierarchicalRiskParity(linkage_method="single")
    weights = hrp.allocate(synthetic_covariance)

    assert isinstance(weights, pd.Series)
    assert len(weights) == 5
    # Sum of weights must equal 1.0
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    # All weights must be strictly positive
    assert (weights > 0.0).all()

    # In Cluster A (BTC vs ETH): ETH has lower volatility (0.02 vs 0.05), so ETH gets higher weight
    assert weights["ETH"] > weights["BTC"]

    # In Cluster B (SOL vs AVAX): AVAX has lower vol (0.03 vs 0.04), so AVAX gets higher weight
    assert weights["AVAX"] > weights["SOL"]


def test_hrp_on_collinear_singular_matrix():
    # Markowitz Mean-Variance fails on collinear/singular matrices.
    # Create rank-deficient covariance matrix with duplicate assets
    vols = np.array([0.02, 0.02, 0.02, 0.02])
    corr = np.array([
        [1.0, 1.0, 0.2, 0.2],
        [1.0, 1.0, 0.2, 0.2],  # Duplicate of asset 0
        [0.2, 0.2, 1.0, 0.9],
        [0.2, 0.2, 0.9, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    assert np.linalg.matrix_rank(cov) < 4  # Singular matrix!

    hrp = HierarchicalRiskParity(linkage_method="single")
    weights = hrp.allocate(cov)

    # Must solve without any LinAlgError or NaN
    assert not weights.isna().any()
    assert pytest.approx(weights.sum(), abs=1e-6) == 1.0
    assert (weights > 0.0).all()


def test_hrp_from_returns_and_benchmarks():
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    returns = pd.DataFrame(
        {
            "BTC": np.random.normal(0.0001, 0.02, n_bars),
            "ETH": np.random.normal(0.0001, 0.025, n_bars),
            "SOL": np.random.normal(0.0002, 0.035, n_bars),
        },
        index=dates,
    )

    hrp = HierarchicalRiskParity()
    hrp_w = hrp.allocate_from_returns(returns)
    eq_w = hrp.compute_equal_weight(3, names=list(returns.columns))
    ivp_w = hrp.compute_inverse_variance(returns.cov())

    assert len(hrp_w) == 3
    assert len(eq_w) == 3
    assert len(ivp_w) == 3

    metrics_hrp = hrp.evaluate_portfolio_metrics(hrp_w, returns)
    assert "sharpe_ratio" in metrics_hrp
    assert "max_drawdown" in metrics_hrp
    assert metrics_hrp["annualized_volatility"] > 0.0
