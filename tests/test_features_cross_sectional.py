import numpy as np
import pytest

from src.features.cross_sectional import (
    gaussian_rank_transform,
    cross_sectional_zscore,
    dollar_neutralize,
    neutralize_beta,
)


def test_gaussian_rank_transform():
    raw_scores = np.array([10.5, -2.3, 0.0, 50.1, -15.0])
    gauss = gaussian_rank_transform(raw_scores)

    assert len(gauss) == len(raw_scores)
    # Order preservation: highest raw score should have highest gaussian rank
    assert np.argmax(gauss) == np.argmax(raw_scores)
    assert np.argmin(gauss) == np.argmin(raw_scores)


def test_dollar_and_beta_neutralization():
    weights = np.array([0.4, 0.3, -0.1, -0.2])
    dn_weights = dollar_neutralize(weights)

    # Sum of positive should equal sum of absolute negative
    assert np.isclose(np.sum(dn_weights[dn_weights > 0]), 0.5)
    assert np.isclose(np.sum(dn_weights[dn_weights < 0]), -0.5)
    assert np.isclose(np.sum(dn_weights), 0.0, atol=1e-7)

    # Beta neutralization
    betas = np.array([1.2, 0.8, 1.0, 1.1])
    bn_weights = neutralize_beta(weights, betas)
    assert np.isclose(np.dot(bn_weights, betas), 0.0, atol=1e-7)
