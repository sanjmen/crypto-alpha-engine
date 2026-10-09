import numpy as np
import pytest

from src.domain.enums import MarketRegime
from src.models.gmm_regime import GaussianMixtureJumpDetector
from src.models.student_t_model import StudentTDensityEstimator
from src.models.regime_classifier import MarketRegimeClassifier


def test_gmm_jump_detector():
    np.random.seed(42)
    normal_returns = np.random.normal(0, 0.01, 80)
    jump_returns = np.random.normal(0, 0.08, 20)
    returns = np.concatenate([normal_returns, jump_returns])

    detector = GaussianMixtureJumpDetector(random_state=42)
    detector.fit(returns)

    assert detector.is_fitted
    assert detector.sigma_jump > detector.sigma_norm

    p_norm = detector.predict_jump_probability(0.001)
    p_jump = detector.predict_jump_probability(0.12)
    assert p_jump > p_norm


def test_student_t_density_and_var():
    np.random.seed(42)
    returns = np.random.standard_t(df=4.0, size=150) * 0.02

    model = StudentTDensityEstimator()
    model.fit(returns)

    assert model.is_fitted
    assert 2.5 <= model.df <= 30.0

    var_5 = model.value_at_risk(alpha=0.05)
    cvar_5 = model.conditional_value_at_risk(alpha=0.05, num_simulations=1000)
    assert var_5 > 0.0
    assert cvar_5 >= var_5


def test_market_regime_classifier():
    np.random.seed(42)
    n = 50
    # Trending series
    closes = np.linspace(100, 150, n) + np.random.normal(0, 0.5, n)
    highs = closes + 1.0
    lows = closes - 1.0
    opens = closes - 0.2

    clf = MarketRegimeClassifier()
    regime = clf.classify(opens, highs, lows, closes)
    assert regime in [MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE]
