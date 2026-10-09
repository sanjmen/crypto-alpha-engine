"""
2-State Gaussian Mixture Model (GMM) Jump and Regime Detector.
Ported and adapted from CrunchDAO Synth for risk management and jump protection.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureJumpDetector:
    """
    2-Component Gaussian Mixture Model:
    State 0: Quiescent / Normal diffusion regime (lower volatility).
    State 1: Jump / Breakout / High-tail risk regime (higher volatility).
    """

    def __init__(
        self,
        normal_weight_prior: float = 0.85,
        jump_scale_multiplier: float = 2.5,
        random_state: int = 42,
    ):
        self.normal_weight_prior = normal_weight_prior
        self.jump_scale_multiplier = jump_scale_multiplier
        self.random_state = random_state

        self.w_norm: float = normal_weight_prior
        self.w_jump: float = 1.0 - normal_weight_prior
        self.mu_norm: float = 0.0
        self.mu_jump: float = 0.0
        self.sigma_norm: float = 0.01
        self.sigma_jump: float = 0.03
        self.is_fitted: bool = False

    def fit(self, returns: Union[np.ndarray, list]) -> "GaussianMixtureJumpDetector":
        r = np.asarray(returns, dtype=np.float64).flatten()
        if len(r) < 30:
            std = float(np.std(r)) if len(r) > 1 else 0.02
            self.sigma_norm = max(std, 1e-4)
            self.sigma_jump = max(std * self.jump_scale_multiplier, 1e-4)
            self.is_fitted = True
            return self

        X = r.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            max_iter=100,
            random_state=self.random_state,
        )
        gmm.fit(X)

        weights = gmm.weights_
        means = gmm.means_.flatten()
        sigmas = np.sqrt(np.maximum(gmm.covariances_.flatten(), 1e-8))

        # Component 0 is normal (smaller sigma), Component 1 is jump (larger sigma)
        if sigmas[0] > sigmas[1]:
            idx_norm, idx_jump = 1, 0
        else:
            idx_norm, idx_jump = 0, 1

        self.w_norm = float(weights[idx_norm])
        self.w_jump = float(weights[idx_jump])
        self.mu_norm = float(means[idx_norm])
        self.mu_jump = float(means[idx_jump])
        self.sigma_norm = float(sigmas[idx_norm])
        self.sigma_jump = float(sigmas[idx_jump])
        self.is_fitted = True

        return self

    def predict_jump_probability(self, current_return: float) -> float:
        """
        Computes the posterior probability of being in the jump regime given a return shock:
        P(State = Jump | r) = (w_jump * N(r | mu_jump, sigma_jump^2)) / f(r)
        """
        if not self.is_fitted:
            return 0.15

        def normal_pdf(x: float, mu: float, sigma: float) -> float:
            return float((1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2))

        pdf_norm = normal_pdf(current_return, self.mu_norm, self.sigma_norm)
        pdf_jump = normal_pdf(current_return, self.mu_jump, self.sigma_jump)

        numerator = self.w_jump * pdf_jump
        denominator = self.w_norm * pdf_norm + self.w_jump * pdf_jump

        if denominator <= 1e-12:
            return float(self.w_jump)

        return float(np.clip(numerator / denominator, 0.0, 1.0))
