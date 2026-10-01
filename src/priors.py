"""
priors.py
=========
Prior distributions for the unknown Weibull rate parameter `beta`
(the base paper's scale parameter beta_i, Eq. 23).

Three priors are implemented, matching the three priors this project's
proposal (Section 4, "Prior sensitivity" objective) asks us to compare:

    * RayleighPrior  -- the paper's own choice (Eq. 23):
                         g(beta) = (beta/b^2) * exp(-beta^2 / (2 b^2))
    * BetaPrior      -- a Beta(mu, nu) prior rescaled onto (0, scale)
    * UniformPrior   -- Uniform(a, b)   (the paper's own robustness
                         comparison prior, Eq. 10, generalised from a
                         fixed upper bound "beta" in the paper's notation
                         to a general [a, b] interval here)

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class RayleighPrior:
    """
    Rayleigh(b) prior on beta: g(beta) = (beta/b^2) exp(-beta^2/(2 b^2)).

    """

    b: float

    @property
    def mean(self) -> float:
        return self.b * np.sqrt(np.pi / 2.0)

    def sample(self, size, rng: np.random.Generator):
        # Rayleigh(b) === Weibull(shape=2, scale=b*sqrt(2)); sample via the
        # inverse-CDF method used throughout the base paper (Eq. 21/29):
        #   beta = b * sqrt(-2 * log(U)),   U ~ Uniform(0,1)
        u = rng.random(size)
        return self.b * np.sqrt(-2.0 * np.log(u))

    def logpdf(self, beta):
        beta = np.asarray(beta, dtype=float)
        with np.errstate(divide="ignore"):
            return np.where(
                beta > 0,
                np.log(beta) - 2 * np.log(self.b) - beta**2 / (2 * self.b**2),
                -np.inf,
            )


@dataclass
class BetaPrior:

    mu: float
    nu: float
    scale: float

    @property
    def mean(self) -> float:
        return self.scale * self.mu / (self.mu + self.nu)

    def sample(self, size, rng: np.random.Generator):
        x = rng.beta(self.mu, self.nu, size=size)
        return self.scale * x

    def logpdf(self, beta):
        beta = np.asarray(beta, dtype=float)
        x = beta / self.scale
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = stats.beta.logpdf(x, self.mu, self.nu) - np.log(self.scale)
        return np.where((x > 0) & (x < 1), lp, -np.inf)


@dataclass
class UniformPrior:

    a: float
    b: float

    @property
    def mean(self) -> float:
        return 0.5 * (self.a + self.b)

    def sample(self, size, rng: np.random.Generator):
        return rng.uniform(self.a, self.b, size=size)

    def logpdf(self, beta):
        beta = np.asarray(beta, dtype=float)
        width = self.b - self.a
        return np.where(
            (beta >= self.a) & (beta <= self.b), -np.log(width), -np.inf
        )


def make_comparable_priors(center: float, spread: float = 1.0):

    center = float(center)
    # Rayleigh(b): mean = b*sqrt(pi/2)  =>  b = center / sqrt(pi/2)
    rayleigh = RayleighPrior(b=spread * center / np.sqrt(np.pi / 2.0))

    # Beta(mu,nu) rescaled to (0, scale): choose mu=nu=2 (a mild unimodal
    # bump, mean = scale/2) with scale = 2*center*spread so the mean
    # matches `center`.
    beta_p = BetaPrior(mu=2.0, nu=2.0, scale=2.0 * center * spread)

    # Uniform(0, 2*center*spread): mean = center*spread
    uniform = UniformPrior(a=1e-9, b=2.0 * center * spread)

    return {"Rayleigh": rayleigh, "Beta": beta_p, "Uniform": uniform}
