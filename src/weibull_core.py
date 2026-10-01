from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FailureData:

    failure_times: np.ndarray
    censor_times: np.ndarray
    kappa: float

    def __post_init__(self):
        self.failure_times = np.asarray(self.failure_times, dtype=float)
        self.censor_times = np.asarray(self.censor_times, dtype=float)
        if np.any(self.failure_times <= 0) or np.any(self.censor_times <= 0):
            raise ValueError("All times must be strictly positive.")
        if self.kappa <= 0:
            raise ValueError("kappa (shape) must be positive.")

    # ---- sufficient statistics -------------------------------------------------
    @property
    def r(self) -> int:
        """Number of observed failures."""
        return len(self.failure_times)

    @property
    def n(self) -> int:
        """Total number of units (failed + censored)."""
        return len(self.failure_times) + len(self.censor_times)

    @property
    def S(self) -> float:
        """Sufficient statistic S = sum t_i^kappa over ALL units (failed and
        censored)."""
        k = self.kappa
        return float(
            np.sum(self.failure_times**k) + np.sum(self.censor_times**k)
        )

    def summary(self) -> dict:
        return {
            "n_total": self.n,
            "n_failed": self.r,
            "n_censored": len(self.censor_times),
            "kappa": self.kappa,
            "S": self.S,
        }


# --------------------------------------------------------------------------- #
# Weibull pdf / reliability (given beta, kappa)
# --------------------------------------------------------------------------- #
def weibull_pdf(t, beta, kappa):
    t = np.asarray(t, dtype=float)
    return beta * t ** (kappa - 1.0) * np.exp(-beta * t**kappa / kappa)


def weibull_reliability(t, beta, kappa):
    """R(t; beta, kappa) = exp(-beta * t^kappa / kappa)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-beta * t**kappa / kappa)


def beta_to_scale(beta, kappa):
    """Convert the paper's rate parameter beta to the textbook Weibull scale
    eta, for reporting purposes only."""
    return (kappa / beta) ** (1.0 / kappa)


def scale_to_beta(eta, kappa):
    """Inverse of beta_to_scale."""
    return kappa / eta**kappa


# --------------------------------------------------------------------------- #
# Classical estimators
# --------------------------------------------------------------------------- #
def mle_beta(data: FailureData) -> float:

    if data.r == 0:
        return 0.0
    return data.r * data.kappa / data.S


def mle_reliability(t, data: FailureData):

    t = np.asarray(t, dtype=float)
    if data.r == 0:
        return np.ones_like(t)
    return np.exp(-data.r * t**data.kappa / data.S)


def mvue_reliability(t, data: FailureData):

    t = np.asarray(t, dtype=float)
    r = data.r
    S = data.S
    out = np.zeros_like(t)
    if r >= 1:
        base = 1.0 - t**data.kappa / S
        valid = base > 0
        out[valid] = base[valid] ** (r - 1)
    return out


def mse_mvue_analytic(t, data: FailureData):

    t = np.asarray(t, dtype=float)
    r = data.r
    if r < 2:
        return np.full_like(t, np.nan)
    beta_hat = mle_beta(data)
    x = t**data.kappa
    mean_x = data.kappa / beta_hat  # E[T^kappa] under the MLE plug-in
    # crude variance approximation of R_hat via delta method around beta_hat
    var_beta_hat = beta_hat**2 / r
    dR_dbeta = -x / data.kappa * np.exp(-beta_hat * x / data.kappa)
    return (dR_dbeta**2) * var_beta_hat
