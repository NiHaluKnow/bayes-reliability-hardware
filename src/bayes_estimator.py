"""
takes the same failure/censoring data, adds a prior belief about β, obtains the posterior distribution of β, and then averages the reliability over that posterior.
"""

from __future__ import annotations
import numpy as np
from scipy import integrate, optimize

from .weibull_core import FailureData
from .priors import RayleighPrior, BetaPrior, UniformPrior


# --------------------------------------------------------------------------- #
# 1. Numerical quadrature (near-exact ground truth)
# --------------------------------------------------------------------------- #
def _log_unnorm_posterior(beta, r, S, kappa, prior):
    """log[ beta**r * g(beta) * exp(-beta*S/kappa) ], vectorised-safe."""
    beta = np.asarray(beta, dtype=float)
    with np.errstate(divide="ignore"):
        log_beta_term = np.where(beta > 0, r * np.log(beta), -np.inf)
    return log_beta_term + prior.logpdf(beta) - beta * S / kappa


def _find_mode_and_std(r, S, kappa, prior, beta_search_upper):

    def neg_log_post(beta):
        if beta <= 0:
            return 1e12
        val = _log_unnorm_posterior(beta, r, S, kappa, prior)
        return 1e12 if not np.isfinite(val) else -val

    res = optimize.minimize_scalar(
        neg_log_post, bounds=(1e-10, beta_search_upper), method="bounded",
        options={"xatol": beta_search_upper * 1e-10},
    )
    mode = float(res.x)

    h = max(mode * 1e-4, 1e-10)
    f0 = _log_unnorm_posterior(mode, r, S, kappa, prior)
    fp = _log_unnorm_posterior(mode + h, r, S, kappa, prior)
    fm = _log_unnorm_posterior(max(mode - h, 1e-12), r, S, kappa, prior)
    second_deriv = (fp - 2 * f0 + fm) / h**2

    if not np.isfinite(second_deriv) or second_deriv >= -1e-10:
        std = beta_search_upper * 0.1  # conservative fallback: wide window
    else:
        std = 1.0 / np.sqrt(-second_deriv)
    return mode, std, float(f0)


def _integral_for_shift(shift, r, S, kappa, prior, beta_upper):

    S_shifted = S + shift
    mode, std, f0 = _find_mode_and_std(r, S_shifted, kappa, prior, beta_upper)

    # Generous window: mode +/- 25 std, clipped to the valid [0, beta_upper]
    # support. 25 std comfortably covers all mass for these log-concave-ish
    # posteriors (Gaussian tail probability beyond 25 std is ~0).
    lo = max(1e-12, mode - 25 * std)
    hi = min(beta_upper, mode + 25 * std)
    if hi <= lo:
        lo, hi = 1e-12, beta_upper

    def integrand(beta):
        lv = _log_unnorm_posterior(beta, r, S_shifted, kappa, prior)
        return np.exp(lv - f0)

    val, _ = integrate.quad(integrand, lo, hi, points=[mode] if lo < mode < hi else None, limit=400)

    # Safety net: also integrate the leftover tails outside [lo, hi] in case
    # the posterior is skewed/heavier-tailed than the Laplace window
    # assumed (cheap, and keeps the estimator robust for small-r / diffuse
    # posteriors where the Gaussian approximation is weakest).
    if lo > 1e-12:
        left, _ = integrate.quad(integrand, 1e-12, lo, limit=100)
        val += left
    if hi < beta_upper:
        right, _ = integrate.quad(integrand, hi, beta_upper, limit=100)
        val += right

    return val, f0


def bayes_reliability_quadrature(t, data: FailureData, prior, beta_upper=None):

    t = np.asarray(t, dtype=float)
    r, S, kappa = data.r, data.S, data.kappa

    if beta_upper is None:
        # IMPORTANT: all multipliers below are purely MULTIPLICATIVE (never
        # additive) -- beta is often on a tiny absolute scale (e.g. ~1e-6
        # for the real hard-drive data in this project), so any added
        # constant like "+1.0" would completely swamp the true scale and
        # break mode-finding. An earlier version of this function had
        # exactly that bug; run_04's prior-sensitivity script caught it via
        # a suspicious "zero difference between all three priors" result.
        if isinstance(prior, BetaPrior):  # hard support ceiling at .scale
            beta_upper = prior.scale
        elif isinstance(prior, UniformPrior):  # hard support ceiling at .b
            beta_upper = prior.b
        else:
            # unbounded prior (e.g. Rayleigh): scale search range off
            # whichever is larger -- the prior's own mean, or the MLE
            # implied by the likelihood alone -- since for large samples
            # the posterior is likelihood-dominated and can sit far from
            # the prior's mean.
            mle_est = (r * kappa / S) if r > 0 else 0.0
            beta_upper = max(prior.mean, mle_est, 1e-12) * 30.0

    denom_val, denom_logshift = _integral_for_shift(0.0, r, S, kappa, prior, beta_upper)

    out = np.empty_like(t)
    for i, ti in enumerate(t):
        shift = ti**kappa
        num_val, num_logshift = _integral_for_shift(shift, r, S, kappa, prior, beta_upper)
        # r~(t) = (num_val * exp(num_logshift)) / (denom_val * exp(denom_logshift))
        out[i] = (num_val / denom_val) * np.exp(num_logshift - denom_logshift)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 2. Self-normalised importance sampling (generic, any prior)
# --------------------------------------------------------------------------- #
def bayes_reliability_importance_sampling(
    t, data: FailureData, prior, n_samples: int = 200_000, rng=None, return_samples=False
):

    if rng is None:
        rng = np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    r, S, kappa = data.r, data.S, data.kappa

    beta = prior.sample(n_samples, rng)
    beta = beta[beta > 0]

    log_w = r * np.log(beta) - beta * S / kappa
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    ess = 1.0 / np.sum(w**2)

    R_samples = np.exp(-np.outer(beta, t**kappa) / kappa)  # shape (n, len(t))
    estimate = (w[:, None] * R_samples).sum(axis=0)

    if return_samples:
        return np.clip(estimate, 0.0, 1.0), beta, w, ess
    return np.clip(estimate, 0.0, 1.0), ess


# --------------------------------------------------------------------------- #
# 3. The base paper's own two-stage Monte-Carlo replication (Eq. 29-30)
# --------------------------------------------------------------------------- #
def sample_posterior_beta_sir(data: FailureData, prior, k_s: int, rng, n_proposal=200_000):

    _, beta, w, ess = bayes_reliability_importance_sampling(
        t=np.array([1.0]), data=data, prior=prior, n_samples=n_proposal, rng=rng, return_samples=True
    )
    idx = rng.choice(len(beta), size=k_s, replace=True, p=w)
    return beta[idx], ess


def simulate_weibull_lifetime(beta, kappa, rng, size=1):

    u = rng.random(size)
    return (-kappa * np.log(u) / beta) ** (1.0 / kappa)


def bayes_reliability_monte_carlo_paper(
    t, data: FailureData, prior, k_s: int = 300, k_r: int = 300, rng=None
):

    if rng is None:
        rng = np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    kappa = data.kappa

    beta_draws, ess = sample_posterior_beta_sir(data, prior, k_s, rng)

    N = k_s * k_r
    out = np.zeros_like(t)
    # simulate k_r lifetimes for every posterior beta draw, vectorised
    lifetimes = simulate_weibull_lifetime(
        np.repeat(beta_draws, k_r), kappa, rng, size=k_s * k_r
    )
    for i, ti in enumerate(t):
        k_f = np.sum(lifetimes <= ti)
        out[i] = (N - k_f) / N
    return out, N, ess


# --------------------------------------------------------------------------- #
# Credible intervals (for prior-sensitivity plots)
# --------------------------------------------------------------------------- #
def bayes_credible_interval(
    t, data: FailureData, prior, n_samples=200_000, rng=None, level=0.90
):

    if rng is None:
        rng = np.random.default_rng(0)
    t = np.asarray(t, dtype=float)
    _, beta, w, _ = bayes_reliability_importance_sampling(
        t, data, prior, n_samples=n_samples, rng=rng, return_samples=True
    )
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lower = np.empty_like(t)
    upper = np.empty_like(t)
    order = np.argsort(beta)
    beta_sorted = beta[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    for i, ti in enumerate(t):
        R_sorted = np.exp(-beta_sorted * ti**data.kappa / data.kappa)
        # R(t;beta) is monotone decreasing in beta, so sorting by beta
        # reverses the order for R; find weighted quantiles directly.
        order_R = np.argsort(R_sorted)
        Rv = R_sorted[order_R]
        cw = np.cumsum(w_sorted[order_R])
        lower[i] = np.interp(lo_q, cw, Rv)
        upper[i] = np.interp(hi_q, cw, Rv)
    return lower, upper
