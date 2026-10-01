"""

OBJECTIVE 1 (Re-implementation / validation half):

    "Reproduce the Weibull / prior-distribution / Monte-Carlo pipeline
    described in the base paper in Python, and validate the implementation
    against the base paper's published results."

"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.weibull_core import FailureData, weibull_reliability, mle_beta, mle_reliability, mvue_reliability
from src.priors import RayleighPrior
from src.bayes_estimator import (
    bayes_reliability_quadrature,
    bayes_reliability_importance_sampling,
    bayes_reliability_monte_carlo_paper,
)
from src.plotting import plot_validation_recovery

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_censored_weibull_sample(n, beta_true, kappa, censor_time, rng):
    lifetimes = (-kappa * np.log(rng.random(n)) / beta_true) ** (1.0 / kappa)
    failed = lifetimes <= censor_time
    failure_times = lifetimes[failed]
    censor_times = np.full(np.sum(~failed), censor_time)
    return FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)


def run_recovery_test(n, beta_true, kappa, censor_time, rayleigh_b, seed, tag):
    rng = np.random.default_rng(seed)
    data = simulate_censored_weibull_sample(n, beta_true, kappa, censor_time, rng)

    beta_hat = mle_beta(data)
    t_grid = np.linspace(0.01, censor_time * 1.3, 60)

    R_true = weibull_reliability(t_grid, beta_true, kappa)
    R_mle = mle_reliability(t_grid, data)
    R_mvue = mvue_reliability(t_grid, data)

    prior = RayleighPrior(b=rayleigh_b)
    R_bayes_quad = bayes_reliability_quadrature(t_grid, data, prior)
    R_bayes_is, ess = bayes_reliability_importance_sampling(t_grid, data, prior, n_samples=300_000, rng=rng)
    R_bayes_mc, N_mc, ess_mc = bayes_reliability_monte_carlo_paper(
        t_grid, data, prior, k_s=300, k_r=300, rng=rng
    )

    rmse = lambda a, b: float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))

    metrics = {
        "tag": tag,
        "n_units": data.n,
        "n_failures": data.r,
        "n_censored": data.n - data.r,
        "beta_true": beta_true,
        "beta_mle_hat": beta_hat,
        "relative_error_beta_mle_pct": 100.0 * abs(beta_hat - beta_true) / beta_true,
        "rmse_MLE_vs_TRUE": rmse(R_mle, R_true),
        "rmse_MVUE_vs_TRUE": rmse(R_mvue, R_true),
        "rmse_BayesQuadrature_vs_TRUE": rmse(R_bayes_quad, R_true),
        "rmse_BayesMonteCarlo_vs_TRUE": rmse(R_bayes_mc, R_true),
        "rmse_BayesMonteCarlo_vs_BayesQuadrature": rmse(R_bayes_mc, R_bayes_quad),
        "rmse_BayesImportanceSampling_vs_BayesQuadrature": rmse(R_bayes_is, R_bayes_quad),
        "importance_sampling_effective_sample_size": float(ess),
        "monte_carlo_total_replications_N": int(N_mc),
    }

    plot_path = os.path.join(RESULTS_DIR, f"validation_{tag}.png")
    plot_validation_recovery(
        t_grid, R_true, R_mle, R_mvue, R_bayes_quad, R_bayes_mc,
        title=f"Objective 1 validation ({tag}): n={n}, r={data.r} failures, kappa={kappa}",
        outpath=plot_path,
    )
    metrics["plot"] = plot_path
    return metrics


def main():
    print("=" * 78)
    print("OBJECTIVE 1 -- Re-implementation validation (parameter recovery)")
    print("=" * 78)

    scenarios = [
        # (n, beta_true, kappa, censor_time, rayleigh_b, seed, tag)
        (40, 0.015, 1.3, 40.0, 0.02, 1, "small_sample_n40"),
        (300, 0.015, 1.3, 40.0, 0.02, 2, "medium_sample_n300"),
        (5000, 0.015, 1.3, 40.0, 0.02, 3, "large_sample_n5000"),
        (300, 0.30, 2.0, 8.0, 0.4, 4, "rayleigh_special_case_kappa2"),
        (300, 0.20, 1.0, 10.0, 0.25, 5, "exponential_special_case_kappa1"),
    ]

    all_metrics = []
    for scen in scenarios:
        m = run_recovery_test(*scen)
        all_metrics.append(m)
        print(f"\n--- {m['tag']} ---")
        print(f"  n={m['n_units']}, failures={m['n_failures']}, censored={m['n_censored']}")
        print(f"  beta_true={m['beta_true']:.5f}  beta_hat(MLE)={m['beta_mle_hat']:.5f}"
              f"  (rel. error {m['relative_error_beta_mle_pct']:.2f}%)")
        print(f"  RMSE vs TRUE R(t):  MLE={m['rmse_MLE_vs_TRUE']:.4f}"
              f"  MVUE={m['rmse_MVUE_vs_TRUE']:.4f}"
              f"  Bayes(quad)={m['rmse_BayesQuadrature_vs_TRUE']:.4f}"
              f"  Bayes(MC)={m['rmse_BayesMonteCarlo_vs_TRUE']:.4f}")
        print(f"  Bayes MC vs Bayes quadrature agreement (RMSE): "
              f"{m['rmse_BayesMonteCarlo_vs_BayesQuadrature']:.5f}  "
              f"[N={m['monte_carlo_total_replications_N']} replications, "
              f"as in the paper's Eq. 30]")

    out_json = os.path.join(RESULTS_DIR, "objective1_validation_metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {out_json}")
    print("Saved plots   -> results/validation_*.png")

    # Sanity checks (assert, so this script fails loudly if the
    # implementation is actually broken, not just numerically imprecise)
    large_n = next(m for m in all_metrics if m["tag"] == "large_sample_n5000")
    assert large_n["relative_error_beta_mle_pct"] < 5.0, "MLE did not converge to beta_true at n=5000"
    assert large_n["rmse_BayesQuadrature_vs_TRUE"] < 0.03, "Bayes estimator did not converge to truth at n=5000"
    assert large_n["rmse_BayesMonteCarlo_vs_BayesQuadrature"] < 0.03, "MC Bayes estimator disagrees with quadrature"
    print("\nAll convergence sanity checks PASSED.")


if __name__ == "__main__":
    main()
