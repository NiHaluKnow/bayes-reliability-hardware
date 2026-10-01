"""
OBJECTIVE 4 (Prior sensitivity): "Compare how different priors -- Rayleigh,
Beta, and Uniform -- on the scale parameter affect the reliability
estimates and their robustness on the chosen data."

"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.weibull_core import FailureData, mle_beta
from src.priors import make_comparable_priors
from src.bayes_estimator import bayes_reliability_quadrature, bayes_credible_interval
from src.plotting import plot_prior_sensitivity
from src.config import ASSUMED_KAPPA
from src.data_pipeline import REAL_MODEL_STATS
from src.sampling import deterministic_small_sample

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_failure_data(csv_path, kappa):
    df = pd.read_csv(csv_path)
    failure_times = df.loc[df.event == 1, "time_days"].to_numpy()
    censor_times = df.loc[df.event == 0, "time_days"].to_numpy()
    return FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)


def run_for_scenario(data_fit: FailureData, center: float, t_grid, tag, title, xlabel):
    priors = make_comparable_priors(center=center, spread=1.0)

    curves, intervals = {}, {}
    for name, prior in priors.items():
        curves[name] = bayes_reliability_quadrature(t_grid, data_fit, prior)
        lo, hi = bayes_credible_interval(t_grid, data_fit, prior, n_samples=150_000)
        intervals[name] = (lo, hi)

    plot_path = os.path.join(RESULTS_DIR, f"prior_sensitivity_{tag}.png")
    plot_prior_sensitivity(t_grid, curves, intervals, title=title, outpath=plot_path, xlabel=xlabel)

    # pairwise max-abs-difference across priors, at every t -> single robustness number
    names = list(curves.keys())
    max_diff = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.max(np.abs(curves[names[i]] - curves[names[j]]))
            max_diff = max(max_diff, float(d))

    metrics = {
        "tag": tag,
        "n_fit": data_fit.n,
        "r_fit_failures": data_fit.r,
        "prior_means": {name: p.mean for name, p in priors.items()},
        "max_pairwise_abs_difference_in_R": max_diff,
        "plot": plot_path,
    }
    return metrics


def main():
    print("=" * 78)
    print("OBJECTIVE 4 -- Prior sensitivity: Rayleigh vs Beta vs Uniform")
    print("=" * 78)

    all_metrics = []
    for key, stats in REAL_MODEL_STATS.items():
        csv_path = os.path.join(DATA_DIR, f"{key}_per_drive_dataset.csv")
        data_full = load_failure_data(csv_path, ASSUMED_KAPPA)
        data_small = deterministic_small_sample(key, data_full)
        beta_hat_full = mle_beta(data_full)

        # NOTE: this range deliberately extends well BEYOND the observed
        # censoring horizon (up to 4x), not just up to it. Prior choice has
        # the least effect on R(t) close to the data and the LARGEST effect
        # when *extrapolating* reliability beyond the observed horizon --
        # exactly the regime a reliability engineer cares about most (e.g.
        # "what fraction will still be alive at 5 years?" when only ~3
        # months of field data exist yet). Restricting the range to only
        # the observed horizon would understate genuine prior sensitivity.
        horizon = data_full.censor_times.max() if len(data_full.censor_times) else 1000.0
        t_grid = np.linspace(1.0, horizon * 4.0, 60)

        print(f"\n=== {stats.label} ===")

        m_full = run_for_scenario(
            data_full, center=beta_hat_full, t_grid=t_grid,
            tag=f"{key}_full_data",
            title=f"{stats.short_label} -- FULL data (n={data_full.n}, {data_full.r} failures)\n"
                  f"Prior sensitivity: Rayleigh vs Beta vs Uniform (90% credible bands)",
            xlabel="Time in service (days)",
        )
        m_full["scenario"] = "full_data"
        m_full["label"] = stats.label
        all_metrics.append(m_full)
        print(f"  [full_data]    n={m_full['n_fit']}, failures={m_full['r_fit_failures']}  "
              f"max pairwise |R difference| across priors = {m_full['max_pairwise_abs_difference_in_R']:.5f}"
              f"  (data-dominated -> should be small)")

        m_small = run_for_scenario(
            data_small, center=beta_hat_full, t_grid=t_grid,
            tag=f"{key}_limited_data",
            title=f"{stats.short_label} -- LIMITED data (n={data_small.n}, {data_small.r} failures)\n"
                  f"Prior sensitivity: Rayleigh vs Beta vs Uniform (90% credible bands)",
            xlabel="Time in service (days)",
        )
        m_small["scenario"] = "limited_data"
        m_small["label"] = stats.label
        all_metrics.append(m_small)
        print(f"  [limited_data] n={m_small['n_fit']}, failures={m_small['r_fit_failures']}  "
              f"max pairwise |R difference| across priors = {m_small['max_pairwise_abs_difference_in_R']:.5f}"
              f"  (prior-influenced -> typically larger)")

    out_json = os.path.join(RESULTS_DIR, "objective4_prior_sensitivity_metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {out_json}")
    print("Saved plots   -> results/prior_sensitivity_*.png")


if __name__ == "__main__":
    main()
