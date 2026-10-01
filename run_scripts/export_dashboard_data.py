import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.weibull_core import (
    FailureData, weibull_reliability, mle_beta, mle_reliability, mvue_reliability,
)
from src.priors import RayleighPrior, make_comparable_priors
from src.bayes_estimator import (
    bayes_reliability_quadrature,
    bayes_reliability_monte_carlo_paper,
    bayes_credible_interval,
)
from src.nonparametric import kaplan_meier
from src.config import ASSUMED_KAPPA
from src.data_pipeline import REAL_MODEL_STATS
from src.sampling import deterministic_small_sample

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def arr(x):
    return [round(float(v), 6) for v in np.asarray(x)]


# --------------------------------------------------------------------------- #
# Objective 1 -- validation / parameter-recovery scenarios (same as run_01)
# --------------------------------------------------------------------------- #
def simulate_censored_weibull_sample(n, beta_true, kappa, censor_time, rng):
    lifetimes = (-kappa * np.log(rng.random(n)) / beta_true) ** (1.0 / kappa)
    failed = lifetimes <= censor_time
    failure_times = lifetimes[failed]
    censor_times = np.full(np.sum(~failed), censor_time)
    return FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)


def build_objective1():
    scenarios = [
        (40, 0.015, 1.3, 40.0, 0.02, 1, "small_sample_n40"),
        (300, 0.015, 1.3, 40.0, 0.02, 2, "medium_sample_n300"),
        (5000, 0.015, 1.3, 40.0, 0.02, 3, "large_sample_n5000"),
        (300, 0.30, 2.0, 8.0, 0.4, 4, "rayleigh_special_case_kappa2"),
        (300, 0.20, 1.0, 10.0, 0.25, 5, "exponential_special_case_kappa1"),
    ]
    out = []
    for n, beta_true, kappa, censor_time, rayleigh_b, seed, tag in scenarios:
        rng = np.random.default_rng(seed)
        data = simulate_censored_weibull_sample(n, beta_true, kappa, censor_time, rng)
        t_grid = np.linspace(0.01, censor_time * 1.3, 60)

        prior = RayleighPrior(b=rayleigh_b)
        R_true = weibull_reliability(t_grid, beta_true, kappa)
        R_mle = mle_reliability(t_grid, data)
        R_mvue = mvue_reliability(t_grid, data)
        R_bayes_quad = bayes_reliability_quadrature(t_grid, data, prior)
        R_bayes_mc, N_mc, _ = bayes_reliability_monte_carlo_paper(t_grid, data, prior, k_s=300, k_r=300, rng=rng)

        out.append({
            "tag": tag,
            "n": data.n, "r_failures": data.r,
            "beta_true": beta_true, "kappa": kappa,
            "beta_hat_mle": round(mle_beta(data), 6),
            "t": arr(t_grid),
            "R_true": arr(R_true),
            "R_mle": arr(R_mle),
            "R_mvue": arr(R_mvue),
            "R_bayes_quadrature": arr(R_bayes_quad),
            "R_bayes_monte_carlo": arr(R_bayes_mc),
        })
    return out


# --------------------------------------------------------------------------- #
# Objectives 3 & 4 -- real hard-drive data (same as run_03 / run_04)
# --------------------------------------------------------------------------- #
def load_failure_data(csv_path, kappa):
    import pandas as pd
    df = pd.read_csv(csv_path)
    failure_times = df.loc[df.event == 1, "time_days"].to_numpy()
    censor_times = df.loc[df.event == 0, "time_days"].to_numpy()
    return FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)


def build_objective3_and_4():
    obj3, obj4 = [], []
    for key, stats in REAL_MODEL_STATS.items():
        csv_path = os.path.join(DATA_DIR, f"{key}_per_drive_dataset.csv")
        data_full = load_failure_data(csv_path, ASSUMED_KAPPA)
        data_small = deterministic_small_sample(key, data_full)
        beta_hat_full = mle_beta(data_full)

        t_grid3 = np.linspace(1.0, data_full.censor_times.max() * 1.1 if len(data_full.censor_times) else 1000, 60)
        km_times, km_surv = kaplan_meier(data_full)

        for tag_suffix, data_fit in (("full_data", data_full), ("limited_data", data_small)):
            prior = RayleighPrior(b=beta_hat_full / np.sqrt(np.pi / 2.0))
            R_mle = mle_reliability(t_grid3, data_fit)
            R_mvue = mvue_reliability(t_grid3, data_fit)
            R_bayes = bayes_reliability_quadrature(t_grid3, data_fit, prior)
            obj3.append({
                "tag": f"{key}_{tag_suffix}",
                "model_key": key, "scenario": tag_suffix, "label": stats.short_label,
                "n_fit": data_fit.n, "r_fit_failures": data_fit.r,
                "t": arr(t_grid3),
                "km_t": arr(km_times), "km_surv": arr(km_surv),
                "R_mle": arr(R_mle), "R_mvue": arr(R_mvue), "R_bayes": arr(R_bayes),
            })

        horizon = data_full.censor_times.max() if len(data_full.censor_times) else 1000.0
        t_grid4 = np.linspace(1.0, horizon * 4.0, 60)
        for tag_suffix, data_fit in (("full_data", data_full), ("limited_data", data_small)):
            priors = make_comparable_priors(center=beta_hat_full, spread=1.0)
            curves = {}
            for name, prior in priors.items():
                R = bayes_reliability_quadrature(t_grid4, data_fit, prior)
                lo, hi = bayes_credible_interval(t_grid4, data_fit, prior, n_samples=150_000)
                curves[name] = {"R": arr(R), "lo": arr(lo), "hi": arr(hi)}
            obj4.append({
                "tag": f"{key}_{tag_suffix}",
                "model_key": key, "scenario": tag_suffix, "label": stats.short_label,
                "n_fit": data_fit.n, "r_fit_failures": data_fit.r,
                "t": arr(t_grid4),
                "priors": curves,
            })
    return obj3, obj4


def main():
    print("Exporting dashboard data (re-running Objective 1/3/4 computations to capture full curves)...")
    dashboard = {
        "objective1": build_objective1(),
        "objective3": None,
        "objective4": None,
    }
    obj3, obj4 = build_objective3_and_4()
    dashboard["objective3"] = obj3
    dashboard["objective4"] = obj4

    out_path = os.path.join(RESULTS_DIR, "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(dashboard, f)
    print(f"Saved -> {out_path}  ({os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
