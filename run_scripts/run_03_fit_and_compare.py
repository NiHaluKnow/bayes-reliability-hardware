"""
OBJECTIVE 3 (Comparative analysis): "Compare Bayes reliability estimates
against classical estimators (MLE/MVUE) on the real dataset for accuracy
and robustness" -- including, per the proposal's Expected Outcomes, "how
each performs with limited failure counts."

Accuracy is quantified as RMSE against the reference Kaplan-Meier curve
in each scenario, and both the curves and the RMSE bar charts are saved.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.weibull_core import FailureData, mle_beta, mle_reliability, mvue_reliability
from src.priors import RayleighPrior
from src.bayes_estimator import bayes_reliability_quadrature
from src.nonparametric import kaplan_meier, km_at
from src.plotting import plot_estimator_comparison, plot_rmse_bar
from src.config import ASSUMED_KAPPA, SMALL_SAMPLE_SIZE
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


def rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_scenario(data_full: FailureData, data_fit: FailureData, prior, t_grid, tag, title, xlabel):
    R_mle = mle_reliability(t_grid, data_fit)
    R_mvue = mvue_reliability(t_grid, data_fit)
    R_bayes = bayes_reliability_quadrature(t_grid, data_fit, prior)

    km_times, km_surv = kaplan_meier(data_full)
    km_ref = km_at(t_grid, km_times, km_surv)

    metrics = {
        "tag": tag,
        "n_fit": data_fit.n,
        "r_fit_failures": data_fit.r,
        "beta_mle": mle_beta(data_fit),
        "rmse_MLE_vs_KM": rmse(R_mle, km_ref),
        "rmse_MVUE_vs_KM": rmse(R_mvue, km_ref),
        "rmse_Bayes_vs_KM": rmse(R_bayes, km_ref),
    }

    plot_path = os.path.join(RESULTS_DIR, f"comparison_{tag}.png")
    plot_estimator_comparison(
        t_grid,
        {"MLE": R_mle, "MVUE": R_mvue, "Bayes (Rayleigh prior)": R_bayes},
        km_times, km_surv,
        title=title, outpath=plot_path, xlabel=xlabel,
    )
    metrics["plot"] = plot_path
    return metrics


def main():
    print("=" * 78)
    print("OBJECTIVE 3 -- Bayes vs classical (MLE/MVUE) comparison on real data")
    print("=" * 78)

    all_metrics = []

    for key, stats in REAL_MODEL_STATS.items():
        csv_path = os.path.join(DATA_DIR, f"{key}_per_drive_dataset.csv")
        data_full = load_failure_data(csv_path, ASSUMED_KAPPA)
        t_grid = np.linspace(1.0, data_full.censor_times.max() * 1.1 if len(data_full.censor_times) else 1000, 60)

        # ---- FULL-DATA scenario -------------------------------------------------
        beta_hat_full = mle_beta(data_full)
        prior_full = RayleighPrior(b=beta_hat_full / np.sqrt(np.pi / 2.0))  # weakly-informative, centred on full MLE
        m_full = run_scenario(
            data_full, data_full, prior_full, t_grid,
            tag=f"{key}_full_data",
            title=f"{stats.short_label}\nFull data (n={data_full.n}, {data_full.r} failures): MLE vs MVUE vs Bayes vs KM",
            xlabel="Time in service (days)",
        )
        m_full["scenario"] = "full_data"
        m_full["label"] = stats.label
        all_metrics.append(m_full)

        # ---- LIMITED-DATA scenario ------------------------------------------------
        data_small = deterministic_small_sample(key, data_full)
        # Prior centred on the FULL dataset's MLE (realistic: represents
        # fleet-historical / manufacturer-spec prior knowledge available
        # before the small new sample was collected), moderately informative.
        prior_small = RayleighPrior(b=beta_hat_full / np.sqrt(np.pi / 2.0))
        m_small = run_scenario(
            data_full, data_small, prior_small, t_grid,
            tag=f"{key}_limited_data",
            title=f"{stats.short_label}\nLimited data (n={data_small.n}, {data_small.r} failures): "
                  f"MLE vs MVUE vs Bayes vs full-data KM",
            xlabel="Time in service (days)",
        )
        m_small["scenario"] = "limited_data"
        m_small["label"] = stats.label
        all_metrics.append(m_small)

        print(f"\n=== {stats.label} ===")
        for m in (m_full, m_small):
            print(f"  [{m['scenario']}] n={m['n_fit']}, failures={m['r_fit_failures']}, "
                  f"beta_hat={m['beta_mle']:.3e}")
            print(f"      RMSE vs Kaplan-Meier:  MLE={m['rmse_MLE_vs_KM']:.4f}  "
                  f"MVUE={m['rmse_MVUE_vs_KM']:.4f}  Bayes={m['rmse_Bayes_vs_KM']:.4f}")

        # RMSE bar chart: full vs limited, for this model
        bar_path = os.path.join(RESULTS_DIR, f"rmse_bars_{key}.png")
        plot_rmse_bar(
            ["MLE\n(full)", "MVUE\n(full)", "Bayes\n(full)", "MLE\n(limited)", "MVUE\n(limited)", "Bayes\n(limited)"],
            [m_full["rmse_MLE_vs_KM"], m_full["rmse_MVUE_vs_KM"], m_full["rmse_Bayes_vs_KM"],
             m_small["rmse_MLE_vs_KM"], m_small["rmse_MVUE_vs_KM"], m_small["rmse_Bayes_vs_KM"]],
            title=f"{stats.short_label}: accuracy vs Kaplan-Meier, full vs limited data",
            outpath=bar_path,
        )

    out_json = os.path.join(RESULTS_DIR, "objective3_comparison_metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics -> {out_json}")
    print("Saved plots   -> results/comparison_*.png, results/rmse_bars_*.png")


if __name__ == "__main__":
    main()
