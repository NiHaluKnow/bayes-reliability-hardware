"""

Turns Backblaze-style hard-drive records into the per-drive right-censored
time-to-failure dataset (`FailureData`) that weibull_core.py /
bayes_estimator.py consume.

"""

from __future__ import annotations
from dataclasses import dataclass, field
import glob
import os
import numpy as np
import pandas as pd

from .weibull_core import FailureData


# --------------------------------------------------------------------------- #
# A. Real Backblaze data, keyed off actual SMART power-on-hours (default)
# --------------------------------------------------------------------------- #
def load_real_backblaze_dataset(
    csv_path: str,
    full_model_name: str,
    kappa: float,
    date_col: str = "date",
    serial_col: str = "serial_number",
    model_col: str = "model",
    failure_col: str = "failure",
    power_on_hours_col: str = "smart_9_raw_power_on_hours",
) -> tuple[FailureData, dict]:
    
    df = pd.read_csv(csv_path)
    df = df[df[model_col] == full_model_name].copy()
    if df.empty:
        raise ValueError(f"No rows found for model={full_model_name!r} in {csv_path}")
    df[date_col] = pd.to_datetime(df[date_col])

    last_rows = df.sort_values(date_col).groupby(serial_col, as_index=False).last()

    age_days = (last_rows[power_on_hours_col].to_numpy(dtype=float)) / 24.0
    age_days = np.maximum(age_days, 1e-6)  # FailureData requires strictly positive times
    is_failure = last_rows[failure_col].to_numpy() == 1

    failure_times = age_days[is_failure]
    censor_times = age_days[~is_failure]

    data = FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)

    info = {
        "source": "real Backblaze Drive Stats, Q1 2026 (data_Q1_2026.zip, downloaded from backblaze.com)",
        "source_csv": csv_path,
        "model": full_model_name,
        "date_range": [str(df[date_col].min().date()), str(df[date_col].max().date())],
        "n_drives": int(len(last_rows)),
        "n_failures": int(is_failure.sum()),
        "n_censored": int((~is_failure).sum()),
        "assumed_kappa": kappa,
        "age_measure": "SMART attribute 9 (power-on hours) at each drive's last recorded day in the quarter, converted to days",
    }
    return data, info


# --------------------------------------------------------------------------- #
# A. Real Backblaze raw-CSV parser (schema-accurate; for the user's own
#    locally-downloaded quarterly files)
# --------------------------------------------------------------------------- #
def load_backblaze_raw_csvs(
    csv_dir: str,
    model: str,
    kappa: float,
    date_col: str = "date",
    serial_col: str = "serial_number",
    model_col: str = "model",
    failure_col: str = "failure",
) -> FailureData:
    paths = sorted(glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(
            p, usecols=[date_col, serial_col, model_col, failure_col], low_memory=False
        )
        df = df[df[model_col] == model]
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError(f"No rows found for model={model!r} in {csv_dir}")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows[date_col] = pd.to_datetime(all_rows[date_col])

    failure_times, censor_times = [], []
    for serial, grp in all_rows.groupby(serial_col):
        grp = grp.sort_values(date_col)
        first_day = grp[date_col].iloc[0]
        last_day = grp[date_col].iloc[-1]
        service_days = max((last_day - first_day).days, 1)
        if int(grp[failure_col].iloc[-1]) == 1:
            failure_times.append(service_days)
        else:
            censor_times.append(service_days)

    return FailureData(
        failure_times=np.array(failure_times, dtype=float),
        censor_times=np.array(censor_times, dtype=float),
        kappa=kappa,
    )


# --------------------------------------------------------------------------- #
# B. Real, cited aggregate statistics (Backblaze Q1 2026 Drive Stats report)
# --------------------------------------------------------------------------- #
@dataclass
class RealModelStats:

    label: str
    short_label: str
    manufacturer: str
    model: str
    capacity_tb: float
    drive_count: int          # real N
    avg_age_months: float     # real average fleet age
    drive_days: int           # real cumulative observed drive-days
    drive_failures: int       # real cumulative observed failures
    afr_reported_pct: float   # real reported annualised failure rate (%)
    source_url: str
    source_note: str


# Figures below are taken verbatim from Backblaze's own Q1 2026 Drive Stats
# report and press materials (published July 2026), retrieved 2026-09-23:
#   https://www.backblaze.com/blog/backblaze-drive-stats-for-q1-2026/
#   https://www.backblaze.com/blog/?p=113117   (cumulative / lifetime table)
#   https://www.backblaze.com/hard-drive.html  (fleet-level Q1 2026 snapshot)
REAL_MODEL_STATS = {
    "HGST_12TB": RealModelStats(
        label="HGST HUH721212ALE600 (12 TB) -- large, long-tenured cohort",
        short_label="HGST HUH721212ALE600 (12TB)",
        manufacturer="HGST",
        model="HUH721212ALE600",
        capacity_tb=12,
        drive_count=2608,
        avg_age_months=74.5,
        drive_days=6_115_413,
        drive_failures=104,
        afr_reported_pct=0.62,
        source_url="https://www.backblaze.com/blog/?p=113117",
        source_note=(
            "Cumulative/lifetime Drive Stats table row for this model, as "
            "published by Backblaze."
        ),
    ),
    "WDC_22TB": RealModelStats(
        label="WDC WUH722222ALE6L4 (22 TB) -- Backblaze's largest single-model "
        "cohort, young fleet",
        short_label="WDC WUH722222ALE6L4 (22TB)",
        manufacturer="WDC",
        model="WUH722222ALE6L4",
        capacity_tb=22,
        drive_count=45638,
        avg_age_months=16.1,
        drive_days=3_992_942,
        drive_failures=42,
        afr_reported_pct=0.38,
        source_url="https://www.backblaze.com/blog/backblaze-drive-stats-for-q1-2026/",
        source_note=(
            "Q1 2026 Drive Stats quarterly table row for this model, as "
            "published by Backblaze (10,220 drives deployed that quarter; "
            "this is Backblaze's largest single drive-model population, "
            "over 45,000 units)."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Calibration: turn a RealModelStats summary into a synthetic per-drive
# FailureData object consistent with a Weibull(kappa, beta) survival model.
# --------------------------------------------------------------------------- #
def calibrate_beta_from_aggregate(stats: RealModelStats, kappa: float) -> tuple[float, float]:
    """Solve for the Weibull rate beta such that, under a common censoring
    horizon C = drive_days / drive_count (the empirical average observed
    service time per drive), the model-implied failure probability
    1 - exp(-beta*C^kappa/kappa) matches the real observed failure
    proportion (drive_failures / drive_count).

    Returns (beta, C).
    """
    C = stats.drive_days / stats.drive_count
    p_fail = stats.drive_failures / stats.drive_count
    p_fail = min(max(p_fail, 1e-6), 1 - 1e-9)  # numerical safety
    beta = -kappa * np.log(1.0 - p_fail) / (C**kappa)
    return beta, C


def build_real_stat_calibrated_dataset(
    stats_key: str, kappa: float, rng: np.random.Generator
) -> tuple[FailureData, dict]:
    
    stats = REAL_MODEL_STATS[stats_key]
    beta, C = calibrate_beta_from_aggregate(stats, kappa)

    N = stats.drive_count
    lifetimes = (-kappa * np.log(rng.random(N)) / beta) ** (1.0 / kappa)
    failed_mask = lifetimes <= C

    failure_times = lifetimes[failed_mask]
    censor_times = np.full(np.sum(~failed_mask), C)

    data = FailureData(failure_times=failure_times, censor_times=censor_times, kappa=kappa)

    info = {
        "stats_key": stats_key,
        "label": stats.label,
        "source_url": stats.source_url,
        "source_note": stats.source_note,
        "real_drive_count": stats.drive_count,
        "real_drive_days": stats.drive_days,
        "real_drive_failures": stats.drive_failures,
        "real_afr_reported_pct": stats.afr_reported_pct,
        "assumed_kappa": kappa,
        "censoring_horizon_days_C": C,
        "calibrated_beta": beta,
        "implied_eta_scale_days": (kappa / beta) ** (1.0 / kappa),
        "realised_failure_count": int(failed_mask.sum()),
        "realised_failure_rate_pct": 100.0 * failed_mask.mean(),
    }
    return data, info
