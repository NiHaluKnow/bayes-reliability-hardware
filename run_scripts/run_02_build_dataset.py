
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline import load_real_backblaze_dataset, REAL_MODEL_STATS
from src.config import ASSUMED_KAPPA

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "raw_backblaze_HGST12TB_WDC22TB_Q1_2026.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 78)
    print("OBJECTIVE 2 -- Building real hard-drive datasets from genuine Backblaze data")
    print("=" * 78)

    all_info = {}
    for key, stats in REAL_MODEL_STATS.items():
        full_model_name = f"{stats.manufacturer} {stats.model}"
        data, info = load_real_backblaze_dataset(RAW_CSV, full_model_name, ASSUMED_KAPPA)
        info["label"] = stats.label
        all_info[key] = info

        print(f"\n--- {info['label']} ---")
        print(f"  Source: {info['source']}")
        print(f"  Quarter date range:          {info['date_range'][0]} to {info['date_range'][1]}")
        print(f"  Real drives observed (N):    {info['n_drives']:,}")
        print(f"  Real failures this quarter:  {info['n_failures']}")
        print(f"  Real censored (still alive): {info['n_censored']:,}")
        print(f"  Assumed Weibull shape kappa: {info['assumed_kappa']}")
        print(f"  Age measure: {info['age_measure']}")

        # persist the per-drive dataset itself (times, in days, + event flag)
        rows = [{"time_days": float(t), "event": 1} for t in data.failure_times]
        rows += [{"time_days": float(t), "event": 0} for t in data.censor_times]
        out_csv = os.path.join(DATA_DIR, f"{key}_per_drive_dataset.csv")
        import csv as csv_module
        with open(out_csv, "w", newline="") as f:
            w = csv_module.DictWriter(f, fieldnames=["time_days", "event"])
            w.writeheader()
            w.writerows(rows)
        info["dataset_csv"] = out_csv
        print(f"  Saved per-drive dataset -> {out_csv}  ({len(rows)} rows)")

    out_json = os.path.join(DATA_DIR, "objective2_dataset_provenance.json")
    with open(out_json, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f"\nSaved full provenance record -> {out_json}")


if __name__ == "__main__":
    main()
