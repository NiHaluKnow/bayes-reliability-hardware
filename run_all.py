"""
Runs the full pipeline end-to-end, in order, exactly matching the four
objectives in Section 4 of the project proposal:

    01. Re-implementation & validation  (Objective: Re-implementation)
    02. Build the real hard-drive datasets from genuine Backblaze data (Objective: Extension)
    03. Bayes vs MLE/MVUE comparison    (Objective: Comparative analysis)
    04. Rayleigh vs Beta vs Uniform     (Objective: Prior sensitivity)

Usage:
    python run_all.py
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "run_scripts/run_01_validate_implementation.py",
    "run_scripts/run_02_build_dataset.py",
    "run_scripts/run_03_fit_and_compare.py",
    "run_scripts/run_04_prior_sensitivity.py",
]


def main():
    for script in SCRIPTS:
        path = os.path.join(HERE, script)
        print("\n" + "#" * 78)
        print(f"# Running {script}")
        print("#" * 78)
        result = subprocess.run([sys.executable, path], cwd=HERE)
        if result.returncode != 0:
            print(f"\n*** {script} FAILED (exit code {result.returncode}) -- stopping. ***")
            sys.exit(result.returncode)
    print("\n" + "=" * 78)
    print("ALL STAGES COMPLETE. See results/ for plots and metrics, "
          "data/ for the datasets.")
    print("=" * 78)


if __name__ == "__main__":
    main()
