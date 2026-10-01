"""

Shared subsampling helper. Uses a per-model-key deterministic seed (derived
from SUBSAMPLE_SEED + a stable hash of the model key) so that run_03 and
run_04 can each independently reconstruct the *exact same* "limited data"
subsample for a given model without having to share state between scripts.
"""

from __future__ import annotations
import hashlib
import numpy as np

from .weibull_core import FailureData
from .config import SUBSAMPLE_SEED, SMALL_SAMPLE_SIZE


def _seed_for_key(key: str) -> int:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return (SUBSAMPLE_SEED + h) % (2**32 - 1)


def subsample(data: FailureData, n_sub: int, rng: np.random.Generator) -> FailureData:
    """Uniform random subsample of n_sub units (without replacement) from a
    FailureData object, preserving each unit's failed/censored status."""
    is_failure = np.concatenate([np.ones(data.r), np.zeros(data.n - data.r)]).astype(bool)
    times = np.concatenate([data.failure_times, data.censor_times])
    n_sub = min(n_sub, len(times))
    idx = rng.choice(len(times), size=n_sub, replace=False)
    sub_failed = is_failure[idx]
    return FailureData(
        failure_times=times[idx][sub_failed],
        censor_times=times[idx][~sub_failed],
        kappa=data.kappa,
    )


def deterministic_small_sample(key: str, data_full: FailureData, n_sub: int = SMALL_SAMPLE_SIZE) -> FailureData:
    """Reconstruct the same "limited field data" subsample for a given
    model key every time, independent of call order or which script calls
    it first."""
    rng = np.random.default_rng(_seed_for_key(key))
    return subsample(data_full, n_sub, rng)
