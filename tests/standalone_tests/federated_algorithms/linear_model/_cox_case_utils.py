from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoxCase:
    name: str
    seed: int
    n_obs: int
    beta: tuple[float, ...]
    baseline_hazard: float
    censor_scale: float
    coef_atol: float
    hazard_corr_min: float


CASE_MATRIX = [
    CoxCase("balanced_small", 11, 60, (0.45, -0.30), 0.09, 12.0, 0.28, 0.96),
    CoxCase("balanced_medium", 23, 90, (0.35, -0.25, 0.40), 0.08, 13.0, 0.24, 0.97),
    CoxCase("single_feature", 37, 70, (0.55,), 0.10, 11.0, 0.22, 0.97),
    CoxCase("heavier_censoring", 41, 80, (0.50, -0.15), 0.08, 6.5, 0.35, 0.94),
    CoxCase(
        "four_features",
        53,
        110,
        (0.25, -0.20, 0.30, 0.45),
        0.07,
        15.0,
        0.26,
        0.97,
    ),
    CoxCase("mixed_signals", 67, 75, (0.15, 0.45, -0.35), 0.09, 10.0, 0.30, 0.95),
    CoxCase("low_signal", 79, 85, (0.12, -0.10), 0.10, 14.0, 0.32, 0.93),
    CoxCase("one_feature_high_signal", 83, 65, (0.70,), 0.07, 9.0, 0.26, 0.97),
    CoxCase(
        "three_features_sparse_events",
        97,
        95,
        (0.28, -0.22, 0.33),
        0.05,
        9.5,
        0.34,
        0.94,
    ),
    CoxCase("two_features_large_n", 101, 140, (0.22, -0.40), 0.06, 16.0, 0.22, 0.98),
]


def simulate_cox_case(case: CoxCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    p = len(case.beta)
    beta = np.asarray(case.beta, dtype=float)

    for attempt in range(8):
        X = rng.normal(size=(case.n_obs, p))
        linpred = np.clip(X @ beta, -1.5, 1.5)
        event_rate = case.baseline_hazard * np.exp(linpred)
        event_time = rng.exponential(scale=1.0 / event_rate, size=case.n_obs)
        censor_time = rng.exponential(
            scale=case.censor_scale * (1.0 + 0.3 * attempt),
            size=case.n_obs,
        )
        observed_time = np.minimum(event_time, censor_time)
        events = (event_time <= censor_time).astype(float)
        observed_time = np.clip(observed_time, 0.05, None)
        if int(events.sum()) >= max(8, p + 3):
            y = np.column_stack([observed_time, events])
            return X.astype(float), y.astype(float)

    raise RuntimeError(f"Could not synthesize a stable Cox case for {case.name}")
