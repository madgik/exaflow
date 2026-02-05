import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from exaflow.algorithms.federated.pearson_correlation import FederatedPearsonCorrelation
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient

TEST_CASES = [
    {
        "data": {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [2, 1, 2, 1, 2, 1],
            "y1": [1.2, 2.1, 3.0, 3.9, 5.1, 6.2],
            "y2": [2.2, 2.8, 2.5, 3.1, 2.7, 3.3],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
    {
        "data": {
            "x1": [10, 11, 12, 13, 14, 15, 16],
            "x2": [3, 5, 4, 6, 5, 7, 6],
            "y1": [8, 7, 9, 10, 11, 10, 12],
            "y2": [1, 2, 1, 3, 2, 4, 3],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.1,
    },
    {
        "data": {
            "x1": [0.5, 1.0, 1.5, 2.1, 2.6, 3.2],
            "x2": [2.2, 2.1, 2.3, 2.2, 2.4, 2.3],
            "y1": [3.1, 2.9, 3.0, 3.2, 3.3, 3.4],
            "y2": [1.0, 0.8, 1.1, 0.9, 1.2, 1.0],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
    {
        "data": {
            "x1": [4, 5, 6, 7, 8, 9, 10, 11],
            "x2": [1.5, 1.7, 1.6, 1.8, 1.9, 2.0, 2.1, 2.2],
            "y1": [4.1, 5.2, 5.9, 7.1, 7.8, 9.2, 9.9, 10.8],
            "y2": [2.0, 2.1, 1.9, 2.2, 2.0, 2.3, 2.1, 2.4],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.2,
    },
    {
        "data": {
            "x1": [2, 4, 6, 8, 10, 12],
            "x2": [1, 3, 2, 4, 3, 5],
            "y1": [2.1, 4.1, 5.8, 8.2, 9.9, 12.2],
            "y2": [5.1, 4.9, 5.2, 5.0, 5.3, 5.1],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
    {
        "data": {
            "x1": [1, 2, 1, 2, 1, 2, 1],
            "x2": [5, 4, 6, 5, 7, 6, 8],
            "y1": [1.1, 2.2, 1.0, 2.1, 1.2, 2.3, 1.3],
            "y2": [3.0, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.15,
    },
    {
        "data": {
            "x1": [0, 1, 2, 3, 4, 5, 6, 7],
            "x2": [7, 6, 5, 4, 3, 2, 1, 0],
            "y1": [0.1, 0.9, 2.1, 2.9, 4.1, 4.8, 6.2, 7.1],
            "y2": [1.0, 1.2, 0.8, 1.1, 0.9, 1.3, 1.0, 1.4],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
    {
        "data": {
            "x1": [3, 5, 7, 9, 11, 13, 15],
            "x2": [2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7],
            "y1": [3.2, 5.1, 6.9, 9.2, 11.1, 13.3, 14.8],
            "y2": [0.5, 0.6, 0.7, 0.6, 0.8, 0.7, 0.9],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
    {
        "data": {
            "x1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
            "x2": [2.1, 2.0, 2.2, 2.1, 2.3, 2.2, 2.4, 2.3],
            "y1": [0.9, 1.0, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2],
            "y2": [3.0, 2.9, 3.1, 3.0, 3.2, 3.1, 3.3, 3.2],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.1,
    },
    {
        "data": {
            "x1": [2.0, 2.4, 2.8, 3.2, 3.6, 4.0],
            "x2": [1.0, 1.2, 1.1, 1.3, 1.2, 1.4],
            "y1": [2.2, 2.5, 2.7, 3.1, 3.5, 3.9],
            "y2": [4.1, 4.0, 4.2, 4.1, 4.3, 4.2],
        },
        "x_vars": ["x1", "x2"],
        "y_vars": ["y1", "y2"],
        "alpha": 0.05,
    },
]


def _expected_results(df, x_vars, y_vars, alpha):
    n_obs = len(df)
    correlations = []
    p_values = []
    ci_lo = []
    ci_hi = []
    z = st.norm.ppf(1 - alpha / 2)
    se = 1 / np.sqrt(n_obs - 3)

    for y_var in y_vars:
        y = df[y_var].to_numpy(dtype=float)
        row_corr = []
        row_p = []
        row_lo = []
        row_hi = []
        for x_var in x_vars:
            x = df[x_var].to_numpy(dtype=float)
            corr = float(np.corrcoef(x, y)[0, 1])
            corr = float(np.clip(corr, -1.0, 1.0))
            row_corr.append(corr)
            if abs(corr) == 1.0:
                p_val = 0.0
            else:
                p_val = float(st.pearsonr(x, y).pvalue)
            row_p.append(p_val)
            r_z = np.arctanh(corr)
            lo_z, hi_z = r_z - z * se, r_z + z * se
            ci_low, ci_high = np.tanh((lo_z, hi_z))
            row_lo.append(float(ci_low))
            row_hi.append(float(ci_high))
        correlations.append(row_corr)
        p_values.append(row_p)
        ci_lo.append(row_lo)
        ci_hi.append(row_hi)

    return {
        "n_obs": n_obs,
        "correlations": correlations,
        "p_values": p_values,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


@pytest.mark.parametrize("case", TEST_CASES)
def test_federated_pearson_correlation_matches_numpy_and_scipy(case):
    df = pd.DataFrame(case["data"])
    agg = DummyAggClient()
    model = FederatedPearsonCorrelation(agg_client=agg)

    result = model.corrcoef(
        data=df,
        x_vars=case["x_vars"],
        y_vars=case["y_vars"],
        alpha=case["alpha"],
    )

    expected = _expected_results(df, case["x_vars"], case["y_vars"], case["alpha"])

    assert result.n_obs == expected["n_obs"]
    np.testing.assert_allclose(
        np.asarray(result.correlations),
        np.asarray(expected["correlations"]),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(result.p_values),
        np.asarray(expected["p_values"]),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(result.ci_lo),
        np.asarray(expected["ci_lo"]),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(result.ci_hi),
        np.asarray(expected["ci_hi"]),
        rtol=1e-7,
        atol=1e-7,
        equal_nan=True,
    )
