from collections import Counter

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from exaflow.algorithms.federated.descriptive_stats import (
    FederatedDescriptiveStatistics,
)
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient

ALL_DATASET_LABEL = "all datasets"

TEST_CASES = [
    {
        "name": "two_datasets_numeric_nominal",
        "seed": 1,
        "layout": [("alpha", 3), ("beta", 4)],
        "numerical_vars": ["value"],
        "nominal_vars": ["group"],
        "min_row_count": 1,
        "nominal_levels": {"group": ["a", "b", "c", "d"]},
    },
    {
        "name": "numeric_only_two_datasets",
        "seed": 2,
        "layout": [("alpha", 5), ("beta", 5)],
        "numerical_vars": ["value"],
        "nominal_vars": [],
        "min_row_count": 1,
        "nominal_levels": {},
    },
    {
        "name": "nominal_only_three_datasets",
        "seed": 3,
        "layout": [("alpha", 4), ("beta", 3), ("gamma", 2)],
        "numerical_vars": [],
        "nominal_vars": ["group"],
        "min_row_count": 1,
        "nominal_levels": {"group": ["a", "b", "c", "d", "e"]},
    },
    {
        "name": "min_row_exclusion",
        "seed": 4,
        "layout": [("alpha", 2), ("beta", 3)],
        "numerical_vars": ["value"],
        "nominal_vars": ["group"],
        "min_row_count": 3,
        "nominal_levels": {"group": ["x", "y", "z"]},
    },
    {
        "name": "constant_values",
        "seed": 5,
        "layout": [("alpha", 4), ("beta", 4)],
        "numerical_vars": ["value"],
        "nominal_vars": ["group"],
        "min_row_count": 1,
        "nominal_levels": {"group": ["a", "b"]},
        "constant_value": 5.0,
    },
    {
        "name": "with_missing",
        "seed": 6,
        "layout": [("alpha", 5), ("beta", 5)],
        "numerical_vars": ["value"],
        "nominal_vars": ["group"],
        "min_row_count": 1,
        "nominal_levels": {"group": ["a", "b", "c"]},
        "missing": {
            "value": {"alpha": {1, 3}, "beta": {0}},
            "group": {"beta": {2}},
        },
    },
    {
        "name": "two_numeric_vars",
        "seed": 7,
        "layout": [("alpha", 3), ("beta", 3)],
        "numerical_vars": ["value1", "value2"],
        "nominal_vars": [],
        "min_row_count": 1,
        "nominal_levels": {},
    },
    {
        "name": "two_nominal_vars",
        "seed": 8,
        "layout": [("alpha", 4), ("beta", 4)],
        "numerical_vars": [],
        "nominal_vars": ["group", "flag"],
        "min_row_count": 1,
        "nominal_levels": {
            "group": ["a", "b", "c"],
            "flag": ["x", "y"],
        },
    },
    {
        "name": "four_datasets",
        "seed": 9,
        "layout": [("alpha", 2), ("beta", 3), ("gamma", 4), ("delta", 2)],
        "numerical_vars": ["value"],
        "nominal_vars": ["group"],
        "min_row_count": 2,
        "nominal_levels": {"group": ["a", "b", "c"]},
    },
    {
        "name": "global_none_due_to_min",
        "seed": 10,
        "layout": [("alpha", 2), ("beta", 2)],
        "numerical_vars": ["value"],
        "nominal_vars": [],
        "min_row_count": 4,
        "nominal_levels": {},
    },
]


def _should_missing(missing, column, dataset, row_idx):
    if not missing:
        return False
    column_map = missing.get(column, {})
    dataset_rows = column_map.get(dataset)
    return bool(dataset_rows and row_idx in dataset_rows)


def _build_dataframe(case):
    rows = []
    missing = case.get("missing", {})
    for ds_idx, (dataset, count) in enumerate(case["layout"]):
        part_rng = np.random.default_rng(case["seed"] + ds_idx * 10)
        numeric_rows = {}
        for var_idx, var in enumerate(case["numerical_vars"]):
            if "constant_value" in case:
                numeric_rows[var] = np.full(count, float(case["constant_value"]))
            else:
                numeric_rows[var] = part_rng.normal(
                    loc=var_idx + ds_idx + 1, scale=1.0, size=count
                )
        for row_idx in range(count):
            row = {"dataset": dataset}
            for var in case["numerical_vars"]:
                value = numeric_rows[var][row_idx]
                if _should_missing(missing, var, dataset, row_idx):
                    value = np.nan
                row[var] = float(value)
            for var in case["nominal_vars"]:
                levels = case["nominal_levels"][var]
                value = levels[(row_idx + ds_idx) % len(levels)]
                if _should_missing(missing, var, dataset, row_idx):
                    value = None
                row[var] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _expected_describe(df, case):
    numerical_vars = case["numerical_vars"]
    nominal_vars = case["nominal_vars"]
    min_row_count = case["min_row_count"]
    dataset_groups = {
        dataset: group for dataset, group in df.groupby("dataset", sort=False)
    }
    expected_varbased = {}
    dataset_info = {
        dataset: {"numerical": {}, "nominal": {}} for dataset in dataset_groups
    }

    for dataset, group in dataset_groups.items():
        for var in numerical_vars:
            arr = group[var].dropna()
            num_total = len(group)
            num_dtps = int(len(arr))
            num_na = num_total - num_dtps
            if num_dtps < min_row_count:
                expected_varbased[(var, dataset)] = None
                dataset_info[dataset]["numerical"][var] = None
                continue
            quantiles = arr.quantile([0.25, 0.5, 0.75])
            desc = DescrStatsW(arr)
            std = None if num_dtps <= 1 else float(np.std(arr, ddof=1))
            expected_varbased[(var, dataset)] = {
                "num_dtps": num_dtps,
                "num_na": num_na,
                "num_total": num_total,
                "mean": float(desc.mean),
                "std": std,
                "min": float(arr.min()),
                "q1": float(quantiles.loc[0.25]),
                "q2": float(quantiles.loc[0.5]),
                "q3": float(quantiles.loc[0.75]),
                "max": float(arr.max()),
            }
            dataset_info[dataset]["numerical"][var] = {
                "num_dtps": num_dtps,
                "num_na": num_na,
                "num_total": num_total,
                "arr": arr.reset_index(drop=True),
            }
        for var in nominal_vars:
            arr = group[var].dropna()
            num_total = len(group)
            num_dtps = int(len(arr))
            num_na = num_total - num_dtps
            if num_dtps < min_row_count:
                expected_varbased[(var, dataset)] = None
                dataset_info[dataset]["nominal"][var] = None
                continue
            counts = arr.value_counts().to_dict()
            expected_varbased[(var, dataset)] = {
                "num_dtps": num_dtps,
                "num_na": num_na,
                "num_total": num_total,
                "counts": counts,
            }
            dataset_info[dataset]["nominal"][var] = {
                "num_dtps": num_dtps,
                "num_na": num_na,
                "num_total": num_total,
                "counts": Counter(counts),
            }

    expected_global = {}
    for var in numerical_vars:
        contributions = [
            info["numerical"][var]
            for info in dataset_info.values()
            if info["numerical"].get(var)
        ]
        if not contributions:
            expected_global[(var, ALL_DATASET_LABEL)] = None
            continue
        total_dtps = sum(info["num_dtps"] for info in contributions)
        total_total = sum(info["num_total"] for info in contributions)
        total_na = sum(info["num_na"] for info in contributions)
        combined = pd.concat([info["arr"] for info in contributions], ignore_index=True)
        desc = DescrStatsW(combined)
        std = None if total_dtps <= 1 else float(np.std(combined, ddof=1))
        expected_global[(var, ALL_DATASET_LABEL)] = {
            "num_dtps": total_dtps,
            "num_na": total_na,
            "num_total": total_total,
            "mean": float(desc.mean),
            "std": std,
            "min": float(combined.min()),
            "max": float(combined.max()),
        }
    for var in nominal_vars:
        contributions = [
            info["nominal"][var]
            for info in dataset_info.values()
            if info["nominal"].get(var)
        ]
        if not contributions:
            expected_global[(var, ALL_DATASET_LABEL)] = None
            continue
        total_dtps = sum(info["num_dtps"] for info in contributions)
        total_total = sum(info["num_total"] for info in contributions)
        total_na = sum(info["num_na"] for info in contributions)
        counts = Counter()
        for info in contributions:
            counts.update(info["counts"])
        counts_dict = {level: int(val) for level, val in counts.items() if val > 0}
        expected_global[(var, ALL_DATASET_LABEL)] = {
            "num_dtps": total_dtps,
            "num_na": total_na,
            "num_total": total_total,
            "counts": counts_dict,
        }
    return expected_varbased, expected_global


def _numeric_close(actual, expected, check_quantiles):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    ints = ["num_dtps", "num_na", "num_total"]
    for key in ints:
        assert actual[key] == expected[key]
    assert np.isclose(actual["mean"], expected["mean"])
    if expected["std"] is None:
        assert actual["std"] is None
    else:
        assert np.isclose(actual["std"], expected["std"])
    for bound in ("min", "max"):
        assert np.isclose(actual[bound], expected[bound])
    if check_quantiles:
        for quant in ("q1", "q2", "q3"):
            assert np.isclose(actual[quant], expected[quant])


def _nominal_close(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual["num_dtps"] == expected["num_dtps"]
    assert actual["num_na"] == expected["num_na"]
    assert actual["num_total"] == expected["num_total"]
    assert actual["counts"] == expected["counts"]


@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_federated_descriptive_statistics_matches_statsmodels(case):
    df = _build_dataframe(case)
    describe = FederatedDescriptiveStatistics(agg_client=DummyAggClient())
    result = describe.describe(
        data=df,
        numerical_vars=case["numerical_vars"],
        nominal_vars=case["nominal_vars"],
        min_row_count=case["min_row_count"],
        nominal_levels=case["nominal_levels"],
        dataset_col="dataset",
    )

    varbased_map = {
        (rec["variable"], rec["dataset"]): rec["data"] for rec in result.recs_varbased
    }
    global_map = {
        (rec["variable"], rec["dataset"]): rec["data"] for rec in result.global_varbased
    }
    expected_varbased, expected_global = _expected_describe(df, case)

    assert set(varbased_map) == set(expected_varbased)
    for key, expected in expected_varbased.items():
        if key[0] in case["numerical_vars"]:
            _numeric_close(varbased_map[key], expected, check_quantiles=True)
        else:
            _nominal_close(varbased_map[key], expected)

    assert set(global_map) == set(expected_global)
    for key, expected in expected_global.items():
        if key[0] in case["numerical_vars"]:
            _numeric_close(global_map[key], expected, check_quantiles=False)
        else:
            _nominal_close(global_map[key], expected)
