import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.histogram import FederatedGroupedHistogram
from exaflow.algorithms.feta import ColumnMetadata
from exaflow.algorithms.feta.feta_histogram import FetaHistogram
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def initialize(runtime, data, metadata, grouped):
    columns = {"y": ["value"]}
    if grouped:
        columns["x"] = ["group"]
    return runtime.initialize(
        Inputdata(data_model="test:1.0", datasets=["a"], variables=list(data)),
        columns,
        set(columns),
        metadata,
    )


def reference(data, metadata, grouped, strategy, minimum):
    native_metadata = {
        name: {
            "sql_type": item.source_type,
            "is_categorical": item.is_categorical,
            "enumerations": {str(code): str(code) for code in item.categories},
        }
        for name, item in metadata.items()
    }
    payload = (
        FederatedGroupedHistogram(CentralAggregation())
        .histogram(
            data=data,
            y_var="value",
            x_vars=["group"] if grouped else [],
            metadata=native_metadata,
            bins=5,
            histogram_type=strategy,
            min_row_count=minimum,
        )
        .as_payload()
    )
    result = [
        {
            "var": "value",
            "grouping_var": None,
            "grouping_enum": None,
            "bins": payload["bins"],
            "counts": payload["counts"],
        }
    ]
    for name, entry in payload["grouped"].items():
        for group, counts in zip(entry["groups"], entry["counts"]):
            result.append(
                {
                    "var": "value",
                    "grouping_var": name,
                    "grouping_enum": group,
                    "bins": payload["bins"],
                    "counts": counts,
                }
            )
    return {"histogram": result}


@pytest.mark.parametrize(
    "kind", ["real", "int", "text_category", "int_category", "constant"]
)
@pytest.mark.parametrize("strategy", ["simple", "wilkinson"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("minimum", [0, 8])
def test_histogram_parity(kind, strategy, grouped, minimum):
    rng = np.random.default_rng(91)
    values = rng.normal(size=48)
    metadata = {
        "value": ColumnMetadata("real", False),
        "group": ColumnMetadata("text", True, ("b", "a", "absent")),
    }
    if kind == "int":
        values = np.arange(48) % 7
        metadata["value"] = ColumnMetadata("int", False)
    elif kind == "constant":
        values = np.ones(48)
    elif kind == "text_category":
        values = np.resize(["λ", "a"], 48)
        metadata["value"] = ColumnMetadata("text", True, ("λ", "a", "absent"))
    elif kind == "int_category":
        values = np.resize([2, 10], 48)
        metadata["value"] = ColumnMetadata("int", True, (10, 2, 99))
    data = pd.DataFrame({"value": values, "group": np.resize(["a", "b", "a"], 48)})
    runtime, workers, aggregation = federation(
        [data.iloc[:3], data.iloc[3:25], data.iloc[25:]], minimum_rows=minimum
    )
    eligible = data if minimum == 0 else data.iloc[3:]
    with runtime:
        result = (
            FetaHistogram(parameters={"bins": 5, "histogram_type": strategy})
            .run(**initialize(runtime, data, metadata, grouped))
            .model_dump()
        )
        assert_nested(result, reference(eligible, metadata, grouped, strategy, minimum))
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


def test_integer_category_unknown_code_rejected():
    data = pd.DataFrame({"value": [1, 2, 999]})
    runtime, workers, _ = federation([data])
    with pytest.raises(BadUserInput, match="undeclared codes"):
        with runtime:
            FetaHistogram().run(
                **initialize(
                    runtime, data, {"value": ColumnMetadata("int", True, (1, 2))}, False
                )
            )
    assert all(not worker.service.store.contexts for worker in workers)


def test_text_categories_and_groups_without_enumerations():
    data = pd.DataFrame(
        {
            "value": ["z", "λ", "a", "z", "a", "λ"],
            "group": ["b", "a", "b", "a", "b", "a"],
        }
    )
    metadata = {
        "value": ColumnMetadata("text", True),
        "group": ColumnMetadata("text", True),
    }
    runtime, _, _ = federation([data.iloc[:2], data.iloc[2:]])
    with runtime:
        result = FetaHistogram().run(**initialize(runtime, data, metadata, True))
        assert_nested(result.model_dump(), reference(data, metadata, True, "simple", 0))


def test_nonfinite_numerical_bounds_rejected_and_cleaned():
    data = pd.DataFrame({"value": [1.0, np.inf]})
    runtime, workers, _ = federation([data])
    with pytest.raises(BadUserInput, match="No finite data"):
        with runtime:
            FetaHistogram().run(
                **initialize(
                    runtime, data, {"value": ColumnMetadata("real", False)}, False
                )
            )
    assert all(not worker.service.store.contexts for worker in workers)
