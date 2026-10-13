from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from exaflow.algorithms.exareme3.linear_model.logistic_regression import (
    LogisticRegression,
)
from exaflow.algorithms.feta import ColumnMetadata
from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta.feta_logistic_regression import FetaLogisticRegression
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_anova_oneway import CentralAggregation
from tests.standalone_tests.feta.test_feta_anova_oneway import assert_nested


def frame():
    rng = np.random.default_rng(141)
    age = rng.normal(size=180)
    group = np.resize(["a", "long", "λ"], 180)
    codes = np.resize([2, 10], 180)
    eta = -0.2 + age * 0.4 + (group == "long") * 0.3 + (codes == 10) * 0.2
    target = rng.random(180) < 1 / (1 + np.exp(-eta))
    return pd.DataFrame(
        {
            "age": age,
            "group": group,
            "code": codes,
            "outcome": np.where(target, "yes", "no"),
        }
    )


def metadata():
    return {
        "age": ColumnMetadata("real", False),
        "group": ColumnMetadata("text", True, ("a", "long", "λ")),
        "code": ColumnMetadata("int", True, (2, 10)),
    }


def reference(data, names):
    categorical = [name for name in names if metadata()[name].is_categorical]
    proxy = SimpleNamespace(
        x=names,
        y=["outcome"],
        metadata={name: {"is_categorical": name in categorical} for name in names},
        get_parameter=lambda name: "yes",
        run_local_udf=lambda func, kw_args, identical_results: func(
            agg_client=CentralAggregation(), data=data, **kw_args
        ),
    )
    return LogisticRegression.run(proxy).model_dump()


def initialize(runtime, names):
    return runtime.initialize(
        Inputdata(data_model="test:1.0", datasets=["a"], variables=[*names, "outcome"]),
        {"x": names, "y": ["outcome"]},
        {"x"},
        metadata(),
    )


@pytest.mark.parametrize("names", [["age"], ["group"], ["age", "group", "code"]])
@pytest.mark.parametrize("sizes", [[180], [50, 60, 70], [0, 85, 95]])
@pytest.mark.parametrize("minimum_rows", [0, 55])
def test_logistic_parity(names, sizes, minimum_rows):
    data = frame()
    edges = np.cumsum([0, *sizes])
    parts = [data.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    eligible = pd.concat([part for part in parts if len(part) >= minimum_rows])
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    with runtime:
        tensors = initialize(runtime, names)
        assert isinstance(tensors["x"], FedColumns)
        assert tensors["x"].names == tuple(names)
        assert tensors["x"].nrows == len(eligible)
        if "code" in names:
            assert tensors["x"]["code"].dtype.kind == "i"
        result = FetaLogisticRegression(
            x_names=names, y_names=["outcome"], parameters={"positive_class": "yes"}
        ).run(**tensors)
        assert_nested(result.model_dump(), reference(eligible, names))
        # Independent statsmodels check of the public coefficient ordering.
        cat_names = [name for name in names if name != "age"]
        covariates = (
            pd.get_dummies(
                eligible[cat_names], columns=cat_names, drop_first=True, dtype=float
            )
            if cat_names
            else pd.DataFrame(index=eligible.index)
        )
        if "age" in names:
            covariates["age"] = eligible.age
        design = sm.add_constant(covariates, has_constant="add")
        fitted = sm.Logit((eligible.outcome == "yes").astype(float), design).fit(
            disp=False
        )
        np.testing.assert_allclose(
            result.summary.coefficients, fitted.params, rtol=1e-6, atol=1e-7
        )
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


def test_column_collection_is_read_only_and_reuses_references():
    runtime, workers, _ = federation([frame()])
    with runtime:
        x = initialize(runtime, ["age", "code"])["x"]
        assert x["age"] is x["age"]
        assert not workers[0].operations
        with pytest.raises(TypeError):
            x.metadata["age"] = ColumnMetadata("int", True)
        with pytest.raises(KeyError):
            x["unknown"]


def test_undeclared_numeric_category_is_rejected():
    data = frame()
    data.loc[0, "code"] = 999
    runtime, _, _ = federation([data])
    with runtime:
        with pytest.raises(BadUserInput, match="undeclared"):
            FetaLogisticRegression(
                x_names=["code"],
                y_names=["outcome"],
                parameters={"positive_class": "yes"},
            ).run(**initialize(runtime, ["code"]))


def test_class_size_errors_cleanup():
    data = frame().iloc[:10].copy()
    data.outcome = "yes"
    runtime, workers, _ = federation([data])
    with pytest.raises(BadUserInput, match="one category"):
        with runtime:
            FetaLogisticRegression(
                x_names=["age"],
                y_names=["outcome"],
                parameters={"positive_class": "yes"},
            ).run(**initialize(runtime, ["age"]))
    assert all(not worker.service.store.contexts for worker in workers)


def test_overlapping_matrix_and_column_groups_preserve_their_dtypes():
    data = frame()
    runtime, _, _ = federation([data])
    with runtime:
        inputs = runtime.initialize(
            Inputdata(data_model="test:1.0", datasets=["a"], variables=["code"]),
            {"x": ["code"], "y": ["code"]},
            {"x"},
            metadata(),
        )
        assert inputs["x"]["code"].dtype.kind == "i"
        assert inputs["y"].dtype == np.dtype("float64")


def test_nonconvergence_cleans_up_context(monkeypatch):
    import exaflow.algorithms.feta.feta_logistic_regression as module

    monkeypatch.setattr(module, "MAX_ITER", 1)
    runtime, workers, _ = federation([frame()])
    with pytest.raises(BadUserInput, match="cannot converge"):
        with runtime:
            FetaLogisticRegression(
                x_names=["age"],
                y_names=["outcome"],
                parameters={"positive_class": "yes"},
            ).run(**initialize(runtime, ["age"]))
    assert all(not worker.service.store.contexts for worker in workers)


def test_singular_hessian_uses_pseudoinverse(monkeypatch):
    def singular(matrix):
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "inv", singular)
    data = frame()
    data["age2"] = data.age
    info = {**metadata(), "age2": ColumnMetadata("real", False)}
    runtime, _, _ = federation([data.iloc[:70], data.iloc[70:]])
    with runtime:
        inputs = runtime.initialize(
            Inputdata(
                data_model="test:1.0",
                datasets=["a"],
                variables=["age", "age2", "outcome"],
            ),
            {"x": ["age", "age2"], "y": ["outcome"]},
            {"x"},
            info,
        )
        result = FetaLogisticRegression(
            x_names=["age", "age2"],
            y_names=["outcome"],
            parameters={"positive_class": "yes"},
        ).run(**inputs)
        proxy = SimpleNamespace(
            x=["age", "age2"],
            y=["outcome"],
            metadata={name: {"is_categorical": False} for name in ["age", "age2"]},
            get_parameter=lambda name: "yes",
            run_local_udf=lambda func, kw_args, identical_results: func(
                agg_client=CentralAggregation(), data=data, **kw_args
            ),
        )
        assert_nested(result.model_dump(), LogisticRegression.run(proxy).model_dump())


def test_nullable_integer_storage_without_missing_values_stays_numerical():
    data = frame()
    data.code = data.code.astype("Int64")
    runtime, _, _ = federation([data])
    with runtime:
        x = initialize(runtime, ["code"])["x"]
        assert x["code"].dtype == np.dtype("int64")
