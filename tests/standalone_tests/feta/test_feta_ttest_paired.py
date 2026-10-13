import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exaflow.algorithms.exareme3.statistics.ttest_paired import TTestPaired
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired
from exaflow.algorithms.feta.feta_ttest_paired import FetaTTestPaired
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import (
    IdentityAggregation,
)


def assert_reference(result, sample_x, sample_y, parameters):
    reference = FederatedTTestPaired(IdentityAggregation()).compute(
        sample_x=sample_x,
        sample_y=sample_y,
        alpha=parameters.get("alpha", 0.05),
        alternative=parameters.get("alt_hypothesis", "two-sided"),
    )
    reference["p"] = reference.pop("p_value")
    actual = result.model_dump() if hasattr(result, "model_dump") else result
    assert actual.keys() == reference.keys()
    for key, expected in reference.items():
        if isinstance(expected, str):
            assert actual[key] == expected
        else:
            np.testing.assert_allclose(
                actual[key], expected, rtol=1e-10, atol=1e-12, equal_nan=True
            )


def selection():
    return Inputdata(
        data_model="test:1.0", datasets=["a"], variables=["first", "second"]
    )


@pytest.mark.parametrize("sizes", [[39], [17, 13, 9], [0, 20, 19]])
@pytest.mark.parametrize("minimum_rows", [0, 10])
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_paired_parity_across_uneven_workers(sizes, minimum_rows, alternative, alpha):
    rng = np.random.default_rng(39)
    first = rng.normal(2.0, 1.5, sum(sizes))
    frame = pd.DataFrame(
        {"first": first, "second": first + rng.normal(0.3, 0.7, len(first))}
    )
    boundaries = np.cumsum([0, *sizes])
    parts = [
        frame.iloc[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    eligible = pd.concat([part for part in parts if len(part) >= minimum_rows])
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    parameters = {"alpha": alpha, "alt_hypothesis": alternative}
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["second"], "y": ["first"]})
        result = FetaTTestPaired(parameters=parameters).run(**tensors)
        assert_reference(
            result,
            eligible["second"].to_numpy(),
            eligible["first"].to_numpy(),
            parameters,
        )
        centralized = stats.ttest_rel(
            eligible["second"], eligible["first"], alternative=alternative
        )
        np.testing.assert_allclose(
            [result.t_stat, result.p],
            [centralized.statistic, centralized.pvalue],
            atol=1e-12,
        )
        assert runtime.step == 5
        assert sum(kind == "prepare" for kind, _ in aggregation.events) == 5
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize(
    "first, second",
    [
        ([], []),
        ([1.0], [2.0]),
        ([1.0, 3.0], [2.0, 5.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
    ],
)
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_small_and_constant_differences(first, second, alternative):
    frame = pd.DataFrame(
        {
            "first": np.asarray(first, dtype=float),
            "second": np.asarray(second, dtype=float),
        }
    )
    runtime, workers, aggregation = federation([frame])
    parameters = {"alt_hypothesis": alternative}
    with runtime:
        tensors = runtime.initialize(selection(), {"x": ["second"], "y": ["first"]})
        if len(first) <= 1:
            with pytest.raises(BadUserInput, match="Not enough observations"):
                FetaTTestPaired(parameters=parameters).run(**tensors)
        else:
            result = FetaTTestPaired(parameters=parameters).run(**tensors)
            assert_reference(result, np.asarray(second), np.asarray(first), parameters)
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("size", [19, 20])
def test_pairs_cannot_mix_lengths_or_execution_contexts(size):
    frame = pd.DataFrame({"first": np.arange(20.0), "second": np.arange(20.0) + 1})
    first, workers_a, _ = federation([frame])
    second, workers_b, _ = federation([frame.iloc[:size]])
    with first, second:
        x = first.initialize(selection(), {"x": ["second"]})["x"]
        y = second.initialize(selection(), {"y": ["first"]})["y"]
        if size == 19:
            with pytest.raises(BadUserInput, match="same length"):
                FetaTTestPaired().run(x=x, y=y)
        else:
            with pytest.raises(ValueError, match="different contexts"):
                FetaTTestPaired().run(x=x, y=y)
        assert first.step == second.step == 0
    assert all(not worker.service.store.contexts for worker in [*workers_a, *workers_b])


def test_specification_preserves_paired_contract_without_preprocessing():
    specification = FetaTTestPaired.get_specification()
    normal = TTestPaired.get_specification()
    assert specification.name == "feta_ttest_paired"
    assert specification.type == AlgorithmType.FETA
    assert specification.required_preprocessing == []
    assert specification.x == normal.x and specification.y == normal.y
    assert specification.parameters == normal.parameters
