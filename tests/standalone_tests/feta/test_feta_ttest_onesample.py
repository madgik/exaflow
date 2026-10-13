import numpy as np
import pandas as pd
import pytest
from scipy import stats

from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)
from exaflow.algorithms.feta.feta_ttest_onesample import FetaTTestOneSample
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.feta.inputdata import Inputdata
from exaflow.worker_communication import BadUserInput
from tests.standalone_tests.feta.helpers import federation
from tests.standalone_tests.feta.test_feta_pearson_correlation import (
    IdentityAggregation,
)


def assert_reference(result, sample, parameters):
    reference = FederatedTTestOneSample(IdentityAggregation()).compute(
        sample=sample,
        mu=parameters.get("mu", 0.0),
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


@pytest.mark.parametrize("sizes", [[39], [17, 13, 9], [0, 20, 19]])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("mu", [0.0, 0.75])
@pytest.mark.parametrize("minimum_rows", [0, 10])
def test_federated_one_sample_parity(sizes, alternative, mu, minimum_rows):
    sample = np.random.default_rng(19).normal(1.2, 2.0, sum(sizes))
    frame = pd.DataFrame({"value": sample})
    boundaries = np.cumsum([0, *sizes])
    parts = [
        frame.iloc[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    runtime, workers, aggregation = federation(parts, minimum_rows=minimum_rows)
    expected_sample = np.concatenate(
        [part.value.to_numpy() for part in parts if len(part) >= minimum_rows]
    )
    parameters = {"mu": mu, "alpha": 0.05, "alt_hypothesis": alternative}
    with runtime:
        tensors = runtime.initialize(
            Inputdata(data_model="test:1.0", datasets=["a"], variables=["value"]),
            {"y": ["value"]},
        )
        y = tensors["y"]
        assert set(tensors) == {"y"}
        assert all(
            len(worker.service.store.contexts[runtime.context_id].values) == 1
            for worker in runtime.workers.values()
        )
        result = FetaTTestOneSample(parameters=parameters).run(y=y)
        assert runtime.step == 3
        assert_reference(result, expected_sample, parameters)
        reference = stats.ttest_1samp(expected_sample, mu, alternative=alternative)
        np.testing.assert_allclose(
            [result.t_stat, result.p],
            [reference.statistic, reference.pvalue],
            atol=1e-12,
        )
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


@pytest.mark.parametrize("values", [[], [1.0], [3.0, 3.0, 3.0], [1.0, 4.0]])
def test_small_and_constant_samples(values):
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float)})
    runtime, workers, aggregation = federation([frame])
    with runtime:
        tensors = runtime.initialize(
            Inputdata(data_model="test:1.0", datasets=["a"], variables=["value"]),
            {"y": ["value"]},
        )
        y = tensors["y"]
        if len(values) <= 1:
            with pytest.raises(BadUserInput, match="Not enough observations"):
                FetaTTestOneSample().run(y=y)
        else:
            assert_reference(FetaTTestOneSample().run(y=y), np.asarray(values), {})
    assert not aggregation.rounds
    assert all(not worker.service.store.contexts for worker in workers)


def test_specification_is_separate_and_preprocessing_free():
    specification = FetaTTestOneSample.get_specification()
    assert specification.name == "feta_ttest_onesample"
    assert specification.type == AlgorithmType.FETA
    assert specification.y.max_count == 1
    assert specification.required_preprocessing == []
