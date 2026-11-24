"""Partition, replay and numerical edge-case regressions."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pytest

from exaflow.algorithms.federated.feature_engineering.simple_imputer import (
    SimpleImputer,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    fit as _fit,
)
from tests.standalone_tests.federated_algorithms.feature_engineering.transformation_test_support import (
    transform as _transform,
)
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


class _CountsOnlyClient(SimulatedAggClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.submissions = []

    def union(self, values):
        pytest.fail("Observed values must never enter union.")

    def sum(self, values):
        array = np.asarray(values)
        assert array.dtype.kind in "fiu"
        self.submissions.append(array.copy())
        return super().sum(values)


def _fit_sites(parts, vocabularies):
    coordinator = AggregationCoordinator(len(parts))
    clients = [
        _CountsOnlyClient(worker_id=i, coordinator=coordinator)
        for i in range(len(parts))
    ]

    def fit_one(i):
        try:
            return _fit(
                SimpleImputer("most_frequent", categories=vocabularies[i]),
                parts[i],
                agg_client=clients[i],
            )
        except BadInputError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, i) for i in range(len(parts))]
        try:
            results = [future.result(timeout=20) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise
    return (results, clients)


def test_global_mode_and_replay_use_metadata_order_and_counts_only():
    parts = [
        np.array([["low"]] * 8 + [["high"]] * 2 + [[None]], dtype=object),
        np.array([["low"]] + [["high"]] * 9 + [[None]], dtype=object),
    ]
    models, clients = _fit_sites(parts, [[["low", "high"]]] * 2)
    for model in models:
        assert model.statistics_.tolist() == ["high"]
        replay = SimpleImputer.from_recipe(
            json.loads(json.dumps(asdict(model.recipe_), allow_nan=False))
        )
        assert _transform(replay, [[None]]).tolist() == [["high"]]
        with pytest.raises(BadInputError, match="outside"):
            _transform(replay, [["patient_123"]])
    np.testing.assert_array_equal(clients[0].submissions[-1], [8, 2])
    np.testing.assert_array_equal(clients[1].submissions[-1], [1, 9])


@pytest.mark.parametrize("vocabulary", [[["1.37", "8.21"]], [[1.37, 8.21]]])
def test_real_vs_string_mismatch_is_rejected_by_every_site_before_counts(vocabulary):
    models, clients = _fit_sites(
        [
            np.array([[1.37], [8.21]], dtype=object),
            np.array([["1.37"], ["8.21"]], dtype=object),
        ],
        [vocabulary, vocabulary],
    )
    assert all(
        (
            isinstance(result, BadInputError) and "outside" in str(result)
            for result in models
        )
    )
    assert [len(client.submissions) for client in clients] == [1, 1]


@pytest.mark.parametrize(
    "vocabulary, parts, expected",
    [
        ([["1", "2"]], [[["1"], [None]], [["2"], ["2"]]], "2"),
        ([[1, 2]], [[[1], [None]], [[2.0], [2.0]]], 2),
        ([[True, False]], [[[True], [None]], [[False], [False]]], False),
    ],
)
def test_explicit_categorical_codes_are_accepted(vocabulary, parts, expected):
    models, _ = _fit_sites(parts, [vocabulary] * 2)
    assert all(
        (_transform(model, [[None]]).tolist() == [[expected]] for model in models)
    )


@pytest.mark.parametrize("unknown", ["patient_123", "free text", "9.999", 1.37, ["a"]])
def test_unknown_values_fail_without_sharing_them(unknown):
    parts = [[["a"]], [[unknown]]]
    results, clients = _fit_sites(parts, [[["a", "b"]]] * 2)
    assert all((isinstance(result, BadInputError) for result in results))
    assert [len(client.submissions) for client in clients] == [1, 1]
    assert all((str(unknown) not in str(result) for result in results))


@pytest.mark.parametrize("second", [None, [["b", "a"]], [["a", "b", "c"]]])
def test_missing_or_inconsistent_metadata_fails_collectively(second):
    results, clients = _fit_sites([[["a"]], [["b"]]], [[["a", "b"]], second])
    assert all((isinstance(result, BadInputError) for result in results))
    assert [len(client.submissions) for client in clients] == [1, 1]


def test_all_missing_or_empty_local_site_still_uses_global_mode():
    models, _ = _fit_sites(
        [np.empty((0, 1), dtype=object), [[None], [None]], [["b"], [None]]],
        [[["a", "b"]]] * 3,
    )
    assert all((model.statistics_.tolist() == ["b"] for model in models))


def test_tie_breaking_follows_approved_metadata_order():
    models, _ = _fit_sites([[["a"]], [["b"]]], [[["b", "a"]]] * 2)
    assert all((model.statistics_.tolist() == ["b"] for model in models))


@pytest.mark.parametrize(
    "vocabulary", [[], [["a", "a"]], [[None]], [[np.nan]], [[1, 1.0]], [[{}]]]
)
def test_invalid_metadata_vocabulary_is_bad_input(vocabulary):
    with pytest.raises(BadInputError, match="metadata vocabulary"):
        _fit(
            SimpleImputer("most_frequent", categories=vocabulary),
            [["a"]],
            agg_client=DummyAggClient(),
        )


def test_two_features_keep_their_own_vocabularies_and_global_modes():
    vocab = [["a", "b"], ["yes", "no"]]
    models, _ = _fit_sites(
        [[["a", "yes"], ["a", "no"]], [["b", "no"], [None, None]]], [vocab] * 2
    )
    assert all(
        (
            _transform(model, [[None, None]]).tolist() == [["a", "no"]]
            for model in models
        )
    )


def test_categorical_replay_rejects_unapproved_fitted_mode():
    model = _fit(
        SimpleImputer("most_frequent", categories=[["a", "b"]]),
        [["a"]],
        agg_client=DummyAggClient(),
    )
    recipe = asdict(model.recipe_)
    recipe["statistics"] = ["patient_123"]
    with pytest.raises(BadInputError):
        SimpleImputer.from_recipe(recipe)
