"""Present existing numerical regression fixtures through the named-column API."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.feature_engineering.transformer import (
    FeatureTransformer,
)
from exaflow.algorithms.federated.utils import BadInputError
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    AggregationCoordinator,
)
from tests.standalone_tests.federated_algorithms.utils.simulated_agg_client import (
    SimulatedAggClient,
)


def _frame_and_columns(model, values):
    # Preserve nested categorical cells for collective validation by the primitive.
    values = (
        values if isinstance(values, np.ndarray) else pd.DataFrame(values).to_numpy()
    )
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise BadInputError("Input must be one- or two-dimensional.")
    names = [f"f{i}" for i in range(values.shape[1])]
    categorical = list(getattr(model, "categorical_features_", []))
    if getattr(model, "strategy", None) == "most_frequent":
        categorical = names
    if getattr(model, "strategy", None) == "constant" and values.dtype.kind in "OUS":
        observed = values[~pd.isna(values)]
        if any(isinstance(value, str) for value in observed):
            categorical = names
    return pd.DataFrame(values, columns=names), {
        "categorical_vars": categorical,
        "numerical_vars": [name for name in names if name not in categorical],
    }


def fit(model, values, *, agg_client=None):
    if not isinstance(model, FeatureTransformer):
        return model.fit(values)
    frame, columns = _frame_and_columns(model, values)
    model.fit(agg_client=agg_client, data=frame, **columns)
    return model


def transform(model, values):
    if not isinstance(model, FeatureTransformer):
        return model.transform(values)
    frame, columns = _frame_and_columns(model, values)
    result = model.transform(frame, **columns)
    return result[:, 0] if np.asarray(values).ndim == 1 else result


def fit_transform(model, values, *, agg_client=None):
    return transform(fit(model, values, agg_client=agg_client), values)


def fit_in_workers(factory, parts):
    coordinator = AggregationCoordinator(len(parts))

    def fit_one(item):
        worker_id, part = item
        transformer = factory()
        try:
            fit(
                transformer,
                part,
                agg_client=SimulatedAggClient(
                    worker_id=worker_id, coordinator=coordinator
                ),
            )
        except Exception as exc:
            coordinator.abort(exc)
            raise
        return transformer

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [executor.submit(fit_one, item) for item in enumerate(parts)]
        try:
            return [future.result(timeout=30) for future in futures]
        except Exception as exc:
            coordinator.abort(exc)
            raise
