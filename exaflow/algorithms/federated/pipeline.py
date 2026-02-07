from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer

PipelineStep = Tuple[str, object]


class FederatedPipeline:
    """Federated pipeline with sklearn-like fit/transform flow."""

    def __init__(self, steps: Iterable[PipelineStep]) -> None:
        self.steps: List[PipelineStep] = list(steps)

    def fit(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> FederatedEstimatorResults | "FederatedPipeline":
        num_vars = list(numerical_vars or [])
        cat_vars = list(categorical_vars)

        if not self.steps:
            return self

        for name, step in self.steps[:-1]:
            if not isinstance(step, FederatedTransformer):
                raise TypeError(f"Pipeline step '{name}' is not a transformer.")
            step.fit(
                agg_client=agg_client,
                data=data,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )

        last_name, last = self.steps[-1]
        if isinstance(last, FederatedTransformer):
            last.fit(
                agg_client=agg_client,
                data=data,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )
            return self

        if isinstance(last, FederatedTransformer):
            last.fit(
                agg_client=agg_client,
                data=data,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )
            return self
        if y is None:
            raise ValueError("Pipeline estimator fit requires y.")

        X = self.transform(
            data,
            categorical_vars=cat_vars,
            numerical_vars=num_vars,
        )
        return last.fit(X, y, agg_client=agg_client)

    def transform(
        self,
        data: pd.DataFrame,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        X = data
        num_vars = list(numerical_vars or [])
        cat_vars = list(categorical_vars)
        steps = self.steps
        if steps and not isinstance(steps[-1][1], FederatedTransformer):
            steps = steps[:-1]
        for idx, (name, step) in enumerate(steps):
            if not isinstance(step, FederatedTransformer):
                raise TypeError(f"Pipeline step '{name}' is not a transformer.")
            if not isinstance(X, pd.DataFrame) and idx < len(steps) - 1:
                raise TypeError(
                    "Pipeline step outputs ndarray before final step. "
                    "Only the last step may output an array."
                )
            X = step.transform(
                X,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )
        if not isinstance(X, np.ndarray):
            raise TypeError("Pipeline transformers must return ndarray.")
        return X

    def fit_transform(
        self,
        *,
        agg_client: AggregationClient,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        self.fit(
            agg_client=agg_client,
            data=data,
            y=None,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
        )
        return self.transform(
            data,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
        )

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        for name, step in reversed(self.steps):
            if isinstance(step, FederatedTransformer):
                return step.get_feature_names_out(
                    categorical_vars=categorical_vars,
                    numerical_vars=numerical_vars,
                )
        return list(numerical_vars or [])
