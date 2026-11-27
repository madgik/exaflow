from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.exceptions import FedErrorReason
from exaflow.algorithms.federated.exceptions import FedTypeError
from exaflow.algorithms.federated.exceptions import FedValueError
from exaflow.algorithms.federated.utils.agg_client import AggregationClient
from exaflow.algorithms.federated.utils.interfaces import FederatedEstimatorResults
from exaflow.algorithms.federated.utils.interfaces import FederatedTransformer
from exaflow.algorithms.federated.validation import check_consistent_length
from exaflow.algorithms.federated.validation import check_dataframe
from exaflow.algorithms.federated.validation import check_named_steps

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
        check_dataframe(data)
        check_named_steps(self.steps)
        if y is not None:
            check_consistent_length(data, y)
        num_vars = list(numerical_vars or [])
        cat_vars = list(categorical_vars)

        if not self.steps:
            return self

        for name, step in self.steps[:-1]:
            if not isinstance(step, FederatedTransformer):
                raise FedTypeError(
                    FedErrorReason.PIPELINE_TRANSFORMER_REQUIRED,
                    name=name,
                )
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

        if y is None:
            raise FedValueError(FedErrorReason.PIPELINE_TARGET_REQUIRED)

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
                raise FedTypeError(
                    FedErrorReason.PIPELINE_TRANSFORMER_REQUIRED,
                    name=name,
                )
            if not isinstance(X, pd.DataFrame) and idx < len(steps) - 1:
                raise FedTypeError(FedErrorReason.PIPELINE_INTERMEDIATE_TYPE_INVALID)
            X = step.transform(
                X,
                categorical_vars=cat_vars,
                numerical_vars=num_vars,
            )
        if not isinstance(X, np.ndarray):
            raise FedTypeError(FedErrorReason.PIPELINE_OUTPUT_TYPE_INVALID)
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
