from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import List

import pandas as pd
from pydantic import ValidationError

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.cluster.kmeans import KMeansReusablePreprocessing
from exaflow.algorithms.exareme3.cluster.kmeans import make_kmeans_input_fingerprint
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.cluster.kmeans import assign_clusters
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput


class KMeansClusterCreator(PreprocessingStep):
    """Replay a fitted KMeans model using its stored centers."""

    def __init__(self, *, params: Dict[str, object]):
        super().__init__(params=params)
        self._code = str(self._params.get("code", ""))
        try:
            self._reusable_preprocessing = KMeansReusablePreprocessing.model_validate(
                self._params.get("reusable_preprocessing")
            )
        except ValidationError as exc:
            raise BadUserInput(
                "'reusable_preprocessing' should be a complete K-means replay payload. "
                f"{exc.errors()[0]['msg']}"
            ) from exc
        self._cluster_variables = self._reusable_preprocessing.cluster_variables
        self._centers = self._reusable_preprocessing.centers

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="kmeans_cluster_creator",
            desc="Creates a categorical KMeans cluster column from fitted centers.",
            documentation=(
                "Uses the centers returned by a K-means analysis to assign each "
                "observation to its nearest cluster. The step only supports the "
                "full output, preserving one category for every fitted cluster. "
                "It replays the fitted model and does not fit K-means again. "
                "Provide the complete reusable_preprocessing output from the "
                "K-means result. Its data model, datasets, filters, and cluster "
                "variables must match the analysis that creates the column. "
                "Every center must contain all clustering features; incomplete "
                "centers are rejected during input validation."
            ),
            label="KMeans Column Creator",
            enabled=True,
            parameters={
                "code": specs.ParameterSpecification(
                    label="New column code",
                    desc="Code/name of the new categorical cluster column.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                ),
                "reusable_preprocessing": specs.ParameterSpecification(
                    label="K-means result",
                    desc="Saved K-means result used to create cluster assignments.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                ),
            },
            output=specs.PreprocessingOutputSpecification(
                type=specs.PreprocessingOutputType.NEW_CATEGORICAL_COLUMN,
                code_parameter="code",
            ),
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        source_context = self._reusable_preprocessing.source_context
        if source_context.data_model != inputdata.data_model:
            raise BadUserInput(
                "K-means replay source data model does not match the current input."
            )
        if sorted(source_context.datasets) != sorted(inputdata.datasets):
            raise BadUserInput(
                "K-means replay source datasets do not match the current input."
            )
        missing_features = sorted(
            set(self._cluster_variables) - set(inputdata.variables)
        )
        if missing_features:
            raise BadUserInput(
                "K-means replay requires clustering features missing from the "
                f"current input: {missing_features}."
            )

        current_fingerprint = make_kmeans_input_fingerprint(
            data_model=inputdata.data_model,
            datasets=inputdata.datasets,
            filters=inputdata.filters,
            feature_names=self._cluster_variables,
        )
        if source_context.input_fingerprint != current_fingerprint:
            raise BadUserInput(
                "K-means replay source does not match the current input filters "
                "or cluster variables, or its fingerprint is invalid."
            )

    def transform_variables(self, *, variables: List[str]) -> List[str]:
        return list(variables) + [self._code]

    def transform_metadata(self, *, metadata: Dict[str, dict]) -> Dict[str, dict]:
        transformed = deepcopy(metadata)
        transformed[self._code] = {
            "code": self._code,
            "label": self._code,
            "sql_type": "text",
            "is_categorical": True,
            "enumerations": {cluster_id: cluster_id for cluster_id in self._centers},
        }
        return transformed

    def transform_data(self, *, data: pd.DataFrame) -> pd.DataFrame:
        centers = [
            [float(center[variable]) for variable in self._cluster_variables]
            for _, center in sorted(self._centers.items(), key=_cluster_sort_key)
        ]
        labels = assign_clusters(data.loc[:, self._cluster_variables], centers)
        cluster_ids = [
            cluster_id
            for cluster_id, _ in sorted(self._centers.items(), key=_cluster_sort_key)
        ]
        data[self._code] = pd.Series(
            [cluster_ids[int(label)] for label in labels],
            index=data.index,
            dtype=object,
        )
        return data


def _cluster_sort_key(item):
    cluster_id, _ = item
    try:
        return int(cluster_id.split("_", 1)[1])
    except (AttributeError, IndexError, ValueError):
        return cluster_id
