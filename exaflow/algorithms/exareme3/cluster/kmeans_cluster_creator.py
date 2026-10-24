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
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans import assign_clusters
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.output_specifications import resolve_output_metadata
from exaflow.worker_communication import BadUserInput


class KMeansClusterCreator(PreprocessingStep):
    """Replay a fitted KMeans model using its stored centers."""

    def __init__(self, *, params: Dict[str, object]):
        super().__init__(params=params)
        declared = resolve_output_metadata(self.get_specification(), self._params, {})
        self._code = next(iter(declared))
        self._cluster_ids = list(declared[self._code]["enumerations"])
        self._reusable_preprocessing = self._parse_reusable_preprocessing()
        self._cluster_variables = self._resolve_cluster_variables()
        self._centers = None
        if self._reusable_preprocessing is not None:
            self._centers = self._reusable_preprocessing.centers
            if (
                set(self._cluster_ids) != set(self._centers)
                or self._params.get("enumeration_prefix", "cluster_")
                != self._reusable_preprocessing.enumeration_prefix
            ):
                raise BadUserInput(
                    "K-means enumeration count or prefix does not match the replay model."
                )

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="kmeans_cluster_creator",
            desc="Creates a categorical KMeans cluster column from fitted centers.",
            documentation=(
                "Creates one categorical cluster column either by replaying a "
                "stored K-means result or by fitting K-means directly on explicitly "
                "selected numerical variables. The full output preserves one "
                "category for every fitted cluster."
            ),
            label="KMeans Column Creator",
            enabled=True,
            parameters={
                "new_column_names": specs.ParameterSpecification(
                    label="New column names",
                    desc="Names assigned to the cluster columns.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                ),
                "variables": specs.ParameterSpecification(
                    label="Clustering variables",
                    desc="Numerical variables used when no K-means result is provided.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=True,
                ),
                "k": specs.ParameterSpecification(
                    label="Number of clusters",
                    desc="Number of clusters in the fitted model.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    min=1,
                ),
                "enumeration_prefix": specs.ParameterSpecification(
                    label="Cluster category prefix",
                    desc="Prefix used to name the fitted cluster categories.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default="cluster_",
                ),
                "reusable_preprocessing": specs.ParameterSpecification(
                    label="K-means result",
                    desc="Optional saved K-means result used to create cluster assignments.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                ),
                "init_method": specs.ParameterSpecification(
                    label="Initialization method",
                    desc="Method used to initialize K-means centers when fitting.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="random_range",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["random_range", "multi_start_random_range"],
                    ),
                ),
                "n_init": specs.ParameterSpecification(
                    label="Number of initializations",
                    desc="Number of K-means initializations used when fitting.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=1,
                    min=1,
                ),
                "tol": specs.ParameterSpecification(
                    label="Convergence tolerance",
                    desc="Tolerance used to stop K-means fitting.",
                    types=[specs.ParameterType.REAL],
                    required=False,
                    multiple=False,
                    default=0.0001,
                    min=0,
                ),
                "maxiter": specs.ParameterSpecification(
                    label="Maximum iterations",
                    desc="Maximum number of K-means fitting iterations.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=100,
                    min=1,
                ),
                "random_state": specs.ParameterSpecification(
                    label="Random seed",
                    desc="Seed used for reproducible K-means initialization.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=123,
                    min=0,
                ),
            },
            outputs=specs.PreprocessingOutputsSpecification(
                new_column_names_param="new_column_names",
                columns=[
                    specs.PreprocessingOutputColumnSpecification(
                        type=specs.InputDataType.TEXT,
                        stattype=specs.InputDataStatType.NOMINAL,
                        enumerations=specs.PreprocessingOutputEnumerationsSpecification(
                            source_parameter_name="k",
                            source_type=specs.PreprocessingOutputEnumerationSource.PREFIXED_RANGE,
                            prefix_parameter_name="enumeration_prefix",
                        ),
                    )
                ],
            ),
            type=specs.PreprocessingStepType.EXAREME3_PREPROCESSING_STEP,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def validate_params(
        self,
        *,
        inputdata: Inputdata,
        metadata: Dict[str, dict],
    ) -> None:
        if self._reusable_preprocessing is None:
            self._validate_independent_variables(inputdata=inputdata, metadata=metadata)
            return

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
        transformed.update(
            resolve_output_metadata(self.get_specification(), self._params, metadata)
        )
        return transformed

    def transform_data(self, *, data: pd.DataFrame, agg_client=None) -> pd.DataFrame:
        if self._centers is None:
            if agg_client is None:
                raise BadUserInput(
                    "An aggregation client is required when fitting K-means "
                    "without reusable_preprocessing."
                )
            model = FederatedKMeans(
                n_clusters=self._params["k"],
                init_method=self._params.get("init_method", "random_range"),
                n_init=self._params.get("n_init", 1),
                tol=self._params.get("tol", 1e-4),
                maxiter=self._params.get("maxiter", 100),
                random_state=self._params.get("random_state", 123),
            ).fit(
                data.loc[:, self._cluster_variables],
                agg_client=agg_client,
                feature_names=self._cluster_variables,
            )
            centers = model.cluster_centers_
        else:
            centers = [
                [
                    float(self._centers[cluster_id][variable])
                    for variable in self._cluster_variables
                ]
                for cluster_id in self._cluster_ids
            ]
        labels = assign_clusters(data.loc[:, self._cluster_variables], centers)
        cluster_ids = self._cluster_ids
        data[self._code] = pd.Series(
            [cluster_ids[int(label)] for label in labels],
            index=data.index,
            dtype=object,
        )
        return data

    def _parse_reusable_preprocessing(self):
        if self._params.get("reusable_preprocessing") is None:
            return None
        try:
            return KMeansReusablePreprocessing.model_validate(
                self._params["reusable_preprocessing"]
            )
        except ValidationError as exc:
            raise BadUserInput(
                "'reusable_preprocessing' should be a complete K-means replay payload. "
                f"{exc.errors()[0]['msg']}"
            ) from exc

    def _resolve_cluster_variables(self):
        if self._reusable_preprocessing is not None:
            return self._reusable_preprocessing.cluster_variables
        variables = self._params.get("variables")
        if not isinstance(variables, list) or not variables:
            raise BadUserInput(
                "'variables' should contain at least one variable when "
                "'reusable_preprocessing' is not provided."
            )
        return variables

    def _validate_independent_variables(self, *, inputdata, metadata):
        if len(self._cluster_variables) != len(set(self._cluster_variables)):
            raise BadUserInput("'variables' should not contain duplicates.")
        missing_variables = [
            variable
            for variable in self._cluster_variables
            if variable not in inputdata.variables
        ]
        if missing_variables:
            raise BadUserInput(
                "K-means variables are not present in inputdata.variables: "
                f"{missing_variables}."
            )
        missing_metadata = [
            variable for variable in self._cluster_variables if variable not in metadata
        ]
        if missing_metadata:
            raise BadUserInput(
                f"K-means variables are missing from metadata: {missing_metadata}."
            )
