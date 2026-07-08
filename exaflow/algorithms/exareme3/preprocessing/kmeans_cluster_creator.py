from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_ELBOW
from exaflow.algorithms.exareme3.cluster.kmeans import K_SELECTION_MANUAL
from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
from exaflow.algorithms.federated.cluster.kmeans import INIT_MULTI_START_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import INIT_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans
from exaflow.algorithms.federated.cluster.kmeans_privacy import (
    validate_binary_cluster_privacy,
)
from exaflow.algorithms.federated.cluster.kmeans_selection import (
    FederatedKMeansSelector,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker_communication import BadUserInput

OUTPUT_MODE_FULL = "full"
OUTPUT_MODE_BINARY = "binary"
OUTPUT_MODE_SUBSET = "subset"
UNSELECTED_STRATEGY_OTHER = "other"


class KMeansClusterCreator(PreprocessingStep):
    def __init__(
        self,
        *,
        params: Dict[str, object],
    ):
        super().__init__(params=params)
        self._code = str(self._params.get("code", ""))
        self._cluster_variables = [
            str(value) for value in self._params.get("cluster_variables", []) or []
        ]
        self._k_selection = str(self._params.get("k_selection", K_SELECTION_MANUAL))
        self._k = int(self._params.get("k", 4))
        self._k_min = int(self._params.get("k_min", 2))
        self._k_max = int(self._params.get("k_max", 8))
        self._tol = float(self._params.get("tol", 1e-4))
        self._maxiter = int(self._params.get("maxiter", 100))
        self._init_method = str(self._params.get("init_method", INIT_RANDOM_RANGE))
        self._n_init = int(self._params.get("n_init", 5))
        self._output_mode = str(self._params.get("output_mode", OUTPUT_MODE_FULL))
        self._binary_cluster = self._params.get("binary_cluster")
        self._binary_cluster = (
            str(self._binary_cluster) if self._binary_cluster is not None else None
        )
        self._selected_clusters = [
            str(value) for value in self._params.get("selected_clusters", []) or []
        ]
        self._runtime_n_clusters = None
        self._unselected_clusters_strategy = str(
            self._params.get(
                "unselected_clusters_strategy",
                UNSELECTED_STRATEGY_OTHER,
            )
        )

    @classmethod
    def get_specification(cls) -> specs.PreprocessingStepSpecification:
        return specs.PreprocessingStepSpecification(
            name="kmeans_cluster_creator",
            desc="Creates a categorical KMeans cluster column.",
            documentation=(
                "Fits federated K-means on selected numerical variables and "
                "creates a new categorical cluster variable for downstream "
                "algorithms.\n\n"
                "The output_mode controls the generated variable:\n"
                "  - 'full' creates one category per cluster.\n"
                "  - 'binary' creates yes/no membership for one cluster.\n"
                "  - 'subset' keeps selected clusters and maps all other "
                "clusters to 'other'. When exactly one cluster is selected, "
                "the step automatically creates a binary yes/no variable.\n\n"
                "Use init_method='multi_start_random_range' with n_init > 1 to "
                "fit multiple random-range initializations and create the "
                "cluster variable from the lowest-inertia fitted model.\n\n"
                "Exact cluster counts and row labels are not returned in the "
                "API result. Runtime validation rejects binary/subset outputs "
                "when any exposed class is below the privacy threshold."
            ),
            label="KMeans Cluster Creator",
            enabled=True,
            parameters={
                "code": specs.ParameterSpecification(
                    label="New column code",
                    desc="Code/name of the new categorical cluster column.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                ),
                "cluster_variables": specs.ParameterSpecification(
                    label="Cluster variables",
                    desc="Numerical variables used to fit KMeans.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["variables"],
                    ),
                ),
                "k_selection": specs.ParameterSpecification(
                    label="K selection",
                    desc="How to choose the number of clusters.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[K_SELECTION_MANUAL, K_SELECTION_ELBOW],
                    ),
                    default=K_SELECTION_MANUAL,
                ),
                "k": specs.ParameterSpecification(
                    label="Number of clusters",
                    desc="Number of clusters when k_selection is manual.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=4,
                    min=1,
                    max=100,
                ),
                "k_min": specs.ParameterSpecification(
                    label="Minimum K",
                    desc="Smallest k evaluated by elbow selection.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=2,
                    min=1,
                    max=100,
                ),
                "k_max": specs.ParameterSpecification(
                    label="Maximum K",
                    desc="Largest k evaluated by elbow selection.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=8,
                    min=1,
                    max=100,
                ),
                "maxiter": specs.ParameterSpecification(
                    label="Maximum iterations",
                    desc="Maximum number of fitting iterations.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=100,
                    min=1,
                    max=100,
                ),
                "tol": specs.ParameterSpecification(
                    label="Convergence tolerance",
                    desc="Tolerance used to decide convergence.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.0001,
                    min=0.0,
                    max=1.0,
                ),
                "output_mode": specs.ParameterSpecification(
                    label="Output mode",
                    desc="Cluster variable output mode.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[
                            OUTPUT_MODE_FULL,
                            OUTPUT_MODE_BINARY,
                            OUTPUT_MODE_SUBSET,
                        ],
                    ),
                    default=OUTPUT_MODE_FULL,
                ),
                "init_method": specs.ParameterSpecification(
                    label="Initialization method",
                    desc=(
                        "How initial centers are generated. 'random_range' "
                        "uses one random draw from global feature ranges. "
                        "'multi_start_random_range' tries multiple random-range "
                        "initializations and keeps the lowest-inertia result."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[INIT_RANDOM_RANGE, INIT_MULTI_START_RANDOM_RANGE],
                    ),
                    default=INIT_RANDOM_RANGE,
                ),
                "n_init": specs.ParameterSpecification(
                    label="Number of initializations",
                    desc=(
                        "Number of random-range initializations evaluated when "
                        "init_method is multi_start_random_range."
                    ),
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=5,
                    min=1,
                    max=20,
                ),
                "binary_cluster": specs.ParameterSpecification(
                    label="Binary cluster",
                    desc="Cluster id used when output_mode is binary.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                ),
                "selected_clusters": specs.ParameterSpecification(
                    label="Selected clusters",
                    desc="Cluster ids used when output_mode is subset.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=True,
                ),
                "unselected_clusters_strategy": specs.ParameterSpecification(
                    label="Unselected clusters strategy",
                    desc="How subset mode handles clusters not selected.",
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=[UNSELECTED_STRATEGY_OTHER],
                    ),
                    default=UNSELECTED_STRATEGY_OTHER,
                ),
            },
            output=specs.PreprocessingOutputSpecification(
                type=specs.PreprocessingOutputType.NEW_CATEGORICAL_COLUMN,
                code_parameter="code",
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
        if not self._code.strip():
            raise BadUserInput("'code' parameter should not be blank.")
        if self._code in metadata:
            raise BadUserInput(
                f"Preprocessing step 'kmeans_cluster_creator' cannot create CDE '{self._code}' because it already exists."
            )
        if not self._cluster_variables:
            raise BadUserInput("'cluster_variables' parameter should not be blank.")

        requested_vars = set(inputdata.variables)
        unknown_vars = sorted(set(self._cluster_variables) - requested_vars)
        if unknown_vars:
            raise BadUserInput(
                "'cluster_variables' includes variables not present in inputdata.variables: "
                f"{unknown_vars}."
            )
        for variable in self._cluster_variables:
            if metadata.get(variable, {}).get("is_categorical"):
                raise BadUserInput(
                    f"KMeans cluster variable '{variable}' must be numerical."
                )

        if self._k_selection not in {K_SELECTION_MANUAL, K_SELECTION_ELBOW}:
            raise BadUserInput(
                f"'k_selection' should be '{K_SELECTION_MANUAL}' or '{K_SELECTION_ELBOW}'."
            )
        if self._k < 1:
            raise BadUserInput("'k' should be greater than or equal to 1.")
        if self._k_min < 1:
            raise BadUserInput("'k_min' should be greater than or equal to 1.")
        if self._k_max < self._k_min:
            raise BadUserInput("'k_max' should be greater than or equal to 'k_min'.")
        if self._output_mode not in {
            OUTPUT_MODE_FULL,
            OUTPUT_MODE_BINARY,
            OUTPUT_MODE_SUBSET,
        }:
            raise BadUserInput(
                f"'output_mode' should be one of: {[OUTPUT_MODE_FULL, OUTPUT_MODE_BINARY, OUTPUT_MODE_SUBSET]}."
            )
        if self._init_method not in {INIT_RANDOM_RANGE, INIT_MULTI_START_RANDOM_RANGE}:
            raise BadUserInput(
                f"'init_method' should be one of: {[INIT_RANDOM_RANGE, INIT_MULTI_START_RANDOM_RANGE]}."
            )
        if self._n_init < 1:
            raise BadUserInput("'n_init' should be greater than or equal to 1.")
        if self._unselected_clusters_strategy != UNSELECTED_STRATEGY_OTHER:
            raise BadUserInput(
                f"'unselected_clusters_strategy' should be '{UNSELECTED_STRATEGY_OTHER}'."
            )
        self._validate_output_mode_configuration()

    def transform_variables(
        self,
        *,
        variables: List[str],
    ) -> List[str]:
        return list(variables) + [self._code]

    def transform_metadata(
        self,
        *,
        metadata: Dict[str, dict],
    ) -> Dict[str, dict]:
        if self._runtime_n_clusters is not None:
            return self._transform_runtime_metadata(
                metadata=metadata,
                n_clusters=self._runtime_n_clusters,
            )
        transformed_metadata = deepcopy(metadata)
        transformed_metadata[self._code] = _new_categorical_metadata(
            code=self._code,
            enumerations=self._enumerations(),
        )
        return transformed_metadata

    def transform_data(
        self,
        *,
        data: pd.DataFrame,
        agg_client,
    ) -> pd.DataFrame:
        self._runtime_n_clusters = None
        if agg_client is None:
            raise BadUserInput(
                "kmeans_cluster_creator requires an aggregation server during preprocessing."
            )

        model = self._fit_model(data=data, agg_client=agg_client)
        output = self._build_output_series(
            labels=model.labels_,
            counts=model.cluster_counts_,
            index=data.index,
        )
        data[self._code] = output
        self._runtime_n_clusters = int(model.n_clusters)
        return data

    def _fit_model(self, *, data: pd.DataFrame, agg_client):
        X = data.loc[:, self._cluster_variables]
        if self._k_selection == K_SELECTION_MANUAL:
            return FederatedKMeans(
                agg_client=agg_client,
                n_clusters=self._k,
                init_method=self._init_method,
                n_init=self._n_init,
                tol=self._tol,
                maxiter=self._maxiter,
            ).fit(X, feature_names=self._cluster_variables)

        selector = FederatedKMeansSelector(
            agg_client=agg_client,
            k_min=self._k_min,
            k_max=self._k_max,
            init_method=self._init_method,
            n_init=self._n_init,
            tol=self._tol,
            maxiter=self._maxiter,
        ).fit(X, feature_names=self._cluster_variables)
        return selector.best_model_

    def _build_output_series(self, *, labels, counts, index):
        from exaflow.worker import config as worker_config

        effective_mode = self._effective_output_mode()
        if effective_mode == OUTPUT_MODE_FULL:
            self._validate_full_privacy(counts)
            return pd.Series(
                [f"cluster_{int(label)}" for label in labels],
                index=index,
                dtype=object,
            )
        if effective_mode == OUTPUT_MODE_BINARY:
            selected_idx = self._binary_cluster_index()
            self._validate_cluster_index(selected_idx, len(counts))
            selected_count = int(counts[selected_idx])
            other_count = int(sum(counts) - selected_count)
            try:
                validate_binary_cluster_privacy(
                    selected_count=selected_count,
                    other_count=other_count,
                    minimum_row_count=worker_config.privacy.minimum_row_count,
                )
            except ValueError as exc:
                raise BadUserInput(str(exc)) from exc
            return pd.Series(
                ["yes" if int(label) == selected_idx else "no" for label in labels],
                index=index,
                dtype=object,
            )

        selected_indices = self._selected_cluster_indices()
        for selected_idx in selected_indices:
            self._validate_cluster_index(selected_idx, len(counts))
        self._validate_subset_privacy(counts, selected_indices)
        selected_set = set(selected_indices)
        return pd.Series(
            [
                f"cluster_{int(label)}" if int(label) in selected_set else "other"
                for label in labels
            ],
            index=index,
            dtype=object,
        )

    def _validate_full_privacy(self, counts):
        from exaflow.worker import config as worker_config

        small_clusters = [
            f"cluster_{idx}"
            for idx, count in enumerate(counts)
            if int(count) < worker_config.privacy.minimum_row_count
        ]
        if small_clusters:
            raise BadUserInput(
                "Cannot create full KMeans cluster variable because clusters are "
                f"below the privacy threshold: {small_clusters}."
            )

    def _validate_subset_privacy(self, counts, selected_indices):
        from exaflow.worker import config as worker_config

        minimum_row_count = worker_config.privacy.minimum_row_count
        small_selected = [
            f"cluster_{idx}"
            for idx in selected_indices
            if int(counts[idx]) < minimum_row_count
        ]
        other_count = int(
            sum(
                count
                for idx, count in enumerate(counts)
                if idx not in set(selected_indices)
            )
        )
        if small_selected or other_count < minimum_row_count:
            raise BadUserInput(
                "Cannot create subset KMeans cluster variable because one exposed "
                "class is below the privacy threshold."
            )

    def _effective_output_mode(self):
        if (
            self._output_mode == OUTPUT_MODE_SUBSET
            and len(self._selected_clusters) == 1
        ):
            return OUTPUT_MODE_BINARY
        return self._output_mode

    def _validate_output_mode_configuration(self):
        effective_mode = self._effective_output_mode()
        if effective_mode == OUTPUT_MODE_BINARY and not self._binary_cluster_for_mode():
            raise BadUserInput(
                "'binary_cluster' is required for binary output mode. In subset "
                "mode, provide exactly one selected cluster to create a binary output."
            )
        if self._output_mode == OUTPUT_MODE_SUBSET and not self._selected_clusters:
            raise BadUserInput(
                "'selected_clusters' is required for subset output mode."
            )
        self._validate_configured_cluster_ids()

    def _validate_configured_cluster_ids(self):
        effective_mode = self._effective_output_mode()
        if effective_mode == OUTPUT_MODE_BINARY:
            cluster_ids = [self._binary_cluster_for_mode()]
        elif effective_mode == OUTPUT_MODE_SUBSET:
            cluster_ids = list(self._selected_clusters)
        else:
            cluster_ids = []

        if len(set(cluster_ids)) != len(cluster_ids):
            raise BadUserInput("KMeans cluster ids should not contain duplicates.")

        k_upper_bound = (
            self._k if self._k_selection == K_SELECTION_MANUAL else self._k_max
        )
        for cluster_id in cluster_ids:
            cluster_idx = _parse_cluster_id(cluster_id)
            self._validate_cluster_index(cluster_idx, k_upper_bound)

    def _binary_cluster_for_mode(self):
        if (
            self._output_mode == OUTPUT_MODE_SUBSET
            and len(self._selected_clusters) == 1
        ):
            return self._selected_clusters[0]
        return self._binary_cluster

    def _binary_cluster_index(self):
        return _parse_cluster_id(self._binary_cluster_for_mode())

    def _selected_cluster_indices(self):
        return [_parse_cluster_id(cluster) for cluster in self._selected_clusters]

    def _validate_cluster_index(self, cluster_idx, n_clusters):
        if cluster_idx < 0 or cluster_idx >= n_clusters:
            raise BadUserInput(
                f"Cluster id 'cluster_{cluster_idx}' is outside the fitted KMeans cluster range."
            )

    def _enumerations(self):
        effective_mode = self._effective_output_mode()
        if effective_mode == OUTPUT_MODE_BINARY:
            return {"yes": "yes", "no": "no"}
        if effective_mode == OUTPUT_MODE_SUBSET:
            return {
                **{cluster: cluster for cluster in self._selected_clusters},
                "other": "other",
            }

        k_upper_bound = (
            self._k if self._k_selection == K_SELECTION_MANUAL else self._k_max
        )
        return {f"cluster_{idx}": f"cluster_{idx}" for idx in range(k_upper_bound)}

    def _transform_runtime_metadata(
        self,
        *,
        metadata: Dict[str, dict],
        n_clusters: int,
    ) -> Dict[str, dict]:
        transformed_metadata = deepcopy(metadata)
        transformed_metadata[self._code] = _new_categorical_metadata(
            code=self._code,
            enumerations=self._runtime_enumerations(n_clusters=n_clusters),
        )
        return transformed_metadata

    def _runtime_enumerations(self, *, n_clusters: int):
        effective_mode = self._effective_output_mode()
        if effective_mode != OUTPUT_MODE_FULL:
            return self._enumerations()
        return {f"cluster_{idx}": f"cluster_{idx}" for idx in range(n_clusters)}


def _new_categorical_metadata(*, code: str, enumerations: Dict[str, str]) -> dict:
    return {
        "code": code,
        "label": code,
        "sql_type": "text",
        "is_categorical": True,
        "enumerations": enumerations,
    }


def _parse_cluster_id(value: str) -> int:
    if not isinstance(value, str) or not value.startswith("cluster_"):
        raise BadUserInput(
            f"Cluster id should use the format 'cluster_<number>'. Value provided: {value!r}."
        )
    try:
        return int(value.split("_", 1)[1])
    except ValueError as exc:
        raise BadUserInput(
            f"Cluster id should use the format 'cluster_<number>'. Value provided: {value!r}."
        ) from exc
