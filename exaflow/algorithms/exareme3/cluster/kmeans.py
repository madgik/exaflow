import hashlib
import json
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pydantic import model_validator

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.cluster.kmeans import INIT_MULTI_START_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import INIT_RANDOM_RANGE
from exaflow.algorithms.federated.cluster.kmeans import FederatedKMeans

K_SELECTION_MANUAL = "manual"
K_SELECTION_ELBOW = "elbow"


class KMeansClusterQuality(BaseModel):
    compactness: Optional[str] = None


class KMeansClusterReport(BaseModel):
    cluster_id: str
    label: str
    size_interval: str
    center: Dict[str, float]
    profile: List[str]
    interpretation: str
    quality: KMeansClusterQuality


class KMeansElbowReport(BaseModel):
    k_min: int
    k_max: int
    selected_k: int
    inertia_by_k: Dict[int, float]
    warning: Optional[str] = None


class KMeansReplaySourceContext(BaseModel):
    data_model: str
    datasets: List[str]
    input_fingerprint: str


class KMeansReusableOutput(BaseModel):
    output_mode: str
    semantic_operation: str
    variable_type: str
    number_of_variables: int
    eligible_roles: List[str]
    cardinality: Optional[int] = None
    selection_rule: str


class KMeansClusterChoice(BaseModel):
    cluster_id: str
    label: str


class KMeansReusablePreprocessing(BaseModel):
    schema_version: str
    preprocessing_name: str
    cluster_variables: List[str]
    centers: Dict[str, Dict[str, float]]
    source_context: KMeansReplaySourceContext
    available_outputs: List[KMeansReusableOutput]
    cluster_choices: List[KMeansClusterChoice]

    @model_validator(mode="after")
    def validate_center_features(self):
        required_features = set(self.cluster_variables)
        for cluster_id, center in self.centers.items():
            missing_features = sorted(required_features - set(center))
            if missing_features:
                raise ValueError(
                    f"Center '{cluster_id}' is missing required clustering "
                    f"features: {missing_features}."
                )
        return self


def build_kmeans_preprocessing_parameters(
    reusable_preprocessing: KMeansReusablePreprocessing,
    *,
    code: str,
):
    return {
        "code": code,
        "reusable_preprocessing": reusable_preprocessing.model_dump(),
    }


class KMeansResult(BaseModel):
    title: str
    result_type: str
    variables: List[str]
    k_selection: str
    selected_k: int
    initialization_method: str
    n_init: int
    selected_initialization: int
    n_obs_interval: str
    center_definition: str
    intended_use: List[str]
    privacy_note: str
    clusters: List[KMeansClusterReport]
    elbow: Optional[KMeansElbowReport] = None
    converged: bool
    n_iter: int
    warnings: List[str]
    limitations: List[str]
    reusable_preprocessing: Optional[KMeansReusablePreprocessing] = None


class KMeans(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="kmeans",
            desc="K-means clustering report for numerical variables.",
            documentation=(
                "Partitions observations into k clusters based on selected "
                "numerical variables and returns a privacy-safe cluster report. "
                "Cluster centers are statistical mean profiles, not individual "
                "patients.\n\n"
                "Use k_selection='manual' to provide k directly. Use "
                "k_selection='elbow' to evaluate k_min..k_max and select k from "
                "the inertia curve. Elbow selection uses the K-means objective "
                "(within-cluster sum of squared distances), not a user-defined "
                "cost function. Use init_method='multi_start_random_range' with "
                "n_init > 1 to fit multiple random-range initializations and keep "
                "the lowest-inertia result.\n\n"
                "The report always includes named cluster centers. Cluster sizes "
                "are returned as privacy-safe intervals, not exact counts.\n\n"
                "K-means can support exploratory subgroup discovery and baseline "
                "phenotype grouping. It is not a diagnostic model, a causal "
                "subgroup discovery method, or a validated clinical risk model "
                "without external clinical evaluation."
            ),
            label="K-means",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Variables",
                desc="Numerical variables used for clustering.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
            ),
            parameters={
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
                    desc="Number of clusters to fit when k_selection is manual.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=4,
                    min=1,
                    max=100,
                ),
                "k_min": specs.ParameterSpecification(
                    label="Minimum K",
                    desc="Smallest number of clusters evaluated by elbow selection.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=2,
                    min=1,
                    max=100,
                ),
                "k_max": specs.ParameterSpecification(
                    label="Maximum K",
                    desc="Largest number of clusters evaluated by elbow selection.",
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
                "random_state": specs.ParameterSpecification(
                    label="Random seed",
                    desc="Seed used to generate reproducible initial centers.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=123,
                    min=0,
                    max=4294967276,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        k_selection = str(self.get_parameter("k_selection", K_SELECTION_MANUAL))
        k = int(self.get_parameter("k", 4))
        k_min = int(self.get_parameter("k_min", 2))
        k_max = int(self.get_parameter("k_max", 8))
        tol = float(self.get_parameter("tol", 1e-4))
        maxiter = int(self.get_parameter("maxiter", 100))
        init_method = str(self.get_parameter("init_method", INIT_RANDOM_RANGE))
        n_init = int(self.get_parameter("n_init", 5))
        random_state = int(self.get_parameter("random_state", 123))

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "variables": self.y,
                "k_selection": k_selection,
                "n_clusters": k,
                "k_min": k_min,
                "k_max": k_max,
                "tol": tol,
                "maxiter": maxiter,
                "init_method": init_method,
                "n_init": n_init,
                "random_state": random_state,
            },
            identical_results=True,
        )

        result_model = KMeansResult(**result)
        result_model.reusable_preprocessing = _build_reusable_preprocessing(
            result=result_model,
            cluster_variables=self.y,
            inputdata=self.inputdata,
        )
        return result_model


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client,
    data,
    variables,
    k_selection,
    n_clusters,
    k_min,
    k_max,
    tol,
    maxiter,
    init_method=INIT_RANDOM_RANGE,
    n_init=5,
    random_state=123,
):
    from exaflow.worker import config as worker_config

    X = data.loc[:, variables]
    global_mean = _compute_global_mean(
        agg_client=agg_client,
        data=X,
    )
    if k_selection == K_SELECTION_MANUAL:
        model = FederatedKMeans(
            n_clusters=n_clusters,
            init_method=init_method,
            n_init=n_init,
            tol=tol,
            maxiter=maxiter,
            random_state=random_state,
        ).fit(X, agg_client=agg_client, feature_names=variables)
        elbow = None
    elif k_selection == K_SELECTION_ELBOW:
        selection = FederatedKMeans.select_k(
            X,
            k_min=k_min,
            k_max=k_max,
            agg_client=agg_client,
            init_method=init_method,
            n_init=n_init,
            tol=tol,
            maxiter=maxiter,
            random_state=random_state,
            feature_names=variables,
        )
        model = selection.best_model
        elbow = {
            "k_min": k_min,
            "k_max": k_max,
            "selected_k": selection.selected_k,
            "inertia_by_k": _round_inertia_by_k(selection.inertia_by_k),
            "warning": selection.warning,
        }
    else:
        raise ValueError(f"Unsupported k_selection: '{k_selection}'.")

    return _build_report(
        model=model,
        variables=variables,
        k_selection=k_selection,
        global_mean=global_mean,
        minimum_row_count=worker_config.privacy.minimum_row_count,
        elbow=elbow,
    )


def _compute_global_mean(*, agg_client, data):
    X = data.to_numpy(dtype=float, copy=False)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_local = float(X.shape[0])
    local_sum = (
        np.einsum("ij->j", X) if n_local > 0 else np.zeros((X.shape[1],), dtype=float)
    )
    total_sum = np.asarray(agg_client.sum(local_sum), dtype=float)
    total_n = float(np.asarray(agg_client.sum([n_local]), dtype=float)[0])
    if total_n <= 0:
        return np.zeros((X.shape[1],), dtype=float)
    return total_sum / total_n


def _build_reusable_preprocessing(
    *,
    result: KMeansResult,
    cluster_variables,
    inputdata,
):
    selected_k = int(result.selected_k)
    centers = {cluster.cluster_id: cluster.center for cluster in result.clusters}
    available_outputs = [
        KMeansReusableOutput(
            output_mode="full",
            semantic_operation="all_clusters",
            variable_type="nominal",
            number_of_variables=1,
            eligible_roles=["predictor"],
            cardinality=selected_k,
            selection_rule="all clusters",
        )
    ]

    return KMeansReusablePreprocessing(
        schema_version="1",
        preprocessing_name="kmeans_cluster_creator",
        cluster_variables=list(cluster_variables),
        centers=centers,
        source_context=KMeansReplaySourceContext(
            data_model=inputdata.data_model,
            datasets=list(inputdata.datasets),
            input_fingerprint=make_kmeans_input_fingerprint(
                data_model=inputdata.data_model,
                datasets=inputdata.datasets,
                filters=inputdata.filters,
                feature_names=cluster_variables,
            ),
        ),
        available_outputs=available_outputs,
        cluster_choices=[
            KMeansClusterChoice(cluster_id=cluster.cluster_id, label=cluster.label)
            for cluster in result.clusters
        ],
    )


def make_kmeans_input_fingerprint(*, data_model, datasets, filters, feature_names):
    payload = {
        "data_model": data_model,
        "datasets": sorted(str(dataset) for dataset in datasets),
        "filters": filters,
        "feature_names": list(feature_names),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_report(
    *,
    model,
    variables,
    k_selection,
    global_mean,
    minimum_row_count,
    elbow,
):
    size_intervals = [
        _count_interval(count, minimum_row_count=minimum_row_count)
        for count in model.cluster_counts_
    ]
    n_obs_interval = _count_interval(model.n_obs_, minimum_row_count=minimum_row_count)
    compactness = _compactness_labels(
        counts=model.cluster_counts_,
        cluster_inertia=model.cluster_inertia_,
        can_show=[True] * len(model.cluster_counts_),
    )
    centers = np.asarray(model.cluster_centers_, dtype=float)
    clusters = []
    warnings = []
    for cluster_idx, size_interval in enumerate(size_intervals):
        center_values = centers[cluster_idx]
        center = _named_center(variables=variables, center=center_values)
        profile = _profile_cluster(
            variables=variables,
            center=center_values,
            global_mean=global_mean,
        )
        quality = KMeansClusterQuality(compactness=compactness.get(cluster_idx))
        interpretation = _cluster_interpretation(
            cluster_id=f"cluster_{cluster_idx}",
            profile=profile,
            compactness=quality.compactness,
        )
        clusters.append(
            KMeansClusterReport(
                cluster_id=f"cluster_{cluster_idx}",
                label=f"Cluster {cluster_idx}",
                size_interval=size_interval,
                center=center,
                profile=profile,
                interpretation=interpretation,
                quality=quality,
            )
        )

    if not model.converged_:
        warnings.append("K-means reached maxiter before convergence.")
    if model.empty_clusters_:
        warnings.append(
            "Empty clusters were produced: "
            + ", ".join(f"cluster_{idx}" for idx in model.empty_clusters_)
            + "."
        )

    return {
        "title": "K-Means Cluster Report",
        "result_type": "privacy_safe_cluster_report",
        "variables": list(variables),
        "k_selection": k_selection,
        "selected_k": int(model.n_clusters),
        "initialization_method": str(model.init_method_),
        "n_init": int(model.n_init_),
        "selected_initialization": int(model.best_init_),
        "n_obs_interval": n_obs_interval,
        "center_definition": (
            "Each center is the per-variable mean of observations assigned to "
            "that cluster. It is not a real patient or row."
        ),
        "intended_use": [
            "Explore baseline covariate patterns.",
            "Create clinically reviewed subgroup hypotheses.",
            "Optionally create a downstream categorical covariate with kmeans_cluster_creator.",
        ],
        "privacy_note": (
            "Cluster sizes are reported as privacy-safe intervals. Cluster centers "
            "are always returned as model summaries, not individual records."
        ),
        "clusters": [cluster.model_dump() for cluster in clusters],
        "elbow": elbow,
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "warnings": warnings,
        "limitations": [
            "K-means clusters are not diagnoses or validated risk groups.",
            "Clinical interpretation must come from domain review of the selected variables.",
            "Feature scaling, outliers, and the selected variables can change the clusters.",
        ],
    }


def _count_interval(count, *, minimum_row_count):
    count = int(count)
    minimum_row_count = int(minimum_row_count)
    if minimum_row_count <= 0:
        raise ValueError("minimum_row_count must be positive.")
    if count == 0:
        return "0"
    if count < minimum_row_count:
        return f"<{minimum_row_count}"
    lower = (count // minimum_row_count) * minimum_row_count
    return f"{lower}-{lower + minimum_row_count - 1}"


def _named_center(*, variables, center):
    return {variable: _round_float(value) for variable, value in zip(variables, center)}


def _profile_cluster(*, variables, center, global_mean):
    profile = []
    for variable, value, mean in zip(variables, center, global_mean):
        if np.isclose(value, mean, rtol=1e-6, atol=1e-9):
            relation = "close to the overall mean"
        elif value > mean:
            relation = "above the overall mean"
        else:
            relation = "below the overall mean"
        profile.append(f"{variable} is {relation}")
    return profile


def _cluster_interpretation(*, cluster_id, profile, compactness):
    compactness_text = (
        f" Compactness is {compactness} relative to the visible clusters."
        if compactness
        else ""
    )
    if not profile:
        return f"{cluster_id} has no visible profile statements.{compactness_text}"
    return (
        f"{cluster_id} is summarized by: " + "; ".join(profile) + "." + compactness_text
    )


def _compactness_labels(*, counts, cluster_inertia, can_show):
    avg_inertia = {}
    for idx, (count, inertia, visible) in enumerate(
        zip(counts, cluster_inertia, can_show)
    ):
        if not visible or count <= 0:
            continue
        avg_inertia[idx] = float(inertia) / float(count)

    if not avg_inertia:
        return {}

    values = np.asarray(list(avg_inertia.values()), dtype=float)
    if len(values) == 1 or np.allclose(values, values[0]):
        return {idx: "high" for idx in avg_inertia}

    low_threshold = float(np.quantile(values, 1 / 3))
    high_threshold = float(np.quantile(values, 2 / 3))
    labels = {}
    for idx, value in avg_inertia.items():
        if value <= low_threshold:
            labels[idx] = "high"
        elif value <= high_threshold:
            labels[idx] = "medium"
        else:
            labels[idx] = "low"
    return labels


def _round_inertia_by_k(inertia_by_k):
    return {int(k): _round_float(value) for k, value in inertia_by_k.items()}


def _round_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(round(float(value), 6))
