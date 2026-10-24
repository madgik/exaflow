from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
from pydantic import BaseModel
from pydantic import Field

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.decomposition.pca import FederatedPCA


class PCAComponent(BaseModel):
    component_id: str
    explained_variance: float
    explained_variance_ratio: float
    cumulative_explained_variance: float
    loadings: Dict[str, float]


class PCAComponentChoice(BaseModel):
    component_id: str
    explained_variance_ratio: float


class PCAReusableBaseParameters(BaseModel):
    variables: List[str]
    pca_variant: Literal["pca"] = "pca"


class PCAReusablePreprocessing(BaseModel):
    schema_version: str = "1"
    preprocessing_name: str = "pca_column_creator"
    base_parameters: PCAReusableBaseParameters
    component_choices: List[PCAComponentChoice]


class PCAResult(BaseModel):
    title: str
    n_obs: int
    eigenvalues: List[float]
    eigenvectors: List[List[float]]
    variables: List[str] = Field(default_factory=list)
    components: List[PCAComponent] = Field(default_factory=list)
    reusable_preprocessing: Optional[PCAReusablePreprocessing] = None


class PCA(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="pca",
            desc="Principal component analysis for numerical variables.",
            documentation=(
                "Computes principal components for selected numerical "
                "variables from their covariance structure. PCA summarizes "
                "multivariate variation by producing eigenvalues and "
                "eigenvectors for orthogonal component directions.\n\n"
                "The result includes the observation count, eigenvalues, and "
                "eigenvectors.\n\n"
                "Reference behavior is aligned with covariance-based PCA "
                "methodology as exposed by scikit-learn PCA. The method "
                "computes the covariance quantities from aggregated sufficient "
                "statistics without sharing raw data."
            ),
            label="Principal Component Analysis",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Variables",
                desc="Numerical variables used to compute principal components.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_vars": self.y,
            },
            identical_results=True,
        )
        return build_pca_result(
            payload=result,
            variables=self.y,
            pca_variant="pca",
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_vars):
    X_values = np.array(data.loc[:, y_vars], dtype=float, copy=True)
    model = FederatedPCA(agg_client=agg_client)
    model.fit(X_values)
    return model_result_payload(model)


def model_result_payload(model: FederatedPCA) -> Dict[str, Any]:
    return {
        "n_obs": model.n_samples_seen_,
        "eigenvalues": model.explained_variance_.tolist(),
        "eigenvectors": model.components_.tolist(),
    }


def build_pca_result(
    *,
    payload: Dict[str, Any],
    variables: List[str],
    pca_variant: Literal["pca"] = "pca",
) -> PCAResult:
    eigenvalues = [float(value) for value in payload["eigenvalues"]]
    eigenvectors = [
        [float(value) for value in component] for component in payload["eigenvectors"]
    ]
    total_variance = float(np.sum(eigenvalues))
    if total_variance == 0:
        ratios = [0.0 for _ in eigenvalues]
    else:
        ratios = [value / total_variance for value in eigenvalues]
    cumulative = np.cumsum(ratios).tolist()

    components = [
        PCAComponent(
            component_id=f"PC{idx + 1}",
            explained_variance=eigenvalue,
            explained_variance_ratio=ratio,
            cumulative_explained_variance=float(cumulative[idx]),
            loadings={
                variable: eigenvectors[idx][feature_idx]
                for feature_idx, variable in enumerate(variables)
            },
        )
        for idx, (eigenvalue, ratio) in enumerate(zip(eigenvalues, ratios))
    ]

    reusable = PCAReusablePreprocessing(
        base_parameters=PCAReusableBaseParameters(
            variables=list(variables),
            pca_variant=pca_variant,
        ),
        component_choices=[
            PCAComponentChoice(
                component_id=component.component_id,
                explained_variance_ratio=component.explained_variance_ratio,
            )
            for component in components
        ],
    )
    return PCAResult(
        title="Eigenvalues and Eigenvectors",
        n_obs=int(payload["n_obs"]),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        variables=list(variables),
        components=components,
        reusable_preprocessing=reusable,
    )
