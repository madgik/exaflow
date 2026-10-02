from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic import Field

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
    pca_variant: Literal["pca", "pca_with_transformation"]
    data_transformation: Optional[Dict[str, List[str]]] = None


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


@dataclass
class PCAExecutionResult:
    model: FederatedPCA
    scores: Optional[np.ndarray] = None


def run_federated_pca(
    *,
    agg_client,
    data: pd.DataFrame,
    variables: List[str],
    data_transformation: Optional[Dict[str, List[str]]] = None,
    compute_scores: bool = False,
) -> PCAExecutionResult:
    """Apply the existing PCA pipeline and return its fitted model.

    Transformation handling intentionally mirrors the existing
    ``pca_with_transformation`` UDF, including its operation order and the
    second standardization performed by ``FederatedPCA.fit``.
    """
    if data_transformation is None:
        data_transformation = {}

    allowed_keys = {"log", "exp", "center", "standardize"}
    for key in data_transformation:
        if key not in allowed_keys:
            raise ValueError(f"Unknown transformation: {key}")

    X = data.loc[:, variables].copy()

    for col in data_transformation.get("log", []) or []:
        if col not in X.columns:
            continue
        if (X[col] <= 0).any():
            raise ValueError(
                f"Log transformation cannot be applied to non-positive values in column '{col}'."
            )
        X[col] = np.log(X[col])

    for col in data_transformation.get("exp", []) or []:
        if col not in X.columns:
            continue
        X[col] = np.exp(X[col])

    center_cols = set(data_transformation.get("center", []) or [])
    standardize_cols = set(data_transformation.get("standardize", []) or [])

    if center_cols or standardize_cols:
        X_values = X.to_numpy(dtype=float, copy=False)
        if not X_values.flags.writeable:
            X_values = np.array(X_values, copy=True)

        n_obs_local = float(X_values.shape[0])
        if n_obs_local > 0:
            sx_local = np.einsum("ij->j", X_values)
            sxx_local = np.einsum("ij,ij->j", X_values, X_values)
        else:
            sx_local = np.zeros(X_values.shape[1], dtype=float)
            sxx_local = np.zeros(X_values.shape[1], dtype=float)

        total_n_obs_arr = agg_client.sum(np.array([n_obs_local], dtype=float))
        total_sx_arr = agg_client.sum(sx_local)
        total_sxx_arr = agg_client.sum(sxx_local)
        total_n_obs = float(np.asarray(total_n_obs_arr, dtype=float).reshape(-1)[0])
        if total_n_obs <= 1:
            means = np.zeros_like(sx_local)
            sigmas = np.ones_like(sx_local)
        else:
            total_sx = np.asarray(total_sx_arr, dtype=float)
            total_sxx = np.asarray(total_sxx_arr, dtype=float)
            means = total_sx / total_n_obs
            variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
            variances = np.maximum(variances, 0.0)
            sigmas = np.sqrt(variances)

        col_to_idx = {name: idx for idx, name in enumerate(X.columns)}
        for col in standardize_cols:
            if col not in col_to_idx:
                continue
            if sigmas[col_to_idx[col]] == 0:
                raise ValueError(
                    f"Standardization cannot be applied to column '{col}' because its standard deviation is zero."
                )

        if X_values.size > 0:
            for col_name, idx in col_to_idx.items():
                if col_name in standardize_cols:
                    X_values[:, idx] = (X_values[:, idx] - means[idx]) / sigmas[idx]
                elif col_name in center_cols:
                    X_values[:, idx] = X_values[:, idx] - means[idx]
            X.iloc[:, :] = X_values
    else:
        X_values = X.to_numpy(dtype=float, copy=False)
        if not X_values.flags.writeable:
            X_values = np.array(X_values, copy=True)

    # Keep the fit/projection matrix owned by this helper. FederatedPCA mutates
    # writeable inputs during fit and transform when copy=False.
    X_values = np.array(X_values, dtype=float, copy=True)
    model = FederatedPCA(agg_client=agg_client)
    model.fit(X_values)
    scores = model.transform(X_values) if compute_scores else None
    return PCAExecutionResult(model=model, scores=scores)


def model_result_payload(model: FederatedPCA) -> Dict[str, Any]:
    return {
        "n_obs": model.n_samples_seen_,
        "eigenvalues": model.explained_variance_.tolist(),
        "eigenvectors": model.components_.tolist(),
    }


def _normalized_transformation(
    data_transformation: Optional[Dict[str, List[str]]],
) -> Optional[Dict[str, List[str]]]:
    if data_transformation is None:
        return None
    return {
        str(key): [str(value) for value in values]
        for key, values in data_transformation.items()
    }


def build_pca_result(
    *,
    payload: Dict[str, Any],
    variables: List[str],
    pca_variant: Literal["pca", "pca_with_transformation"],
    data_transformation: Optional[Dict[str, List[str]]] = None,
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
            data_transformation=_normalized_transformation(data_transformation),
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
