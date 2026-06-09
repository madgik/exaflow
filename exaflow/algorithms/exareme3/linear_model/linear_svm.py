from typing import List

import numpy as np
from pydantic import BaseModel
from sklearn.svm import SVC

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from exaflow.worker_communication import BadUserInput


class SVMResult(BaseModel):
    title: str
    n_obs: int
    weights: List[float]
    intercept: float


class LinearSVM(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="linear_svm",
            desc="Linear support vector machine for classification with numerical features.",
            documentation=(
                "Trains a linear support vector machine for a categorical "
                "target using selected numerical features. A linear SVC model "
                "is fitted separately on each dataset, then learned "
                "coefficients and intercepts are averaged.\n\n"
                "The gamma parameter is passed to scikit-learn SVC with a "
                "linear kernel. The C parameter controls the misclassification "
                "penalty and regularization strength.\n\n"
                "The result includes the total observation count, averaged "
                "weights, and averaged intercept.\n\n"
                "Reference behavior is aligned with scikit-learn SVC using "
                "kernel='linear' for each local fit. This method differs from "
                "a single pooled SVC because it averages fitted parameters "
                "rather than optimizing one global margin."
            ),
            label="Linear SVM",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Target",
                    desc="Nominal target variable defining the classes.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                ),
                x=specs.InputDataSpecification(
                    label="Features",
                    desc="Numerical covariates used to train the linear SVM.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                ),
            ),
            parameters={
                "gamma": specs.ParameterSpecification(
                    label="Gamma",
                    desc="Kernel coefficient passed to the linear SVM.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.1,
                    min=0.0,
                    max=1.0,
                ),
                "C": specs.ParameterSpecification(
                    label="Regularization",
                    desc="Regularization penalty for misclassification.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=1.0,
                    min=0.0,
                    max=1.0,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x

        gamma = self.get_parameter("gamma")
        C = self.get_parameter("C")

        udf_results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "gamma": float(gamma),
                "C": float(C),
            },
        )

        coeff_sum = None
        intercept_sum = 0.0
        total_n_obs = 0
        for res in udf_results:
            coeff_vec = np.asarray(res["coeff_local"], dtype=float)
            coeff_sum = coeff_vec if coeff_sum is None else coeff_sum + coeff_vec
            intercept_sum += float(res["intercept_local"])
            total_n_obs += int(res["n_obs"])

        num_workers = float(len(udf_results) or 1.0)
        coeff_mean = (coeff_sum / num_workers).reshape(-1)
        intercept_mean = float(intercept_sum / num_workers)
        return SVMResult(
            title="Federated Linear SVM (Averaged Parameters)",
            n_obs=int(total_n_obs),
            weights=coeff_mean.tolist(),
            intercept=intercept_mean,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var, x_vars, gamma, C):
    """
    Train a linear SVM locally and return local model summaries for global aggregation.
    """
    n_features = len(x_vars)
    if n_features == 0:
        raise BadUserInput("SVM requires at least one covariate (x).")

    X = data[x_vars].to_numpy(dtype=float, copy=False)
    y = data[y_var].to_numpy(copy=False)
    aggregator = NumpyAggregator(agg_client)

    n_obs_local = float(len(y))
    unique_y = aggregator.fed_union(y)
    if unique_y.size < 2:
        raise BadUserInput("Cannot perform SVM. Covariable has only one level.")

    model = SVC(kernel="linear", gamma=gamma, C=C)
    model.fit(X, y)

    coeff_arr = np.asarray(model.coef_, dtype=float)
    # Fix shape across workers: average across class rows to a single vector
    coeff_local = coeff_arr.mean(axis=0)

    intercept_arr = np.asarray(model.intercept_, dtype=float)
    intercept_local = float(intercept_arr.mean())

    return {
        "n_obs": int(n_obs_local),
        "coeff_local": np.asarray(coeff_local, dtype=float).reshape(-1).tolist(),
        "intercept_local": intercept_local,
    }
