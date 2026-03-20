from typing import List

import numpy as np
from pydantic import BaseModel
from sklearn.svm import SVC

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
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
            desc="Federated linear SVM trained locally and averaged across workers (coefficients and intercept).",
            label="Linear SVM",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Target classes",
                    desc="Nominal target variable defining the classes.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Features",
                    desc="Numerical covariates used to train the linear SVM.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "gamma": specs.ParameterSpecification(
                    label="Gamma",
                    desc="Gamma parameter passed to scikit-learn's SVC (linear kernel).",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.1,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=0.0,
                    max=1.0,
                ),
                "C": specs.ParameterSpecification(
                    label="C",
                    desc="Regularization parameter (penalty for misclassification).",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=1.0,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=0.0,
                    max=1.0,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[],
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


@exareme3_udf()
def local_step(data, y_var, x_vars, metadata, gamma, C):
    """
    Train a linear SVM locally and return local model summaries for global aggregation.
    """
    # Keep only required columns and drop rows with missing values
    cols = list(dict.fromkeys(list(x_vars) + [y_var]))
    data = data[cols].dropna()

    y_enums = (metadata.get(y_var) or {}).get("enumerations") or {}
    y_levels = list(y_enums.keys())
    if len(y_levels) < 2:
        raise BadUserInput(
            f"The variable {y_var} has less than 2 levels and SVM cannot be "
            "performed. Please choose another variable."
        )

    n_features = len(x_vars)
    if n_features == 0:
        raise BadUserInput("SVM requires at least one covariate (x).")

    X = data[x_vars].to_numpy(dtype=float, copy=False)
    y = data[y_var].to_numpy(copy=False)

    n_obs_local = float(len(y))
    unique_y = np.unique(y)
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
