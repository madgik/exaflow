from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._logistic import ALPHA
from exaflow.algorithms.feta._logistic import MAX_ITER
from exaflow.algorithms.feta._logistic import TOL
from exaflow.algorithms.feta._logistic import _compute_summary
from exaflow.algorithms.feta._logistic import _handle_logreg_errors
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.worker_communication import BadUserInput


class LogisticRegressionSummary(BaseModel):
    n_obs: int
    coefficients: List[float]
    stderr: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    z_scores: List[float]
    pvalues: List[float]
    df_model: int
    df_resid: int
    r_squared_cs: float
    r_squared_mcf: float
    ll0: float
    ll: float
    aic: float
    bic: float


class LogisticRegressionResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    summary: LogisticRegressionSummary


class FetaLogisticRegression(Algorithm):
    column_inputs = frozenset({"x"})

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_logistic_regression",
            desc="Logistic regression for a binary outcome and one or more covariates.",
            documentation=(
                "Fits a logistic regression model for a binary outcome using "
                "numerical and/or categorical covariates. The dependent "
                "variable is converted to binary by assigning 1 to the "
                "selected positive class and 0 to all other classes. "
                "Categorical covariates are one-hot encoded before estimation.\n\n"
                "The model includes an intercept and estimates coefficients by "
                "maximizing the binomial log-likelihood with the logistic link "
                "function. The positive class is selected using the "
                "positive_class parameter.\n\n"
                "The result includes coefficients, standard errors, z-scores, "
                "p-values, confidence intervals, degrees of freedom, pseudo "
                "R-squared values, log-likelihoods, AIC, and BIC.\n\n"
                "Reference behavior is aligned with statsmodels logistic "
                "regression using smf.logit(formula, data).fit(method='newton') "
                "for a model with an intercept. The method computes the same "
                "model quantities from aggregated gradient, Hessian, and "
                "log-likelihood statistics without sharing raw data.\n\n"
                "Inputs containing missing values are rejected. Preprocessing is not supported."
            ),
            label="Logistic Regression",
            enabled=True,
            required_preprocessing=[],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Nominal outcome converted using the positive class.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Covariates",
                desc="Numerical or categorical covariates.",
                types=[
                    specs.InputDataType.INT,
                    specs.InputDataType.REAL,
                    specs.InputDataType.TEXT,
                ],
                stattypes=[
                    specs.InputDataStatType.NUMERICAL,
                    specs.InputDataStatType.NOMINAL,
                ],
                required=True,
            ),
            parameters={
                "positive_class": specs.ParameterSpecification(
                    label="Positive class",
                    desc="Outcome category treated as the positive outcome.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y"],
                    ),
                ),
            },
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self, x: FedColumns, y: FedTensor) -> LogisticRegressionResult:
        target = (y == str(self.parameters["positive_class"])) * 1.0
        n, y_sum = len(target), float(fnp.sum(target))
        columns, feature_names = [], []
        for name in x.names:
            column, metadata = x[name], x.metadata[name]
            if not metadata.is_categorical:
                continue
            if column.dtype.kind == "U":
                levels = fnp.unique(column).tolist()
            else:
                # Numeric categories are discovered from declared codes and
                # counts; continuous observations are never released as unique.
                counts = [
                    (level, int(fnp.sum(column == level)))
                    for level in metadata.categories
                ]
                if sum(count for _, count in counts) != x.nrows:
                    raise BadUserInput(
                        f"Categorical variable {name} contains undeclared codes."
                    )
                levels = [level for level, count in counts if count]
            for level in levels[1:]:
                columns.append((column == level) * 1.0)
                feature_names.append(f"{name}[{level}]")
        for name in x.names:
            if not x.metadata[name].is_categorical:
                columns.append(x[name])
                feature_names.append(name)
        design = fnp.column_stack([fnp.ones_like(target), *columns])
        p = design.shape[1]
        _handle_logreg_errors(n, p, y_sum)
        coefficients = np.zeros((p, 1))
        for _ in range(MAX_ITER):
            eta = design @ coefficients
            mu = 1.0 / (1.0 + fnp.exp(0.0 - eta))
            weights = mu * (1.0 - mu)
            gradient = design.T @ (target - mu)
            hessian = design.T @ (weights * design)
            ll = float(
                fnp.sum(fnp.where(target == 1.0, fnp.log(mu), fnp.log(1.0 - mu)))
            )
            try:
                inverse = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                inverse = np.linalg.pinv(hessian)
            coefficients = coefficients + inverse @ gradient
            if np.max(np.abs(gradient)) <= TOL:
                break
        else:
            raise BadUserInput("Logistic regression cannot converge. Cancelling run.")
        summary = _compute_summary(
            coefficients=coefficients.reshape(-1),
            h_inv=inverse,
            ll=ll,
            n_obs=n,
            y_sum=y_sum,
            alpha=ALPHA,
        )
        return LogisticRegressionResult(
            dependent_var=self.y_names[0],
            indep_vars=["Intercept", *feature_names],
            summary=LogisticRegressionSummary(**summary),
        )
