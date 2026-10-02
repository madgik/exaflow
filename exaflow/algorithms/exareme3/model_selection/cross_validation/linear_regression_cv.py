from typing import List
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated import FederatedOLS
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.cross_validator import (
    FederatedCrossValidator,
)
from exaflow.algorithms.federated.model_selection.cross_validation.scorer_regression import (
    FederatedRegressionScorer,
)
from exaflow.algorithms.federated.model_selection.cross_validation.splitter_kfold import (
    FederatedKFoldSplitter,
)
from exaflow.algorithms.federated.pipeline import FederatedPipeline
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder

ALPHA = 0.05


class BasicStats(NamedTuple):
    mean: float | None
    std: float | None

    @classmethod
    def from_values(cls, values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            return cls(mean=None, std=None)
        return cls(mean=float(values.mean()), std=float(values.std(ddof=1)))


class LinearRegressionCVResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    n_obs: List[int]
    root_mean_sq_error: BasicStats
    r_squared: BasicStats
    mean_abs_error: BasicStats
    f_stat: BasicStats


class LinearRegressionCV(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="linear_regression_cv",
            desc="Linear regression evaluated with K-fold cross-validation.",
            documentation=(
                "Evaluates a linear regression model with K-fold "
                "cross-validation. Each fold trains a model and reports "
                "metrics over held-out data. Nominal predictors are one-hot "
                "encoded with a consistent global schema.\n\n"
                "The 'n_splits' setting controls the number of cross-validation "
                "folds. It must be between 2 and 20. Default is 5.\n\n"
                "Results contain root_mean_sq_error (RMSE), r_squared, "
                "mean_abs_error, and f_stat as [mean, sample standard deviation] "
                "across folds. A metric summary is [null, null] if any fold "
                "is undefined or infinite. R-squared can be negative and is "
                "undefined for constant test outcomes or fewer than two test "
                "observations. n_obs lists training observations per fold. "
                "The F diagnostic uses (TSS - RSS) * (n_train - p - 1) / "
                "(p * RSS), with held-out sums of squares and p encoded "
                "predictors excluding the intercept; it is not a conventional "
                "fitted-model significance test. It is undefined with no "
                "predictors, no outcome variation, or nonpositive residual "
                "degrees of freedom, and diverges for a perfect fit.\n\n"
                "Reference behavior is aligned with scikit-learn KFold "
                "cross-validation around an OLS-style linear regression model. "
                "Fold metrics are computed from aggregated prediction and "
                "residual statistics without sharing raw data."
            ),
            label="Linear Regression Cross-validation",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Numerical outcome variable.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Covariates",
                desc="Numerical or categorical covariates.",
                types=[
                    specs.InputDataType.REAL,
                    specs.InputDataType.INT,
                    specs.InputDataType.TEXT,
                ],
                stattypes=[
                    specs.InputDataStatType.NUMERICAL,
                    specs.InputDataStatType.NOMINAL,
                ],
                required=True,
            ),
            parameters={
                "n_splits": specs.ParameterSpecification(
                    label="Number of folds",
                    desc="Fold count used for cross-validation.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=5,
                    min=2,
                    max=20,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.y[0]
        n_splits = int(self.get_parameter("n_splits"))
        x_vars = list(self.x)
        categorical_vars = [
            var for var in x_vars if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in x_vars if not self.metadata[var]["is_categorical"]
        ]

        metrics = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "n_splits": n_splits,
            },
            identical_results=True,
        )
        indep_var_names = metrics["feature_names"]

        nobs = [int(v) for v in metrics["n_obs"]]

        result = LinearRegressionCVResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            n_obs=nobs,
            root_mean_sq_error=BasicStats.from_values(metrics["rmse"]),
            r_squared=BasicStats.from_values(metrics["r2"]),
            mean_abs_error=BasicStats.from_values(metrics["mae"]),
            f_stat=BasicStats.from_values(metrics["f_stat"]),
        )
        return result


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client,
    data,
    y_var,
    x_vars,
    categorical_vars,
    numerical_vars,
    n_splits,
):
    """
    Run K-fold CV locally on each worker, but use agg_client to:

    - Train a global linear model per fold (aggregated X'X, X'y, n_train).
    - Aggregate residual statistics on the test set.

    Returns identical global metrics from every worker.
    """
    cv_pipeline = FederatedPipeline(
        [
            (
                "features",
                FederatedColumnTransformer(
                    [("cat", FederatedOneHotEncoder(), categorical_vars)],
                    remainder="passthrough",
                ),
            ),
            ("model", FederatedOLS(fit_intercept=True)),
        ]
    )
    y = data[y_var].astype(float).to_numpy()

    splitter = FederatedKFoldSplitter(n_splits=n_splits, shuffle=False)
    cross_validator = FederatedCrossValidator(
        estimator=cv_pipeline,
        splitter=splitter,
        scorer=FederatedRegressionScorer(),
    )
    metrics = cross_validator.evaluate(
        None,
        y,
        data=data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
        agg_client=agg_client,
    )

    # Get global feature names
    feature_transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), categorical_vars)],
        remainder="passthrough",
    )
    feature_transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = feature_transformer.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = ["Intercept"] + feature_names

    metrics["feature_names"] = feature_names
    return metrics
