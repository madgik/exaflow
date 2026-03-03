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
    mean: float
    std: float


class LinearRegressionCVResult(BaseModel):
    dependent_var: str
    indep_vars: List[str]
    n_obs: List[int]
    mean_sq_error: BasicStats
    r_squared: BasicStats
    mean_abs_error: BasicStats
    f_stat: BasicStats


class LinearRegressionCV(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="linear_regression_cv",
            desc="Evaluates a federated linear regression model using K-fold cross-validation. Each fold trains a global model and reports metrics over the held-out data.",
            label="Linear Regression (K-Fold CV)",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Dependent Variable",
                    desc="Single numerical target variable.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Independent Variables",
                    desc="One or more predictors (numerical or nominal). Nominal variables are one-hot encoded with a global schema across workers.",
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
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "n_splits": specs.ParameterSpecification(
                    label="Number of Folds",
                    desc="Number of folds for K-fold cross-validation.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=5,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=2,
                    max=20,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        n_splits = int(self.get_parameter("n_splits"))

        # Identify categorical vs numerical predictors
        categorical_vars = [
            var for var in self.inputdata.x if self.metadata[var]["is_categorical"]
        ]
        numerical_vars = [
            var for var in self.inputdata.x if not self.metadata[var]["is_categorical"]
        ]

        udf_results = self.run_local_udf(
            func=linear_regression_cv_local_step,
            kw_args={
                "y_var": y_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
                "n_splits": n_splits,
            },
        )

        # All workers should return identical global metrics; take the first
        metrics = udf_results[0]
        indep_var_names = metrics["feature_names"]

        rmse = np.asarray(metrics["rmse"], dtype=float)
        r2 = np.asarray(metrics["r2"], dtype=float)
        mae = np.asarray(metrics["mae"], dtype=float)
        fstats = np.asarray(metrics["f_stat"], dtype=float)
        nobs = [int(v) for v in metrics["n_obs"]]

        result = LinearRegressionCVResult(
            dependent_var=y_var,
            indep_vars=indep_var_names,
            n_obs=nobs,
            mean_sq_error=BasicStats(
                mean=float(rmse.mean()), std=float(rmse.std(ddof=1))
            ),
            r_squared=BasicStats(mean=float(r2.mean()), std=float(r2.std(ddof=1))),
            mean_abs_error=BasicStats(
                mean=float(mae.mean()), std=float(mae.std(ddof=1))
            ),
            f_stat=BasicStats(mean=float(fstats.mean()), std=float(fstats.std(ddof=1))),
        )
        return result


@exareme3_udf(with_aggregation_server=True)
def linear_regression_cv_local_step(
    agg_client,
    data,
    y_var,
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
