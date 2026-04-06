from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.mixed_effects_common import get_group_ids
from exaflow.algorithms.exareme3.mixed_effects_common import split_grouping_var
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.linear_model.logistic_regression import (
    FederatedLogisticRegression,
)
from exaflow.algorithms.federated.mixed_effects import FederatedGLMMBinary
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder


class GLMMBinaryResult(BaseModel):
    dependent_var: str
    grouping_var: str
    indep_vars: List[str]
    n_obs: int
    n_groups: int
    coefficients: List[float]
    sigma_u2: float
    converged: bool
    n_iter: int


class GLMMBinary(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="glmm_binary",
            desc="Federated binary generalized linear mixed model with one random-intercept grouping variable.",
            label="Binary GLMM",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Dependent variable (binary)",
                    desc="A unique nominal variable converted to 0/1 using the positive_class parameter.",
                    types=[specs.InputDataType.INT, specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates and grouping variable",
                    desc="One or more covariates plus exactly one grouping variable, referenced by the 'grouping_var' parameter.",
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
                "positive_class": specs.ParameterSpecification(
                    label="Positive class (y=1)",
                    desc="Positive class of y. All other classes are considered negative.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["y"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "grouping_var": specs.ParameterSpecification(
                    label="Grouping variable",
                    desc="Variable from x to use as the random-intercept grouping factor.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        positive_class = self.get_parameter("positive_class")
        grouping_var = self.get_parameter("grouping_var")
        categorical_vars, numerical_vars = split_grouping_var(
            self.inputdata.x, grouping_var, self.metadata
        )

        udf_results = self.run_local_udf(
            func=glmm_binary_local_step,
            kw_args={
                "y_var": y_var,
                "positive_class": positive_class,
                "grouping_var": grouping_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
        )
        model_stats = udf_results[0]
        return GLMMBinaryResult(
            dependent_var=y_var,
            grouping_var=grouping_var,
            indep_vars=model_stats["feature_names"],
            n_obs=model_stats["n_obs"],
            n_groups=model_stats["n_groups"],
            coefficients=model_stats["coefficients"],
            sigma_u2=model_stats["sigma_u2"],
            converged=model_stats["converged"],
            n_iter=model_stats["n_iter"],
        )


@exareme3_udf(with_aggregation_server=True)
def glmm_binary_local_step(
    agg_client,
    data,
    y_var,
    positive_class,
    grouping_var,
    categorical_vars,
    numerical_vars,
):
    transformer = FederatedColumnTransformer(
        [("cat", FederatedOneHotEncoder(), categorical_vars)],
        remainder="passthrough",
    )
    transformer.fit(
        agg_client=agg_client,
        data=data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    positive_class = FederatedLogisticRegression.coerce_positive_class(
        data[y_var], positive_class
    )
    y = data[y_var].eq(positive_class).to_numpy(dtype=float, copy=False)
    center_ids = get_group_ids(data, grouping_var)
    X = transformer.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    model = FederatedGLMMBinary(fit_intercept=True)
    results = model.fit(
        X,
        y,
        agg_client=agg_client,
        center_ids=center_ids,
    )
    feature_names = transformer.get_feature_names_out(
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    feature_names = ["Intercept"] + feature_names
    return {
        "feature_names": feature_names,
        "n_obs": results.nobs,
        "n_groups": results.n_groups,
        "coefficients": results.params.tolist(),
        "sigma_u2": results.sigma_u2,
        "converged": results.converged,
        "n_iter": results.n_iter,
    }
