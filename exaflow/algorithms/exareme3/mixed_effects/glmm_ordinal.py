from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import (
    encode_ordinal_response,
)
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import get_group_ids
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import (
    split_grouping_var,
)
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.mixed_effects import FederatedGLMMOrdinal
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder


class GLMMOrdinalResult(BaseModel):
    dependent_var: str
    grouping_var: str
    indep_vars: List[str]
    category_order: List[str]
    n_obs: int
    n_groups: int
    coefficients: List[float]
    cutpoints: List[float]
    sigma_u2: float
    converged: bool
    n_iter: int


class GLMMOrdinal(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="glmm_ordinal",
            desc=(
                "Ordinal generalized linear mixed model with a random "
                "intercept for ordered outcomes."
            ),
            documentation=(
                "Fits an ordinal generalized linear mixed model with fixed "
                "covariate effects and a single random-intercept grouping "
                "variable. The dependent variable categories are interpreted "
                "using the explicitly provided order. This model is not a "
                "general-purpose count model.\n\n"
                "The 'grouping_var' setting selects the variable from x used as "
                "the random-intercept grouping factor.\n\n"
                "The 'category_order' setting provides y categories from lowest "
                "to highest outcome level.\n\n"
                "The result includes threshold and fixed-effect coefficients, "
                "standard errors, z-scores, p-values, confidence intervals, "
                "random-effect variance, log-likelihood, AIC, BIC, convergence "
                "status, and iteration count.\n\n"
                "Reference behavior is aligned with standard ordinal "
                "random-intercept GLMM methodology, similar to cumulative-link "
                "mixed models. Model quantities are computed from aggregated "
                "sufficient statistics without sharing raw data."
            ),
            label="Ordinal GLMM",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Ordered outcome",
                    desc=(
                        "A unique ordered categorical or integer-coded ordered "
                        "outcome. The category order must be provided "
                        "explicitly by the 'category_order' parameter. This "
                        "is not intended as a general-purpose count-model "
                        "interface."
                    ),
                    types=[specs.InputDataType.INT, specs.InputDataType.TEXT],
                    stattypes=[
                        specs.InputDataStatType.NOMINAL,
                        specs.InputDataStatType.NUMERICAL,
                    ],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates and grouping variable",
                    desc="Covariates plus the random-intercept grouping variable.",
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
                    min_count=2,
                ),
            ),
            parameters={
                "grouping_var": specs.ParameterSpecification(
                    label="Grouping variable",
                    desc="Random-intercept grouping factor.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
                ),
                "category_order": specs.ParameterSpecification(
                    label="Ordinal category order",
                    desc="Ordered outcome levels from lowest to highest.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=True,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        grouping_var = self.get_parameter("grouping_var")
        category_order = self.get_parameter("category_order")
        categorical_vars, numerical_vars = split_grouping_var(
            self.inputdata.x, grouping_var, self.metadata
        )

        udf_results = self.run_local_udf(
            func=glmm_ordinal_local_step,
            kw_args={
                "y_var": y_var,
                "grouping_var": grouping_var,
                "category_order": category_order,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
        )
        model_stats = udf_results[0]
        return GLMMOrdinalResult(
            dependent_var=y_var,
            grouping_var=grouping_var,
            indep_vars=model_stats["feature_names"],
            category_order=model_stats["category_order"],
            n_obs=model_stats["n_obs"],
            n_groups=model_stats["n_groups"],
            coefficients=model_stats["coefficients"],
            cutpoints=model_stats["cutpoints"],
            sigma_u2=model_stats["sigma_u2"],
            converged=model_stats["converged"],
            n_iter=model_stats["n_iter"],
        )


@exareme3_udf(with_aggregation_server=True)
def glmm_ordinal_local_step(
    agg_client,
    data,
    y_var,
    grouping_var,
    category_order,
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
    y, category_order = encode_ordinal_response(data[y_var], category_order)
    center_ids = get_group_ids(data, grouping_var)
    X = transformer.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    model = FederatedGLMMOrdinal(K=len(category_order), fit_intercept=True)
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
        "category_order": category_order,
        "n_obs": results.nobs,
        "n_groups": results.n_groups,
        "coefficients": results.params.tolist(),
        "cutpoints": results.cutpoints.tolist(),
        "sigma_u2": results.sigma_u2,
        "converged": results.converged,
        "n_iter": results.n_iter,
    }
