from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import get_group_ids
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import (
    split_grouping_var,
)
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.mixed_effects import FederatedLMM
from exaflow.algorithms.federated.preprocessing import FederatedOneHotEncoder


class LMMResult(BaseModel):
    dependent_var: str
    grouping_var: str
    indep_vars: List[str]
    n_obs: int
    n_groups: int
    df_model: int
    df_resid: int
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalues: List[float]
    lower_ci: List[float]
    upper_ci: List[float]
    sigma2: float
    sigma_u2: float
    ll_reml: float
    aic: float
    bic: float
    converged: bool
    n_iter: int


class LMM(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="lmm",
            desc="Federated linear mixed model with a single random-intercept grouping variable.",
            label="Linear Mixed Model",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome (continuous)",
                    desc="Single numerical dependent variable.",
                    types=[specs.InputDataType.REAL],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
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
        grouping_var = self.get_parameter("grouping_var")
        categorical_vars, numerical_vars = split_grouping_var(
            self.inputdata.x, grouping_var, self.metadata
        )

        udf_results = self.run_local_udf(
            func=lmm_local_step,
            kw_args={
                "y_var": y_var,
                "grouping_var": grouping_var,
                "categorical_vars": categorical_vars,
                "numerical_vars": numerical_vars,
            },
        )
        model_stats = udf_results[0]
        return LMMResult(
            dependent_var=y_var,
            grouping_var=grouping_var,
            indep_vars=model_stats["feature_names"],
            n_obs=model_stats["n_obs"],
            n_groups=model_stats["n_groups"],
            df_model=model_stats["df_model"],
            df_resid=model_stats["df_resid"],
            coefficients=model_stats["coefficients"],
            std_err=model_stats["std_err"],
            t_stats=model_stats["t_stats"],
            pvalues=model_stats["pvalues"],
            lower_ci=model_stats["lower_ci"],
            upper_ci=model_stats["upper_ci"],
            sigma2=model_stats["sigma2"],
            sigma_u2=model_stats["sigma_u2"],
            ll_reml=model_stats["ll_reml"],
            aic=model_stats["aic"],
            bic=model_stats["bic"],
            converged=model_stats["converged"],
            n_iter=model_stats["n_iter"],
        )


@exareme3_udf(with_aggregation_server=True)
def lmm_local_step(
    agg_client,
    data,
    y_var,
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
    y = data[y_var].to_numpy(dtype=float, copy=False)
    center_ids = get_group_ids(data, grouping_var)
    X = transformer.transform(
        data,
        categorical_vars=categorical_vars,
        numerical_vars=numerical_vars,
    )
    model = FederatedLMM(fit_intercept=True)
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
        "df_model": results.df_model,
        "df_resid": results.df_resid,
        "coefficients": results.params.tolist(),
        "std_err": results.bse.tolist(),
        "t_stats": results.tvalues.tolist(),
        "pvalues": results.pvalues.tolist(),
        "lower_ci": results.conf_int_low.tolist(),
        "upper_ci": results.conf_int_high.tolist(),
        "sigma2": results.sigma2,
        "sigma_u2": results.sigma_u2,
        "ll_reml": results.ll_reml,
        "aic": results.aic,
        "bic": results.bic,
        "converged": results.converged,
        "n_iter": results.n_iter,
    }
