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
    grouping_var: List[str]
    indep_vars: List[str]
    n_obs: int
    n_groups: int
    df_model: int
    df_resid: int
    coefficients: List[float]
    std_err: List[float]
    t_stats: List[float]
    pvalue_label: str
    pvalues: List[float]
    pvalues_display: List[str]
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
    @staticmethod
    def _format_pvalue_display(pvalue: float) -> str:
        if pvalue != pvalue:
            return "NaN"
        if pvalue < 0.001:
            return "<0.001"
        return f"{pvalue:.3f}"

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="lmm",
            desc="Linear mixed model for a numerical outcome with a random intercept.",
            documentation=(
                "Fit a linear mixed model for a continuous outcome with fixed "
                "covariate effects and a random-intercept grouping factor.\n\n"
                "The 'grouping_var' setting selects one variable from x used as "
                "the random-intercept grouping factor, or two variables used to "
                "build one composite nested-like grouping factor. Composite "
                "grouping still estimates a single random-intercept variance "
                "component.\n\n"
                "The result includes fixed-effect coefficients, standard errors, "
                "t-statistics, p-values, confidence intervals, random-effect "
                "variance, residual variance, log-likelihood, AIC, BIC, "
                "convergence status, and iteration count.\n\n"
                "Reference behavior is aligned with standard random-intercept "
                "linear mixed-model methodology, similar to statsmodels MixedLM "
                "for a single grouping factor. Model quantities are computed "
                "from aggregated sufficient statistics without sharing raw data."
            ),
            label="Linear Mixed Model",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome",
                    desc=(
                        "Single numerical dependent variable. Continuous or "
                        "integer-valued outcomes are supported when a linear "
                        "mixed-model assumption is appropriate."
                    ),
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Covariates and grouping variable",
                    desc=(
                        "Covariates plus one or two random-intercept grouping "
                        "variables."
                    ),
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
                    desc=(
                        "One grouping factor, or two variables combined into one "
                        "composite nested-like grouping factor."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=True,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x"],
                    ),
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
        pvalues = model_stats["pvalues"]
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
            pvalue_label="P(>|t|)",
            pvalues=pvalues,
            pvalues_display=[
                self._format_pvalue_display(float(pvalue)) for pvalue in pvalues
            ],
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
