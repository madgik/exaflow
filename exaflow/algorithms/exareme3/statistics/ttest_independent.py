from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)


class TTestIndependentResult(BaseModel):
    t_stat: float
    df: float
    p: float
    mean_diff: float
    se_diff: float
    ci_upper: str | float
    ci_lower: str | float
    cohens_d: float


class TTestIndependent(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="ttest_independent",
            desc="Federated Student's independent t-test for the difference in means between two independent groups (pooled variance; df = n_a + n_b - 2). Reports confidence intervals and Cohen's d (pooled SD).",
            label="Student's Independent T-Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable of interest",
                    desc="A numerical variable.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Grouping variable",
                    desc="A nominal variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative Hypothesis",
                    desc="Specifies whether group A is different from, greater than, or less than group B.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default="two-sided",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["two-sided", "less", "greater"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "alpha": specs.ParameterSpecification(
                    label="Alpha",
                    desc="The significance level.",
                    types=[specs.ParameterType.REAL],
                    required=True,
                    multiple=False,
                    default=0.05,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=0.0,
                    max=1.0,
                ),
                "groupA": specs.ParameterSpecification(
                    label="Group A",
                    desc="Category of the grouping variable used as group A.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=None,
                    max=None,
                ),
                "groupB": specs.ParameterSpecification(
                    label="Group B",
                    desc="Category of the grouping variable used as group B.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default=None,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
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
        alpha = self.get_parameter("alpha")
        alternative = self.get_parameter("alt_hypothesis")
        group_a = self.get_parameter("groupA")
        group_b = self.get_parameter("groupB")

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "group_var": self.inputdata.x[0],
                "value_var": self.inputdata.y[0],
                "alpha": alpha,
                "alternative": alternative,
                "group_a": group_a,
                "group_b": group_b,
            },
            identical_results=True,
        )
        return TTestIndependentResult(
            t_stat=result["t_stat"],
            df=result["df"],
            p=result["p_value"],
            mean_diff=result["mean_diff"],
            se_diff=result["se_diff"],
            ci_upper=result["ci_upper"],
            ci_lower=result["ci_lower"],
            cohens_d=result["cohens_d"],
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client, data, group_var, value_var, alpha, alternative, group_a, group_b
):
    ttest = FederatedTTestIndependent(agg_client=agg_client)
    return ttest.compute(
        data=data,
        group_var=group_var,
        value_var=value_var,
        group_a=group_a,
        group_b=group_b,
        alpha=alpha,
        alternative=alternative,
    )
