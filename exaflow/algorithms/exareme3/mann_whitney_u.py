from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from library.statistical_tests.nonparametric.mann_whitney_utest import (
    MannWhitneyUTest as LibMannWhitneyU,
)
from library.utils.aggregators.numpy_aggregator import NumpyAggregator


class MannWhitneyUResult(BaseModel):
    title: str
    statistic: float
    p_value: float


class MannWhitneyU(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="mann_whitney_u",
            desc="Federated histogram-based Mann-Whitney U test.",
            label="Mann-Whitney U Test",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variable",
                    desc="Numeric variable to compare.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Group",
                    desc="Categorical variable with two levels.",
                    types=[specs.InputDataType.TEXT, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "alternative": specs.ParameterSpecification(
                    label="Alternative Hypothesis",
                    desc="Specifies the alternative hypothesis (two-sided, less, or greater).",
                    types=[specs.ParameterType.TEXT],
                    required=False,
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
                "num_bins": specs.ParameterSpecification(
                    label="Number of bins",
                    desc="Number of histogram bins used for rank approximation.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=10,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=2,
                    max=1000,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        alternative = self.get_parameter("alternative", "two-sided")
        num_bins = self.get_parameter("num_bins", 10)

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": self.inputdata.y[0],
                "x_var": self.inputdata.x[0],
                "alternative": alternative,
                "num_bins": num_bins,
            },
        )
        u_stat, p_value = results[0]

        return MannWhitneyUResult(
            title="Mann-Whitney U Test",
            statistic=u_stat,
            p_value=p_value,
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var, x_var, alternative, num_bins):
    aggregator = NumpyAggregator(agg_client)

    # Need to split data based on x_var
    # We assume x_var has exactly two categories
    groups = data[x_var].unique()
    if len(groups) != 2:
        # If one client has only 1 group, it's okay, but globally it needs 2.
        # However, local UDF might only see what's local.
        # But wait, MannWhitneyUTest.rank uses global_min/max.
        pass

    # The library implementation expects x and y as separate arrays.
    # In exareme3, we usually have one dataframe.
    group0_mask = data[x_var] == groups[0]
    group1_mask = data[x_var] == groups[1]
    x = data.loc[group0_mask, y_var].to_numpy()
    y = data.loc[group1_mask, y_var].to_numpy()

    mw = LibMannWhitneyU(aggregator)
    return mw.compute(
        x=x,
        y=y,
        alternative=alternative,
        num_bins=num_bins,
    )
