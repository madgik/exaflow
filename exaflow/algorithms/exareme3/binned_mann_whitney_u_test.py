from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.binned_mann_whitney_u_test import (
    FederatedBinnedMannWhitneyUTest,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class BinnedMannWhitneyUTestResult(BaseModel):
    u_stat: float
    p_value: float
    z_score: float
    n1: int
    n2: int


class BinnedMannWhitneyUTest(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="binned_mann_whitney_u_test",
            desc="Binned Mann-Whitney U test comparing a numerical variable across two independent groups via histogram rank approximation.",
            documentation=(
                "Tests whether the distributions of a numerical variable differ between "
                "two independent groups without sharing raw data across workers.\n\n"
                "Ranks are approximated via federated histogram binning: values within "
                "the same bin receive the same average rank. The U statistic and p-value "
                "are derived from the normal approximation used by "
                "scipy.stats.mannwhitneyu with method='asymptotic'.\n\n"
                "The 'alt_hypothesis' setting selects whether the distributions differ "
                "('two-sided'), group A is stochastically less than group B ('less'), "
                "or greater ('greater').\n\n"
                "The 'num_bins' setting controls histogram resolution: more bins improve "
                "rank approximation accuracy at the cost of more data sent per worker.\n\n"
                "Returns u_stat, p_value, z_score, and the two sample sizes n1 and n2."
            ),
            label="Binned Mann-Whitney U Test",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Numerical variable compared between groups.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Grouping variable",
                desc="Categorical variable containing the two groups.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            parameters={
                "alt_hypothesis": specs.ParameterSpecification(
                    label="Alternative hypothesis",
                    desc="Direction of the alternative hypothesis.",
                    types=[specs.ParameterType.TEXT],
                    required=True,
                    multiple=False,
                    default="two-sided",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["two-sided", "less", "greater"],
                    ),
                ),
                "groupA": specs.ParameterSpecification(
                    label="Group A",
                    desc="Grouping-variable category used as group A.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
                    ),
                ),
                "groupB": specs.ParameterSpecification(
                    label="Group B",
                    desc="Grouping-variable category used as group B.",
                    types=[specs.ParameterType.TEXT, specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_CDE_ENUMS,
                        source=["x"],
                    ),
                ),
                "num_bins": specs.ParameterSpecification(
                    label="Number of bins",
                    desc="Histogram bins used to approximate ranks. More bins increase accuracy.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=40,
                    min=2,
                    max=200,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        alternative = self.get_parameter("alt_hypothesis")
        group_a = self.get_parameter("groupA")
        group_b = self.get_parameter("groupB")
        num_bins = int(self.get_parameter("num_bins", 40) or 40)

        if group_a == group_b:
            raise BadInputError(
                "groupA and groupB must select two different categories; "
                f"both were set to '{group_a}'."
            )

        result = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": self.y[0],
                "group_var": self.x[0],
                "group_a": group_a,
                "group_b": group_b,
                "alternative": alternative,
                "num_bins": num_bins,
            },
            identical_results=True,
        )
        return BinnedMannWhitneyUTestResult(**result)


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client, data, y_var, group_var, group_a, group_b, alternative, num_bins
):
    import numpy as np

    grouping = (
        data[group_var].squeeze()
        if hasattr(data[group_var], "squeeze")
        else data[group_var]
    )
    values = data[y_var]

    sample_a = np.asarray(values[grouping == group_a], dtype=float).reshape(-1)
    sample_b = np.asarray(values[grouping == group_b], dtype=float).reshape(-1)

    aggregator = NumpyAggregator(agg_client)
    test = FederatedBinnedMannWhitneyUTest(aggregator)
    return test.compute(
        sample_a,
        sample_b,
        alternative=str(alternative),
        num_bins=int(num_bins),
    )
