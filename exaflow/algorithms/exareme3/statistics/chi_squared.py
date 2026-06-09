from typing import List

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.chi_squared import (
    ChiSquared as FedChiSquared,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class ChiSquaredResult(BaseModel):
    chi2: float
    p_value: float
    dof: int
    expected: List[List[float]]
    x_labels: List[str]
    y_labels: List[str]


class ChiSquared(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="chi_squared",
            desc="Chi-squared test of independence for two categorical variables.",
            documentation=(
                "Tests whether two categorical variables are independent by "
                "forming a contingency table and comparing observed counts "
                "with expected counts under independence. Rows with missing "
                "values in either selected variable are excluded.\n\n"
                "The result includes the chi-squared statistic, p-value, "
                "degrees of freedom, expected counts, and category labels for "
                "both variables.\n\n"
                "Reference behavior is aligned with scipy.stats.chi2_contingency "
                "for a contingency table built from the selected variables. "
                "The method computes the table and expected counts from "
                "aggregated category counts without sharing raw data."
            ),
            label="Chi-squared Test",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Outcome",
                    desc="Categorical outcome variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Factor",
                    desc="Categorical factor variable.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=True,
                    max_count=1,
                ),
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_var = self.inputdata.x[0]

        results = self.run_local_udf(
            func=chi_squared_local_step,
            kw_args={
                "factor": x_var,
                "outcome": y_var,
            },
        )

        res = results[0]
        chi2, p, dof, expected, x_labels, y_labels = res
        return ChiSquaredResult(
            chi2=chi2,
            p_value=p,
            dof=dof,
            expected=expected,
            x_labels=x_labels,
            y_labels=y_labels,
        )


@exareme3_udf(with_aggregation_server=True)
def chi_squared_local_step(agg_client, data, metadata, factor, outcome):
    factor_categories = get_enum_codes(metadata, factor)
    outcome_categories = get_enum_codes(metadata, outcome)
    aggregator = NumpyAggregator(agg_client)
    model = FedChiSquared(aggregator=aggregator)
    res = model.compute(
        dataset=data,
        factor=factor,
        outcome=outcome,
        factor_categories=factor_categories,
        outcome_categories=outcome_categories,
        # Missing values are always dropped: NaN rows are excluded from the
        # contingency table. Including NaN as a category is not supported
        # because chi-squared requires clean, non-missing observations.
        dropna=True,
    )
    return (
        res[0],
        res[1],
        res[2],
        res[3].tolist() if hasattr(res[3], "tolist") else res[3],
        res.x_labels,
        res.y_labels,
    )
