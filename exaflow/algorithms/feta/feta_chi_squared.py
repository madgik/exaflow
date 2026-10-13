from typing import List

from pydantic import BaseModel
from scipy.stats import chi2_contingency

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta._contingency import contingency_table
from exaflow.algorithms.feta.algorithm import Algorithm


class ChiSquaredResult(BaseModel):
    chi2: float
    p_value: float
    dof: int
    expected: List[List[float]]
    x_labels: List[str]
    y_labels: List[str]


class FetaChiSquared(Algorithm):
    column_inputs = frozenset({"x", "y"})

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_chi_squared",
            desc="Chi-squared test of independence for two categorical variables.",
            documentation=(
                "Tests whether two categorical variables are independent by "
                "forming a contingency table and comparing observed counts "
                "with expected counts under independence. Inputs with missing "
                "values are rejected; preprocessing is not supported.\n\n"
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
            required_preprocessing=[],
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
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self, x: FedColumns, y: FedColumns) -> ChiSquaredResult:
        table, x_labels, y_labels = contingency_table(x, y)
        chi2, p_value, dof, expected = chi2_contingency(table)
        return ChiSquaredResult(
            chi2=chi2,
            p_value=p_value,
            dof=dof,
            expected=expected.tolist(),
            x_labels=x_labels,
            y_labels=y_labels,
        )
