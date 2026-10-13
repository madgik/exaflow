from typing import List

from pydantic import BaseModel
from scipy.stats import fisher_exact

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._contingency import indicator_table
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.worker_communication import BadUserInput


class FisherExactResult(BaseModel):
    odds_ratio: float
    p_value: float
    x_labels: List[str]
    y_labels: List[str]


class FetaFisherExact(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_fisher_exact",
            desc="Fisher's exact test for association between two binary categorical variables.",
            documentation=(
                "Tests association between two categorical variables by "
                "building a 2x2 contingency table and applying Fisher's exact "
                "test. Each selected variable must have exactly two observed "
                "levels. Missing values are rejected; preprocessing is not supported.\n\n"
                "The result includes the odds ratio, p-value, and category "
                "labels for both variables.\n\n"
                "Reference behavior is aligned with scipy.stats.fisher_exact "
                "for a 2x2 contingency table. The method computes the table "
                "from aggregated category counts without sharing raw data."
            ),
            label="Fisher's Exact Test",
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

    def run(self, x: FedTensor, y: FedTensor) -> FisherExactResult:
        x_categories = fnp.unique(x).tolist()
        y_categories = fnp.unique(y).tolist()
        if len(x_categories) != 2 or len(y_categories) != 2:
            raise BadUserInput(
                "Fisher's Exact Test requires exactly 2 observed levels for both "
                f"the factor and the outcome, but factor {self.x_names[0]!r} has "
                f"{len(x_categories)} and outcome {self.y_names[0]!r} has "
                f"{len(y_categories)}."
            )
        table = indicator_table(x, y, x_categories, y_categories)
        odds_ratio, p_value = fisher_exact(table)
        return FisherExactResult(
            odds_ratio=odds_ratio,
            p_value=p_value,
            x_labels=[str(code) for code in x_categories],
            y_labels=[str(code) for code in y_categories],
        )
