from typing import List
from typing import Optional

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._anova_twoway import compute_anova_table
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.worker_communication import BadUserInput


class AnovaResult(BaseModel):
    terms: List[str]
    sum_sq: List[float]
    df: List[int]
    f_stat: List[Optional[float]]
    f_pvalue: List[Optional[float]]


class FetaAnovaTwoWay(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_anova_twoway",
            desc=(
                "Two-way ANOVA for a numerical outcome and two categorical "
                "factors, including their interaction."
            ),
            documentation=(
                "Tests whether a numerical outcome differs across the levels "
                "of two categorical factors in a factorial model with main "
                "effects and their interaction. The fitted model is equivalent "
                "to y ~ A * B, where A and B are the two selected factors.\n\n"
                "The ANOVA table contains rows for the first factor, the "
                "second factor, their interaction, and residuals. For each "
                "testable term it reports sum of squares, degrees of freedom, "
                "F statistic, and p-value; residual rows report sum of squares "
                "and degrees of freedom only.\n\n"
                "The 'sstype' setting selects the sums-of-squares type:\n"
                "  - 1 computes Type I sequential sums of squares.\n"
                "  - 2 computes Type II marginal sums of squares. Default is 2.\n\n"
                "Reference behavior is aligned with statsmodels OLS two-way "
                "ANOVA using ols('y ~ A * B') and sm.stats.anova_lm(lm, "
                "typ=sstype). The method computes the same ANOVA quantities "
                "from aggregated sufficient statistics without sharing raw data.\n\n"
                "Inputs containing missing values are rejected. Preprocessing is not supported."
            ),
            label="Two-way ANOVA",
            enabled=True,
            required_preprocessing=[],
            y=specs.InputDataSpecification(
                label="Outcome",
                desc="Numerical outcome variable.",
                types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                stattypes=[specs.InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=specs.InputDataSpecification(
                label="Factors",
                desc="Two categorical factors.",
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=True,
                min_count=2,
                max_count=2,
            ),
            parameters={
                "sstype": specs.ParameterSpecification(
                    label="Sum of squares type",
                    desc="Sums-of-squares decomposition used for the ANOVA table.",
                    types=[specs.ParameterType.INT],
                    required=True,
                    multiple=False,
                    default="2",
                    min=1,
                    max=2,
                ),
            },
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self, x: FedTensor, y: FedTensor) -> AnovaResult:
        if len(self.x_names) != 2 or x.shape[1] != 2:
            raise BadUserInput("ANOVA two-way requires exactly two covariates (x).")
        a, b = x[:, 0], x[:, 1]
        levels_a, levels_b = fnp.unique(a).tolist(), fnp.unique(b).tolist()
        for name, levels in zip(self.x_names, (levels_a, levels_b)):
            if len(levels) < 2:
                raise BadUserInput(
                    f"The variable {name} has less than 2 levels and Anova cannot be "
                    "performed. Please choose another variable."
                )
        intercept = fnp.ones_like(y)
        main_a = [(a == level) * 1.0 for level in levels_a[1:]]
        main_b = [(b == level) * 1.0 for level in levels_b[1:]]
        interactions = [left * right for left in main_a for right in main_b]
        # The reference OLS adds a second intercept to each existing design.
        # Keep it to preserve Gram matrices, rank and solver rounding behavior.
        designs = {
            "const": [intercept, intercept],
            "a": [intercept, intercept, *main_a],
            "b": [intercept, intercept, *main_b],
            "ab": [intercept, intercept, *main_a, *main_b],
            "full": [intercept, intercept, *main_a, *main_b, *interactions],
        }
        n = len(y)
        models = {}
        for name, columns in designs.items():
            design = fnp.column_stack(columns)
            gram = design.T @ design
            cross = design.T @ y
            coefficients = np.linalg.pinv(gram) @ cross
            residuals = y - design @ coefficients
            models[name] = {
                "rss": float(fnp.sum(residuals * residuals)),
                "rank": int(np.linalg.matrix_rank(gram)),
                "n_obs": n,
            }
        table = compute_anova_table(
            model_stats=models,
            sstype=self.parameters.get("sstype", 2),
            x1_name=self.x_names[0],
            x2_name=self.x_names[1],
        )
        terms = table["terms"]
        return AnovaResult(
            terms=terms,
            **{
                key: [table[key][term] for term in terms]
                for key in ("sum_sq", "df", "f_stat", "f_pvalue")
            },
        )
