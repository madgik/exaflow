from typing import Dict
from typing import List

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._anova import AnovaStatistics
from exaflow.algorithms.feta._anova import get_min_max_ci_info
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.algorithms.specifications import AlgorithmSpecification
from exaflow.algorithms.specifications import AlgorithmType
from exaflow.algorithms.specifications import ComponentType
from exaflow.algorithms.specifications import InputDataSpecification
from exaflow.algorithms.specifications import InputDataStatType
from exaflow.algorithms.specifications import InputDataType
from exaflow.worker_communication import BadUserInput


class AnovaResult(BaseModel):
    anova_table: Dict
    tuckey_test: List[Dict]
    min_max_per_group: Dict
    ci_info: Dict


class FetaAnovaOneWay(Algorithm):
    @classmethod
    def get_specification(cls):
        specification = AlgorithmSpecification(
            name="feta_anova_oneway",
            desc="One-way ANOVA for a numerical outcome and one categorical factor.",
            documentation=(
                "Tests whether a numerical outcome differs across groups "
                "defined by one categorical factor. The ANOVA table decomposes "
                "variation into between-group and within-group components and "
                "reports the F statistic and p-value for the group effect.\n\n"
                "The result includes the ANOVA table, Tukey HSD pairwise "
                "comparisons, group min/max summaries, group means, sample "
                "standard deviations, and confidence interval information.\n\n"
                "Reference behavior is aligned with classical one-way ANOVA "
                "and Tukey HSD methodology as exposed by statsmodels and "
                "scipy.stats. The method computes the ANOVA quantities from "
                "aggregated group counts, means, and sums of squares without "
                "sharing raw data."
            ),
            label="One-way ANOVA",
            enabled=True,
            required_preprocessing=[],
            y=InputDataSpecification(
                label="Outcome",
                desc="Numerical outcome variable.",
                types=[InputDataType.REAL, InputDataType.INT],
                stattypes=[InputDataStatType.NUMERICAL],
                required=True,
                max_count=1,
            ),
            x=InputDataSpecification(
                label="Grouping variable",
                desc="Categorical variable defining the groups.",
                types=[InputDataType.TEXT],
                stattypes=[InputDataStatType.NOMINAL],
                required=True,
                max_count=1,
            ),
            type=AlgorithmType.FETA,
            components=[ComponentType.AGGREGATION_SERVER],
        )
        return specification.model_copy(
            update={
                "name": "feta_anova_oneway",
                "type": AlgorithmType.FETA,
                "required_preprocessing": [],
                "documentation": specification.documentation
                + "\n\nInputs containing missing values are rejected. Preprocessing is not supported.",
            }
        )

    def run(self, x: FedTensor, y: FedTensor) -> AnovaResult:
        grouping = x[:, 0]
        categories = fnp.unique(grouping).tolist()
        if len(categories) < 2:
            raise BadUserInput(
                "Cannot perform Anova one-way. Covariable has only one level."
            )
        counts, sums, squares, minima, maxima = [], [], [], [], []
        for category in categories:
            sample = y[grouping == category]
            counts.append(len(sample))
            sums.append(fnp.sum(sample))
            squares.append(fnp.sum(sample * sample))
            minima.append(fnp.min(sample))
            maxima.append(fnp.max(sample))
        counts = np.asarray(counts, dtype=float)
        sums = np.asarray(sums, dtype=float)
        squares = np.asarray(squares, dtype=float)
        statistics = {
            "group_stats_count": counts,
            "group_stats_sum": sums,
            "group_ssq": squares,
            "var_min_per_group": np.asarray(minima),
            "var_max_per_group": np.asarray(maxima),
            "n_obs": int(counts.sum()),
            "overall_sum": sums.sum(),
            "overall_count": counts.sum(),
            "overall_ssq": squares.sum(),
            "group_stats_index": categories,
        }
        model = AnovaStatistics().fit(statistics, categories)
        min_max, ci_info = get_min_max_ci_info(
            means=model.means_,
            sample_stds=model.sample_stds_,
            categories=categories,
            var_min_per_group=model.var_min_per_group_,
            var_max_per_group=model.var_max_per_group_,
            group_stats_index=model.group_stats_index_,
        )
        tukey = model.thsd_.rename(
            columns={
                "A": "groupA",
                "B": "groupB",
                "mean(A)": "meanA",
                "mean(B)": "meanB",
                "Std.Err.": "se",
                "t value": "t_stat",
                "Pr(>|t|)": "p_tuckey",
            }
        ).to_dict(orient="records")
        return AnovaResult(
            anova_table={
                "n_obs": model.nobs,
                "y_label": self.y_names[0],
                "x_label": self.x_names[0],
                "df_residual": model.df_within,
                "df_explained": model.df_between,
                "ss_residual": model.ss_within,
                "ss_explained": model.ss_between,
                "ms_residual": model.ms_within,
                "ms_explained": model.ms_between,
                "p_value": model.pvalue,
                "f_stat": model.fvalue,
            },
            tuckey_test=tukey,
            min_max_per_group=min_max,
            ci_info=ci_info,
        )
