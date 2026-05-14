from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.preprocessing.outlier_winsorizer import (
    OutlierWinsorizer,
)
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.outlier_report import (
    FederatedOutlierReport,
)
from exaflow.algorithms.federated.statistics.outlier_report import OutlierStrategy
from exaflow.algorithms.federated.statistics.outlier_report import OutlierTail
from exaflow.algorithms.federated.statistics.outlier_report import WinsorizationHelper
from exaflow.worker_communication import BadUserInput


class OutlierReportData(BaseModel):
    strategy: str
    tail: str
    fold: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_outlier_count: Optional[int] = None
    upper_outlier_count: Optional[int] = None
    total_outlier_count: Optional[int] = None
    total_outlier_percentage: Optional[float] = None


class OutlierReportRecord(BaseModel):
    variable: str
    dataset: str
    data: OutlierReportData


class OutlierReportResult(BaseModel):
    featurewise: List[OutlierReportRecord]


class OutlierReport(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        strategy_values = [strategy.value for strategy in OutlierStrategy]
        tail_values = [tail.value for tail in OutlierTail]
        documentation = (
            "Report numerical outliers per dataset using winsorization bounds "
            "computed independently on each worker's local dataset, not from "
            "global statistics. The report returns the strategy, tail, fold, "
            "computed lower and upper bounds, outlier counts, and outlier "
            "percentage for each configured variable.\n\n"
            "Configure one reporting strategy per variable with 'strategies':\n"
            "  - 'gaussian' reports values outside mean +/- fold * sample "
            "standard deviation, computed locally on each worker. Default fold "
            "is 3.0.\n"
            "  - 'iqr' reports values outside Q1 - fold * IQR and Q3 + fold * "
            "IQR, where IQR=Q3-Q1, computed locally on each worker. Default fold "
            "is 1.5.\n"
            "  - 'mad' reports values outside median +/- fold * normalized MAD, "
            "where normalized MAD=1.4826 * median absolute deviation. Default "
            "fold is 3.0. The median and MAD are computed locally on each worker.\n"
            "  - 'quantile' reports values below the fold quantile and above "
            "the 1 - fold quantile. Default fold is 0.05. Its fold must be "
            "greater than 0 and less than 0.5. Quantiles are computed locally "
            "on each worker.\n\n"
            "The optional 'tails' setting controls which side is inspected "
            "per variable:\n"
            "  - 'left' reports only values below the lower bound.\n"
            "  - 'right' reports only values above the upper bound.\n"
            "  - 'both' (default) reports both sides for variables without an "
            "explicit tail.\n\n"
            "The optional 'folds' setting overrides the strategy default per "
            "variable:\n"
            "  - 'gaussian', 'iqr', and 'mad' folds must be positive finite "
            "numbers.\n"
            "  - 'quantile' folds must be finite probabilities in (0, 0.5)."
        )
        return specs.AlgorithmSpecification(
            name="outlier_report",
            desc="Report numerical outliers per dataset using winsorization bounds.",
            documentation=documentation,
            label="Outlier Report",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Variables",
                    desc="One or more numerical variables to inspect for outliers.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=True,
                ),
                x=specs.InputDataSpecification(
                    label="Additional variables",
                    desc="Optional numerical variables to inspect for outliers.",
                    types=[specs.InputDataType.REAL, specs.InputDataType.INT],
                    stattypes=[specs.InputDataStatType.NUMERICAL],
                    required=False,
                ),
            ),
            parameters={
                "strategies": specs.ParameterSpecification(
                    label="Strategies",
                    desc="Outlier reporting strategy for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=True,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=strategy_values,
                    ),
                ),
                "tails": specs.ParameterSpecification(
                    label="Tails",
                    desc="Side of the distribution to inspect for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=tail_values,
                    ),
                ),
                "folds": specs.ParameterSpecification(
                    label="Folds",
                    desc="Strategy-specific reporting threshold for each variable.",
                    types=[specs.ParameterType.DICT],
                    required=False,
                    multiple=False,
                    dict_keys_enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.INPUT_VAR_NAMES,
                        source=["x", "y"],
                    ),
                    dict_values_type=specs.ParameterDictValueType.REAL,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[],
        )

    @property
    def check_min_rows(self) -> bool:
        return False

    @property
    def add_dataset_variable(self) -> bool:
        return True

    def run(self):
        strategies = self._dict_parameter("strategies")
        tails = self._dict_parameter("tails")
        folds = self._dict_parameter("folds")
        self._validate_parameters(strategies=strategies, folds=folds)

        rules = WinsorizationHelper.make_rules(
            strategies=strategies,
            tails=tails,
            folds=folds,
        )
        local_results = self.run_local_udf(
            func=local_step,
            kw_args={
                "rules": {
                    variable: {
                        "strategy": rule.strategy,
                        "tail": rule.tail,
                        "fold": rule.fold,
                    }
                    for variable, rule in rules.items()
                },
            },
        )
        records = [
            OutlierReportRecord(**record)
            for worker_result in local_results
            for record in worker_result["featurewise"]
        ]
        return OutlierReportResult(featurewise=records)

    def _dict_parameter(self, name: str) -> Dict[str, object]:
        value = self.get_parameter(name, {}) or {}
        if not isinstance(value, dict):
            raise BadUserInput(f"Parameter '{name}' should be a dictionary.")
        return {str(key): val for key, val in value.items()}

    def _validate_parameters(
        self,
        *,
        strategies: Dict[str, object],
        folds: Dict[str, object],
    ) -> None:
        OutlierWinsorizer.validate_outlier_configuration(
            strategies=strategies,
            folds=folds,
            metadata=self.metadata,
            parameter_label="Outlier strategy",
        )


@exareme3_udf()
def local_step(data, rules):
    from exaflow.worker import config as worker_config

    typed_rules = {
        variable: WinsorizationHelper.make_rules(
            strategies={variable: payload["strategy"]},
            tails={variable: payload["tail"]},
            folds={variable: payload["fold"]},
        )[variable]
        for variable, payload in rules.items()
    }
    report = FederatedOutlierReport()
    return {
        "featurewise": report.report(
            data=data,
            rules=typed_rules,
            min_row_count=worker_config.privacy.minimum_row_count,
        )
    }
