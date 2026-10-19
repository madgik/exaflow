from __future__ import annotations

from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from exaflow.column_names import DATASET_COL
from exaflow.worker_communication import InsufficientDataError


class OutlierStrategy(str, Enum):
    GAUSSIAN = "gaussian"
    IQR = "iqr"
    MAD = "mad"
    QUANTILE = "quantile"


class OutlierTail(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


DEFAULT_FOLDS = {
    OutlierStrategy.GAUSSIAN.value: 3.0,
    OutlierStrategy.IQR.value: 1.5,
    OutlierStrategy.MAD.value: 3.0,
    OutlierStrategy.QUANTILE.value: 0.05,
}


class OutlierRule:
    def __init__(
        self,
        variable: str,
        strategy: str,
        tail: str,
        fold: float,
    ):
        self.variable = variable
        self.strategy = strategy
        self.tail = tail
        self.fold = fold


class OutlierBounds:
    def __init__(
        self,
        lower: Optional[float],
        upper: Optional[float],
    ):
        self.lower = lower
        self.upper = upper


class WinsorizationHelper:
    @classmethod
    def make_rules(
        cls,
        *,
        strategies: Dict[str, str],
        tails: Optional[Dict[str, str]] = None,
        folds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, OutlierRule]:
        return {
            variable: OutlierRule(
                variable=variable,
                strategy=strategy,
                tail=(tails or {}).get(variable, OutlierTail.BOTH.value),
                fold=cls.resolve_fold(strategy, (folds or {}).get(variable)),
            )
            for variable, strategy in strategies.items()
        }

    @staticmethod
    def resolve_fold(strategy: str, fold: Optional[float]) -> float:
        value = DEFAULT_FOLDS[strategy] if fold is None else float(fold)
        if not np.isfinite(value):
            raise ValueError("Fold should be finite.")
        if strategy == OutlierStrategy.QUANTILE.value:
            if value <= 0 or value >= 0.5:
                raise ValueError(
                    "Quantile fold should be greater than 0 and less than 0.5."
                )
            return value
        if value <= 0:
            raise ValueError("Fold should be greater than 0.")
        return value

    @staticmethod
    def compute_bounds(series: pd.Series, rule: OutlierRule) -> OutlierBounds:
        values = pd.to_numeric(series.dropna(), errors="coerce").dropna()

        if rule.strategy == OutlierStrategy.GAUSSIAN.value:
            mean = float(values.mean())
            std = float(values.std(ddof=1))
            if not np.isfinite(std):
                std = 0.0
            lower = mean - rule.fold * std
            upper = mean + rule.fold * std
        elif rule.strategy == OutlierStrategy.IQR.value:
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - rule.fold * iqr
            upper = q3 + rule.fold * iqr
        elif rule.strategy == OutlierStrategy.MAD.value:
            median = float(values.median())
            mad = float((values - median).abs().median())
            normalized_mad = 1.4826 * mad
            lower = median - rule.fold * normalized_mad
            upper = median + rule.fold * normalized_mad
        elif rule.strategy == OutlierStrategy.QUANTILE.value:
            lower = float(values.quantile(rule.fold))
            upper = float(values.quantile(1.0 - rule.fold))
        else:  # pragma: no cover - validated before this point
            raise ValueError(f"Unsupported outlier strategy '{rule.strategy}'.")

        if rule.tail == OutlierTail.LEFT.value:
            return OutlierBounds(lower=lower, upper=None)
        if rule.tail == OutlierTail.RIGHT.value:
            return OutlierBounds(lower=None, upper=upper)
        return OutlierBounds(lower=lower, upper=upper)

    @staticmethod
    def count_outliers(series: pd.Series, bounds: OutlierBounds) -> tuple[int, int]:
        values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        lower_count = (
            int((values < bounds.lower).sum()) if bounds.lower is not None else 0
        )
        upper_count = (
            int((values > bounds.upper).sum()) if bounds.upper is not None else 0
        )
        return lower_count, upper_count


class FederatedOutlierReport:
    """Local per-dataset outlier report using winsorization bounds."""

    def report(
        self,
        *,
        data: pd.DataFrame,
        rules: Dict[str, OutlierRule],
        min_row_count: int,
        dataset_col: str = DATASET_COL,
    ) -> List[Dict[str, object]]:
        if dataset_col in data.columns:
            dataset_values = data[dataset_col]
            if isinstance(dataset_values, pd.DataFrame):
                dataset_values = dataset_values.iloc[:, 0]
            datasets = sorted(dataset_values.dropna().unique().tolist())
        else:
            dataset_values = None
            datasets = ["unknown"]

        records: List[Dict[str, object]] = []
        for dataset in datasets:
            dataset_data = (
                data.loc[dataset_values == dataset]
                if dataset_values is not None
                else data
            )
            for variable, rule in rules.items():
                records.append(
                    {
                        "variable": variable,
                        "dataset": str(dataset),
                        "data": self._report_variable(
                            data=dataset_data,
                            rule=rule,
                            min_row_count=min_row_count,
                        ),
                    }
                )
        return records

    def _report_variable(
        self,
        *,
        data: pd.DataFrame,
        rule: OutlierRule,
        min_row_count: int,
    ) -> Dict[str, object]:
        series = data[rule.variable]
        values = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        num_dtps = int(len(values))
        if num_dtps < min_row_count:
            raise InsufficientDataError(
                f"Insufficient non-missing data for variable '{rule.variable}': "
                f"{num_dtps} rows; minimum required is {min_row_count}."
            )

        bounds = WinsorizationHelper.compute_bounds(values, rule)
        lower_count, upper_count = WinsorizationHelper.count_outliers(values, bounds)
        lower_masked = self._mask_count(lower_count, min_row_count)
        upper_masked = self._mask_count(upper_count, min_row_count)
        total_count = lower_count + upper_count

        has_suppressed_side = lower_masked is None or upper_masked is None
        total_masked = None if has_suppressed_side else total_count
        total_percentage = (
            None if total_masked is None else float((total_count / num_dtps) * 100.0)
        )

        return {
            "strategy": rule.strategy,
            "tail": rule.tail,
            "fold": rule.fold,
            "lower_bound": self._json_float(bounds.lower),
            "upper_bound": self._json_float(bounds.upper),
            "lower_outlier_count": (None if bounds.lower is None else lower_masked),
            "upper_outlier_count": (None if bounds.upper is None else upper_masked),
            "total_outlier_count": total_masked,
            "total_outlier_percentage": self._json_float(total_percentage),
        }

    @staticmethod
    def _mask_count(count: int, min_row_count: int) -> Optional[int]:
        if count == 0:
            return 0
        if count < min_row_count:
            return None
        return count

    @staticmethod
    def _json_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not np.isfinite(value):
            return None
        return float(value)
