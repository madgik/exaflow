from __future__ import annotations

from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms.federated.statistics.describe import DescribeResult
from exaflow.algorithms.federated.statistics.describe import FederatedDescribe
from exaflow.algorithms.federated.statistics.histogram import FederatedGroupedHistogram
from exaflow.algorithms.federated.statistics.histogram import HistogramResult
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    FederatedPearsonCorrelation,
)
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    PearsonCorrelationResult,
)
from exaflow.column_names import DATASET_COL


class FederatedDescriptiveStatistics:
    """
    Statsmodels-style descriptive interface backed by federated primitives.

    ``describe`` returns per-feature summaries for each dataset together
    with aggregation-backed global aggregates.
    ``hist`` provides federated histograms (numerical or categorical) with
    optional grouping via ``x_vars``.

    Notes
    -----
    - Quantiles stay local per dataset (global summaries remove quantiles once
      multiple datasets participate).
    - Means and variances come from aggregated sufficient statistics (sx, sxx).
    - Nominal counts flow through the aggregation client and zero-count levels are
      dropped in the global result per our contract.
    - NA handling remains pairwise complete per variable.
    - This is descriptive only—no inference or modeling.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client
        self._describe = FederatedDescribe(agg_client)
        self._hist = FederatedGroupedHistogram(agg_client)
        self._pearson = FederatedPearsonCorrelation(agg_client)

    def describe(
        self,
        *,
        data: pd.DataFrame,
        numerical_vars: List[str],
        nominal_vars: List[str],
        min_row_count: int,
        nominal_levels: Dict[str, List],
        dataset_col: str = DATASET_COL,
    ) -> DescribeResult:
        return self._describe.describe(
            data=data,
            numerical_vars=numerical_vars,
            nominal_vars=nominal_vars,
            min_row_count=min_row_count,
            nominal_levels=nominal_levels,
            dataset_col=dataset_col,
        )

    def hist(
        self,
        *,
        data: pd.DataFrame,
        y_var: str,
        x_vars: List[str],
        metadata: Dict,
        bins: int = 20,
        histogram_type: str = "wilkinson",
        is_integer: bool = False,
        min_row_count: int,
    ) -> HistogramResult:
        """Compute federated histogram with discovery-based categorization.

        Always discovers actual categories from data, ensuring no values are
        silently dropped when metadata enumerations are outdated. Supports
        both numerical (Wilkinson/Simple binning) and categorical histograms
        with optional grouping and privacy masking.

        ``is_integer`` tells the Wilkinson strategy that ``y_var`` holds
        integer values, so it never produces a sub-unit bin step.
        """
        return self._hist.histogram(
            data=data,
            y_var=y_var,
            x_vars=x_vars,
            metadata=metadata,
            bins=bins,
            histogram_type=histogram_type,
            is_integer=is_integer,
            min_row_count=min_row_count,
        )

    def corrcoef(
        self,
        *,
        data: pd.DataFrame,
        x_vars: List[str],
        y_vars: List[str],
        alpha: float,
    ) -> PearsonCorrelationResult:
        return self._pearson.corrcoef(
            data=data,
            x_vars=x_vars,
            y_vars=y_vars,
            alpha=alpha,
        )

    def pearson_correlation(
        self,
        *,
        data: pd.DataFrame,
        x_vars: List[str],
        y_vars: List[str],
        alpha: float,
    ) -> PearsonCorrelationResult:
        return self.corrcoef(
            data=data,
            x_vars=x_vars,
            y_vars=y_vars,
            alpha=alpha,
        )
