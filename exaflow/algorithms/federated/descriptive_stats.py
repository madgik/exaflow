from __future__ import annotations

from typing import Dict
from typing import List

import pandas as pd

from exaflow.algorithms.federated.describe import DescribeResult
from exaflow.algorithms.federated.describe import FederatedDescribe
from exaflow.algorithms.federated.histogram import FederatedHistogram
from exaflow.algorithms.federated.histogram import HistogramResult


class FederatedDescriptiveStatistics:
    """
    Statsmodels-style descriptive interface backed by federated primitives.

    ``describe`` returns per-variable summaries for each dataset together with
    aggregation-backed global aggregates. ``hist`` provides federated histograms
    (numerical or categorical) with optional grouping via ``x_vars``.

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
        self._hist = FederatedHistogram(agg_client)

    def describe(
        self,
        *,
        data: pd.DataFrame,
        numerical_vars: List[str],
        nominal_vars: List[str],
        min_row_count: int,
        nominal_levels: Dict[str, List],
        dataset_col: str = "dataset",
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
        bins: int,
        min_row_count: int,
    ) -> HistogramResult:
        return self._hist.hist(
            data=data,
            y_var=y_var,
            x_vars=x_vars,
            metadata=metadata,
            bins=bins,
            min_row_count=min_row_count,
        )
