from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated import FederatedDescriptiveStatistics
from exaflow.algorithms.specifications import AlgorithmName

HistogramBin = Union[float, str]


class HistogramResultItem(BaseModel):
    var: str
    grouping_var: Optional[str]
    grouping_enum: Optional[str]
    bins: List[HistogramBin]
    counts: List[Optional[int]]


class HistogramResult(BaseModel):
    histogram: List[HistogramResultItem]


class Histogram(Algorithm, algname=AlgorithmName.HISTOGRAM):
    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x or []

        default_bins = 20
        bins = self.get_parameter("bins", default_bins)
        if bins is None:
            bins = default_bins

        metadata_subset = {var: self.metadata[var] for var in {y_var, *x_vars}}

        results = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "metadata": metadata_subset,
                "bins": bins,
            },
        )
        payload = results[0]

        histograms: List[HistogramResultItem] = []
        base_bins = payload["bins"]
        histograms.append(
            HistogramResultItem(
                var=y_var,
                grouping_var=None,
                grouping_enum=None,
                bins=base_bins,
                counts=payload["counts"],
            )
        )

        for grouping_var, grouped in payload.get("grouped", {}).items():
            groups = grouped["groups"]
            counts_per_group = grouped["counts"]
            for group, counts in zip(groups, counts_per_group):
                histograms.append(
                    HistogramResultItem(
                        var=y_var,
                        grouping_var=grouping_var,
                        grouping_enum=group,
                        bins=base_bins,
                        counts=counts,
                    )
                )

        return HistogramResult(histogram=histograms)


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, y_var, x_vars, metadata, bins):
    from exaflow.worker import config as worker_config

    min_row_count = worker_config.privacy.minimum_row_count
    descriptive_stats = FederatedDescriptiveStatistics(agg_client=agg_client)
    result = descriptive_stats.hist(
        data=data,
        y_var=y_var,
        x_vars=x_vars,
        metadata=metadata,
        bins=bins,
        min_row_count=min_row_count,
    )
    return result.as_payload()
