from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated import FederatedDescriptiveStatistics

HistogramBin = Union[float, str]


class HistogramResultItem(BaseModel):
    var: str
    grouping_var: Optional[str] = None
    grouping_enum: Optional[str] = None
    bins: List[HistogramBin]
    counts: List[Optional[int]]


class HistogramResult(BaseModel):
    histogram: List[HistogramResultItem]


class Histogram(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="histogram",
            desc="Federated histogram with optional grouping and privacy masking.",
            label="Histogram",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="Target variable",
                    desc="Numerical or categorical variable to bin into a histogram.",
                    types=[
                        specs.InputDataType.REAL,
                        specs.InputDataType.INT,
                        specs.InputDataType.TEXT,
                    ],
                    stattypes=[
                        specs.InputDataStatType.NUMERICAL,
                        specs.InputDataStatType.NOMINAL,
                    ],
                    required=True,
                    multiple=False,
                    enumslen=None,
                ),
                x=specs.InputDataSpecification(
                    label="Grouping variables",
                    desc="Optional categorical variables used to produce grouped histograms.",
                    types=[specs.InputDataType.INT, specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=False,
                    multiple=True,
                    enumslen=None,
                ),
                validation=None,
            ),
            parameters={
                "bins": specs.ParameterSpecification(
                    label="Number of bins",
                    desc="Bin count for numerical histograms (ignored for categorical targets).",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=20,
                    enums=None,
                    dict_keys_enums=None,
                    dict_values_enums=None,
                    min=1,
                    max=100,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x or []

        default_bins = 20
        bins = self.get_parameter("bins", default_bins)
        if bins is None:
            bins = default_bins

        payload = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "metadata": self.metadata,
                "bins": bins,
            },
            identical_results=True,
        )

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

    metadata_subset = {var: metadata[var] for var in {y_var, *x_vars}}
    min_row_count = worker_config.privacy.minimum_row_count
    descriptive_stats = FederatedDescriptiveStatistics(agg_client=agg_client)
    result = descriptive_stats.hist(
        data=data,
        y_var=y_var,
        x_vars=x_vars,
        metadata=metadata_subset,
        bins=bins,
        min_row_count=min_row_count,
    )
    return result.as_payload()
