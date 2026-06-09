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
            desc="Histogram with optional grouping.",
            documentation=(
                "Compute a histogram for a numerical or categorical target "
                "variable, optionally grouped by categorical variables. Counts "
                "are privacy-masked according to worker privacy rules.\n\n"
                "The 'bins' setting controls the number of bins for numerical "
                "targets and is ignored for categorical targets. Default is 20.\n\n"
                "The 'histogram_type' setting controls the binning strategy for "
                "numerical targets: 'simple' (default) divides the range into "
                "equal-width bins, 'wilkinson' snaps bin edges to nice numbers. "
                "Ignored for categorical targets.\n\n"
                "The result includes histogram bins or categories and their "
                "counts, with grouped series when grouping variables are provided."
            ),
            label="Histogram",
            enabled=True,
            required_preprocessing=["missing_values_handler"],
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
                    max_count=1,
                ),
                x=specs.InputDataSpecification(
                    label="Grouping variables",
                    desc="Optional categorical variables used to produce grouped histograms.",
                    types=[specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=False,
                ),
            ),
            parameters={
                "bins": specs.ParameterSpecification(
                    label="Number of bins",
                    desc="Bin count used for numerical histograms.",
                    types=[specs.ParameterType.INT],
                    required=False,
                    multiple=False,
                    default=20,
                    min=1,
                    max=100,
                ),
                "histogram_type": specs.ParameterSpecification(
                    label="Histogram type",
                    desc=(
                        "Binning strategy for numerical histograms: "
                        "'simple' for equal-width bins, "
                        "'wilkinson' for nice-number boundaries."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="simple",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["wilkinson", "simple"],
                    ),
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

        default_histogram_type = "simple"
        histogram_type = self.get_parameter("histogram_type", default_histogram_type)
        if histogram_type is None:
            histogram_type = default_histogram_type

        is_integer = (
            self.metadata.get(y_var, {}).get("sql_type")
            == specs.InputDataType.INT.value
        )

        payload = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "bins": bins,
                "histogram_type": histogram_type,
                "is_integer": is_integer,
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
def local_step(
    agg_client, data, y_var, x_vars, metadata, bins, histogram_type, is_integer
):
    from exaflow.worker import config as worker_config

    selected_columns = list(dict.fromkeys([y_var, *x_vars]))
    data = data[selected_columns]

    metadata_subset = {var: metadata[var] for var in {y_var, *x_vars}}
    min_row_count = worker_config.privacy.minimum_row_count
    descriptive_stats = FederatedDescriptiveStatistics(agg_client=agg_client)
    result = descriptive_stats.hist(
        data=data,
        y_var=y_var,
        x_vars=x_vars,
        metadata=metadata_subset,
        bins=bins,
        histogram_type=histogram_type,
        is_integer=is_integer,
        min_row_count=min_row_count,
    )
    return result.as_payload()
