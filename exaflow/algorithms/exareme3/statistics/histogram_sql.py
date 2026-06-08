from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.worker_communication import InsufficientDataError

HistogramBin = Union[float, str]


class HistogramResultItem(BaseModel):
    var: str
    grouping_var: Optional[str] = None
    grouping_enum: Optional[str] = None
    bins: List[HistogramBin]
    counts: List[Optional[int]]


class HistogramResult(BaseModel):
    histogram: List[HistogramResultItem]


class HistogramSQL(Algorithm):
    """Histogram algorithm backed directly by histogram_base classes.

    Supports ``WilkinsonHistogram`` (nice-number bin boundaries) and
    ``SimpleHistogram`` (equal-width bins) for numerical variables, and
    ``CategoricalHistogram`` for nominal variables.  Grouped histograms are
    produced for each variable listed in ``x``.
    """

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="histogram_sql",
            desc="Federated histogram with optional grouping.",
            documentation=(
                "Compute a federated histogram over a numerical or categorical variable "
                "without sharing raw data across workers.\n\n"
                "For numerical variables, bins are determined either by a Wilkinson "
                "nice-number algorithm ('wilkinson') or by equal-width division "
                "('simple').  For categorical variables, counts are computed per "
                "category using a federated union to discover global categories.\n\n"
                "An optional set of categorical grouping variables (x) produces "
                "one histogram per group level.  Counts below the privacy threshold "
                "are masked as null."
            ),
            label="Histogram (SQL)",
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
                    desc="Optional categorical variables for grouped histograms.",
                    types=[specs.InputDataType.INT, specs.InputDataType.TEXT],
                    stattypes=[specs.InputDataStatType.NOMINAL],
                    required=False,
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
                "histogram_type": specs.ParameterSpecification(
                    label="Histogram type",
                    desc=(
                        "Binning strategy for numerical histograms: "
                        "'wilkinson' for nice-number boundaries, "
                        "'simple' for equal-width bins."
                    ),
                    types=[specs.ParameterType.TEXT],
                    required=False,
                    multiple=False,
                    default="wilkinson",
                    enums=specs.ParameterEnumSpecification(
                        type=specs.ParameterEnumType.LIST,
                        source=["wilkinson", "simple"],
                    ),
                    dict_keys_enums=None,
                    dict_values_enums=None,
                ),
            },
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self):
        y_var = self.inputdata.y[0]
        x_vars = self.inputdata.x or []

        bins = self.get_parameter("bins", 20) or 20
        histogram_type = (
            self.get_parameter("histogram_type", "wilkinson") or "wilkinson"
        )

        payload = self.run_local_udf(
            func=local_step,
            kw_args={
                "y_var": y_var,
                "x_vars": x_vars,
                "bins": bins,
                "histogram_type": histogram_type,
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
            for group, counts in zip(grouped["groups"], grouped["counts"]):
                histograms.append(
                    HistogramResultItem(
                        var=y_var,
                        grouping_var=grouping_var,
                        grouping_enum=str(group),
                        bins=base_bins,
                        counts=counts,
                    )
                )

        return HistogramResult(histogram=histograms)


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client, data, y_var, x_vars, metadata, bins, histogram_type="wilkinson"
):
    import numpy as np

    from exaflow.algorithms.federated.statistics.histogram_base import (
        CategoricalHistogram,
    )
    from exaflow.algorithms.federated.statistics.histogram_base import SimpleHistogram
    from exaflow.algorithms.federated.statistics.histogram_base import (
        WilkinsonHistogram,
    )
    from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
        NumpyAggregator,
    )
    from exaflow.worker import config as worker_config

    selected_columns = list(dict.fromkeys([y_var, *x_vars]))
    data = data[selected_columns].dropna(axis=0, how="any")
    _check_min_rows_or_raise(
        data=data, min_required=worker_config.privacy.minimum_row_count
    )

    min_row_count = worker_config.privacy.minimum_row_count
    aggregator = NumpyAggregator(agg_client)
    y_meta = metadata.get(y_var, {})
    is_categorical = y_meta.get("is_categorical", False)

    if is_categorical:
        # When enumerations are defined in metadata, honour their order and
        # perform a simple local count + fed_sum.  Otherwise let
        # CategoricalHistogram discover global categories via fed_union.
        if y_meta.get("enumerations"):
            categories = list(y_meta["enumerations"].keys())
            y_str = data[y_var].astype(str)
            local_counts = np.array(
                [float((y_str == str(c)).sum()) for c in categories], dtype=float
            )
            global_counts = aggregator.fed_sum(local_counts)
        else:
            cat_hist = CategoricalHistogram(aggregator)
            cats, global_counts = cat_hist.compute(data[y_var].to_numpy(dtype=object))
            categories = cats.tolist()

        masked = _mask_counts(global_counts, min_row_count)

        grouped = {}
        for x_var in x_vars:
            groups = _get_groups(aggregator, metadata, x_var, data[x_var])
            n_g, n_c = len(groups), len(categories)
            matrix = np.zeros((n_g, n_c), dtype=float)
            y_str = data[y_var].astype(str)
            x_str = data[x_var].astype(str)
            for i, group in enumerate(groups):
                subset_y = y_str[x_str == str(group)]
                matrix[i] = [float((subset_y == str(c)).sum()) for c in categories]
            global_matrix = np.asarray(
                aggregator.fed_sum(matrix.flatten()), dtype=float
            ).reshape(n_g, n_c)
            grouped[x_var] = {
                "groups": [str(g) for g in groups],
                "counts": [_mask_counts(row, min_row_count) for row in global_matrix],
            }

        return {
            "bins": [str(c) for c in categories],
            "counts": masked,
            "grouped": grouped,
        }

    else:
        values = data[y_var].to_numpy(dtype=float, copy=False)
        bins = max(1, int(round(bins)))

        if histogram_type == "wilkinson":
            hist_algo = WilkinsonHistogram(aggregator)
        else:
            hist_algo = SimpleHistogram(aggregator)

        global_hist, bin_edges = hist_algo.compute(values, bins)
        masked = _mask_counts(global_hist, min_row_count)

        grouped = {}
        for x_var in x_vars:
            groups = _get_groups(aggregator, metadata, x_var, data[x_var])
            n_g, n_b = len(groups), len(bin_edges) - 1
            matrix = np.zeros((n_g, n_b), dtype=float)
            for i, group in enumerate(groups):
                mask = data[x_var].astype(str) == str(group)
                subset = data.loc[mask, y_var].to_numpy(dtype=float, copy=False)
                local_hist, _ = np.histogram(subset, bins=bin_edges)
                matrix[i] = local_hist.astype(float)
            global_matrix = np.asarray(
                aggregator.fed_sum(matrix.flatten()), dtype=float
            ).reshape(n_g, n_b)
            grouped[x_var] = {
                "groups": [str(g) for g in groups],
                "counts": [_mask_counts(row, min_row_count) for row in global_matrix],
            }

        return {"bins": bin_edges.tolist(), "counts": masked, "grouped": grouped}


def _get_groups(aggregator, metadata, x_var, series):
    """Return groups for a grouping variable.

    Uses metadata enumerations when available so that all workers agree on the
    same ordered list without a network round-trip.  Falls back to a federated
    union when enumerations are absent, ensuring consistency across workers.
    """
    import numpy as np

    x_meta = metadata.get(x_var, {})
    if x_meta and x_meta.get("enumerations"):
        return list(x_meta["enumerations"].keys())
    local_groups = series.dropna().astype(str).unique()
    global_groups = aggregator.fed_union(np.array(local_groups, dtype=object))
    return sorted([str(g) for g in global_groups])


def _mask_counts(values, min_row_count: int):
    """Replace counts below *min_row_count* with ``None`` (privacy mask)."""
    result = []
    for v in values:
        c = int(round(float(v)))
        result.append(c if c >= min_row_count else None)
    return result


def _check_min_rows_or_raise(*, data, min_required: int) -> None:
    num_rows = len(data)
    if num_rows < min_required:
        raise InsufficientDataError(
            f"Insufficient data: {num_rows} rows; minimum required is {min_required}."
        )
