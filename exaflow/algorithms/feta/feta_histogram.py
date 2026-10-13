from typing import List
from typing import Optional
from typing import Union

import numpy as np
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.algorithms.feta._histogram import _wilkinson_step
from exaflow.algorithms.feta._histogram import mask_counts
from exaflow.algorithms.feta.algorithm import Algorithm
from exaflow.worker_communication import BadUserInput

HistogramBin = Union[float, str]


class HistogramResultItem(BaseModel):
    var: str
    grouping_var: Optional[str] = None
    grouping_enum: Optional[str] = None
    bins: List[HistogramBin]
    counts: List[Optional[int]]


class HistogramResult(BaseModel):
    histogram: List[HistogramResultItem]


class FetaHistogram(Algorithm):
    column_inputs = frozenset({"x", "y"})

    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="feta_histogram",
            desc="Histogram counts for a numerical or categorical variable, with optional grouping.",
            documentation=(
                "Compute a histogram for a numerical or categorical target "
                "variable, optionally grouped by categorical variables. Counts "
                "that do not satisfy the minimum row-count privacy threshold "
                "are masked.\n\n"
                "The 'bins' setting controls the number of bins for numerical "
                "targets and is ignored for categorical targets. Default is 20.\n\n"
                "The 'histogram_type' setting controls the binning strategy for "
                "numerical targets: 'simple' (default) divides the range into "
                "equal-width bins, 'wilkinson' snaps bin edges to nice numbers. "
                "Ignored for categorical targets.\n\n"
                "The result includes histogram bins or categories and their "
                "counts, with grouped series when grouping variables are "
                "provided.\n\n"
                "Reference behavior is aligned with standard histogram "
                "counting methodology. Numerical histograms use selected bin "
                "boundaries and categorical histograms count observed category "
                "levels from aggregated counts without sharing raw data. "
                "Missing values are rejected; preprocessing is not supported."
            ),
            label="Histogram",
            enabled=True,
            required_preprocessing=[],
            y=specs.InputDataSpecification(
                label="Variable",
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
                types=[specs.InputDataType.TEXT],
                stattypes=[specs.InputDataStatType.NOMINAL],
                required=False,
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
            type=specs.AlgorithmType.FETA,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    def run(self, y: FedColumns, x: FedColumns | None = None) -> HistogramResult:
        name = y.names[0]
        target = y[name][:, 0]
        metadata = y.metadata[name]
        minimum = target.runtime.minimum_group_size
        if minimum is None:
            raise ValueError("FETA worker privacy minimum is unavailable.")
        if metadata.is_categorical:
            if target.dtype.kind == "U":
                observed = fnp.unique(target).tolist()
                categories = list(metadata.categories)
                categories += [code for code in observed if code not in categories]
                counts = np.array([int(fnp.sum(target == code)) for code in categories])
            else:
                categories = list(metadata.categories)
                counts = np.array([int(fnp.sum(target == code)) for code in categories])
                if int(counts.sum()) != y.nrows:
                    raise BadUserInput(
                        f"Categorical variable {name} contains undeclared codes."
                    )
            bins = [str(code) for code in categories]

            def count_sample(sample):
                return [int(fnp.sum(sample == code)) for code in categories]
        else:
            lower, upper = float(fnp.min(target)), float(fnp.max(target))
            if not np.isfinite(lower) or not np.isfinite(upper):
                raise BadUserInput("No finite data available to compute histogram.")
            requested = self.parameters.get("bins") or 20
            strategy = self.parameters.get("histogram_type") or "simple"
            if strategy == "wilkinson":
                _, lower, upper, requested = _wilkinson_step(
                    lower,
                    upper,
                    requested,
                    min_step=1.0 if metadata.source_type == "int" else 0.0,
                )
            elif lower == upper:
                upper = lower + 1.0
            counts, edges = fnp.histogram(target, bins=requested, range=(lower, upper))
            bins = edges.tolist()

            def count_sample(sample):
                return fnp.histogram(sample, bins=edges)[0]

        result = [
            HistogramResultItem(
                var=name, bins=bins, counts=mask_counts(counts, minimum)
            )
        ]
        if x is not None:
            for grouping_name in x.names:
                grouping = x[grouping_name][:, 0]
                groups = list(x.metadata[grouping_name].categories)
                if not groups:
                    groups = fnp.unique(grouping).tolist()
                for code in groups:
                    sample = target[grouping == code]
                    result.append(
                        HistogramResultItem(
                            var=name,
                            grouping_var=grouping_name,
                            grouping_enum=str(code),
                            bins=bins,
                            counts=mask_counts(count_sample(sample), minimum),
                        )
                    )
        return HistogramResult(histogram=result)
