from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from pydantic import BaseModel

from exaflow.algorithms import specifications as specs
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.statistics.descriptive_stats import (
    FederatedDescriptiveStatistics,
)
from exaflow.column_names import DATASET_COL

DATASET_VAR_NAME = DATASET_COL


class NumericalDescriptiveStats(BaseModel):
    num_dtps: int
    num_na: int
    num_total: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    q2: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None


class NominalDescriptiveStats(BaseModel):
    num_dtps: int
    num_na: int
    num_total: int
    counts: Dict[str, int]


class Variable(BaseModel):
    variable: str
    dataset: str
    data: Union[NominalDescriptiveStats, NumericalDescriptiveStats, None]

    @classmethod
    def from_record(cls, rec):
        data = rec["data"]
        if data is None:
            return cls(**rec)
        if "counts" in data:
            rec["data"] = NominalDescriptiveStats(**data)
        else:
            rec["data"] = NumericalDescriptiveStats(**data)
        return cls(**rec)


class Result(BaseModel):
    featurewise: List[Variable]


class Describe(Algorithm):
    @classmethod
    def get_specification(cls) -> specs.AlgorithmSpecification:
        return specs.AlgorithmSpecification(
            name="describe",
            desc="Descriptive statistics for selected variables.",
            documentation=(
                "Compute descriptive statistics for numerical and nominal "
                "variables with pairwise feature-level missing-value handling.\n\n"
                "The result includes featurewise summaries and analysis-set "
                "summaries for the selected variables."
            ),
            label="Descriptive stats (Describe)",
            enabled=True,
            inputdata=specs.InputDataSpecifications(
                y=specs.InputDataSpecification(
                    label="y",
                    desc="Variables to summarize.",
                    types=[
                        specs.InputDataType.INT,
                        specs.InputDataType.REAL,
                        specs.InputDataType.TEXT,
                    ],
                    stattypes=[
                        specs.InputDataStatType.NUMERICAL,
                        specs.InputDataStatType.NOMINAL,
                    ],
                    required=True,
                    multiple=True,
                ),
                x=specs.InputDataSpecification(
                    label="x",
                    desc="Optional additional variables to summarize.",
                    types=[
                        specs.InputDataType.INT,
                        specs.InputDataType.REAL,
                        specs.InputDataType.TEXT,
                    ],
                    stattypes=[
                        specs.InputDataStatType.NUMERICAL,
                        specs.InputDataStatType.NOMINAL,
                    ],
                    required=False,
                    multiple=True,
                ),
            ),
            type=specs.AlgorithmType.EXAREME3,
            components=[specs.ComponentType.AGGREGATION_SERVER],
        )

    @property
    def check_min_rows(self) -> bool:
        return False

    @property
    def add_dataset_variable(self) -> bool:
        return True

    def run(self):
        x_vars = self.inputdata.x or []
        y_vars = self.inputdata.y or []
        variable_names = [v for v in (x_vars + y_vars) if v != DATASET_VAR_NAME]
        numerical_vars = [
            var for var in variable_names if not self.metadata[var]["is_categorical"]
        ]
        nominal_vars = [
            var for var in variable_names if self.metadata[var]["is_categorical"]
        ]
        nominal_levels = {}
        for var in nominal_vars:
            enums = self.metadata[var].get("enumerations") or {}
            nominal_levels[var] = [code for code in enums.keys()]

        local_results = self.run_local_udf(
            func=local_step,
            kw_args={
                "variable_names": variable_names,
                "numerical_vars": numerical_vars,
                "nominal_vars": nominal_vars,
                "nominal_levels": nominal_levels,
            },
        )

        recs = [rec for worker_res in local_results for rec in worker_res["recs"]]

        recs = _append_missing_datasets(
            records=recs,
            variables=variable_names,
            datasets=self.inputdata.datasets,
        )
        recs += local_results[0]["global_recs"]

        return Result(featurewise=[Variable.from_record(rec) for rec in recs])


@exareme3_udf(with_aggregation_server=True)
def local_step(
    agg_client,
    data,
    variable_names,
    numerical_vars,
    nominal_vars,
    nominal_levels,
):
    from exaflow.worker import config as worker_config

    min_row_count = worker_config.privacy.minimum_row_count
    if "dataset" in data.columns:
        ds = data["dataset"]
        if isinstance(ds, pd.DataFrame):
            data["dataset"] = ds.iloc[:, 0]

    descriptive_stats = FederatedDescriptiveStatistics(agg_client=agg_client)
    result = descriptive_stats.describe(
        data=data,
        numerical_vars=numerical_vars,
        nominal_vars=nominal_vars,
        min_row_count=min_row_count,
        nominal_levels=nominal_levels,
        dataset_col=DATASET_VAR_NAME,
    )
    return {
        "recs": result.recs,
        "global_recs": result.global_recs,
    }


def _append_missing_datasets(*, records, variables, datasets):
    def missing(var, dataset):
        return not any(
            r for r in records if r["variable"] == var and r["dataset"] == dataset
        )

    for var in variables:
        for dataset in datasets:
            if missing(var, dataset):
                records.append({"variable": var, "dataset": dataset, "data": None})
    return records
