from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from pydantic import BaseModel

from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.federated.descriptive_stats import (
    FederatedDescriptiveStatistics,
)

DATASET_VAR_NAME = "dataset"
ALGORITHM_NAME = "descriptive_stats"


class NumericalDescriptiveStats(BaseModel):
    num_dtps: int
    num_na: int
    num_total: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    q1: Optional[float]
    q2: Optional[float]
    q3: Optional[float]
    max: Optional[float]


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
    variable_based: List[Variable]
    model_based: List[Variable]


class DescriptiveStatisticsAlgorithm(Algorithm, algname=ALGORITHM_NAME):
    @property
    def drop_na_rows(self) -> bool:
        return False

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
                "numerical_vars": numerical_vars,
                "nominal_vars": nominal_vars,
                "nominal_levels": nominal_levels,
            },
        )

        recs_varbased = [
            rec for worker_res in local_results for rec in worker_res["recs_varbased"]
        ]
        recs_modbased = [
            rec for worker_res in local_results for rec in worker_res["recs_modbased"]
        ]

        recs_varbased = _append_missing_datasets(
            records=recs_varbased,
            variables=variable_names,
            datasets=self.inputdata.datasets,
        )
        recs_modbased = _append_missing_datasets(
            records=recs_modbased,
            variables=variable_names,
            datasets=self.inputdata.datasets,
        )
        recs_varbased += local_results[0][
            "global_varbased"
        ]  # The global stats are the same in all responses
        recs_modbased += local_results[0][
            "global_modbased"
        ]  # The global stats are the same in all responses

        return Result(
            variable_based=[Variable.from_record(rec) for rec in recs_varbased],
            model_based=[Variable.from_record(rec) for rec in recs_modbased],
        )


@exareme3_udf(with_aggregation_server=True)
def local_step(agg_client, data, numerical_vars, nominal_vars, nominal_levels):
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
        "recs_varbased": result.recs_varbased,
        "recs_modbased": result.recs_modbased,
        "global_varbased": result.global_varbased,
        "global_modbased": result.global_modbased,
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
