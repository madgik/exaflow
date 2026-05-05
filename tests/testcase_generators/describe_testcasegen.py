import pandas as pd

from exaflow.algorithms.exareme3.statistics.describe import Variable
from tests.testcase_generators.testcase_generator import TestCaseGenerator

# TODO privacy threshold is hardcoded. Find beter solution.
# https://team-1617704806227.atlassian.net/browse/MIP-689
MIN_ROW_COUNT = 10


class DesciptiveStatistics(TestCaseGenerator):
    full_data = True
    dropna = False

    def compute_expected_output(self, data: pd.DataFrame, _, metadata: dict):
        numerical_vars = [md["code"] for md in metadata if not md["isCategorical"]]
        nominal_vars = [
            md["code"]
            for md in metadata
            if md["isCategorical"] and md["code"] != "dataset"
        ]
        vars = numerical_vars + nominal_vars
        enums = {
            var: next(md["enumerations"] for md in metadata if md["code"] == var)
            for var in nominal_vars
        }
        # datasets = data.dataset.unique()
        # XXX  MIN_ROW_COUNT is hardcoded in descriptive stats algorithm because
        # a  dynamic  solution  is  currently  lacking. Thus, we reject all test
        # cases with less rows.
        if len(data) < MIN_ROW_COUNT:
            return

        datasets = data.dataset.unique()

        # variable based
        recs_varbased = []
        for dataset, group in group_by_dataset(data, datasets):
            recs_varbased += get_numerical_records(group, numerical_vars, dataset)
            recs_varbased += get_nominal_records(group, nominal_vars, dataset, enums)
        recs_varbased += [reduce_recs_for_var(recs_varbased, var) for var in vars]

        result_varbased = [Variable.from_record(rec) for rec in recs_varbased]

        # model based
        data = data.dropna()
        recs_modbased = []
        for dataset, group in group_by_dataset(data, datasets):
            recs_modbased += get_numerical_records(group, numerical_vars, dataset)
            recs_modbased += get_nominal_records(group, nominal_vars, dataset, enums)
        recs_modbased += [reduce_recs_for_var(recs_modbased, var) for var in vars]

        result_modbased = [Variable.from_record(rec) for rec in recs_modbased]

        return {
            "featurewise": [rec.model_dump() for rec in result_varbased],
            "analysis_set": [rec.model_dump() for rec in result_modbased],
        }


def get_numerical_records(data, numerical_vars, dataset):
    num_total = len(data)
    description = data.describe(include="all")
    num_dtps = description.loc["count"]
    mean = description.loc["mean"]
    std = description.loc["std"]
    min = description.loc["min"]
    max = description.loc["max"]
    q1 = description.loc["25%"]
    q2 = description.loc["50%"]
    q3 = description.loc["75%"]
    sx = data[numerical_vars].sum().to_dict()
    sxx = (data[numerical_vars] ** 2).sum().to_dict()
    return [
        dict(
            variable=var,
            dataset=dataset,
            data=(
                dict(
                    num_dtps=num_dtps[var],
                    num_total=num_total,
                    num_na=num_total - num_dtps[var],
                    mean=mean[var],
                    std=std[var],
                    min=min[var],
                    max=max[var],
                    q1=q1[var] if dataset != "all datasets" else None,
                    q2=q2[var] if dataset != "all datasets" else None,
                    q3=q3[var] if dataset != "all datasets" else None,
                    sx=sx[var],
                    sxx=sxx[var],
                )
                if num_dtps[var] >= MIN_ROW_COUNT
                else None
            ),
        )
        for var in numerical_vars
    ]


def get_nominal_records(data, nominal_vars, dataset, enums):
    num_total = len(data)
    description = data.describe(include="all")
    num_dtps = description.loc["count"]
    return [
        dict(
            variable=var,
            dataset=dataset,
            data=(
                dict(
                    num_dtps=num_dtps[var],
                    num_total=num_total,
                    num_na=num_total - num_dtps[var],
                    counts=data[var].value_counts().to_dict(),
                )
                if num_dtps[var] >= MIN_ROW_COUNT
                else None
            ),
        )
        for var in nominal_vars
    ]


def group_by_dataset(data, datasets):
    for dataset in datasets:
        yield dataset, data[data.dataset == dataset]


def reduce_recs_for_var(recs, var):
    var_recs = [rec for rec in recs if rec["variable"] == var and rec["data"]]
    if not var_recs:
        return {"variable": var, "dataset": "all datasets", "data": None}

    first_data = var_recs[0]["data"]
    if "counts" in first_data:
        counts = {}
        for rec in var_recs:
            for level, value in rec["data"]["counts"].items():
                counts[level] = counts.get(level, 0) + int(value)
        return {
            "variable": var,
            "dataset": "all datasets",
            "data": {
                "num_dtps": sum(int(rec["data"]["num_dtps"]) for rec in var_recs),
                "num_total": sum(int(rec["data"]["num_total"]) for rec in var_recs),
                "num_na": sum(int(rec["data"]["num_na"]) for rec in var_recs),
                "counts": counts,
            },
        }

    sx = sum(float(rec["data"]["sx"]) for rec in var_recs)
    sxx = sum(float(rec["data"]["sxx"]) for rec in var_recs)
    num_dtps = sum(int(rec["data"]["num_dtps"]) for rec in var_recs)
    num_total = sum(int(rec["data"]["num_total"]) for rec in var_recs)
    num_na = sum(int(rec["data"]["num_na"]) for rec in var_recs)
    if num_dtps == 0:
        return {"variable": var, "dataset": "all datasets", "data": None}

    mean = sx / num_dtps
    std = None
    if num_dtps > 1:
        variance = (sxx - num_dtps * mean**2) / (num_dtps - 1)
        std = variance**0.5 if variance >= 0 else 0.0
    return {
        "variable": var,
        "dataset": "all datasets",
        "data": {
            "num_dtps": num_dtps,
            "num_total": num_total,
            "num_na": num_na,
            "mean": mean,
            "std": std,
            "min": min(float(rec["data"]["min"]) for rec in var_recs),
            "q1": None,
            "q2": None,
            "q3": None,
            "max": max(float(rec["data"]["max"]) for rec in var_recs),
            "sx": sx,
            "sxx": sxx,
        },
    }


if __name__ == "__main__":
    with open("exareme3/algorithms/describe.json") as specs_file:
        gen = DesciptiveStatistics(specs_file)
    with open("describe_expected.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 50)
