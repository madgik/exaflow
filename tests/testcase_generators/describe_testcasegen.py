import pandas as pd

from exaflow.algorithms.exareme3.describe import Result
from exaflow.algorithms.exareme3.describe import Variable
from exaflow.algorithms.exareme3.describe import reduce_recs_for_var
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

        result = Result(variable_based=result_varbased, model_based=result_modbased)

        return result.model_dump()


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


if __name__ == "__main__":
    with open("exareme3/algorithms/describe.json") as specs_file:
        gen = DesciptiveStatistics(specs_file)
    with open("describe_expected.json", "w") as expected_file:
        gen.write_test_cases(expected_file, 50)
