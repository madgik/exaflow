from pathlib import Path

import numpy as np
import pandas as pd
import rpy2
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

from tests.testcase_generators.testcase_generator import TestCaseGenerator

pandas2ri.activate()

base = importr("base")
stats = importr("stats")
car = importr("car")

SPECS_PATH = Path("exareme3", "algorithms", "anova_twoway.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "anova_twoway_expected.json",
)


class ANOVATwoWayTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, parameters, metadata):
        Y, X = input_data
        sstype = parameters["sstype"]
        enums = {
            md["code"]: [enum["code"] for enum in md["enumerations"]]
            for md in metadata
            if md["code"] in X.columns
        }

        y = Y.columns[0]
        x = X.columns
        if len(x) != 2:
            return

        x1 = x[0]
        x2 = x[1]
        dict_to_df = {
            y: Y[y],
            x1: pd.Categorical(X[x1], categories=enums[x1]),
            x2: pd.Categorical(X[x2], categories=enums[x2]),
        }
        data = pd.DataFrame(dict_to_df)
        n1_groups = len(set(data[x1]))
        n2_groups = len(set(data[x2]))

        if n1_groups < 2 or n2_groups < 2:
            return

        formula = f"{y} ~ {x1}*{x2}"
        try:
            model = stats.lm(formula, data=data)
        except rpy2.rinterface_lib.embedded.RRuntimeError:
            return  # unknown rpy2 exception

        if sstype == 1:
            aov = stats.anova(model)
        else:
            aov = car.Anova(model, type=2)

        ss = np.nan_to_num(aov.rx2("Sum Sq"), nan=0).tolist()
        df = aov.rx2("Df").astype(int).tolist()
        f_stat = np.nan_to_num(aov.rx2("F value"), nan=0).tolist()
        f_pvalue = np.nan_to_num(aov.rx2("Pr(>F)"), nan=0).tolist()

        termnames = list(aov.rownames)

        result = {
            "sum_sq": dict(zip(termnames, ss)),
            "df": dict(zip(termnames, df)),
            "f_stat": dict(zip(termnames, f_stat)),
            "f_pvalue": dict(zip(termnames, f_pvalue)),
        }
        return result


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        paired_gen = ANOVATwoWayTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        paired_gen.write_test_cases(expected_file, num_test_cases=50)
