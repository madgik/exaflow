from pathlib import Path

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("exareme3", "algorithms", "ttest_paired.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "exareme3",
    "expected",
    "ttest_paired_expected.json",
)

jsonlite = importr("jsonlite")
dplyr = importr("dplyr")
effsize = importr("effsize")
stats = importr("stats")


class PairedTtestTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters="alt_hypothesis"):
        Y, X = input_data
        if input_parameters["alt_hypothesis"] == "two-sided":
            alt_hyp = "two.sided"
        else:
            alt_hyp = input_parameters["alt_hypothesis"]
        alpha = input_parameters["alpha"]
        n_obs = len(Y)
        y_name = Y.columns[0]
        x_name = X.columns[0]

        t_test_res = stats.t_test(
            ro.vectors.FloatVector(X[x_name]),
            ro.vectors.FloatVector(Y[y_name]),
            paired=True,
            alternative=alt_hyp,
            conf_level=1 - alpha,
        )
        cohens_d_res = effsize.cohen_d(
            ro.vectors.FloatVector(X[x_name]),
            ro.vectors.FloatVector(Y[y_name]),
            paired=True,
        )
        t_test_res_py = dict(zip(t_test_res.names, map(list, list(t_test_res))))
        cohens_d_res_py = dict(zip(cohens_d_res.names, map(list, list(cohens_d_res))))

        expected_out = {
            "t_stat": t_test_res_py["statistic"][0],
            "p_value": t_test_res_py["p.value"][0],
            "df": t_test_res_py["parameter"][0],
            "mean_diff": t_test_res_py["estimate"][0],
            "se_diff": t_test_res_py["stderr"][0],
            "ci_upper": t_test_res_py["conf.int"][1],
            "ci_lower": t_test_res_py["conf.int"][0],
            "cohens_d": cohens_d_res_py["estimate"][0],
        }

        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        paired_gen = PairedTtestTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        paired_gen.write_test_cases(expected_file)
