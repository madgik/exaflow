from pathlib import Path

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
import scipy.stats as stats
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("exareme3", "algorithms", "ttest_independent.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "expected",
    "independent_ttest_expected.json",
)

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
packnames = ("stats", "jsonlite", "dplyr", "effsize")

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

stats_r = rpackages.importr("stats")
jsonlite = rpackages.importr("jsonlite")
dplyr = rpackages.importr("dplyr")
effsize = rpackages.importr("effsize")


class IndependentTtestTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters="alt_hypothesis"):
        Y, X = input_data
        if input_parameters["alt_hypothesis"] == "two-sided":
            alt_hyp = "two.sided"
        else:
            alt_hyp = input_parameters["alt_hypothesis"]
        alpha = input_parameters["alpha"]
        n_obs = len(Y) + len(X)
        y_name = Y.columns[0]
        x_name = X.columns[0]

        t_test_res = stats_r.t_test(
            ro.vectors.FloatVector(X[x_name]),
            ro.vectors.FloatVector(Y[y_name]),
            paired=False,
            alternative=alt_hyp,
            var_equal=True,
            conf_level=1 - alpha,
        )

        cohens_d_res = effsize.cohen_d(
            ro.vectors.FloatVector(X[x_name]), ro.vectors.FloatVector(Y[y_name])
        )
        t_test_res_py = dict(zip(t_test_res.names, map(list, list(t_test_res))))
        cohens_d_res_py = dict(zip(cohens_d_res.names, map(list, list(cohens_d_res))))

        expected_out = {
            "n_obs": n_obs,
            "statistic": t_test_res_py["statistic"][0],
            "p_value": t_test_res_py["p.value"][0],
            "df": t_test_res_py["parameter"][0],
            "mean_diff": t_test_res_py["estimate"][0] - t_test_res_py["estimate"][1],
            "se_difference": t_test_res_py["stderr"][0],
            "ci_upper": t_test_res_py["conf.int"][1],
            "ci_lower": t_test_res_py["conf.int"][0],
            "cohens_d": cohens_d_res_py["estimate"][0],
        }

        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        paired_gen = IndependentTtestTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        paired_gen.write_test_cases(expected_file)
