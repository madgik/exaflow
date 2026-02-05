from pathlib import Path

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.vectors import StrVector

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("exareme3", "algorithms", "ttest_onesample.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "exareme3",
    "expected",
    "ttest_onesample_expected.json",
)
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
packnames = ("stats", "effsize")

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

stats = rpackages.importr("stats")
effsize = rpackages.importr("effsize")


class OneSampleTtestTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters="alt_hypothesis"):
        Y, _ = input_data
        if input_parameters["alt_hypothesis"] == "two-sided":
            alt_hyp = "two.sided"
        else:
            alt_hyp = input_parameters["alt_hypothesis"]
        alpha = input_parameters["alpha"]
        mu = input_parameters["mu"]
        n_obs = len(Y)
        y_name = Y.columns[0]
        t_test_res = stats.t_test(
            ro.vectors.FloatVector(Y[y_name]),
            mu=mu,
            paired=False,
            alternative=alt_hyp,
            conf_level=1 - alpha,
        )

        cohens_d_res = effsize.cohen_d(FloatVector(Y[y_name]), mu=mu)

        t_test_res_py = dict(zip(t_test_res.names, map(list, list(t_test_res))))
        expected_out = {
            "n_obs": n_obs,
            "t_value": t_test_res_py["statistic"][0],
            "p_value": t_test_res_py["p.value"][0],
            "df": t_test_res_py["parameter"][0],
            "mean_diff": t_test_res_py["estimate"][0],
            "se_diff": t_test_res_py["stderr"][0],
            "ci_upper": t_test_res_py["conf.int"][1],
            "ci_lower": t_test_res_py["conf.int"][0],
            "cohens_d": cohens_d_res[0],
        }

        return expected_out


if __name__ == "__main__":
    with open(SPECS_PATH) as specs_file:
        one_sample_gen = OneSampleTtestTestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        one_sample_gen.write_test_cases(expected_file)
