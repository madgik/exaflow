import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import PerfectSeparationError
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from exaflow.algorithms.exareme3.logistic_regression import LogisticRegressionSummary
from tests.testcase_generators.testcase_generator import TestCaseGenerator

warnings.filterwarnings("error")


class LogisticRegressionTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters):
        y, X = input_data
        positive_class = input_parameters["positive_class"]
        ybin = (y == positive_class).astype(int)

        # reject if there are fewer than 2 classes
        if len(ybin.iloc[:, 0].unique()) < 2:
            return

        [yname] = y.columns
        xnames = X.columns
        formula = f"{yname}~{'+'.join(xnames)}"
        data = pd.concat([ybin, X], axis=1)
        data.dropna(inplace=True)

        # reject if class counts are less than the number of columns of X
        positive_count = sum(data[yname])
        negative_count = len(data) - positive_count
        # The actual number of columns is larger than len(X.columns) due to
        # dummy variables. It is too much hussle to compute this here, so I
        # just catch these cases durring algorithm validation testing, and
        # replace them with newly generated ones.
        n_cols = len(X.columns) + 1
        if positive_count <= n_cols or negative_count <= n_cols:
            return

        try:
            # statsmodels is very chatty so I suppress stdout. Remove next line
            # to see what is printed.
            sys.stdout = None
            model = smf.logit(formula, data).fit()
            sys.stdout = sys.__stdout__
        except np.linalg.LinAlgError:
            # statsmodels throws an error when encountering a singular matrix.
            # In our implementation we take the Penrose-Moore pseudo-inverse in
            # these cases.
            return
        except PerfectSeparationError:
            # Perfect separation happens sometimes because there are seemingly
            # different variables with the same contents in the testing
            # datasets. This can lead to infinities. There isn't much we can do
            # about it, except hoping that there are no such variables in
            # production.
            return
        except ConvergenceWarning:
            # not sure why it didn't converge but we certainly can't compare
            # results when this happens
            return
        except RuntimeWarning as exc:
            if "overflow encountered in exp" == exc.args[0]:
                return  # this is another manifestation of perfect separation
            raise

        summary = LogisticRegressionSummary(
            aic=model.aic,
            bic=model.bic,
            df_model=model.df_model,
            df_resid=model.df_resid,
            ll=model.llf,
            ll0=model.llnull,
            stderr=model.bse.tolist(),
            pvalues=model.pvalues.tolist(),
            r_squared_mcf=model.prsquared,
            r_squared_cs=0,  # statsmodels doesn't compute Cox-Snell R^2
            z_scores=model.tvalues.tolist(),
            lower_ci=model.conf_int()[0].tolist(),
            upper_ci=model.conf_int()[1].tolist(),
            n_obs=model.nobs,
            coefficients=model.params.tolist(),
        )

        return summary.model_dump()


if __name__ == "__main__":
    with open("exareme3/algorithms/logistic_regression.json") as specs_file:
        gen = LogisticRegressionTestCaseGenerator(specs_file)
    with open("logistic_regression.json", "w") as expected_file:
        gen.write_test_cases(expected_file, num_test_cases=100)
