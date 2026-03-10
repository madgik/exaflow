import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score

from exaflow.algorithms.exareme3.linear_regression_cv import CVLinearRegressionResult
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class StatsmodelsWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for statsmodels regression model with formula, exposing the same
    methods as regression in sklearn. It is used as a wrapper for statsmodels
    OLS in order to be able to use sklearn's `cross_val_score`."""

    def __init__(self, model_class, formula):
        self.model_class = model_class
        self.formula = formula

    def fit(self, X, y):
        data = pd.concat([y, X], axis=1)
        self.model_ = self.model_class(self.formula, data=data, missing="drop").fit()
        return self

    def predict(self, X):
        X = sm.add_constant(X)
        return self.model_.predict(X)


class LinearRegressionTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, params):
        y, X = input_data
        n_splits = params["n_splits"]

        if n_splits >= len(y):
            return  # Discard invalid test case

        [yname] = y.columns
        xnames = X.columns
        formula = f"{yname}~{'+'.join(xnames)}"

        model = StatsmodelsWrapper(smf.ols, formula)

        try:
            neg_rms_errors = cross_val_score(
                model,
                X,
                y,
                cv=n_splits,
                scoring="neg_root_mean_squared_error",
            )
            rms_errors = np.array([-e for e in neg_rms_errors])
            r2s = np.array(cross_val_score(model, X, y, cv=n_splits, scoring="r2"))
            neg_maes = cross_val_score(
                model, X, y, cv=n_splits, scoring="neg_mean_absolute_error"
            )
            maes = np.array([-e for e in neg_maes])
        except patsy.PatsyError:
            return  # Discard test case if patsy cannot parse formula

        result = CVLinearRegressionResult(
            dependent_var=yname,
            indep_vars=[""],
            n_obs=[0],
            mean_sq_error=(rms_errors.mean(), rms_errors.std(ddof=1)),
            r_squared=(r2s.mean(), r2s.std(ddof=1)),
            mean_abs_error=(maes.mean(), maes.std(ddof=1)),
        )

        if result_has_nan(result):
            return  # Some results have nans, not sure why but discard

        return result.model_dump()


def result_has_nan(result):
    if np.isnan(result.mean_sq_error).any():
        return True
    if np.isnan(result.mean_abs_error).any():
        return True
    if np.isnan(result.r_squared).any():
        return True
    return False


if __name__ == "__main__":
    with open("exareme3/algorithms/linear_regression_cv.json") as specs_file:
        pcagen = LinearRegressionTestCaseGenerator(specs_file)
    with open("linear_regression_cv_expected.json", "w") as expected_file:
        pcagen.write_test_cases(expected_file, num_test_cases=50)
