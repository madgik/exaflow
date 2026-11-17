import pandas as pd
import statsmodels.api as sm

from tests.testcase_generators.design_matrix import coded_design_matrix
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class LinearRegressionTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, _, metadata):
        y, X = input_data

        [yname] = y.columns
        data = pd.concat([y, X], axis=1).dropna()
        design = sm.add_constant(
            coded_design_matrix(data[X.columns], metadata), prepend=True
        ).rename(columns={"const": "Intercept"})
        model = sm.OLS(data[yname], design).fit()
        dummy_xnames = list(design.columns)

        conf_int = model.conf_int().to_numpy().T.tolist()
        return {
            "dependent_var": yname,
            "n_obs": int(model.nobs),
            "df_resid": model.df_resid,
            "df_model": model.df_model,
            "rse": model.mse_resid**0.5,
            "r_squared": model.rsquared,
            "r_squared_adjusted": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "indep_vars": dummy_xnames,
            "coefficients": model.params.tolist(),
            "std_err": model.bse.tolist(),
            "t_stats": model.tvalues.tolist(),
            "pvalues": model.pvalues.tolist(),
            "lower_ci": conf_int[0],
            "upper_ci": conf_int[1],
        }


if __name__ == "__main__":
    with open("exareme3/algorithms/linear_regression.json") as specs_file:
        pcagen = LinearRegressionTestCaseGenerator(specs_file)
    with open("linear_regression_expected.json", "w") as expected_file:
        pcagen.write_test_cases(expected_file)
