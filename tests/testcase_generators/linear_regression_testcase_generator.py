import re

import pandas as pd
import statsmodels.formula.api as smf

from exaflow.algorithms.exareme3.linear_regression import LinearRegressionResult
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class LinearRegressionTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, _, __):
        y, X = input_data

        [yname] = y.columns
        xnames = X.columns
        formula = f"{yname}~{'+'.join(xnames)}"
        data = pd.concat([y, X], axis=1)

        model = smf.ols(formula, data=data, missing="drop").fit()

        # NOTE: Statsmodels uses patsy to parse formula which names dummy variables
        # using a "T" to signify "Treatment" (another name for dummy coding).
        # The line below removes the "T" to match the dummy variable names from
        # Exareme2. E.g. gender[T.F] -> gender[F]
        dummy_xnames = [
            re.sub(r"(\w+)\[T\.([\w\d-]+)\]", r"\1[\2]", name)
            for name in model.model.exog_names
        ]

        result = LinearRegressionResult(
            dependent_var=model.model.endog_names,
            n_obs=model.nobs,
            df_resid=model.df_resid,
            df_model=model.df_model,
            rse=model.mse_resid**0.5,
            r_squared=model.rsquared,
            r_squared_adjusted=model.rsquared_adj,
            f_stat=model.fvalue,
            f_pvalue=model.f_pvalue,
            indep_vars=dummy_xnames,
            coefficients=model.params.tolist(),
            std_err=model.bse.tolist(),
            t_stats=model.tvalues.tolist(),
            pvalues=model.pvalues.tolist(),
            lower_ci=model.conf_int().to_numpy().T.tolist()[0],
            upper_ci=model.conf_int().to_numpy().T.tolist()[1],
        )
        return result.model_dump()


if __name__ == "__main__":
    with open("exareme3/algorithms/linear_regression.json") as specs_file:
        pcagen = LinearRegressionTestCaseGenerator(specs_file)
    with open("linear_regression_expected.json", "w") as expected_file:
        pcagen.write_test_cases(expected_file)
