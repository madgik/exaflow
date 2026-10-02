import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from exaflow.algorithms.exareme3.model_selection.cross_validation.linear_regression_cv import (
    LinearRegressionCV,
)
from exaflow.algorithms.exareme3.model_selection.cross_validation.linear_regression_cv import (
    LinearRegressionCVResult,
)
from tests.testcase_generators.testcase_generator import TestCaseGenerator


class LinearRegressionTestCaseGenerator(TestCaseGenerator):
    """Centralized reference using unshuffled folds in the supplied row order.

    Distributed comparisons must use the same fold membership; worker-local
    KFold splits generally differ from KFold over concatenated rows.
    """

    def compute_expected_output(self, input_data, params, metadata):
        y, X = input_data
        n_splits = params["n_splits"]
        if n_splits > len(y):
            return None

        categorical = [
            cde["code"]
            for cde in metadata
            if cde["code"] in X.columns and cde["isCategorical"]
        ]
        numerical = [name for name in X.columns if name not in categorical]

        def design(frame, levels):
            columns = [
                (frame[name] == level).to_numpy(dtype=float)
                for name in categorical
                for level in levels[name][1:]
            ] + [frame[name].to_numpy(dtype=float) for name in numerical]
            return np.column_stack(columns) if columns else np.empty((len(frame), 0))

        metrics = {key: [] for key in ["rmse", "mae", "r2", "f_stat"]}
        n_obs = []
        y_values = y.iloc[:, 0].to_numpy(dtype=float)
        for train, test in KFold(n_splits=n_splits, shuffle=False).split(X):
            levels = {
                name: sorted(X.iloc[train][name].unique()) for name in categorical
            }
            X_train, X_test = (
                design(X.iloc[train], levels),
                design(X.iloc[test], levels),
            )
            y_train, y_test = y_values[train], y_values[test]
            p = X_train.shape[1]
            if p:
                model = LinearRegression().fit(X_train, y_train)
                predicted = model.predict(X_test)
            else:
                predicted = np.full(len(test), y_train.mean())
            rss = np.square(y_test - predicted).sum()
            tss = np.square(y_test - y_test.mean()).sum()
            f_stat = float("nan")
            if p > 0 and len(train) - p - 1 > 0 and tss > 0:
                if rss <= tss * np.finfo(float).eps * 100:
                    f_stat = float("inf")
                else:
                    f_stat = (tss - rss) * (len(train) - p - 1) / (p * rss)
            metrics["rmse"].append(np.sqrt(mean_squared_error(y_test, predicted)))
            metrics["mae"].append(mean_absolute_error(y_test, predicted))
            metrics["r2"].append(
                r2_score(y_test, predicted, force_finite=False)
                if len(test) >= 2 and tss > 0
                else float("nan")
            )
            metrics["f_stat"].append(f_stat)
            n_obs.append(len(train))

        def summary(values):
            if not np.isfinite(values).all():
                return None, None
            return float(np.mean(values)), float(np.std(values, ddof=1))

        feature_names = [
            f"{name}[{level}]"
            for name in categorical
            for level in sorted(X[name].unique())[1:]
        ] + numerical
        return LinearRegressionCVResult(
            dependent_var=y.columns[0],
            indep_vars=["Intercept"] + feature_names,
            n_obs=n_obs,
            root_mean_sq_error=summary(metrics["rmse"]),
            r_squared=summary(metrics["r2"]),
            mean_abs_error=summary(metrics["mae"]),
            f_stat=summary(metrics["f_stat"]),
        ).model_dump(mode="json")

    def generate_test_case(self):
        case = super().generate_test_case()
        generated_input = case["input"]
        inputdata = dict(generated_input["inputdata"])
        x, y = list(inputdata.pop("x")), list(inputdata.pop("y"))
        inputdata["variables"] = x + y
        case["input"] = {
            "inputdata": inputdata,
            "preprocessing": [
                {
                    "name": "missing_values_handler",
                    "parameters": {"strategies": {name: "drop" for name in x + y}},
                }
            ],
            "algorithm": {
                "name": "linear_regression_cv",
                "x": x,
                "y": y,
                "parameters": generated_input["parameters"],
            },
        }
        return case


if __name__ == "__main__":
    specification = LinearRegressionCV.get_specification()
    # The shared random-input generator takes grouped variable specifications.
    generator_spec = {
        "inputdata": {
            name: {
                **getattr(specification, name).model_dump(
                    mode="json", exclude_none=True
                ),
                "multiple": getattr(specification, name).max_count != 1,
            }
            for name in ["x", "y"]
        },
        "parameters": {
            name: parameter.model_dump(mode="json", exclude_none=True)
            for name, parameter in specification.parameters.items()
        },
    }
    generator = LinearRegressionTestCaseGenerator(generator_spec)
    with open("linear_regression_cv_expected.json", "w") as expected_file:
        generator.write_test_cases(expected_file, num_test_cases=50)
