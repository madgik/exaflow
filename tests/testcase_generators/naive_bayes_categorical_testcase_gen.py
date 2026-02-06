import json
from collections import Counter

from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

from tests.testcase_generators.testcase_generator import TestCaseGenerator


class CategoricalNBFitTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, _, metadata):
        y, X = input_data

        xcat = [get_enums(col, metadata) for col in X.columns]
        ycat = [get_enums(col, metadata) for col in y.columns]

        X = X.values
        y = y.iloc[:, 0].values[:, None]

        X = OrdinalEncoder(categories=xcat, dtype=int).fit_transform(X)
        y = OrdinalEncoder(categories=ycat, dtype=int).fit_transform(y)

        nb = CategoricalNB(alpha=1, force_alpha=True)
        nb.fit(X, y.ravel())

        return {
            "class_count": nb.class_count_.tolist(),
            "category_count": [cc.tolist() for cc in nb.category_count_],
        }


class CategoricalNBPredictTestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, _, metadata):
        y, X = input_data

        xcat = [get_enums(col, metadata) for col in X.columns]
        ycat = [get_enums(col, metadata) for col in y.columns]

        X = X.values
        y = y.iloc[:, 0].values[:, None]

        X = OrdinalEncoder(categories=xcat, dtype=int).fit_transform(X)
        yenc = OrdinalEncoder(categories=ycat, dtype=int)
        y = yenc.fit_transform(y)

        nb = CategoricalNB(alpha=1, force_alpha=True)
        nb.fit(X, y.ravel())

        predictions = Counter(nb.predict(X))
        predictions = {
            yenc.categories_[0][key]: val for key, val in predictions.items()
        }

        return {"predictions": predictions}


def get_enums(code, metadata):
    md = next(md for md in metadata if md["code"] == code)
    return sorted([enum["code"] for enum in md["enumerations"]])


if __name__ == "__main__":
    with open(
        "exaflow/algorithms/exareme3/naive_bayes_categorical_cv.json"
    ) as specifications_file:
        specs = json.loads(specifications_file.read())
    gen = CategoricalNBFitTestCaseGenerator(specifications_file.read())
    fname = "tests/algorithm_validation_tests/expected/naive_bayes_categorical_fit_expected.json"
    with open(fname, "w") as output_file:
        gen.write_test_cases(output_file, num_test_cases=50)

    gen = CategoricalNBPredictTestCaseGenerator(specs)
    fname = "tests/algorithm_validation_tests/expected/naive_bayes_categorical_predict_expected.json"
    with open(fname, "w") as output_file:
        gen.write_test_cases(output_file, num_test_cases=50)
