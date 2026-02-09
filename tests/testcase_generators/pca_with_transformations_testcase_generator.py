from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from tests.testcase_generators.testcase_generator import TestCaseGenerator

SPECS_PATH = Path("exaflow", "algorithms", "exareme3", "pca_with_transformations.json")
EXPECTED_PATH = Path(
    "tests",
    "algorithm_validation_tests",
    "exareme3",
    "expected",
    "pca_with_transformation_expected.json",
)


class PCATestCaseGenerator(TestCaseGenerator):
    def compute_expected_output(self, input_data, input_parameters=None, metadata=None):
        X, _ = input_data

        if "data_transformation" in input_parameters:
            for transformation, variables in input_parameters[
                "data_transformation"
            ].items():
                for variable in variables:
                    try:
                        if transformation == "log":
                            if (X[variable] <= 0).any():
                                raise ValueError(
                                    f"Log transformation cannot be applied to non-positive values in column '{variable}'."
                                )
                            X[variable] = np.log(X[variable])
                        elif transformation == "exp":
                            X[variable] = np.exp(X[variable])
                        elif transformation == "center":
                            mean = np.mean(X[variable])
                            X[variable] = X[variable] - mean
                        elif transformation == "standardize":
                            mean = np.mean(X[variable])
                            std = np.std(X[variable])
                            if std == 0:
                                raise ValueError(
                                    f"Standardization cannot be applied to column '{variable}' because the standard deviation is zero."
                                )
                            X[variable] = (X[variable] - mean) / std
                        else:
                            raise ValueError(
                                f"Unknown transformation: {transformation}"
                            )

                    except Exception as e:
                        return {"errors": [str(e)]}

        X -= X.mean(axis=0)
        X /= X.std(axis=0, ddof=1)
        pca = PCA()
        pca.fit(X)

        output = {
            "n_obs": len(X),
            "eigen_vals": pca.explained_variance_.tolist(),
            "eigen_vecs": pca.components_.tolist(),
        }
        return output


if __name__ == "__main__":
    with open(
        "exaflow/algorithms/exareme3/pca_with_transformations.json"
    ) as specs_file:
        pcagen = PCATestCaseGenerator(specs_file)
    with open(EXPECTED_PATH, "w") as expected_file:
        pcagen.write_test_cases(expected_file)
