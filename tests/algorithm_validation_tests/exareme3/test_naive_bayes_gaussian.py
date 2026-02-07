import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

assert_allclose = partial(np.testing.assert_allclose, rtol=1e-2, atol=1e-2)
fit_exp = Path(__file__).parent / "expected" / "naive_bayes_gauss_fit_expected.json"
pred_exp = (
    Path(__file__).parent / "expected" / "naive_bayes_gauss_predict_expected.json"
)
cv_inputs = Path(__file__).parent / "expected" / "naive_bayes_gauss_cv_expected.json"


def get_test_inputs(file):
    with file.open() as f:
        params = json.load(f)["test_cases"]
    return [p["input"] for p in params]


class TestGaussianNB:
    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__class_count(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian", test_input)
        assert result["class_count"] == expected["class_count"]

    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__theta(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian", test_input)
        assert_allclose(result["theta"], expected["theta"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(fit_exp))
    def test_fit__var(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian", test_input)
        assert_allclose(result["var"], expected["var"])

    @pytest.mark.parametrize("test_input, expected", get_test_params(pred_exp))
    def test_predict(self, test_input, expected, get_algorithm_result):
        result = get_algorithm_result("test_nb_gaussian_predict", test_input)
        assert result["predictions"] == expected["predictions"]

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__confusion_matrix(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        confmat = np.array(result["confusion_matrix"]["data"])
        assert (0 <= confmat).all()

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__accuracy(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        accuracy = result["classification_summary"]["accuracy"]
        self._assert_all_between_0_and_1(accuracy)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__precision(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        precision = result["classification_summary"]["precision"]
        self._assert_all_between_0_and_1(precision)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__recall(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        recall = result["classification_summary"]["recall"]
        self._assert_all_between_0_and_1(recall)

    @pytest.mark.parametrize("test_input", get_test_inputs(cv_inputs))
    def test_cv__fscore(self, test_input, get_algorithm_result):
        result = get_algorithm_result("naive_bayes_gaussian_cv", test_input)
        fscore = result["classification_summary"]["fscore"]
        self._assert_all_between_0_and_1(fscore)

    @staticmethod
    def _assert_all_between_0_and_1(quantity):
        for key1, val1 in quantity.items():
            for key2, val2 in val1.items():
                msg = f"{key1}:{key2} is out of bounds"
                assert 0 <= val2 <= 1, msg
