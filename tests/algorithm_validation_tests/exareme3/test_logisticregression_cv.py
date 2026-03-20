from pathlib import Path

import numpy as np
import pytest

from tests.algorithm_validation_tests.exareme3.helpers import get_test_params

expected_file = (
    Path(__file__).parent / "expected" / "logistic_regression_cv_expected.json"
)
ALGNAME = "logistic_regression_cv"


class TestLogisticRegressionCV:
    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_accuracy(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        accuracy = np.array(result["summary"]["accuracy"])
        assert (0 <= accuracy).all() and (accuracy <= 1).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_precision(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        precision = np.array(result["summary"]["precision"])
        assert (0 <= precision).all() and (precision <= 1).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_recall(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        recall = np.array(result["summary"]["recall"])
        assert (0 <= recall).all() and (recall <= 1).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_fscore(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        fscore = np.array(result["summary"]["fscore"])
        assert (0 <= fscore).all() and (fscore <= 1).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_confusion_matrix(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        confmat = np.array(list(result["confusion_matrix"].values()))
        assert (confmat >= 0).all()

    @pytest.mark.parametrize("test_input, _", get_test_params(expected_file))
    def test_roc_curves(self, test_input, _, get_algorithm_result):
        result = get_algorithm_result(ALGNAME, test_input)
        for roc in result["roc_curves"]:
            tpr = np.array(roc["tpr"])
            fpr = np.array(roc["fpr"])
            auc = roc["auc"]

            assert (0 <= tpr).all() and (tpr <= 1).all()
            assert (0 <= fpr).all() and (fpr <= 1).all()
            assert 0 <= auc <= 1
