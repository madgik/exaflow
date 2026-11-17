import logging

import pytest

from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadata,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadataPerWorker,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _remove_incompatible_data_models_from_data_models_metadata_per_worker,
)
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from tests.standalone_tests.controller.services.test_worker_landscape_aggregator import (
    _build_metadata,
)

LEVELS = {"1": "Low", "2": "Mid", "3": "High"}
LEVELS_REORDERED = {"3": "High", "1": "Low", "2": "Mid"}
BINARY = {"0": "No", "1": "Yes"}
BINARY_REORDERED = {"1": "Yes", "0": "No"}


def _cde(cde_type, enumerations):
    return CommonDataElement(
        code="var", label="Var", type=cde_type, enumerations=enumerations
    )


@pytest.mark.parametrize(
    "cde_type, enumerations, reordered, order_matters",
    [
        ("nominal", LEVELS, LEVELS_REORDERED, False),
        ("ordinal", LEVELS, LEVELS_REORDERED, True),
        ("binary", BINARY, BINARY_REORDERED, True),
    ],
)
class TestCategoricalLevelOrder:
    def test_same_order_is_equal(
        self, cde_type, enumerations, reordered, order_matters
    ):
        assert _cde(cde_type, enumerations) == _cde(cde_type, dict(enumerations))

    def test_different_order_equality_follows_type(
        self, cde_type, enumerations, reordered, order_matters
    ):
        assert (
            _cde(cde_type, enumerations) != _cde(cde_type, reordered)
        ) is order_matters

    def test_collection_with_different_order_follows_type(
        self, cde_type, enumerations, reordered, order_matters
    ):
        cdes1 = CommonDataElements(values={"var": _cde(cde_type, enumerations)})
        cdes2 = CommonDataElements(values={"var": _cde(cde_type, reordered)})
        assert (cdes1 != cdes2) is order_matters

    def test_different_order_marks_the_data_model_incompatible(
        self, cde_type, enumerations, reordered, order_matters, caplog
    ):
        def metadata(levels):
            built = _build_metadata(["dataset1"])
            cdes = dict(built.cdes.values)
            cdes["var"] = _cde(cde_type, levels)
            return built.model_copy(update={"cdes": CommonDataElements(values=cdes)})

        per_worker = DataModelsMetadataPerWorker(
            data_models_metadata_per_worker={
                "localworker1": DataModelsMetadata(
                    data_models_metadata={"dm:1": metadata(enumerations)}
                ),
                "localworker2": DataModelsMetadata(
                    data_models_metadata={"dm:1": metadata(reordered)}
                ),
            }
        )
        logger = logging.getLogger("cde-level-order-test")
        with caplog.at_level(logging.INFO, logger=logger.name):
            result = (
                _remove_incompatible_data_models_from_data_models_metadata_per_worker(
                    per_worker, logger
                )
            )

        if order_matters:
            assert all(
                not models.data_models_metadata
                for models in result.data_models_metadata_per_worker.values()
            )
            assert "have incompatibility on the following cdes" in caplog.text
        else:
            assert all(
                models.data_models_metadata
                for models in result.data_models_metadata_per_worker.values()
            )
            assert "have incompatibility on the following cdes" not in caplog.text


def test_non_categorical_cdes_compare_normally():
    first = CommonDataElement(code="v", label="V", type="continuous")
    assert first == CommonDataElement(code="v", label="V", type="continuous")


def test_dataset_cde_enumerations_may_still_differ():
    def dataset_cdes(enumerations):
        return CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    type="nominal",
                    enumerations=enumerations,
                )
            }
        )

    assert dataset_cdes({"1": "A", "2": "B"}) == dataset_cdes({"3": "C", "1": "A"})


def test_dataset_cde_shared_code_with_conflicting_labels_is_not_equal():
    def dataset_cdes(enumerations):
        return CommonDataElements(
            values={
                "dataset": CommonDataElement(
                    code="dataset",
                    label="Dataset",
                    type="nominal",
                    enumerations=enumerations,
                )
            }
        )

    assert dataset_cdes({"1": "A", "2": "B"}) != dataset_cdes({"1": "C", "3": "D"})
