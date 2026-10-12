import ipaddress
import logging
from typing import Dict
from typing import List

import pytest

from exaflow import AttrDict
from exaflow.controller import DeploymentType
from exaflow.controller.services.worker_landscape_aggregator import (
    worker_landscape_aggregator as wla_module,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelMetadata,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelRegistry,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsAttributes,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsCDES,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadata,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DataModelsMetadataPerWorker,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    DatasetsLocations,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerLandscapeAggregator,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    WorkerRegistry,
)
from exaflow.controller.services.worker_landscape_aggregator.worker_landscape_aggregator import (
    _crunch_data_model_registry_data,
)
from exaflow.controller.worker_client.app import WorkerClientConnectionError
from exaflow.controller.worker_client.app import WorkerClientTimeoutException
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetInfo
from exaflow.worker_communication import WorkerInfo
from exaflow.worker_communication import WorkerRole

_AUTO = object()


def _build_dataset_cde(
    enumerations: Dict[str, str],
    min_value: float | None = None,
    max_value: float | None = None,
):
    return CommonDataElement(
        code="dataset",
        label="Dataset",
        sql_type="text",
        is_categorical=True,
        enumerations=enumerations,
        min=min_value,
        max=max_value,
    )


def _build_metadata(
    dataset_codes: List[str],
    *,
    enumerations=_AUTO,
    tags: List[str] | None = None,
    properties: Dict[str, str] | None = None,
    extra_cdes: Dict[str, CommonDataElement] | None = None,
    attributes=_AUTO,
):
    dataset_infos = [
        DatasetInfo(code=code, label=code.upper()) for code in dataset_codes
    ]
    if enumerations is _AUTO:
        enumerations = {code: code.upper() for code in dataset_codes}

    if enumerations is None:
        cdes = None
    else:
        values = {"dataset": _build_dataset_cde(enumerations)}
        if extra_cdes:
            values.update(extra_cdes)
        cdes = CommonDataElements(values=values)

    if attributes is _AUTO:
        attributes = DataModelAttributes(
            tags=tags or [],
            properties=properties or {},
        )

    return DataModelMetadata(
        dataset_infos=dataset_infos,
        cdes=cdes,
        attributes=attributes,
    )


@pytest.fixture
def worker_landscape_aggregator():
    return WorkerLandscapeAggregator(
        logger=logging.getLogger("worker-landscape-tests"),
        update_interval=0,
        tasks_timeout=0,
        deployment_type=DeploymentType.LOCAL,
        localworkers=AttrDict({}),
    )


@pytest.fixture
def mocked_datasets_locations():
    return DatasetsLocations(
        datasets_locations={
            "tbi:0.1": {
                "dummy_tbi0": "localworker1",
                "dummy_tbi1": "localworker2",
                "dummy_tbi3": "localworker2",
            },
            "dementia:0.1": {
                "ppmi0": "localworker1",
                "ppmi1": "localworker2",
                "ppmi3": "localworker2",
                "edsd0": "localworker1",
                "edsd1": "localworker2",
                "edsd3": "localworker2",
                "desd-synthdata0": "localworker1",
                "desd-synthdata1": "localworker2",
                "desd-synthdata3": "localworker2",
            },
        }
    )


@pytest.fixture
def mocked_data_models_cdes():
    dementia_enums = {
        "ppmi": "PPMI",
        "edsd": "EDSD",
        "desd-synthdata": "DESD-synthdata",
    }
    tbi_enums = {"dummy_tbi": "Dummy TBI"}
    return DataModelsCDES(
        data_models_cdes={
            "dementia:0.1": CommonDataElements(
                values={
                    "dataset": CommonDataElement(
                        code="dataset",
                        label="Dataset",
                        sql_type="text",
                        is_categorical=True,
                        enumerations=dementia_enums,
                    ),
                    "alzheimerbroadcategory": CommonDataElement(
                        code="alzheimerbroadcategory",
                        label="Alzheimer Broad Category",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "AD": "Alzheimer’s disease",
                            "CN": "Cognitively Normal",
                            "Other": "Other",
                            "MCI": "Mild cognitive impairment",
                        },
                    ),
                    "minimentalstate": CommonDataElement(
                        code="minimentalstate",
                        label="MMSE Total scores",
                        sql_type="int",
                        is_categorical=False,
                        min=0,
                        max=30,
                    ),
                }
            ),
            "tbi:0.1": CommonDataElements(
                values={
                    "dataset": CommonDataElement(
                        code="dataset",
                        label="Dataset",
                        sql_type="text",
                        is_categorical=True,
                        enumerations=tbi_enums,
                    ),
                    "gender_type": CommonDataElement(
                        code="gender_type",
                        label="Gender",
                        sql_type="text",
                        is_categorical=True,
                        enumerations={
                            "M": "Male",
                            "F": "Female",
                        },
                    ),
                }
            ),
        }
    )


@pytest.fixture
def mocked_data_models_attributes():
    return DataModelsAttributes(
        data_models_attributes={
            "dementia:0.1": DataModelAttributes(
                tags=["dementia"],
                properties={"key": ["value"]},
            ),
            "tbi:0.1": DataModelAttributes(
                tags=["tbi", "longitudinal"],
                properties={"key": ["value"]},
            ),
        }
    )


@pytest.fixture
def mocked_data_model_registry(
    mocked_data_models_cdes,
    mocked_datasets_locations,
    mocked_data_models_attributes,
):
    return DataModelRegistry(
        data_models_cdes=mocked_data_models_cdes,
        datasets_locations=mocked_datasets_locations,
        data_models_attributes=mocked_data_models_attributes,
    )


@pytest.fixture
def mocked_worker_infos():
    return [
        WorkerInfo(
            id="globalworker",
            role=WorkerRole.GLOBALWORKER,
            ip=ipaddress.ip_address("127.0.0.1"),
            port=50000,
        ),
        WorkerInfo(
            id="localworker1",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.2"),
            port=50001,
        ),
        WorkerInfo(
            id="localworker2",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.3"),
            port=50002,
        ),
        WorkerInfo(
            id="localworker3",
            role=WorkerRole.LOCALWORKER,
            ip=ipaddress.ip_address("127.0.0.4"),
            port=50003,
        ),
    ]


@pytest.fixture
def mocked_wla(
    worker_landscape_aggregator,
    mocked_worker_infos,
    mocked_data_model_registry,
):
    worker_registry = WorkerRegistry(workers_info=mocked_worker_infos)
    worker_landscape_aggregator._set_new_registries(
        worker_registry=worker_registry,
        data_model_registry=mocked_data_model_registry,
    )
    return worker_landscape_aggregator


def test_get_all_available_datasets_per_data_model(mocked_data_model_registry):
    assert mocked_data_model_registry.get_all_available_datasets_per_data_model() == {
        "tbi:0.1": ["dummy_tbi0", "dummy_tbi1", "dummy_tbi3"],
        "dementia:0.1": [
            "ppmi0",
            "ppmi1",
            "ppmi3",
            "edsd0",
            "edsd1",
            "edsd3",
            "desd-synthdata0",
            "desd-synthdata1",
            "desd-synthdata3",
        ],
    }


def test_data_model_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.data_model_exists("tbi:0.1")
    assert not mocked_data_model_registry.data_model_exists("non-existing")


def test_dataset_exists(mocked_data_model_registry):
    assert mocked_data_model_registry.dataset_exists("tbi:0.1", "dummy_tbi0")
    assert not mocked_data_model_registry.dataset_exists("tbi:0.1", "missing")


def test_get_workers_with_any_of_datasets(mocked_data_model_registry):
    workers = mocked_data_model_registry.get_worker_ids_with_any_of_datasets(
        "tbi:0.1", ["dummy_tbi0", "dummy_tbi1"]
    )
    assert set(workers) == {"localworker1", "localworker2"}


def test_get_worker_specific_datasets(mocked_data_model_registry):
    datasets = mocked_data_model_registry.get_worker_specific_datasets(
        "localworker1", "dementia:0.1", ["edsd0", "ppmi0", "edsd1"]
    )
    assert set(datasets) == {"edsd0", "ppmi0"}


def test_is_longitudinal(mocked_data_model_registry):
    assert mocked_data_model_registry.is_longitudinal("tbi:0.1")
    assert not mocked_data_model_registry.is_longitudinal("dementia:0.1")


def test_get_data_models_attributes(mocked_data_model_registry):
    attributes = mocked_data_model_registry.get_data_models_attributes()
    assert attributes["tbi:0.1"].tags == ["tbi", "longitudinal"]


def test_data_model_registry_empty_initialization():
    data_model_registry = DataModelRegistry()
    assert not data_model_registry.data_models_cdes.data_models_cdes
    assert not data_model_registry.datasets_locations.datasets_locations


def get_parametrization_cases():
    common_case_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset1", "dataset2"],
                        tags=["tag1"],
                        properties={"key1": "value1"},
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset1"],
                        tags=["tag1"],
                        properties={"key1": "value1"},
                    ),
                }
            ),
            "localworker2": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset3", "dataset4"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset5", "dataset6"],
                        tags=["tag3"],
                        properties={"key3": "value3"},
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset2"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
        }
    )
    common_case_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:1": DataModelAttributes(
                    tags=["tag1", "tag2", "tag3"],
                    properties={
                        "key1": ["value1"],
                        "key2": ["value2"],
                        "key3": ["value3"],
                    },
                ),
                "data_model:2": DataModelAttributes(
                    tags=["tag1", "tag2"],
                    properties={
                        "key1": ["value1"],
                        "key2": ["value2"],
                    },
                ),
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:1": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {
                                "dataset1": "DATASET1",
                                "dataset2": "DATASET2",
                                "dataset3": "DATASET3",
                                "dataset4": "DATASET4",
                                "dataset5": "DATASET5",
                                "dataset6": "DATASET6",
                            }
                        )
                    }
                ),
                "data_model:2": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {"dataset1": "DATASET1", "dataset2": "DATASET2"}
                        )
                    }
                ),
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:1": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker1",
                    "dataset3": "localworker2",
                    "dataset4": "localworker2",
                    "dataset5": "localworker3",
                    "dataset6": "localworker3",
                },
                "data_model:2": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker3",
                },
            }
        ),
    )

    none_cdes_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset1", "dataset2"],
                        enumerations=None,
                        tags=["tag1"],
                        properties={"key1": "value1"},
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset1"],
                        tags=["tag1"],
                        properties={"key1": "value1"},
                    ),
                }
            ),
            "localworker2": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset3", "dataset4"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset5", "dataset6"],
                        tags=["tag3"],
                        properties={"key3": "value3"},
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset2"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
        }
    )
    none_cdes_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:1": DataModelAttributes(
                    tags=["tag2", "tag3"],
                    properties={
                        "key2": ["value2"],
                        "key3": ["value3"],
                    },
                ),
                "data_model:2": DataModelAttributes(
                    tags=["tag1", "tag2"],
                    properties={
                        "key1": ["value1"],
                        "key2": ["value2"],
                    },
                ),
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:1": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {
                                "dataset3": "DATASET3",
                                "dataset4": "DATASET4",
                                "dataset5": "DATASET5",
                                "dataset6": "DATASET6",
                            }
                        )
                    }
                ),
                "data_model:2": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {"dataset1": "DATASET1", "dataset2": "DATASET2"}
                        )
                    }
                ),
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:1": {
                    "dataset3": "localworker2",
                    "dataset4": "localworker2",
                    "dataset5": "localworker3",
                    "dataset6": "localworker3",
                },
                "data_model:2": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker3",
                },
            }
        ),
    )

    incompatible_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset1", "dataset2"],
                        extra_cdes={
                            "age": CommonDataElement(
                                code="age",
                                label="Age",
                                sql_type="int",
                                is_categorical=False,
                                min=0,
                                max=130,
                            )
                        },
                    ),
                    "data_model:2": _build_metadata(["dataset1"]),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset5", "dataset6"],
                        extra_cdes={
                            "age": CommonDataElement(
                                code="age",
                                label="Age",
                                sql_type="int",
                                is_categorical=False,
                                min=1,
                                max=130,
                            )
                        },
                    ),
                    "data_model:2": _build_metadata(["dataset2"]),
                }
            ),
        }
    )
    incompatible_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:2": DataModelAttributes(
                    tags=[],
                    properties={},
                )
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:2": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {"dataset1": "DATASET1", "dataset2": "DATASET2"}
                        )
                    }
                )
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:2": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker3",
                }
            }
        ),
    )

    no_data_model_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(data_models_metadata={}),
            "localworker2": DataModelsMetadata(data_models_metadata={}),
        }
    )
    no_data_model_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(data_models_attributes={}),
        data_models_cdes=DataModelsCDES(data_models_cdes={}),
        datasets_locations=DatasetsLocations(datasets_locations={}),
    )

    duplicated_dataset_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:2": _build_metadata(["dataset1"]),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:2": _build_metadata(["dataset1", "dataset2"]),
                }
            ),
        }
    )
    duplicated_dataset_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:2": DataModelAttributes(tags=[], properties={})
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:2": CommonDataElements(
                    values={"dataset": _build_dataset_cde({"dataset2": "DATASET2"})}
                )
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:2": {
                    "dataset2": "localworker3",
                }
            }
        ),
    )

    duplicated_tags_metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset1", "dataset2"],
                        tags=["tag1"],
                        properties={
                            "key1": "value1",
                            "common-key": "different-value1",
                        },
                    ),
                }
            ),
            "localworker2": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset3", "dataset4"],
                        tags=["tag2"],
                        properties={
                            "key2": "value2",
                            "common-key": "different-value2",
                        },
                    ),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset5", "dataset6"],
                        tags=["tag3"],
                        properties={"key3": "value3"},
                    ),
                }
            ),
        }
    )
    duplicated_tags_expected = DataModelRegistry(
        data_models_attributes=DataModelsAttributes(
            data_models_attributes={
                "data_model:1": DataModelAttributes(
                    tags=["tag1", "tag2", "tag3"],
                    properties={
                        "key1": ["value1"],
                        "key2": ["value2"],
                        "key3": ["value3"],
                        "common-key": [
                            "different-value1",
                            "different-value2",
                        ],
                    },
                ),
            }
        ),
        data_models_cdes=DataModelsCDES(
            data_models_cdes={
                "data_model:1": CommonDataElements(
                    values={
                        "dataset": _build_dataset_cde(
                            {
                                "dataset1": "DATASET1",
                                "dataset2": "DATASET2",
                                "dataset3": "DATASET3",
                                "dataset4": "DATASET4",
                                "dataset5": "DATASET5",
                                "dataset6": "DATASET6",
                            }
                        )
                    }
                )
            }
        ),
        datasets_locations=DatasetsLocations(
            datasets_locations={
                "data_model:1": {
                    "dataset1": "localworker1",
                    "dataset2": "localworker1",
                    "dataset3": "localworker2",
                    "dataset4": "localworker2",
                    "dataset5": "localworker3",
                    "dataset6": "localworker3",
                }
            }
        ),
    )

    return [
        pytest.param(common_case_metadata, common_case_expected, id="common_case"),
        pytest.param(
            none_cdes_metadata,
            none_cdes_expected,
            id="none_cdes_on_data_model_1",
        ),
        pytest.param(
            incompatible_metadata,
            incompatible_expected,
            id="incompatible_cdes_on_data_model1",
        ),
        pytest.param(
            no_data_model_metadata,
            no_data_model_expected,
            id="no_data_model_or_dataset_case",
        ),
        pytest.param(
            duplicated_dataset_metadata,
            duplicated_dataset_expected,
            id="duplicated_dataset1_on_data_model2",
        ),
        pytest.param(
            duplicated_tags_metadata,
            duplicated_tags_expected,
            id="properties_with_common_keys",
        ),
    ]


@pytest.mark.parametrize(
    "data_models_metadata_per_worker,expected_registry",
    get_parametrization_cases(),
)
def test_data_model_registry(
    worker_landscape_aggregator,
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
    expected_registry: DataModelRegistry,
):
    dmr = _crunch_data_model_registry_data(
        data_models_metadata_per_worker,
        worker_landscape_aggregator._logger,
    )
    assert (
        dmr.data_models_cdes.data_models_cdes
        == expected_registry.data_models_cdes.data_models_cdes
    )
    assert (
        dmr.datasets_locations.datasets_locations
        == expected_registry.datasets_locations.datasets_locations
    )
    assert (
        dmr.data_models_attributes.data_models_attributes
        == expected_registry.data_models_attributes.data_models_attributes
    )


def test_data_model_registry_missing_data_model_attributes(
    worker_landscape_aggregator,
):
    metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset1", "dataset2"],
                        attributes=None,
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset1"],
                        tags=["tag1"],
                        properties={"key1": "value1"},
                    ),
                }
            ),
            "localworker2": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset3", "dataset4"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
            "localworker3": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": _build_metadata(
                        ["dataset5", "dataset6"],
                        tags=["tag3"],
                        properties={"key3": "value3"},
                    ),
                    "data_model:2": _build_metadata(
                        ["dataset2"],
                        tags=["tag2"],
                        properties={"key2": "value2"},
                    ),
                }
            ),
        }
    )
    dmr = _crunch_data_model_registry_data(
        metadata, worker_landscape_aggregator._logger
    )
    assert "data_model:1" in dmr.data_models_attributes.data_models_attributes
    assert dmr.data_models_attributes.data_models_attributes["data_model:1"].tags == [
        "tag2",
        "tag3",
    ]


def test_crunch_data_model_registry_data_includes_datasets_variables(
    worker_landscape_aggregator,
):
    metadata = DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            "localworker1": DataModelsMetadata(
                data_models_metadata={
                    "data_model:1": DataModelMetadata(
                        dataset_infos=[
                            DatasetInfo(
                                code="dataset1",
                                label="DATASET1",
                                variables=["var1", "var2"],
                            )
                        ],
                        cdes=CommonDataElements(
                            values={
                                "dataset": _build_dataset_cde({"dataset1": "DATASET1"}),
                                "var1": CommonDataElement(
                                    code="var1",
                                    label="Variable 1",
                                    sql_type="text",
                                    is_categorical=False,
                                ),
                                "var2": CommonDataElement(
                                    code="var2",
                                    label="Variable 2",
                                    sql_type="text",
                                    is_categorical=False,
                                ),
                            }
                        ),
                        attributes=DataModelAttributes(tags=[], properties={}),
                    )
                }
            )
        }
    )
    dmr = _crunch_data_model_registry_data(
        metadata, worker_landscape_aggregator._logger
    )
    assert dmr.datasets_variables.datasets_variables == {
        "data_model:1": {"dataset1": ["var1", "var2"]}
    }


def test_get_datasets_per_worker(mocked_wla):
    datasets = mocked_wla.get_datasets_per_worker("dementia:0.1")
    assert datasets["localworker1"] == ["ppmi0", "edsd0", "desd-synthdata0"]


def test_get_training_and_validation_datasets(mocked_wla):
    training, validation = mocked_wla.get_training_and_validation_datasets("tbi:0.1")
    assert set(training) == {"dummy_tbi0", "dummy_tbi1", "dummy_tbi3"}
    assert not validation


def test_get_training_and_validation_datasets_without_global_worker(
    worker_landscape_aggregator, mocked_worker_infos, mocked_data_model_registry
):
    local_workers = [
        info for info in mocked_worker_infos if info.role == WorkerRole.LOCALWORKER
    ]
    worker_landscape_aggregator._set_new_registries(
        worker_registry=WorkerRegistry(workers_info=local_workers),
        data_model_registry=mocked_data_model_registry,
    )

    training, validation = (
        worker_landscape_aggregator.get_training_and_validation_datasets("tbi:0.1")
    )

    assert set(training) == {"dummy_tbi0", "dummy_tbi1", "dummy_tbi3"}
    assert validation == []


def test_get_training_and_validation_datasets_unknown_model(mocked_wla):
    with pytest.raises(BadUserInput):
        mocked_wla.get_training_and_validation_datasets("missing")


def test_get_workers_info_properly_handles_errors(
    worker_landscape_aggregator, monkeypatch
):
    class FakeHandler:
        def __init__(self, worker_queue_addr, *args, **kwargs):
            self.worker_queue_addr = worker_queue_addr

        def get_worker_info_task(self):
            if "fail" in self.worker_queue_addr:
                raise WorkerClientConnectionError("boom")
            return WorkerInfo(
                id="worker-ok",
                role=WorkerRole.LOCALWORKER,
                ip=ipaddress.ip_address("127.0.0.10"),
                port=50010,
            )

    monkeypatch.setattr(wla_module, "WorkerInfoTasksHandler", FakeHandler)
    workers = worker_landscape_aggregator._get_workers_info(["fail", "ok"])
    assert len(workers) == 1
    assert workers[0].id == "worker-ok"


def test_get_worker_datasets_per_data_model_properly_handles_errors(
    worker_landscape_aggregator, monkeypatch
):
    class FakeHandler:
        def __init__(self, *args, **kwargs):
            pass

        def get_worker_datasets_per_data_model_task(self):
            raise WorkerClientTimeoutException("timeout")

    monkeypatch.setattr(wla_module, "WorkerInfoTasksHandler", FakeHandler)
    datasets = worker_landscape_aggregator._get_worker_datasets_per_data_model("fail")
    assert datasets == {}


def test_get_worker_cdes_properly_handles_errors(
    worker_landscape_aggregator, monkeypatch
):
    class FakeHandler:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_model_cdes_task(self, *args, **kwargs):
            raise WorkerClientConnectionError("boom")

    monkeypatch.setattr(wla_module, "WorkerInfoTasksHandler", FakeHandler)
    cdes = worker_landscape_aggregator._get_worker_cdes("fail", "dementia:0.1")
    assert cdes is None
