import threading
import time
import traceback
from abc import ABC
from collections import OrderedDict
from collections import defaultdict
from logging import Logger
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel

from exaflow.controller import DeploymentType
from exaflow.controller.federation_info_logs import log_datamodel_added
from exaflow.controller.federation_info_logs import log_datamodel_removed
from exaflow.controller.federation_info_logs import log_dataset_added
from exaflow.controller.federation_info_logs import log_dataset_removed
from exaflow.controller.federation_info_logs import log_worker_joined_federation
from exaflow.controller.federation_info_logs import log_worker_left_federation
from exaflow.controller.services.worker_landscape_aggregator.worker_info_tasks_handler import (
    WorkerInfoTasksHandler,
)
from exaflow.controller.worker_client.app import WorkerClientConnectionError
from exaflow.controller.worker_client.app import WorkerClientTimeoutException
from exaflow.controller.workers_addresses import WorkersAddressesFactory
from exaflow.utils import AttrDict
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import CommonDataElement
from exaflow.worker_communication import CommonDataElements
from exaflow.worker_communication import DataModelAttributes
from exaflow.worker_communication import DatasetInfo
from exaflow.worker_communication import DatasetsInfoPerDataModel
from exaflow.worker_communication import WorkerInfo
from exaflow.worker_communication import WorkerRole

WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID = "WORKER_LANDSCAPE_AGGREGATOR"
LONGITUDINAL = "longitudinal"


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


def _get_worker_socket_addr(worker_info: WorkerInfo):
    return f"{worker_info.ip}:{worker_info.port}"


def _have_common_elements(a: List[Any], b: List[Any]):
    return bool(set(a) & set(b))


class DataModelsCDES(ImmutableBaseModel):
    """
    A dictionary representation of the cdes of each data model.
    Key values are data models.
    Values are CommonDataElements.
    """

    data_models_cdes: Optional[Dict[str, CommonDataElements]] = {}


class DataModelsAttributes(ImmutableBaseModel):
    """
    A dictionary representation of the attributes of each data model.
    Key values are data models.
    Values are DataModelAttributes.
    """

    data_models_attributes: Optional[Dict[str, DataModelAttributes]] = {}


class DatasetsLocations(ImmutableBaseModel):
    """
    A dictionary representation of the locations of each dataset in the federation.
    Key values are data models because a dataset may be available in multiple data_models.
    Values are Dictionaries of datasets and their locations.
    """

    datasets_locations: Optional[Dict[str, Dict[str, str]]] = {}


class DatasetsVariables(ImmutableBaseModel):
    """
    A dictionary representation of the available variables per dataset in the federation.
    Key values are data models because a dataset may be available in multiple data_models.
    Values are dictionaries where the keys are dataset codes and the values are lists of variables.
    """

    datasets_variables: Optional[Dict[str, Dict[str, List[str]]]] = {}


class DataModelRegistry(ImmutableBaseModel):
    data_models_attributes: Optional[DataModelsAttributes] = DataModelsAttributes()
    data_models_cdes: Optional[DataModelsCDES] = DataModelsCDES()
    datasets_locations: Optional[DatasetsLocations] = DatasetsLocations()
    datasets_variables: Optional[DatasetsVariables] = DatasetsVariables()

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def is_longitudinal(self, data_model: str) -> bool:
        return (
            LONGITUDINAL
            in self.data_models_attributes.data_models_attributes[data_model].tags
        )

    def get_cdes_specific_data_model(self, data_model) -> CommonDataElements:
        return self.data_models_cdes.data_models_cdes[data_model]

    def get_data_models_attributes(self) -> Dict[str, DataModelAttributes]:
        return self.data_models_attributes.data_models_attributes

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary with all the currently available data_models on the
        system as keys and lists of datasets as values. Without duplicates
        """
        return (
            {
                data_model: list(datasets_and_locations_of_specific_data_model.keys())
                for data_model, datasets_and_locations_of_specific_data_model in self.datasets_locations.datasets_locations.items()
            }
            if self.datasets_locations
            else {}
        )

    def data_model_exists(self, data_model: str) -> bool:
        return data_model in self.datasets_locations.datasets_locations

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return (
            data_model in self.datasets_locations.datasets_locations
            and dataset in self.datasets_locations.datasets_locations[data_model]
        )

    def get_worker_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        if not self.data_model_exists(data_model):
            return []

        local_workers_with_datasets = [
            self.datasets_locations.datasets_locations[data_model][dataset]
            for dataset in self.datasets_locations.datasets_locations[data_model]
            if dataset in datasets
            if self.datasets_locations.datasets_locations[data_model][dataset]
            != "globalworker"
        ]
        return list(set(local_workers_with_datasets))

    def get_worker_specific_datasets(
        self, worker_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        """
        From the datasets provided, returns only the ones located in the worker.

        Parameters
        ----------
        worker_id: the id of the worker
        data_model: the data model of the datasets
        wanted_datasets: the datasets to look for

        Returns
        -------
        some, all or none of the wanted_datasets that are located in the worker
        """
        if not self.data_model_exists(data_model):
            raise ValueError(
                f"Data model '{data_model}' is not available in the worker '{worker_id}'."
            )

        datasets_in_worker = [
            dataset
            for dataset in self.datasets_locations.datasets_locations[data_model]
            if dataset in wanted_datasets
            and worker_id
            == self.datasets_locations.datasets_locations[data_model][dataset]
        ]
        return datasets_in_worker


class WorkerRegistry(ImmutableBaseModel):
    def __init__(self, workers_info=None):
        if workers_info:
            global_workers = [
                worker_info
                for worker_info in workers_info
                if worker_info.role == WorkerRole.GLOBALWORKER
            ]
            local_workers = [
                worker_info
                for worker_info in workers_info
                if worker_info.role == WorkerRole.LOCALWORKER
            ]
            _workers_per_id = {
                worker_info.id: worker_info
                for worker_info in global_workers + local_workers
            }
            super().__init__(
                global_workers=global_workers,
                local_workers=local_workers,
                workers_per_id=_workers_per_id,
            )
        else:
            super().__init__()

    global_workers: List[WorkerInfo] = []
    local_workers: List[WorkerInfo] = []
    workers_per_id: Dict[str, WorkerInfo] = {}

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class _wlaRegistries(ImmutableBaseModel):
    worker_registry: Optional[WorkerRegistry] = WorkerRegistry()
    data_model_registry: Optional[DataModelRegistry] = DataModelRegistry()

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True


class DataModelMetadata(ImmutableBaseModel):
    """
    A representation of a data model's Metadata datasets info, cdes and attributes for a specific data model
    """

    dataset_infos: List[DatasetInfo]
    cdes: Optional[CommonDataElements]
    attributes: Optional[DataModelAttributes]


class DataModelsMetadata(ImmutableBaseModel):
    """
    A dictionary representation of a data model's Metadata.
    Key values are data models.
    Values are DataModelMetadata.
    """

    data_models_metadata: Dict[str, DataModelMetadata]


class DataModelsMetadataPerWorker(ImmutableBaseModel):
    """
    A dictionary representation of all information for the data model's Metadata per worker.
    Key values are workers.
    Values are data model's Metadata.
    """

    data_models_metadata_per_worker: Dict[str, DataModelsMetadata]


class WorkerLandscapeAggregator:
    def __init__(
        self,
        logger: Logger,
        update_interval: int,
        tasks_timeout: int,
        deployment_type: DeploymentType,
        localworkers: AttrDict,
    ):
        self._logger = logger
        self._update_interval = update_interval
        self._worker_info_tasks_timeout = tasks_timeout
        self._deployment_type = deployment_type
        self._localworkers = localworkers
        self._registries = _wlaRegistries()
        self._keep_updating = True
        self._update_loop_thread = None
        self._loaded_workers = set()

    def update(self):
        """
        Worker Landscape Aggregator(wla) is a module that handles the aggregation of necessary information,
        to keep up-to-date and in sync the Worker Registry and the Data Model Registry.
        The Worker Registry contains information about the worker such as id, ip, port etc.
        The Data Model Registry contains two types of information, data_models and datasets_locations.
        data_models contains information about the data models and their corresponding cdes.
        datasets_locations contains information about datasets and their locations(workers).
        wla periodically will send requests (get_worker_info, get_worker_datasets_per_data_model, get_data_model_cdes),
        to the workers to retrieve the current information that they contain.
        Once all information about data models and cdes is aggregated,
        any data model that is incompatible across workers will be removed.
        A data model is incompatible when the cdes across workers are not identical, except one edge case.
        The edge case is that the cdes can only contain a difference in the field of 'enumerations' in
        the cde with code 'dataset' and still be considered compatible.
        For each data model the 'enumerations' field in the cde with code 'dataset' is updated with all datasets across workers.
        Once all the information is aggregated and validated the wla will provide the information to the Worker Registry and to the Data Model Registry.
        """
        (
            workers_info,
            data_models_metadata_per_worker,
        ) = self._fetch_workers_metadata()
        worker_registry = WorkerRegistry(workers_info=workers_info)
        dmr = _crunch_data_model_registry_data(
            data_models_metadata_per_worker, self._logger
        )

        self._set_new_registries(worker_registry, dmr)

        self._logger.debug(
            f"Workers:{[worker_info.id for worker_info in self.get_workers()]}"
        )

    def _update_loop(self):
        while self._keep_updating:
            try:
                self.update()
            except Exception as exc:
                self._logger.warning(
                    f"WorkerLandscapeAggregator caught an exception but will continue to "
                    f"update {exc=}"
                )
                tr = traceback.format_exc()
                self._logger.error(tr)
            finally:
                time.sleep(self._update_interval)

    def start(self):
        self._logger.info("WorkerLandscapeAggregator starting ...")

        self.stop()
        self._keep_updating = True
        self._update_loop_thread = threading.Thread(
            target=self._update_loop, daemon=True
        )
        self._update_loop_thread.start()

        self._logger.info("WorkerLandscapeAggregator started.")

    def stop(self):
        if self._update_loop_thread and self._update_loop_thread.is_alive():
            self._keep_updating = False
            self._update_loop_thread.join()

    def healthcheck(self):
        worker_info_tasks_handlers = [
            WorkerInfoTasksHandler(
                worker_queue_addr=worker_queue_addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for worker_queue_addr in WorkersAddressesFactory(
                self._deployment_type, self._localworkers
            )
            .get_workers_addresses()
            .socket_addresses
        ]
        for task_handler in worker_info_tasks_handlers:
            task_handler.get_healthcheck_task(False)

    def _get_workers_info(self, workers_socket_addr: List[str]) -> List[WorkerInfo]:
        worker_info_tasks_handlers = [
            WorkerInfoTasksHandler(
                worker_queue_addr=worker_queue_addr,
                tasks_timeout=self._worker_info_tasks_timeout,
                request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
            )
            for worker_queue_addr in workers_socket_addr
        ]
        workers_info = []
        for tasks_handler in worker_info_tasks_handlers:
            try:
                result = tasks_handler.get_worker_info_task()
                workers_info.append(result)
            except (WorkerClientConnectionError, WorkerClientTimeoutException) as exc:
                # just log the exception do not reraise it
                self._logger.warning(exc)
            except Exception:
                # just log full traceback exception as error and do not reraise it
                self._logger.error(traceback.format_exc())
        return workers_info

    def _get_worker_datasets_per_data_model(
        self,
        worker_queue_addr: str,
    ) -> DatasetsInfoPerDataModel:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            datasets_per_data_model = (
                tasks_handler.get_worker_datasets_per_data_model_task()
            )
        except (WorkerClientConnectionError, WorkerClientTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
            return {}
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())
            return {}
        return datasets_per_data_model

    def _get_worker_cdes(
        self, worker_queue_addr: str, data_model: str
    ) -> CommonDataElements:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            worker_cdes = tasks_handler.get_data_model_cdes_task(
                data_model=data_model,
            )
            return worker_cdes
        except (WorkerClientConnectionError, WorkerClientTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())

    def _get_data_model_attributes(
        self, worker_queue_addr: str, data_model: str
    ) -> DataModelAttributes:
        tasks_handler = WorkerInfoTasksHandler(
            worker_queue_addr=worker_queue_addr,
            tasks_timeout=self._worker_info_tasks_timeout,
            request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
        )
        try:
            attributes = tasks_handler.get_data_model_attributes_task(
                data_model=data_model
            )
            return attributes
        except (WorkerClientConnectionError, WorkerClientTimeoutException) as exc:
            # just log the exception do not reraise it
            self._logger.warning(exc)
        except Exception:
            # just log full traceback exception as error and do not reraise it
            self._logger.error(traceback.format_exc())

    def _set_new_registries(self, worker_registry, data_model_registry):
        _log_worker_changes(
            self.get_workers(),
            list(worker_registry.workers_per_id.values()),
            self._logger,
        )
        _log_data_model_changes(
            self._registries.data_model_registry.data_models_cdes.data_models_cdes,
            data_model_registry.data_models_cdes.data_models_cdes,
            self._logger,
        )
        _log_dataset_changes(
            self._registries.data_model_registry.datasets_locations.datasets_locations,
            data_model_registry.datasets_locations.datasets_locations,
            self._logger,
        )
        self._registries = _wlaRegistries(
            worker_registry=worker_registry, data_model_registry=data_model_registry
        )

    def get_datasets_per_worker(self, data_model: str):
        dataset_worker_mapping = (
            self._registries.data_model_registry.datasets_locations.datasets_locations[
                data_model
            ]
        )
        # Initialize an empty dictionary for worker_id -> list of datasets
        worker_to_datasets = {}

        # Iterate through the mapping and invert it
        for dataset, worker_id in dataset_worker_mapping.items():
            if worker_id in worker_to_datasets:
                worker_to_datasets[worker_id].append(dataset)
            else:
                worker_to_datasets[worker_id] = [dataset]
        return worker_to_datasets

    def get_workers(self) -> List[WorkerInfo]:
        return list(self._registries.worker_registry.workers_per_id.values())

    def get_global_worker(self) -> Optional[WorkerInfo]:
        if not self._registries.worker_registry.global_workers:
            return None
        return self._registries.worker_registry.global_workers[0]

    def get_all_local_workers(self) -> List[WorkerInfo]:
        return self._registries.worker_registry.local_workers

    def get_worker_info(self, worker_id: str) -> WorkerInfo:
        return self._registries.worker_registry.workers_per_id[worker_id]

    def get_cdes(self, data_model: str) -> Dict[str, CommonDataElement]:
        return self._registries.data_model_registry.get_cdes_specific_data_model(
            data_model
        ).values

    def get_metadata(self, data_model: str, variable_names: List[str]):
        common_data_elements = self.get_cdes(data_model)
        metadata = {
            variable_name: cde.dict()
            for variable_name, cde in common_data_elements.items()
            if variable_name in variable_names
        }
        return metadata

    def get_cdes_per_data_model(self) -> DataModelsCDES:
        return self._registries.data_model_registry.data_models_cdes

    def get_datasets_locations(self) -> DatasetsLocations:
        return self._registries.data_model_registry.datasets_locations

    def get_datasets_variables(self) -> DatasetsVariables:
        return self._registries.data_model_registry.datasets_variables

    def get_training_and_validation_datasets(
        self, data_model: str
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieves all available training and validation datasets for a specific data model.

        Parameters:
            data_model (str): The data model for which to retrieve datasets.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists:
                - The first list contains training datasets (Data loaded inside the localworkers).
                - The second list contains validation datasets (Data loaded inside the globalworker).
        """
        training_datasets = []
        validation_datasets = []

        if data_model not in self.get_datasets_locations().datasets_locations.keys():
            raise BadUserInput(f"Data model '{data_model}' does not exist.")
        datasets_locations = self.get_datasets_locations().datasets_locations[
            data_model
        ]

        for dataset, worker_id in datasets_locations.items():
            if worker_id == self.get_global_worker().id:
                validation_datasets.append(dataset)
            else:
                training_datasets.append(dataset)

        return training_datasets, validation_datasets

    def get_all_available_datasets_per_data_model(self) -> Dict[str, List[str]]:
        return self._registries.data_model_registry.get_all_available_datasets_per_data_model()

    def data_model_exists(self, data_model: str) -> bool:
        return self._registries.data_model_registry.data_model_exists(data_model)

    def dataset_exists(self, data_model: str, dataset: str) -> bool:
        return self._registries.data_model_registry.dataset_exists(data_model, dataset)

    def get_worker_ids_with_any_of_datasets(
        self, data_model: str, datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_worker_ids_with_any_of_datasets(
            data_model, datasets
        )

    def get_worker_specific_datasets(
        self, worker_id: str, data_model: str, wanted_datasets: List[str]
    ) -> List[str]:
        return self._registries.data_model_registry.get_worker_specific_datasets(
            worker_id, data_model, wanted_datasets
        )

    def get_data_models_attributes(self) -> Dict[str, DataModelAttributes]:
        return self._registries.data_model_registry.get_data_models_attributes()

    def _fetch_workers_metadata(
        self,
    ) -> Tuple[List[WorkerInfo], DataModelsMetadataPerWorker]:
        """
        Returns a list of all the workers in the federation and their metadata (data_models, datasets, cdes).
        """
        workers_addresses = (
            WorkersAddressesFactory(self._deployment_type, self._localworkers)
            .get_workers_addresses()
            .socket_addresses
        )
        workers_info = self._get_workers_info(workers_addresses)
        cached_datasets = self._ensure_workers_data_loaded(workers_info)
        data_models_metadata_per_worker = self._get_data_models_metadata_per_worker(
            workers_info, cached_datasets
        )
        return workers_info, data_models_metadata_per_worker

    def _ensure_workers_data_loaded(
        self, workers: List[WorkerInfo]
    ) -> Dict[str, DatasetsInfoPerDataModel]:
        cached_datasets: Dict[str, DatasetsInfoPerDataModel] = {}
        for worker_info in workers:
            if worker_info.id in self._loaded_workers:
                continue
            try:
                handler = WorkerInfoTasksHandler(
                    worker_queue_addr=worker_info.socket_addr,
                    tasks_timeout=self._worker_info_tasks_timeout,
                    request_id=WORKER_LANDSCAPE_AGGREGATOR_REQUEST_ID,
                )
                datasets_per_data_model: DatasetsInfoPerDataModel | None = None
                try:
                    datasets_per_data_model = (
                        handler.get_worker_datasets_per_data_model_task()
                    )
                except (WorkerClientConnectionError, WorkerClientTimeoutException):
                    # Most commonly raised when metadata tables are missing (first boot)
                    self._logger.debug(
                        "Failed to fetch datasets for worker %s before loading data",
                        worker_info.id,
                        exc_info=True,
                    )

                if (
                    datasets_per_data_model
                    and datasets_per_data_model.datasets_info_per_data_model
                ):
                    cached_datasets[worker_info.id] = datasets_per_data_model
                    self._loaded_workers.add(worker_info.id)
                    continue

                self._loaded_workers.add(worker_info.id)
                try:
                    datasets_per_data_model = (
                        handler.get_worker_datasets_per_data_model_task()
                    )
                    if (
                        datasets_per_data_model
                        and datasets_per_data_model.datasets_info_per_data_model
                    ):
                        cached_datasets[worker_info.id] = datasets_per_data_model
                except (WorkerClientConnectionError, WorkerClientTimeoutException):
                    self._logger.debug(
                        "Failed to fetch datasets after loading data for worker %s",
                        worker_info.id,
                        exc_info=True,
                    )
            except Exception:
                self._logger.error(
                    "Failed to auto-load data for worker %s", worker_info.id
                )
                self._logger.debug(traceback.format_exc())
        return cached_datasets

    def _get_data_models_metadata_per_worker(
        self,
        workers: List[WorkerInfo],
        cached_datasets: Optional[Dict[str, DatasetsInfoPerDataModel]] = None,
    ) -> DataModelsMetadataPerWorker:
        cached_datasets = cached_datasets or {}
        data_models_metadata_per_worker = {}

        for worker_info in workers:
            data_models_metadata = {}

            worker_socket_addr = worker_info.socket_addr
            datasets_per_data_model = cached_datasets.get(worker_info.id)
            if not datasets_per_data_model:
                datasets_per_data_model = self._get_worker_datasets_per_data_model(
                    worker_socket_addr
                )
            if datasets_per_data_model:
                for (
                    data_model,
                    dataset_infos,
                ) in datasets_per_data_model.datasets_info_per_data_model.items():
                    cdes = self._get_worker_cdes(worker_socket_addr, data_model)
                    attributes = self._get_data_model_attributes(
                        worker_socket_addr, data_model
                    )
                    cdes = cdes if cdes else None
                    attributes = attributes if attributes else None
                    data_models_metadata[data_model] = DataModelMetadata(
                        dataset_infos=dataset_infos,
                        cdes=cdes,
                        attributes=attributes,
                    )
                    data_models_metadata_per_worker[worker_info.id] = (
                        DataModelsMetadata(data_models_metadata=data_models_metadata)
                    )

        return DataModelsMetadataPerWorker(
            data_models_metadata_per_worker=data_models_metadata_per_worker
        )


def _crunch_data_model_registry_data(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker, logger
) -> DataModelRegistry:
    data_models_metadata_per_worker_with_compatible_data_models = (
        _remove_incompatible_data_models_from_data_models_metadata_per_worker(
            data_models_metadata_per_worker, logger
        )
    )

    cleaned_data_models_metadata_per_worker = _remove_duplicate_datasets(
        data_models_metadata_per_worker_with_compatible_data_models, logger
    )
    data_models_cdes = _aggregate_data_models_cdes(
        cleaned_data_models_metadata_per_worker, logger
    )

    data_models_attributes = _aggregate_data_models_attributes(
        cleaned_data_models_metadata_per_worker,
    )
    datasets_locations = _extract_datasets_locations(
        cleaned_data_models_metadata_per_worker
    )
    datasets_variables = _extract_datasets_variables(
        cleaned_data_models_metadata_per_worker
    )
    return DataModelRegistry(
        data_models_cdes=data_models_cdes,
        datasets_locations=datasets_locations,
        data_models_attributes=data_models_attributes,
        datasets_variables=datasets_variables,
    )


def _remove_duplicate_datasets(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker, logger
) -> DataModelsMetadataPerWorker:
    dataset_to_workers = defaultdict(lambda: defaultdict(set))
    updated_data_models_metadata_per_worker = {}

    # First pass to identify duplicates and log them
    for (
        worker,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            for dataset in model_metadata.dataset_infos:
                if dataset.code in dataset_to_workers[data_model]:
                    dataset_to_workers[data_model][dataset.code].add(worker)
                    _log_duplicated_dataset(
                        list(dataset_to_workers[data_model][dataset.code]),
                        data_model,
                        dataset,
                        logger,
                    )
                else:
                    dataset_to_workers[data_model][dataset.code].add(worker)

    # Second pass to create new instances without duplicates
    for (
        worker,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        updated_data_models_metadata = {}
        for (
            data_model,
            model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            unique_datasets = []
            for dataset in model_metadata.dataset_infos:
                if len(dataset_to_workers[data_model][dataset.code]) == 1:
                    unique_datasets.append(dataset)
            updated_data_models_metadata[data_model] = DataModelMetadata(
                dataset_infos=unique_datasets,
                cdes=model_metadata.cdes,
                attributes=model_metadata.attributes,
            )
        updated_data_models_metadata_per_worker[worker] = DataModelsMetadata(
            data_models_metadata=updated_data_models_metadata
        )

    return DataModelsMetadataPerWorker(
        data_models_metadata_per_worker=updated_data_models_metadata_per_worker
    )


def _aggregate_data_models_cdes(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker, logger
) -> DataModelsCDES:
    # Keep per-data-model enumerations in an OrderedDict to preserve insertion order.
    data_models_dataset_enumerations: dict[str, OrderedDict[str, str]] = {}
    # First pass: collect all dataset enumerations per data model in a stable order.
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            if data_model not in data_models_dataset_enumerations:
                data_models_dataset_enumerations[data_model] = OrderedDict()

            enum_dict = data_models_dataset_enumerations[data_model]

            # Assuming dataset_infos is already in the desired order.
            for dataset_info in data_model_metadata.dataset_infos:
                # Only set once to avoid bumping items to the end if the same code
                # appears from another worker.
                if dataset_info.code not in enum_dict:
                    enum_dict[dataset_info.code] = dataset_info.label

    # Second pass: build the final DataModelsCDES structure.
    data_models: Dict[str, CommonDataElements] = {}

    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            data_models[data_model] = data_model_metadata.cdes

            dataset_cde = data_model_metadata.cdes.values["dataset"]
            new_dataset_cde = CommonDataElement(
                code=dataset_cde.code,
                label=dataset_cde.label,
                sql_type=dataset_cde.sql_type,
                is_categorical=dataset_cde.is_categorical,
                enumerations=data_models_dataset_enumerations[data_model],
                min=dataset_cde.min,
                max=dataset_cde.max,
            )
            data_models[data_model].values["dataset"] = new_dataset_cde
    return DataModelsCDES(data_models_cdes=data_models)


def _aggregate_data_models_attributes(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
) -> DataModelsAttributes:
    data_models_attributes = {}
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            current_tags = _get_updated_tags(
                data_model=data_model,
                data_models_attributes=data_models_attributes,
                tags=data_model_metadata.attributes.tags,
            )
            current_properties = _get_updated_properties(
                data_model=data_model,
                data_models_attributes=data_models_attributes,
                properties_to_be_added=data_model_metadata.attributes.properties,
            )
            data_models_attributes[data_model] = DataModelAttributes(
                tags=current_tags, properties=current_properties
            )
    return DataModelsAttributes(data_models_attributes=data_models_attributes)


def _extract_datasets_locations(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
) -> DatasetsLocations:
    datasets_locations_dict = defaultdict(lambda: defaultdict(dict))

    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            for dataset in model_metadata.dataset_infos:
                datasets_locations_dict[data_model][dataset.code] = worker_id
    return DatasetsLocations(datasets_locations=datasets_locations_dict)


def _extract_datasets_variables(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
) -> DatasetsVariables:
    datasets_variables_dict = defaultdict(lambda: defaultdict(list))

    for (
        _,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            for dataset in model_metadata.dataset_infos:
                variables = getattr(dataset, "variables", None)
                datasets_variables_dict[data_model][dataset.code] = (
                    list(variables) if variables else []
                )

    return DatasetsVariables(datasets_variables=datasets_variables_dict)


def _get_updated_properties(data_model, data_models_attributes, properties_to_be_added):
    if data_model not in data_models_attributes:
        return {key: [value] for key, value in properties_to_be_added.items()}

    properties = data_models_attributes[data_model].properties
    for key, value in properties_to_be_added.items():
        properties[key] = (
            (properties[key] if value in properties[key] else properties[key] + [value])
            if key in properties
            else [value]
        )
    return properties


def _get_updated_tags(data_model, data_models_attributes, tags):
    return (
        data_models_attributes[data_model].tags
        + list(set(tags) - set(data_models_attributes[data_model].tags))
        if data_model in data_models_attributes
        else tags
    )


def _remove_incompatible_data_models_from_data_models_metadata_per_worker(
    data_models_metadata_per_worker: DataModelsMetadataPerWorker,
    logger: Logger,
) -> DataModelsMetadataPerWorker:
    """
    Each worker has its own data models definition.
    We need to check for each data model if the definitions across all workers is the same.
    If the data model is not the same across all workers containing it, we log the incompatibility.
    The data models with similar definitions across all workers are returned.
    Parameters
    ----------
        data_models_metadata_per_worker: DataModelsMetadataPerWorker
    Returns
    ----------
        DataModelsMetadataPerWorker
            The data_models_metadata_per_worker but with removed the incompatible data models
    """
    validation_dictionary = {}

    incompatible_data_models = []
    for (
        worker_id,
        data_models_metadata,
    ) in data_models_metadata_per_worker.data_models_metadata_per_worker.items():
        for (
            data_model,
            data_model_metadata,
        ) in data_models_metadata.data_models_metadata.items():
            if (
                data_model in incompatible_data_models
                or not data_model_metadata.cdes
                or not data_model_metadata.attributes
            ):
                continue

            if data_model in validation_dictionary:
                valid_worker_id, valid_cdes = validation_dictionary[data_model]
                if not valid_cdes == data_model_metadata.cdes:
                    workers = [worker_id, valid_worker_id]
                    incompatible_data_models.append(data_model)
                    _log_incompatible_data_models(
                        workers,
                        data_model,
                        [data_model_metadata.cdes, valid_cdes],
                        logger,
                    )
                    break
            else:
                validation_dictionary[data_model] = (
                    worker_id,
                    data_model_metadata.cdes,
                )

    return DataModelsMetadataPerWorker(
        data_models_metadata_per_worker={
            worker_id: DataModelsMetadata(
                data_models_metadata={
                    data_model: data_model_metadata
                    for data_model, data_model_metadata in data_models_metadata.data_models_metadata.items()
                    if data_model not in incompatible_data_models
                    and data_model_metadata.cdes
                    and data_model_metadata.attributes
                }
            )
            for worker_id, data_models_metadata in data_models_metadata_per_worker.data_models_metadata_per_worker.items()
        }
    )


def _log_worker_changes(old_workers, new_workers, logger):
    old_workers_per_worker_id = {worker.id: worker for worker in old_workers}
    new_workers_per_worker_id = {worker.id: worker for worker in new_workers}
    added_workers = set(new_workers_per_worker_id.keys()) - set(
        old_workers_per_worker_id.keys()
    )
    for worker in added_workers:
        log_worker_joined_federation(logger, worker)
    removed_workers = set(old_workers_per_worker_id.keys()) - set(
        new_workers_per_worker_id.keys()
    )
    for worker in removed_workers:
        log_worker_left_federation(logger, worker)


def _log_data_model_changes(old_data_models, new_data_models, logger):
    added_data_models = new_data_models.keys() - old_data_models.keys()
    for data_model in added_data_models:
        log_datamodel_added(data_model, logger)
    removed_data_models = old_data_models.keys() - new_data_models.keys()
    for data_model in removed_data_models:
        log_datamodel_removed(data_model, logger)


def _log_dataset_changes(old_datasets_locations, new_datasets_locations, logger):
    _log_datasets_added(old_datasets_locations, new_datasets_locations, logger)
    _log_datasets_removed(old_datasets_locations, new_datasets_locations, logger)


def _log_datasets_added(old_datasets_locations, new_datasets_locations, logger):
    for data_model in new_datasets_locations:
        added_datasets = new_datasets_locations[data_model].keys()
        if data_model in old_datasets_locations:
            added_datasets -= old_datasets_locations[data_model].keys()
        for dataset in added_datasets:
            log_dataset_added(
                data_model,
                dataset,
                logger,
                new_datasets_locations[data_model][dataset],
            )


def _log_datasets_removed(old_datasets_locations, new_datasets_locations, logger):
    for data_model in old_datasets_locations:
        removed_datasets = old_datasets_locations[data_model].keys()
        if data_model in new_datasets_locations:
            removed_datasets -= new_datasets_locations[data_model].keys()
        for dataset in removed_datasets:
            log_dataset_removed(
                data_model,
                dataset,
                logger,
                old_datasets_locations[data_model][dataset],
            )


def _log_incompatible_data_models(workers, data_model, conflicting_cdes, logger):
    logger.info(
        f"""Workers: '[{", ".join(workers)}]' on data model '{data_model}' have incompatibility on the following cdes: '[{", ".join([cdes.__str__() for cdes in conflicting_cdes])}]' """
    )


def _log_duplicated_dataset(workers, data_model, dataset, logger):
    logger.info(
        f"Dataset '{dataset.code}' of the data_model: '{data_model}' is not unique in the federation. "
        f"Workers that contain the dataset: '[{', '.join(workers)}]'"
    )
