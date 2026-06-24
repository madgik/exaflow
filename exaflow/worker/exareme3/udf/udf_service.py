import inspect
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from exaflow import exareme3_preprocessing_step_classes
from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exareme3.utils.metadata_enums import enforce_enum_order
from exaflow.algorithms.exareme3.utils.registry import exareme3_registry
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.utils.pandas_utils import convert_to_pandas_dataframe
from exaflow.worker import config as worker_config
from exaflow.worker.exareme3.lazy_aggregation import lazy_agg
from exaflow.worker.exareme3.udf.udf_db import load_algorithm_arrow_table
from exaflow.worker.utils.logger import get_logger
from exaflow.worker.utils.logger import initialise_logger
from exaflow.worker_communication import BadUserInput
from exaflow.worker_communication import InsufficientDataError
from exaflow.worker_communication import RunUdfSystemArgs


@initialise_logger
def run_udf(
    request_id,
    udf_registry_key: str,
    kw_args: dict,
    system_args: RunUdfSystemArgs,
):
    udf = _get_udf_or_raise(udf_registry_key)
    udf = _wrap_udf_with_lazy_aggregation_if_enabled(udf_registry_key, udf)
    agg_client = _create_aggregation_client_if_required(
        request_id,
        udf_registry_key,
        system_args.preprocessing,
    )
    preprocessing = system_args.preprocessing
    metadata = enforce_enum_order(system_args.metadata)
    data = _load_worker_data_for_udf(
        inputdata=system_args.inputdata,
        preprocessing=preprocessing,
        add_dataset_variable=system_args.add_dataset_variable,
    )
    data, metadata = _apply_preprocessing_steps_to_data_and_metadata(
        data=data,
        metadata=metadata,
        preprocessing=preprocessing,
        check_min_rows=system_args.check_min_rows,
        agg_client=agg_client,
    )

    try:
        return _execute_udf(
            udf=udf,
            kw_args=kw_args,
            data=data,
            metadata=metadata,
            agg_client=agg_client,
            agg_client_name=exareme3_registry.agg_client_name(udf_registry_key),
        )
    except BadInputError as e:
        logger = get_logger()
        logger.info(
            f"Bad input while calling udf. (request_id={request_id})(udf={udf_registry_key})(error={e})"
        )
        raise BadUserInput(str(e)) from e
    except TypeError as e:
        logger = get_logger()
        logger.error(
            f"Error calling udf. (udf={udf_registry_key})({kw_args=})({system_args=})(error={e})"
        )
        raise


def _get_udf_or_raise(udf_registry_key: str):
    udf = exareme3_registry.get_func(udf_registry_key)
    if not udf:
        error_msg = f"udf '{udf_registry_key}' not found in EXAREME3_REGISTRY."
        raise ImportError(error_msg)
    return udf


def _wrap_udf_with_lazy_aggregation_if_enabled(udf_registry_key: str, udf):
    if exareme3_registry.lazy_aggregation_enabled(udf_registry_key):
        agg_client_name = exareme3_registry.agg_client_name(udf_registry_key)
        return lazy_agg(agg_client_name=agg_client_name)(udf)
    return udf


def _create_aggregation_client_if_required(
    request_id,
    udf_registry_key: str,
    preprocessing: Optional[List[Dict[str, Any]]],
) -> Optional[AggregationClient]:
    if not exareme3_registry.aggregation_server_required(
        udf_registry_key
    ) and not _preprocessing_requires_aggregation_server(preprocessing):
        return None
    agg_dns = worker_config.aggregation_server.dns
    return AggregationClient(request_id, aggregator_dns=agg_dns)


def _preprocessing_requires_aggregation_server(
    preprocessing: Optional[List[Dict[str, Any]]],
) -> bool:
    for preprocessing_step in preprocessing or []:
        preprocessing_step_name = preprocessing_step["name"]
        preprocessing_step_cls = exareme3_preprocessing_step_classes[
            preprocessing_step_name
        ]
        components = preprocessing_step_cls.get_specification().components
        if any(component.value == "AGGREGATION_SERVER" for component in components):
            return True
    return False


def _collect_extra_columns(preprocessing: Optional[List[Dict[str, Any]]]) -> set[str]:
    extra_columns = set()
    for preprocessing_step in preprocessing or []:
        preprocessing_step_name = preprocessing_step["name"]
        preprocessing_step_cls = exareme3_preprocessing_step_classes[
            preprocessing_step_name
        ]
        extra_columns.update(preprocessing_step_cls.required_input_variables())
    return extra_columns


def _load_worker_data_for_udf(
    *,
    inputdata,
    preprocessing: Optional[List[Dict[str, Any]]],
    add_dataset_variable: bool,
):
    include_dataset = False
    return load_algorithm_arrow_table(
        inputdata,
        include_dataset=(include_dataset or add_dataset_variable),
        extra_columns=_collect_extra_columns(preprocessing),
    )


def _check_min_rows_or_raise(
    *,
    data,
    check_min_rows: bool,
    agg_client: Optional[AggregationClient],
) -> None:
    if not check_min_rows:
        return

    num_rows = getattr(data, "num_rows", len(data))
    min_required = worker_config.privacy.minimum_row_count
    if num_rows >= min_required:
        return

    if agg_client:
        try:
            agg_client.unregister()
        finally:
            agg_client.close()

    raise InsufficientDataError(
        f"Insufficient data returned {num_rows} rows; minimum required is {min_required}."
    )


def _apply_preprocessing_steps_to_data_and_metadata(
    data,
    metadata: dict,
    preprocessing: Optional[List[Dict[str, Any]]],
    check_min_rows: bool,
    agg_client: Optional[AggregationClient],
):
    data = convert_to_pandas_dataframe(data)
    _check_min_rows_or_raise(
        data=data,
        check_min_rows=check_min_rows,
        agg_client=agg_client,
    )

    for preprocessing_step in preprocessing or []:
        preprocessing_step_name = preprocessing_step["name"]
        preprocessing_step_params = preprocessing_step.get("parameters", {})
        preprocessing_step_cls = exareme3_preprocessing_step_classes[
            preprocessing_step_name
        ]
        step_metadata = metadata
        preprocessing_step = preprocessing_step_cls(
            params=preprocessing_step_params,
        )
        transform_kwargs = {
            "data": data,
            "metadata": step_metadata,
        }
        if (
            "agg_client"
            in inspect.signature(
                preprocessing_step.transform_data_and_metadata
            ).parameters
        ):
            transform_kwargs["agg_client"] = agg_client
        data, metadata = preprocessing_step.transform_data_and_metadata(
            **transform_kwargs
        )
        _check_min_rows_or_raise(
            data=data,
            check_min_rows=check_min_rows,
            agg_client=agg_client,
        )
    return data, metadata


def _execute_udf(
    *,
    udf,
    kw_args: dict,
    data,
    metadata: dict,
    agg_client: Optional[AggregationClient],
    agg_client_name: str = "agg_client",
):
    udf_parameters = inspect.signature(udf).parameters
    if agg_client and agg_client_name in udf_parameters:
        kw_args[agg_client_name] = agg_client
    kw_args["data"] = data
    if "metadata" in udf_parameters:
        kw_args["metadata"] = metadata
    return udf(**kw_args)
