from typing import Optional

from exaflow.aggregation_clients.exareme3_udf_aggregation_client import (
    Exareme3UDFAggregationClient as AggregationClient,
)
from exaflow.algorithms.exareme3.longitudinal_transformer import (
    apply_longitudinal_transformation,
)
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


def enforce_enum_order(data_dict):
    for key, field in data_dict.items():
        if field.get("enumerations"):
            ordered = field.get("ordered_enums")

            # Only process if ordered_enums exists
            if ordered and "enumerations" in field:
                enums = field["enumerations"]

                # Rebuild the enumerations dict using the list order
                new_enums = {code: enums[code] for code in ordered if code in enums}

                field["enumerations"] = new_enums

                # Remove the ordered_enums entry
                del field["ordered_enums"]
    return data_dict


@initialise_logger
def run_udf(
    request_id,
    udf_registry_key: str,
    kw_args: dict,
    system_args: RunUdfSystemArgs,
):
    udf = exareme3_registry.get_func(udf_registry_key)
    if not udf:
        error_msg = f"udf '{udf_registry_key}' not found in EXAREME3_REGISTRY."
        raise ImportError(error_msg)

    if exareme3_registry.lazy_aggregation_enabled(udf_registry_key):
        agg_client_name = exareme3_registry.agg_client_name(udf_registry_key)
        udf = lazy_agg(agg_client_name=agg_client_name)(udf)

    agg_client: Optional[AggregationClient] = None
    if exareme3_registry.aggregation_server_required(udf_registry_key):
        agg_dns = worker_config.aggregation_server.dns
        agg_client = AggregationClient(request_id, aggregator_dns=agg_dns)

    inputdata = system_args.inputdata
    # GRPC will mess with the order of dict when sending from controller to worker we need a list with the order to we can re-arrange them properly
    if (
        "metadata" in kw_args
    ):  # TODO We should not expect the metadata hard coded in a specific name, try to remove
        kw_args["metadata"] = enforce_enum_order(kw_args["metadata"])

    preprocessing = system_args.preprocessing
    include_dataset = False
    extra_columns = set()
    if preprocessing and "longitudinal_transformer" in preprocessing:
        include_dataset = True
        extra_columns.update(
            preprocessing["longitudinal_transformer"].get("raw_x", [])
            + preprocessing["longitudinal_transformer"].get("raw_y", [])
        )
        extra_columns.update({"subjectid", "visitid"})

    data = load_algorithm_arrow_table(
        inputdata,
        dropna=system_args.drop_na,
        include_dataset=(include_dataset or system_args.add_dataset_variable),
        extra_columns=extra_columns if extra_columns else None,
    )

    if system_args.check_min_rows:
        num_rows = data.num_rows
        min_required = worker_config.privacy.minimum_row_count
        if num_rows < min_required:
            if agg_client:
                try:
                    agg_client.unregister()
                finally:
                    agg_client.close()
            raise InsufficientDataError(
                f"Insufficient data returned {num_rows} rows; minimum required is {min_required}."
            )

    data = convert_to_pandas_dataframe(data)
    if preprocessing and "longitudinal_transformer" in preprocessing:
        data = apply_longitudinal_transformation(
            data, preprocessing["longitudinal_transformer"]
        )

    try:
        if agg_client:
            kw_args["agg_client"] = agg_client
        kw_args["data"] = data
        result = udf(**kw_args)
        return result
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
