import enum
import json
from logging import Logger
from typing import List
from typing import Optional

import requests
from pydantic import BaseModel

ADD_DATASET_ENDPOINT = "/api/update-dataset/"
TRIGGER_COMPUTATION_ENDPOINT = "/api/secure-aggregation/job-id/"
GET_RESULT_ENDPOINT = "/api/get-result/job-id/"


# ~~~~~~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class SMPCRequestType(enum.Enum):
    INT_SUM = "sum"
    INT_MIN = "min"
    INT_MAX = "max"
    SUM = "fsum"
    MIN = "fmin"
    MAX = "fmax"

    def __str__(self):
        return self.name


class SMPCResponseStatus(enum.Enum):
    IN_QUEUE = "IN_QUEUE"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    def __str__(self):
        return self.name


class DPRequestData(BaseModel):
    c: Optional[float] = None  # Differential Privacy - sensitivity
    e: Optional[float] = None  # Differential Privacy - privacy budget


class SMPCRequestData(BaseModel):
    computationType: SMPCRequestType
    clients: List[str]
    dp: Optional[DPRequestData] = None


class DifferentialPrivacyParams(BaseModel):
    sensitivity: float
    privacy_budget: float


class SMPCResponse(BaseModel):
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus


class SMPCResponseWithOutput(BaseModel):
    computationOutput: List[float]
    computationType: SMPCRequestType
    jobId: str
    status: SMPCResponseStatus


# ~~~~~~~~~~~~~~~~~~~~~~~~ Exceptions ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class SMPCUsageError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class SMPCCommunicationError(Exception):
    pass


class SMPCComputationError(Exception):
    pass


# ~~~~~~~~~~~~~~~~~~~~~~~~ SMPC cluster communication helper methods ~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _get_smpc_load_data_request_data_structure(data_values: str):
    """
    The current approach with the SMPC cluster is to send all computations as floats.
    That way we don't need to have separate operations for sum-int and sum-float.
    """
    data = {"type": "float", "data": json.loads(data_values)}
    return json.dumps(data)


def load_data_to_smpc_client(client_address: str, jobid: str, values: str):
    request_url = client_address + ADD_DATASET_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.post(
        url=request_url,
        data=_get_smpc_load_data_request_data_structure(values),
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


def get_smpc_result(coordinator_address: str, jobid: str) -> str:
    request_url = coordinator_address + GET_RESULT_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    response = requests.get(
        url=request_url,
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )
    return response.text


def trigger_smpc(
    logger: Logger,
    coordinator_address: str,
    jobid: str,
    payload: SMPCRequestData,
):
    request_url = coordinator_address + TRIGGER_COMPUTATION_ENDPOINT + jobid
    request_headers = {"Content-type": "application/json", "Accept": "text/plain"}
    logger.info(f"Starting SMPC with {jobid=}...")
    logger.debug(f"{request_url=}")
    logger.debug(f"{payload=}")
    response = requests.post(
        url=request_url,
        data=payload.model_dump_json(),
        headers=request_headers,
    )
    if response.status_code != 200:
        raise SMPCCommunicationError(
            f"Response status code: {response.status_code} \n Body:{response.text}"
        )


def create_payload(
    computation_type: SMPCRequestType,
    clients: List[str],
    dp_params: DifferentialPrivacyParams = None,
) -> SMPCRequestData:
    if dp_params:
        return SMPCRequestData(
            computationType=computation_type,
            clients=clients,
            dp=DPRequestData(
                c=dp_params.sensitivity,
                e=dp_params.privacy_budget,
            ),
        )
    else:
        return SMPCRequestData(computationType=computation_type, clients=clients)


def validate_smpc_usage(use_smpc: bool, smpc_enabled: bool, smpc_optional: bool):
    """
    Validates if smpc can be used or if it must be used based on the configs.
    """
    if use_smpc and not smpc_enabled:
        raise SMPCUsageError("SMPC cannot be used, since it's not enabled.")

    if not use_smpc and smpc_enabled and not smpc_optional:
        raise SMPCUsageError(
            "The computation cannot be made without SMPC. SMPC usage is not optional."
        )
