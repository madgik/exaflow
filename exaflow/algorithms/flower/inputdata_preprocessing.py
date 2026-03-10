import json
import os
import time
from math import log2
from math import pow
from typing import Optional

import flwr as fl
import requests
from flwr.common.logger import FLOWER_LOGGER
from sklearn import preprocessing

from exaflow.algorithms.utils.inputdata_utils import Inputdata

# Constants for project directories and environment configurations
CONTROLLER_IP = os.getenv("CONTROLLER_IP", "127.0.0.1")
CONTROLLER_PORT = os.getenv("CONTROLLER_PORT", 5000)
RESULT_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/flower/result"
INPUT_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/flower/input"
CDES_URL = f"http://{CONTROLLER_IP}:{CONTROLLER_PORT}/cdes_metadata"
HEADERS = {"Content-type": "application/json", "Accept": "text/plain"}


def preprocess_data(inputdata, full_data):
    # Ensure x and y are specified and correct
    if not inputdata.x or not inputdata.y:
        raise ValueError("Input features 'x' and labels 'y' must be specified")

    # Select features and target based on inputdata configuration
    features = full_data[inputdata.x]  # This should be a DataFrame
    target = full_data[inputdata.y].values.ravel()  # Flatten the array if it's 2D

    # Encode target variable
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(get_enumerations(inputdata.data_model, inputdata.y[0]))
    y_train = label_encoder.transform(target)

    return features, y_train


def error_handling(error):
    error_msg = {"error": str(error)}
    FLOWER_LOGGER.error(
        f"Error will try to save error message: {error_msg}! Running: {RESULT_URL}..."
    )
    requests.post(RESULT_URL, data=json.dumps(error_msg), headers=HEADERS)


def post_result(result: dict) -> None:
    FLOWER_LOGGER.debug(f"Posting result at: {RESULT_URL} ...")
    response = requests.post(RESULT_URL, data=json.dumps(result), headers=HEADERS)
    if response.status_code != 200:
        error_handling(response.text)


def get_input() -> Inputdata:
    FLOWER_LOGGER.debug(f"Getting inputdata from: {INPUT_URL} ...")
    response = requests.get(INPUT_URL)
    if response.status_code != 200:
        error_handling(response.text)
    inputdata = Inputdata.model_validate_json(response.text)
    return inputdata


def get_enumerations(data_model: str, variable_name: str) -> Optional[list]:
    try:
        FLOWER_LOGGER.debug(f"Getting enumerations from: {CDES_URL} ...")
        response = requests.get(CDES_URL)
        if response.status_code != 200:
            error_handling(response.text)
        cdes_metadata = response.json()
        if data_model not in cdes_metadata:
            raise KeyError(f"'{data_model}' key not found in cdes_metadata")

        if variable_name not in cdes_metadata[data_model]:
            raise KeyError(f"'{variable_name}' key not found in {data_model}")

        enumerations = cdes_metadata[data_model][variable_name].get("enumerations")
        if enumerations is not None:
            return [code for code, label in enumerations.items()]
        else:
            raise KeyError(f"'enumerations' key not found in {variable_name}")
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        error_handling(str(e))


def connect_with_retries(client, client_name):
    """
    Attempts to connect the client to the Flower server with retries.

    Args:
        client: The client instance to connect.
        client_name: The name of the client (for logging purposes).
    """
    attempts = 0
    max_attempts = int(log2(int(os.environ["TIMEOUT"])))

    while True:
        try:
            fl.client.start_client(
                server_address=os.environ["SERVER_ADDRESS"], client=client.to_client()
            )
            FLOWER_LOGGER.debug(
                f"{client_name} - Connection successful on attempt: {attempts + 1}"
            )
            break
        except Exception as e:
            FLOWER_LOGGER.warning(
                f"{client_name} - Connection with the server failed. Attempt {attempts + 1} failed: {e}"
            )
            time.sleep(pow(2, attempts))  # Exponential backoff
            attempts += 1
            if attempts >= max_attempts:
                FLOWER_LOGGER.error(
                    f"{client_name} - Could not establish connection to the server."
                )
                raise e
