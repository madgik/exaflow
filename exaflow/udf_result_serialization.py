import json
from typing import Any

import numpy as np

UDF_RESULT_FORMAT_JSON_BYTES_V1 = "json-bytes-v1"


def encode_udf_result(result: Any) -> bytes:
    return json.dumps(_normalize_for_json(result)).encode("utf-8")


def decode_udf_result(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalize_for_json(value.tolist())

    if isinstance(value, np.generic):
        return _normalize_for_json(value.item())

    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]

    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]

    if isinstance(value, dict):
        return {
            _normalize_dict_key(key): _normalize_for_json(item)
            for key, item in value.items()
        }

    return value


def _normalize_dict_key(key: Any) -> Any:
    if isinstance(key, np.generic):
        return key.item()
    return key
