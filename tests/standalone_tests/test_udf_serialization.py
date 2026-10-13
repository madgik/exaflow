import math

import numpy as np

from exaflow.udf_serialization import decode_udf_payload
from exaflow.udf_serialization import encode_udf_payload


def test_udf_payload_json_bytes_round_trip_normalizes_nested_numpy_and_tuples():
    result = {
        "nested": {
            "list": [np.int64(3), np.float64(4.5), (np.bool_(True), "x")],
            "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        },
        "tuple": (np.float32(1.25), np.array([5.5, 6.5])),
    }

    decoded = decode_udf_payload(encode_udf_payload(result))

    assert decoded == {
        "nested": {
            "list": [3, 4.5, [True, "x"]],
            "array": [[1, 2], [3, 4]],
        },
        "tuple": [1.25, [5.5, 6.5]],
    }


def test_udf_payload_json_bytes_round_trip_preserves_non_finite_floats():
    decoded = decode_udf_payload(
        encode_udf_payload(
            {
                "nan": float("nan"),
                "inf": float("inf"),
                "neg_inf": np.float64(float("-inf")),
            }
        )
    )

    assert math.isnan(decoded["nan"])
    assert decoded["inf"] == float("inf")
    assert decoded["neg_inf"] == float("-inf")
