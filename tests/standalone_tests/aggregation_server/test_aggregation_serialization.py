import numpy as np
import pytest

from exaflow import aggregation_serialization


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (3,),
        (2, 3),
    ],
)
def test_arrow_ndarray_roundtrip_preserves_dtype_shape_and_values(dtype, shape):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    result = aggregation_serialization.bytes_to_ndarray(
        aggregation_serialization.ndarray_to_bytes(values)
    )

    assert result.dtype == values.dtype
    assert result.shape == values.shape
    np.testing.assert_array_equal(result, values, strict=True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_arrow_ndarray_roundtrip_preserves_non_finite_float_values(
    dtype,
):
    values = np.array([np.nan, np.inf, -np.inf, 1.5], dtype=dtype)

    result = aggregation_serialization.bytes_to_ndarray(
        aggregation_serialization.ndarray_to_bytes(values)
    )

    assert result.dtype == values.dtype
    assert result.shape == values.shape
    np.testing.assert_array_equal(result, values, strict=True)


def test_json_values_roundtrip_preserves_union_values_as_object_array():
    values = ["b", "a", 3, None]

    result = aggregation_serialization.bytes_to_values(
        aggregation_serialization.values_to_bytes(values)
    )

    assert result.dtype == object
    np.testing.assert_array_equal(result, np.asarray(values, dtype=object))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_json_values_rejects_non_finite_float_values(bad_value):
    with pytest.raises(
        ValueError,
        match="UNION payload contains non-JSON-compliant float values",
    ):
        aggregation_serialization.values_to_bytes(["a", bad_value])
