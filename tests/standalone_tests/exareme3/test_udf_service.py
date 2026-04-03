import pandas as pd
import pytest

from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import InsufficientDataError


class _DummyAggClient:
    def __init__(self):
        self.unregistered = False
        self.closed = False

    def unregister(self):
        self.unregistered = True

    def close(self):
        self.closed = True


class _DropAllStep:
    def __init__(self, *, params):
        self._params = params

    def transform_data_and_metadata(self, *, data, metadata):
        return data.iloc[0:0], metadata


def test_check_min_rows_uses_pandas_row_count(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"x": [1, 2]})
    agg_client = _DummyAggClient()

    with pytest.raises(InsufficientDataError, match="minimum required is 3"):
        udf_service._check_min_rows_or_raise(
            data=data,
            check_min_rows=True,
            agg_client=agg_client,
        )

    assert agg_client.unregistered is True
    assert agg_client.closed is True


def test_apply_preprocessing_checks_min_rows_after_each_step(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 1)
    monkeypatch.setattr(
        udf_service,
        "exareme3_preprocessing_step_classes",
        {"drop_all": _DropAllStep},
    )
    data = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(InsufficientDataError, match="minimum required is 1"):
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata={"x": {"is_categorical": False}},
            preprocessing={"drop_all": {}},
            check_min_rows=True,
            agg_client=None,
        )


def test_apply_preprocessing_checks_min_rows_when_no_steps(monkeypatch):
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 3)
    data = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(InsufficientDataError, match="minimum required is 3"):
        udf_service._apply_preprocessing_steps_to_data_and_metadata(
            data=data,
            metadata={"x": {"is_categorical": False}},
            preprocessing={},
            check_min_rows=True,
            agg_client=None,
        )
