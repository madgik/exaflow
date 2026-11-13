"""The worker decides whether the minimum-row check runs, not the request.

The decision comes from the UDF registration on the worker. A caller cannot
switch the privacy threshold off through the request payload.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from exaflow.worker.exareme3.udf import udf_service
from exaflow.worker_communication import InsufficientDataError
from exaflow.worker_communication import RunUdfSystemArgs

UDF_KEY = "min_rows_probe_udf"


def _count_rows(data):
    return {"rows": len(data)}


@pytest.fixture
def two_row_worker(monkeypatch):
    """A worker that holds two rows and requires three."""
    monkeypatch.setattr(udf_service.worker_config.privacy, "minimum_row_count", 3)
    monkeypatch.setattr(udf_service, "_get_udf_or_raise", lambda _: _count_rows)
    monkeypatch.setattr(
        udf_service,
        "_wrap_udf_with_lazy_aggregation_if_enabled",
        lambda _, udf: udf,
    )
    monkeypatch.setattr(
        udf_service, "_create_aggregation_client_if_required", lambda **_: None
    )
    monkeypatch.setattr(
        udf_service,
        "_load_worker_data_for_udf",
        lambda **_: pd.DataFrame({"x": [1, 2]}),
    )


def _system_args(**extra):
    return SimpleNamespace(
        preprocessing=[],
        metadata={"x": {"is_categorical": False}},
        inputdata=None,
        add_dataset_variable=False,
        **extra,
    )


def _run(system_args):
    return udf_service.run_udf(
        request_id="req",
        udf_registry_key=UDF_KEY,
        kw_args={},
        system_args=system_args,
    )


def test_request_cannot_disable_the_minimum_row_check(monkeypatch, two_row_worker):
    monkeypatch.setattr(
        udf_service.exareme3_registry, "min_rows_check_required", lambda key: True
    )

    with pytest.raises(InsufficientDataError, match="minimum required is 3"):
        _run(_system_args(check_min_rows=False))


def test_registered_exemption_skips_the_minimum_row_check(monkeypatch, two_row_worker):
    monkeypatch.setattr(
        udf_service.exareme3_registry, "min_rows_check_required", lambda key: False
    )

    assert _run(_system_args()) == {"rows": 2}


def test_minimum_row_decision_is_looked_up_by_udf_registry_key(
    monkeypatch, two_row_worker
):
    looked_up = []

    def _required(key):
        looked_up.append(key)
        return False

    monkeypatch.setattr(
        udf_service.exareme3_registry, "min_rows_check_required", _required
    )

    _run(_system_args())

    assert looked_up == [UDF_KEY]


def test_run_udf_system_args_carry_no_minimum_row_flag():
    assert "check_min_rows" not in RunUdfSystemArgs.model_fields


def test_a_minimum_row_flag_in_the_payload_is_ignored():
    parsed = RunUdfSystemArgs.model_validate(
        {
            "inputdata": {"data_model": "dm:1", "datasets": ["ds"], "variables": ["x"]},
            "metadata": {},
            "add_dataset_variable": False,
            "check_min_rows": False,
        }
    )

    assert not hasattr(parsed, "check_min_rows")
