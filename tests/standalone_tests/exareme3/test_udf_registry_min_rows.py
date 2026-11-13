"""The minimum-row privacy exemption is declared where a UDF is registered.

A UDF is checked against the worker's minimum row count unless its
registration says otherwise. Only the UDFs that mask small groups themselves
may opt out.
"""

import pytest

from exaflow.algorithms.exareme3.outliers import outlier_report
from exaflow.algorithms.exareme3.statistics import describe
from exaflow.algorithms.exareme3.utils.registry import exareme3_registry
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf
from exaflow.algorithms.exareme3.utils.registry import get_udf_registry_key
from exaflow.worker.exareme3.udf import udf_service

RUNTIME_PACKAGE = "exaflow.algorithms.exareme3."


def _strict_udf(data):
    return {}


def _exempt_udf(data):
    return {}


@pytest.fixture
def register():
    keys = []

    def _register(func, **decorator_kwargs):
        exareme3_udf(**decorator_kwargs)(func)
        key = get_udf_registry_key(func)
        keys.append(key)
        return key

    yield _register

    for key in keys:
        exareme3_registry._registry.pop(key, None)  # noqa: SLF001


def test_udf_requires_the_minimum_row_check_by_default(register):
    key = register(_strict_udf)

    assert exareme3_registry.min_rows_check_required(key) is True


def test_udf_can_declare_a_minimum_row_exemption(register):
    key = register(_exempt_udf, check_min_rows=False)

    assert exareme3_registry.min_rows_check_required(key) is False


def test_unknown_udf_key_requires_the_minimum_row_check():
    assert exareme3_registry.min_rows_check_required("no_such_udf_key") is True


def test_only_describe_and_outlier_report_are_exempt():
    exempt = {
        key
        for key, info in exareme3_registry._registry.items()  # noqa: SLF001
        if info.func.__module__.startswith(RUNTIME_PACKAGE)
        and not exareme3_registry.min_rows_check_required(key)
    }

    assert exempt == {
        get_udf_registry_key(describe.local_step),
        get_udf_registry_key(outlier_report.local_step),
    }


@pytest.mark.parametrize(
    "algorithm_cls", [describe.Describe, outlier_report.OutlierReport]
)
def test_exempt_algorithms_no_longer_override_check_min_rows(algorithm_cls):
    assert "check_min_rows" not in vars(algorithm_cls)


def test_unknown_udf_key_lookup_returns_none():
    assert exareme3_registry.get_func("no_such_udf_key") is None


def test_worker_reports_an_unknown_udf_key_as_not_found():
    with pytest.raises(ImportError, match="not found"):
        udf_service._get_udf_or_raise("no_such_udf_key")
