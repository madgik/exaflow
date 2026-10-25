from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.statistics.anova_oneway import FederatedAnovaOneWay
from exaflow.algorithms.federated.statistics.anova_twoway import FederatedAnovaTwoWay
from exaflow.algorithms.federated.statistics.binned_mann_whitney_u_test import (
    FederatedBinnedMannWhitneyUTest,
)
from exaflow.algorithms.federated.statistics.cross_tabular_table import CrossTabTable
from exaflow.algorithms.federated.statistics.describe import FederatedDescribe
from exaflow.algorithms.federated.statistics.histogram import CategoricalHistogram
from exaflow.algorithms.federated.statistics.histogram import SimpleHistogram
from exaflow.algorithms.federated.statistics.histogram import WilkinsonHistogram
from exaflow.algorithms.federated.statistics.pearson_correlation import (
    FederatedPearsonCorrelation,
)
from exaflow.algorithms.federated.statistics.percentile import Percentile
from exaflow.algorithms.federated.statistics.primitive_statistics import (
    PrimitiveStatistics,
)
from exaflow.algorithms.federated.statistics.standardized_mean_difference import (
    FederatedStandardizedMeanDifference,
)
from exaflow.algorithms.federated.statistics.ttest_independent import (
    FederatedTTestIndependent,
)
from exaflow.algorithms.federated.statistics.ttest_onesample import (
    FederatedTTestOneSample,
)
from exaflow.algorithms.federated.statistics.ttest_paired import FederatedTTestPaired
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


def compute_ttest(kind, client, **parameters):
    settings = {"alpha": 0.05, "alternative": "two-sided", **parameters}
    aggregator = NumpyAggregator(client)
    if kind == "one":
        return FederatedTTestOneSample(aggregator).compute(
            sample=[1, 2], mu=0, **settings
        )
    if kind == "paired":
        return FederatedTTestPaired(aggregator).compute(
            sample_x=[1, 2], sample_y=[2, 3], **settings
        )
    return FederatedTTestIndependent(aggregator).compute(
        data=pd.DataFrame({"group": [0, 1], "value": [1, 2]}),
        group_var="group",
        value_var="value",
        group_a=0,
        group_b=1,
        **settings,
    )


@pytest.mark.parametrize("kind", ["one", "paired", "independent"])
@pytest.mark.parametrize(
    "parameters",
    [
        {"alpha": np.nan},
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alternative": "private-value"},
    ],
)
def test_ttest_parameters_fail_before_aggregation(kind, parameters):
    client = Mock()
    with pytest.raises(BadInputError) as error:
        compute_ttest(kind, client, **parameters)
    assert "private-value" not in str(error.value)
    assert client.mock_calls == []


@pytest.mark.parametrize("kind", ["one", "paired", "independent"])
@pytest.mark.parametrize("alpha", [0, 1])
def test_confidence_endpoints_retain_scipy_semantics(kind, alpha):
    client = Mock()
    client.sum.side_effect = RuntimeError("Validation passed")
    with pytest.raises(RuntimeError, match="Validation passed"):
        compute_ttest(kind, client, alpha=alpha)


@pytest.mark.parametrize("kind", [SimpleHistogram, WilkinsonHistogram])
@pytest.mark.parametrize("bins", [0, -1, 1.5, np.nan])
def test_histogram_bins_fail_before_aggregation(kind, bins):
    aggregator = Mock()
    with pytest.raises(BadInputError):
        kind(aggregator).compute(np.array([1, 2]), bins)
    assert aggregator.mock_calls == []


@pytest.mark.parametrize(
    "kind", [SimpleHistogram, CategoricalHistogram, WilkinsonHistogram]
)
def test_histogram_rank_errors_remain_local(kind):
    aggregator = Mock()
    args = () if kind is CategoricalHistogram else (2,)
    with pytest.raises(BadInputError, match="must be a 1D array"):
        kind(aggregator).compute(np.ones((2, 2)), *args)
    assert aggregator.mock_calls == []


@pytest.mark.parametrize("bounds", [(2, 1), (0, np.inf), (0, np.nan), [0]])
def test_histogram_explicit_bounds_fail_before_aggregation(bounds):
    aggregator = Mock()
    with pytest.raises(BadInputError):
        SimpleHistogram(aggregator).compute(np.array([1, 2]), 2, bounds=bounds)
    assert aggregator.mock_calls == []


@pytest.mark.parametrize("q", [-0.1, 1.1, np.nan, np.inf])
def test_percentile_probability_errors_remain_local(q):
    aggregator = Mock()
    with pytest.raises(BadInputError):
        Percentile(aggregator).compute(pd.Series([1, 2]), q)
    assert aggregator.mock_calls == []


def test_oneway_group_labels_match_before_aggregation():
    client = Mock()
    with pytest.raises(BadInputError, match="category label"):
        FederatedAnovaOneWay(client).fit([[1, 2]], ["a", "b"])
    assert client.mock_calls == []


def test_twoway_ss_type_is_checked_before_model_fitting():
    client = Mock()
    with pytest.raises(BadInputError, match="sstype"):
        FederatedAnovaTwoWay(client, sstype=3).fit(
            data=pd.DataFrame({"y": [1], "a": [0], "b": [0]}),
            y="y",
            x1="a",
            x2="b",
            levels_a=[0, 1],
            levels_b=[0, 1],
        )
    assert client.mock_calls == []


@pytest.mark.parametrize(
    "parameters", [{"alternative": "private-value"}, {"num_bins": 1}, {"num_bins": 2.5}]
)
def test_binned_test_parameters_fail_without_collectives(parameters):
    aggregator = Mock()
    with pytest.raises(BadInputError):
        FederatedBinnedMannWhitneyUTest(aggregator).compute(
            np.array([1]), np.array([2]), **parameters
        )
    assert aggregator.mock_calls == []


@pytest.mark.parametrize("method", ["mean", "variance", "covariance"])
def test_primitive_ddof_nonfinite_values_fail_locally(method):
    aggregator = Mock()
    args = (np.array([1, 2]),) * (2 if method == "covariance" else 1)
    with pytest.raises(BadInputError, match="ddof"):
        getattr(PrimitiveStatistics(aggregator), method)(*args, ddof=np.nan)
    assert aggregator.mock_calls == []


def test_cross_tab_checks_columns_without_changing_category_policy():
    aggregator = Mock()
    with pytest.raises(BadInputError, match="required columns"):
        CrossTabTable(aggregator).compute(pd.DataFrame({"a": [None]}), "a", "b")
    assert aggregator.mock_calls == []


def test_pearson_checks_alpha_before_aggregation():
    client = Mock()
    with pytest.raises(BadInputError, match="alpha"):
        FederatedPearsonCorrelation(client).corrcoef(
            data=pd.DataFrame({"a": [1]}), x_vars=["a"], y_vars=["a"], alpha=np.inf
        )
    assert client.mock_calls == []


def test_describe_checks_frame_without_aggregating():
    client = Mock()
    with pytest.raises(BadInputError, match="DataFrame"):
        FederatedDescribe(client).describe(
            data=None,
            numerical_vars=[],
            nominal_vars=[],
            min_row_count=1,
            nominal_levels={},
        )
    assert client.mock_calls == []


def test_smd_group_columns_are_checked_before_sql_execution():
    client = Mock()
    with pytest.raises(BadInputError, match="required columns"):
        FederatedStandardizedMeanDifference(client).pairwise_by_group(
            data=pd.DataFrame({"x": [1]}), value_var="x", group_var="missing"
        )
    assert client.mock_calls == []


@pytest.mark.parametrize("kind", ["percentile", "binned", "cross_tab"])
def test_unconverted_boolean_options_are_checked_before_aggregation(kind):
    aggregator = Mock()
    with pytest.raises(BadInputError, match="boolean"):
        if kind == "percentile":
            Percentile(aggregator).compute(pd.Series([1, 2]), 0.5, is_integer="false")
        elif kind == "binned":
            FederatedBinnedMannWhitneyUTest(aggregator).compute(
                np.array([1]), np.array([2]), use_continuity="false"
            )
        else:
            CrossTabTable(aggregator).compute(
                pd.DataFrame({"x": [0], "y": [1]}), "x", "y", dropna="false"
            )
    assert aggregator.mock_calls == []
