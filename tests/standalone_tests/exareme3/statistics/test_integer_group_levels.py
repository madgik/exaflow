"""Integer groupA/groupB values select the text codes stored for those rows."""

from unittest.mock import Mock

import pandas as pd

from exaflow.algorithms.exareme3.statistics.ttest_independent import TTestIndependent
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from tests.standalone_tests.federated_algorithms.utils.dummy_agg_client import (
    DummyAggClient,
)


def test_integer_group_levels_select_text_coded_rows():
    data = pd.DataFrame(
        {"group": ["0", "0", "1", "1"], "value": [1.0, 3.0, 10.0, 14.0]}
    )

    class _Engine:
        def run_udf(self, func, add_dataset_variable, kw_args):
            return [func(DummyAggClient(), data, **kw_args)]

    algorithm = TTestIndependent(
        engine=_Engine(),
        logger=Mock(),
        inputdata=Inputdata(
            data_model="dementia:0.1",
            datasets=["dataset_a"],
            variables=["group", "value"],
        ),
        x=["group"],
        y=["value"],
        parameters={
            "alt_hypothesis": "two-sided",
            "alpha": 0.05,
            "groupA": 0,
            "groupB": 1,
        },
    )

    result = algorithm.run()

    assert result.mean_diff == -10.0
