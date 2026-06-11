import pandas as pd
import pytest

from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import (
    display_grouping_var,
)
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import get_group_ids
from exaflow.algorithms.exareme3.mixed_effects.mixed_effects_common import (
    split_grouping_var,
)
from exaflow.worker_communication import BadUserInput


def test_get_group_ids_preserves_single_grouping_var():
    data = pd.DataFrame({"dataset": ["ppmi0", "ppmi1"]})

    group_ids = get_group_ids(data, "dataset")

    assert group_ids.tolist() == ["ppmi0", "ppmi1"]


def test_get_group_ids_builds_composite_grouping_key():
    data = pd.DataFrame(
        {
            "dataset": ["ppmi0", "ppmi0", "ppmi1"],
            "gender": ["F", "M", "F"],
        }
    )

    group_ids = get_group_ids(data, ["dataset", "gender"])

    assert group_ids.tolist() == [
        "dataset=ppmi0|gender=F",
        "dataset=ppmi0|gender=M",
        "dataset=ppmi1|gender=F",
    ]


def test_split_grouping_var_removes_all_grouping_vars_from_fixed_effects():
    metadata = {
        "lefthippocampus": {"is_categorical": False},
        "agegroup": {"is_categorical": True},
        "dataset": {"is_categorical": True},
        "gender": {"is_categorical": True},
    }

    categorical_vars, numerical_vars = split_grouping_var(
        ["lefthippocampus", "agegroup", "dataset", "gender"],
        ["dataset", "gender"],
        metadata,
    )

    assert categorical_vars == ["agegroup"]
    assert numerical_vars == ["lefthippocampus"]


def test_display_grouping_var_keeps_single_value_backward_compatible():
    assert display_grouping_var(["dataset"]) == "dataset"
    assert display_grouping_var(["dataset", "gender"]) == ["dataset", "gender"]


def test_split_grouping_var_rejects_no_fixed_effects():
    metadata = {
        "dataset": {"is_categorical": True},
        "gender": {"is_categorical": True},
    }

    with pytest.raises(BadUserInput):
        split_grouping_var(["dataset", "gender"], ["dataset", "gender"], metadata)
