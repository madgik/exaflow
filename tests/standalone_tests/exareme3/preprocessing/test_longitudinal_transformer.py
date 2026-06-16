import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from exaflow.algorithms.exareme3.preprocessing.longitudinal_transformer import (
    LongitudinalStrategy,
)
from exaflow.algorithms.exareme3.preprocessing.longitudinal_transformer import (
    LongitudinalTransformer,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.column_names import DATASET_COL
from exaflow.column_names import SUBJECT_ID_COL
from exaflow.column_names import VISIT_ID_COL
from exaflow.worker_communication import BadUserInput

STRATEGY_FIRST = LongitudinalStrategy.FIRST.value
STRATEGY_SECOND = LongitudinalStrategy.SECOND.value
STRATEGY_DIFF = LongitudinalStrategy.DIFF.value

BASE_METADATA = {
    "age": {"is_categorical": False, "label": "Age"},
    "weight": {"is_categorical": False, "label": "Weight"},
    "sex": {"is_categorical": True, "label": "Sex"},
    "score": {"is_categorical": False, "label": "Score"},
    "group": {"is_categorical": True, "label": "Group"},
}


def _make_transformer(strategies=None, visit1="BL", visit2="FL1"):
    return LongitudinalTransformer(
        params={
            "visit1": visit1,
            "visit2": visit2,
            "strategies": strategies or {},
        }
    )


def _make_inputdata(x=None, y=None):
    return Inputdata(
        data_model="dm:0.1",
        datasets=["d1"],
        x=x,
        y=y,
    )


def _base_longitudinal_df(include_dataset=True):
    rows = [
        # dataset d1
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s1",
            VISIT_ID_COL: "BL",
            "age": 10,
            "weight": 70,
            "sex": "F",
            "score": 1,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s1",
            VISIT_ID_COL: "FL1",
            "age": 11,
            "weight": 72,
            "sex": "F",
            "score": 3,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s1",
            VISIT_ID_COL: "FL2",
            "age": 12,
            "weight": 73,
            "sex": "F",
            "score": 4,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s2",
            VISIT_ID_COL: "BL",
            "age": 20,
            "weight": 80,
            "sex": "M",
            "score": 5,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s2",
            VISIT_ID_COL: "FL1",
            "age": 21,
            "weight": 78,
            "sex": "M",
            "score": 4,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s3",
            VISIT_ID_COL: "BL",
            "age": 30,
            "weight": 90,
            "sex": "F",
            "score": 2,
        },
        {
            DATASET_COL: "d1",
            SUBJECT_ID_COL: "s4",
            VISIT_ID_COL: "FL1",
            "age": 41,
            "weight": 100,
            "sex": "M",
            "score": 6,
        },
        # same subject id in another dataset
        {
            DATASET_COL: "d2",
            SUBJECT_ID_COL: "s1",
            VISIT_ID_COL: "BL",
            "age": 15,
            "weight": 60,
            "sex": "M",
            "score": 10,
        },
        {
            DATASET_COL: "d2",
            SUBJECT_ID_COL: "s1",
            VISIT_ID_COL: "FL1",
            "age": 16,
            "weight": 62,
            "sex": "M",
            "score": 11,
        },
    ]
    df = pd.DataFrame(rows)
    if not include_dataset:
        df = df[df[DATASET_COL] == "d1"].drop(columns=[DATASET_COL]).copy()
    return df


def _expected_df(rows):
    return pd.DataFrame(rows)


def _assert_frames_equal(actual, expected):
    sort_cols = [c for c in [DATASET_COL, SUBJECT_ID_COL] if c in actual.columns]
    if sort_cols:
        actual = actual.sort_values(sort_cols).reset_index(drop=True)
        expected = expected.sort_values(sort_cols).reset_index(drop=True)
    actual = actual[expected.columns]
    pdt.assert_frame_equal(actual, expected, check_dtype=False)


def test_get_specification_has_expected_shape():
    spec = LongitudinalTransformer.get_specification()

    assert spec.name == "longitudinal_transformer"
    assert spec.enabled is True
    assert set(spec.parameters.keys()) == {"visit1", "visit2", "strategies"}
    assert spec.parameters["strategies"].dict_values_enums.source == [
        STRATEGY_DIFF,
        STRATEGY_FIRST,
        STRATEGY_SECOND,
    ]


def test_required_input_variables_returns_fixed_required_columns():
    assert LongitudinalTransformer.required_input_variables() == [
        DATASET_COL,
        SUBJECT_ID_COL,
        VISIT_ID_COL,
    ]


@pytest.mark.parametrize(
    "strategies,x,y,expected_x,expected_y",
    [
        ({"age": STRATEGY_FIRST}, ["age"], [], ["age"], []),
        ({"age": STRATEGY_SECOND}, ["age"], [], ["age"], []),
        ({"age": STRATEGY_DIFF}, ["age"], [], ["age"], []),
        ({"score": STRATEGY_DIFF}, [], ["score"], [], ["score"]),
        (
            {"age": STRATEGY_DIFF, "weight": STRATEGY_FIRST, "sex": STRATEGY_SECOND},
            ["age", "weight"],
            ["sex"],
            ["age", "weight"],
            ["sex"],
        ),
        ({"age": STRATEGY_DIFF}, ["unknown"], [], ["unknown"], []),
        ({"age": STRATEGY_DIFF}, ["age", "age"], [], ["age", "age"], []),
        ({"score": STRATEGY_DIFF}, [], ["score", "age"], [], ["score", "age"]),
        ({"age": STRATEGY_FIRST}, ["age", "score"], [], ["age", "score"], []),
        ({"age": STRATEGY_DIFF}, ["age"], ["age"], ["age"], ["age"]),
    ],
)
def test_transform_inputdata_variables_10_cases(
    strategies, x, y, expected_x, expected_y
):
    transformer = _make_transformer(strategies=strategies)

    actual_x, actual_y = transformer.transform_inputdata_variables(x=x, y=y)

    assert actual_x == expected_x
    assert actual_y == expected_y


@pytest.mark.parametrize(
    "strategies,metadata,expected",
    [
        ({"age": STRATEGY_FIRST}, BASE_METADATA, BASE_METADATA),
        ({"age": STRATEGY_SECOND}, BASE_METADATA, BASE_METADATA),
        (
            {"age": STRATEGY_DIFF},
            BASE_METADATA,
            {
                "weight": {"is_categorical": False, "label": "Weight"},
                "sex": {"is_categorical": True, "label": "Sex"},
                "score": {"is_categorical": False, "label": "Score"},
                "group": {"is_categorical": True, "label": "Group"},
                "age": {"is_categorical": False, "label": "Age"},
            },
        ),
        (
            {"age": STRATEGY_DIFF, "sex": STRATEGY_FIRST},
            BASE_METADATA,
            {
                "weight": {"is_categorical": False, "label": "Weight"},
                "sex": {"is_categorical": True, "label": "Sex"},
                "score": {"is_categorical": False, "label": "Score"},
                "group": {"is_categorical": True, "label": "Group"},
                "age": {"is_categorical": False, "label": "Age"},
            },
        ),
        (
            {"age": STRATEGY_DIFF, "weight": STRATEGY_DIFF},
            BASE_METADATA,
            {
                "sex": {"is_categorical": True, "label": "Sex"},
                "score": {"is_categorical": False, "label": "Score"},
                "group": {"is_categorical": True, "label": "Group"},
                "age": {"is_categorical": False, "label": "Age"},
                "weight": {"is_categorical": False, "label": "Weight"},
            },
        ),
        ({"unknown": STRATEGY_DIFF}, BASE_METADATA, BASE_METADATA),
        (
            {"score": STRATEGY_FIRST},
            {**BASE_METADATA, "extra": {"is_categorical": False, "label": "Extra"}},
            {**BASE_METADATA, "extra": {"is_categorical": False, "label": "Extra"}},
        ),
        ({"age": STRATEGY_DIFF}, {}, {}),
        ({}, BASE_METADATA, BASE_METADATA),
        (
            {"group": STRATEGY_SECOND, "score": STRATEGY_DIFF},
            BASE_METADATA,
            {
                "age": {"is_categorical": False, "label": "Age"},
                "weight": {"is_categorical": False, "label": "Weight"},
                "sex": {"is_categorical": True, "label": "Sex"},
                "group": {"is_categorical": True, "label": "Group"},
                "score": {"is_categorical": False, "label": "Score"},
            },
        ),
    ],
)
def test_transform_metadata_10_cases(strategies, metadata, expected):
    transformer = _make_transformer(strategies=strategies)

    actual = transformer.transform_metadata(metadata=metadata)

    assert actual == expected


def test_transform_metadata_returns_deepcopy():
    metadata = {"age": {"is_categorical": False, "label": "Age"}}
    transformer = _make_transformer(strategies={})

    transformed = transformer.transform_metadata(metadata=metadata)
    transformed["age"]["label"] = "Changed"

    assert metadata["age"]["label"] == "Age"


@pytest.mark.parametrize(
    "strategies,data,expected",
    [
        (
            {"age": STRATEGY_FIRST},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "age": 10},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "age": 20},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "age": 15},
                ]
            ),
        ),
        (
            {"age": STRATEGY_SECOND},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "age": 11},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "age": 21},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "age": 16},
                ]
            ),
        ),
        (
            {"age": STRATEGY_DIFF},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "age": 1},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "age": 1},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "age": 1},
                ]
            ),
        ),
        (
            {"age": STRATEGY_DIFF, "weight": STRATEGY_SECOND},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {
                        SUBJECT_ID_COL: "s1",
                        DATASET_COL: "d1",
                        "age": 1,
                        "weight": 72,
                    },
                    {
                        SUBJECT_ID_COL: "s2",
                        DATASET_COL: "d1",
                        "age": 1,
                        "weight": 78,
                    },
                    {
                        SUBJECT_ID_COL: "s1",
                        DATASET_COL: "d2",
                        "age": 1,
                        "weight": 62,
                    },
                ]
            ),
        ),
        (
            {"age": STRATEGY_DIFF},
            _base_longitudinal_df(include_dataset=False),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", "age": 1},
                    {SUBJECT_ID_COL: "s2", "age": 1},
                ]
            ),
        ),
        (
            {"score": STRATEGY_SECOND},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "score": 3},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "score": 4},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "score": 11},
                ]
            ),
        ),
        (
            {"weight": STRATEGY_FIRST},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "weight": 70},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "weight": 80},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "weight": 60},
                ]
            ),
        ),
        (
            {"weight": STRATEGY_DIFF},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "weight": 2},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "weight": -2},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "weight": 2},
                ]
            ),
        ),
        (
            {"weight": STRATEGY_DIFF},
            _base_longitudinal_df(include_dataset=True).assign(
                weight=lambda df: np.where(
                    (df[DATASET_COL] == "d1")
                    & (df[SUBJECT_ID_COL] == "s2")
                    & (df[VISIT_ID_COL] == "FL1"),
                    np.nan,
                    df["weight"],
                )
            ),
            _expected_df(
                [
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "weight": 2.0},
                    {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "weight": np.nan},
                    {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "weight": 2.0},
                ]
            ),
        ),
        (
            {"weight": STRATEGY_FIRST, "age": STRATEGY_DIFF, "score": STRATEGY_SECOND},
            _base_longitudinal_df(include_dataset=True),
            _expected_df(
                [
                    {
                        SUBJECT_ID_COL: "s1",
                        DATASET_COL: "d1",
                        "weight": 70,
                        "age": 1,
                        "score": 3,
                    },
                    {
                        SUBJECT_ID_COL: "s2",
                        DATASET_COL: "d1",
                        "weight": 80,
                        "age": 1,
                        "score": 4,
                    },
                    {
                        SUBJECT_ID_COL: "s1",
                        DATASET_COL: "d2",
                        "weight": 60,
                        "age": 1,
                        "score": 11,
                    },
                ]
            ),
        ),
    ],
)
def test_transform_data_10_cases(strategies, data, expected):
    transformer = _make_transformer(strategies=strategies)

    actual = transformer.transform_data(data=data)

    _assert_frames_equal(actual, expected)


def test_transform_data_raises_when_required_columns_are_missing():
    transformer = _make_transformer(strategies={"age": STRATEGY_DIFF})
    data = pd.DataFrame(
        [
            {SUBJECT_ID_COL: "s1", "age": 1},
            {SUBJECT_ID_COL: "s1", "age": 2},
        ]
    )

    with pytest.raises(BadUserInput, match="Missing required columns"):
        transformer.transform_data(data=data)


def test_transform_data_and_metadata_combines_both_transformations():
    transformer = _make_transformer(strategies={"age": STRATEGY_DIFF})
    data = _base_longitudinal_df(include_dataset=True)
    metadata = {"age": {"is_categorical": False, "label": "Age"}}

    transformed_data, transformed_metadata = transformer.transform_data_and_metadata(
        data=data,
        metadata=metadata,
    )

    expected_data = _expected_df(
        [
            {SUBJECT_ID_COL: "s1", DATASET_COL: "d1", "age": 1},
            {SUBJECT_ID_COL: "s2", DATASET_COL: "d1", "age": 1},
            {SUBJECT_ID_COL: "s1", DATASET_COL: "d2", "age": 1},
        ]
    )
    expected_metadata = {"age": {"is_categorical": False, "label": "Age"}}

    _assert_frames_equal(transformed_data, expected_data)
    assert transformed_metadata == expected_metadata


@pytest.mark.parametrize(
    "params,inputdata,metadata,error_match",
    [
        (
            {"visit1": "", "visit2": "FL1", "strategies": {"age": STRATEGY_FIRST}},
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "Both 'visit1' and 'visit2' parameters are required",
        ),
        (
            {"visit1": "BL", "visit2": "", "strategies": {"age": STRATEGY_FIRST}},
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "Both 'visit1' and 'visit2' parameters are required",
        ),
        (
            {"visit1": "BL", "visit2": "BL", "strategies": {"age": STRATEGY_FIRST}},
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "must be different",
        ),
        (
            {"visit1": "BL", "visit2": "FL1", "strategies": "not-a-dict"},
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "'strategies' must be a dictionary",
        ),
        (
            {"visit1": "BL", "visit2": "FL1", "strategies": {"age": STRATEGY_FIRST}},
            _make_inputdata(x=["age", "weight"], y=[]),
            BASE_METADATA,
            "missing",
        ),
        (
            {
                "visit1": "BL",
                "visit2": "FL1",
                "strategies": {"age": STRATEGY_FIRST, "weight": STRATEGY_FIRST},
            },
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "extra",
        ),
        (
            {"visit1": "BL", "visit2": "FL1", "strategies": {"age": "invalid"}},
            _make_inputdata(x=["age"], y=[]),
            BASE_METADATA,
            "Invalid strategy values",
        ),
        (
            {"visit1": "BL", "visit2": "FL1", "strategies": {"sex": STRATEGY_DIFF}},
            _make_inputdata(x=["sex"], y=[]),
            BASE_METADATA,
            "Cannot take the difference for the nominal variable",
        ),
    ],
)
def test_validate_params_raises_for_invalid_configs(
    params, inputdata, metadata, error_match
):
    transformer = LongitudinalTransformer(params=params)

    with pytest.raises(BadUserInput, match=error_match):
        transformer.validate_params(inputdata=inputdata, metadata=metadata)


def test_validate_params_accepts_valid_configuration():
    transformer = _make_transformer(
        strategies={
            "age": STRATEGY_DIFF,
            "weight": STRATEGY_FIRST,
            "sex": STRATEGY_SECOND,
        }
    )
    inputdata = _make_inputdata(x=["weight", "age"], y=["sex"])

    transformer.validate_params(inputdata=inputdata, metadata=BASE_METADATA)
