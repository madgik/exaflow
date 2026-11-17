from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.inputdata_utils import fetch_data


def _inputdata(**overrides):
    values = {
        "data_model": "dementia:0.1",
        "datasets": ["2", "3"],
        "validation_datasets": None,
        "filters": None,
        "variables": ["lefthippocampus", "gender"],
    }
    values.update(overrides)
    return Inputdata(**values)


def test_fetch_data_selects_integer_dataset_codes_by_string_request(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "dataset,lefthippocampus,gender\n1,1.0,0\n2,2.0,1\n3,3.0,0\n3,4.0,1\n"
    )

    data = fetch_data(_inputdata(), [str(path)])

    assert sorted(data["lefthippocampus"]) == [2.0, 3.0, 4.0]
