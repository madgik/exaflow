import pytest
from pydantic import ValidationError

from exaflow.controller.services.api.analysis_request_dtos import AnalysisRequestDTO
from tests.standalone_tests.controller.analysis_support import ANALYSIS_BODY
from tests.standalone_tests.controller.analysis_support import INVALID_REQUEST_IDS
from tests.standalone_tests.controller.analysis_support import VALID_REQUEST_IDS


@pytest.mark.parametrize("request_id", VALID_REQUEST_IDS)
def test_request_ids_are_preserved(request_id):
    request = AnalysisRequestDTO.model_validate(
        {**ANALYSIS_BODY, "request_id": request_id}
    )
    assert request.request_id == request_id


@pytest.mark.parametrize("request_id", INVALID_REQUEST_IDS)
def test_invalid_request_ids_are_rejected(request_id):
    with pytest.raises(ValidationError) as exc:
        AnalysisRequestDTO.model_validate({**ANALYSIS_BODY, "request_id": request_id})
    assert exc.value.errors()[0]["loc"] == ("request_id",)


@pytest.mark.parametrize("fields", [{}, {"request_id": None}])
def test_request_id_can_be_omitted_or_null(fields):
    assert (
        AnalysisRequestDTO.model_validate({**ANALYSIS_BODY, **fields}).request_id
        is None
    )
