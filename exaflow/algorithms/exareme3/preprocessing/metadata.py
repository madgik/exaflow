from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import Iterable


def promote_int_variables_to_real(
    *,
    metadata: Dict[str, dict],
    variables: Iterable[str],
) -> Dict[str, dict]:
    transformed_metadata = deepcopy(metadata)
    for variable in variables:
        variable_metadata = transformed_metadata.get(variable)
        if not variable_metadata:
            continue
        if variable_metadata.get("is_categorical"):
            continue
        if variable_metadata.get("sql_type") == "int":
            variable_metadata["sql_type"] = "real"
    return transformed_metadata
