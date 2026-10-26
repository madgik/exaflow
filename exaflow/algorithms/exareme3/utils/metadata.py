from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import Iterable

from exaflow.datatypes import CDEType


def promote_discrete_variables_to_continuous(
    *,
    metadata: Dict[str, dict],
    variables: Iterable[str],
) -> Dict[str, dict]:
    transformed_metadata = deepcopy(metadata)
    for variable in variables:
        variable_metadata = transformed_metadata.get(variable)
        if not variable_metadata:
            continue
        if variable_metadata.get("type") == CDEType.DISCRETE:
            variable_metadata["type"] = CDEType.CONTINUOUS.value
    return transformed_metadata
