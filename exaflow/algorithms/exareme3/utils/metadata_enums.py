from typing import Any
from typing import Dict
from typing import List


def get_enum_codes(metadata: Dict[str, Dict[str, Any]], variable: str) -> List[str]:
    enums = metadata.get(variable, {}).get("enumerations") or {}
    if isinstance(enums, dict):
        return list(enums.keys())
    return list(enums)
