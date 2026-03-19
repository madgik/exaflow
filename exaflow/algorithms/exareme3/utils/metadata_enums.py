from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List


def add_ordered_enums(metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Attach an explicit enum order next to categorical metadata entries so order can
    be reconstructed after gRPC map serialization.
    """
    metadata_copy = deepcopy(metadata)
    for field in metadata_copy.values():
        if not isinstance(field, dict):
            continue
        enums = field.get("enumerations")
        if field.get("is_categorical") and isinstance(enums, dict):
            field["ordered_enums"] = list(enums.keys())
    return metadata_copy


def enforce_enum_order(
    metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Rebuild enumerations dicts using ordered_enums and remove the helper field.
    """
    metadata_copy = deepcopy(metadata)
    for field in metadata_copy.values():
        if not isinstance(field, dict):
            continue
        enums = field.get("enumerations")
        ordered = field.get("ordered_enums")
        if isinstance(enums, dict) and isinstance(ordered, list):
            field["enumerations"] = {
                code: enums[code] for code in ordered if code in enums
            }
            field.pop("ordered_enums", None)
    return metadata_copy


def get_enum_codes(metadata: Dict[str, Dict[str, Any]], variable: str) -> List[str]:
    enums = metadata.get(variable, {}).get("enumerations") or {}
    if isinstance(enums, dict):
        return list(enums.keys())
    return list(enums)
