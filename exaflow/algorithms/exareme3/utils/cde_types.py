"""Defensive checks that an algorithm received exactly the CDE type it expects.

The controller validates input slots against the algorithm specification before
an algorithm runs. These helpers repeat the check inside algorithms that
require binary or ordinal semantics, so a nominal variable is never silently
reinterpreted as binary or ordinal, even if the request validation is bypassed.
"""

from typing import Any
from typing import Dict
from typing import List

from exaflow.algorithms.exareme3.utils.metadata_enums import get_enum_codes
from exaflow.datatypes import CDEType
from exaflow.worker_communication import BadUserInput


def require_cde_type(
    metadata: Dict[str, Dict[str, Any]],
    variable: str,
    expected: CDEType,
    *,
    role: str,
) -> None:
    """Raise BadUserInput unless ``variable`` is declared with exactly ``expected``."""
    actual = (metadata.get(variable) or {}).get("type")
    if actual != expected:
        raise BadUserInput(
            f"The {role} '{variable}' has type '{_type_value(actual)}' but the "
            f"algorithm requires a '{expected.value}' variable. Variable types are "
            "matched exactly and are never converted."
        )
    if expected == CDEType.BINARY:
        levels = get_enum_codes(metadata, variable)
        if len(levels) != 2:
            raise BadUserInput(
                f"The {role} '{variable}' is declared binary but has {len(levels)} "
                "enumerations instead of exactly two."
            )


def declared_levels(metadata: Dict[str, Dict[str, Any]], variable: str) -> List[str]:
    """Enumeration codes of ``variable`` in the order the data model declares them."""
    return [str(code) for code in get_enum_codes(metadata, variable)]


def _type_value(cde_type: Any) -> str:
    return cde_type.value if isinstance(cde_type, CDEType) else str(cde_type)


def require_declared_level(
    metadata: Dict[str, Dict[str, Any]],
    variable: str,
    level: Any,
    *,
    parameter: str,
) -> None:
    """Raise BadUserInput unless ``level`` is one of the levels declared for ``variable``."""
    levels = declared_levels(metadata, variable)
    if str(level) not in levels:
        raise BadUserInput(
            f"Parameter '{parameter}' must be one of the levels declared for "
            f"'{variable}': {levels}. Got '{level}'."
        )
