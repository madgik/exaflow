import re
from enum import Enum

VARCHAR_SIZE = 500


class CDEType(str, Enum):
    """Variable type of a common data element (CDE)."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    BINARY = "binary"


CATEGORICAL_CDE_TYPES = frozenset({CDEType.NOMINAL, CDEType.ORDINAL, CDEType.BINARY})
# Every categorical CDE is stored as an integer (binary 0/1, nominal and ordinal
# 1..K, the dataset CDE 1..N, visitid included); the enumeration label keeps the
# readable name. Subject identifiers are discrete whole numbers, not categorical
# codes. Categorical CDEs without enumerations are rejected when data is loaded.
_INTEGER_CODE = re.compile(r"0|-?[1-9][0-9]*")


def _enumeration_codes(enumerations):
    if isinstance(enumerations, dict):
        return [str(code) for code in enumerations]
    return [
        str(item["code"] if isinstance(item, dict) else item)
        for item in enumerations or []
    ]


def is_integer_coded(code, cde_type, enumerations) -> bool:
    """True when the CDE `code` is an integer code column.

    `enumerations` is the CDE's enumerations (a code->label mapping or a list of
    {"code", "label"} items). Derived categorical columns whose codes are text
    (for example the categorical_column_creator output) are not integer coded.
    """
    if cde_type not in CATEGORICAL_CDE_TYPES:
        return False
    codes = _enumeration_codes(enumerations)
    return bool(codes) and all(_INTEGER_CODE.fullmatch(c) for c in codes)


class DType(Enum):
    """Members of DType represent data types in any language (python or sql).
    Each member has methods to_py and to_sql to convert its value to a concrete
    type.  There are also class methods from_py and from_sql to construct DType
    members from python/sql concrete types. The entire py2dtype and sql2dtype
    mappings are also provided as class methods for convenience."""

    INT = "INT"
    FLOAT = "FLOAT"
    STR = "STR"
    JSON = "JSON"
    BINARY = "BINARY"

    def to_py(self):
        mapping = self.dtype2py()
        return mapping[self]

    def to_sql(self):
        mapping = self.dtype2sql()
        return mapping[self]

    @classmethod
    def from_py(cls, pytype):
        mapping = cls.py2dtype()
        return mapping[pytype]

    @classmethod
    def from_sql(cls, sql_type):
        mapping = cls.sql2dtype()
        return mapping[sql_type.upper()]  # In SQL always print upper-case by convention

    @classmethod
    def from_cde(cls, cde_type):
        """Storage type of a CDE type: continuous is FLOAT, every other type
        (discrete and the categorical types) is INT. Categorical CDEs are integer
        codes; use `from_cde_variable` for derived text-coded categoricals."""
        if cde_type == CDEType.CONTINUOUS:
            return cls.FLOAT
        return cls.INT

    @classmethod
    def from_cde_variable(cls, code, cde_type, enumerations):
        """Like `from_cde`, but STR for derived categorical columns whose codes are
        text (for example the categorical_column_creator output)."""
        if cde_type in CATEGORICAL_CDE_TYPES and not is_integer_coded(
            code, cde_type, enumerations
        ):
            return cls.STR
        return cls.from_cde(cde_type)

    @classmethod
    def dtype2py(cls):
        mapping = {val: key for key, val in cls.py2dtype().items()}
        # Add here items for many-to-one mapping
        mapping[cls.JSON] = str
        return mapping

    @classmethod
    def dtype2sql(cls):
        return {
            cls.INT: "INT",
            cls.FLOAT: "DOUBLE",
            cls.STR: f"VARCHAR({VARCHAR_SIZE})",
            cls.BINARY: "BLOB",
        }

    @classmethod
    def py2dtype(cls):
        return {
            int: cls.INT,
            float: cls.FLOAT,
            str: cls.STR,
            bytes: cls.BINARY,
        }

    @classmethod
    def sql2dtype(cls):
        mapping = {val: key for key, val in cls.dtype2sql().items()}
        # Add here items for many-to-one mapping
        return mapping

    def __repr__(self):
        cls = type(self).__name__
        return f"{cls}.{self.name}"
