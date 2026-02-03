from enum import Enum


class AggregationType(Enum):
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"
    UNION = "UNION"

    def __str__(self):
        return self.name
