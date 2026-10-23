from enum import Enum


class AggregationType(str, Enum):
    """Aggregation operations shared by the aggregation server and its clients.

    A `str` enum, so members compare equal to the `aggregation_type` strings
    sent over gRPC.
    """

    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"
    UNION = "UNION"

    def __str__(self):
        return self.name
