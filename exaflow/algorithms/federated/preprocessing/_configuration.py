"""Shared public-configuration checks before feature-dependent rounds."""

import json
from hashlib import sha256

import numpy as np

from exaflow.algorithms.federated.utils import BadInputError


def check_common_configuration(configuration, agg_client):
    """Compare metadata/settings fingerprints, never observed data values.

    All participants enter the same fixed-size check. SHA-256 words are exactly
    representable in float64; observed categories must never enter this hash.
    """
    encoded = json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
    fingerprint = np.frombuffer(sha256(encoded).digest(), dtype=">u4").astype(float)
    minimum = np.asarray(agg_client.min(fingerprint))
    maximum = np.asarray(agg_client.max(fingerprint))
    if not np.array_equal(minimum, maximum):
        raise BadInputError(
            "Participants must use the same ordered metadata and settings."
        )
