"""Require a token on the controller's HTTP API when EXAFLOW_API_TOKEN is set.

The API has no user authentication: platform-backend makes the dataset checks, so
whatever else reaches it must not get in. Kubernetes probes carry no headers.
"""

import hmac
import os

from quart import jsonify
from quart import request

PROBE_PATHS = frozenset({"/livez", "/healthcheck"})


async def require_api_token():
    expected = os.environ.get("EXAFLOW_API_TOKEN", "")
    if not expected or request.path in PROBE_PATHS:
        return None
    presented = request.headers.get("Authorization", "")
    if hmac.compare_digest(presented.encode(), f"Bearer {expected}".encode()):
        return None
    return jsonify({"detail": "Unauthorized"}), 401
