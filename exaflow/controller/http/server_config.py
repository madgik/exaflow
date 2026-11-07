"""Uvicorn settings for the controller.

The shutdown budget lives in ``exaflow/controller/services/controller_shutdown.py``.
"""

from exaflow.controller.services.controller_shutdown import GRACEFUL_TIMEOUT

# In-process server: with workers = 1 the process Kubernetes signals as PID 1
# is the one serving. workers > 1 starts a supervisor, and the app never sees
# that SIGTERM. `inv kill-controller` signals the same process.
workers = 1
# Uvicorn's timeout_graceful_shutdown: drain in-flight HTTP requests, then
# cancel the ones still running and run the lifespan shutdown hook, which
# cancels the analysis RPCs (see controller_shutdown.py).
graceful_timeout = GRACEFUL_TIMEOUT
