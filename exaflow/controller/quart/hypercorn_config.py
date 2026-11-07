from exaflow.controller.http.loggers import loggers
from exaflow.controller.services.controller_shutdown import GRACEFUL_TIMEOUT
from exaflow.controller.services.controller_shutdown import SHUTDOWN_TIMEOUT

# In-process server: the process Kubernetes signals as PID 1 is the one serving,
# and `inv kill-controller` signals the same process.
workers = 0
# Drain in-flight HTTP requests for GRACEFUL_TIMEOUT after SIGTERM, then run the
# lifespan shutdown hook, which cancels the analysis RPCs that are still
# running (see controller_shutdown.py). The pod grace period is derived from
# the budget in that module.
graceful_timeout = GRACEFUL_TIMEOUT
shutdown_timeout = SHUTDOWN_TIMEOUT
accesslog = "-"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
logconfig_dict = loggers
