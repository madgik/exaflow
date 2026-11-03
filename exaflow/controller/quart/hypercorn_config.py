from exaflow.controller.quart.loggers import loggers

# In-process server. workers >= 1 runs a supervisor as PID 1 and the app in a
# child that does not observe the SIGTERM Kubernetes sends to the supervisor,
# so shutdown cannot cancel in-flight RPCs.
workers = 0
# Drain in-flight HTTP requests for up to 120 s on SIGTERM. Hypercorn then
# cancels the ones still running and waits for them to finish. Those requests
# cancel their worker and aggregation RPCs within shutdown_rpc_cancel_timeout
# and run aggregation cleanup within shutdown_cleanup_timeout, instead of
# waiting out worker_tasks_timeout. after_serving then has shutdown_timeout.
# The pod grace period is the sum of those bounds plus shutdown_grace_margin.
graceful_timeout = 120
shutdown_timeout = 60
shutdown_rpc_cancel_timeout = 5
shutdown_cleanup_timeout = 10
shutdown_grace_margin = 20
accesslog = "-"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
logconfig_dict = loggers
