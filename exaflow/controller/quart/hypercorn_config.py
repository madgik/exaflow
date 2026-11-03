from exaflow.controller.quart.loggers import loggers

workers = 1
# Drain in-flight HTTP requests for up to 120 s on SIGTERM, then run the
# shutdown hooks for shutdown_timeout. A worker RPC can start near the end of
# the drain and still run for worker_tasks_timeout, so the Helm grace period is
# worker_tasks_timeout + graceful_timeout + shutdown_timeout + 20 s.
graceful_timeout = 120
shutdown_timeout = 60
accesslog = "-"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
logconfig_dict = loggers
