from exaflow.controller.quart.loggers import loggers

workers = 1
# Gracefully drain in-flight requests for up to 120 s on SIGTERM. The Helm
# terminationGracePeriodSeconds must exceed this plus Hypercorn's shutdown_timeout
# (60 s default) so the shutdown hooks can complete before the pod is killed.
graceful_timeout = 120
accesslog = "-"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
logconfig_dict = loggers
