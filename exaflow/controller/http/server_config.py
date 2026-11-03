"""Shutdown budget shared by Uvicorn, Helm, and the controller process.

``workers`` is 1 because Uvicorn serves in-process at that value. ``workers``
greater than 1 starts a supervisor as PID 1, and Kubernetes' SIGTERM never
reaches the app.

``graceful_timeout`` is Uvicorn's ``timeout_graceful_shutdown``: drain
in-flight HTTP requests, then cancel the ones still running. Uvicorn has no
separate hook timeout. Lifespan shutdown runs after that cancel. The app
bounds RPC cancel, aggregation cleanup, and the worker-landscape join.
``shutdown_timeout`` stays in the pod grace so those hooks are not SIGKILLed.
"""

workers = 1
graceful_timeout = 120
shutdown_timeout = 60
shutdown_rpc_cancel_timeout = 5
shutdown_cleanup_timeout = 10
shutdown_grace_margin = 20
