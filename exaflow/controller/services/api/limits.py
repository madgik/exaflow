"""Request limits shared by the HTTP and AMQP transports."""

# Uvicorn does not limit request bodies. Retain the former HTTP limit for both.
MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
