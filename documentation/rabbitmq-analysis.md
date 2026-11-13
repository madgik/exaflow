# Analysis submission through RabbitMQ

RabbitMQ is an optional second transport for the existing analysis service.
`POST /analysis`, metadata routes, HTTP errors, and algorithm result shapes remain
unchanged. There is no job database or HTTP result polling endpoint.

## Execution and identifiers

The FastAPI lifespan starts one asynchronous consumer after the first worker
landscape refresh. HTTP and AMQP share an asyncio loop; blocking native analysis
work stays in the dedicated analysis executor. The landscape refresher retains
its existing background thread. Broker connection failures do not prevent HTTP
startup or operation.

`request_id` identifies the logical analysis and is used for tracing across the
controller and workers. The AMQP `correlation_id` must equal it. An omitted body
`request_id` is populated from the correlation ID; conflicting values are rejected.
Use the same ID only for the same logical request. Additional collision detection
is deliberately outside this transport's contract.

Each strategy generates a UUID `context_id` for its execution attempt. UDF system
arguments carry that context to workers. Aggregation Configure, Aggregate,
Unregister, Cleanup, and cancellation tombstones use the context ID as their key;
aggregation logs retain both IDs and Cleanup still enforces its ownership token.
Before this change context IDs appeared only in experiment-start logs and were
stored, unused, in the flow engine.

**Compatibility:** the aggregation RPC and UDF system-argument contracts change.
Upgrade controller, workers, and aggregation server together, even when AMQP is
disabled. There is no fallback for old clients omitting the execution context.
Regenerate bindings using `exaflow/protos/README.md`.

## Message contract

Publish persistent JSON messages with publisher confirms and mandatory routing to
`exaflow.analysis.requests` (configurable). The body is the existing
`AnalysisRequestDTO`, including `inputdata`, `preprocessing`, `algorithm`, and any
existing flags. Required AMQP properties are:

| Property | Meaning |
| --- | --- |
| `correlation_id` | Logical request ID, preserved on replies. |
| `reply_to` | Name of a durable queue provisioned by the producer. |
| `content_type` | Producers should set `application/json`. |
| `delivery_mode` | Producers must use persistent delivery for restart recovery. |

The consumer parses the DTO, waits for shared analysis capacity, then validates
against current worker metadata and constructs the execution strategy. HTTP
keeps its original validation/admission order. Both transports share
`max_concurrent_experiments`; there are no reserved slots. Other HTTP endpoints
do not acquire analysis slots. Queue prefetch and active delivery handlers are
bounded by that same limit, also across reconnects.

Replies contain parsed result JSON, rather than a JSON string inside JSON:

```json
{"status": "succeeded", "result": {"example": "algorithm-specific result"}}
```

```json
{"status": "failed", "error": {"code": "invalid_request", "message": "Request does not match the analysis schema or identifiers."}}
```

Protocol error codes include `invalid_request`, `request_too_large`,
`result_too_large`, `insufficient_data`, and `internal_error`. Recognized user-input
and method errors use their existing exception class name, for example
`BadRequest`. Unexpected errors return a generic message without exception details.
Validation and algorithm errors are terminal replies, not automatic retry jobs.

## Delivery, recovery, and shutdown

A request is acknowledged only after a persistent reply has been routed and
confirmed by the broker. Missing routing metadata and unroutable replies cause
the original request to be dead-lettered. Consumer declarations create bounded,
durable quorum request and dead-letter queues, with at-least-once dead-lettering.
A full queue rejects publication; producers must handle negative confirms.

Processing is **at least once**. A controller crash after reply publication but
before request acknowledgement can repeat the analysis and its reply. Clients
must deduplicate by correlation ID; recomputation can yield different results for
randomized algorithms or changing datasets. No completed-result database exists.
Active duplicate deliveries in a surviving controller await the existing attempt.
Fresh execution contexts prevent stale aggregation calls from affecting retries.

The single-controller consumer checks subscription registration once per second.
If the broker cancels it without closing the connection, zero registered consumers
interrupts the iterator, marks consumption unavailable, and restarts the connection
and subscription.

Connection failures close the affected delivery channel and reconnect with
exponential backoff from 1 to 30 seconds. Deliveries belong to their original
channel and are never acknowledged through a replacement channel.

At lifespan shutdown, intake stops, active deliveries drain for up to 30 seconds,
and existing shutdown logic cancels RPCs and performs aggregation cleanup. Then
remaining AMQP tasks and the connection close. Unfinished requests remain
unacknowledged for redelivery. The shutdown phases fit the existing 60-second
lifespan budget with default settings; cancellation does not forcibly terminate
Python threads. Uvicorn's existing HTTP drain precedes the lifespan hook.

## Configuration and provisioning

AMQP defaults to disabled. Provision an external RabbitMQ 4.x broker; the normal
Helm chart does not install it. Use one controller replica and one Uvicorn process.
Provide broker credentials and queue permissions through the existing settings
layer. Restrict producer access to the appropriate request and reply queues;
use a dedicated vhost and `amqps://` when TLS is required by the deployment.

For local configuration, add `[rabbitmq]` to the deployment TOML, enable it, and
set `url = "${RABBITMQ_URL:-}"`. Run the existing config-generation task.
Credentials are interpolated by the controller loader, not written into the
sample file. Never commit generated private configuration.

For Helm, enable `rabbitmq.enabled` and supply `rabbitmq.existing_secret` containing
an AMQP URL at `rabbitmq.url_secret_key` (default `url`). Other settings are wired
through the controller's TOML template:

| Setting | Default |
| --- | --- |
| `request_queue` | `exaflow.analysis.requests` |
| `dead_letter_queue` | `exaflow.analysis.dead` |
| `max_result_bytes` | 16 MiB, including the result envelope |
| `queue_max_bytes` | 256 MiB per request/dead-letter queue |
| `reply_ttl_ms` | 24 hours, applied to each result message |
| `dead_letter_ttl_ms` | 7 days |
| `operation_timeout` | 5 seconds, maximum 5 |
| `shutdown_drain_timeout` | 30 seconds, maximum 30 |

Request bodies have the same 16 MiB limit as HTTP. Configure the broker's
`max_message_size` high enough to let the application report oversized-request
errors; the production test broker uses 32 MiB. Provision bounded reply queues
with appropriate retention; reply messages also carry the configured expiration.
There is no default request expiry. Changing queue declaration arguments requires
an explicit queue migration; RabbitMQ rejects conflicting declarations.
Configure the broker consumer acknowledgement timeout above legitimate capacity
waits plus analysis and reply publication. The test broker uses two hours.
Persistent broker storage and high availability remain deployment responsibilities.

`GET /healthcheck/rabbitmq` returns enabled/connected/consuming state, active job
count, and the last safe error type. It returns 503 when enabled but unavailable,
and 200 when disabled or consuming. Existing `/livez` and `/healthcheck` semantics
remain unchanged. Background logs record connection transitions and delivery
failures without credentials or message bodies.

## Producer example

With `RABBITMQ_URL` supplied privately and controller consumption enabled:

```bash
uv run python examples/rabbitmq_analysis.py analysis-request.json
```

The example declares a durable reply queue per logical request, confirms request
publication, prints the matching reply, and acknowledges it. Supply a stable body
`request_id` when retrying after a client interruption. Idle example reply queues
expire after two days; result messages expire after one day.

## Testing and rollback

- Standalone tests use fake AMQP boundaries for parsing, admission, replies,
  acknowledgements, reconnect decisions, health, limits, and shutdown. Real gRPC
  aggregation tests check attempt isolation and cleanup ownership. Uvicorn process
  tests cover cancellation of queue analyses that run in executor threads.
- Algorithm validation tests continue submitting through HTTP, with no RabbitMQ
  dependency or transport mocks.
- Production environment tests deploy pinned RabbitMQ in kind with persistent
  storage and an isolated generated Secret. Set `RABBITMQ_TEST_URL` to enable the
  AMQP cases. `RABBITMQ_TEST_RECOVERY=1` additionally permits replacing the test
  controller and restarting this dedicated test broker; never use that flag
  against a shared or production deployment. CI enables it only in a separate
  recovery step after the normal production test suite. Recovery cleanup waits
  for the replacement controller rollout and HTTP/AMQP readiness before returning.
  The test broker uses NodePort 30001, mapped to localhost port 5679 by kind;
  this endpoint survives broker restarts without a port-forward process.

Disable consumption to roll back the transport; pending durable requests stay in
the broker and HTTP remains usable. Revert protocol changes across controller,
workers, and aggregation server together. Retention and dead-letter inspection
must be considered before removing broker resources.
