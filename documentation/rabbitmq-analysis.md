# Analysis submission through RabbitMQ

RabbitMQ is an optional second transport for the existing analysis service.
HTTP routes, response formats, and algorithm result shapes remain unchanged.
Both transports validate request IDs using the shared rule below. There is no
job database or HTTP result polling endpoint.

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

Caller-provided request IDs must match `^[A-Za-z0-9_-]{1,128}$`: 1–128 ASCII
letters, digits, underscores, or hyphens. This rule applies to HTTP body IDs,
AMQP `correlation_id`, and any body `request_id`. Accepted values are preserved
exactly, without trimming or rewriting. These are Exaflow restrictions on the
standard AMQP correlation property, not RabbitMQ format requirements.

Invalid supplied HTTP IDs return `422`; omitted/null HTTP IDs retain server-side
generation. Nonempty but invalid AMQP correlation IDs, invalid body IDs, and
conflicting IDs receive `invalid_request` replies without execution or cache
lookup. Missing/empty correlation IDs retain existing dead-letter handling.
Clients previously using IDs outside this format must update them. No validation
is repeated at individual analysis log statements.

Controller and worker request logs use fixed service loggers with per-request
adapters. Unique IDs do not register permanent loggers or handlers. Worker
context remains isolated between concurrent UDFs; control characters in logging
identifiers are escaped without changing the protocol IDs.

Each strategy generates a UUID `context_id` for its execution attempt. UDF system
arguments carry that context to workers. Aggregation Configure, Aggregate,
Unregister, Cleanup, and cancellation tombstones use the context ID as their key;
aggregation logs retain both IDs and Cleanup still enforces its ownership token.
Before this change context IDs appeared only in experiment-start logs and were
stored, unused, in the flow engine.

RPCs for an existing context must pair its ID with the request ID that configured it; mismatches
return `INVALID_ARGUMENT` without changing the context. Fresh context IDs make
`ALREADY_CONFIGURED` a defensive check for accidental context reuse.

**Compatibility:** the aggregation RPC and UDF system-argument contracts change.
Upgrade controller, workers, and aggregation server together, even when AMQP is
disabled. There is no fallback for old clients omitting the execution context.
Regenerate bindings using `exaflow/protos/README.md`.

## Message contract

Publish persistent JSON messages with publisher confirms and mandatory routing to
`exaflow.analysis.requests` (configurable). The body is the existing
`AnalysisRequestDTO`, including `inputdata`, `preprocessing`, and `algorithm`.
Required AMQP properties are:

| Property | Meaning |
| --- | --- |
| `correlation_id` | Logical request ID, preserved on replies. |
| `reply_to` | Name of a durable queue provisioned by the producer. |
| `content_type` | Producers should set `application/json`. |
| `delivery_mode` | Producers must use persistent delivery for restart recovery. |

Provision the durable reply queue before submitting an analysis. Every delivery
checks its destination with a passive declaration on a separate reply channel,
before analysis execution. That channel is opened on a dedicated publisher
connection (`exaflow-analysis-publisher`), not on the connection that consumes
requests and sends `basic.ack`. The check adds a channel and one broker round
trip. Missing queues dead-letter the request without running the analysis.
There are no queue-prefix restrictions or additional application permission
checks; the broker is dedicated to Exaflow.

The consumer parses the DTO, waits for shared analysis capacity, then validates
against current worker metadata and constructs the execution strategy. HTTP
keeps its original validation/admission order. Both transports share
`max_concurrent_experiments`; there are no reserved slots. Other HTTP endpoints
do not acquire analysis slots. Queue prefetch and active delivery handlers are
bounded by that same limit, also across reconnects.

A sustained queue backlog can keep HTTP `POST /analysis` returning 503 until
waiting queue analyses have been admitted. This shared-capacity policy is
intentional; metadata and health routes remain available.

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
confirmed by the broker. Missing routing metadata, missing destinations,
unroutable replies, and negative broker confirmations dead-letter only the
affected request. Each delivery publishes through an isolated reply channel on
the publisher connection. These failures do not close the consuming channel or
interrupt other analyses. A memory or disk alarm blocks connections that publish
(`connection.blocked`). Keeping acknowledgements on the consumer connection
prevents that block from stalling `basic.ack` and deadlocking recovery.
The destination can disappear or fill up after the initial check, so replies
retain mandatory routing and publisher confirms. Consumer declarations create bounded,
durable quorum request and dead-letter queues, with at-least-once dead-lettering.
A full queue rejects publication; producers must handle negative confirms.

Processing is **at least once**. A controller crash after reply publication but
before request acknowledgement can repeat the analysis and its reply. Clients
must deduplicate by correlation ID; recomputation can yield different results for
randomized algorithms or changing datasets. No completed-result database exists.
Active duplicate deliveries in a surviving controller await the existing attempt.
Fresh execution contexts prevent stale aggregation calls from affecting retries.

Completed serialized replies for unsettled deliveries are retained in memory for
up to five minutes, bounded to 128 entries and 64 MiB total. Expired entries are
removed first, then the oldest entries; replies larger than the cache budget are
not retained. Redelivery in the same controller can reuse a retained reply after
its execution has finished. Successful acknowledgement or terminal rejection
removes it; a subsequent submission starts a fresh attempt. Cache expiry,
eviction, and controller replacement can still lead to recomputation.

The single-controller consumer checks subscription registration once per second.
If the broker cancels it without closing the connection, zero registered consumers
interrupts the iterator, marks consumption unavailable, and restarts the connection
and subscription.

Transient reply publication failures retry the existing result on a fresh reply
channel, with exponential backoff from 1 to 30 seconds. A confirmation timeout
can still produce duplicate replies. If the publisher connection drops on its
own, it is reconnected and the same reply is retried. Consumption stays up, the
consuming channel stays open, and completed replies are kept. Connection or
consumer-channel failures restart consumption with the same backoff, retaining
completed replies while the controller survives. Deliveries belong to their
original channel and are never acknowledged through a replacement channel. Only
redelivered messages consult the retained-reply cache.

At lifespan shutdown, intake stops, active deliveries drain for up to 20 seconds,
and existing shutdown logic cancels RPCs and performs aggregation cleanup. Then
remaining AMQP tasks stop, and the consumer and publisher connections close
concurrently within one operation timeout. Unfinished requests remain
unacknowledged for redelivery. The shutdown phases total at most 50 seconds with
default settings, leaving ten seconds of headroom in the existing 60-second
lifespan budget. Cancellation does not forcibly terminate Python threads.
Uvicorn's existing HTTP drain precedes the lifespan hook.

## Configuration and provisioning

AMQP defaults to disabled. Provision an external RabbitMQ 4.x broker; the normal
Helm chart does not install it. Use one controller replica and one Uvicorn process.
Disabled consumption ignores broker-specific settings. When enabled, invalid
configuration fails startup; an unreachable broker retries independently of HTTP.
Provide broker credentials and queue permissions through the existing settings
layer. Restrict producer access to the appropriate request and reply queues;
use a dedicated vhost and `amqps://` when TLS is required by the deployment.

Starting with RabbitMQ 4.3.1, including 4.4.1, a passive queue declaration
requires at least one of `configure`, `write`, or `read` on that queue.
RabbitMQ 4.3.0 and 4.2.6 required `configure` specifically; later patches in
those lines accept any one permission. Least-privilege Exaflow users need a
permission on each reply queue the consumer checks, and producers that
passively declare the request queue need a permission on it. Active
declarations of the request and dead-letter queues still require `configure`.

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
| `publish_timeout` | 30 seconds, positive; independent of operation timeout |
| `shutdown_drain_timeout` | 20 seconds, maximum 20 |

Existing configurations setting `shutdown_drain_timeout` above 20 seconds must
lower it before upgrading. Reply publication is interruptible during shutdown,
including when `publish_timeout` exceeds the drain budget.

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
count, and the last safe error type. The JSON fields are unchanged. `connected`
is true only when both the consumer and publisher connections are open, so a
dropped publisher is unavailable even if the subscription is still up. It returns
503 when enabled but not both connected and consuming, and 200 when disabled or
ready. Existing `/livez` and `/healthcheck` semantics remain unchanged. Background
logs record connection transitions and delivery failures without credentials or
message bodies.
Broker channel failures include bounded, sanitized diagnostic text so queue
declaration mismatches can be diagnosed. Connection errors and publication errors
that can contain URLs or message bodies log only their error type.

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
- Production broker regressions cover missing/deleted reply destinations and a
  full reply queue rejecting one reply while another analysis succeeds. Unit
  tests cover a destination disappearing between validation and publication,
  transient publish retries, completed-result reuse, and cache bounds.
- Algorithm validation tests continue submitting through HTTP, with no RabbitMQ
  dependency or transport mocks.
- Production environment tests deploy pinned `rabbitmq:4.4.1` in kind with
  persistent storage and an isolated generated Secret. The broker requests
  512Mi and limits memory to 2Gi. Kind nodes have no memory cap, the pod is
  created after image builds, and the public GitHub runner has 16 GiB, so the
  limit has room. The default high watermark is 0.6 of that limit. Set
  `RABBITMQ_TEST_URL` to enable the AMQP cases. `RABBITMQ_TEST_RECOVERY=1`
  additionally permits replacing the test
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
