# Exaflow Aggregation Server

A lightweight gRPC microservice that performs **federated vector aggregation**
(`SUM`, `MIN`, `MAX`) from multiple worker nodes.

______________________________________________________________________

## Features

| Capability | Description |
| --- | --- |
| **gRPC API** | Defined in [`exaflow/protos/aggregation_server/aggregation_server.proto`](exaflow/protos/aggregation_server/aggregation_server.proto) – unary RPCs `Configure`, `Aggregate`, `Cleanup`, `Unregister`. |
| **Aggregation modes** | `SUM`, `MIN`, `MAX`, `UNION` over vectors; `Aggregate` accepts repeated operations in one call. |

Note: `UNION` returns JSON-encoded bytes in the `tensors` field.
| **Payload formats** | Payloads are sent in `tensor` bytes (`Arrow` for numeric vectors, `JSON` bytes for `UNION`). |
| **Concurrency** | Thread-pool server (configurable worker pool). |
| **Request lifecycle** | A `request_id` can span multiple steps until `Cleanup` resets the context. |

## Docker

Build a production-ready image with the supplied **Dockerfile**:

```bash
docker build -f exaflow/aggregation_server/Dockerfile -t exaflow/aggregation_server:latest .
docker run -p 50051:50051 exaflow/aggregation_server:latest
```
