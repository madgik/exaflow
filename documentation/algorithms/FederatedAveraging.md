## Federated Averaging

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

#### Algorithm Description

This algorithm aggregates the parameters of the local models and returns their average.

<b><h4>Algorithm Implementation</b></h4>

[FedAvg](../../exaflow/algorithms/exareme3/utils/fedaverage.py)
