<b><h2><center>Support Vector Machine (SVM)</center></h1></b>

<b><h4>Aggregation Server</h4></b>
Some algorithms use the aggregation server to combine partial vectors (e.g., sums)
from workers into a single global result. The controller coordinates the flow and
workers send partial aggregates to the aggregation server via gRPC; the combined
result is then used in the algorithm’s global step.

<b><h4>Algorithm Description</b></h4>

The SVM algorithm uses the state-of-art python library, scikit-learn to calculate the local models. The model from each Worker is then averaged on the Master to return the result of the averaging process.

<b><h4>Exareme3 Notes</h4></b>

- Implemented as a linear SVM using scikit-learn on each worker.
- Global weights/intercept are computed by averaging local models.
- Parameters include `C` and `gamma` (see algorithm spec).
- Inputs must be numeric; target must have at least two classes.

<b><h4>Algorithm Implementation</b></h4>

[SVM](../../exaflow/algorithms/exareme3/linear_svm.py)

[Federated Averaging Strategy](../../exaflow/algorithms/exareme3/utils/fedaverage.py)
