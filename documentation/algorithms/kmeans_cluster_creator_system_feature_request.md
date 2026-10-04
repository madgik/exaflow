# System Feature Request: K-means Cluster Creator

System feature needed for algorithm: `kmeans_cluster_creator`

Needed capability:
Allow preprocessing steps to declare and receive runtime components, specifically an aggregation-server client, while transforming worker-local data before an algorithm runs.

Current limitation:
Current algorithm-owned interfaces allow an Exareme3 algorithm UDF to request the aggregation server with `@exareme3_udf(with_aggregation_server=True)`, but preprocessing steps do not currently make the controller choose an aggregation-server-backed execution strategy or pass an aggregation client into `transform_data_and_metadata`. A K-means cluster creator needs to fit cluster centers from aggregated sufficient statistics, then assign local rows to cluster labels during preprocessing. Without system support for aggregation-aware preprocessing, an algorithm developer would have to edit controller strategy selection and worker UDF preprocessing plumbing, which is outside the Algorithm Developer Profile.

Minimal requested system change:
Expose a supported preprocessing-step component contract so `PreprocessingStepSpecification.components` can affect execution strategy selection, and pass a scoped aggregation client to preprocessing transforms that declare they need it. The system behavior should preserve existing privacy checks, request cleanup, and worker-local row protection.

Algorithm-side impact:
Algorithm-owned work can define the preprocessing specification, K-means fitting logic, cluster-label metadata, documentation, and standalone tests once the runtime can provide aggregation during preprocessing. Until then, implementing `kmeans_cluster_creator` or a K-means-derived categorical variable variant is blocked because the required distributed fit cannot run inside the current preprocessing interface.

Evidence:
- Current K-means algorithm support uses an aggregation-server-backed UDF in `exaflow/algorithms/exareme3/cluster/kmeans.py`, which works for an algorithm result but does not create a transformed categorical column for downstream analyses.
- The reference branch `k_means_enhancements` adds `exaflow/algorithms/exareme3/preprocessing/kmeans_cluster_creator.py`, which calls `transform_data_and_metadata(..., agg_client)` to fit K-means and create local cluster labels.
- The same reference branch also changes system-owned files:
  - `exaflow/controller/services/algorithm_execution_strategy_factory.py` to include preprocessing component requirements in strategy selection.
  - `exaflow/worker/exareme3/udf/udf_service.py` to create an aggregation client when preprocessing requires it and to pass `agg_client` into preprocessing transforms.
- Those controller and worker changes are not allowed under the Algorithm Developer Profile, so the algorithm variant should not be implemented in this guardrail exercise.
