# Exareme3 Algorithms- Author Guide

## Core contracts (what every algorithm relies on)

- Base class: `exaflow/algorithms/exareme3/utils/algorithm.py` -> `Algorithm`

  - Each algorithm must implement `@classmethod get_specification()` and return a
    `AlgorithmSpecification` (see `exaflow/algorithms/specifications.py`).
  - Constructor receives `inputdata`, `metadata`, `parameters`, and the engine.
  - Use `self.run_local_udf(func=..., kw_args=...)` to execute UDFs on workers.
  - Override these properties when needed:
    - `drop_na_rows` (default `True`) to keep rows with NA values.
    - `check_min_rows` (default `True`) to skip the privacy minimum-row check.
    - `add_dataset_variable` (default `False`) to include the dataset column.

- Transformer base class: `exaflow/algorithms/exareme3/utils/transformer.py` -> `Transformer`

  - Each transformer must implement `@classmethod get_specification()` and return a
    `TransformerSpecification` (see `exaflow/algorithms/specifications.py`).

- Input payloads:

  - `inputdata` is the Pydantic model in
    `exaflow/algorithms/utils/inputdata_utils.Inputdata`.
  - `metadata` is `dict[var] -> {is_categorical: bool, enumerations: {...}}`.
  - `parameters` is a plain dict validated against the specification.

## Specifications

- Prefer using the enums and models from `exaflow/algorithms/specifications.py`
  to build the specification objects. A common pattern is:

  - `from exaflow.algorithms import specifications as specs`
  - `return specs.AlgorithmSpecification(...)`

## Discovery and loading

- The controller discovers Exareme3 algorithms/transformers by importing the
  modules under `EXAREME3_ALGORITHM_FOLDERS` and collecting subclasses of the
  base classes.

- Class maps are keyed by the specification name:

  - `exaflow.exareme3_algorithm_classes[AlgorithmSpecification.name] -> class`
  - `exaflow.exareme3_transformer_classes[TransformerSpecification.name] -> class`

## UDFs, registry, and aggregation

- Decorate worker UDFs with `@exareme3_udf(...)` in
  `exaflow/algorithms/exareme3/utils/registry.py`.

  - UDF registry keys are stable and derived from `__qualname__` + module.
  - Duplicate keys for different callables raise to avoid ambiguity.
  - `with_aggregation_server=True` injects an `agg_client` argument.

- Aggregation client contract:

  - Interface: `exaflow/algorithms/exareme3/utils/udf_aggregation_client_interface.py`
    (`Exareme3UDFAggregationClientI`).
  - `agg_client.sum/min/max(...)` return numpy arrays; convert to lists for JSON.

- Lazy aggregation (now default for aggregation UDFs):

  - `exareme3_udf(with_aggregation_server=True)` enables lazy aggregation.
    The worker applies `lazy_agg` from `exaflow/worker/exareme3/lazy_aggregation`.
  - Disable batching with `enable_lazy_aggregation=False`.
  - Use `agg_client_name="client"` if your UDF uses a custom argument name.

## Preprocessing and metadata helpers

- Metadata validation: `metadata_utils.validate_metadata_vars` (requires
  `is_categorical`), `validate_metadata_enumerations` (requires `enumerations`).
- Variable checks: `validation_utils` has `require_dependent_var`,
  `require_covariates`, and exact-count variants.
- Dummy encoding: use `preprocessing.get_dummy_categories` with
  `run_local_udf_func=self.run_local_udf` to collect categories, then
  `metrics.build_design_matrix` inside UDFs.

## Cross-validation utilities

- `crossvalidation.kfold_indices` yields train/test index arrays for K-folds.
- `crossvalidation.split_dataframe` yields `(train_df, test_df)` pairs.

## Patterns to follow

- Validate input variables and metadata at the start of `run()`.
- Keep UDF inputs minimal (only the columns you use).
- Prefer `self.run_local_udf(...)` for worker dispatch instead of direct engine access.
- When using aggregation, aggregate numpy arrays and convert to lists before
  returning to stay JSON-serializable.
- Preserve privacy checks (minimum row count) unless explicitly opting out via
  `check_min_rows`.
