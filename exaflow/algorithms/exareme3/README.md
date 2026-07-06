# Exareme3 Algorithms- Author Guide

## Core contracts (what every algorithm relies on)

- Base class: `exaflow/algorithms/exareme3/utils/algorithm.py` -> `Algorithm`

  - Each algorithm must implement `@classmethod get_specification()` and return a
    `AlgorithmSpecification` (see `exaflow/algorithms/specifications.py`).
  - Constructor receives `engine`, `inputdata`, and optional `parameters`.
  - Use `self.run_local_udf(func=..., kw_args=...)` to execute UDFs on workers.
  - `self.run_local_udf(..., identical_results=True)` enforces that all worker
    responses are identical and returns one result.
  - Global step contract (`Algorithm.run`): it only has direct access to the
    preprocessed `inputdata` and algorithm `parameters`; it does not directly
    access worker dataframes or metadata.
  - Override these properties when needed:
    - `check_min_rows` (default `True`) to skip the privacy minimum-row check.
    - `add_dataset_variable` (default `False`) to include the dataset column.
  - Missing-value handling is modeled as preprocessing (`missing_values_handler`)
    with a per-variable `strategies` map
    (`drop`, `mean`, `median`, `most_frequent`, `constant`), instead of a
    hardcoded algorithm property.
  - For categorical variables using `constant`, `fill_values[var]` must be one
    of the metadata enum codes for that variable.
  - For `mean`/`median`/`most_frequent`, if a worker has only missing values for
    a selected variable, preprocessing raises `InsufficientDataError` so that
    worker can be skipped (instead of injecting zeros).

- PreprocessingStep base class: `exaflow/algorithms/exareme3/utils/preprocessing_step.py` -> `PreprocessingStep`

  - Each preprocessing step must implement `@classmethod get_specification()` and return a
    `PreprocessingStepSpecification` (see `exaflow/algorithms/specifications.py`).
  - Constructor accepts only `params`.
  - Validation and transforms are explicit contracts:
    - `validate_params(inputdata=..., metadata=...)`
    - `transform_inputdata_variables(x=..., y=...)`
    - `transform_metadata(metadata=...)`
    - `transform_data(data=...)`

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

- The controller discovers Exareme3 algorithms/preprocessing_steps by importing the
  modules under `EXAREME3_ALGORITHM_FOLDERS` and collecting subclasses of the
  base classes.

- Class maps are keyed by the specification name:

  - `exaflow.exareme3_algorithm_classes[AlgorithmSpecification.name] -> class`
  - `exaflow.exareme3_preprocessing_step_classes[PreprocessingStepSpecification.name] -> class`

## UDFs, registry, and aggregation

- Decorate worker UDFs with `@exareme3_udf(...)` in
  `exaflow/algorithms/exareme3/utils/registry.py`.

  - UDF registry keys are stable and derived from `__qualname__` + module.
  - Duplicate keys for different callables raise to avoid ambiguity.
  - `with_aggregation_server=True` injects an `agg_client` argument.
  - In local UDF steps, `metadata` is passed by the system only if the UDF
    function includes `metadata` in its parameters.

- Aggregation client contract:

  - Interface: `exaflow/algorithms/exareme3/utils/udf_aggregation_client_interface.py`
    (`Exareme3UDFAggregationClientI`).
  - `agg_client.sum/min/max(...)` return numpy arrays; convert to lists for JSON.

- Lazy aggregation (now default for aggregation UDFs):

  - `exareme3_udf(with_aggregation_server=True)` enables lazy aggregation.
    The worker applies `lazy_agg` from `exaflow/worker/exareme3/lazy_aggregation`.
  - Disable batching with `enable_lazy_aggregation=False`.

## Preprocessing and metadata helpers

- Metadata validation: `metadata_utils.validate_metadata_vars` (requires
  `is_categorical`), `validate_metadata_enumerations` (requires `enumerations`).
- Variable checks: `validation_utils` has `require_dependent_var`,
  `require_covariates`, and exact-count variants.
- Dummy encoding: use `preprocessing.get_dummy_categories` with
  `run_local_udf_func=self.run_local_udf` to collect categories, then
  `metrics.build_design_matrix` inside UDFs.
- Preprocessing contract:
  - `PreprocessingStep.required_input_variables()` returns extra columns that
    must be loaded by workers before UDF execution.
  - Default implementation returns `[]`; override it for steps that need
    system columns outside user-selected `x`/`y`.
  - `LongitudinalTransformer.required_input_variables()` returns
    `[dataset, subjectid, visitid]`, so worker UDF loading can include the
    longitudinal keys explicitly.
  - Controller preprocessing orchestration follows the request list order and applies:
    `validate_params -> transform_inputdata_variables -> transform_metadata`.
  - Worker preprocessing applies runtime transforms in the same request order via
    `transform_data_and_metadata(data, metadata)`.

## Longitudinal notes

- Longitudinal preprocessing validates `diff` strategies against the provided
  metadata (categorical variables cannot use `diff`).
- `transform_inputdata_variables()` returns transformed `x`/`y` names; the
  strategy rebuilds a new `Inputdata` with `model_copy(...)`.

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
