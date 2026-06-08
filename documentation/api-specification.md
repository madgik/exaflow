# Exaflow API Specification

This document describes the Exaflow API contract used by clients that discover
available algorithms, render request forms, validate user input, and execute an
algorithm.

## Table of Contents

- [1) API Flow](#1-api-flow)
- [2) Endpoints](#2-endpoints)
- [3) Algorithm Specification DTOs](#3-algorithm-specification-dtos)
- [4) Metadata Endpoints Used By The UI](#4-metadata-endpoints-used-by-the-ui)
- [5) Algorithm Request DTOs](#5-algorithm-request-dtos)
- [6) Building A Valid Algorithm Request](#6-building-a-valid-algorithm-request)
- [7) Error And Status Mapping](#7-error-and-status-mapping)
- [8) Source Of Truth](#8-source-of-truth)

______________________________________________________________________

## 1) API Flow

The intended client flow is:

1. Call `GET /algorithms` to discover available algorithms and their DTO specifications.
1. Call metadata endpoints to populate data model, dataset, variable, CDE, and filter controls.
1. Build an `AlgorithmRequestDTO` from the selected `AlgorithmSpecificationDTO`.
1. Run client-side checks that mirror the server validator.
1. Submit `POST /algorithms/<algorithm_name>`.
1. Display either the algorithm result or the server validation/error message.

The most important rule: `GET /algorithms` is the runtime source of truth for
algorithm availability and per-algorithm form shape. Do not hardcode request
requirements that are already expressed by the specification DTO.

______________________________________________________________________

## 2) Endpoints

| Endpoint | Purpose | Main UI Use |
| --- | --- | --- |
| `GET /algorithms` | Returns enabled algorithm specifications. | Build algorithm catalog and dynamic algorithm forms. |
| `POST /algorithms/<algorithm_name>` | Executes an algorithm request. | Submit the validated request payload. |
| `GET /datasets` | Returns available datasets grouped by data model. | Populate data model and dataset selectors. |
| `GET /datasets_locations` | Returns dataset-to-worker mapping per data model. | Optional diagnostics or data location display. |
| `GET /datasets_variables` | Returns available variables per dataset and data model. | Inform variable availability in the UI. |
| `GET /cdes_metadata` | Returns CDE metadata per data model. | Validate variable types, stattypes, filters, and enum-derived parameters. |
| `GET /data_models_attributes` | Returns data model metadata/attributes. | Display data-model-level metadata. |

______________________________________________________________________

## 3) Algorithm Specification DTOs

`GET /algorithms` returns `AlgorithmSpecificationsDTO`, a root JSON array of
enabled `AlgorithmSpecificationDTO` objects.

Example:

```json
[
  {
    "name": "linear_regression",
    "desc": "Short description.",
    "documentation": "Longer explanation.",
    "label": "Linear Regression",
    "inputdata": {
      "data_model": {},
      "datasets": {},
      "filter": {},
      "y": {},
      "x": {},
      "validation_datasets": {}
    },
    "parameters": {},
    "preprocessing": [],
    "type": "exareme3"
  }
]
```

### `AlgorithmSpecificationsDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| root | `AlgorithmSpecificationDTO[]` | The enabled algorithms available in the current deployment. | Treat this list as authoritative; deployments may expose different algorithms. |

### `AlgorithmSpecificationDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `name` | `string` | Stable algorithm id. | Use this exact value in `POST /algorithms/<algorithm_name>`. |
| `desc` | `string` | Short algorithm description. | Use in compact UI surfaces such as algorithm lists. |
| `documentation` | `string` | Longer algorithm documentation. | Use in details/help views. |
| `label` | `string` | Human-readable algorithm name. | Use as display label. |
| `inputdata` | `InputDataSpecificationsDTO` | Describes required data model, datasets, filter, variable slots, and validation datasets. | Drives the `inputdata` part of the request form. |
| `parameters` | `object \| null` | Map of parameter name to `ParameterSpecificationDTO`. | Request parameter keys outside this map are rejected. |
| `preprocessing` | `PreprocessingStepSpecificationDTO[] \| null` | Enabled preprocessing steps exposed with this algorithm. | Required steps are listed separately in `required_preprocessing`. |
| `required_preprocessing` | `string[]` | Preprocessing step ids that must be included in execution requests. | The server rejects requests that omit any listed step. |
| `type` | `string` | Algorithm backend type, for example `exareme3`. | Informational for clients; execution still uses `POST /algorithms/<name>`. |

During request validation, the server uses this DTO to decide which variables
are required, which CDEs are valid for each variable slot, which parameters are
accepted, which preprocessing steps are accepted, and how parameter values are
validated.

### `InputDataSpecificationsDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `data_model` | `InputDataSpecificationDTO` | Specification for request field `inputdata.data_model`. | Engine-provided and required. |
| `datasets` | `InputDataSpecificationDTO` | Specification for request field `inputdata.datasets`. | Engine-provided and required. |
| `filter` | `InputDataSpecificationDTO` | Specification for request field `inputdata.filters`. | The spec field is singular `filter`; the request field is plural `filters`. |
| `y` | `InputDataSpecificationDTO` | Algorithm-defined primary variable slot. | Usually dependent, outcome, or primary variables. |
| `x` | `InputDataSpecificationDTO \| null` | Optional algorithm-defined secondary variable slot. | Usually independent, covariate, or input variables. If absent, do not render/send `inputdata.x`. |
| `validation_datasets` | `InputDataSpecificationDTO \| null` | Validation dataset slot. | If present, request `inputdata.validation_datasets` is required. If absent, that request field is rejected. |

### `InputDataSpecificationDTO`

Example:

```json
{
  "label": "Variable",
  "desc": "Variable used by the algorithm.",
  "types": ["real"],
  "required": true,
  "stattypes": ["numerical"],
  "min_count": 1,
  "max_count": 1
}
```

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `label` | `string` | UI label for the input slot. | Use in form labels and validation messages. |
| `desc` | `string` | Short input description. | Use as helper text. |
| `types` | `string[]` | Allowed value/CDE types. | Variable slots use `real`, `int`, or `text`; filter specs use `jsonObject`. A slot that allows `real` also accepts `int` CDEs. |
| `required` | `boolean` | Whether the slot is required when `min_count` is not set. | Effective minimum is `min_count` if present, otherwise `1` when required and `0` when optional. |
| `stattypes` | `string[] \| null` | Allowed statistical types for variable slots. | `numerical` means non-categorical CDEs; `nominal` means categorical CDEs. Engine fields use `null`. |
| `min_count` | `integer \| null` | Minimum number of selected values. | Must be non-negative. |
| `max_count` | `integer \| null` | Maximum number of selected values. | `null` means no explicit maximum; when set, must be non-negative and not less than `min_count`. |

### `ParameterSpecificationDTO`

Algorithm parameters and preprocessing parameters use the same specification.

Example:

```json
{
  "label": "Parameter label",
  "desc": "What the parameter controls.",
  "types": ["text"],
  "required": true,
  "multiple": false,
  "default": null,
  "enums": null,
  "dict_keys_enums": null,
  "dict_values_type": null,
  "dict_values_enums": null,
  "min": null,
  "max": null
}
```

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `label` | `string` | UI label and server error reference. | Use in form controls and error messages. |
| `desc` | `string` | Short parameter description. | Use as helper text. |
| `types` | `string[]` | Allowed request value types. | Supported values are `text`, `int`, `real`, `boolean`, and `dict`. |
| `required` | `boolean` | Whether the parameter must be submitted. | Required values cannot be `null`, blank strings, empty lists, or empty objects. |
| `multiple` | `boolean` | Whether the submitted value must be a list. | If true, validate every list item. If false, submit one scalar/object value. |
| `default` | any | Optional default value. | Use to initialize the UI control. |
| `enums` | `ParameterEnumSpecificationDTO \| null` | Allowed values for non-dict parameters. | Do not use for dict parameters. |
| `dict_keys_enums` | `ParameterEnumSpecificationDTO \| null` | Allowed keys for dict parameters. | Valid only when `types` contains `dict`. |
| `dict_values_type` | `string \| null` | Required type for every dict value. | Valid only when `types` contains `dict`. |
| `dict_values_enums` | `ParameterEnumSpecificationDTO \| null` | Allowed values for every dict value. | Valid only when `types` contains `dict`. |
| `min` | `number \| null` | Numeric lower bound. | Apply to numeric parameter values. |
| `max` | `number \| null` | Numeric upper bound. | Apply to numeric parameter values. |

Spec-load validation:

- `dict` cannot be combined with other `types`.
- `dict_keys_enums`, `dict_values_type`, and `dict_values_enums` are valid only for `dict` parameters.
- `enums` is not valid for `dict` parameters.
- `input_var_names` enum specs require `types` to be exactly `["text"]`.
- `input_var_CDE_enums` must use one source (`x` or `y`), cannot use `multiple=true`, and requires the referenced input slot to have `max_count=1`.
- `fixed_var_CDE_enums` must use exactly one fixed CDE source.

### `ParameterEnumSpecificationDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `type` | `string` | Enum source strategy. | Determines how `source` is interpreted. |
| `source` | `array` | Source values for the enum strategy. | Values are interpreted according to `type`. |

Enum types:

- `list`: `source` is the literal list of allowed values.
- `input_var_names`: `source` contains `x`, `y`, or both; allowed values are selected variable names from those request fields.
- `input_var_CDE_enums`: `source` contains one value, `x` or `y`; allowed values come from the CDE enumerations of the single selected variable in that slot.
- `fixed_var_CDE_enums`: `source` contains one fixed CDE name; allowed values come from that CDE's enumerations.

### `PreprocessingStepSpecificationDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `name` | `string` | Stable preprocessing step id. | Use as key under request `preprocessing`. |
| `desc` | `string` | Short preprocessing description. | Use in compact UI. |
| `documentation` | `string` | Longer preprocessing explanation. | Use in details/help views. |
| `label` | `string` | Human-readable preprocessing name. | Use as display label. |
| `parameters` | `object \| null` | Map of step parameter name to `ParameterSpecificationDTO`. | Validate like algorithm parameters. |
| `order` | `integer` | Server execution order. | Display steps in this order; JSON insertion order does not control execution. |

Preprocessing behavior:

- Request step names must exist in the enabled preprocessing specifications.
- Step names listed in `required_preprocessing` must be present under request
  `preprocessing`; the server does not inject required steps automatically.
- Step parameters are validated with the same rules as algorithm parameters.
- Each step then runs implementation-specific `validate_params`.
- Steps may transform selected `x`/`y` variable names and metadata before final algorithm validation.

______________________________________________________________________

## 4) Metadata Endpoints Used By The UI

| Endpoint | Data Used By UI |
| --- | --- |
| `GET /datasets` | Map keyed by data model; use the selected `inputdata.data_model` key to get valid dataset names. |
| `GET /datasets_variables` | Map keyed by data model and dataset; use it to inspect variable availability. |
| `GET /cdes_metadata` | Map keyed by data model; use the selected `inputdata.data_model` key to get CDE `sql_type`, categorical status, and enumerations. |
| `GET /data_models_attributes` | Data-model-level display metadata. |
| `GET /datasets_locations` | Optional dataset location/diagnostic information. |

`GET /datasets` and `GET /cdes_metadata` do not accept a path parameter. Call
the endpoint once, then index the returned map by the selected
`inputdata.data_model`.

`GET /cdes_metadata` is the key endpoint for avoiding validator errors on
variable selections and filter values, because request validation checks CDE
existence, SQL type compatibility, categorical status, and enum-derived
parameter values against metadata.

______________________________________________________________________

## 5) Algorithm Request DTOs

`POST /algorithms/<algorithm_name>` accepts an `AlgorithmRequestDTO`.

Example:

```json
{
  "request_id": "optional-string",
  "inputdata": {
    "data_model": "dementia:0.1",
    "datasets": ["dataset_a"],
    "validation_datasets": ["dataset_validation"],
    "filters": {},
    "y": ["outcome"],
    "x": ["age", "sex"]
  },
  "parameters": {
    "parameter_name": "value"
  },
  "preprocessing": {
    "missing_values_handler": {
      "strategy": "drop"
    }
  }
}
```

### `AlgorithmRequestDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `request_id` | `string \| null` | Optional client request id. | If omitted, the server generates one. |
| `inputdata` | `AlgorithmInputDataDTO` | Data model, dataset, filter, and variable selection. | Required. |
| `parameters` | `object \| null` | Algorithm parameter values. | Keys must exist in selected algorithm `parameters` spec. |
| `preprocessing` | `object \| null` | Preprocessing step configuration. | Step names must exist in enabled preprocessing specs, and required preprocessing steps must be included. |

### `AlgorithmInputDataDTO`

| Field | Type | Meaning | Validation / UI Notes |
| --- | --- | --- | --- |
| `data_model` | `string` | Data model code. | Required; must exist in worker landscape metadata. |
| `datasets` | `string[]` | Training dataset names. | Required; every value must exist for `data_model`. |
| `validation_datasets` | `string[] \| null` | Validation dataset names. | Required only when the algorithm spec exposes `inputdata.validation_datasets`; forbidden otherwise. |
| `filters` | `object \| null` | Query-builder filter JSON. | Optional; must use valid CDE ids, operators, conditions, and value types. |
| `y` | `string[] \| null` | Primary/dependent variables. | Requiredness, count, types, and stattypes come from `AlgorithmSpecificationDTO.inputdata.y`. |
| `x` | `string[] \| null` | Secondary/independent variables. | Submit only when `AlgorithmSpecificationDTO.inputdata.x` exists. |

______________________________________________________________________

## 6) Building A Valid Algorithm Request

Build the request in this order:

1. Select an algorithm from `GET /algorithms`.
1. Select `inputdata.data_model` from metadata/data endpoints.
1. Select `inputdata.datasets` from the `GET /datasets` response at the selected `data_model` key.
1. Render `x`, `y`, parameters, and preprocessing from the selected algorithm specification.
1. Use the `GET /cdes_metadata` response at the selected `data_model` key to validate variable types, stattypes, filters, and enum-derived options.
1. Submit `POST /algorithms/<algorithm_name>` using `AlgorithmSpecificationDTO.name`.

Inputdata checks:

- Submit `x` and `y` as lists, never scalars.
- Enforce each slot's `min_count` and `max_count`.
- Reject duplicates inside `x` or inside `y`.
- Reject variables that appear in both `x` and `y`.
- Only allow CDEs present in the `GET /cdes_metadata` response for the selected `data_model`.
- Check each CDE `sql_type` against the slot `types`.
- Allow `int` CDEs for slots that allow `real`.
- Check categorical status against `stattypes`.
- Include `validation_datasets` only when the spec includes it.
- Require `validation_datasets` when the spec includes it.

Parameter checks:

- Do not submit parameter names missing from the selected spec.
- Include every required parameter.
- Reject `null`, blank strings, empty lists, and empty objects for required/provided parameters.
- For `multiple=true`, submit a list and validate each item.
- For `multiple=false`, submit one scalar/object value.
- Enforce `types`, `min`, `max`, `enums`, and dict key/value constraints.
- For `boolean`, submit actual JSON booleans, including `false`.

Preprocessing checks:

- Only allow step names from `AlgorithmSpecificationDTO.preprocessing`.
- Validate each step's parameters using `ParameterSpecificationDTO`.
- Display steps in `order`.
- Expect preprocessing to possibly transform effective `x`, `y`, and metadata before algorithm validation.

### Filter Contract

`inputdata.filters` is optional. When present, it must be a dictionary in
query-builder style.

Group node:

```json
{
  "condition": "AND",
  "rules": [
    {
      "id": "age",
      "operator": "greater_or_equal",
      "type": "integer",
      "value": 18
    }
  ]
}
```

Rule node:

```json
{
  "id": "gender",
  "operator": "equal",
  "type": "string",
  "value": "F"
}
```

Validator requirements:

- Top-level value must be a dictionary.
- Group `condition` must be `AND` or `OR`.
- Rule `id` must be a CDE in the selected `data_model`.
- Rule `operator` must be one of: `equal`, `not_equal`, `less`, `greater`, `less_or_equal`, `greater_or_equal`, `between`, `not_between`, `is_null`, `is_not_null`, `in`, `not_in`.
- `value` may be `null`, a scalar `int`/`float`/`str`, or a list of those scalars.
- Non-null values must be convertible to the selected CDE's type.

______________________________________________________________________

## 7) Error And Status Mapping

Exaflow maps API exceptions to status codes:

| Status | Meaning | Client Handling |
| --- | --- | --- |
| `400` | Malformed request or filter format error. | Treat as client payload/schema issue. |
| `460` | Semantic user-input validation error. | Show the server message near the relevant form field when possible. |
| `461` | Insufficient data for algorithm execution. | Explain that the chosen datasets/filters/variables do not leave enough usable data. |
| `512` | Worker unresponsive during execution. | Treat as execution infrastructure failure. |
| `513` | Worker task timeout during execution. | Treat as execution timeout. |

Clients should expose the response body message as the primary diagnostic text.

______________________________________________________________________

## 8) Source Of Truth

- DTOs:
  - `exaflow/controller/services/api/algorithm_request_dtos.py`
- Semantic validation:
  - `exaflow/controller/services/api/algorithm_request_validator.py`
- Algorithm spec DTO shape:
  - `exaflow/controller/services/api/algorithm_spec_dtos.py`
- Route surface:
  - `exaflow/controller/quart/endpoints.py`
- Error mapping:
  - `exaflow/controller/quart/error_handlers.py`
- Algorithm/preprocessing names:
  - `exaflow/algorithms/specifications.py`
