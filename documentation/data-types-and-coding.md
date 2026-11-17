# Data Types, Storage and Variable Coding

This page is the single reference for how Common Data Elements (CDEs) are
stored and how algorithms code them. Every algorithm page links here and adds a
short "Variable types and coding" note for its own inputs.

## Storage

| CDE `type` | Stored as | Values |
| --- | --- | --- |
| `continuous` | `DOUBLE` | Any finite number, within `minValue`/`maxValue` when set. |
| `discrete` | `INTEGER` | Whole numbers. |
| `binary` | `INTEGER` | Exactly the codes `0` and `1`. |
| `nominal` | `INTEGER` | Codes `1..K` declared in `enumerations`. |
| `ordinal` | `INTEGER` | Codes `1..K` declared in `enumerations`, in rank order. |
| `dataset` (a `nominal` CDE) | `INTEGER` | Codes `1..N` per data model. |

- Every categorical CDE has `enumerations` with integer codes. The enumeration
  `label` carries the readable name (`1` = `Female`). `visitid` and `dataset`
  stay integer-coded nominal variables. `subjectid` and `subjectcode` are
  `discrete` whole numbers without enumerations, so subject identifiers are
  not published as category levels.
- A categorical CDE without enumerations, or with non-integer codes, is rejected
  by the exadata-validator and by the worker CSV loader.
- `dataset` codes are numeric strings in requests (`"datasets": ["12", "14"]`)
  and results. `GET /datasets` returns the codes; the dataset name is the label
  of the `dataset` enumeration in `GET /cdes_metadata`.
- Filter values on categorical columns and on `dataset` are integers.
- Parameters that name a level (`positive_class`, `groupA`, `groupB`,
  `category_order`, `visit1`, `visit2`) take the level code as a string, for
  example `"1"`. Results keyed by level use the same codes.
- Existing deployed data is not migrated; only newly loaded data uses this
  storage.

## Coding inside algorithms

Variable types are matched exactly: an algorithm slot that asks for `binary` or
`ordinal` never accepts a `nominal` CDE and nothing is converted between
categorical types.

| CDE `type` | Coding as a covariate | Column name |
| --- | --- | --- |
| `continuous`, `discrete` | Used as is. | `var` |
| `binary` | One 0/1 column. The first declared level (code `0`) is the reference. | `var[level coded 1]` |
| `ordinal` | One integer column `0..K-1` in declared level order. The model assumes equal spacing between consecutive levels. | `var` |
| `nominal` | One-hot columns. The first level in sorted code order is the reference and is dropped. | `var[code]` |

- A `binary` or `ordinal` covariate with fewer than two distinct observed
  values after filtering (counted across all workers) stops the analysis with a
  clear error such as: `Covariate 'X' has only one observed value (V) after filtering; it cannot be estimated. Remove it or change the filters.` There is
  no warning or silent drop. Nominal covariates are not checked. In
  cross-validation the check also runs on the full data before the folds.
- The `positive_class` of a binary outcome or event variable (logistic
  regression, its cross-validation, the binary GLMM and both Cox regressions) is
  always required and must be one of the codes declared for that variable.
  Observations equal to it are coded `1` and all others `0`; nothing is
  inferred from the stored values.
- Level order for `ordinal` variables and for `binary` 0/1 coding comes from
  the declared enumeration order. Workers that declare the same levels in a
  different order make the data model incompatible, and it is not served.
- `subjectid` is a `discrete` whole number and `visitid` is an integer-coded
  nominal variable; the longitudinal transformer's `visit1` and `visit2` take
  visit codes (for example `"1"` for the baseline visit, whose label may be `BL`).

## Checklist for a new data model

1. Give every `nominal`, `ordinal` and `binary` CDE `enumerations` with integer
   codes (`binary`: `0` and `1`; others: from `1`).
1. Recode the CSV values to those codes; keep the readable value as the label.
1. Run the exadata-validator on the data model folder before loading it.
