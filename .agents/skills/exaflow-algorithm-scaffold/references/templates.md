# Scaffold Templates and Placement Rules

## Standalone Subfolder Inference

Priority order:

1. Explicit `--subfolder`.
1. `--family` inferred subfolder.
1. Algorithm-name inference rules.
1. Final fallback: `tests/standalone_tests/federated_algorithms/_generated/` (reported as warning).

## Generated File Contracts

### Exareme3 algorithm module

- Path: `exaflow/algorithms/exareme3/<algorithm>.py`
- Contains:
  - `Algorithm` subclass
  - `get_specification()` with matching `name`
  - family-aware inputdata defaults
  - local UDF placeholder (or federated-core call when scaffolded)
- Specification metadata:
  - Treat `label` as compact, standard UI menu copy. Avoid implementation
    details such as "federated", backend names, or helper names unless they are
    clinically or methodologically necessary.

  - Treat every `desc` as compact UI copy. Algorithm/preprocessing `desc`
    values should be one concise method-level sentence suitable for cards or
    tooltips. Input and parameter `desc` values should stay short and describe
    observable behavior.

  - Use the parent algorithm/preprocessing step `documentation` field for the
    detailed explanation: formulas/defaults/ranges, strategy options, parameter
    mechanics, assumptions, output interpretation, response fields, and careful
    reference anchors such as `statsmodels`, `scipy.stats`, `scikit-learn`, or
    standard methodology. Use "aligned with" or "methodology consistent with";
    do not claim exact equivalence when defaults or implementation details
    differ.

  - Avoid platform/execution terms in user-facing spec text, including
    `Exaflow`, `exareme`, `MIP`, `worker`, `engine`, `federated implementation`,
    and `aggregation-server-backed`. Prefer neutral wording such as "computed
    from aggregated sufficient statistics without sharing raw data".

  - Use UI-facing input labels such as `Outcome`, `Covariates`, `Variables`,
    `Features`, `Grouping variable`, or `Additional variables`; do not expose
    raw labels such as `x`, `y`, `var`, or `vars`.

  - Do not add `documentation` to `ParameterSpecification`. Parameter `desc`
    values should explain what the setting does, not restate schema shape,
    options, defaults, min/max bounds, requiredness, or dictionary structure.
    For example:

    ```python
    "strategies": specs.ParameterSpecification(
        label="Strategies",
        desc="Clipping strategy for each variable.",
        types=[specs.ParameterType.DICT],
        required=True,
        multiple=False,
        dict_keys_enums=...,
        dict_values_enums=...,
    )
    ```

  - For option-heavy parameters, use newline-separated sections and indented
    bullet lines in the parent `documentation`, for example:

    ```python
    return specs.AlgorithmSpecification(
        name="example_algorithm",
        desc="Compute the requested workflow result.",
        documentation=(
            "Explain what the workflow computes.\n\n"
            "Configure one strategy per variable with 'strategies':\n"
            "  - 'option_a' explains exactly how option_a is computed. Default is 1.0.\n"
            "  - 'option_b' explains exactly how option_b is computed. Default is 0.05.\n\n"
            "The optional 'folds' setting overrides the strategy default per variable:\n"
            "  - 'option_a' folds must be positive finite numbers.\n"
            "  - 'option_b' folds must be finite probabilities in (0, 0.5)."
        ),
        ...
    )
    ```

  - Keep the copy factual. Do not add "use this when..." recommendations unless
    the product surface explicitly needs decision guidance.

  - Prefer `from exaflow.algorithms import specifications as specs` and use
    `specs.ParameterDictValueType` for dictionary value types such as numeric
    folds.

  - Omit explicit optional `None` specification arguments; do not write
    `parameters=None`, `validation=None`, `enumslen=None`, `default=None`,
    `enums=None`, `dict_values_enums=None`, `min=None`, or `max=None`.

### Federated core module (enabled by default)

- Path: `exaflow/algorithms/federated/<family>/<algorithm>.py`
- Contains:
  - `Federated<Algorithm>` placeholder class
  - `compute()` placeholder to be replaced with real logic

### Standalone test

- Path: `tests/standalone_tests/federated_algorithms/<subfolder>/test_<algorithm>.py`
- Contains placeholder test to replace with parity checks.

### Prod test

- Path: `tests/prod_env_tests/test_<algorithm>.py`
- Uses existing request helpers and expected fixture path.

### Prod expected fixture

- Path: `tests/prod_env_tests/expected/<algorithm>_expected.json`
- Default shape is non-empty sample fixture:

```json
{
  "test_cases": [
    {
      "input": {
        "inputdata": {
          "y": ["__REPLACE_ME_Y__"],
          "x": ["__REPLACE_ME_X__"],
          "data_model": "__REPLACE_ME_DATA_MODEL__",
          "datasets": ["__REPLACE_ME_DATASET__"],
          "filters": null
        },
        "parameters": {}
      },
      "output": {}
    }
  ]
}
```

### Documentation

- Canonical algorithm doc: `documentation/algorithms/<algorithm>.md`
- Federated docs index target: `exaflow/algorithms/federated/docs/<algorithm>.md`

### Optional auto-patching

When enabled, scaffold updates:

- `exaflow/algorithms/federated/<family>/__init__.py`
- `exaflow/algorithms/federated/__init__.py`
- `exaflow/algorithms/specifications.py` (`AlgorithmName`)
- `exaflow/algorithms/federated/README.md`
