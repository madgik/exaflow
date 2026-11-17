"""Shared handling of the ``positive_class`` parameter of binary-outcome algorithms.

Logistic regression, its cross-validated variant, the binary GLMM and both Cox
regressions select the level of a binary variable that counts as the positive
outcome (event) with ``positive_class``. The parameter is always required and
must be one of the levels declared for the variable; nothing is inferred from
the stored values.
"""

from typing import Any
from typing import Dict

import numpy as np
import pandas as pd

from exaflow.algorithms.exareme3.utils.cde_types import require_cde_type
from exaflow.algorithms.exareme3.utils.cde_types import require_declared_level
from exaflow.datatypes import CDEType


def require_binary_positive_class(
    metadata: Dict[str, Dict[str, Any]],
    variable: str,
    positive_class: Any,
    *,
    role: str,
) -> None:
    """Check ``variable`` is binary and ``positive_class`` is one of its declared levels."""
    require_cde_type(metadata, variable, CDEType.BINARY, role=role)
    require_declared_level(
        metadata, variable, positive_class, parameter="positive_class"
    )


def encode_positive_class(series: pd.Series, positive_class: Any) -> np.ndarray:
    """Return 1.0 where ``series`` equals ``positive_class`` and 0.0 elsewhere."""
    # Imported here: the federated module pulls in heavy dependencies that the
    # level checks above do not need.
    from exaflow.algorithms.federated.linear_model.logistic_regression import (
        FederatedLogisticRegression,
    )

    if series.dtype == object:
        # Integer codes are held as text in memory; request values may be ints.
        coerced = str(positive_class)
    else:
        coerced = FederatedLogisticRegression.coerce_positive_class(
            series, positive_class
        )
    return series.eq(coerced).to_numpy(dtype=float, copy=False)
