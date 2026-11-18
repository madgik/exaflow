"""Design-matrix coding of categorical covariates, chosen by CDE type.

Every algorithm that turns categorical covariates into model columns builds its
column transformer here, so the coding rules live in one place:

* ``binary``  -> one 0/1 column. The first level declared in the data model is
  0 and the second is 1. The column is named ``var[level coded 1]``.
* ``ordinal`` -> one integer column ``0..K-1`` following the level order declared
  in the data model. The column keeps the variable name.
* ``nominal`` -> one-hot dummy columns with the first sorted level as reference,
  named ``var[level]``.

Binary and ordinal coding uses the declared order, never the observed or sorted
levels. A missing value or a level that is not declared raises an error instead
of being coded silently, and so does a binary or ordinal covariate with fewer than
two observed values across all workers after filtering (``BadInputError`` naming the
covariate and its single value). The one-hot encoder is the only place nominal
variables are dummy coded, and nominal variables are never integer coded.
"""

from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import pandas as pd

from exaflow.algorithms.exareme3.utils.cde_types import declared_levels
from exaflow.algorithms.federated.compose.column_transformer import (
    FederatedColumnTransformer,
)
from exaflow.algorithms.federated.feature_engineering._one_hot_encoder import (
    _FederatedOneHotEncoder,
)
from exaflow.algorithms.federated.feature_engineering._ordinal_encoder import (
    _FederatedOrdinalEncoder,
)
from exaflow.algorithms.federated.utils import BadInputError
from exaflow.datatypes import CDEType


class _ObservedVariationMixin:
    """Fails the analysis when a covariate has fewer than two observed values.

    A binary or ordinal covariate that is constant after filtering is collinear
    with the intercept and cannot be estimated. The observed values are united
    across all workers, so the check is global, and the error stops the analysis.
    """

    check_observed_variation: bool = True

    def fit(
        self,
        *,
        agg_client,
        data: pd.DataFrame,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> None:
        super().fit(
            agg_client=agg_client,
            data=data,
            categorical_vars=categorical_vars,
            numerical_vars=numerical_vars,
        )
        if not self.check_observed_variation:
            return
        for var in categorical_vars:
            column = data[var]
            if isinstance(column, pd.DataFrame):
                column = column.iloc[:, 0]
            local = sorted({str(value) for value in column.dropna().unique()})
            observed = sorted({str(value) for value in agg_client.union(local)})
            if len(observed) < 2:
                shown = f" ({observed[0]})" if observed else ""
                count = "only one observed value" if observed else "no observed values"
                raise BadInputError(
                    f"Covariate '{var}' has {count}{shown} after filtering; it "
                    "cannot be estimated. Remove it or change the filters."
                )


class _BinaryEncoder(_ObservedVariationMixin, _FederatedOrdinalEncoder):
    """Declared-order 0/1 coding of a two-level variable, named after level 1."""

    def get_feature_names_out(
        self,
        *,
        categorical_vars: List[str],
        numerical_vars: Optional[List[str]] = None,
    ) -> List[str]:
        names = [f"{var}[{self.categories_[var][1]}]" for var in categorical_vars]
        return names + list(numerical_vars or [])


class _OrdinalEncoder(_ObservedVariationMixin, _FederatedOrdinalEncoder):
    """Declared-order 0..K-1 coding of an ordinal variable."""


def build_covariate_transformer(
    categorical_vars: Iterable[str],
    metadata: Dict[str, Dict[str, Any]],
    *,
    check_observed_variation: bool = True,
) -> FederatedColumnTransformer:
    """Column transformer that codes each categorical covariate by its CDE type.

    ``categorical_vars`` are the categorical covariates (grouping and event
    variables excluded). Numerical covariates pass through unchanged, so the
    design matrix is: nominal dummies, binary columns, ordinal columns, then the
    numerical covariates. Set ``check_observed_variation`` to False only for a
    transformer refitted per cross-validation fold, where a rare level may be
    absent from a training split.
    """
    nominal: List[str] = []
    binary: Dict[str, List[str]] = {}
    ordinal: Dict[str, List[str]] = {}
    for var in categorical_vars:
        cde_type = (metadata.get(var) or {}).get("type")
        if cde_type == CDEType.NOMINAL:
            nominal.append(var)
        elif cde_type == CDEType.BINARY:
            levels = declared_levels(metadata, var)
            if len(levels) != 2:
                raise BadInputError(
                    f"Binary variable '{var}' must declare exactly two levels, "
                    f"got {levels}."
                )
            binary[var] = levels
        elif cde_type == CDEType.ORDINAL:
            levels = declared_levels(metadata, var)
            if len(levels) < 2:
                raise BadInputError(
                    f"Ordinal variable '{var}' must declare at least two levels, "
                    f"got {levels}."
                )
            ordinal[var] = levels
        else:
            shown_type = cde_type.value if isinstance(cde_type, CDEType) else cde_type
            raise BadInputError(
                f"Covariate '{var}' has type '{shown_type}', which is not categorical."
            )

    steps = []
    if nominal:
        steps.append(("nominal", _FederatedOneHotEncoder(), nominal))
    if binary:
        encoder = _BinaryEncoder(categories=binary, handle_unknown="error")
        encoder.check_observed_variation = check_observed_variation
        steps.append(("binary", encoder, list(binary)))
    if ordinal:
        encoder = _OrdinalEncoder(categories=ordinal, handle_unknown="error")
        encoder.check_observed_variation = check_observed_variation
        steps.append(("ordinal", encoder, list(ordinal)))
    return FederatedColumnTransformer(steps, remainder="passthrough")
