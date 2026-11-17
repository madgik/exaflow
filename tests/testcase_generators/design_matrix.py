"""Reference design matrix with the algorithms' coding of covariates.

* nominal  -> dummy columns ``var[level]``, first level in code order is the reference
* binary   -> one 0/1 column ``var[level]``, named after the level coded 1; the
  first declared level is 0 and the second is 1
* ordinal  -> one integer column ``var``, 0..K-1 in the declared level order
* numerical -> unchanged

Columns are ordered nominal, binary, ordinal, numerical, as the algorithms do.
"""

import pandas as pd

from exaflow.algorithms.federated.utils import sort_levels

NUMERICAL_TYPES = ("continuous", "discrete")


def declared_levels(cde):
    enumerations = cde["enumerations"]
    if isinstance(enumerations, dict):
        return [str(code) for code in enumerations]
    return [
        str(item["code"] if isinstance(item, dict) else item) for item in enumerations
    ]


def coded_design_matrix(X, metadata, levels_from=None):
    """Design matrix of ``X`` (without intercept) coded by CDE type.

    ``levels_from`` optionally gives the rows used to learn the nominal levels
    (the training rows in cross-validation); default is all rows of ``X``.
    """
    levels_from = X if levels_from is None else levels_from
    cdes = {cde["code"]: cde for cde in metadata}
    nominal, binary, ordinal, numerical = [], [], [], []
    for name in X.columns:
        # Columns without a metadata entry are numerical.
        cde_type = cdes.get(name, {}).get("type", "continuous")
        if cde_type in NUMERICAL_TYPES:
            numerical.append(name)
        elif cde_type == "binary":
            binary.append(name)
        elif cde_type == "ordinal":
            ordinal.append(name)
        else:
            nominal.append(name)

    columns = {}
    for name in nominal:
        for level in sort_levels(levels_from[name].dropna().unique())[1:]:
            columns[f"{name}[{level}]"] = (X[name] == level).astype(float)
    for name in binary:
        levels = declared_levels(cdes[name])
        assert len(levels) == 2, f"{name} must declare two levels"
        columns[f"{name}[{levels[1]}]"] = (X[name] == levels[1]).astype(float)
    for name in ordinal:
        levels = declared_levels(cdes[name])
        columns[name] = X[name].map(levels.index).astype(float)
    for name in numerical:
        columns[name] = X[name].astype(float)
    return pd.DataFrame(columns, index=X.index)
