"""
Lightweight cross-validation helpers adapted from the legacy flows.

These run locally (inside a UDF) and avoid repeated allocations where possible.
"""

from typing import Dict
from typing import Optional


def min_rows_for_cv(
    df, y_var: str, n_splits: int, *, positive_class: Optional[object] = None
) -> Dict[str, object]:
    """
    Common per-worker check used by CV flows to ensure enough rows for splitting.
    """

    if y_var in df.columns:
        series = df[y_var]
        if positive_class is not None:
            series = series == positive_class
        n_obs = int(series.dropna().shape[0])
    else:
        n_obs = 0
    return {"ok": bool(n_obs >= int(n_splits)), "n_obs": n_obs}
