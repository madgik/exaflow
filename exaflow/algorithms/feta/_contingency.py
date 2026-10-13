"""Contingency counts built from worker-local categorical indicators."""

import numpy as np

from exaflow.algorithms.feta import FedColumns
from exaflow.algorithms.feta import FedTensor
from exaflow.algorithms.feta import federated_numpy as fnp
from exaflow.worker_communication import BadUserInput


def contingency_table(
    x: FedColumns, y: FedColumns
) -> tuple[np.ndarray, list[str], list[str]]:
    def indicators(group):
        name = group.names[0]
        column = group[name]
        categories = list(dict.fromkeys(group.metadata[name].categories))
        if not categories:
            categories = fnp.unique(column).tolist()
        if not categories:
            raise BadUserInput("Contingency table is empty after pruning.")
        return column, categories

    factor, x_categories = indicators(x)
    outcome, y_categories = indicators(y)
    table = indicator_table(factor, outcome, x_categories, y_categories)
    rows = np.any(table != 0, axis=1)
    columns = np.any(table != 0, axis=0)
    table = table[np.ix_(rows, columns)]
    if not table.size:
        raise BadUserInput("Contingency table is empty after pruning.")
    return (
        table,
        [str(code) for code, keep in zip(x_categories, rows) if keep],
        [str(code) for code, keep in zip(y_categories, columns) if keep],
    )


def indicator_table(
    factor: FedTensor,
    outcome: FedTensor,
    x_categories: list,
    y_categories: list,
) -> np.ndarray:
    # Boolean matrix multiplication does not count observations in NumPy.
    x_design = fnp.column_stack([(factor == code) * 1.0 for code in x_categories])
    y_design = fnp.column_stack([(outcome == code) * 1.0 for code in y_categories])
    return x_design.T @ y_design
