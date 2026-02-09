from typing import Any
from typing import Dict

import numpy as np


def fed_average(agg_client, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Federated averaging of model parameters using exaflow's aggregation server.

    This mirrors the old exaflow `fed_average` UDF, but in exaflow style:
    it is a plain function that you call *inside* an exaflow UDF which has
    access to `agg_client`.

    Parameters
    ----------
    agg_client
        Aggregation client that provides secure sum operations over lists
        of floats.
    params
        Local model parameters to be averaged.
        Expected shape: {name: array_like} for each parameter.

    Returns
    -------
    dict
        Dictionary with the same keys as `params`, each mapped to the
        *global averaged* parameter (as nested lists).
    """
    averaged_params: Dict[str, Any] = {}

    # First, compute the number of participating workers:
    # every worker contributes "1.0" once for this call.
    num_workers = float(agg_client.sum([1.0])[0])
    if num_workers <= 0.0:
        # No workers contributed (degenerate case)
        return {k: np.asarray(v).tolist() for k, v in params.items()}

    for name, local_val in params.items():
        arr = np.asarray(local_val, dtype=float)

        # Flatten, aggregate with secure sum, then reshape
        flat_local = arr.ravel().tolist()
        flat_sum = agg_client.sum(flat_local)
        summed = np.asarray(flat_sum, dtype=float).reshape(arr.shape)

        # FedAvg: simple average over workers (equal weight per worker)
        averaged = summed / num_workers
        averaged_params[name] = averaged.tolist()

    return averaged_params
