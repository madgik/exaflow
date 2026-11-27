import numpy as np
import pytest

from exaflow.algorithms.federated.exceptions import FedError
from exaflow.algorithms.federated.model_selection.cross_validation.splitter_kfold import (
    FederatedKFoldSplitter,
)


def test_kfold_splitter_zero_rows_raises():
    splitter = FederatedKFoldSplitter(n_splits=2)
    X = np.empty((0, 2), dtype=float)
    y = np.empty((0,), dtype=float)

    with pytest.raises(FedError):
        list(splitter.split(X, y))
