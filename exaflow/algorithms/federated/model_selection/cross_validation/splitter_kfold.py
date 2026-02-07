from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import KFold

from exaflow.algorithms.federated.utils import BadInputError
from exaflow.algorithms.federated.utils.interfaces import FederatedSplitter


class FederatedKFoldSplitter(FederatedSplitter):
    """Deterministic K-Fold splitter with reusable buffers."""

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        n_rows = X.shape[0]
        if n_rows == 0 or n_rows < self.n_splits:
            raise BadInputError(
                "Cross validation cannot run because the number of observations "
                f"({n_rows}) is smaller than the number of splits ({self.n_splits})."
            )

        train_X_buf = np.empty_like(X)
        test_X_buf = np.empty_like(X)
        train_y_buf = np.empty_like(y)
        test_y_buf = np.empty_like(y)

        kf = KFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        def generator():
            for train_idx, test_idx in kf.split(np.arange(n_rows)):
                train_len = len(train_idx)
                test_len = len(test_idx)

                np.take(X, train_idx, axis=0, out=train_X_buf[:train_len])
                np.take(y, train_idx, axis=0, out=train_y_buf[:train_len])
                X_train = train_X_buf[:train_len, ...]
                y_train = train_y_buf[:train_len, ...]

                np.take(X, test_idx, axis=0, out=test_X_buf[:test_len])
                np.take(y, test_idx, axis=0, out=test_y_buf[:test_len])
                X_test = test_X_buf[:test_len, ...]
                y_test = test_y_buf[:test_len, ...]

                yield X_train, y_train, X_test, y_test

        return generator()

    def split_indices(self, n_rows: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if n_rows == 0 or n_rows < self.n_splits:
            raise BadInputError(
                "Cross validation cannot run because the number of observations "
                f"({n_rows}) is smaller than the number of splits ({self.n_splits})."
            )

        kf = KFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state
        )

        def generator():
            for train_idx, test_idx in kf.split(np.arange(n_rows)):
                yield train_idx, test_idx

        return generator()
