from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.utils.agg_client import AggregationClient


class NumpyAggregator:
    """A client for performing federated operations on numpy arrays.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    def __init__(self, client: AggregationClient):
        """Initialize the NumpyAggClient with an aggregation client.

        Args:
            client: An instance of AggregationClient that handles the actual
                   communication with the federated learning server.
        """
        self.client = client

    def fed_union(self, categories: np.ndarray) -> np.ndarray:
        """Compute the union of categories across all federated clients.

        NaN (for numeric arrays) and None (for object arrays) are totally ignored.

        Args:
            categories: A numpy array of categories to be united with
                       categories from other clients.

        Returns:
            A numpy array containing the union of all categories from
            all clients, with duplicates removed (NaN/None excluded).
        """
        if np.issubdtype(categories.dtype, np.number):
            unique_vals = np.unique(categories[~np.isnan(categories)])
            result = self.client.union(unique_vals)
            return np.array(result)
        else:
            unique_vals = np.unique(categories[~pd.isna(categories)])
            result = self.client.union(unique_vals)
            return np.array(result)

    def fed_avg(self, array: np.ndarray):
        """Compute the federated average of an array across all clients, ignoring NaN.

        NaN positions are excluded from the average; a position returns NaN only
        when every client has NaN there.

        Args:
            array: The numpy array to be averaged across all clients.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise nanmean across all clients' arrays.
        """
        nan_mask = np.isnan(array)
        filled = np.where(nan_mask, 0.0, array)
        valid_count = self.client.sum((~nan_mask).astype(float))
        total_sum = self.client.sum(filled)
        return np.where(valid_count == 0, np.nan, total_sum / valid_count)

    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute federated weighted average of an array across all clients, ignoring NaN.

        NaN positions contribute neither to the numerator nor denominator; a
        position returns NaN only when every client has NaN there.

        Args:
            array: The numpy array to be averaged across all clients.
            weight: The scalar weight for this client's contribution.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise weighted nanmean across all clients' arrays.
        """
        nan_mask = np.isnan(array)
        filled = np.where(nan_mask, 0.0, array)
        effective_weight = np.where(nan_mask, 0.0, float(weight))
        global_weight = self.client.sum(effective_weight)
        weighted_sum = self.client.sum(filled * effective_weight)
        return np.where(global_weight == 0, np.nan, weighted_sum / global_weight)

    def fed_sum(self, array: np.ndarray):
        """Compute the federated sum of an array across all clients, ignoring NaN.

        Each worker's NaN values are treated as 0 for the sum. A position in the
        result is NaN only when *all* workers have NaN at that position.

        Args:
            array: The numpy array to be summed across all clients.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise nansum across all clients' arrays, with NaN where
            every client had NaN.
        """
        nan_mask = np.isnan(array)
        filled = np.where(nan_mask, 0.0, array)
        global_sum = self.client.sum(filled)
        global_valid_count = self.client.sum((~nan_mask).astype(float))
        return np.where(global_valid_count == 0, np.nan, global_sum)

    def fed_min(self, array: np.ndarray):
        """Federated element-wise minimum, ignoring NaN.

        NaN positions are replaced with +inf before aggregation so they do not
        win the min; a position returns NaN only when every client had NaN there
        (global min = +inf is converted back to NaN).
        """
        filled = np.where(np.isnan(array), np.inf, array)
        result = self.client.min(filled)
        return np.where(np.isposinf(result), np.nan, result)

    def fed_max(self, array: np.ndarray):
        """Federated element-wise maximum, ignoring NaN.

        NaN positions are replaced with -inf before aggregation; a position
        returns NaN only when every client had NaN there (global max = -inf is
        converted back to NaN).
        """
        filled = np.where(np.isnan(array), -np.inf, array)
        result = self.client.max(filled)
        return np.where(np.isneginf(result), np.nan, result)

    def global_sum(self, array: np.ndarray):
        """Compute nansum along axis=0 and then federated sum across clients.

        NaN values are excluded from the sum; a column returns NaN only when
        every observation across all clients is NaN.

        Args:
            array: The numpy array to be summed (along axis=0 first).

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated nansum of all clients' arrays.
        """
        local_sum = np.nansum(array, axis=0).astype(float)
        local_valid = np.sum(~np.isnan(array), axis=0).astype(float)
        total_sum = self.client.sum(local_sum)
        total_valid = self.client.sum(local_valid)
        return np.where(total_valid == 0, np.nan, total_sum)

    def global_count(self, array: np.ndarray):
        """Compute the federated count of samples across all clients.

        Args:
            array: A numpy array whose first dimension represents samples.

        Returns:
            The total count of samples across all clients.

        Note:
            This effectively sums the first dimension sizes from all clients.
        """
        return self.client.sum(np.sum(~np.isnan(array), axis=0))

    def global_avg(self, array: np.ndarray):
        """Compute federated nanmean across all clients, ignoring NaN.

        NaN values are excluded from both numerator and denominator; a column
        returns NaN only when every observation across all clients is NaN.

        Args:
            array: The numpy array to be averaged (along axis=0 across clients).

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated nanmean across all clients.
        """
        local_sum = np.nansum(array, axis=0).astype(float)
        local_count = np.sum(~np.isnan(array), axis=0).astype(float)
        total_sum = self.client.sum(local_sum)
        total_count = self.client.sum(local_count)
        return np.where(total_count == 0, np.nan, total_sum / total_count)

    def global_min(self, array: np.ndarray):
        """Compute nanmin along axis=0 and then federated min across clients.

        NaN positions are replaced with +inf locally so they don't win the min;
        a column returns NaN only when every observation across all clients is NaN.

        Args:
            array: The numpy array to find minimum values from.

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the element-wise nanmin across all clients.
        """
        local_min = np.nanmin(np.where(np.isnan(array), np.inf, array), axis=0)
        result = self.client.min(local_min)
        return np.where(np.isposinf(result), np.nan, result)

    def global_max(self, array: np.ndarray):
        """Compute nanmax along axis=0 and then federated max across clients.

        NaN positions are replaced with -inf locally so they don't win the max;
        a column returns NaN only when every observation across all clients is NaN.

        Args:
            array: The numpy array to find maximum values from.

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the element-wise nanmax across all clients.
        """
        local_max = np.nanmax(np.where(np.isnan(array), -np.inf, array), axis=0)
        result = self.client.max(local_max)
        return np.where(np.isneginf(result), np.nan, result)


class NumpyUnaryAggregationFunction(ABC):
    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator

    def __call__(self, x: np.ndarray, *args, **kwargs):
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        self.compute(x, **kwargs)

    @abstractmethod
    def compute(self, x: np.ndarray, **kwargs):
        pass


class NumpyBinaryAggregationFunction(ABC):
    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator

    def __call__(self, x: np.ndarray, y: np.ndarray, *args, **kwargs):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")
        self.compute(x, **kwargs)

    @abstractmethod
    def compute(self, x: np.ndarray, **kwargs):
        pass
