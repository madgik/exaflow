from abc import ABC
from abc import abstractmethod

import numpy as np

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

    def fed_union(self, categories: np.ndarray):
        """Compute the union of categories across all federated clients. Does not compute NaN.

        Args:
            categories: A numpy array of categories to be united with
                       categories from other clients.

        Returns:
            A numpy array containing the union of all categories from
            all clients, with duplicates removed.
        """
        if np.issubdtype(categories.dtype, np.number):
            categories = categories[~np.isnan(categories)]
        else:
            categories = np.unique(categories[categories != None])
        return self.client.union(categories)

    def fed_avg(self, array: np.ndarray):
        """Compute the federated average of an array across all clients.

        Args:
            array: The numpy array to be averaged across all clients.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise average across all clients' arrays.

        Note:
            Internally flattens the array for transmission and appends
            a count (1) for proper averaging.
        """
        global_weight = self.client.sum(np.sum(1))
        return self.client.sum(array / global_weight)

    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute federated weighted average of an array across all clients.

        Args:
            array: The numpy array to be averaged across all clients.
            weight: The weight (typically sample count) for this client's array.
                   Weights from all clients will be summed for normalization.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise weighted average across all clients' arrays.

        Note:
            - Follows the formula: sum(weight_i * array_i) / sum(weights)
            - Internally flattens the array for transmission
            - The weight should typically be positive
            - If all weights are 1, this is equivalent to fed_avg()
        """
        global_weight = self.client.sum(np.sum(weight))
        return self.client.sum((array * weight) / global_weight)

    def fed_sum(self, array: np.ndarray):
        """Compute the federated sum of an array across all clients.

        Args:
            array: The numpy array to be summed across all clients.

        Returns:
            A numpy array with the same shape as input, containing the
            element-wise sum across all clients' arrays.
        """
        return self.client.sum(array)

    def fed_min(self, array: np.ndarray):
        return self.client.min(array)

    def fed_max(self, array: np.ndarray):
        return self.client.max(array)

    def global_sum(self, array: np.ndarray):
        """Compute sum along axis=0 and then federated sum across clients.

        Args:
            array: The numpy array to be summed (along axis=0 first).

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated sum of all clients' sums.

        Note:
            This is different from fed_sum as it first reduces the array
            by summing along axis=0 before federated aggregation.
        """
        return self.client.sum(np.sum(array, axis=0))

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
        """Compute federated average of array sums across clients.

        Args:
            array: The numpy array to be processed (summed along axis=0 first).

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated average across all clients.

        Note:
            Similar to global_sum but divides by total sample count.
            More efficient than fed_avg for large arrays as it reduces first.
        """
        return self.global_sum(array) / self.global_count(array)

    def global_min(self, array: np.ndarray):
        """Compute min along axis=0 and then federated min across clients.

        Args:
            array: The numpy array to find minimum values from.

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the element-wise minimum across all clients.
        """
        return self.client.min(np.min(array, axis=0))

    def global_max(self, array: np.ndarray):
        """Compute max along axis=0 and then federated max across clients.

        Args:
            array: The numpy array to find maximum values from.

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the element-wise maximum across all clients.
        """
        return self.client.max(np.max(array, axis=0))


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
