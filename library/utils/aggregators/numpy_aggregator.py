import numpy as np

from library.utils.aggregators.aggregation_client import AggregationClientInterface


class NumpyAggregator:
    """A client for performing federated operations on numpy arrays.

    This class provides an interface for common federated aggregation operations
    that communicate with a central aggregation server through an AggregationClient.
    """

    def __init__(self, client:AggregationClientInterface):
        """Initialize the NumpyAggClient with an aggregation client.

                Args:
                    client: An instance of AggregationClient that handles the actual
                           communication with the federated learning server.
                """
        self.client = client

    def fed_union(self,categories:np.ndarray):
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
            categories=np.unique(categories[categories != None])
        _shape, _flattened,_type = NumpyAggregator._transform2(categories)
        _shape = (-1,) + _shape[1:]
        _ans= np.array(self.client.__global_union__(_flattened,_type))
        return NumpyAggregator._inv_transform(_shape, _ans)

    def fed_avg(self,array:np.ndarray):
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
        _shape, _flattened = NumpyAggregator._transform(array)
        _ans = self.client.__global_sum__(np.append(_flattened,1))
        tmp = np.array(_ans[:-1])/_ans[-1]
        return NumpyAggregator._inv_transform(_shape, tmp)

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
        _shape, _flattened = NumpyAggregator._transform(array)
        # Append weight for weighted averaging
        weighted_array = np.append(_flattened , 1)* weight
        _ans = self.client.__global_sum__(weighted_array)
        tmp = np.array(_ans[:-1]) / _ans[-1]  # weighted sum / total weight
        return NumpyAggregator._inv_transform(_shape, tmp)

    def fed_sum(self,array:np.ndarray):
        """Compute the federated sum of an array across all clients.

                Args:
                    array: The numpy array to be summed across all clients.

                Returns:
                    A numpy array with the same shape as input, containing the
                    element-wise sum across all clients' arrays.
                """
        _shape, _flattened = NumpyAggregator._transform(array)
        _ans = self.client.__global_sum__( _flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def fed_min(self,array:np.ndarray):
        _shape, _flattened = NumpyAggregator._transform(array)
        _ans = self.client.__global_min__( _flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def fed_max(self,array:np.ndarray):
        _shape, _flattened = NumpyAggregator._transform(array)
        _ans = self.client.__global_max__( _flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def global_sum(self,array:np.ndarray):
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
        _shape, _flattened = NumpyAggregator._transform(np.sum(array, axis=0))
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def global_count(self,array:np.ndarray):
        """Compute the federated count of samples across all clients.

                Args:
                    array: A numpy array whose first dimension represents samples.

                Returns:
                    The total count of samples across all clients.

                Note:
                    This effectively sums the first dimension sizes from all clients.
                """
        _shape, _flattened = NumpyAggregator._transform(np.count_nonzero(array, axis=0))
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def global_avg(self,array:np.ndarray):
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
        _shape, _flattened = NumpyAggregator._transform(np.sum(array, axis=0))
        _flattened = np.append(_flattened, array.shape[0])
        _ans = self.client.__global_sum__(_flattened)
        return NumpyAggregator._inv_transform(_shape, _ans[:-1]) / _ans[-1]

    def global_min(self,array:np.ndarray):
        """Compute min along axis=0 and then federated min across clients.

                Args:
                    array: The numpy array to find minimum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise minimum across all clients.
                """
        _shape, _flattened = NumpyAggregator._transform(np.min(array, axis=0))
        _ans = self.client.__global_min__(_flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def global_max(self,array:np.ndarray):
        """Compute max along axis=0 and then federated max across clients.

                Args:
                    array: The numpy array to find maximum values from.

                Returns:
                    A numpy array with reduced dimensions (axis=0 removed),
                    containing the element-wise maximum across all clients.
                """
        _shape, _flattened = NumpyAggregator._transform(np.max(array, axis=0))
        _ans = self.client.__global_max__(_flattened)
        return NumpyAggregator._inv_transform(_shape, _ans)

    def global_sum_squares(self, array: np.ndarray) -> np.ndarray:
        """Compute federated sum of squares along axis=0.

        Args:
            array: The numpy array to be squared and summed across samples.

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated sum of squares across all clients.
        """
        return self.global_sum(array ** 2)

    def global_sum_products(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute federated sum of element-wise products along axis=0.

        Args:
            left: First numpy array (same shape as right).
            right: Second numpy array (same shape as left).

        Returns:
            A numpy array with reduced dimensions (axis=0 removed),
            containing the federated sum of products across all clients.
        """
        if left.shape != right.shape:
            raise ValueError("left and right must have the same shape")
        return self.global_sum(left * right)

    def global_dot(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute federated dot product across all clients.

        Args:
            left: First numpy array (same shape as right).
            right: Second numpy array (same shape as left).

        Returns:
            A scalar float containing the federated dot product.
        """
        if left.shape != right.shape:
            raise ValueError("left and right must have the same shape")
        local_dot = np.asarray(np.sum(left * right))
        _shape, _flattened = NumpyAggregator._transform(local_dot)
        _ans = self.client.__global_sum__(_flattened)
        return float(_ans[0])

    def global_l2_norm(self, array: np.ndarray) -> float:
        """Compute the L2 norm of an array across all clients.

        Args:
            array: The numpy array to compute the norm for.

        Returns:
            A scalar float containing the federated L2 norm.
        """
        return float(np.sqrt(self.global_dot(array, array)))

    def global_var(self, array: np.ndarray, *, ddof: int = 0) -> np.ndarray:
        """Compute federated variance along axis=0.

        Args:
            array: The numpy array to compute variance for.
            ddof: Delta degrees of freedom.

        Returns:
            A numpy array containing the federated variance.
        """
        mean = self.global_avg(array)
        sum_squared_diff = self.global_sum((array - mean) ** 2)
        n = self.global_count(array)
        return sum_squared_diff / (n - ddof)

    def global_std(self, array: np.ndarray, *, ddof: int = 0) -> np.ndarray:
        """Compute federated standard deviation along axis=0."""
        return np.sqrt(self.global_var(array, ddof=ddof))

    def global_cov(self, left: np.ndarray, right: np.ndarray, *, ddof: int = 0) -> np.ndarray:
        """Compute federated covariance between two arrays along axis=0.

        Args:
            left: First numpy array (same shape as right).
            right: Second numpy array (same shape as left).
            ddof: Delta degrees of freedom.

        Returns:
            A numpy array containing the federated covariance.
        """
        if left.shape != right.shape:
            raise ValueError("left and right must have the same shape")
        mean_left = self.global_avg(left)
        mean_right = self.global_avg(right)
        sum_products = self.global_sum((left - mean_left) * (right - mean_right))
        n = self.global_count(left)
        return sum_products / (n - ddof)

    def global_corr(self, left: np.ndarray, right: np.ndarray, *, ddof: int = 0) -> np.ndarray:
        """Compute federated Pearson correlation along axis=0."""
        cov = self.global_cov(left, right, ddof=ddof)
        std_left = self.global_std(left, ddof=ddof)
        std_right = self.global_std(right, ddof=ddof)
        return cov / (std_left * std_right)

    @staticmethod
    def _transform(array:np.ndarray):
        try:
            out = array.flatten().astype(np.float64).tolist()
        except Exception as e:
            out = array.flatten().astype(str).tolist()
        return array.shape,out

    @staticmethod
    def _transform2(array: np.ndarray):
        try:
            out = array.flatten().astype(np.float64).tolist()
            return array.shape, out,np.float64
        except Exception as e:
            out = array.flatten().astype(str).tolist()
            return array.shape, out, str

    @staticmethod
    def _inv_transform(original_shape, answer):
        return np.asarray(answer).reshape(original_shape)
