import math

import numpy as np

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)


class PrimitiveStatistics:
    """Statistical operations using federated NumpyAggregator."""

    def __init__(self, aggregator: NumpyAggregator):
        self.aggregator = aggregator

    def _validate_univariate(self, x: np.ndarray):
        """Validates that the input is a 1D numpy array."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")

    def _validate_bivariate(self, x: np.ndarray, y: np.ndarray):
        """Validates that the inputs are 1D numpy arrays of the same shape."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

    def count(self, x: np.ndarray):
        """
        Computes the total count of valid elements in a federated array.

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The total number of elements across all clients.
        """
        self._validate_univariate(x)
        return self.aggregator.global_count(x)

    def mean(self, x: np.ndarray, *, ddof=0):
        """
        Computes the federated mean (average) of an array across all clients.

        Args:
            x: A 1D numpy array representing the data.
            ddof: Degrees of freedom (default: 0).

        Returns:
            The global mean of the array.
        """
        self._validate_univariate(x)
        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        if n <= ddof:
            raise ValueError(
                f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points."
            )
        return self.aggregator.global_avg(x)

    def variance(self, x: np.ndarray, *, ddof=0):
        """
        Computes the federated variance of an array.

        Args:
            x: A 1D numpy array representing the data.
            ddof: Delta Degrees of Freedom. Use 0 for population variance, 1 for sample variance.

        Returns:
            The global variance of the array.
        """
        self._validate_univariate(x)
        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        if n <= ddof:
            raise ValueError(
                f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points."
            )
        mean = self.aggregator.global_avg(x)
        sum_squared_diff = self.aggregator.global_sum((x - mean) ** 2)
        return sum_squared_diff / (n - ddof)

    def standard_deviation(self, x: np.ndarray, *, ddof=0):
        """
        Computes the federated standard deviation of an array.

        Args:
            x: A 1D numpy array representing the data.
            ddof: Delta Degrees of Freedom. Use 0 for population standard dev, 1 for sample.

        Returns:
            The global standard deviation.
        """
        self._validate_univariate(x)
        return np.sqrt(self.variance(x, ddof=ddof))

    def range(self, x: np.ndarray):
        """
        Computes the federated range (maximum - minimum) of an array.

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The difference between the global max and the global min.
        """
        self._validate_univariate(x)
        return self.aggregator.global_max(x) - self.aggregator.global_min(x)

    def coefficient_of_variation(self, x: np.ndarray):
        """
        Computes the federated coefficient of variation (standard deviation / mean).

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The relative variability. Returns 0 if the mean is 0.
        """
        self._validate_univariate(x)
        avg_data = self.aggregator.global_avg(x)
        if avg_data == 0 or self.aggregator.global_count(x) == 0:
            return 0
        return self.standard_deviation(x) / avg_data

    def mean_absolute_deviation(self, x: np.ndarray):
        """
        Computes the federated average absolute deviation from the mean across all clients.

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The mean absolute deviation, or 0 if array is empty.
        """
        self._validate_univariate(x)
        if self.aggregator.global_count(x) == 0:
            return 0
        avg_data = self.aggregator.global_avg(x)
        return self.aggregator.global_avg(abs(x - avg_data))

    def root_mean_square(self, x: np.ndarray):
        """
        Computes the federated root mean square (RMS) of an array.

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The RMS value of the array.
        """
        self._validate_univariate(x)
        if self.aggregator.global_count(x) == 0:
            return 0
        return np.sqrt(self.aggregator.global_avg(x**2))

    def mean_square(self, x: np.ndarray):
        """
        Computes the federated mean of squared deviations from the mean (population variance).

        Args:
            x: A 1D numpy array representing the data.

        Returns:
            The mean squared deviation.
        """
        self._validate_univariate(x)
        y = self.aggregator.global_avg(x)
        z = y - x
        return self.aggregator.global_avg(z**2)

    def covariance(self, x: np.ndarray, y: np.ndarray, *, ddof=0):
        """
        Computes the federated covariance between two arrays.

        Args:
            x: First 1D numpy array.
            y: Second 1D numpy array. Must be the same shape as x.
            ddof: Delta Degrees of Freedom. 0 for population, 1 for sample.

        Returns:
            The federated covariance between x and y.
        """
        self._validate_bivariate(x, y)

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= ddof:
            raise ValueError(
                f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points."
            )

        avg_x = self.aggregator.global_avg(x)
        avg_y = self.aggregator.global_avg(y)
        sum_products = self.aggregator.global_sum(((x - avg_x) * (y - avg_y)))
        return sum_products / (n - ddof)

    def pearson_correlation(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the federated Pearson correlation coefficient (r) between two arrays.

        Args:
            x: First 1D numpy array.
            y: Second 1D numpy array. Must be the same shape as x.

        Returns:
            The Pearson correlation coefficient. Returns 0 if there's no correlation
            (e.g. constant variables).
        """
        self._validate_bivariate(x, y)

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= 1:
            return 0  # Cannot compute correlation with less than 2 points

        # Use sample statistics (ddof=1)
        avg_x = self.aggregator.global_avg(x)
        avg_y = self.aggregator.global_avg(y)

        cov = self.aggregator.global_sum(((x - avg_x) * (y - avg_y))) / (n - 1)
        var_x = self.aggregator.global_sum(((x - avg_x) ** 2)) / (n - 1)
        var_y = self.aggregator.global_sum(((y - avg_y) ** 2)) / (n - 1)

        # Handle zero variance
        if var_x <= 0 or var_y <= 0:
            return 0

        stddev_x = math.sqrt(var_x)
        stddev_y = math.sqrt(var_y)

        return cov / (stddev_x * stddev_y)

    def standardized_mean_differences(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the federated Cohen's d (standardized mean difference) between two groups.
        Note that the groups x and y do not strictly need the same number of elements.

        Args:
            x: First 1D numpy array representing group 1.
            y: Second 1D numpy array representing group 2.

        Returns:
            The standardized mean difference between group x and group y.
        """
        self._validate_univariate(x)
        self._validate_univariate(y)

        n1 = self.aggregator.global_count(x)
        n2 = self.aggregator.global_count(y)

        if n1 == 0 or n2 == 0:
            raise ValueError("Data arrays cannot be empty")
        if n1 < 2 or n2 < 2:
            raise ValueError("Each group needs at least 2 points for SMD")

        # Calculate means
        mean1 = self.aggregator.global_avg(x)
        mean2 = self.aggregator.global_avg(y)

        # Use sample standard deviations (ddof=1)
        var1 = self.variance(x, ddof=1)
        var2 = self.variance(y, ddof=1)

        # Calculate pooled standard deviation (Cohen's d)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Handle zero pooled standard deviation
        if pooled_sd <= 0:
            return 0

        return (mean1 - mean2) / pooled_sd
