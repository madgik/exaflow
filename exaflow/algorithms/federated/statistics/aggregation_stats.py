import math

import numpy as np

from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyBinaryAggregationFunction,
)
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyUnaryAggregationFunction,
)


class FedCount(NumpyUnaryAggregationFunction):
    """
    Computes the total count of valid elements in a federated array.

    Usage:
        aggregator = NumpyAggregator(client)
        count_algo = FedCount(aggregator)
        total_count = count_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        return self.aggregator.global_count(x)


class Mean(NumpyUnaryAggregationFunction):
    """
    Computes the federated mean (average) of an array across all clients.

    Usage:
        aggregator = NumpyAggregator(client)
        mean_algo = Mean(aggregator)
        avg = mean_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        return self.aggregator.global_avg(x)


class Variance(NumpyUnaryAggregationFunction):
    """
    Computes the federated variance of an array.

    Usage:
        aggregator = NumpyAggregator(client)
        var_algo = Variance(aggregator)
        # For population variance (default):
        pop_var = var_algo(x, ddof=0)
        # For sample variance:
        samp_var = var_algo(x, ddof=1)
    """

    def _compute(self, x: np.ndarray, ddof=0, **kwargs):
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


class StandardDeviation(NumpyUnaryAggregationFunction):
    """
    Computes the federated standard deviation of an array.

    Usage:
        aggregator = NumpyAggregator(client)
        std_algo = StandardDeviation(aggregator)
        # For population standard deviation (default):
        pop_std = std_algo(x, ddof=0)
        # For sample standard deviation:
        samp_std = std_algo(x, ddof=1)
    """

    def _compute(self, x: np.ndarray, ddof=0, **kwargs):
        variance_algo = Variance(self.aggregator)
        return np.sqrt(variance_algo(x, ddof=ddof))


class Range(NumpyUnaryAggregationFunction):
    """
    Computes the federated range (maximum - minimum) of an array.

    Usage:
        aggregator = NumpyAggregator(client)
        range_algo = Range(aggregator)
        data_range = range_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        return self.aggregator.global_max(x) - self.aggregator.global_min(x)


class CoefficientOfVariation(NumpyUnaryAggregationFunction):
    """
    Computes the federated coefficient of variation (standard deviation / mean).
    Returns 0 if the mean is 0.

    Usage:
        aggregator = NumpyAggregator(client)
        cv_algo = CoefficientOfVariation(aggregator)
        cv = cv_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        avg_data = self.aggregator.global_avg(x)
        if avg_data == 0 or self.aggregator.global_count(x) == 0:
            return 0
        std_algo = StandardDeviation(self.aggregator)
        return std_algo(x) / avg_data


class MeanAbsoluteDeviation(NumpyUnaryAggregationFunction):
    """
    Computes the federated average absolute deviation from the mean across all clients.

    Usage:
        aggregator = NumpyAggregator(client)
        mad_algo = MeanAbsoluteDeviation(aggregator)
        mad = mad_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        if self.aggregator.global_count(x) == 0:
            return 0
        avg_data = self.aggregator.global_avg(x)
        return self.aggregator.global_avg(abs(x - avg_data))


class RootMeanSquare(NumpyUnaryAggregationFunction):
    """
    Computes the federated root mean square (RMS) of an array.

    Usage:
        aggregator = NumpyAggregator(client)
        rms_algo = RootMeanSquare(aggregator)
        rms = rms_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        if self.aggregator.global_count(x) == 0:
            return 0
        return np.sqrt(self.aggregator.global_avg(x**2))


class MeanSquare(NumpyUnaryAggregationFunction):
    """
    Computes the federated mean of squared deviations from the mean (population variance).

    Usage:
        aggregator = NumpyAggregator(client)
        ms_algo = MeanSquare(aggregator)
        ms = ms_algo(x)
    """

    def _compute(self, x: np.ndarray, **kwargs):
        y = self.aggregator.global_avg(x)
        z = y - x
        return self.aggregator.global_avg(z**2)


class Covariance(NumpyBinaryAggregationFunction):
    """
    Computes the federated covariance between two arrays.

    Usage:
        aggregator = NumpyAggregator(client)
        cov_algo = Covariance(aggregator)
        # For population covariance (default):
        pop_cov = cov_algo(x, y, ddof=0)
        # For sample covariance:
        samp_cov = cov_algo(x, y, ddof=1)
    """

    def _compute(self, x: np.ndarray, y: np.ndarray, ddof=0, **kwargs):
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

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


class PearsonCorrelation(NumpyBinaryAggregationFunction):
    """
    Computes the federated Pearson correlation coefficient (r) between two arrays.
    Returns 0 if there is no correlation (e.g., zero variance).

    Usage:
        aggregator = NumpyAggregator(client)
        pearson_algo = PearsonCorrelation(aggregator)
        r = pearson_algo(x, y)
    """

    def _compute(self, x: np.ndarray, y: np.ndarray, **kwargs):
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

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


class LeastSquaresRegression(NumpyBinaryAggregationFunction):
    """
    Computes the federated simple linear regression (least squares) returning (slope, intercept).

    Usage:
        aggregator = NumpyAggregator(client)
        lsr_algo = LeastSquaresRegression(aggregator)
        slope, intercept = lsr_algo(x, y)
    """

    def _compute(self, x: np.ndarray, y: np.ndarray, **kwargs):
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= 1:
            raise ValueError("Need at least 2 points for regression")

        # Use sample statistics (ddof=1)
        cov_algo = Covariance(self.aggregator)
        cov = cov_algo(x, y, ddof=1)
        avg_x = self.aggregator.global_avg(x)
        avg_y = self.aggregator.global_avg(y)
        var_algo = Variance(self.aggregator)
        var_x = var_algo(x, ddof=1)

        # Handle zero variance
        if var_x <= 0:
            return 0, avg_y  # Horizontal line at mean of y

        slope = cov / var_x
        intercept = avg_y - slope * avg_x
        return slope, intercept


class StandardizedMeanDifferences(NumpyBinaryAggregationFunction):
    """
    Computes the federated Cohen's d (standardized mean difference) between two groups.

    Usage:
        aggregator = NumpyAggregator(client)
        smd_algo = StandardizedMeanDifferences(aggregator)
        cohens_d = smd_algo(x, y)
    """

    def _compute(self, x: np.ndarray, y: np.ndarray, **kwargs):
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
        var_algo = Variance(self.aggregator)
        var1 = var_algo(x, ddof=1)
        var2 = var_algo(y, ddof=1)

        # Calculate pooled standard deviation (Cohen's d)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Handle zero pooled standard deviation
        if pooled_sd <= 0:
            return 0

        return (mean1 - mean2) / pooled_sd
