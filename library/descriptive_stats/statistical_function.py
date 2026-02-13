
import numpy as np
import math

from library.utils.interfaces.statistical_function import NumpyStatisticalFunction


class FederatedStatistics(NumpyStatisticalFunction):
    """Statistical operations using federated NumpyAggregator."""

    def fed_count(self, x: np.ndarray):
        """Return total count of elements."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        return self.aggregator.global_count(x)

    def mean(self, x: np.ndarray, *, ddof=0):
        """Return mean of array. ddof parameter maintained for API consistency."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points.")
        return self.aggregator.global_avg(x)

    def variance(self, x: np.ndarray, *, ddof=0):
        """Return variance. Use ddof=0 for population, ddof=1 for sample."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data array cannot be empty")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points.")
        mean = self.aggregator.global_avg(x)
        sum_squared_diff = self.aggregator.global_sum((x - mean) ** 2)
        return sum_squared_diff / (n - ddof)

    def standard_deviation(self, x: np.ndarray, *, ddof=0):
        """Return standard deviation. Use ddof=0 for population, ddof=1 for sample."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        return np.sqrt(self.variance(x, ddof=ddof))

    def sum_of_squares(self, x: np.ndarray):
        """Return sum of squared elements."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        return self.aggregator.global_sum(x ** 2)

    def range(self, x: np.ndarray):
        """Return max - min."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        return self.aggregator.global_max(x) - self.aggregator.global_min(x)

    def coefficient_of_variation(self, x: np.ndarray):
        """Return relative variability (std/mean). Returns 0 if mean is 0."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        avg_data = self.aggregator.global_avg(x)
        if avg_data == 0 or self.aggregator.global_count(x) == 0:
            return 0
        return self.standard_deviation(x) / avg_data

    def mean_absolute_deviation(self, x: np.ndarray):
        """Return average absolute deviation from mean."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        if self.aggregator.global_count(x) == 0:
            return 0
        avg_data = self.aggregator.global_avg(x)
        return self.aggregator.global_avg(abs(x - avg_data))

    def root_mean_square(self, x: np.ndarray):
        """Return RMS (root mean square)."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        if self.aggregator.global_count(x) == 0:
            return 0
        return self.aggregator.global_avg(np.sqrt(x ** 2))

    def mean_square(self, x: np.ndarray):
        """Return mean of squared deviations from mean."""
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if x.ndim > 1:
            raise ValueError("Input must be a 1D array")
        y = self.aggregator.global_avg(x)
        z = y - x
        return self.aggregator.global_avg(z ** 2)

    def covariance(self, x: np.ndarray, y: np.ndarray, *, ddof=0):
        """Return covariance. Use ddof=0 for population, ddof=1 for sample."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= ddof:
            raise ValueError(f"Not enough data points for ddof={ddof}. Need at least {ddof + 1} points.")

        avg_x = self.aggregator.global_avg(x)
        avg_y = self.aggregator.global_avg(y)
        sum_products = self.aggregator.global_sum(((x - avg_x) * (y - avg_y)))
        return sum_products / (n - ddof)

    def pearson_correlation(self, x: np.ndarray, y: np.ndarray):
        """Return Pearson correlation coefficient (r). Returns 0 if no correlation."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= 1:
            return 0  # Cannot compute correlation with less than 2 points

        # Use sample descriptive_stats (ddof=1)
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

    def least_squares_regression(self, x: np.ndarray, y: np.ndarray):
        """Return (slope, intercept) for linear regression."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")
        if x.shape != y.shape:
            raise ValueError("Arrays must have the same shape")

        n = self.aggregator.global_count(x)
        if n == 0:
            raise ValueError("Data arrays cannot be empty")
        if n <= 1:
            raise ValueError("Need at least 2 points for regression")

        # Use sample descriptive_stats (ddof=1)
        cov = self.covariance(x, y, ddof=1)
        avg_x = self.aggregator.global_avg(x)
        avg_y = self.aggregator.global_avg(y)
        var_x = self.variance(x, ddof=1)

        # Handle zero variance
        if var_x <= 0:
            return 0, avg_y  # Horizontal line at mean of y

        slope = cov / var_x
        intercept = avg_y - slope * avg_x
        return slope, intercept

    def standardized_mean_differences(self, x: np.ndarray, y: np.ndarray):
        """Return Cohen's d (standardized mean difference)."""
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Inputs must be 1D arrays")

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
        sd1 = self.standard_deviation(x, ddof=1)
        sd2 = self.standard_deviation(y, ddof=1)

        # Calculate pooled standard deviation (Cohen's d)
        pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

        # Handle zero pooled standard deviation
        if pooled_sd <= 0:
            return 0

        return (mean1 - mean2) / pooled_sd
