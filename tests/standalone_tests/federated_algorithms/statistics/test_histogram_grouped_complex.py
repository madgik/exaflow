import numpy as np
import pandas as pd
import pytest

from exaflow.algorithms.federated.sql.sql import FederatedSQL
from exaflow.algorithms.federated.statistics.histogram import CategoricalHistogram
from exaflow.algorithms.federated.statistics.histogram import SimpleHistogram
from exaflow.algorithms.federated.utils.aggregators.numpy_aggregator import (
    NumpyAggregator,
)
from tests.standalone_tests.federated_algorithms.utils.federated_algorithm_test import (
    FederatedAlgorithmTest,
)


class TestFederatedSQLComplexHistograms(FederatedAlgorithmTest):
    """
    Comprehensive test suite for FederatedSQL with both numerical and categorical histograms,
    handling missing groups and missing categories across workers.
    """

    def _split_inputs(self, X, y, n_workers: int):
        # We manually define the partitions in the dataset for this complex case
        if hasattr(self, "_partitions") and self._partitions:
            x_parts = self._partitions
            y_parts = [np.zeros(len(p)) for p in x_parts]
            full_x = pd.concat(x_parts, ignore_index=True)
            return x_parts, y_parts, full_x, np.zeros(len(full_x))
        return super()._split_inputs(X, y, n_workers)

    def compute_centralized_result(self, X, y, **kwargs):
        num_bins = kwargs.get("num_bins", 5)
        results = {}

        for group_key, sub in X.groupby("group", sort=True):
            # Numerical Histogram ground truth
            nums = sub["val_num"].to_numpy(dtype=float)
            nums_clean = nums[~np.isnan(nums)]
            if len(nums_clean) == 0:
                num_res = (np.array([], dtype=float), np.array([], dtype=float))
            else:
                min_v, max_v = np.nanmin(nums_clean), np.nanmax(nums_clean)
                if min_v == max_v:
                    max_v = min_v + 1.0
                counts, edges = np.histogram(
                    nums_clean, bins=num_bins, range=(min_v, max_v)
                )
                num_res = (counts.astype(float), edges)

            # Categorical Histogram ground truth
            cats = sub["val_cat"].to_numpy(dtype=float)
            cats_clean = [str(c) for c in cats if not pd.isna(c)]
            unique_cats = sorted(set(cats_clean))
            if not unique_cats:
                cat_res = (np.array([], dtype=object), np.array([], dtype=float))
            else:
                cat_counts = [float(cats_clean.count(c)) for c in unique_cats]
                cat_res = (
                    np.array(unique_cats, dtype=object),
                    np.array(cat_counts, dtype=float),
                )

            results[group_key] = {"num": num_res, "cat": cat_res}
        return results

    def compute_federated_result(self, X, y, *, agg_client, **kwargs):
        num_bins = kwargs.get("num_bins", 5)
        aggregator = NumpyAggregator(agg_client)

        simple_hist = SimpleHistogram(aggregator)
        cat_hist = CategoricalHistogram(aggregator)

        results = (
            FederatedSQL(X, aggregator)
            .group_by("group")
            .aggregate(
                "num_hist",
                lambda x: simple_hist.compute(x, num_bins=num_bins),
                "val_num",
            )
            .aggregate("cat_hist", lambda x: cat_hist.compute(x), "val_cat")
            .run()
        )

        df = results.dataframe
        out = {}
        for group_key in df.index:
            out[group_key] = {
                "num": df.loc[group_key, "num_hist"],
                "cat": df.loc[group_key, "cat_hist"],
            }
        return out

    def _outputs_equal(self, left, right):
        if left is right:
            return True
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                return False
            return all(self._outputs_equal(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                return False
            return all(self._outputs_equal(l, r) for l, r in zip(left, right))
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left_arr, right_arr = np.asarray(left), np.asarray(right)
            if left_arr.dtype.kind in ("U", "S", "O"):  # Strings or Objects
                return np.array_equal(left_arr, right_arr)
            return np.allclose(
                left_arr, right_arr, atol=1e-8, rtol=1e-8, equal_nan=True
            )
        return super()._outputs_equal(left, right)

    def compare(self, federated_output, centralized_output, **kwargs):
        assert self._outputs_equal(federated_output, centralized_output)

    def test_federated_algorithm_with_one_worker(self):
        # We don't really need a one-worker case for this specific complex setup,
        # but we must implement it for the ABC. We can just run the comparison with 1 worker if we want.
        # But for this specific scenario, let's just make it call the main scenario logic.
        self._run_main_scenario(n_workers=1)

    def test_federated_algorithm_with_multiple_workers(self):
        self._run_main_scenario(n_workers=2)

    def _run_main_scenario(self, n_workers):
        """
        Scenario:
        100 elements total.
        Groups: G1, G2, G3.
        - Worker 1 has G1, G2.
        - Worker 2 has G2, G3. (If n_workers=2)
        """
        if n_workers == 1:
            # Simple 1-worker version of the complex data
            df = pd.DataFrame(
                {
                    "group": ["G1"] * 40 + ["G2"] * 30 + ["G3"] * 30,
                    "val_num": np.random.rand(100) * 100,
                    "val_cat": np.random.choice([1, 2, 3], 100),
                }
            )
            self._partitions = [df]
            self.run_comparison(X=df, y=np.zeros(len(df)), n_workers=1, num_bins=5)
        else:
            # --- Worker 1 Data ---
            w1_g1 = pd.DataFrame(
                {"group": "G1", "val_num": np.linspace(0, 10, 30), "val_cat": 1}
            )
            w1_g2 = pd.DataFrame(
                {
                    "group": "G2",
                    "val_num": np.linspace(10, 20, 20),
                    "val_cat": np.random.choice([1, 2], 20),
                }
            )
            w1_df = pd.concat([w1_g1, w1_g2])

            # --- Worker 2 Data ---
            w2_g2 = pd.DataFrame(
                {
                    "group": "G2",
                    "val_num": np.linspace(15, 25, 20),
                    "val_cat": np.random.choice([2, 3], 20),
                }
            )
            w2_g3 = pd.DataFrame(
                {"group": "G3", "val_num": np.linspace(100, 200, 30), "val_cat": 3}
            )
            w2_df = pd.concat([w2_g2, w2_g3])

            full_df = pd.concat([w1_df, w2_df])

            self._partitions = [w1_df, w2_df]
            self.run_comparison(
                X=full_df, y=np.zeros(len(full_df)), n_workers=2, num_bins=5
            )
