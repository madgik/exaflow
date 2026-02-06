from __future__ import annotations

from typing import Dict
from typing import List

import numpy as np
import pandas as pd

from exaflow.algorithms.federated.agg_client import AggregationClient
from exaflow.algorithms.federated.interfaces import FederatedEstimator
from exaflow.algorithms.federated.interfaces import FederatedEstimatorResults

ALPHA = 1.0


class FederatedCategoricalNBResults(FederatedEstimatorResults):
    """Results container for federated categorical Naive Bayes."""

    nobs: int

    def __init__(
        self,
        *,
        y_var: str,
        x_vars: List[str],
        categories: Dict[str, List[str]],
        class_count: np.ndarray,
        category_count: Dict[str, np.ndarray],
        labels: List[str],
        n_obs: int,
        class_log_prior: np.ndarray,
        category_log_prob: Dict[str, np.ndarray],
    ):
        self.y_var = y_var
        self.x_vars = list(x_vars)
        self.categories = categories
        self.labels = list(labels)
        self.class_count = np.asarray(class_count, dtype=float)
        self.category_count = {
            xvar: np.asarray(counts, dtype=float)
            for xvar, counts in category_count.items()
        }
        self.class_log_prior = np.asarray(class_log_prior, dtype=float)
        self.category_log_prob = {
            xvar: np.asarray(vals, dtype=float)
            for xvar, vals in category_log_prob.items()
        }
        self.nobs = int(n_obs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.class_count is None or self.category_count is None:
            raise ValueError("CategoricalNB is not fitted yet.")

        X_enc = self._prepare_encoded_features(X)
        if X_enc.shape[0] == 0:
            labels = np.asarray(self.labels)
            return np.asarray([], dtype=labels.dtype)
        if len(self.labels) == 0:
            return np.asarray([], dtype=object)

        likelihood = self._compute_likelihood(X_enc)

        class_sum = self.class_count.sum()
        if class_sum == 0.0:
            prior = np.ones_like(self.class_count, dtype=float) / len(self.class_count)
        else:
            prior = self.class_count / class_sum

        unnormalized_post = prior * likelihood
        denom = unnormalized_post.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        posterior = unnormalized_post / denom
        labels = np.asarray(self.labels)
        if posterior.shape[0] == 0:
            return np.asarray([], dtype=labels.dtype)
        idx = posterior.argmax(axis=1)
        return labels[idx]

    def _prepare_encoded_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize encoded feature matrix:
        - accept numpy arrays or DataFrames
        - coerce NaNs to -1 to represent unknown categories
        - ensure a 2D integer array
        """
        if isinstance(X, pd.DataFrame):
            X_enc = X[self.x_vars].to_numpy()
        else:
            X_enc = np.asarray(X)

        if X_enc.ndim == 1:
            X_enc = X_enc.reshape(-1, len(self.x_vars))

        if np.issubdtype(X_enc.dtype, np.floating):
            X_enc = np.where(np.isnan(X_enc), -1, X_enc)
        return X_enc.astype(int, copy=False)

    def _compute_likelihood(self, X_enc: np.ndarray) -> np.ndarray:
        """
        Compute per-class likelihoods for each row.
        Unknown codes (-1) are treated as missing and ignored (factor = 1).
        """
        category_count_list = [self.category_count[xv] for xv in self.x_vars]
        n_class = self.class_count[:, np.newaxis]
        n_cat = np.array([len(self.categories[xv]) for xv in self.x_vars], dtype=float)

        factors_list = []
        for feat_idx, (counts, codes) in enumerate(zip(category_count_list, X_enc.T)):
            if len(counts) == 0:
                continue
            codes = np.asarray(codes, dtype=int)
            safe_codes = np.where(codes < 0, 0, codes)
            feat_counts = counts[:, safe_codes]
            feat_factors = (feat_counts + ALPHA) / (n_class + ALPHA * n_cat[feat_idx])
            if np.any(codes < 0):
                mask = codes < 0
                feat_factors[:, mask] = 1.0
            factors_list.append(feat_factors)

        if not factors_list:
            return np.ones((X_enc.shape[0], len(self.labels)), dtype=float)

        likelihood = np.ones((len(self.labels), X_enc.shape[0]), dtype=float)
        for factors in factors_list:
            likelihood *= factors
        return likelihood.T


class FederatedCategoricalNB(FederatedEstimator):
    """
    Federated categorical Naive Bayes with sklearn-like interface.

    - Discovers target classes from `y` during `fit` (like sklearn's `classes_`).
    - Expects features `X` to be ordinal-encoded externally (encoder handles
      category ordering and unknowns).
    - Aggregates class/feature counts across workers via the aggregation client.
    """

    def __init__(
        self,
        *,
        y_var: str,
        x_vars: List[str],
        categories: Dict[str, List[str]],
    ) -> None:
        self.y_var = y_var
        self.x_vars = list(x_vars)
        self.categories = categories
        self.results: FederatedCategoricalNBResults | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        agg_client: AggregationClient,
    ) -> FederatedCategoricalNBResults:
        X_enc, y_arr = _ensure_encoded_training_data(X, y, self.x_vars)
        if y_arr.size == 0:
            class_cats = []
        else:
            local_levels = pd.Series(y_arr).dropna().unique().tolist()
            global_levels = agg_client.union(
                [lvl for lvl in local_levels if lvl is not None]
            )
            class_cats = sorted(global_levels)
        n_classes = len(class_cats)

        if y_arr.size == 0 or n_classes == 0:
            class_count_full = np.zeros(n_classes, dtype=float)
        else:
            class_count_local = np.array(
                [float(np.sum(y_arr == label)) for label in class_cats], dtype=float
            )
            class_count_full = np.asarray(
                agg_client.sum(class_count_local), dtype=float
            )

        category_count_full = {}
        for feat_idx, xvar in enumerate(self.x_vars):
            feat_cats = self.categories[xvar]
            n_feat_cats = len(feat_cats)
            counts_matrix_local = np.zeros((n_classes, n_feat_cats), dtype=float)
            if y_arr.size and n_classes:
                for class_idx, label in enumerate(class_cats):
                    mask = y_arr == label
                    if not np.any(mask):
                        continue
                    codes = np.asarray(X_enc[mask, feat_idx], dtype=int)
                    codes = codes[codes >= 0]
                    if codes.size == 0:
                        continue
                    counts = np.bincount(codes, minlength=n_feat_cats)
                    counts_matrix_local[class_idx, :] = counts.astype(float)
            aggregated_counts = (
                agg_client.sum(counts_matrix_local)
                if counts_matrix_local.size
                else counts_matrix_local
            )
            category_count_full[xvar] = np.asarray(aggregated_counts, dtype=float)

        labels = list(class_cats)
        class_count_kept = class_count_full
        category_count_kept = category_count_full
        n_obs = int(np.sum(class_count_kept)) if class_count_kept.size else 0
        class_sum = float(class_count_kept.sum())
        if class_sum == 0.0 and len(class_count_kept):
            class_prior = np.ones_like(class_count_kept, dtype=float) / len(
                class_count_kept
            )
        elif class_sum == 0.0:
            class_prior = class_count_kept
        else:
            class_prior = class_count_kept / class_sum

        class_log_prior = np.log(
            class_prior, where=class_prior > 0, out=np.full_like(class_prior, -np.inf)
        )

        category_log_prob = {}
        for xvar in self.x_vars:
            counts = np.asarray(category_count_kept.get(xvar, []), dtype=float)
            n_cat = len(self.categories.get(xvar, []))
            if counts.size == 0 or n_cat == 0 or class_count_kept.size == 0:
                category_log_prob[xvar] = np.array([])
                continue
            denom = class_count_kept[:, np.newaxis] + ALPHA * n_cat
            probs = (counts + ALPHA) / denom
            log_probs = np.log(probs, where=probs > 0, out=np.full_like(probs, -np.inf))
            category_log_prob[xvar] = log_probs

        results = FederatedCategoricalNBResults(
            y_var=self.y_var,
            x_vars=self.x_vars,
            categories=self.categories,
            class_count=class_count_kept,
            category_count=category_count_kept,
            labels=labels,
            n_obs=n_obs,
            class_log_prior=class_log_prior,
            category_log_prob=category_log_prob,
        )
        self.results = results
        return results


def _ensure_encoded_training_data(X, y, x_vars):
    if isinstance(X, pd.DataFrame):
        X_enc = X[x_vars].to_numpy()
    else:
        X_enc = np.asarray(X)

    if X_enc.ndim == 1:
        X_enc = X_enc.reshape(-1, len(x_vars))

    if np.issubdtype(X_enc.dtype, np.floating):
        X_enc = np.where(np.isnan(X_enc), -1, X_enc)
    X_enc = X_enc.astype(int, copy=False)

    if y is None:
        raise ValueError("Missing target values for categorical naive Bayes.")
    y_arr = np.asarray(y)
    return X_enc, y_arr
