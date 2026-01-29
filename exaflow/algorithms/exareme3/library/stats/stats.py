from typing import List

import numpy as np
import pyarrow as pa
import scipy.special as special
import scipy.stats as st

from exaflow.algorithms.exareme3.lazy_aggregation import lazy_agg


def _to_numpy(x) -> np.ndarray:
    """Convert input (Arrow Table/Array or list/array) to NumPy array."""
    if isinstance(x, pa.Table):
        # Prefer zero-copy Arrow->NumPy via pandas destruction when possible
        try:
            return x.to_pandas(split_blocks=True, self_destruct=True).to_numpy(
                dtype=float
            )
        except Exception:
            return x.to_pandas().to_numpy(dtype=float)
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        try:
            return x.to_numpy(zero_copy_only=True)
        except Exception:
            return x.to_numpy(zero_copy_only=False)
    return np.asarray(x, dtype=float)


def kmeans(agg_client, x, n_clusters, tol=1e-4, maxiter=100, random_state=123):
    """
    Distributed K-means clustering using secure aggregation via agg_client.

    Parameters
    ----------
    agg_client
        Aggregation client providing sum/min/max over lists of floats.
    x
        Local data, 2D array-like (n_samples_local, n_features).
    n_clusters
        Number of clusters (k).
    tol
        Convergence tolerance on the Frobenius norm of center difference.
    maxiter
        Maximum number of iterations.
    random_state
        Seed for deterministic center initialization.

    Returns
    -------
    dict with keys:
        n_obs: int
            Total number of observations across all workers.
        centers: List[List[float]]
            Final cluster centers (k x n_features).
    """

    # Convert to numpy array
    X = _to_numpy(x)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_local, n_features = X.shape

    # Global number of observations
    total_n_obs = int(agg_client.sum([float(n_local)])[0])

    # If there is no data at all, return empty centers
    if total_n_obs == 0:
        return dict(n_obs=0, centers=[])

    # ------------------------------------------------------------------
    # Global initialization of centers using global min/max per feature
    # ------------------------------------------------------------------
    if n_local > 0:
        local_min = np.nanmin(X, axis=0)
        local_max = np.nanmax(X, axis=0)
    else:
        # This worker has no rows but we still need to participate
        local_min = np.full((n_features,), np.inf, dtype=float)
        local_max = np.full((n_features,), -np.inf, dtype=float)

    global_min = np.asarray(agg_client.min(local_min), dtype=float)
    global_max = np.asarray(agg_client.max(local_max), dtype=float)

    rng = np.random.RandomState(seed=random_state)
    centers = rng.uniform(
        low=global_min,
        high=global_max,
        size=(int(n_clusters), n_features),
    )

    # ------------------------------------------------------------------
    # Lloyd's algorithm with distributed aggregation of sums/counts
    # ------------------------------------------------------------------
    for _ in range(int(maxiter)):
        # If this worker has no data, it contributes zero sums/counts
        if n_local > 0:
            # Compute squared distances to centers
            # X: (n_local, n_features)
            # centers: (k, n_features)
            # diff: (n_local, k, n_features)
            diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.einsum("ijk,ijk->ij", diff, diff)
            labels = np.argmin(dists_sq, axis=1)

            # Local sums and counts per cluster
            sum_local = np.zeros((n_clusters, n_features), dtype=float)
            count_local = np.zeros((n_clusters,), dtype=float)

            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    sum_local[k] = X[mask].sum(axis=0)
                    count_local[k] = float(mask.sum())
        else:
            sum_local = np.zeros((n_clusters, n_features), dtype=float)
            count_local = np.zeros((n_clusters,), dtype=float)

        # Aggregate sums and counts across workers
        sum_global_arr = agg_client.sum(sum_local.ravel())
        count_global_arr = agg_client.sum(count_local)
        sum_global = np.asarray(sum_global_arr, dtype=float).reshape(
            (n_clusters, n_features)
        )
        count_global = np.asarray(count_global_arr, dtype=float)

        # Update centers; if a cluster has no assigned points, follow the
        # original Exareme behavior by resetting it to the origin so that
        # it can capture the smallest-norm observations in the next step.
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            if count_global[k] > 0.0:
                new_centers[k] = sum_global[k] / count_global[k]
            else:
                new_centers[k] = np.zeros(n_features, dtype=float)

        # Check convergence (Frobenius norm)
        diff_norm = np.linalg.norm(new_centers - centers, ord="fro")
        centers = new_centers
        if diff_norm <= tol:
            break

    return dict(
        n_obs=int(total_n_obs),
        centers=centers.tolist(),
    )


def pearson_correlation(agg_client, x, y, alpha):
    n_obs = len(y)

    x = _to_numpy(x)
    y = _to_numpy(y)

    sx = np.einsum("ij->j", x)
    sy = np.einsum("ij->j", y)
    sxx = np.einsum("ij,ij->j", x, x)
    syy = np.einsum("ij,ij->j", y, y)
    sxy = np.einsum("ji,jk->ki", x, y)

    total_n_obs_arr = agg_client.sum(np.array([float(n_obs)], dtype=float))
    total_sx = agg_client.sum(sx)
    total_sy = agg_client.sum(sy)
    total_sxx = agg_client.sum(sxx)
    total_syy = agg_client.sum(syy)
    total_sxy = agg_client.sum(sxy)
    total_n_obs = float(np.asarray(total_n_obs_arr).reshape(-1)[0])
    total_sx = np.asarray(total_sx, dtype=float)
    total_sy = np.asarray(total_sy, dtype=float)
    total_sxx = np.asarray(total_sxx, dtype=float)
    total_syy = np.asarray(total_syy, dtype=float)
    total_sxy = np.asarray(total_sxy, dtype=float)

    df = total_n_obs - 2
    if total_n_obs == 0:
        raise ValueError("Cannot compute Pearson correlation on empty data.")

    if df <= 0:
        raise ValueError("Not enough observations to compute Pearson correlation.")

    d = (
        np.sqrt(total_n_obs * total_sxx - total_sx * total_sx)
        * np.sqrt(total_n_obs * total_syy - total_sy * total_sy)[:, np.newaxis]
    )
    correlations = (total_n_obs * total_sxy - total_sx * total_sy[:, np.newaxis]) / d
    correlations[d == 0] = 0
    correlations = correlations.clip(-1, 1)
    t_squared = correlations**2 * (df / ((1.0 - correlations) * (1.0 + correlations)))
    p_values = special.betainc(
        0.5 * df, 0.5, np.fmin(np.asarray(df / (df + t_squared)), 1.0)
    )
    p_values[abs(correlations) == 1] = 0
    r_z = np.arctanh(correlations)
    se = 1 / np.sqrt(total_n_obs - 3)
    z = st.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se, r_z + z * se
    ci_lo, ci_hi = np.tanh((lo_z, hi_z))

    return dict(
        n_obs=int(total_n_obs),
        correlations=correlations.tolist(),
        p_values=p_values.tolist(),
        ci_lo=ci_lo.tolist(),
        ci_hi=ci_hi.tolist(),
    )


def ttest_one_sample(agg_client, sample, *, mu: float, alpha: float, alternative: str):
    sample = _to_numpy(sample).reshape(-1)
    n_obs = sample.size

    sum_x = sample.sum()
    sqrd_x = np.dot(sample, sample)
    diff_x = sum_x - n_obs * mu
    diff_sqrd_x = sqrd_x - 2 * mu * sum_x + n_obs * mu**2

    total_n_obs_arr = agg_client.sum(np.array([float(n_obs)], dtype=float))
    total_sum_x_arr = agg_client.sum(np.array([float(sum_x)], dtype=float))
    total_sqrd_x_arr = agg_client.sum(np.array([float(sqrd_x)], dtype=float))
    total_diff_x_arr = agg_client.sum(np.array([float(diff_x)], dtype=float))
    total_diff_sqrd_x_arr = agg_client.sum(np.array([float(diff_sqrd_x)], dtype=float))
    total_n_obs = float(np.asarray(total_n_obs_arr).reshape(-1)[0])
    total_sum_x = float(np.asarray(total_sum_x_arr).reshape(-1)[0])
    total_sqrd_x = float(np.asarray(total_sqrd_x_arr).reshape(-1)[0])
    total_diff_x = float(np.asarray(total_diff_x_arr).reshape(-1)[0])
    total_diff_sqrd_x = float(np.asarray(total_diff_sqrd_x_arr).reshape(-1)[0])

    if total_n_obs <= 1:
        raise ValueError("Not enough observations for one-sample t-test.")

    smpl_mean = total_sum_x / total_n_obs
    sd = np.sqrt(
        (total_diff_sqrd_x - (total_diff_x**2 / total_n_obs)) / (total_n_obs - 1)
    )
    sed = sd / np.sqrt(total_n_obs)
    t_stat = (smpl_mean - mu) / sed
    df = total_n_obs - 1

    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=smpl_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    cohens_d = -(smpl_mean - mu) / sd

    return dict(
        n_obs=int(total_n_obs),
        t_stat=t_stat,
        df=int(df),
        std=sd,
        p_value=p_value,
        mean_diff=smpl_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def ttest_paired(
    agg_client,
    sample_x,
    sample_y,
    *,
    alpha: float,
    alternative: str,
):
    sample_x = _to_numpy(sample_x).reshape(-1)
    sample_y = _to_numpy(sample_y).reshape(-1)
    if sample_x.shape != sample_y.shape:
        raise ValueError("Paired samples must have the same length.")

    n_obs = sample_x.size
    diff = sample_x - sample_y

    sum_x = sample_x.sum()
    sum_y = sample_y.sum()
    diff_sum = diff.sum()
    diff_sq_sum = np.dot(diff, diff)
    x_sq_sum = np.dot(sample_x, sample_x)
    y_sq_sum = np.dot(sample_y, sample_y)

    totals_arr = agg_client.sum(
        np.array(
            [
                float(n_obs),
                float(sum_x),
                float(sum_y),
                float(diff_sum),
                float(diff_sq_sum),
            ],
            dtype=float,
        )
    )
    total_x_sq_arr = agg_client.sum(np.array([float(x_sq_sum)], dtype=float))
    total_y_sq_arr = agg_client.sum(np.array([float(y_sq_sum)], dtype=float))
    total_n_obs, total_sum_x, total_sum_y, total_diff_sum, total_diff_sq_sum = (
        np.asarray(totals_arr, dtype=float).reshape(-1)
    )
    total_x_sq = float(np.asarray(total_x_sq_arr).reshape(-1)[0])
    total_y_sq = float(np.asarray(total_y_sq_arr).reshape(-1)[0])

    if total_n_obs <= 1:
        raise ValueError("Not enough observations for paired t-test.")

    mean_x = total_sum_x / total_n_obs
    mean_y = total_sum_y / total_n_obs

    sd_x = np.sqrt(
        (total_x_sq - 2 * mean_x * total_sum_x + (mean_x**2) * total_n_obs)
        / (total_n_obs - 1)
    )
    sd_y = np.sqrt(
        (total_y_sq - 2 * mean_y * total_sum_y + (mean_y**2) * total_n_obs)
        / (total_n_obs - 1)
    )
    sd_diff = np.sqrt(
        (total_diff_sq_sum - (total_diff_sum**2 / total_n_obs)) / (total_n_obs - 1)
    )
    sed = sd_diff / np.sqrt(total_n_obs)
    t_stat = (mean_x - mean_y) / sed
    df = total_n_obs - 1

    sample_mean = total_diff_sum / total_n_obs
    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=sample_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    cohens_d = (mean_x - mean_y) / np.sqrt((sd_x**2 + sd_y**2) / 2)

    return dict(
        t_stat=t_stat,
        df=int(df),
        p_value=p_value,
        mean_diff=sample_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def ttest_independent(
    agg_client,
    sample_a,
    sample_b,
    *,
    alpha: float,
    alternative: str,
):
    sample_a = _to_numpy(sample_a).reshape(-1)
    sample_b = _to_numpy(sample_b).reshape(-1)

    n_a = sample_a.size
    n_b = sample_b.size

    totals_arr = agg_client.sum(
        np.array(
            [
                float(n_a),
                float(n_b),
                float(sample_a.sum()),
                float(sample_b.sum()),
            ],
            dtype=float,
        )
    )
    sq_sums_arr = agg_client.sum(
        np.array(
            [
                float(np.dot(sample_a, sample_a)),
                float(np.dot(sample_b, sample_b)),
            ],
            dtype=float,
        )
    )
    n_a_total, n_b_total, sum_a_total, sum_b_total = np.asarray(
        totals_arr, dtype=float
    ).reshape(-1)
    sq_sum_a, sq_sum_b = np.asarray(sq_sums_arr, dtype=float).reshape(-1)

    if n_a_total < 1:
        raise ValueError("Group A has no data.")
    if n_b_total < 1:
        raise ValueError("Group B has no data.")
    if n_a_total + n_b_total <= 2:
        raise ValueError("Not enough observations for independent t-test.")

    mean_a = sum_a_total / n_a_total
    mean_b = sum_b_total / n_b_total

    sd_a = np.sqrt(
        (sq_sum_a - 2 * mean_a * sum_a_total + (mean_a**2) * n_a_total)
        / (n_a_total - 1)
    )
    sd_b = np.sqrt(
        (sq_sum_b - 2 * mean_b * sum_b_total + (mean_b**2) * n_b_total)
        / (n_b_total - 1)
    )

    sed_a = sd_a / np.sqrt(n_a_total)
    sed_b = sd_b / np.sqrt(n_b_total)
    sed = np.sqrt(sed_a**2 + sed_b**2)
    t_stat = (mean_a - mean_b) / sed
    df = n_a_total + n_b_total - 2
    diff_mean = mean_a - mean_b

    ci_lower, ci_upper = st.t.interval(1 - alpha, df, loc=diff_mean, scale=sed)

    if alternative == "greater":
        p_value = 1.0 - st.t.cdf(t_stat, df)
        ci_upper = "Infinity"
    elif alternative == "less":
        p_value = 1.0 - st.t.cdf(-t_stat, df)
        ci_lower = "-Infinity"
    else:
        p_value = (1.0 - st.t.cdf(abs(t_stat), df)) * 2.0

    pooled_var = ((n_a_total - 1) * sd_a**2 + (n_b_total - 1) * sd_b**2) / (
        n_a_total + n_b_total - 2
    )
    cohens_d = diff_mean / np.sqrt(pooled_var)

    return dict(
        t_stat=t_stat,
        df=int(df),
        p_value=p_value,
        mean_diff=diff_mean,
        se_diff=sed,
        ci_upper=ci_upper,
        ci_lower=ci_lower,
        cohens_d=cohens_d,
    )


def roc_curve_binary(y_true, y_score):
    """
    Compute ROC curve points (FPR, TPR) for binary classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels {0, 1}.
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for positive class.

    Returns
    -------
    dict with keys:
        "tpr" : List[float]
        "fpr" : List[float]

    Notes
    -----
    - Identical to sklearn.metrics.roc_curve(..., drop_intermediate=False)
    - Does NOT perform secure aggregation — operates on already-local arrays.
    """
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)

    # Sort by descending score
    desc_idx = np.argsort(-y_score)
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    # Count positives/negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    if P == 0 or N == 0:
        # Degenerate case: only one class present
        return {"tpr": [0.0, 1.0], "fpr": [0.0, 1.0]}

    # True positives & false positives cumulative
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    # Threshold changes
    # Find indices where the score changes
    distinct_idx = np.where(np.diff(y_score))[0]
    # Always include last index
    threshold_idxs = np.r_[distinct_idx, y_true.size - 1]

    # Compute TPR, FPR at each threshold
    tpr = tps[threshold_idxs] / P
    fpr = fps[threshold_idxs] / N

    # prepend (0,0) to match sklearn behavior
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]

    return {
        "tpr": tpr.tolist(),
        "fpr": fpr.tolist(),
    }


# Apply lazy aggregation to key aggregated helpers
kmeans = lazy_agg()(kmeans)
pearson_correlation = lazy_agg()(pearson_correlation)
ttest_one_sample = lazy_agg()(ttest_one_sample)
ttest_paired = lazy_agg()(ttest_paired)
ttest_independent = lazy_agg()(ttest_independent)
