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


# Apply lazy aggregation to key aggregated helpers
ttest_one_sample = lazy_agg()(ttest_one_sample)
ttest_paired = lazy_agg()(ttest_paired)
