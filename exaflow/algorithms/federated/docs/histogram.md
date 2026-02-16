## 9. histogram.py - FederatedHistogram

### Name

**Histogram (FederatedHistogram)**

### Type

**Descriptive Statistics** (distribution visualization)

### Goal (Why we need it)

Computes **binned frequency counts** for visualizing data distributions.
In a federated setting, bins are defined globally and each client computes local counts that are aggregated.

### When to use

Use Histogram when:

* you need to visualize **continuous variable distributions**
* you want to identify skewness, modes, outliers
* binning strategy is known or can be defined
* supports both numerical and categorical histograms

### When NOT to use

Avoid / be careful when:

* data are categorical without predefined levels (use frequency tables)
* bin edges are unknown and cannot be pre-specified

---

### Inputs / Outputs

| Item              | Description                                   |
| ----------------- | --------------------------------------------- |
| **data**          | DataFrame with variable                       |
| **y_var**         | Variable for histogram                        |
| **x_vars**        | Optional grouping variables                   |
| **metadata**      | Metadata with enumerations or bin info        |
| **bins**          | Number of bins (for numerical)                |
| **min_row_count** | Minimum count for privacy masking             |
| **agg_client**    | Federated aggregation client                  |

**Outputs (HistogramResult)**

* `bins`: list of bin edges or category labels
* `counts`: list of frequencies per bin
* `grouped`: dict of histograms per grouping variable

### Key Differences from numpy/matplotlib

| Aspect          | numpy.histogram   | MIP Federated Implementation  |
| --------------- | ----------------- | ----------------------------- |
| Data access     | Centralized       | Data remains local per client |
| Bin computation | Local             | Requires pre-specified bins   |
| Counts          | Exact             | Exact (aggregated)            |
| Privacy masking | None              | min_row_count support         |

### Approximation vs Exactness

| Component       | numpy/matplotlib | MIP                   |
| --------------- | ---------------- | --------------------- |
| Bin edges       | Exact            | Pre-specified         |
| Counts per bin  | Exact            | Exact (aggregated)    |
| Grouping        | N/A              | Exact (aggregated)    |

