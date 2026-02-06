# Federated Algorithms User Guide

- [Intro](#intro)
  - [System Overview](#system-overview)
- [Getting started](#getting-started)
  - [Federated algorithms overview](#federated-algorithms-overview)
  - [Writing a federated algorithm](#writing-a-federated-algorithm)
    - [Local computations](#local-computations)
    - [Global computations](#global-computations)
    - [Algorithm flow](#algorithm-flow)
  - [Porting to Exaflow](#porting-to-exaflow)
    - [Declaring worker steps](#declaring-worker-steps)
    - [Algorithm flow in Exaflow](#algorithm-flow-in-exaflow)
    - [Data loading](#data-loading)
    - [Algorithm Specifications](#algorithm-specifications)
    - [Running the algorithm](#running-the-algorithm)
- [Advanced topics](#advanced-topics)
  - [UDF registry and decorator options](#udf-registry-and-decorator-options)
  - [Metadata and preprocessing helpers](#metadata-and-preprocessing-helpers)
  - [Aggregation server (SUM/MIN/MAX)](#aggregation-server-summinmax)
- [Best practices](#best-practices)
  - [Memory efficiency](#memory-efficiency)
  - [Time efficiency](#time-efficiency)
  - [Privacy](#privacy)
    - [Privacy threshold](#privacy-threshold)
    - [Share only aggregates](#share-only-aggregates)
- [Examples](#examples)
  - [Iterative algorithm](#iterative-algorithm)
  - [Class based model](#class-based-model)
  - [Customizing worker inputs](#customizing-worker-inputs)

# Intro

This guide will walk you through the process of writing federated algorithms
using Exaflow. Exaflow's focus is primarily on machine learning and statistical
algorithms, but the framework is so general as to allow any type of algorithm
to be implemented, provided that the input data is scattered across multiple
decentralized data sources.

## System Overview

Exaflow consists of a controller service plus worker services distributed across
remote sites. There are multiple local workers and a single global worker. Local
workers host primary data sources, while the global worker performs global
computations coordinated by the controller.

# Getting started

## Federated algorithms overview

In the highest level, a federated algorithm is composed of three ingredients.
Local computations, global computations and the algorithm flow. A local
computation takes places in each local worker and usually produces some aggregate
of the primary data found in the local worker. A global computation takes place
in a special worker called the global worker, and usually consolidates the local
aggregates into a global aggregate. Finally, the algorithm flow is responsible
for coordinating these computations and the data exchange between workers.

## Writing a federated algorithm

In order to write a federated algorithm we need to define local and global
computations and write an algorithm flow that ties the computations together.

Let's break down the steps one by one. We'll begin by writing a simple
algorithm for computing the mean of a single variable. This algorithm will be
written as a simple python script, running on a single machine. The purpose of
this exercise is to illustrate how an algorithm is decomposed into local and
global steps, and how the flow coordinates these steps. Later, we'll add the
necessary ingredients in order to be able to run the algorithm in an actual
federated environment, using Exaflow.

Since we will run the algorithm on a single machine we will represent the
federated data as a list of dataframes `data: list[pandas.DataFrame]`.

#### Local computations

A local computation is python function taking local data and returning a
dictionary of local aggregates. The example below demonstrates a local
computation function that calculates the sum and count of some variable.

```python
def mean_local(local_data):
    # Compute sum and count
    sx = local_data.sum()
    n = len(local_data)

    # Pack results into single dictionary which will
    # be transferred to global worker
    results = {"sx": sx, "n": n}
    return results
```

The result packs the two aggregates, `sx` and `n` into a dictionary. A separate
instance of this function will run on each local worker, so `sx` and `n` are
different for each worker and reflect each worker's local data.

#### Global computations

A global computation is also a python function, and it usually takes the output
of a local computation as an input. The local computations, coming from
multiple workers, produce a list of results, from which a global aggregate is
computed. We can then perform further computations, as in the current example
where the mean is computed by dividing the global `sx` with the global `n`.

```python
def mean_global(local_results):
    # Sum aggregates from all workers
    sx = sum(res["sx"] for res in local_results)
    n = sum(res["n"] for res in local_results)

    # Compute global mean
    mean = sx / n

    return mean
```

#### Algorithm flow

The algorithm flow coordinates local and global computations, as well as the
exchange of data between workers. We can write the algorithm flow as a python function
`run` that calls `mean_local` on all local workers and then calls `mean_global` on the
global worker, passing the results of the local computations.

```python
def run(all_data):
    local_results = [mean_local(local_data) for local_data in all_data]
    mean = mean_global(local_results)
    return mean
```

We can then run the algorithm flow like this:

```python
import pandas as pd

x1 = pd.DataFrame({"var": [1, 2, 3]})
x2 = pd.DataFrame({"var": [4, 5, 6]})
all_data = [x1, x2]
print(run(all_data))
```

## Porting to Exaflow

Let's now port the above algorithm to the Exaflow runtime used in
`exaflow/algorithms/exareme3`. The high-level idea remains the same: describe
local computations that run next to each dataset and combine their outputs in
the controller. The implementation details, however, are different from the
older Exareme2 stack and revolve around the `Algorithm` base class,
`@exareme3_udf` decorator, and the `Inputdata` pydantic model.

### Declaring worker steps

Exaflow discovers worker-side logic through the
`exaflow.algorithms.exareme3.exareme3_registry.exareme3_udf` decorator. When you
decorate a function with `@exareme3_udf`, it gets registered under a stable
key, can be dispatched to every participating worker, and receives the
arguments that the flow passes via `engine.run_algorithm_udf`.

Every worker step receives:

- `data`: a pandas `DataFrame` containing the columns referenced by the flow.
- `inputdata`: an `Inputdata` model describing the request (datasets, filters,
  x/y variables). This is useful when the same UDF is reused across algorithms.
- Any additional keyword arguments provided by the flow.

```python
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf


@exareme3_udf()
def mean_local(data, inputdata, column):
  # Keep only the column of interest and drop NA locally
  series = data[column].dropna()
  sx = float(series.sum())
  n = int(len(series))
  return {"sx": sx, "n": n}
```

Whenever a UDF needs access to the secure aggregation service (for operations
such as SUM/MIN/MAX without revealing local payloads), declare it with
`with_aggregation_server=True`. The decorator injects an `agg_client` argument
implementing `ExaflowUDFAggregationClientI`, so the local function can issue
`agg_client.sum(array)` calls without worrying about the privacy plumbing.

### Algorithm flow in Exaflow

The controller orchestrates the algorithm by instantiating an `Algorithm`
subclass and calling its `run()` method. The flow can fan out a worker step to
all workers through `engine.run_algorithm_udf`, inspect the results, and drive
subsequent computations. There is no longer a split between "local" and
"global" UDFs—global logic is written directly in Python inside the flow.

```python
from pydantic import BaseModel
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm


class MeanResult(BaseModel):
  variable: str
  mean: float


class MeanAlgorithm(Algorithm, algname="mean"):
  def run(self):
    column = self.inputdata.x[0]

    worker_payloads = self.run_local_udf(
      func=mean_local,
      kw_args={
        "column": column,
      },
    )

    sx = sum(payload["sx"] for payload in worker_payloads)
    n = sum(payload["n"] for payload in worker_payloads)

    return MeanResult(variable=column, mean=sx / n)
```

`run_algorithm_udf` schedules the registered UDF on every participating worker
and returns a list of their outputs (in worker order). The flow is responsible
for validating inputs (`validation_utils`), checking metadata
(`metadata_utils`), and combining local payloads into the final result.
Additional arguments such as `metadata` or algorithm parameters can be passed
in the `positional_args` dictionary and arrive unchanged on each worker.

### Data loading

Data loading is handled by the runtime: the worker takes the serialized
`Inputdata` object, applies filters/dataset selection, and materializes the
requested columns as a pandas `DataFrame`. This means algorithm authors no
longer implement custom `AlgorithmDataLoader` classes. Instead, you control the
loader via parameters passed alongside the UDF call:

- `dropna` (default `True`) drops rows with missing values on the selected
  columns before the `DataFrame` reaches your UDF. Algorithms such as
  `describe` override this by passing `{"dropna": False}` when calling
  `run_algorithm_udf`.
- `include_dataset` adds the `dataset` column to the `DataFrame` so the UDF can
  emit per-dataset metrics.
- `preprocessing`/`raw_inputdata` allow flows to request alternative views of
  the data (for example, the longitudinal transformer injects subject-id and
  visit-id columns by setting these fields).

If a worker ends up with insufficient rows (because of dataset selection,
filters, or the privacy threshold), it raises `InsufficientDataError` and the
controller skips that worker when aggregating.

### Algorithm Specifications

Finally, we need to define a specification for each algorithm. This contains
information about the public API of each algorithm, such as the number and
types of variables, whether they are required or optional, the names and types
of additional parameters etc.

The full description can be found in
[`exaflow.algorithms.specifications.AlgorithmSpecification`](https://github.com/madgik/exaflow/blob/main/exaflow/algorithms/specifications.py).
The algorithm writer needs to provide a JSON file, with the same name and
location as the file where the algorithm is defined. E.g. for an algorithm
defined in `dir1/dir2/my_algorithm.py` we also create
`dir1/dir2/my_algorithm.json`. The contents of the file are a JSON object with
the exact same structure as
`exaflow.algorithms.specifications.AlgorithmSpecification`.

In our example the specs are

```json
{
    "name": "mean",
    "desc": "Computes the mean of a single variable.",
    "label": "Mean",
    "enabled": true,
    "inputdata": {
        "x": {
            "label": "Variable",
            "desc": "A unique numerical variable.",
            "types": [ "real" ],
            "stattypes": [ "numerical" ],
            "notblank": true,
            "multiple": false
        }
    }
}
```

### Running the algorithm

Once all building blocks are in place, and our system is deployed, we can run
the algorithm either by performing a POST request to Exaflow, or by using
[`run_algorithm`](https://github.com/madgik/exaflow/tree/main/run_algorithm) from the
command line.

# Advanced topics

The previous example is enough to get you started, but Exaflow offers a few
more features that make the implementation of complex algorithms manageable.
This section highlights the parts of `exaflow/algorithms/exareme3` you will
interact with most frequently.

## UDF registry and decorator options

`exaflow.algorithms.exareme3.exareme3_registry` exposes the decorator used in
all flows:

- Every decorated function is registered under a stable key derived from the
  module and the function's qualified name. The controller uses this key to ask
  each worker to run the correct function; duplicate keys raise an error to
  avoid ambiguity.
- `with_aggregation_server=True` injects an `agg_client` argument that
  implements `ExaflowUDFAggregationClientI`. This client exposes `sum/min/max`
  helpers that automatically perform secure aggregation across workers.
- `enable_lazy_aggregation` can override the default batching behavior. By
  default it matches `with_aggregation_server` and wraps the function with
  `library.lazy_aggregation.lazy_agg`, which buffers small payloads to reduce
  network chatter.
- `agg_client_name` lets you use a different parameter name if your UDF already
  has an argument called `agg_client`.

```python
import numpy as np
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf


@exareme3_udf(with_aggregation_server=True)
def xtx_local(data, inputdata, covariates, agg_client):
  X = data[covariates].to_numpy(dtype=float, copy=False)
  payload = np.array([X.T @ X, X.sum(axis=0), [len(X)]], dtype=object)
  xtx, sx, count = agg_client.sum(payload)
  return {"xtx": xtx.tolist(), "sx": sx.tolist(), "count": int(count[0])}
```

The return value of a UDF can be any JSON-serializable object; most flows
return dictionaries or pydantic-friendly structures that are easy to combine in
the controller.

## Metadata and preprocessing helpers

The migrated flows share a small set of helpers to keep `run()` methods tidy:

- `validation_utils` contains simple guards (for example,
  `require_dependent_var`, `require_covariates`, `require_exact_covariates`)
  that raise `BadUserInput` (from `exaflow.worker_communication`) with a
  meaningful error message.
- `metadata_utils.validate_metadata_vars` and
  `validate_metadata_enumerations` assert that the `metadata` dictionary
  contains the keys needed by the algorithm (primarily `is_categorical` and
  category enumerations).
- `preprocessing.get_dummy_categories` runs a helper UDF (usually something
  like `logistic_collect_categorical_levels`) on every worker to discover the
  levels that actually appear in the data. The function merges the outputs,
  drops the reference level, and returns the categories that should be dummy
  encoded.
- `metrics.build_design_matrix`, `construct_design_labels`, and
  `collect_categorical_levels_from_df` encapsulate the encoding logic used by
  regression-like algorithms.
- `crossvalidation.kfold_indices` and `split_dataframe` reimplement the
  splitting strategy used by the historic flows so that cross-validation
  algorithms such as `linear_regression_cv` and
  `naive_bayes_categorical_cv` remain reproducible without pulling additional
  dependencies.

Treat these helpers as building blocks: copy the patterns from
`exaflow/algorithms/exareme3/logistic_regression.py`,
`linear_regression.py`, or `describe.py` when you need similar logic.

## Aggregation server (SUM/MIN/MAX)

The aggregation server provides centralized SUM/MIN/MAX operations across
workers. Algorithms opt in by using the aggregation client described earlier
(`with_aggregation_server=True`). The runtime forwards arrays to the aggregation
service and only the aggregated result is returned to the flow.

```python
import numpy as np
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf


@exareme3_udf(with_aggregation_server=True)
def privacy_preserving_counts(data, agg_client, categories):
  matrix = np.stack(
    [data[cat].value_counts(dropna=False).reindex(categories, fill_value=0)]
  )
  # The aggregation server sums the matrices across workers
  total_counts = agg_client.sum(matrix)
  return {"counts": total_counts.tolist()}
```

Inside the flow you treat the aggregated values like any other result; the main
difference is that you do not see per-worker contributions, which satisfies the
privacy guarantees required by Exaflow deployments.

# Best practices

## Memory efficiency

Worker UDFs receive a pandas `DataFrame` materialized by the runtime (see [UDF
registry and decorator options](#udf-registry-and-decorator-options)). If we are
not careful when writing `udf` functions, we could end up performing numerous
unnecessary copies of the original data, effectively inflating memory usage.

For example, say we want to compute the product of three matrices, `A`, `B` and
`C`, and then sum the elements of the final resulting matrix. The result is a
single float and we ought to be able to allocate just a single float. If we are
not careful and write

```python
result = (A @ B @ C).sum()
```

Python will allocate a new matrix for the result of `A @ B`, then another one for `A @ B @ C`
and then it will sum the elements of this last matrix to obtain the final result.

To overcome this we can use `numpy.einsum` like this

```python
result = numpy.einsum('ij,jk,kl->', A, B, C)
```

This will only allocate a single float!

There are multiple tricks like this one that we can use to reduce the memory
footprint of our UDFs. You can find a few in this
[commit](https://github.com/madgik/exaflow/commit/5d15847898930ac44fcf3be3669b4e74f427d5b5)
together with a short summary of the main ideas.

## Time efficiency

When writing a federated algorithm, we usually decompose an existing algorithm
into local and global computations. This decomposition is not unique in
general, thus the same algorithm can be written in more than one ways. However,
different decompositions might lead to important differences in execution time.
This is related to the overall number of local and global steps. Let's call a
sequence of one local and one global step, a **federation round**.. Each
federation round, together with the required data interchange between workers,
takes a non-negligible amount of time. It is therefore desirable to find a
decomposition that minimizes the number of federation rounds.

Consider, for example, the computation of the [total sum of
squares](https://en.wikipedia.org/wiki/Total_sum_of_squares). This quantity is
needed in many algorithms, such as linear regression or ANOVA. The total sum of
squares (TSS) is given by

```math
\text{TSS} = \sum_i^N (y_i - \hat{y})^2
```

We might think that we have to compute $\\hat{y}$ in a single round, then share the
result with the local workers, and finally compute the TSS in a second round.
But in fact, the whole computation can be done in a single round. We first develop
the square of the difference.

```math
\text{TSS} = \sum_i^N y_i^2 - 2 \hat{y} \sum_i^N y_i + N \hat{y}^2
```

It follows that we must compute the sum, the sum of squares and $N$, in the
local step. Then, in the global step, we can compute $\\hat{y}$ and the above
expression for the TSS. We managed to compute the result in a single federation
round, instead of two.

## Privacy

Privacy is a very subtle subject. In the context of federated analytics it
means to protect individual datapoints residing in local workers, as these might
correspond to a person's sensitive personal data. Exaflow allows the user to
learn something about a large group of individuals through statistics, but
tries to prevent any individual's information to leak.

In order to achieve this we follow a few guidelines to ensure maximum
protection of sensible data.

#### Privacy threshold

First, there is a threshold of datapoints that a worker needs in order to
participate in a computation. No worker with datapoints below this threshold is
allowed to participate. This threshold is system wide (its value is usually
around 10) and the algorithm writer has no control over it.

#### Share only aggregates

The most important guideline you should follow, as an algorithm writer, is to
allow local workers to share only aggregates of the sensitive data they hold, and
never the actual data! Every variable in the database contains multiple values,
one for each individual. When some variable is involved in a computation we
must make sure that we only share the result of some function that aggregates
these multiple values into a single value.

In (informal) mathematical language, we can say that, given a number of datapoints $N$,
we want functions whose output size is $O(1)$ and not $O(N)$. For example
`sum`, when applied to a continuous variable, is $\\texttt{sum} : \\mathbb{R}^N \\rightarrow \\mathbb{R}$
and `count`, when applied to a categorical variable with categories $C$, is
$\\texttt{count} : C^N \\rightarrow \\mathbb{N}$. Both these functions have codomains that are
independent of $N$, so they are good examples of aggregate functions. Other examples are `min` and `max`.

These functions are not limited to plain sums of counts, however. Let's say,
for example, that we have a $N \\times p$ design matrix $\\mathbf{X}$, where
$N$ is the number of rows and $p$ is the number of variables. It is often the
case in statistics that we need to compute the quantity $\\mathbf{X}^\\intercal \\cdot \\mathbf{X}$.
Although not scalar, this quantity complies with our definition of aggregate,
since all elements of the resulting matrix are themselves aggregates over $N$,
hence the result size is $O(p^2)$ and doesn't depend on $N$.

# Examples

## Iterative algorithm

In the previous sections we presented a very simple algorithm for computing the
mean of some variable. This algorithm requires a single worker call. More
complex workflows are possible, such as an iterative workflow. Below is an
example of how to structure code to achieve this. For a real world example you
should see the `run_distributed_logistic_regression` helper used by the
[logistic regression
algorithm](https://github.com/madgik/exaflow/blob/main/exaflow/algorithms/exareme3/logistic_regression.py).

```python
from pydantic import BaseModel
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
from exaflow.algorithms.exareme3.utils.registry import exareme3_udf


class IterationResult(BaseModel):
  value: float


@exareme3_udf()
def update_local(data, column, val):
  gradient = data[column].mean() - val
  return {"gradient": gradient}


class MyAlgorithm(Algorithm, algname="iterative"):
  def run(self):
    val = 0.0
    while True:
      local_results = self.run_local_udf(
        func=update_local,
        kw_args={
          "column": self.inputdata.x[0],
          "val": val,
        },
      )
      # Combine local gradients and update the iterate
      gradient = sum(result["gradient"] for result in local_results)
      val -= 0.1 * gradient
      if abs(gradient) < 1e-3:
        break

    return IterationResult(value=val)
```

Here we initialize `val` to 0 and start the iteration. In each step the flow
passes the current value to the worker UDF, sums the gradients it receives
back, and updates the iterate locally. The stopping criterion is computed
entirely in the controller (`abs(gradient) < 1e-3` in this toy example).

## Class based model

When the algorithm logic becomes too complex, we might want to abstract some parts into separate
classes. This is possible and advised. For example, when the algorithm makes use of a supervised learning
model, e.g. [linear regression](https://github.com/madgik/exaflow/blob/main/exaflow/algorithms/exareme3/linear_regression.py),
we can create a separate class for the model and call it from within the algorithm class.
Typically, a model implements two methods, `fit` and `predict`. The first does the learning by fitting the
model to the data, and the second does prediction on new data. Both methods could be used in a cross-validation
algorithm, see for example [here](https://github.com/madgik/exaflow/blob/main/exaflow/algorithms/exareme3/linear_regression_cv.py).

```python
from pydantic import BaseModel
from exaflow.algorithms.exareme3.utils.algorithm import Algorithm


class MyModel:
  def __init__(self, engine):
    self.engine = engine

  def fit(self, inputdata_json):
    self.run_local_udf(
      func=some_local_training_step,
      kw_args={"inputdata": inputdata_json},
    )

  def predict(self, inputdata_json):
    return self.run_local_udf(
      func=some_predict_step,
      kw_args={"inputdata": inputdata_json},
    )


class PredictionResult(BaseModel):
  predictions: list


class MyAlgorithm(Algorithm, algname="complex_algorithm"):
  def run(self):
    model = MyModel(self.run_local_udf)
    model.fit(self.inputdata.json())

    new_inputdata = self.inputdata.copy(update={"datasets": ["validation"]})
    predictions = model.predict(new_inputdata.json())
    return PredictionResult(predictions=predictions)
```

## Customizing worker inputs

While the worker takes care of loading `data` based on `Inputdata`, algorithms
often need subtle tweaks. Instead of writing a custom data loader you can pass
control flags alongside the UDF call:

- `dropna=False` keeps rows with missing values. See
  `describe.DescriptiveStatisticsAlgorithm`, which needs both the raw
  counts and the rows without missing values.
- `include_dataset=True` ensures the `dataset` column is available, something
  histograms and descriptive statistics rely on for per-dataset reports.
- The `preprocessing` argument can carry instructions understood by the worker.
  For example, algorithms using the longitudinal transformer
  (`exaflow/algorithms/exareme3/longitudinal_transformer.py`) pass a dictionary
  describing which raw variables (subject id, visit id, etc.) should be joined
  into the final `DataFrame`.

Because these knobs live next to the `run_algorithm_udf` invocation, the logic
is visible right where it matters and keeps the data-loading behavior tied to
the worker step that consumes it.
