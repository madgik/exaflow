# Lazy Aggregation

`lazy_aggregation.py` lets algorithms call the aggregation client one value at
a time and still send independent aggregations in one round-trip.

The worker wraps every UDF registered with lazy aggregation (the default for
`with_aggregation_server=True`) in `lazy_agg`. On each call `lazy_agg`:

1. wraps the `agg_client` argument in a `LazyAggregationClient`, which queues
   submitted operations and sends the whole queue with one `aggregate_batch`
   call when any result is needed;
1. runs a rewritten copy of the UDF. Every project function or method it calls
   (anything outside the standard library and site-packages) also runs
   rewritten, so helpers such as `NumpyAggregator` or the federated cores batch
   too. Rewrites are cached per function.

## Rewrite

```python
x = np.asarray(agg_client.sum(a), dtype=float)
y = np.asarray(agg_client.sum(b), dtype=float)
```

becomes

```python
_lazy_tmp0 = _lazy_submit(agg_client, "sum", a)
_lazy_tmp1 = _lazy_submit(agg_client, "sum", b)
x = np.asarray(_lazy_force(_lazy_tmp0), dtype=float)  # one round-trip: [a, b]
y = np.asarray(_lazy_force(_lazy_tmp1), dtype=float)
```

- **Extraction**: `<client>.sum|min|max|union(value)` calls become submissions,
  and calls to project functions become deferrals (`_lazy_defer` /
  `_lazy_settle`). Only always-evaluated parts of a statement are extracted:
  never the right side of `and`/`or`, the branches of `a if c else b`, later
  links of chained comparisons, comprehensions or lambdas.

- **Scheduling**: outside `try` bodies, consecutive statements without side
  effects (pure assignments, submissions, and `if`/`for`/`while` statements made
  of those) are reordered by data dependencies: submissions first, statements
  that use results last. Calls with unknown side effects, stores into objects,
  `return`, `raise` and `try` stay where they are.

- **Deferred variants**: a function whose body ends with pure statements that
  only use aggregation results (`return np.where(total == 0, np.nan, total)`,
  also in every branch of a final `if`) has a second version that returns that
  tail as a `_LazyTail`. Consecutive helper calls then share round-trips:

  ```python
  lo = float(aggregator.global_min(x))  # min and max go out together
  hi = float(aggregator.global_max(x))
  ```

## Guarantees

- Each result is forced at the original call site, so every worker sends the
  same operations as the eager code, only grouped into fewer batches. This
  relies on algorithms keeping workers in lock-step: branches that depend on
  local data must make the same aggregation calls in the same structure.
- Values are copied when submitted, so later mutations do not change what is
  sent.
- Submissions whose call site is never reached (an exception was raised first)
  are dropped and never sent.
- Calls on objects that are not aggregation clients (`array.sum(0)`) run as
  usual.
- Tracebacks keep the original file and line numbers.
- `aggregate_batch` errors propagate; clients without `aggregate_batch` get one
  call per operation.

## Limits

- Loop iterations are not interleaved: a loop whose body needs an aggregation
  result costs at least one round-trip per iteration.
- A function call with unknown side effects, a store into an object or a
  `raise`/`return` between two aggregations keeps them in separate batches.
- Functions with closures, name-mangled attributes or unavailable source run
  unchanged (their aggregations are sent one by one).

## Tests

- `tests/standalone_tests/lazy_aggregation/test_lazy_aggregation_patterns.py`:
  rewrite rules.
- `tests/standalone_tests/lazy_aggregation/test_lazy_aggregation_algorithms.py`:
  every aggregation-server algorithm, three in-process workers, the real
  aggregation server servicer; lazy results must equal eager results and stay
  within `MAX_LAZY_ROUNDS` round-trips.
