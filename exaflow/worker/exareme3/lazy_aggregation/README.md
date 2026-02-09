# Lazy Aggregation

`lazy_aggregation.py` rewrites functions at call time to batch global aggregation calls (`agg_client.sum/min/max`) when it is safe to do so, reducing aggregation roundtrips while preserving semantics and debuggability.

## How it works

- The `@lazy_agg` decorator wraps a function and uses `LazyAggregationRewriter` to inspect its source AST on first call.
- Consecutive assignments of the form `var = agg_client.sum|min|max(expr)` that are independent are grouped into a single batch executed by `LazyAggregationExecutor`.
- Agg calls buried inside expressions (e.g., `min([agg_client.min(x)[0], ...])`) are hoisted to temporaries first, so they can be batched while preserving evaluation order. Comprehension bodies are not hoisted to avoid scoping changes.
- If `agg_client.aggregate_batch` exists, it is used; otherwise, execution falls back to individual `sum`/`min`/`max` calls. Single-op batches are also executed directly to avoid noisy `batch(1)` recordings.
- Injected AST nodes are location-copied and line-offset so tracebacks still point to the original source file/line.
- Custom aggregation client names are supported via `@lazy_agg(agg_client_name="client")`.

## Behaviors covered by tests

- **Basic batching**: Independent globals are batched; dependent ones stay ordered.
- **Globals + locals**: Respect dependencies through `_used_names` tracking.
- **Side effects**: Any non-assignment statement between globals forces a flush; no batching across side effects.
- **Loops**: Per-iteration batches occur when two globals sit consecutively inside the loop body.
- **Try/except/else/finally**: Rewriter recurses into nested bodies.
- **Type annotations**: `AnnAssign` targets participate in batching.
- **Min/Max ops**: `min`/`max` calls batch together like `sum`.
- **Expression hoisting**: Agg calls inside expressions are hoisted to temps for batching (excluding comprehensions).
- **Batch fallback**: If `aggregate_batch` raises, execution reverts to eager individual calls.
- **Custom client name**: Works when the aggregation client parameter is named differently.
- **Tracebacks**: Exceptions report original file/line numbers post-rewrite.
- **Globals**: Mutations of globals force batching to flush to preserve order; globals can be read safely.
- **Embedded calls**: Previously eager-only; now hoisted for batching when safe (non-comprehension expressions).

## Usage notes

- Write aggregation calls as direct assignments to enable batching, or keep agg calls in expressions that can be safely hoisted:
  ```python
  @lazy_agg()
  def foo(agg_client, a, b):
      s1 = agg_client.sum(a)
      s2 = agg_client.sum(b)   # batched with s1 if independent
      return s1, s2

  @lazy_agg()
  def bar(agg_client, x, y):
      # expression hoisting: agg calls inside min/max are hoisted then batched
      lo = min([agg_client.min(x)[0], agg_client.min(y)[0]])
      hi = max([agg_client.max(x)[0], agg_client.max(y)[0]])
      return lo, hi
  ```
- If you want batching inside larger expressions, either hoist to temps yourself or rely on the built-in hoisting (it will skip comprehensions).
- The rewriter needs access to the function’s source; dynamically generated or already-rewritten functions will be left untouched.

## Rewrite log

Every time `lazy_agg` rewrites a function, the rewritten source is appended to `lazy_agg_rewrites.log` (in the same directory as `lazy_aggregation.py`). Override the destination by setting `LAZY_AGG_REWRITE_LOG=/path/to/log`. Logging failures are ignored so rewriting is never blocked.
