"""
Lazy aggregation: fewer aggregation-server round-trips for unchanged UDF code.

`lazy_agg` runs a rewritten copy of a UDF, and of every project function it
calls, in which each aggregation call (`<client>.sum|min|max|union(values)`) is
split into a submission and a force at its original site:

    x = np.asarray(agg_client.sum(a), dtype=float)
    y = np.asarray(agg_client.sum(b), dtype=float)

becomes

    _lazy_tmp0 = _lazy_submit(agg_client, "sum", a)
    _lazy_tmp1 = _lazy_submit(agg_client, "sum", b)       # hoisted
    x = np.asarray(_lazy_force(_lazy_tmp0), dtype=float)  # one round-trip: [a, b]
    y = np.asarray(_lazy_force(_lazy_tmp1), dtype=float)

Calls to project functions are split the same way (`_lazy_defer` / `_lazy_settle`).
A function whose body ends with a pure tail computed only from aggregation
results (for example `return np.where(result == 0, np.nan, result)`) also gets a
deferred variant that runs everything up to the tail and returns the tail as a
`_LazyTail`, so consecutive helper calls share round-trips too:

    lo = float(aggregator.global_min(x))   # both submissions go out
    hi = float(aggregator.global_max(x))   # in one round-trip

See `_FunctionRewriter` for the rules that keep the rewrite equivalent to the
original code: every worker sends exactly the aggregation operations of the
eager code, in the same order, only grouped into fewer batches.
"""

import ast
import builtins
import copy
import inspect
import sysconfig
import textwrap
import weakref
from functools import lru_cache
from functools import wraps
from types import FunctionType
from types import MethodType

import numpy as np

from exaflow.aggregation_clients import AggregationType

AGGREGATION_OPS = {
    "sum": AggregationType.SUM,
    "min": AggregationType.MIN,
    "max": AggregationType.MAX,
    "union": AggregationType.UNION,
}


class _Pending:
    __slots__ = ("client", "value", "done")

    def __init__(self, client, value=None, done=False):
        self.client = client
        self.value = value
        self.done = done


class _LazyTail:
    """The pure tail of a function call, run when its result is first needed."""

    __slots__ = ("fn", "handles", "value", "done")

    def __init__(self, fn, handles):
        self.fn = fn
        self.handles = handles
        self.value = None
        self.done = False


class LazyAggregationClient:
    """
    Aggregation client proxy that queues submitted operations and sends every
    queued operation in one `_aggregate_batch` call when any result is needed.
    Direct `sum/min/max/union` calls are sent immediately, together with the queue.
    It exposes only `sum/min/max/union` to the UDF; it does not forward other
    attributes to the wrapped client.
    """

    def __init__(self, client):
        self._client = client
        self._pending = []

    def submit(self, op: str, values) -> _Pending:
        pending = _Pending(self)
        # Copied: the caller may mutate `values` before the batch is sent.
        self._pending.append((op, copy.deepcopy(values), pending))
        return pending

    def discard(self, pending: _Pending) -> None:
        self._pending = [item for item in self._pending if item[2] is not pending]

    def flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        aggregate_batch = getattr(self._client, "_aggregate_batch", None)
        if aggregate_batch is None:
            results = [getattr(self._client, op)(values) for op, values, _ in batch]
        else:
            results = aggregate_batch(
                [(AGGREGATION_OPS[op], values) for op, values, _ in batch]
            )
        for (_, _, pending), result in zip(batch, results, strict=True):
            pending.value = result
            pending.done = True

    def _aggregate_now(self, op: str, values):
        pending = self.submit(op, values)
        self.flush()
        return pending.value

    def sum(self, values):
        return self._aggregate_now("sum", values)

    def min(self, values):
        return self._aggregate_now("min", values)

    def max(self, values):
        return self._aggregate_now("max", values)

    def union(self, values):
        return self._aggregate_now("union", values)


# --- helpers called by rewritten code ----------------------------------------- #


def _lazy_submit(receiver, op, values):
    if isinstance(receiver, LazyAggregationClient):
        return receiver.submit(op, values)
    # Not an aggregation client (e.g. `array.sum(0)`): run it now.
    return _Pending(None, getattr(receiver, op)(values), True)


def _lazy_force(handle: _Pending):
    if not handle.done:
        handle.client.flush()
    return handle.value


def _lazy_settle(result):
    if isinstance(result, _LazyTail):
        if not result.done:
            result.value = result.fn()
            result.done = True
        return result.value
    return result


def _lazy_discard(*handles):
    """Drops queued submissions whose original site was never reached."""
    for handle in handles:
        if isinstance(handle, _Pending) and not handle.done:
            handle.client.discard(handle)
        elif isinstance(handle, _LazyTail) and not handle.done:
            _lazy_discard(*handle.handles)


def _lazy_call(func):
    """The rewritten version of project functions, `func` otherwise."""
    return _rewritten(func, deferred=False)


def _lazy_defer(func):
    """Like `_lazy_call`, preferring the variant that returns a `_LazyTail`."""
    return _rewritten(func, deferred=True)


def _rewritten(func, *, deferred: bool):
    if type(func) is MethodType:
        rewritten = _rewritten_function(func.__func__, deferred)
        if rewritten is func.__func__:
            return func
        return MethodType(rewritten, func.__self__)
    if type(func) is FunctionType:
        return _rewritten_function(func, deferred)
    return func


_HELPERS = {
    "_lazy_submit": _lazy_submit,
    "_lazy_force": _lazy_force,
    "_lazy_settle": _lazy_settle,
    "_lazy_discard": _lazy_discard,
    "_lazy_call": _lazy_call,
    "_lazy_defer": _lazy_defer,
    "_LazyTail": _LazyTail,
}
_PURE_HELPERS = {"_lazy_force", "_lazy_settle"}
_TMP_PREFIX = "_lazy_tmp"

# --- rewriting ---------------------------------------------------------------- #

_PURE_BUILTINS = {
    id(getattr(builtins, name))
    for name in """
        abs all any bool complex dict divmod enumerate float format frozenset
        hasattr hash int isinstance issubclass len list max min pow range repr
        reversed round set slice sorted str sum tuple type zip
    """.split()
}
_PURE_LIBRARIES = {"numpy", "scipy", "pandas", "math"}
_IMPURE_LIBRARY_NAMES = set(
    """
    apply_along_axis apply_over_axes copyto fill_diagonal fromfunction frompyfunc
    piecewise place put putmask save savetxt savez seterr vectorize
    """.split()
)
# Methods that return new values without mutating their receiver or arguments.
_PURE_METHODS = set(
    """
    all any argmax argmin argsort astype copy count cumprod cumsum diagonal
    difference dot dropna endswith fillna flatten format get intersection isin isna
    isnull issubset item items join keys lower max mean median min nonzero notna
    notnull nunique prod ravel replace reshape round sort_values split squeeze
    startswith std strip sum to_list to_numpy tolist transpose union unique upper
    value_counts values var
    """.split()
)
_MUTATING_KEYWORDS = {"out", "inplace"}
_LIBRARY_PATHS = tuple(
    {sysconfig.get_paths()[key] for key in ("stdlib", "platstdlib", "purelib")}
)
_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_REWRITTEN_CODES = set()
_REWRITES = weakref.WeakKeyDictionary()
_MISSING = object()


@lru_cache(maxsize=None)
def _is_project_file(filename: str) -> bool:
    return not filename.startswith(("<", *_LIBRARY_PATHS))


def _rewritten_function(func: FunctionType, deferred: bool) -> FunctionType:
    if func.__code__ in _REWRITTEN_CODES:
        return func
    variants = _REWRITES.get(func)
    if variants is None:
        variants = _rewrite(func) or (func, func)
        _REWRITES[func] = variants
    return variants[deferred]


def _rewrite(func: FunctionType):
    """Returns (rewritten, deferred variant) or None when `func` stays as is."""
    code = func.__code__
    if (
        func.__closure__ is not None
        or not code.co_names
        or not _is_project_file(code.co_filename)
    ):
        return None
    try:
        source_lines, first_line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        return None
    module = ast.parse(textwrap.dedent("".join(source_lines)))
    fdef = module.body[0]
    if not isinstance(fdef, ast.FunctionDef) or fdef.name != func.__name__:
        return None
    if any(
        _is_mangled(node.id if isinstance(node, ast.Name) else node.attr)
        for node in ast.walk(fdef)
        if isinstance(node, (ast.Name, ast.Attribute))
    ):
        return None
    ast.increment_lineno(module, first_line - 1)

    tmp_names, deferred_body = _FunctionRewriter(func.__globals__).function(fdef)
    fdef.decorator_list = []
    fdef.returns = None
    fdef.args.defaults = []
    fdef.args.kw_defaults = [None] * len(fdef.args.kwonlyargs)
    for arg in ast.walk(fdef.args):
        if isinstance(arg, ast.arg):
            arg.annotation = None

    rewritten = _compile(_with_body(fdef, fdef.body, tmp_names), func)
    if deferred_body is None:
        return rewritten, rewritten
    return rewritten, _compile(_with_body(fdef, deferred_body, tmp_names), func)


def _compile(fdef, func: FunctionType) -> FunctionType:
    # The factory closes over the helpers, so module globals are never touched.
    factory = ast.FunctionDef(
        name="_lazy_factory",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(name) for name in _HELPERS],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[fdef, ast.Return(ast.Name(fdef.name, ast.Load()))],
        decorator_list=[],
    )
    module = ast.Module(body=[ast.copy_location(factory, fdef)], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(
        compile(module, func.__code__.co_filename, "exec", dont_inherit=True),
        func.__globals__,
        namespace,
    )
    rewritten = namespace["_lazy_factory"](*_HELPERS.values())
    rewritten.__defaults__ = func.__defaults__
    rewritten.__kwdefaults__ = func.__kwdefaults__
    rewritten.__doc__ = func.__doc__
    rewritten.__qualname__ = func.__qualname__
    rewritten.__annotations__ = func.__annotations__
    rewritten.__dict__.update(func.__dict__)
    _register_codes(rewritten.__code__)
    return rewritten


def _with_body(fdef, body, tmp_names):
    definition = copy.deepcopy(fdef)
    definition.body = _discard_on_exception(copy.deepcopy(body), tmp_names)
    return definition


def _deferred_body(body, tail_start: int):
    """Runs the head and returns the pure tail as a `_LazyTail`."""
    head, tail = body[:tail_start], body[tail_start:]
    tail_fn = ast.FunctionDef(
        name="_lazy_tail",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=tail,
        decorator_list=[],
    )
    handles = sorted(
        name
        for name in _load_names(ast.Module(tail, []))
        if name.startswith(_TMP_PREFIX)
    )
    result = ast.Return(
        ast.Call(
            ast.Name("_LazyTail", ast.Load()),
            [
                ast.Name("_lazy_tail", ast.Load()),
                ast.Tuple([ast.Name(name, ast.Load()) for name in handles], ast.Load()),
            ],
            [],
        )
    )
    for node in (tail_fn, result):
        ast.copy_location(node, tail[0])
    return [*head, tail_fn, result]


def _discard_on_exception(body, tmp_names):
    if not tmp_names:
        return body
    location = body[0]
    init = ast.Assign(
        targets=[ast.Name(name, ast.Store()) for name in tmp_names],
        value=ast.Constant(None),
    )
    handler = ast.ExceptHandler(
        type=ast.Name("BaseException", ast.Load()),
        name=None,
        body=[
            ast.Expr(
                ast.Call(
                    ast.Name("_lazy_discard", ast.Load()),
                    [ast.Name(name, ast.Load()) for name in tmp_names],
                    [],
                )
            ),
            ast.Raise(),
        ],
    )
    wrapper = ast.Try(body=[init, *body], handlers=[handler], orelse=[], finalbody=[])
    for node in [*ast.walk(handler), init, wrapper]:
        ast.copy_location(node, location)
    return [wrapper]


def _register_codes(code) -> None:
    _REWRITTEN_CODES.add(code)
    for const in code.co_consts:
        if inspect.iscode(const):
            _register_codes(const)


def _is_mangled(name: str) -> bool:
    return name.startswith("__") and not name.endswith("__")


def _load_names(node) -> frozenset:
    return frozenset(
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    )


def _store_names(node) -> set:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }


def _walk_scope(node):
    """ast.walk that does not enter nested functions, lambdas or classes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _SCOPE_BOUNDARIES):
            yield from _walk_scope(child)


def _local_names(fdef) -> set:
    args = fdef.args
    names = {
        arg.arg
        for arg in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *(arg for arg in (args.vararg, args.kwarg) if arg is not None),
        ]
    }
    for node in ast.walk(fdef):
        if node is fdef:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


class _ExtractState:
    __slots__ = ("blocked", "unsafe", "statements")

    def __init__(self):
        self.blocked = False  # an impure expression was evaluated
        self.unsafe = False  # a local value was read
        self.statements = []


class _FunctionRewriter:
    """
    Rewrites one function definition (and the functions nested in it).

    Extraction. In `x = expr`, `return expr`, `if expr:` ... statements:
      - an aggregation call with pure receiver and argument becomes
        `_lazy_tmpN = _lazy_submit(receiver, op, argument)` (a submission) placed
        before the statement, and `_lazy_force(_lazy_tmpN)` at the call site;
      - another (intercepted) call with pure arguments becomes
        `_lazy_tmpN = _lazy_defer(func)(arguments)` (a deferral) and
        `_lazy_settle(_lazy_tmpN)` at the call site, when nothing evaluated
        before it in the statement reads a local value.
      Only always-evaluated parts are extracted (not the right side of
      `and`/`or`, the branches of `a if c else b`, chained comparisons,
      comprehensions or lambdas), and nothing after an impure part.

    Scheduling. Outside `try` bodies, each run of consecutive movable statements
    (submissions, deferrals, pure assignments and `if`/`for`/`while` statements
    made of those) is reordered: submissions and deferrals as early as their
    data dependencies allow, statements that force results as late as possible.
    Pure statements have no side effects, so only name dependencies order them.
    A deferral runs arbitrary code, so it keeps its order with every statement
    that reads a local value. Calls, stores into objects and other side effects
    are never crossed, and each result is still forced at its original site.

    Interception. Remaining calls go through `_lazy_call`, so the project
    functions they reach run rewritten too.

    Deferred variant. When the function ends with pure assignments and a
    `return` that read only aggregation results (in every branch of a final
    `if`, possibly), `_lazy_defer` callers get a variant returning that tail as a
    `_LazyTail`.
    """

    def __init__(self, globals_: dict):
        self._globals = globals_
        self._scopes = []
        self._tmp_counter = 0

    # --- function / blocks ------------------------------------------------- #

    def function(self, fdef):
        """
        Rewrites `fdef` in place (the body is not wrapped by
        `_discard_on_exception`); returns its temporaries and the body of its
        deferred variant (None when there is none).
        """
        self._scopes.append(_local_names(fdef))
        first_tmp = self._tmp_counter
        fdef.decorator_list = [
            decorator
            for decorator in fdef.decorator_list
            if "lazy_agg" not in _load_names(decorator)
        ]
        fdef.body = self._block(fdef.body, hoist=True)
        for stmt in fdef.body:
            self._intercept(stmt)
        is_generator = any(
            isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await))
            for stmt in fdef.body
            for node in _walk_scope(stmt)
        )
        deferred_body = None if is_generator else self._deferred_block(fdef.body)
        self._scopes.pop()
        tmp_names = [
            f"{_TMP_PREFIX}{idx}" for idx in range(first_tmp, self._tmp_counter)
        ]
        return tmp_names, deferred_body

    def _block(self, stmts, *, hoist: bool):
        out = []
        segment = []
        for stmt in stmts:
            self._children(stmt, hoist=hoist)
            for new_stmt in self._extract(stmt):
                if hoist and self._movable(new_stmt):
                    segment.append(new_stmt)
                else:
                    out += self._schedule(segment)
                    segment = []
                    out.append(new_stmt)
        return out + self._schedule(segment)

    def _children(self, stmt, *, hoist: bool) -> None:
        if isinstance(stmt, (ast.If, ast.For, ast.While)):
            stmt.body = self._block(stmt.body, hoist=hoist)
            stmt.orelse = self._block(stmt.orelse, hoist=hoist)
        elif isinstance(stmt, ast.With):
            # A context manager may swallow exceptions, unless it is a known pure
            # one such as numpy.errstate.
            pure_context = all(self._pure(item.context_expr) for item in stmt.items)
            stmt.body = self._block(stmt.body, hoist=hoist and pure_context)
        elif isinstance(stmt, ast.Try):
            stmt.body = self._block(stmt.body, hoist=False)
            for handler in stmt.handlers:
                handler.body = self._block(handler.body, hoist=hoist)
            stmt.orelse = self._block(stmt.orelse, hoist=hoist)
            stmt.finalbody = self._block(stmt.finalbody, hoist=hoist)
        elif isinstance(stmt, ast.FunctionDef):
            tmp_names, _ = self.function(stmt)
            stmt.body = _discard_on_exception(stmt.body, tmp_names)

    # --- scheduling --------------------------------------------------------- #

    def _movable(self, stmt, *, in_loop: bool = False) -> bool:
        if hasattr(stmt, "_lazy_reads") or isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return in_loop
        if isinstance(stmt, ast.Expr):
            return not isinstance(stmt.value, ast.Constant) and self._pure(stmt.value)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return _binds_names_only(stmt) and (
                stmt.value is None or self._pure(stmt.value)
            )
        if isinstance(stmt, ast.If):
            header, nested_loop = stmt.test, in_loop
        elif isinstance(stmt, ast.While):
            header, nested_loop = stmt.test, True
        elif isinstance(stmt, ast.For) and _binds_names_only(stmt):
            header, nested_loop = stmt.iter, True
        else:
            return False
        # Deferrals run arbitrary code: inside a moved compound statement they would
        # be reordered with everything around it.
        return self._pure(header) and all(
            self._movable(child, in_loop=nested_loop) and not _has_deferral(child)
            for child in [*stmt.body, *stmt.orelse]
        )

    def _schedule(self, segment):
        """Orders `segment` putting submissions first and forces last."""
        reads = [_load_names(stmt) for stmt in segment]
        binds = [_store_names(stmt) for stmt in segment]
        defers = [getattr(stmt, "_lazy_defer", False) for stmt in segment]
        # Whether a deferral (arbitrary code) may run before/after the statement:
        # it must read no value that code could mutate. A submission's receiver is
        # the aggregation client, so only its argument counts.
        inert = [
            all(
                self._inert(name)
                for name in (
                    _load_names(stmt.value.args[2])
                    if hasattr(stmt, "_lazy_reads") and not stmt._lazy_defer
                    else reads[idx]
                )
            )
            for idx, stmt in enumerate(segment)
        ]
        priority = [
            0
            if hasattr(stmt, "_lazy_reads")
            else 2
            if any(_is_helper_call(node, _PURE_HELPERS) for node in ast.walk(stmt))
            else 1
            for stmt in segment
        ]
        before = [
            {
                i
                for i in range(j)
                if binds[i] & (reads[j] | binds[j])
                or reads[i] & binds[j]
                or (defers[i] and (defers[j] or not inert[j]))
                or (defers[j] and not inert[i])
            }
            for j in range(len(segment))
        ]
        order = []
        while len(order) < len(segment):
            done = set(order)
            order.append(
                min(
                    (
                        j
                        for j in range(len(segment))
                        if j not in done and before[j] <= done
                    ),
                    key=lambda j: (priority[j], j),
                )
            )
        return [segment[j] for j in order]

    # --- deferred variant --------------------------------------------------- #

    def _deferred_block(self, body):
        """`body` with its final pure tails returned as `_LazyTail`s, or None."""
        if not body:
            return None
        last = body[-1]
        if isinstance(last, ast.If):
            then_body = self._deferred_block(last.body)
            else_body = self._deferred_block(last.orelse)
            if then_body is None and else_body is None:
                return None
            deferred_if = copy.copy(last)
            deferred_if.body = then_body or last.body
            deferred_if.orelse = else_body or last.orelse
            return [*body[:-1], deferred_if]
        if not isinstance(last, ast.Return) or last.value is None:
            return None
        for start in range(len(body)):
            if self._valid_tail(body[start:]):
                return _deferred_body(body, start)
        return None

    def _valid_tail(self, tail) -> bool:
        """Pure statements reading only aggregation results, ending in a return."""
        bound = set()
        has_result = False
        for stmt in tail:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if hasattr(stmt, "_lazy_reads") or not _binds_names_only(stmt):
                    return False
            elif not isinstance(stmt, ast.Return):
                return False
            if stmt.value is None or not self._pure(stmt.value):
                return False
            if any(
                self._is_local(name) and name not in bound
                for name in _load_names(stmt.value)
            ):
                return False
            has_result = has_result or any(
                _is_helper_call(node, _PURE_HELPERS) for node in ast.walk(stmt.value)
            )
            bound |= _store_names(stmt)
        return has_result

    # --- extraction --------------------------------------------------------- #

    def _extract(self, stmt):
        state = _ExtractState()
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)):
            if stmt.value is not None:
                stmt.value = self._x(stmt.value, state)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            state.unsafe = True  # the target is read first
            stmt.value = self._x(stmt.value, state)
        elif isinstance(stmt, ast.If):
            stmt.test = self._x(stmt.test, state)
        elif isinstance(stmt, ast.For):
            stmt.iter = self._x(stmt.iter, state)
        return [*state.statements, stmt]

    def _is_candidate(self, node) -> bool:
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in AGGREGATION_OPS
            and len(node.args) == 1
            and not isinstance(node.args[0], ast.Starred)
            and not node.keywords
            and self._resolve(node.func.value) is _MISSING
        )

    def _x(self, node, state: _ExtractState):
        """Extracts calls from `node`, visiting it in evaluation order."""
        if state.blocked:
            return node
        if isinstance(node, ast.Call):
            return self._x_call(node, state)
        if isinstance(node, ast.Name):
            if self._is_local(node.id):
                state.unsafe = True
        elif isinstance(node, (ast.Attribute, ast.Starred)):
            node.value = self._x(node.value, state)
        elif isinstance(node, ast.Subscript):
            node.value = self._x(node.value, state)
            node.slice = self._x(node.slice, state)
        elif isinstance(node, ast.Slice):
            for field in ("lower", "upper", "step"):
                if getattr(node, field) is not None:
                    setattr(node, field, self._x(getattr(node, field), state))
        elif isinstance(node, ast.BinOp):
            node.left = self._x(node.left, state)
            node.right = self._x(node.right, state)
        elif isinstance(node, ast.UnaryOp):
            node.operand = self._x(node.operand, state)
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            node.elts = [self._x(elt, state) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            for idx, (key, value) in enumerate(zip(node.keys, node.values)):
                if key is not None:
                    node.keys[idx] = self._x(key, state)
                node.values[idx] = self._x(value, state)
        elif isinstance(node, ast.JoinedStr):
            node.values = [self._x(value, state) for value in node.values]
        elif isinstance(node, ast.FormattedValue):
            node.value = self._x(node.value, state)
        elif isinstance(node, ast.BoolOp):
            node.values[0] = self._x(node.values[0], state)
            self._conditional(node.values[1:], state)
        elif isinstance(node, ast.Compare):
            node.left = self._x(node.left, state)
            node.comparators[0] = self._x(node.comparators[0], state)
            self._conditional(node.comparators[1:], state)
        elif isinstance(node, ast.IfExp):
            node.test = self._x(node.test, state)
            self._conditional([node.body, node.orelse], state)
        else:
            self._conditional([node], state)
        return node

    def _conditional(self, nodes, state: _ExtractState) -> None:
        """Parts that are not extracted: they only taint the state."""
        if not all(self._pure(node) for node in nodes):
            state.blocked = True
        if any(self._is_local(name) for node in nodes for name in _load_names(node)):
            state.unsafe = True

    def _x_call(self, node, state: _ExtractState):
        unsafe_before = state.unsafe
        if (
            self._is_candidate(node)
            and self._pure(node.func.value)
            and self._pure(node.args[0])
        ):
            receiver = self._x(node.func.value, state)
            values = self._x(node.args[0], state)
            state.unsafe = unsafe_before
            return self._lift(
                node,
                ast.Call(
                    ast.Name("_lazy_submit", ast.Load()),
                    [receiver, ast.Constant(node.func.attr), values],
                    [],
                ),
                "_lazy_force",
                state,
                defer=False,
            )
        node.func = self._x(node.func, state)
        node.args = [self._x(arg, state) for arg in node.args]
        for keyword in node.keywords:
            keyword.value = self._x(keyword.value, state)
        if state.blocked or self._pure_call(node):
            return node
        if (
            not unsafe_before
            and not self._skip_interception(node.func)
            and self._pure(node.func)
            and all(self._pure(arg) for arg in node.args)
            and all(self._pure(keyword.value) for keyword in node.keywords)
        ):
            state.unsafe = unsafe_before
            node.func = ast.Call(ast.Name("_lazy_defer", ast.Load()), [node.func], [])
            return self._lift(node, node, "_lazy_settle", state, defer=True)
        state.blocked = True
        return node

    def _lift(self, call, value, resolver: str, state: _ExtractState, *, defer: bool):
        tmp = f"{_TMP_PREFIX}{self._tmp_counter}"
        self._tmp_counter += 1
        stmt = ast.Assign(targets=[ast.Name(tmp, ast.Store())], value=value)
        stmt._lazy_reads = _load_names(value)
        stmt._lazy_defer = defer
        state.statements.append(ast.copy_location(stmt, call))
        result = ast.Call(
            ast.Name(resolver, ast.Load()), [ast.Name(tmp, ast.Load())], []
        )
        return ast.copy_location(result, call)

    # --- purity ------------------------------------------------------------- #

    def _is_local(self, name: str) -> bool:
        return not name.startswith(_TMP_PREFIX) and any(
            name in scope for scope in self._scopes
        )

    def _inert(self, name: str) -> bool:
        """Lazy temporaries, helpers and global modules, functions and classes."""
        if name.startswith(_TMP_PREFIX) or name in _HELPERS:
            return True
        value = self._resolve(ast.Name(name, ast.Load()))
        return value is not _MISSING and (inspect.ismodule(value) or callable(value))

    def _resolve(self, node):
        """Statically resolves a global (non-local) name or module attribute."""
        if isinstance(node, ast.Name):
            if node.id in _HELPERS or self._is_local(node.id):
                return _MISSING
            value = self._globals.get(node.id, _MISSING)
            if value is _MISSING:
                value = getattr(builtins, node.id, _MISSING)
            return value
        if isinstance(node, ast.Attribute):
            value = self._resolve(node.value)
            if value is _MISSING or not inspect.ismodule(value):
                return _MISSING
            return getattr(value, node.attr, _MISSING)
        return _MISSING

    def _pure_callee(self, func) -> bool:
        if isinstance(func, ast.Name) and func.id in _PURE_HELPERS:
            return True
        value = self._resolve(func)
        if value is not _MISSING:
            if id(value) in _PURE_BUILTINS or isinstance(value, np.ufunc):
                return True
            module = getattr(value, "__module__", None) or ""
            return (
                module.split(".")[0] in _PURE_LIBRARIES
                and not module.startswith("numpy.random")
                and getattr(value, "__name__", "") not in _IMPURE_LIBRARY_NAMES
            )
        return (
            isinstance(func, ast.Attribute)
            and func.attr in _PURE_METHODS
            and self._pure(func.value)
        )

    def _pure_call(self, node) -> bool:
        return self._pure_callee(node.func) and not any(
            keyword.arg in _MUTATING_KEYWORDS
            or (keyword.arg == "key" and not isinstance(keyword.value, ast.Lambda))
            for keyword in node.keywords
        )

    def _pure(self, node) -> bool:
        """True when evaluating `node` cannot mutate state (calls are whitelisted)."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if not self._pure_call(child):
                    return False
            elif isinstance(
                child, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)
            ):
                return False
        return True

    # --- interception ------------------------------------------------------- #

    def _intercept(self, stmt) -> None:
        rewriter = self

        class _Interceptor(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Nested functions intercepted their own bodies.
                node.decorator_list = [self.visit(d) for d in node.decorator_list]
                return node

            def visit_ClassDef(self, node):
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                if rewriter._skip_interception(node.func) or rewriter._pure_call(node):
                    return node
                node.func = ast.copy_location(
                    ast.Call(ast.Name("_lazy_call", ast.Load()), [node.func], []),
                    node.func,
                )
                return node

        _Interceptor().visit(stmt)

    def _skip_interception(self, func) -> bool:
        if isinstance(func, ast.Call):
            return _is_helper_call(func, {"_lazy_defer", "_lazy_call"})
        if isinstance(func, ast.Name) and (func.id in _HELPERS or func.id == "super"):
            return True
        value = self._resolve(func)
        return value is not _MISSING and not isinstance(value, FunctionType)


def _binds_names_only(stmt) -> bool:
    """True for Assign/AnnAssign/For targets without attribute or item stores."""
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    return all(
        isinstance(node, (ast.Name, ast.Tuple, ast.List, ast.Starred))
        for target in targets
        for node in ast.walk(target)
        if not isinstance(node, ast.expr_context)
    )


def _has_deferral(stmt) -> bool:
    return any(getattr(node, "_lazy_defer", False) for node in ast.walk(stmt))


def _is_helper_call(node, names) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    )


def lazy_agg(func=None, *, agg_client_name: str = "agg_client"):
    """
    Runs `func` with lazy aggregation: the `agg_client_name` argument is wrapped in
    a `LazyAggregationClient` and `func` (plus every project function it calls)
    runs rewritten so independent aggregation calls share round-trips.
    """

    def decorator(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            client = bound.arguments.get(agg_client_name)
            if client is not None and not isinstance(client, LazyAggregationClient):
                bound.arguments[agg_client_name] = LazyAggregationClient(client)
            return _lazy_call(fn)(*bound.args, **bound.kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class RecordingAggClient:
    """Test double that records calls and returns its inputs."""

    def __init__(self):
        self.calls = []

    def _aggregate_batch(self, ops):
        self.calls.append(("batch", len(ops)))
        return [value for _, value in ops]

    def sum(self, value):
        self.calls.append(("sum", np.asarray(value).size))
        return value

    def min(self, value):
        self.calls.append(("min", np.asarray(value).size))
        return value

    def max(self, value):
        self.calls.append(("max", np.asarray(value).size))
        return value

    def union(self, value):
        self.calls.append(("union", np.asarray(value).size))
        return value
