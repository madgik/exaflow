import ast
import datetime
import inspect
import os
import textwrap
from collections import defaultdict
from enum import Enum
from functools import wraps
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple


class LazyAggregationExecutor:
    """
    Small helper that batches global aggregations when possible.

    - Tries to use agg_client.aggregate_batch with AggregationType when available.
    - Falls back to issuing individual sum/min/max calls to preserve behaviour.
    """

    def __init__(self, agg_client):
        self.agg_client = agg_client

    def execute(self, batch: List[Tuple[str, object]]):
        if not batch:
            return []

        # Prefer a batch call if the client supports it
        aggregate_batch = getattr(self.agg_client, "aggregate_batch", None)
        if aggregate_batch:
            try:

                class AggregationType(Enum):
                    SUM = "SUM"
                    MIN = "MIN"
                    MAX = "MAX"

                    def __str__(self):
                        return self.name

                mapped = [
                    (getattr(AggregationType, op.upper()), value) for op, value in batch
                ]
                return list(aggregate_batch(mapped))
            except Exception:
                # Fallback to eager execution
                pass

        results = []
        for op, value in batch:
            if op == "sum":
                results.append(self.agg_client.sum(value))
            elif op == "min":
                results.append(self.agg_client.min(value))
            elif op == "max":
                results.append(self.agg_client.max(value))
            else:
                raise ValueError(f"Unsupported aggregation op '{op}'.")
        return results


class _NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)


def _used_names(node) -> set:
    collector = _NameCollector()
    collector.visit(node)
    return collector.names


def _rewrite_log_path() -> str:
    """
    Resolve the path where rewritten UDFs are logged.

    Override via LAZY_AGG_REWRITE_LOG if you want a custom location.
    """
    default_path = os.path.join(os.path.dirname(__file__), "lazy_agg_rewrites.log")
    return os.environ.get("LAZY_AGG_REWRITE_LOG", default_path)


class LazyAggregationRewriter:
    """
    Prototype AST rewriter that:
      - Finds assignments to agg_client.sum/min/max (treated as global ops)
      - Reorders consecutive independent globals into a batch
      - Inserts a LazyAggregationExecutor that performs the batch
    """

    def _log_rewritten(self, func_def: ast.FunctionDef, filename: str):
        """
        Persist the rewritten function source to a log file for inspection.
        """
        log_path = _rewrite_log_path()
        try:
            rewritten_source = (
                ast.unparse(func_def)
                if hasattr(ast, "unparse")
                else ast.dump(func_def, include_attributes=False)
            )
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            header = f"# --- lazy_agg rewrite: {func_def.name} ({filename}) at {timestamp} ---"
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{header}\n{rewritten_source}\n\n")
        except Exception:
            # Logging must never interfere with the rewrite process.
            pass

    def rewrite(self, func, agg_client_name: str = "agg_client"):
        try:
            source_lines, start_line = inspect.getsourcelines(func)
            source = textwrap.dedent("".join(source_lines))
        except OSError:
            # If we can't retrieve source (e.g., already rewritten), bail out.
            return func

        self._tmp_counter = 0
        module_ast = ast.parse(source)
        func_def = next(
            (node for node in module_ast.body if isinstance(node, ast.FunctionDef)),
            None,
        )
        if func_def is None:
            raise ValueError("Expected a single function definition to rewrite.")

        # Strip decorators to avoid re-entering lazy_agg inside the rewritten body.
        func_def.decorator_list = []

        new_body, has_lazy = self._rewrite_body(func_def.body, agg_client_name)
        if not has_lazy:
            return func  # Nothing to rewrite
        func_def.body = new_body

        ast.fix_missing_locations(module_ast)
        ast.increment_lineno(module_ast, start_line - 1)

        filename = inspect.getsourcefile(func) or getattr(
            getattr(func, "__code__", None), "co_filename", ""
        )
        if not filename:
            filename = f"<lazy-{func.__name__}>"
        # Uncomment for debugging
        # self._log_rewritten(func_def, filename)

        code_obj = compile(module_ast, filename=filename, mode="exec")
        env = func.__globals__
        env["LazyAggregationExecutor"] = LazyAggregationExecutor
        original_binding = env.get(func.__name__)
        exec(code_obj, env)
        rewritten = env[func.__name__]
        if original_binding is not None:
            env[func.__name__] = original_binding
        return rewritten

    def _get_assign_target(self, stmt):
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                return stmt.targets[0].id
            return None
        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name):
                return stmt.target.id
            return None
        return None

    def _is_global_call_assign(self, stmt, agg_client_name: str):
        target_name = self._get_assign_target(stmt)
        if target_name is None:
            return False
        call = stmt.value
        if not isinstance(call, ast.Call):
            return False
        if not isinstance(call.func, ast.Attribute):
            return False
        if not isinstance(call.func.value, ast.Name):
            return False
        if call.func.value.id != agg_client_name:
            return False
        if call.func.attr not in {"sum", "min", "max"}:
            return False
        return True

    def _next_tmp(self):
        name = f"_lazy_tmp{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def _hoist_globals_in_expr(self, expr, agg_client_name: str):
        """
        Returns (prefix_statements, new_expr) where prefix_statements are Assign nodes
        hoisting any agg_client.{sum|min|max} calls found in the expression.
        """

        def combine(parts):
            prefixes = []
            new_parts = []
            for pre, val in parts:
                prefixes.extend(pre)
                new_parts.append(val)
            return prefixes, new_parts

        if isinstance(expr, ast.Call):
            is_global_call = (
                isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == agg_client_name
                and expr.func.attr in {"sum", "min", "max"}
            )
            contains_comp = any(
                isinstance(
                    expr_arg,
                    (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp),
                )
                for expr_arg in (*expr.args, *(kw.value for kw in expr.keywords))
            )
            if is_global_call and not contains_comp:
                tmp_name = self._next_tmp()
                tmp_assign = ast.Assign(
                    targets=[ast.Name(tmp_name, ctx=ast.Store())],
                    value=expr,
                )
                tmp_assign = ast.copy_location(tmp_assign, expr)
                return [tmp_assign], ast.copy_location(
                    ast.Name(tmp_name, ctx=ast.Load()), expr
                )

            func_prefixes, new_func = self._hoist_globals_in_expr(
                expr.func, agg_client_name
            )
            arg_results = [
                self._hoist_globals_in_expr(arg, agg_client_name) for arg in expr.args
            ]
            kw_results = [
                self._hoist_globals_in_expr(kw.value, agg_client_name)
                for kw in expr.keywords
            ]
            arg_prefixes, new_args = combine(arg_results)
            kw_prefixes, new_kw_vals = combine(kw_results)
            new_keywords = []
            for kw, new_val in zip(expr.keywords, new_kw_vals):
                new_kw = ast.keyword(arg=kw.arg, value=new_val)
                new_kw = ast.copy_location(new_kw, kw)
                new_keywords.append(new_kw)
            expr.func = new_func
            expr.args = new_args
            expr.keywords = new_keywords
            return [*func_prefixes, *arg_prefixes, *kw_prefixes], expr

        if isinstance(expr, ast.BoolOp):
            parts = [
                self._hoist_globals_in_expr(v, agg_client_name) for v in expr.values
            ]
            prefixes, new_vals = combine(parts)
            expr.values = new_vals
            return prefixes, expr

        if isinstance(expr, ast.BinOp):
            left_p, new_left = self._hoist_globals_in_expr(expr.left, agg_client_name)
            right_p, new_right = self._hoist_globals_in_expr(
                expr.right, agg_client_name
            )
            expr.left = new_left
            expr.right = new_right
            return [*left_p, *right_p], expr

        if isinstance(expr, ast.UnaryOp):
            pre, operand = self._hoist_globals_in_expr(expr.operand, agg_client_name)
            expr.operand = operand
            return pre, expr

        if isinstance(expr, ast.Compare):
            left_p, new_left = self._hoist_globals_in_expr(expr.left, agg_client_name)
            right_parts = [
                self._hoist_globals_in_expr(c, agg_client_name)
                for c in expr.comparators
            ]
            right_p, new_right = combine(right_parts)
            expr.left = new_left
            expr.comparators = new_right
            return [*left_p, *right_p], expr

        if isinstance(expr, ast.IfExp):
            test_p, new_test = self._hoist_globals_in_expr(expr.test, agg_client_name)
            body_p, new_body = self._hoist_globals_in_expr(expr.body, agg_client_name)
            orelse_p, new_orelse = self._hoist_globals_in_expr(
                expr.orelse, agg_client_name
            )
            expr.test = new_test
            expr.body = new_body
            expr.orelse = new_orelse
            return [*test_p, *body_p, *orelse_p], expr

        if isinstance(expr, ast.Subscript):
            val_p, new_val = self._hoist_globals_in_expr(expr.value, agg_client_name)
            slice_p, new_slice = self._hoist_globals_in_expr(
                expr.slice, agg_client_name
            )
            expr.value = new_val
            expr.slice = new_slice
            return [*val_p, *slice_p], expr

        if isinstance(expr, ast.Attribute):
            pre, new_val = self._hoist_globals_in_expr(expr.value, agg_client_name)
            expr.value = new_val
            return pre, expr

        if isinstance(expr, ast.Tuple):
            parts = [
                self._hoist_globals_in_expr(elt, agg_client_name) for elt in expr.elts
            ]
            prefixes, new_elts = combine(parts)
            expr.elts = new_elts
            return prefixes, expr

        if isinstance(expr, ast.List):
            parts = [
                self._hoist_globals_in_expr(elt, agg_client_name) for elt in expr.elts
            ]
            prefixes, new_elts = combine(parts)
            expr.elts = new_elts
            return prefixes, expr

        if isinstance(expr, ast.Dict):
            key_parts = [
                (
                    self._hoist_globals_in_expr(k, agg_client_name)
                    if k is not None
                    else ([], None)
                )
                for k in expr.keys
            ]
            val_parts = [
                self._hoist_globals_in_expr(v, agg_client_name) for v in expr.values
            ]
            key_prefixes, new_keys = combine(key_parts)
            val_prefixes, new_vals = combine(val_parts)
            expr.keys = new_keys
            expr.values = new_vals
            return [*key_prefixes, *val_prefixes], expr

        return [], expr

    def _hoist_globals_in_stmt(self, stmt, agg_client_name: str):
        """
        Returns a list of statements including hoisted global-call assignments
        needed by this statement, with the original statement last.
        """
        new_stmts = []
        if isinstance(
            stmt, (ast.Assign, ast.AnnAssign)
        ) and self._is_global_call_assign(stmt, agg_client_name):
            return [stmt]
        if isinstance(stmt, ast.Assign):
            pre, new_value = self._hoist_globals_in_expr(stmt.value, agg_client_name)
            stmt.value = new_value
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                pre, new_value = self._hoist_globals_in_expr(
                    stmt.value, agg_client_name
                )
                stmt.value = new_value
                new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.Expr):
            pre, new_value = self._hoist_globals_in_expr(stmt.value, agg_client_name)
            stmt.value = new_value
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                pre, new_value = self._hoist_globals_in_expr(
                    stmt.value, agg_client_name
                )
                stmt.value = new_value
                new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.If):
            pre, new_test = self._hoist_globals_in_expr(stmt.test, agg_client_name)
            stmt.test = new_test
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.While):
            pre, new_test = self._hoist_globals_in_expr(stmt.test, agg_client_name)
            stmt.test = new_test
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            pre, new_iter = self._hoist_globals_in_expr(stmt.iter, agg_client_name)
            stmt.iter = new_iter
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                pre, new_ctx = self._hoist_globals_in_expr(
                    item.context_expr, agg_client_name
                )
                new_stmts.extend(pre)
                item.context_expr = new_ctx
            new_stmts.append(stmt)
            return new_stmts
        if isinstance(stmt, ast.Assert):
            pre, new_test = self._hoist_globals_in_expr(stmt.test, agg_client_name)
            stmt.test = new_test
            new_stmts.extend(pre)
            new_stmts.append(stmt)
            return new_stmts
        return [stmt]

    def _flush_batch(self, batch, output_order, location):
        if not batch:
            return []

        def _locate(node):
            return ast.copy_location(node, location) if location is not None else node

        batch_assign = ast.Assign(
            targets=[ast.Name("_lazy_batch", ctx=ast.Store())],
            value=ast.List(
                elts=[
                    ast.Tuple(
                        elts=[
                            ast.Constant(op),
                            expr,
                        ],
                        ctx=ast.Load(),
                    )
                    for op, expr in batch
                ],
                ctx=ast.Load(),
            ),
        )
        batch_assign = _locate(batch_assign)
        execute_call = ast.Assign(
            targets=[ast.Name("_lazy_results", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name("_lazy_exec", ctx=ast.Load()),
                    attr="execute",
                    ctx=ast.Load(),
                ),
                args=[ast.Name("_lazy_batch", ctx=ast.Load())],
                keywords=[],
            ),
        )
        execute_call = _locate(execute_call)

        assigns = []
        for idx, target_name in enumerate(output_order):
            assign_stmt = ast.Assign(
                targets=[ast.Name(target_name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name("_lazy_results", ctx=ast.Load()),
                    slice=ast.Index(ast.Constant(idx)),
                    ctx=ast.Load(),
                ),
            )
            assigns.append(_locate(assign_stmt))

        return [batch_assign, execute_call, *assigns]

    def _rewrite_body(self, body, agg_client_name: str, insert_exec: bool = True):
        new_body = []
        has_lazy = False

        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
        ):
            new_body.append(body[0])
            body = body[1:]

        expanded_body = []
        for stmt in body:
            expanded_body.extend(self._hoist_globals_in_stmt(stmt, agg_client_name))
        body = expanded_body

        new_body = []
        pending_batch: List[Tuple[str, ast.AST]] = []
        pending_targets: List[str] = []
        pending_location = None
        has_lazy = False

        exec_assign = (
            ast.Assign(
                targets=[ast.Name("_lazy_exec", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name("LazyAggregationExecutor", ctx=ast.Load()),
                    args=[ast.Name(agg_client_name, ctx=ast.Load())],
                    keywords=[],
                ),
            )
            if insert_exec
            else None
        )
        inserted_exec = not insert_exec

        def flush():
            nonlocal pending_batch, pending_targets, inserted_exec, has_lazy, pending_location
            if pending_batch:
                new_body.extend(
                    self._flush_batch(pending_batch, pending_targets, pending_location)
                )
                if insert_exec and not inserted_exec:
                    new_body.insert(1 if new_body else 0, exec_assign)
                    inserted_exec = True
                has_lazy = True
            pending_batch = []
            pending_targets = []
            pending_location = None

        i = 0
        while i < len(body):
            stmt = body[i]

            if self._is_global_call_assign(stmt, agg_client_name):
                target = self._get_assign_target(stmt)
                # If this global depends on earlier pending outputs, flush first
                call_used = _used_names(stmt.value)
                if call_used.intersection(set(pending_targets)):
                    flush()
                pending_batch.append((stmt.value.func.attr, stmt.value.args[0]))
                pending_targets.append(target)
                pending_location = pending_location or stmt
                i += 1
                continue

            used = _used_names(stmt)
            defined = (
                {t.id for t in stmt.targets if isinstance(t, ast.Name)}
                if isinstance(stmt, ast.Assign)
                else (
                    {stmt.target.id}
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    else set()
                )
            )
            allow_hoist = isinstance(stmt, (ast.Assign, ast.AnnAssign))

            if pending_targets and not allow_hoist:
                flush()

            # Non-global statement: optionally hoist following independent globals
            j = i + 1
            if allow_hoist:
                while j < len(body) and self._is_global_call_assign(
                    body[j], agg_client_name
                ):
                    call_used = _used_names(body[j].value)
                    call_used_no_client = {
                        name for name in call_used if name != agg_client_name
                    }
                    used_no_client = {name for name in used if name != agg_client_name}
                    if call_used_no_client.intersection(
                        used_no_client
                    ) or call_used_no_client.intersection(defined):
                        break
                    target = self._get_assign_target(body[j])
                    pending_batch.append(
                        (body[j].value.func.attr, body[j].value.args[0])
                    )
                    pending_targets.append(target)
                    pending_location = pending_location or body[j]
                    j += 1

                if pending_targets and used.intersection(set(pending_targets)):
                    flush()

            rewritten_stmt, child_has_lazy = self._rewrite_children(
                stmt, agg_client_name
            )
            has_lazy = has_lazy or child_has_lazy
            new_body.append(rewritten_stmt)
            i = j

        flush()
        if insert_exec and has_lazy and not inserted_exec:
            new_body.insert(1 if new_body else 0, exec_assign)
            inserted_exec = True
        return new_body, has_lazy

    def _rewrite_children(self, stmt, agg_client_name: str):
        # Recursively rewrite statements that contain bodies.
        child_has_lazy = False
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith)):
            stmt.body, body_lazy = self._rewrite_body(
                stmt.body, agg_client_name, insert_exec=False
            )
            stmt.orelse, orelse_lazy = self._rewrite_body(
                stmt.orelse, agg_client_name, insert_exec=False
            )
            child_has_lazy = body_lazy or orelse_lazy
        elif isinstance(stmt, ast.Try):
            stmt.body, body_lazy = self._rewrite_body(
                stmt.body, agg_client_name, insert_exec=False
            )
            stmt.orelse, orelse_lazy = self._rewrite_body(
                stmt.orelse, agg_client_name, insert_exec=False
            )
            stmt.finalbody, final_lazy = self._rewrite_body(
                stmt.finalbody, agg_client_name, insert_exec=False
            )
            new_handlers = []
            handlers_lazy = False
            for handler in stmt.handlers:
                rewritten_handler, handler_lazy = self._rewrite_exc_handler(
                    handler, agg_client_name
                )
                new_handlers.append(rewritten_handler)
                handlers_lazy = handlers_lazy or handler_lazy
            stmt.handlers = new_handlers
            child_has_lazy = body_lazy or orelse_lazy or final_lazy or handlers_lazy
        elif isinstance(stmt, ast.FunctionDef):
            inner_agg_name = (
                stmt.args.args[0].arg if stmt.args.args else agg_client_name
            )
            # Avoid double-decoration; the parent rewrite will handle batching.
            stmt.decorator_list = []
            stmt.body, child_has_lazy = self._rewrite_body(
                stmt.body, inner_agg_name, insert_exec=True
            )
        return stmt, child_has_lazy

    def _rewrite_exc_handler(self, handler, agg_client_name: str):
        handler.body, handler_lazy = self._rewrite_body(
            handler.body, agg_client_name, insert_exec=False
        )
        return handler, handler_lazy


def lazy_agg(func=None, *, agg_client_name: str = "agg_client"):
    """
    Decorator to apply LazyAggregationRewriter to a function.
    """

    def decorator(fn):
        rewritten_func = None

        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal rewritten_func
            if rewritten_func is None:
                rewriter = LazyAggregationRewriter()
                try:
                    rewritten_func = rewriter.rewrite(fn, agg_client_name)
                except Exception:
                    rewritten_func = fn
            return rewritten_func(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


import numpy as np


class RecordingAggClient:
    def __init__(self):
        self.calls = []

    def aggregate_batch(self, ops):
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


class DependencyGraphBuilder(ast.NodeVisitor):
    """
    Lightweight dependency graph builder for Python function bodies.

    Useful for visualizing how values flow through a function before/after
    applying lazy aggregation rewrites.
    """

    def __init__(self):
        self.nodes: List[Dict[str, object]] = []
        self.edges: List[Dict[str, object]] = []
        self.var_definitions: Dict[str, int] = {}
        self.var_uses = defaultdict(list)
        self.node_counter = 0
        self.current_scope = []

    def add_node(self, node_type: str, details: str = "") -> int:
        """Add a node to the graph and return its index."""
        node_id = self.node_counter
        self.nodes.append({"id": node_id, "type": node_type, "details": details})
        self.node_counter += 1
        return node_id

    def add_edge(self, source_id: int, target_id: int, var_name: str):
        """Add an edge representing data dependency."""
        if source_id != target_id:
            self.edges.append(
                {"source": source_id, "target": target_id, "variable": var_name}
            )

    def get_names_from_expr(self, node) -> Set[str]:
        """Extract all variable names used in an expression."""
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def visit_Assign(self, node):
        """Handle assignment: target = value."""
        targets_str = ", ".join(self.get_target_names(node.targets))

        value_node_id = self.visit_expr(node.value)
        node_id = self.add_node("Assign", f"{targets_str} = ...")

        if value_node_id is not None:
            self.add_edge(value_node_id, node_id, "result")

        for target in node.targets:
            for name in self.get_target_names([target]):
                self.var_definitions[name] = node_id

    def visit_expr(self, node) -> int:
        """Visit an expression and return the node id that produces it."""
        if isinstance(node, ast.BinOp):
            left_id = self.visit_expr(node.left)
            right_id = self.visit_expr(node.right)
            op_name = node.op.__class__.__name__
            node_id = self.add_node("BinOp", op_name)
            if left_id is not None:
                self.add_edge(left_id, node_id, "left")
            if right_id is not None:
                self.add_edge(right_id, node_id, "right")
            return node_id

        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node.func)
            node_id = self.add_node("Call", func_name)
            for i, arg in enumerate(node.args):
                arg_id = self.visit_expr(arg)
                if arg_id is not None:
                    self.add_edge(arg_id, node_id, f"arg{i}")
            return node_id

        if isinstance(node, ast.Name):
            if node.id in self.var_definitions:
                return self.var_definitions[node.id]
            return None

        if isinstance(node, ast.Constant):
            return self.add_node("Constant", repr(node.value))

        return None

    def _get_func_name(self, node) -> str:
        """Extract function name from a Call node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            obj = self._get_func_name(node.value)
            return f"{obj}.{node.attr}"
        return "?"

    def visit_AugAssign(self, node):
        """Handle augmented assignment: x += 1."""
        target_name = node.target.id if isinstance(node.target, ast.Name) else "?"
        op_str = node.op.__class__.__name__

        target_id = (
            self.var_definitions.get(target_name)
            if isinstance(node.target, ast.Name)
            else None
        )
        value_id = self.visit_expr(node.value)

        binop_id = self.add_node("BinOp", op_str)
        if target_id is not None:
            self.add_edge(target_id, binop_id, target_name)
        if value_id is not None:
            self.add_edge(value_id, binop_id, "rhs")

        node_id = self.add_node("Assign", f"{target_name} {op_str}=")
        self.add_edge(binop_id, node_id, "result")

        if isinstance(node.target, ast.Name):
            self.var_definitions[node.target.id] = node_id

    def visit_For(self, node):
        """Handle for loops."""
        loop_var = node.target.id if isinstance(node.target, ast.Name) else "?"
        iter_id = self.visit_expr(node.iter)
        node_id = self.add_node("For", f"for {loop_var}")
        if iter_id is not None:
            self.add_edge(iter_id, node_id, "iterator")

        if isinstance(node.target, ast.Name):
            self.var_definitions[node.target.id] = node_id

        for child in node.body:
            self.visit(child)

    def visit_If(self, node):
        """Handle if statements."""
        test_id = self.visit_expr(node.test)
        node_id = self.add_node("If", "if")
        if test_id is not None:
            self.add_edge(test_id, node_id, "condition")

        for child in node.body + node.orelse:
            self.visit(child)

    def get_target_names(self, targets) -> List[str]:
        """Extract variable names from assignment targets."""
        names: List[str] = []
        for target in targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    names.extend(self.get_target_names([elt]))
        return names

    def _get_expr_repr(self, node) -> str:
        """Get a string representation of an expression."""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = ", ".join(self._get_expr_repr(arg) for arg in node.args)
                return f"{node.func.id}({args})"
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._get_expr_repr(node.left)
            right = self._get_expr_repr(node.right)
            op = node.op.__class__.__name__
            return f"{left} {op} {right}"
        return "..."
