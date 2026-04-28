"""
Injects needed params for tuning into source code
"""

import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider


@dataclass
class Finding:
    f_name: str
    lineno: int
    call_type: str
    kwarg: str
    param: str
    default: object
    search_values: list
    already_present: bool

_SCAN_NAMES = {"jax.lax.scan", "lax.scan"}
_DOT_NAMES = {"jnp.dot", "jax.numpy.dot", "np.dot"}
_MATMUL_NAMES = {"jnp.matmul", "jax.numpy.matmul"}
_JIT_NAMES = {"jax.jit", "jit"}
_MAP_NAMES = {"jax.lax.map", "lax.map"}
_CHECKPOINT_NAMES = {"jax.checkpoint", "jax.remat", "checkpoint", "remat"}


def _cst_dotted_name(node) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        return _cst_dotted_name(node.value) + "." + node.attr.value
    return ""


def _cst_has_kwarg(call, name) -> bool:
    for arg in call.args:
        if isinstance(arg.keyword, cst.Name) and arg.keyword.value == name:
            return True
    return False


def _make_kwarg(kw, value_expr) -> cst.Arg:
    return cst.Arg(
        keyword=cst.Name(kw),
        value=cst.parse_expression(value_expr),
        equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")),
        comma=cst.MaybeSentinel.DEFAULT
    )


def _add_comma_to_last(args) -> tuple[cst.Arg, ...]:
    if not args:
        return tuple(args)
    last = args[-1]
    if isinstance(last.comma, cst.MaybeSentinel):
        last = last.with_changes(comma=cst.Comma(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(" ")))
        return (*args[:-1], last)
    return tuple(args)


def _param_name_for_occurrence(base, count) -> str:
    return base if count == 0 else f"{base}_{count}"


class _ScannerVisitor(cst.CSTVisitor):
    """Walk the CST and collect all tunable pattern occurrences as Finding objects."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, include_autotune = False):
        self._stack = []
        self._scan_counts = {}
        self.findings = []
        self._include_autotune = include_autotune

    def _top(self) -> str | None:
        return self._stack[0] if self._stack else None

    def visit_FunctionDef(self, node) -> bool:
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, node) -> None:
        fn_name = node.name.value
        if self._include_autotune and len(self._stack) == 1:
            pos = self.get_metadata(PositionProvider, node)
            self.findings.append(Finding(
                f_name=fn_name,
                lineno=pos.start.line,
                call_type="autotune",
                kwarg="XLA_FLAGS",
                param="autotune_level",
                default=0,
                search_values=[0, 1, 2, 3, 4],
                already_present=False))
        self._stack.pop()

    def visit_Decorator(self, node) -> bool:
        top = self._top()
        if top is None:
            return True
        pos = self.get_metadata(PositionProvider, node)
        dec = node.decorator
        name = _cst_dotted_name(dec)
        if name in _JIT_NAMES:
            already = isinstance(dec, cst.Call)
            self.findings.append(Finding(
                f_name=top,
                lineno=pos.start.line,
                call_type="jit",
                kwarg="donate_argnums",
                param="jit_donate_argnums",
                default=(),
                search_values=[(), (0,), (1,), (0, 1)],
                already_present=already))
        elif name in _CHECKPOINT_NAMES:
            already = isinstance(dec, cst.Call)
            self.findings.append(Finding(
                f_name=top,
                lineno=pos.start.line,
                call_type="checkpoint",
                kwarg="policy",
                param="checkpoint_policy",
                default=None,
                search_values=[None, "nothing_saveable", "everything_saveable", "dots_with_no_batch_dims_saveable"],
                already_present=already))
        return True

    def visit_Call(self, node) -> bool:
        top = self._top()
        if top is None:
            return True
        pos = self.get_metadata(PositionProvider, node)
        lineno = pos.start.line
        name = _cst_dotted_name(node.func)
        self._check_scan(node, name, top, lineno)
        self._check_dot_matmul(node, name, top, lineno)
        self._check_map(node, name, top, lineno)
        return True

    def _check_scan(self, node, name, top, lineno):
        if name not in _SCAN_NAMES:
            return
        count = self._scan_counts.get(top, 0)
        self._scan_counts[top] = count + 1
        unroll_param = _param_name_for_occurrence("scan_unroll", count)
        reverse_param = _param_name_for_occurrence("scan_reverse", count)
        self.findings.append(Finding(
            f_name=top, lineno=lineno, call_type="scan",
            kwarg="unroll", param=unroll_param,
            default=1, search_values=[1, 2, 4, 8, 16],
            already_present=_cst_has_kwarg(node, "unroll")))
        self.findings.append(Finding(
            f_name=top, lineno=lineno, call_type="scan",
            kwarg="reverse", param=reverse_param,
            default=False, search_values=[False, True],
            already_present=_cst_has_kwarg(node, "reverse")))

    def _check_dot_matmul(self, node, name, top, lineno):
        if name in _DOT_NAMES:
            call_type, param = "dot", "dot_precision"
        elif name in _MATMUL_NAMES:
            call_type, param = "matmul", "matmul_precision"
        else:
            return
        self.findings.append(Finding(
            f_name=top, lineno=lineno, call_type=call_type,
            kwarg="precision", param=param,
            default=None, search_values=[None, "high", "highest"],
            already_present=_cst_has_kwarg(node, "precision")))

    def _check_map(self, node, name, top, lineno):
        if name not in _MAP_NAMES:
            return
        self.findings.append(Finding(
            f_name=top, lineno=lineno, call_type="map",
            kwarg="batch_size", param="map_chunk_size",
            default=None, search_values=[1, 4, 8, 16, None],
            already_present=_cst_has_kwarg(node, "batch_size")))

class _InjectTransformer(cst.CSTTransformer):
    """
    Rewrite the CST to inject tunable params into signatures and calls, based on the list of Findings.
    """

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, to_inject: dict[str, list[Finding]]):
        self._to_inject = to_inject
        self._active_fn = None
        self._fn_depth = 0
        self._needs_functools = False

    def leave_Module(self, original_node, updated_node) -> cst.Module:
        if not self._needs_functools:
            return updated_node
        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for s in stmt.body:
                    if isinstance(s, cst.Import):
                        imp = s.names
                        if isinstance(imp, cst.ImportStar):
                            continue
                        for alias in imp:
                            if _cst_dotted_name(alias.name) == "functools":
                                return updated_node
            elif isinstance(stmt, cst.ImportFrom):
                pass
        import_stmt = cst.parse_statement("import functools\n")
        return updated_node.with_changes(body=(import_stmt, *updated_node.body))

    def visit_FunctionDef(self, node) -> bool:
        if self._active_fn is None and node.name.value in self._to_inject:
            self._active_fn = node.name.value
            self._fn_depth = 0
        elif self._active_fn is not None:
            self._fn_depth += 1
        return True

    def leave_FunctionDef(self, original_node, updated_node) -> cst.FunctionDef:
        if self._active_fn == original_node.name.value and self._fn_depth == 0:
            findings = self._to_inject[self._active_fn]
            updated_node = self._inject_params(updated_node, findings)
            updated_node = self._inject_autotune_stmt(updated_node, findings)
            self._active_fn = None
        elif self._active_fn is not None and self._fn_depth > 0:
            self._fn_depth -= 1
        return updated_node

    def leave_Decorator(self, original_node,updated_node) -> cst.Decorator:
        if self._active_fn is None:
            return updated_node
        findings = self._to_inject.get(self._active_fn, [])
        pos = self.get_metadata(PositionProvider, original_node)
        lineno = pos.start.line
        dec = original_node.decorator
        name = _cst_dotted_name(dec)
        for f in findings:
            if f.lineno != lineno:
                continue
            if f.call_type == "jit" and name in _JIT_NAMES and not isinstance(dec, cst.Call):
                self._needs_functools = True
                new_dec = cst.parse_expression(f"functools.partial({name}, donate_argnums={f.param})")
                return updated_node.with_changes(decorator=new_dec)
            if f.call_type == "checkpoint" and name in _CHECKPOINT_NAMES and not isinstance(dec, cst.Call):
                self._needs_functools = True
                new_dec = cst.parse_expression(f"functools.partial({name}, policy={f.param})")
                return updated_node.with_changes(decorator=new_dec)
        return updated_node

    def leave_Call(self, original_node, updated_node) -> cst.Call:
        if self._active_fn is None:
            return updated_node
        findings = self._to_inject.get(self._active_fn, [])
        pos = self.get_metadata(PositionProvider, original_node)
        lineno = pos.start.line
        name = _cst_dotted_name(original_node.func)

        new_kwargs: list[cst.Arg] = []
        for f in findings:
            if f.lineno != lineno:
                continue
            if f.call_type == "scan" and name not in _SCAN_NAMES: continue
            if f.call_type == "dot" and name not in _DOT_NAMES: continue
            if f.call_type == "matmul" and name not in _MATMUL_NAMES: continue
            if f.call_type == "map" and name not in _MAP_NAMES: continue
            new_kwargs.append(_make_kwarg(f.kwarg, f.param))

        if not new_kwargs:
            return updated_node

        existing = _add_comma_to_last(updated_node.args)
        return updated_node.with_changes(args=(*existing, *new_kwargs))

    @staticmethod
    def _inject_params(node, findings) -> cst.FunctionDef:
        params = node.params
        existing_names = {
            p.name.value
            for p in (
                *params.params,
                *([] if params.star_kwarg is None else [params.star_kwarg]),
                *params.kwonly_params,
            )
        }
        new_params = []
        seen = set()
        for f in findings:
            if f.param in existing_names or f.param in seen:
                continue
            seen.add(f.param)
            default_val = _default_to_cst(f.default)
            new_params.append(cst.Param(
                name=cst.Name(f.param),
                default=default_val,
                equal=cst.AssignEqual(whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace(""))))

        if not new_params:
            return node

        old_params = list(params.params)
        if old_params:
            last = old_params[-1]
            if isinstance(last.comma, cst.MaybeSentinel):
                last = last.with_changes(comma=cst.Comma(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(" "),
                ))
                old_params[-1] = last

        all_params = old_params + new_params
        last = all_params[-1]
        if not isinstance(last.comma, cst.MaybeSentinel):
            all_params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

        return node.with_changes(params=params.with_changes(params=all_params))

    @staticmethod
    def _inject_autotune_stmt(
        node: cst.FunctionDef,
        findings: list[Finding],
    ) -> cst.FunctionDef:
        autotune = [f for f in findings if f.call_type == "autotune"]
        if not autotune:
            return node
        import_stmt = cst.parse_statement("import os as _os\n")
        set_stmt = cst.parse_statement(
            '_os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_autotune_level={autotune_level}")\n'
        )
        body = node.body
        if isinstance(body, cst.IndentedBlock):
            return node.with_changes(
                body=body.with_changes(body=(import_stmt, set_stmt, *body.body))
            )
        return node



def _default_to_cst(value: object) -> cst.BaseExpression:
    if value is None:
        return cst.Name("None")
    if isinstance(value, bool):
        return cst.Name("True" if value else "False")
    if isinstance(value, int):
        return cst.Integer(str(value))
    if isinstance(value, tuple):
        elements = [
            cst.Element(
                value=cst.Integer(str(v)),
                comma=cst.Comma(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(" "),
                ) if i < len(value) - 1 else cst.MaybeSentinel.DEFAULT,
            )
            for i, v in enumerate(value)
        ]
        return cst.Tuple(elements=elements, lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
    return cst.parse_expression(repr(value))

def _run_transform(
    source: str,
    skip_present: bool = True,
    include_autotune: bool = False,
) -> tuple[str, dict[str, str], dict[str, list]]:
    """
    Parse source, scan for patterns, inject params. Returns
    (instrumented_source, param_map, search_space).
    """
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    visitor = _ScannerVisitor(include_autotune=include_autotune)
    wrapper.visit(visitor)

    to_inject: dict[str, list[Finding]] = {}
    param_map: dict[str, str] = {}
    search_space: dict[str, list] = {}

    for f in visitor.findings:
        if skip_present and f.already_present:
            continue
        to_inject.setdefault(f.f_name, []).append(f)
        param_map[f.param] = f.param
        search_space[f.param] = f.search_values

    if not to_inject:
        return source, {}, {}

    transformer = _InjectTransformer(to_inject)
    new_module = wrapper.visit(transformer)
    return new_module.code, param_map, search_space


def scan(path: str | Path) -> list[Finding]:
    """
    Scan a Python source file for tunable JAX patterns.
    Non-destructive — does not write any file.
    Returns a list of Finding objects (both already-present and missing).
    """
    source = Path(path).read_text()
    module = cst.parse_module(source)
    wrapper = MetadataWrapper(module)
    visitor = _ScannerVisitor(include_autotune=False)
    wrapper.visit(visitor)
    return visitor.findings


def instrument(
    path: str | Path,
    output_path: str | Path | None = None,
    skip_present: bool = True,
    include_autotune: bool = False,
) -> tuple[Path, dict, dict]:
    """
    Write an instrumented copy of the file with tunable params injected.

    Parameters
    ----------
    path             : source file to instrument
    output_path      : destination (default: <stem>_instrumented.py next to source)
    skip_present     : if True, skip params already in the call
    include_autotune : if True, also inject autotune_level param

    Returns
    -------
    (output_path, param_map, search_space)
    """
    path = Path(path)
    source = path.read_text()

    instrumented_src, param_map, search_space = _run_transform(source, skip_present=skip_present, include_autotune=include_autotune)

    out = Path(output_path or path.with_stem(path.stem + "_instrumented"))
    out.write_text(instrumented_src)
    return out, param_map, search_space


def from_fn(
    fn: Callable,
    skip_present: bool = True,
    include_autotune: bool = False,
) -> tuple[str, dict, dict]:
    """
    Accept a live function object, instrument it in-memory via LibCST.

    Parameters
    ----------
    fn               : a Python function (must be defined in a .py file)
    skip_present     : if True, skip params already present in calls
    include_autotune : if True, inject autotune_level env-flag param
    """
    raw = fn
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    src = textwrap.dedent(inspect.getsource(raw))
    return _run_transform(src, skip_present=skip_present, include_autotune=include_autotune)
