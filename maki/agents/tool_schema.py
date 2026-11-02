"""
Typed tool schemas and argument validation for plugin methods.

``parameters_schema(method)`` builds the JSON Schema a model sees from the method's type
hints and the ``Args:`` section of its docstring. ``coerce_args(method, args)`` checks a
model-supplied argument dict against the same signature, converting what is unambiguous
(``"3"`` -> ``3``) and returning a message the model can act on for what is not. Parameters
without an annotation are advertised as strings and their values are passed through unchanged, so untyped plugins behave as before.
"""

import enum
import inspect
import json
import math
import re
import types
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}
_UNION_TYPES = (Union, getattr(types, "UnionType", Union))
_ARGS_HEADER = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
_ARG_LINE = re.compile(r"^\s+(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _type_hints(method: Callable) -> Dict[str, Any]:
    """Resolved type hints. When one annotation cannot be resolved (a name only imported under
    ``TYPE_CHECKING``) the others still are; the unresolvable parameter counts as unannotated."""
    try:
        return typing.get_type_hints(method)
    except Exception:
        pass
    namespace = getattr(inspect.unwrap(method), "__globals__", {})
    hints: Dict[str, Any] = {}
    for name, annotation in (getattr(method, "__annotations__", {}) or {}).items():
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, namespace)  # noqa: S307 - the plugin's own annotation
            except Exception:
                continue
        hints[name] = annotation
    return hints


def _doc_args(method: Callable) -> Dict[str, str]:
    """Per-parameter descriptions from a Google-style ``Args:`` section."""
    lines = (inspect.getdoc(method) or "").splitlines()
    out: Dict[str, str] = {}
    in_args, current, indent = False, None, None
    for line in lines:
        if not in_args:
            in_args = bool(_ARGS_HEADER.match(line))
            continue
        if not line.strip():
            current = None
            continue
        match = _ARG_LINE.match(line)
        line_indent = len(line) - len(line.lstrip())
        if match and (indent is None or line_indent <= indent):
            indent = line_indent
            current = match.group(1).lstrip("*")
            out[current] = match.group(2).strip()
        elif current and indent is not None and line_indent > indent:
            out[current] = f"{out[current]} {line.strip()}".strip()
        else:
            break  # next docstring section (Returns:, Raises:, ...)
    return out


def _enum_schema(tp: Any) -> dict:
    values = [m.value for m in tp]
    kind = _SCALARS.get(type(values[0])) if values else None
    return {"type": kind, "enum": values} if kind else {"enum": values}


def _array_schema(args: tuple) -> dict:
    schema: dict = {"type": "array"}
    if args and args[-1] is not Ellipsis:
        item = type_to_schema(args[0])
        if item:
            schema["items"] = item
    return schema


def _literal_schema(args: tuple) -> dict:
    kinds = {_SCALARS.get(type(a)) for a in args}
    schema: dict = {"enum": list(args)}
    if len(kinds) == 1 and None not in kinds:
        schema["type"] = kinds.pop()
    return schema


def _union_schema(args: tuple) -> dict:
    schemas = [type_to_schema(a) for a in args if a is not type(None)]
    if any(not s for s in schemas):
        return {}
    return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}


def type_to_schema(tp: Any) -> dict:
    """JSON Schema for a type hint; ``{}`` (anything goes) for types it cannot express."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    if tp in _SCALARS:
        return {"type": _SCALARS[tp]}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return _enum_schema(tp)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if tp in (list, tuple, set, frozenset) or origin in (list, tuple, set, frozenset):
        return _array_schema(args)
    if tp is dict or origin is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return _literal_schema(args)
    if origin in _UNION_TYPES:
        return _union_schema(args)
    return {}


def method_description(method: Callable, fallback: str) -> str:
    """The docstring's summary paragraph (up to the first blank line) on one line."""
    summary = (inspect.getdoc(method) or "").split("\n\n", 1)[0]
    return " ".join(summary.split()) or fallback


def parameters_schema(method: Callable) -> dict:
    """JSON Schema ``object`` for *method*'s parameters (``self`` excluded)."""
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return {"type": "object", "properties": {}, "required": []}
    hints, docs = _type_hints(method), _doc_args(method)
    properties: Dict[str, dict] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        schema = type_to_schema(hints[name]) if name in hints else {"type": "string"}
        schema = dict(schema)
        if name in docs and docs[name]:
            schema["description"] = docs[name]
        if param.default is not inspect.Parameter.empty and param.default is not None:
            try:
                json.dumps(param.default)
                schema["default"] = param.default
            except (TypeError, ValueError):
                pass
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise ValueError("expected a string")


def _to_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("expected a boolean")


def _to_integer(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError("expected an integer")


def _to_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("expected a number")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError("expected a number") from None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value  # "nan" / "inf" parse as floats but are never a quantity a tool should get
    raise ValueError("expected a finite number")


_SCALAR_CONVERTERS = {str: _to_string, bool: _to_boolean, int: _to_integer, float: _to_number}


def _by_value(options: Any, value: Any) -> Any:
    """The option equal to *value*, matching ``"3"`` to ``3``; ``ValueError`` if none."""
    for option in options:
        raw = option.value if isinstance(option, enum.Enum) else option
        if value == raw or (isinstance(value, str) and str(raw) == value):
            return option
    raise ValueError(f"must be one of {[o.value if isinstance(o, enum.Enum) else o for o in options]}")


def _convert_union(value: Any, args: tuple) -> Any:
    if value is None and type(None) in args:
        return None
    branches = [a for a in args if a is not type(None)]
    if len(branches) == 1:  # Optional[T]: report T's own complaint
        return _convert(value, branches[0])
    for branch in branches:
        try:
            return _convert(value, branch)
        except ValueError:
            continue
    raise ValueError("matches none of the allowed types")


def _convert_list(value: Any, args: tuple) -> list:
    if isinstance(value, (tuple, set, frozenset)):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("expected a list")
    if not args or args[0] is Ellipsis:
        return value
    out = []
    for i, element in enumerate(value):
        try:
            out.append(_convert(element, args[0]))
        except ValueError as exc:
            raise ValueError(f"item {i}: {exc}") from None
    return out


def _convert(value: Any, hint: Any) -> Any:
    """Convert *value* to what the type hint *hint* asks for, or raise ``ValueError``.

    Driven by the hint itself (not the JSON Schema) so the method receives the real type:
    enum members, not their values, and enums nested in lists.
    """
    if hint is inspect.Parameter.empty or hint is Any:
        return value
    if hint in _SCALAR_CONVERTERS:
        return _SCALAR_CONVERTERS[hint](value)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return _by_value(list(hint), value)
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in _UNION_TYPES:
        return _convert_union(value, args)
    if origin is typing.Literal:
        return _by_value(args, value)
    if hint in (list, tuple, set, frozenset) or origin in (list, tuple, set, frozenset):
        return _convert_list(value, args)
    if hint is dict or origin is dict:
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    return value


def _accepts_none(hint: Any) -> bool:
    return (hint is inspect.Parameter.empty or hint is Any or hint is type(None)
            or (typing.get_origin(hint) in _UNION_TYPES and type(None) in typing.get_args(hint)))


def _convert_arg(value: Any, hint: Any, param: inspect.Parameter) -> Tuple[bool, Any]:
    """``(keep, value)`` for one argument; ``keep`` is False for a null that means "use the default"."""
    if value is None and not _accepts_none(hint):
        if param.default is inspect.Parameter.empty:
            raise ValueError("must not be null")
        return False, None
    return True, _convert(value, hint)


def _check_arg(name: str, value: Any, named: dict, hints: dict, takes_kwargs: bool, clean: dict) -> Optional[str]:
    """Validate one argument into *clean*; return the problem for the model, or None."""
    param = named.get(name)
    if param is None:
        if takes_kwargs:
            clean[name] = value
            return None
        return f"unknown argument '{name}' (expected: {', '.join(named) or 'none'})"
    try:
        keep, converted = _convert_arg(value, hints.get(name, inspect.Parameter.empty), param)
    except ValueError as exc:
        return f"argument '{name}' {exc}" if value is None else f"argument '{name}' {exc}, got {value!r}"
    if keep:
        clean[name] = converted
    return None


def coerce_args(method: Callable, args: dict) -> Tuple[dict, Optional[str]]:
    """Validate *args* against *method*'s signature.

    Returns ``(clean_args, None)`` on success or ``(args, message)`` when a required argument
    is missing, an unknown one is passed, or a value cannot be converted to its declared type.
    The message is written for the model so it can correct the call. A ``null`` for a parameter
    that has a default means "use the default" and is dropped; a ``null`` for a required
    parameter that does not accept it is an error. Unannotated parameters are not converted.
    """
    try:
        params = inspect.signature(method).parameters
    except (ValueError, TypeError):
        return args, None
    hints = _type_hints(method)
    named = {n: p for n, p in params.items()
             if n != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    takes_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
    problems: List[str] = []
    clean: dict = {}
    for name, value in args.items():
        problem = _check_arg(name, value, named, hints, takes_kwargs, clean)
        if problem:
            problems.append(problem)
    missing = [n for n, p in named.items() if p.default is inspect.Parameter.empty and n not in args]
    if missing:
        problems.append(f"missing required argument(s): {', '.join(missing)}")
    if problems:
        return args, "Invalid arguments: " + "; ".join(problems)
    return clean, None
