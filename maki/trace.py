"""
Run traces: a JSON-lines record of what an agent did, span by span.

``Agent(trace="runs.jsonl")`` (or ``Agent(callbacks=[TraceCallback("runs.jsonl")])``) appends one line per event:
a *task* span (one ``execute_task``), the *model* rounds inside it and the *tool* calls each round asked for.
Every span has an id and the id of its parent, and every span of one run shares a trace id, so a trace answers
"why did the model fail at round 4" without a debugger. Read it with ``maki trace show runs.jsonl``.

Across ``maki serve`` the trace id travels in the ``X-Maki-Trace-Id`` header (with the caller's span in
``X-Maki-Parent-Span``), so a client's trace file and a server's trace file join into one tree:
``maki trace show client.jsonl server.jsonl``.

Standard library only. The file is created with mode ``0600`` (traces hold prompts and tool arguments), values are
clipped to ``max_chars``, argument values under keys that look like credentials are masked, and the file rolls
over to ``<path>.1`` at ``max_bytes``. ``include_content=False`` records timing, token counts and names only.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .callbacks import AgentCallback
from .connector import redact

logger = logging.getLogger(__name__)

TRACE_HEADER = "X-Maki-Trace-Id"
PARENT_SPAN_HEADER = "X-Maki-Parent-Span"
SCHEMA_VERSION = 1

_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
_SECRET_KEY = re.compile(r"(key|token|secret|passw|authoriz|credential|cookie)", re.IGNORECASE)


def new_id() -> str:
    """A fresh 16-hex-digit id for a trace or a span."""
    return uuid.uuid4().hex[:16]


def valid_id(value: Any) -> Optional[str]:
    """*value* if it is safe to write into a trace file and a log line (1-64 of ``A-Za-z0-9._-``), else None.
    Ids arrive in request headers, so they are never trusted as they are."""
    return value if isinstance(value, str) and _ID_PATTERN.fullmatch(value) else None


@dataclass(eq=False)
class _Frame:
    owner: int
    trace: str
    span: str
    parent: Optional[str]
    kind: str
    started: float = field(default_factory=time.perf_counter)
    closed: bool = False  # ended (possibly on another thread, whose context still lists it)


#: the open spans of the current thread or asyncio task, outermost first
_stack: contextvars.ContextVar = contextvars.ContextVar("maki_trace_stack", default=())
#: (trace id, parent span id) a caller handed us, e.g. from request headers
_inherited: contextvars.ContextVar = contextvars.ContextVar("maki_trace_inherited", default=None)


def current_trace() -> Optional[Tuple[str, str]]:
    """``(trace id, span id)`` of the innermost open span in this context, or None. The remote proxy sends
    these so the server's spans hang under the caller's."""
    stack = [f for f in _stack.get() if not f.closed]
    return (stack[-1].trace, stack[-1].span) if stack else None


def set_inherited(trace_id: Optional[str], parent_span: Optional[str] = None) -> None:
    """Make the spans started in this context part of *trace_id*, under *parent_span*. Ids that are not
    safe (see :func:`valid_id`) are ignored."""
    trace_id, parent_span = valid_id(trace_id), valid_id(parent_span)
    _inherited.set((trace_id, parent_span) if trace_id else None)


class trace_context:
    """``with trace_context(trace_id, parent_span): agent.execute_task(...)`` joins that run to a trace."""

    def __init__(self, trace_id: Optional[str], parent_span: Optional[str] = None) -> None:
        self._value = ((valid_id(trace_id), valid_id(parent_span)) if valid_id(trace_id) else None)
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "trace_context":
        self._token = _inherited.set(self._value)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            try:
                _inherited.reset(self._token)
            except ValueError:  # exited in another context than it was entered in (an async generator)
                _inherited.set(None)


def _clip(text: Any, limit: int) -> str:
    text = redact(text if isinstance(text, str) else str(text))  # a URL with ?api_key=... in a result or error
    return text if len(text) <= limit else text[:limit] + f"…[+{len(text) - limit} chars]"


def _scrub(value: Any, depth: int = 0) -> Any:
    """*value* with the values under credential-looking keys masked."""
    if depth > 6:
        return "…"
    if isinstance(value, dict):
        return {str(k): ("***" if _SECRET_KEY.search(str(k)) else _scrub(v, depth + 1)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, depth + 1) for v in value]
    return value


class _Sink:
    """One append-only file shared by every TraceCallback that writes to the same path in this process, so
    their lines never interleave inside a write and the size count and rollover agree."""

    def __init__(self, path: str, max_bytes: int) -> None:
        self.path, self.max_bytes = path, max_bytes
        self.lock = threading.Lock()
        self.fd: Optional[int] = None
        self.size = 0

    def _open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        self.size = os.fstat(self.fd).st_size

    def append(self, line: bytes) -> None:
        with self.lock:
            if self.fd is None:
                self._open()
            if self.max_bytes and self.size and self.size + len(line) > self.max_bytes:
                os.close(self.fd)  # type: ignore[arg-type]
                self.fd = None  # never keep a closed descriptor: its number may be reused by another file
                os.replace(self.path, self.path + ".1")
                self._open()
            os.write(self.fd, line)  # type: ignore[arg-type]
            self.size += len(line)

    def close(self) -> None:
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


_sinks: Dict[Tuple[str, int], _Sink] = {}
_sinks_lock = threading.Lock()


def _sink_for(path: str, max_bytes: int) -> _Sink:
    key = (os.path.realpath(path), max_bytes)
    with _sinks_lock:
        if key not in _sinks:
            _sinks[key] = _Sink(*key)
        return _sinks[key]


def close_all() -> None:
    """Close every open trace file (they reopen on the next event). For tests and orderly shutdown."""
    with _sinks_lock:
        sinks = list(_sinks.values())
    for sink in sinks:
        sink.close()


class TraceCallback(AgentCallback):
    """Writes the events of an agent's runs to a JSON-lines file; see the module docstring.

    Args:
        path: the trace file (appended to, created ``0600``).
        max_chars: longest text (task, result, arguments) kept per field.
        max_bytes: roll the file over to ``<path>.1`` beyond this size (one backup is kept); 0 disables.
        include_content: False keeps only timing, token counts and names.
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"], max_chars: int = 500, max_bytes: int = 50 * 1024 * 1024,
                 include_content: bool = True) -> None:
        if max_chars < 1 or max_bytes < 0:
            raise ValueError("max_chars must be >= 1 and max_bytes >= 0")
        self.path = os.path.abspath(os.fspath(path))  # absolute: a later chdir must not split the file
        self.max_chars, self.max_bytes, self.include_content = max_chars, max_bytes, include_content
        self._sink = _sink_for(self.path, max_bytes)
        self._lock = threading.Lock()  # guards the open-span tables below
        self._warned = False
        # Streaming hops threads between chunks, which loses context variables, so task spans are also
        # found by (agent, task) here; model and tool spans only ever occur on one thread.
        self._open_tasks: Dict[Tuple[int, str], List[_Frame]] = {}
        self._open_tools: Dict[int, _Frame] = {}
        self._open_models: Dict[int, List[_Frame]] = {}

    # -- frames --------------------------------------------------------------------------------------------

    def _mine(self) -> List[_Frame]:
        return [f for f in _stack.get() if f.owner == id(self) and not f.closed]

    def _push(self, kind: str) -> _Frame:
        stack = _stack.get()
        mine = [f for f in stack if f.owner == id(self) and not f.closed]
        inherited = _inherited.get()
        if mine:
            trace, parent = mine[-1].trace, mine[-1].span
        elif inherited:
            trace, parent = inherited
        else:
            trace, parent = None, None
        frame = _Frame(id(self), trace or new_id(), new_id(), parent, kind)
        _stack.set(tuple(f for f in stack if not f.closed) + (frame,))
        return frame

    def _pop(self, frame: _Frame) -> List[_Frame]:
        """Remove *frame* from this context's stack; the frames above it were never closed and come back."""
        frame.closed = True
        stack = list(_stack.get())
        if frame not in stack:
            return []
        at = stack.index(frame)
        orphans = [f for f in stack[at + 1:] if f.owner == id(self)]
        _stack.set(tuple(stack[:at]) + tuple(f for f in stack[at + 1:] if f.owner != id(self)))
        return orphans

    def _top(self, kind: str) -> Optional[_Frame]:
        for frame in reversed(self._mine()):
            if frame.kind == kind:
                return frame
        return None

    # -- output --------------------------------------------------------------------------------------------

    def _write(self, frame: _Frame, phase: str, agent: Any, **fields: Any) -> None:
        record: Dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace": frame.trace, "span": frame.span, "parent": frame.parent,
            "kind": frame.kind, "ph": phase, "agent": getattr(agent, "name", None),
        }
        if phase != "start" and phase != "retry":
            record["seconds"] = round(time.perf_counter() - frame.started, 4)
        record.update({k: v for k, v in fields.items() if v is not None})
        try:
            self._sink.append((json.dumps(record, default=str, ensure_ascii=False) + "\n").encode("utf-8"))
        except Exception:
            if not self._warned:  # one warning, not one per event
                self._warned = True
                logger.warning("TraceCallback could not write %s", self.path, exc_info=True)

    def close(self) -> None:
        """Close the file (it is reopened by the next event)."""
        self._sink.close()

    def _error(self, error: BaseException) -> str:
        """The error for a record: type and message, or just the type when content is not recorded."""
        if not self.include_content:
            return type(error).__name__
        return _clip(f"{type(error).__name__}: {error}", 300)

    def _text(self, value: Any) -> Optional[str]:
        return _clip(value, self.max_chars) if self.include_content and value is not None else None

    def _close_orphans(self, orphans: Iterable[_Frame], agent: Any) -> None:
        for frame in orphans:
            frame.closed = True
            self._write(frame, "end", agent, status="abandoned")
            with self._lock:
                self._open_tools = {k: v for k, v in self._open_tools.items() if v is not frame}
                self._open_models = {k: [f for f in v if f is not frame] for k, v in self._open_models.items()}
                self._open_models = {k: v for k, v in self._open_models.items() if v}

    # -- task ----------------------------------------------------------------------------------------------

    def on_task_start(self, agent: Any, task: str) -> None:
        frame = self._push("task")
        with self._lock:
            self._open_tasks.setdefault((id(agent), task), []).append(frame)
        self._write(frame, "start", agent, task=self._text(task))

    def _finish_task(self, agent: Any, task: str, phase: str, **fields: Any) -> None:
        with self._lock:
            frames = self._open_tasks.get((id(agent), task))
            if not frames:
                return
            frame = frames.pop(0)
            if not frames:
                del self._open_tasks[(id(agent), task)]
        self._close_orphans(self._pop(frame), agent)
        self._write(frame, phase, agent, **fields)

    def on_task_end(self, agent: Any, task: str, result: str, elapsed: float) -> None:
        self._finish_task(agent, task, "end", status="ok", result=self._text(result))

    def on_task_error(self, agent: Any, task: str, error: BaseException, elapsed: float) -> None:
        self._finish_task(agent, task, "error", status="error", error=self._error(error))

    # -- model ---------------------------------------------------------------------------------------------

    def on_model_start(self, agent: Any, messages: list) -> None:
        frame = self._push("model")
        with self._lock:
            self._open_models.setdefault(id(agent), []).append(frame)
        self._write(frame, "start", agent, messages=len(messages) if hasattr(messages, "__len__") else None)

    def on_model_end(self, agent: Any, response: Any, tool_calls: Any) -> None:
        with self._lock:
            open_here = self._open_models.get(id(agent))
            frame = open_here.pop() if open_here else None  # a stream can end on another thread than it began
            if open_here is not None and not open_here:
                del self._open_models[id(agent)]
        if frame is None:
            return
        self._close_orphans(self._pop(frame), agent)
        names = [getattr(c, "name", str(c)) for c in tool_calls] if tool_calls else None
        self._write(frame, "end", agent, status="ok", tool_calls=names,
                    prompt_tokens=getattr(response, "prompt_tokens", None),
                    completion_tokens=getattr(response, "completion_tokens", None),
                    cached_tokens=getattr(response, "cached_tokens", None) or None)

    # -- tool ----------------------------------------------------------------------------------------------

    def on_tool_start(self, agent: Any, call: Any) -> None:
        frame = self._push("tool")
        with self._lock:
            self._open_tools[id(call)] = frame
        args = getattr(call, "args", None)
        text = None
        if self.include_content and args is not None:
            try:
                text = _clip(json.dumps(_scrub(args), default=str, ensure_ascii=False), self.max_chars)
            except Exception:
                text = "<unprintable>"
        self._write(frame, "start", agent, tool=getattr(call, "name", None), args=text)

    def _tool_frame(self, call: Any) -> Optional[_Frame]:
        with self._lock:
            known = self._open_tools.get(id(call))
        return known or self._top("tool")

    def on_tool_end(self, agent: Any, call: Any, result: str, elapsed: float) -> None:
        frame = self._tool_frame(call)
        if frame is None:
            return
        with self._lock:
            self._open_tools.pop(id(call), None)
        self._close_orphans(self._pop(frame), agent)
        self._write(frame, "end", agent, status="ok", tool=getattr(call, "name", None), result=self._text(result))

    def on_tool_error(self, agent: Any, call: Any, error: BaseException, elapsed: float) -> None:
        frame = self._tool_frame(call)
        if frame is None:
            return
        with self._lock:
            self._open_tools.pop(id(call), None)
        self._close_orphans(self._pop(frame), agent)
        self._write(frame, "error", agent, status="error", tool=getattr(call, "name", None),
                    error=self._error(error))

    def on_tool_retry(self, agent: Any, call: Any, error: BaseException, attempt: int, delay: float) -> None:
        frame = self._tool_frame(call)
        if frame is not None:
            self._write(frame, "retry", agent, attempt=attempt, delay=round(delay, 3),
                        error=self._error(error))


# ----------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------


def read_trace(*paths: Union[str, "os.PathLike[str]"]) -> List[Dict[str, Any]]:
    """The records of one or more trace files, in time order. Lines that are not JSON objects (a torn last
    line after a crash, a foreign file) are skipped."""
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("trace") and record.get("span"):
                    records.append(record)
    records.sort(key=lambda r: str(r.get("ts", "")))
    return records


@dataclass
class Span:
    """One span of a trace: its start and end records merged."""
    id: str
    trace: str
    parent: Optional[str]
    kind: str
    agent: Optional[str]
    start: Dict[str, Any]
    end: Optional[Dict[str, Any]] = None
    retries: int = 0
    children: List["Span"] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.end is None:
            return "running"
        return str(self.end.get("status") or ("error" if self.end.get("ph") == "error" else "ok"))

    @property
    def seconds(self) -> Optional[float]:
        return self.end.get("seconds") if self.end else None


def build_spans(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Span]]:
    """``{trace id: root spans}`` from trace records. A span whose parent is not in the records (the caller's
    trace file was not given) is a root."""
    spans: Dict[Tuple[str, str], Span] = {}
    for r in records:
        key = (r["trace"], r["span"])
        if r.get("ph") == "start":
            spans[key] = Span(r["span"], r["trace"], r.get("parent"), str(r.get("kind")), r.get("agent"), r)
        elif key in spans:
            if r.get("ph") == "retry":
                spans[key].retries += 1
            else:
                spans[key].end = r
    roots: Dict[str, List[Span]] = {}
    for (trace, _), span in spans.items():
        parent = spans.get((trace, span.parent)) if span.parent else None
        (parent.children if parent else roots.setdefault(trace, [])).append(span)
    return roots


def list_traces(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One summary row per trace, oldest first: id, start time, agents, span count, errors, tokens, seconds."""
    rows: Dict[str, Dict[str, Any]] = {}
    for r in records:
        row = rows.setdefault(r["trace"], {"trace": r["trace"], "start": r.get("ts"), "agents": [], "spans": 0,
                                           "errors": 0, "tokens": 0, "seconds": 0.0})
        agent = r.get("agent")
        if agent and agent not in row["agents"]:
            row["agents"].append(agent)
        if r.get("ph") == "start":
            row["spans"] += 1
        if r.get("ph") == "error":
            row["errors"] += 1
        if r.get("kind") == "model" and r.get("ph") == "end":
            row["tokens"] += int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
        if r.get("kind") == "task" and r.get("ph") != "start" and not r.get("parent"):
            row["seconds"] = round(row["seconds"] + float(r.get("seconds") or 0), 3)
    return list(rows.values())


def _describe(span: Span) -> str:
    s, e = span.start, span.end or {}
    if span.kind == "task":
        what = repr(s.get("task", ""))
    elif span.kind == "model":
        tokens = ""
        if "prompt_tokens" in e or "completion_tokens" in e:
            tokens = f" {e.get('prompt_tokens', 0)} in / {e.get('completion_tokens', 0)} out tokens"
        calls = f" -> {', '.join(e['tool_calls'])}" if e.get("tool_calls") else ""
        what = f"{s.get('messages', '?')} messages{tokens}{calls}"
    else:
        what = f"{s.get('tool', '?')} {s.get('args', '')}".rstrip()
    detail = e.get("error") or (f"-> {e['result']!r}" if span.kind == "tool" and e.get("result") else "")
    parts = [f"{span.kind:<5}", what]
    if span.retries:
        parts.append(f"({span.retries} retr{'y' if span.retries == 1 else 'ies'})")
    parts.append(span.status if span.status != "ok" else "")
    seconds = "" if span.seconds is None else f"{span.seconds:.2f}s"
    return "  ".join(p for p in parts + [seconds, str(detail)] if p)


def render_trace(trace_id: str, roots: Sequence[Span], width: int = 160) -> str:
    """The span tree of one trace as text."""
    lines = [f"trace {trace_id}"]

    def walk(span: Span, prefix: str, last: bool, top: bool) -> None:
        branch = "" if top else ("└─ " if last else "├─ ")
        note = f" [parent {span.parent} is in another file]" if top and span.parent else ""
        text = f"{prefix}{branch}{_describe(span)}{note}"
        lines.append(text if len(text) <= width else text[:width - 1] + "…")
        child_prefix = prefix if top else prefix + ("   " if last else "│  ")
        for i, child in enumerate(span.children):
            walk(child, child_prefix, i == len(span.children) - 1, False)

    for root in roots:
        walk(root, "", True, True)
    return "\n".join(lines)


def find_trace(records: Sequence[Dict[str, Any]], wanted: Optional[str] = None) -> Optional[str]:
    """The id of the trace *wanted* (an id or unambiguous prefix), or the most recent one when None."""
    rows = list_traces(records)
    if not rows:
        return None
    if wanted is None:
        return rows[-1]["trace"]
    matches = [r["trace"] for r in rows if r["trace"].startswith(wanted)]
    exact = [m for m in matches if m == wanted]
    if exact:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"{wanted!r} matches {len(matches)} traces; give more of the id")
    return None


# ----------------------------------------------------------------------------------------------------------
# CLI: maki trace list|show
# ----------------------------------------------------------------------------------------------------------


def add_cli_arguments(parser: Any) -> None:
    sub = parser.add_subparsers(dest="trace_command", metavar="<list|show>")
    ls = sub.add_parser("list", help="List the traces in one or more trace files")
    ls.add_argument("files", nargs="+", metavar="FILE")
    show = sub.add_parser("show", help="Show a trace as a span tree (the latest one by default)")
    show.add_argument("files", nargs="+", metavar="FILE", help="trace files; give a client's and a server's to join them")
    show.add_argument("--trace", default=None, metavar="ID", help="trace id or prefix (default: the latest)")
    show.add_argument("--json", action="store_true", help="print the raw records of the trace as JSON lines")
    show.add_argument("--width", type=int, default=160, help="cut lines at this width (default 160)")


def run_cli(args: Any, out: Any = None) -> int:
    import sys
    out = out or sys.stdout
    command = getattr(args, "trace_command", None)
    if command not in ("list", "show"):
        print("usage: maki trace {list,show} FILE...", file=sys.stderr)
        return 2
    try:
        records = read_trace(*args.files)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if command == "list":
        rows = list_traces(records)
        if not rows:
            print("no traces found", file=out)
        for row in rows:
            print(f"{row['trace']}  {row['start']}  {row['spans']:>3} spans  {row['errors']} errors  "
                  f"{row['tokens']:>6} tokens  {row['seconds']:.2f}s  {', '.join(row['agents'])}", file=out)
        return 0
    try:
        trace_id = find_trace(records, args.trace)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if trace_id is None:
        print("Error: no such trace" if args.trace else "Error: no traces found", file=sys.stderr)
        return 1
    chosen = [r for r in records if r["trace"] == trace_id]
    if args.json:
        for r in chosen:
            print(json.dumps(r, ensure_ascii=False), file=out)
        return 0
    print(render_trace(trace_id, build_spans(chosen).get(trace_id, []), max(40, args.width)), file=out)
    return 0
