"""
Async bridges for the synchronous agent API.

``Agent``, ``AgentManager`` and ``AgentProxy`` do their work on blocking calls (HTTP through
``Connector``, a per-agent execution lock, plugin methods). The ``a``-prefixed methods run those calls
on worker threads so an event loop keeps running while a model answers:

    answer = await agent.aexecute_task("Summarise this")
    async for chunk in agent.astream_task("Tell a story"):
        ...
    results = await manager.agather_tasks([{"agent": "a", "task": "x"}, {"agent": "b", "task": "y"}])

Different agents run at the same time; calls to one agent are still serialised by that agent's lock.
Cancelling an awaiting task stops the wait, not the worker thread: a call already sent to the model
finishes (and is billed). Worker threads come from the loop's default executor, so very many
concurrent calls queue behind its size (calls waiting on one busy agent occupy a thread each).
Callbacks and plugin methods run on those threads, not on the loop.
"""

import asyncio
import contextvars
import functools
import threading
import weakref
from typing import Any, AsyncIterator, Callable, Iterable, TypeVar

T = TypeVar("T")

_DATA, _ERROR, _DONE = 0, 1, 2


def async_gate(owner: Any) -> asyncio.Lock:
    """The asyncio lock that serialises *owner*'s async calls on the running event loop.

    An agent runs one task at a time (a threading lock inside ``execute_task``). Waiting for that lock
    on a worker thread would park one thread per queued call; waiting on this gate parks none, so the
    executor stays free for other agents. One lock per event loop, created on first use.
    """
    loop = asyncio.get_running_loop()
    try:
        gates = owner.__dict__.setdefault("_async_gates", weakref.WeakKeyDictionary())
        key = loop
        gates.get(key)  # probes that this loop type can be weakly referenced
    except TypeError:  # an event loop implementation without weakref support: key by id (a tiny leak)
        gates = owner.__dict__.setdefault("_async_gates_by_id", {})
        key = id(loop)
    gate = gates.get(key)
    if gate is None:
        gate = gates[key] = asyncio.Lock()
    return gate


async def run_in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``fn(*args, **kwargs)`` on a worker thread and await its result.

    The caller's context variables (trace ids, logging context) are visible to ``fn``, as with
    :func:`asyncio.to_thread`.
    """
    context = contextvars.copy_context()
    call = functools.partial(context.run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(None, call)


def _pump(make_iterable: Callable[[], Iterable[T]], post: Callable[..., None], stop: threading.Event) -> None:
    """Worker-thread body of :func:`aiterate`: feed items (then the end or the error) to the consumer."""
    iterator = None
    try:
        iterator = iter(make_iterable())
        for item in iterator:
            if stop.is_set():
                break
            post(_DATA, item)
    except BaseException as exc:  # forwarded to the consumer, which re-raises it
        post(_ERROR, exc)
    else:
        post(_DONE)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


async def aiterate(make_iterable: Callable[[], Iterable[T]]) -> AsyncIterator[T]:
    """Iterate a blocking iterable on a worker thread, yielding its items to the event loop.

    *make_iterable* is called on the worker thread (so a generator function's body, including its
    first blocking step, never runs on the loop). Errors raised by the iterable are re-raised here.
    Closing this generator early (``break``, cancellation) stops the worker after its current item
    and closes the underlying iterable, which releases whatever it holds.
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[tuple]" = asyncio.Queue()
    stop = threading.Event()

    def post(kind: int, value: Any = None) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
        except RuntimeError:  # the loop is gone; nobody is listening
            stop.set()

    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(_pump, make_iterable, post, stop), name="maki-aiterate",
                     daemon=True).start()
    try:
        while True:
            kind, value = await queue.get()
            if kind == _DATA:
                yield value
            elif kind == _ERROR:
                raise value
            else:
                return
    finally:
        stop.set()
