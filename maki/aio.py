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
concurrent calls queue behind its size.
"""

import asyncio
import functools
import threading
from typing import Any, AsyncIterator, Callable, Iterable, TypeVar

T = TypeVar("T")

_DATA, _ERROR, _DONE = 0, 1, 2


async def run_in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``fn(*args, **kwargs)`` on a worker thread and await its result."""
    return await asyncio.get_running_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))


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

    threading.Thread(target=_pump, args=(make_iterable, post, stop), name="maki-aiterate", daemon=True).start()
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
