"""The a-prefixed API: results, real concurrency, per-agent serialisation, streaming and cancellation."""
import asyncio
import threading
import time
import unittest
from unittest.mock import MagicMock

from maki.agents import Agent, AgentManager
from maki.aio import aiterate, run_in_thread
from maki.backend import LLMBackend
from maki.exceptions import MakiNetworkError
from maki.objects import LLMResponse


def reply(text):
    return LLMResponse(content=text, model="m", prompt_tokens=1, completion_tokens=1, total_tokens=2,
                       elapsed_seconds=0.0)


class Slow(LLMBackend):
    """Answers after ``hold`` seconds; records overlap between concurrent calls."""
    supports_history = True

    def __init__(self, label="x", hold=0.0, chunks=("a", "b", "c")):
        self.model, self.temperature, self.label, self.hold, self.chunks = label, 0.0, label, hold, chunks
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.stream_closed = threading.Event()
        self.fail_times = 0

    def _enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def _leave(self):
        with self.lock:
            self.active -= 1

    def request(self, prompt):
        return self.chat(prompt)

    def chat(self, prompt, history=None, config=None, system=None, images=None):
        self._enter()
        try:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise MakiNetworkError("down")
            time.sleep(self.hold)
            return reply(f"{self.label}:{prompt.splitlines()[0][:20]}")
        finally:
            self._leave()

    def stream(self, prompt, history=None, config=None, system=None):
        try:
            for chunk in self.chunks:
                time.sleep(self.hold)
                yield chunk
        finally:
            self.stream_closed.set()


class TestBridges(unittest.IsolatedAsyncioTestCase):
    async def test_run_in_thread_returns_and_raises(self):
        self.assertEqual(await run_in_thread(lambda a, b=1: a + b, 2, b=3), 5)
        with self.assertRaises(ZeroDivisionError):
            await run_in_thread(lambda: 1 / 0)

    async def test_context_variables_reach_the_worker(self):
        import contextvars
        trace = contextvars.ContextVar("trace", default="none")
        trace.set("abc")
        self.assertEqual(await run_in_thread(trace.get), "abc")
        self.assertEqual([x async for x in aiterate(lambda: [trace.get()])], ["abc"])
        agent_seen = []

        class Spy(Slow):
            def chat(self, *a, **k):
                agent_seen.append(trace.get())
                return super().chat(*a, **k)
        await Agent("a", Spy()).aexecute_task("x")
        self.assertEqual(agent_seen, ["abc"])

    async def test_run_in_thread_does_not_block_the_loop(self):
        ticks = []

        async def heartbeat():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)
        beat = asyncio.create_task(heartbeat())
        await run_in_thread(time.sleep, 0.2)
        beat.cancel()
        self.assertGreater(len(ticks), 5)

    async def test_aiterate_yields_in_order_and_runs_off_the_loop(self):
        main = threading.get_ident()
        seen = []

        def make():
            seen.append(threading.get_ident())
            yield from range(4)
        self.assertEqual([x async for x in aiterate(make)], [0, 1, 2, 3])
        self.assertNotEqual(seen[0], main)

    async def test_aiterate_forwards_errors_after_the_items(self):
        def make():
            yield 1
            raise MakiNetworkError("boom")
        got = []
        with self.assertRaises(MakiNetworkError):
            async for x in aiterate(make):
                got.append(x)
        self.assertEqual(got, [1])

    async def test_an_error_in_creating_the_iterable_is_forwarded(self):
        def make():
            raise ValueError("bad")
        with self.assertRaises(ValueError):
            async for _ in aiterate(make):
                pass

    async def test_breaking_out_closes_the_source(self):
        closed = threading.Event()

        def make():
            try:
                for i in range(1000):
                    time.sleep(0.005)
                    yield i
            finally:
                closed.set()
        gen = aiterate(make)
        async for x in gen:
            if x == 2:
                break
        await gen.aclose()
        self.assertTrue(await run_in_thread(closed.wait, 2))


class TestAgent(unittest.IsolatedAsyncioTestCase):
    async def test_aexecute_task_matches_the_sync_result(self):
        agent = Agent("a", Slow("one"))
        self.assertEqual(await agent.aexecute_task("hello"), agent.execute_task("hello"))

    async def test_different_agents_overlap(self):
        backend = Slow(hold=0.3)
        agents = [Agent(f"a{i}", backend) for i in range(3)]
        started = time.monotonic()
        await asyncio.gather(*(a.aexecute_task("t") for a in agents))
        self.assertLess(time.monotonic() - started, 0.7)  # three sequential calls would take 0.9
        self.assertEqual(backend.peak, 3)

    async def test_one_agent_runs_one_task_at_a_time(self):
        backend = Slow(hold=0.1)
        agent = Agent("a", backend, stateful=True)
        await asyncio.gather(*(agent.aexecute_task(f"t{i}") for i in range(4)))
        self.assertEqual(backend.peak, 1)
        self.assertEqual(len(agent._conversation_memory), 8)  # four whole turns, none interleaved

    async def test_queued_calls_on_one_agent_do_not_starve_other_agents(self):
        from concurrent.futures import ThreadPoolExecutor
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        busy = Agent("busy", Slow(hold=0.25))
        other = Agent("other", Slow(hold=0.0))
        queued = [asyncio.create_task(busy.aexecute_task(f"t{i}")) for i in range(6)]
        await asyncio.sleep(0.05)  # let the busy agent's calls line up
        started = time.monotonic()
        await asyncio.wait_for(other.aexecute_task("quick"), 5)
        self.assertLess(time.monotonic() - started, 0.5)  # not stuck behind five queued busy calls
        await asyncio.gather(*queued)

    async def test_a_cancelled_waiter_does_not_hold_up_the_queue(self):
        agent = Agent("a", Slow(hold=0.1))
        first = asyncio.create_task(agent.aexecute_task("first"))
        second = asyncio.create_task(agent.aexecute_task("second"))
        third = asyncio.create_task(agent.aexecute_task("third"))
        await asyncio.sleep(0.02)
        second.cancel()
        self.assertIn("x:", await asyncio.wait_for(third, 3))
        await first
        with self.assertRaises(asyncio.CancelledError):
            await second

    async def test_errors_propagate(self):
        backend = Slow()
        backend.fail_times = 1
        with self.assertRaises(MakiNetworkError):
            await Agent("a", backend).aexecute_task("x")

    async def test_retry_variant_recovers(self):
        backend = Slow()
        backend.fail_times = 1
        self.assertIn("ok", await Agent("a", Slow("ok")).aexecute_task_with_retry("go", retry_delay=0))
        agent = Agent("b", backend)
        self.assertIn("x:", await agent.aexecute_task_with_retry("go", retry_delay=0.01))

    async def test_astream_task_yields_chunks(self):
        agent = Agent("a", Slow(chunks=("he", "llo")))
        self.assertEqual("".join([c async for c in agent.astream_task("hi")]), "hello")

    async def test_astream_task_without_streaming_raises(self):
        class NoStream(Slow):
            stream = LLMBackend.stream
        with self.assertRaises(NotImplementedError):
            async for _ in Agent("a", NoStream()).astream_task("hi"):
                pass

    async def test_leaving_a_stream_early_releases_the_agent(self):
        backend = Slow(hold=0.02, chunks=tuple("abcdefghij" * 30))
        agent = Agent("a", backend)
        gen = agent.astream_task("hi")
        async for chunk in gen:
            break
        await gen.aclose()
        self.assertTrue(await run_in_thread(backend.stream_closed.wait, 3))
        self.assertTrue(await asyncio.wait_for(agent.aexecute_task("again"), 3))  # lock is free again

    async def test_arun_loop_delegates(self):
        agent = Agent("a", Slow())
        agent.run_loop = MagicMock(return_value="done")
        self.assertEqual(await agent.arun_loop("goal", ["r"], "judge", overwrite=True), "done")
        agent.run_loop.assert_called_once_with("goal", ["r"], "judge", overwrite=True)


class TestManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = Slow(hold=0.25)
        self.manager = AgentManager(self.backend)
        for name in ("a", "b", "c"):
            self.manager.add_agent(name)

    async def test_aassign_task(self):
        self.assertIn("x:", await self.manager.aassign_task("a", "hello"))
        with self.assertRaises(ValueError):
            await self.manager.aassign_task("nope", "hello")

    async def test_gather_runs_agents_together_and_keeps_order(self):
        started = time.monotonic()
        results = await self.manager.agather_tasks([{"agent": n, "task": f"task-{n}"} for n in "abc"])
        self.assertLess(time.monotonic() - started, 0.6)  # sequential would be 0.75
        self.assertEqual([r.split("task-")[1][0] for r in results], ["a", "b", "c"])

    async def test_aassign_task_wraps_errors_like_assign_task(self):
        self.backend.fail_times = 1
        with self.assertRaises(RuntimeError) as caught:
            await self.manager.aassign_task("a", "hello")
        self.assertIsInstance(caught.exception.__cause__, MakiNetworkError)
        self.assertIn("agent 'a'", str(caught.exception))

    async def test_gathering_many_tasks_for_one_agent_does_not_starve_another(self):
        from concurrent.futures import ThreadPoolExecutor
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        self.backend.hold = 0.2
        busy = asyncio.create_task(self.manager.agather_tasks([{"agent": "a", "task": f"t{i}"} for i in range(5)]))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        quick = Slow(hold=0.0)
        self.manager.agents["b"].maki = quick
        await asyncio.wait_for(self.manager.aassign_task("b", "quick"), 5)
        self.assertLess(time.monotonic() - started, 0.5)
        await busy

    async def test_objects_without_an_async_method_still_work(self):
        class Plain:
            def execute_task(self, task, context=None):
                return f"plain:{task}"
        self.manager.agents["p"] = Plain()
        self.assertEqual(await self.manager.aassign_task("p", "t"), "plain:t")

    async def test_gather_errors(self):
        tasks = [{"agent": "a", "task": "ok"}, {"agent": "missing", "task": "x"}]
        with self.assertRaises(ValueError):
            await self.manager.agather_tasks(tasks)  # raised by assign_task for the unknown agent
        results = await self.manager.agather_tasks(tasks, return_exceptions=True)
        self.assertIn("x:", results[0])
        self.assertIsInstance(results[1], ValueError)

    async def test_gather_validates_before_starting_anything(self):
        for bad in ([{"agent": "a"}], ["a"], [{"task": "x"}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await self.manager.agather_tasks(bad)
        self.assertEqual(self.backend.peak, 0)

    async def test_collaborative_and_workflow_delegate(self):
        self.manager.collaborative_task = MagicMock(return_value="merged")
        self.manager.run_workflow = MagicMock(return_value={"r": 1})
        self.assertEqual(await self.manager.acollaborative_task("t", ["a", "b"], strict=True), "merged")
        self.manager.collaborative_task.assert_called_once_with("t", ["a", "b"], None, True)
        self.assertEqual(await self.manager.arun_workflow([{"agent": "a", "task": "x"}], workflow_id="w"), {"r": 1})
        self.manager.run_workflow.assert_called_once_with([{"agent": "a", "task": "x"}], workflow_id="w")


class TestProxy(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_async_methods_delegate(self):
        from maki.distributed.proxy import AgentProxy
        proxy = object.__new__(AgentProxy)
        proxy.execute_task = MagicMock(return_value="remote")
        proxy.stream_task = MagicMock(return_value=iter(["x", "y"]))
        self.assertEqual(await proxy.aexecute_task("t", {"c": 1}, True, "trace"), "remote")
        proxy.execute_task.assert_called_once_with("t", {"c": 1}, True, "trace")
        self.assertEqual([c async for c in proxy.astream_task("t")], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
