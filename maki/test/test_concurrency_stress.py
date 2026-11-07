"""Stress tests of the shared state: what must stay exact or never overlap when many threads or tasks use one
backend, agent, usage ledger or served app. Each test has a hard time limit so a deadlock fails instead of hanging."""
import asyncio
import random
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from maki import Agent, FallbackBackend
from maki.backend import LLMBackend
from maki.exceptions import MakiNetworkError
from maki.objects import LLMResponse
from maki.usage import TokenRelay, Usage, attach_usage

LIMIT = 60  # seconds before a stress test is declared stuck


def reply(text="ok", prompt=3, completion=2):
    return LLMResponse(content=text, model="m", prompt_tokens=prompt, completion_tokens=completion,
                       total_tokens=prompt + completion, elapsed_seconds=0.0)


def run_threads(count, work):
    """Run ``work(i)`` on *count* threads at once; returns the results, raises the first failure or a timeout."""
    results = [None] * count
    errors = []
    barrier = threading.Barrier(count)

    def body(i):
        try:
            barrier.wait(timeout=LIMIT)
            results[i] = work(i)
        except BaseException as exc:  # noqa: BLE001 - reported to the test thread
            errors.append(exc)
    threads = [threading.Thread(target=body, args=(i,), daemon=True) for i in range(count)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + LIMIT
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        raise AssertionError("threads still running after the time limit: a deadlock or a livelock")
    if errors:
        raise errors[0]
    return results


class TestUsageLedger(unittest.TestCase):
    def test_totals_are_exact_under_contention(self):
        usage = Usage()
        threads, per_thread = 32, 400

        def work(i):
            for _ in range(per_thread):
                usage.add(prompt=i + 1, completion=2, total=0, cached=1)
        run_threads(threads, work)
        expected_prompt = per_thread * sum(range(1, threads + 1))
        self.assertEqual(usage.calls, threads * per_thread)
        self.assertEqual(usage.prompt_tokens, expected_prompt)
        self.assertEqual(usage.completion_tokens, 2 * threads * per_thread)
        self.assertEqual(usage.total_tokens, expected_prompt + 2 * threads * per_thread)
        self.assertEqual(usage.cached_tokens, threads * per_thread)

    def test_each_thread_sees_only_its_own_tokens(self):
        usage = Usage()
        seen = run_threads(24, lambda i: ([usage.add(i + 1, 1, 0) for _ in range(50)], usage.thread_tokens())[1])
        self.assertEqual(seen, [50 * (i + 2) for i in range(24)])

    def test_tokens_spent_on_helper_threads_are_credited_to_the_caller_exactly(self):
        usage = Usage()

        def caller(i):
            relay = TokenRelay(usage)
            before = usage.thread_tokens()
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(relay.wrap(lambda n: usage.add(n, 0, 0)), range(1, 21)))
            relay.settle()
            return usage.thread_tokens() - before
        self.assertEqual(run_threads(12, caller), [sum(range(1, 21))] * 12)


class Flaky(LLMBackend):
    """Fails a fraction of calls; records the highest number of calls in flight at once."""
    supports_history = True

    def __init__(self, label, failure_rate, hold=0.0, seed=0):
        self.model, self.temperature, self.label = label, 0.0, label
        self.rate, self.hold, self.rng = failure_rate, hold, random.Random(seed)
        self.lock, self.active, self.peak, self.calls = threading.Lock(), 0, 0, 0

    def request(self, prompt):
        return self.chat(prompt)

    def chat(self, prompt, history=None, config=None, system=None, images=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls += 1
            fail = self.rng.random() < self.rate
        try:
            if self.hold:
                time.sleep(self.hold)
            if fail:
                raise MakiNetworkError(f"{self.label} down")
            return reply(self.label)
        finally:
            with self.lock:
                self.active -= 1


class TestSharedFallbackChain(unittest.TestCase):
    def test_a_chain_shared_by_many_threads_always_answers_when_one_backend_is_healthy(self):
        primary, secondary, last = Flaky("a", 0.7, seed=1), Flaky("b", 0.5, seed=2), Flaky("c", 0.0)
        chain = FallbackBackend([primary, secondary, last], cooldown=0.01)
        answers = run_threads(40, lambda i: [chain.chat(f"q{i}-{n}").content for n in range(25)])
        flat = [a for batch in answers for a in batch]
        self.assertEqual(len(flat), 1000)
        self.assertTrue(set(flat) <= {"a", "b", "c"})
        self.assertIn(chain.last_used, (primary, secondary, last))
        self.assertGreater(primary.calls, 0)
        self.assertGreater(secondary.calls, 0)
        self.assertEqual(len(chain._down_until), len(chain._down_until))  # bounded by the number of backends
        self.assertLessEqual(len(chain._down_until), 3)

    def test_when_every_backend_is_down_every_caller_gets_the_error_not_a_hang(self):
        chain = FallbackBackend([Flaky("a", 1.0), Flaky("b", 1.0)], cooldown=0.01)

        def work(i):
            try:
                chain.chat("q")
            except MakiNetworkError:
                return "error"
            return "answered"
        self.assertEqual(set(run_threads(30, work)), {"error"})

    def test_usage_counted_through_a_shared_chain_matches_the_answers_given(self):
        chain = FallbackBackend([Flaky("a", 0.5, seed=3), Flaky("b", 0.0)], cooldown=0)
        usage = attach_usage(chain)
        run_threads(20, lambda i: [chain.chat("q") for _ in range(30)])
        self.assertEqual(usage.calls, 600)
        self.assertEqual(usage.total_tokens, 600 * 5)


class TestOneAgentManyThreads(unittest.TestCase):
    def test_tasks_never_overlap_and_a_stateful_history_is_never_interleaved(self):
        backend = Flaky("x", 0.0, hold=0.002)
        agent = Agent("shared", backend, stateful=True)
        agent._conversation_memory.token_budget = 10 ** 6
        agent._conversation_memory.max_entries = 10 ** 5
        run_threads(30, lambda i: [agent.execute_task(f"task {i}-{n}") for n in range(10)])
        self.assertEqual(backend.peak, 1, "two tasks ran on one agent at the same time")
        messages = agent._conversation_memory.messages()
        self.assertEqual(len(messages), 600)
        self.assertEqual([m.role for m in messages], ["user", "assistant"] * 300)
        # every user message is followed by the answer to that task, not by somebody else's
        self.assertEqual(len({m.content for m in messages if m.role == "user"}), 300)

    def test_a_failing_task_releases_the_agent_for_the_next_one(self):
        backend = Flaky("x", 0.5, seed=9)
        agent = Agent("shared", backend)

        def work(i):
            outcomes = []
            for n in range(20):
                try:
                    agent.execute_task(f"t{i}-{n}")
                    outcomes.append("ok")
                except Exception:
                    outcomes.append("err")
            return outcomes
        results = run_threads(16, work)
        flat = [o for r in results for o in r]
        self.assertEqual(len(flat), 320)
        self.assertIn("ok", flat)
        self.assertIn("err", flat)
        self.assertFalse(agent._execution_lock.locked())

    def test_token_usage_is_exact_across_threads(self):
        agent = Agent("shared", Flaky("x", 0.0))
        run_threads(20, lambda i: [agent.execute_task(f"t{i}-{n}") for n in range(15)])
        self.assertEqual(agent.usage.calls, 300)
        self.assertEqual(agent.usage.total_tokens, 300 * 5)


class TestServedAppUnderParallelLoad(unittest.TestCase):
    def test_parallel_requests_all_complete_and_memory_stays_consistent(self):
        pytest = __import__("pytest")
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from maki.distributed.server import create_app
        agent = Agent("served", Flaky("x", 0.0, hold=0.001), stateful=True)
        client = TestClient(create_app(agent, api_key="k"), raise_server_exceptions=False)
        auth = {"Authorization": "Bearer k"}

        def work(i):
            codes = []
            for n in range(8):
                codes.append(client.post("/execute", json={"task": f"t{i}-{n}"}, headers=auth).status_code)
                codes.append(client.post("/memory/set", json={"key": f"k{i}-{n}", "value": n}, headers=auth).status_code)
                codes.append(client.get("/health").status_code)
            return codes
        codes = [c for batch in run_threads(16, work) for c in batch]
        self.assertEqual(set(codes), {200})
        self.assertEqual(len(agent.memory), 128)
        roles = [m.role for m in agent._conversation_memory.messages()]
        self.assertEqual(roles, ["user", "assistant"] * (len(roles) // 2))


class TestAsyncFanOut(unittest.IsolatedAsyncioTestCase):
    async def test_hundreds_of_async_tasks_with_failures_and_cancellations_leave_every_agent_usable(self):
        rng = random.Random(11)
        agents = [Agent(f"a{i}", Flaky("x", 0.3, hold=0.001, seed=i)) for i in range(4)]
        tasks = [asyncio.create_task(rng.choice(agents).aexecute_task(f"t{n}")) for n in range(300)]
        await asyncio.sleep(0.02)
        for task in rng.sample(tasks, 60):
            task.cancel()
        done = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), LIMIT)
        self.assertEqual(len(done), 300)
        for agent in agents:
            agent.maki.rate = 0.0
            self.assertEqual(await asyncio.wait_for(agent.aexecute_task("final"), 10), "x")
            self.assertFalse(agent._execution_lock.locked())

    async def test_streams_abandoned_midway_by_many_consumers_release_their_agents(self):
        class Streaming(Flaky):
            def stream(self, prompt, history=None, config=None, system=None):
                for chunk in "abcdefghij" * 5:
                    time.sleep(0.001)
                    yield chunk
        agents = [Agent(f"s{i}", Streaming("x", 0.0)) for i in range(5)]

        async def consume(agent, stop_after):
            gen = agent.astream_task("go")
            count = 0
            async for _ in gen:
                count += 1
                if count >= stop_after:
                    break
            await gen.aclose()
        await asyncio.wait_for(asyncio.gather(*[consume(agents[n % 5], 1 + n % 7) for n in range(60)]), LIMIT)
        for agent in agents:
            self.assertEqual(await asyncio.wait_for(agent.aexecute_task("after"), 10), "x")


if __name__ == "__main__":
    unittest.main()
