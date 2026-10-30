"""Loop token budget."""
import os
import tempfile
import unittest

from maki.agents.loop import LoopSpec, LoopTask, Requirement, StopReason, resume_loop, run_loop

from .loop_helpers import Clock, FakeAgent, verdict

AB = [Requirement("a", "first"), Requirement("b", "second")]
ONE_FAIL = verdict({"a": True, "b": False})
ALL_PASS = verdict({"a": True, "b": True})


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "loop")

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, **kw):
        return LoopSpec("goal", AB, loop_dir=self.dir, loop_id="b1", **kw)

    def agents(self, judge_replies=(ONE_FAIL,), maker_tokens=100, judge_tokens=20):
        return (FakeAgent(["draft"], tokens_per_call=maker_tokens),
                FakeAgent(list(judge_replies), stateful=False, tokens_per_call=judge_tokens))

    def test_validation(self):
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.spec(max_tokens=bad)

    def test_stops_after_the_iteration_that_crosses_the_budget(self):
        maker, judge = self.agents()
        result = run_loop(maker, judge, self.spec(max_tokens=250, stall_limit=10, max_iterations=10), clock=Clock())
        self.assertEqual(result.stop_reason, StopReason.TOKEN_BUDGET)
        self.assertEqual((result.iterations, result.tokens), (3, 360))  # 120 per iteration: 240 < 250 <= 360
        self.assertEqual([e.tokens for e in result.events], [120, 120, 120])

    def test_success_beats_the_budget(self):
        maker, judge = self.agents(judge_replies=(ALL_PASS,))
        result = run_loop(maker, judge, self.spec(max_tokens=10), clock=Clock())
        self.assertEqual(result.stop_reason, StopReason.SUCCEEDED)

    def test_no_budget_means_no_limit_and_agents_without_usage_count_zero(self):
        maker, judge = self.agents(maker_tokens=0, judge_tokens=0)
        result = run_loop(maker, judge, self.spec(max_tokens=1, max_iterations=2, stall_limit=10), clock=Clock())
        self.assertEqual((result.stop_reason, result.tokens), (StopReason.MAX_ITERATIONS, 0))

    def test_resume_keeps_counting_and_a_spent_budget_stops_immediately(self):
        maker, judge = self.agents()
        first = run_loop(maker, judge, self.spec(max_iterations=1), clock=Clock())
        self.assertEqual((first.stop_reason, first.tokens), (StopReason.MAX_ITERATIONS, 120))
        maker2, judge2 = self.agents()
        again = resume_loop(maker2, judge2, self.spec(max_iterations=5, max_tokens=100, stall_limit=10), clock=Clock())
        self.assertEqual((again.stop_reason, again.iterations, again.tokens), (StopReason.TOKEN_BUDGET, 1, 120))
        self.assertEqual(maker2.prompts, [])
        third_maker, third_judge = self.agents()
        more = resume_loop(third_maker, third_judge, self.spec(max_iterations=5, max_tokens=200, stall_limit=10),
                           clock=Clock())
        self.assertEqual((more.stop_reason, more.iterations, more.tokens), (StopReason.TOKEN_BUDGET, 2, 240))

    def test_real_agents_report_their_backends_tokens_to_the_loop(self):
        from maki.agents.agent import Agent
        from maki.objects import BackendType, LLMResponse

        class Backend:
            supports_native_tools = False

            def __init__(self, reply):
                self.reply = reply

            def request(self, prompt):
                return self.chat(prompt)

            def chat(self, prompt, system=None, **kw):
                return LLMResponse(self.reply, "m", 10, 5, 15, 0.1, backend=BackendType.OLLAMA)

        maker = Agent("maker", Backend("draft"), stateful=True)
        judge = Agent("judge", Backend(ONE_FAIL))
        result = run_loop(maker, judge, self.spec(max_tokens=40, stall_limit=9, max_iterations=5), clock=Clock())
        self.assertEqual((result.stop_reason, result.iterations, result.tokens), (StopReason.TOKEN_BUDGET, 2, 60))

    def test_loop_task_passes_the_budget(self):
        task = LoopTask(judge="j", requirements=["x"], max_tokens=500)
        self.assertEqual(task.to_spec("w/t", "goal").max_tokens, 500)


if __name__ == "__main__":
    unittest.main()
