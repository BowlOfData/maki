"""The memory benchmark itself: pins today's behaviour so later steps are measured against it."""
import unittest

from maki.test import memory_bench as mb


class TestBaseline(unittest.TestCase):
    def test_stateful_prompt_grows_then_plateaus_at_the_token_budget(self):
        tokens = mb.run_stateful(turns=20)
        self.assertGreater(tokens[5], tokens[0])
        self.assertLessEqual(max(tokens), 4096 + 200)  # budget plus the task and system text
        self.assertAlmostEqual(tokens[-1], tokens[-2], delta=60)

    def test_default_eviction_loses_a_turn_one_fact_by_turn_fifteen(self):
        self.assertFalse(mb.recall_at(15))

    def test_the_fact_is_visible_while_it_still_fits(self):
        self.assertTrue(mb.recall_at(4))

    def test_tool_results_accumulate_without_a_cap(self):
        rounds = mb.run_tool_heavy(rounds=4, result_chars=20000)
        self.assertGreaterEqual(rounds[-1] - rounds[0], 4 * 20000 // 4)

    def test_task_history_keeps_every_answer_in_full(self):
        self.assertEqual(mb.history_footprint(turns=10, answer_chars=1000), 10 * (1000 + len("None")))

    def test_run_all_reports_every_metric(self):
        self.assertEqual(set(mb.run_all()), {
            "stateful_prompt_tokens_total", "stateful_prompt_tokens_last_turn", "recall_turn_15",
            "tool_heavy_tokens_last_round", "task_history_chars"})


if __name__ == "__main__":
    unittest.main()
