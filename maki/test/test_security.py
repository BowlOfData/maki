"""
Security-focused tests for the Maki framework.

Covers:
  1.  FTP path validation — remote traversal, null bytes, valid paths
  2.  Plugin security — ALLOWED_METHODS whitelist, private methods,
      argument count/length/type limits
  3.  Workflow condition safety — exceptions in conditions are caught
  4.  RateLimiter — construction, token acquisition, Maki integration
  5.  LLM output parsing — _extract_json_array handles messy LLM output
"""

import unittest
from unittest.mock import MagicMock, patch

from maki.maki import Maki
from maki.objects import LLMResponse, RateLimiter


def _r(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="test", prompt_tokens=0,
                       completion_tokens=0, total_tokens=0, elapsed_seconds=0.0)
from maki.agents import Agent, WorkflowTask
from maki.agents.reasoning import _extract_json_array
from maki.plugins.ftp_client.ftp_client import FTPClient


# ---------------------------------------------------------------------------
# 1. FTP path validation
# ---------------------------------------------------------------------------
class TestFTPRemotePathValidation(unittest.TestCase):
    """Tests for _validate_path with is_local=False (remote FTP paths)."""

    def setUp(self):
        self.client = FTPClient()

    # --- traversal ---

    def test_dotdot_relative_blocked(self):
        result = self.client._validate_path('../../etc/passwd', is_local=False)
        self.assertFalse(result['valid'])
        self.assertIn('traversal', result['error'].lower())

    def test_dotdot_in_middle_blocked(self):
        result = self.client._validate_path('files/../../../etc/passwd', is_local=False)
        self.assertFalse(result['valid'])

    def test_dotdot_trailing_resolves_to_valid(self):
        # posixpath.normpath('uploads/dir/..') == 'uploads' — no '..' remains
        result = self.client._validate_path('uploads/dir/..', is_local=False)
        self.assertTrue(result['valid'])
        self.assertEqual(result['normalized_path'], 'uploads')

    def test_single_dotdot_blocked(self):
        result = self.client._validate_path('..', is_local=False)
        self.assertFalse(result['valid'])

    # --- null bytes ---

    def test_null_byte_remote_blocked(self):
        result = self.client._validate_path('file\x00.txt', is_local=False)
        self.assertFalse(result['valid'])
        self.assertIn('invalid characters', result['error'].lower())

    def test_null_byte_local_blocked(self):
        result = self.client._validate_path('file\x00.txt', is_local=True)
        self.assertFalse(result['valid'])

    # --- valid remote paths ---

    def test_valid_absolute_remote_path(self):
        # Absolute remote paths are treated as path injection (CVE-2.1 / CVE-2.2)
        result = self.client._validate_path('/var/ftp/files/report.txt', is_local=False)
        self.assertFalse(result['valid'])

    def test_valid_relative_remote_path(self):
        result = self.client._validate_path('uploads/document.pdf', is_local=False)
        self.assertTrue(result['valid'])

    def test_valid_remote_root(self):
        # '/' is an absolute path and blocked as path injection
        result = self.client._validate_path('/', is_local=False)
        self.assertFalse(result['valid'])

    def test_valid_remote_current_dir(self):
        result = self.client._validate_path('.', is_local=False)
        self.assertTrue(result['valid'])

    def test_backslash_not_traversal_on_posix_ftp(self):
        """On POSIX FTP servers backslash is a filename character, not a separator."""
        result = self.client._validate_path('..\\..\\etc\\passwd', is_local=False)
        self.assertTrue(result['valid'])

    # --- invalid inputs ---

    def test_none_path_invalid(self):
        result = self.client._validate_path(None, is_local=False)
        self.assertFalse(result['valid'])

    def test_empty_path_invalid(self):
        result = self.client._validate_path('', is_local=False)
        self.assertFalse(result['valid'])

    def test_whitespace_only_invalid(self):
        result = self.client._validate_path('   ', is_local=False)
        self.assertFalse(result['valid'])


class TestFTPLocalPathValidation(unittest.TestCase):
    """Tests for _validate_path with is_local=True (local filesystem paths)."""

    def setUp(self):
        self.client = FTPClient()

    def test_local_traversal_blocked(self):
        result = self.client._validate_path('../secret.txt', is_local=True)
        self.assertFalse(result['valid'])

    def test_local_absolute_path_blocked(self):
        result = self.client._validate_path('/etc/passwd', is_local=True)
        self.assertFalse(result['valid'])

    def test_local_valid_relative_path(self):
        result = self.client._validate_path('uploads/file.txt', is_local=True)
        self.assertTrue(result['valid'])

    def test_local_valid_simple_filename(self):
        result = self.client._validate_path('report.pdf', is_local=True)
        self.assertTrue(result['valid'])


# ---------------------------------------------------------------------------
# 2. Plugin security
# ---------------------------------------------------------------------------
class TestPluginSecurity(unittest.TestCase):
    """Tests for PluginHandler._validate_plugin_call security enforcement."""

    def setUp(self):
        self.maki = Maki("localhost", "11434", "llama3", 0.7)
        self.agent = Agent("SecurityAgent", self.maki, "tester", "Test security")

    def _plain_plugin(self):
        """A mock plugin with no ALLOWED_METHODS whitelist."""
        plugin = MagicMock(spec=[])          # no ALLOWED_METHODS attribute
        plugin.safe_method = MagicMock(return_value="ok")
        return plugin

    def _whitelisted_plugin(self, allowed):
        """A mock plugin with an explicit ALLOWED_METHODS list."""
        plugin = MagicMock(spec=[])
        plugin.ALLOWED_METHODS = allowed
        plugin.safe_method = MagicMock(return_value="ok")
        plugin.other_method = MagicMock(return_value="other")
        return plugin

    # --- method name filtering ---

    def test_private_method_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(plugin, "plug", "_private", {})
        self.assertIsNotNone(error)
        self.assertIn("not callable", error.lower())

    def test_dunder_method_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(plugin, "plug", "__init__", {})
        self.assertIsNotNone(error)

    def test_dunder_call_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(plugin, "plug", "__call__", {})
        self.assertIsNotNone(error)

    # --- ALLOWED_METHODS whitelist ---

    def test_whitelist_blocks_unlisted_method(self):
        plugin = self._whitelisted_plugin(["safe_method"])
        error = self.agent._validate_plugin_call(plugin, "plug", "other_method", {})
        self.assertIsNotNone(error)
        self.assertIn("not in the allowed", error.lower())

    def test_whitelist_permits_listed_method(self):
        plugin = self._whitelisted_plugin(["safe_method"])
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", {})
        self.assertIsNone(error)

    def test_no_whitelist_does_not_block_public_methods(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", {})
        self.assertIsNone(error)

    def test_whitelist_as_set(self):
        plugin = self._plain_plugin()
        plugin.ALLOWED_METHODS = {"safe_method"}
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", {})
        self.assertIsNone(error)

    def test_whitelist_as_frozenset(self):
        plugin = self._plain_plugin()
        plugin.ALLOWED_METHODS = frozenset(["safe_method"])
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", {})
        self.assertIsNone(error)

    # --- argument count ---

    def test_too_many_arguments_blocked(self):
        plugin = self._plain_plugin()
        args = {f"arg{i}": "value" for i in range(25)}   # > _MAX_ARGS_COUNT (20)
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", args)
        self.assertIsNotNone(error)
        self.assertIn("too many arguments", error.lower())

    def test_exactly_max_arguments_allowed(self):
        plugin = self._plain_plugin()
        args = {f"arg{i}": "value" for i in range(20)}   # == _MAX_ARGS_COUNT
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", args)
        self.assertIsNone(error)

    # --- argument string length ---

    def test_argument_string_too_long_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(
            plugin, "plug", "safe_method", {"data": "x" * 10_001}
        )
        self.assertIsNotNone(error)
        self.assertIn("exceeds the maximum", error.lower())

    def test_argument_string_at_limit_allowed(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(
            plugin, "plug", "safe_method", {"data": "x" * 10_000}
        )
        self.assertIsNone(error)

    # --- argument types ---

    def test_bytes_argument_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(
            plugin, "plug", "safe_method", {"data": b"bytes"}
        )
        self.assertIsNotNone(error)
        self.assertIn("unsupported type", error.lower())

    def test_object_argument_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(
            plugin, "plug", "safe_method", {"obj": object()}
        )
        self.assertIsNotNone(error)

    def test_all_safe_types_allowed(self):
        plugin = self._plain_plugin()
        args = {
            "s": "hello",
            "i": 42,
            "f": 3.14,
            "b": True,
            "l": [1, 2, 3],
            "d": {"key": "val"},
            "n": None,
        }
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", args)
        self.assertIsNone(error)

    def test_non_dict_args_blocked(self):
        plugin = self._plain_plugin()
        error = self.agent._validate_plugin_call(
            plugin, "plug", "safe_method", ["not", "a", "dict"]
        )
        self.assertIsNotNone(error)

    # --- integration: handle_plugin_calls blocks unsafe methods ---

    def test_handle_plugin_calls_blocks_private_method(self):
        """TOOL: directive targeting a private method is silently blocked."""
        plugin = self._plain_plugin()
        plugin._secret = MagicMock(return_value="leaked")
        self.agent.plugins["plug"] = plugin

        directive = 'TOOL: {"plugin": "plug", "method": "_secret", "args": {}}\nsome text'
        with patch.object(self.maki, 'request', return_value=_r("final answer")):
            self.agent.handle_plugin_calls(directive, "task", None)
        plugin._secret.assert_not_called()

    def test_handle_plugin_calls_blocks_whitelisted_violation(self):
        """TOOL: directive for a method not in ALLOWED_METHODS is blocked."""
        plugin = self._whitelisted_plugin(["safe_method"])
        self.agent.plugins["plug"] = plugin

        directive = 'TOOL: {"plugin": "plug", "method": "other_method", "args": {}}\ntext'
        with patch.object(self.maki, 'request', return_value=_r("final answer")):
            self.agent.handle_plugin_calls(directive, "task", None)
        plugin.other_method.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Workflow condition safety
# ---------------------------------------------------------------------------
class TestWorkflowConditionSafety(unittest.TestCase):
    """Tests for WorkflowTask.should_execute robustness after our fix."""

    def test_condition_exception_returns_false_not_raise(self):
        """A condition that raises must not propagate; task is skipped."""
        def bad_condition(ctx):
            raise RuntimeError("condition exploded")

        task = WorkflowTask("t", "A", "task", conditions=[bad_condition])
        self.assertFalse(task.should_execute())

    def test_condition_exception_with_context_returns_false(self):
        def bad_condition(ctx):
            raise ValueError(f"bad value: {ctx}")

        task = WorkflowTask("t", "A", "task", conditions=[bad_condition])
        self.assertFalse(task.should_execute({"key": "val"}))

    def test_condition_returns_false_task_skipped(self):
        task = WorkflowTask("t", "A", "task", conditions=[lambda ctx: False])
        self.assertFalse(task.should_execute())

    def test_condition_returns_true_task_runs(self):
        task = WorkflowTask("t", "A", "task", conditions=[lambda ctx: True])
        self.assertTrue(task.should_execute())

    def test_no_conditions_task_runs(self):
        task = WorkflowTask("t", "A", "task")
        self.assertTrue(task.should_execute())

    def test_first_bad_condition_skips_remaining(self):
        """Once a condition raises (treated as False), remaining conditions are not evaluated."""
        second_called = []

        def bad_first(ctx):
            raise RuntimeError("first failed")

        def second(ctx):
            second_called.append(True)
            return True

        task = WorkflowTask("t", "A", "task", conditions=[bad_first, second])
        result = task.should_execute()
        self.assertFalse(result)
        self.assertEqual(second_called, [])   # short-circuit: second was never called

    def test_all_conditions_true_executes(self):
        task = WorkflowTask("t", "A", "task",
                            conditions=[lambda ctx: True, lambda ctx: True])
        self.assertTrue(task.should_execute())

    def test_mixed_conditions_one_false_skips(self):
        task = WorkflowTask("t", "A", "task",
                            conditions=[lambda ctx: True, lambda ctx: False])
        self.assertFalse(task.should_execute())


# ---------------------------------------------------------------------------
# 4. RateLimiter
# ---------------------------------------------------------------------------
class TestRateLimiter(unittest.TestCase):

    def test_zero_rpm_raises(self):
        with self.assertRaises(ValueError):
            RateLimiter(0)

    def test_negative_rpm_raises(self):
        with self.assertRaises(ValueError):
            RateLimiter(-10)

    def test_non_integer_rpm_raises(self):
        with self.assertRaises(ValueError):
            RateLimiter("sixty")

    def test_float_rpm_raises(self):
        with self.assertRaises(ValueError):
            RateLimiter(60.0)

    def test_valid_rpm_constructs(self):
        rl = RateLimiter(60)
        self.assertIsNotNone(rl)

    def test_acquire_succeeds_immediately_when_full(self):
        rl = RateLimiter(600)   # large bucket fills quickly
        rl.acquire()            # should not block

    def test_acquire_multiple_within_capacity(self):
        """A fresh limiter starts full, so burst up to capacity is instant."""
        rl = RateLimiter(10)
        for _ in range(10):
            rl.acquire()        # all 10 should be immediate

    def test_maki_with_rate_limit_creates_limiter(self):
        maki = Maki("localhost", "11434", "llama3", 0.7, rate_limit=60)
        self.assertIsNotNone(maki._rate_limiter)
        self.assertIsInstance(maki._rate_limiter, RateLimiter)

    def test_maki_without_rate_limit_has_no_limiter(self):
        maki = Maki("localhost", "11434", "llama3", 0.7)
        self.assertIsNone(maki._rate_limiter)

    def test_rate_limiter_called_on_request(self):
        """Rate limiter acquire() is called before each request."""
        maki = Maki("localhost", "11434", "llama3", 0.7, rate_limit=60)
        maki._rate_limiter = MagicMock()

        with patch('maki.connector.Connector.simple', return_value={"response": "response"}):
            maki.request("hello")

        maki._rate_limiter.acquire.assert_called_once()


# ---------------------------------------------------------------------------
# 5. LLM output parsing (_extract_json_array)
# ---------------------------------------------------------------------------
class TestExtractJsonArray(unittest.TestCase):

    def test_clean_json_array(self):
        raw = '[{"description": "Step 1"}]'
        self.assertEqual(_extract_json_array(raw), raw)

    def test_json_fence(self):
        raw = '```json\n[{"description": "Step 1"}]\n```'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_plain_fence(self):
        raw = '```\n[{"description": "Step 1"}]\n```'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_preamble_text_before_array(self):
        raw = 'Sure! Here is the decomposition:\n[{"description": "Step 1"}]'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_trailing_text_after_array(self):
        raw = '[{"description": "Step 1"}]\nLet me know if you need more detail!'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_preamble_and_trailing_text(self):
        raw = 'Here you go:\n[{"description": "Step 1"}]\nHope this helps!'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_fence_with_preamble(self):
        raw = 'Here is the JSON:\n```json\n[{"description": "Step 1"}]\n```'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_uppercase_json_fence(self):
        raw = '```JSON\n[{"description": "Step 1"}]\n```'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[{"description": "Step 1"}]')

    def test_multiline_json_array(self):
        raw = '```json\n[\n  {"description": "Step 1"},\n  {"description": "Step 2"}\n]\n```'
        result = _extract_json_array(raw)
        import json
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)

    def test_no_array_returns_cleaned_text(self):
        raw = 'This is plain text with no JSON array.'
        result = _extract_json_array(raw)
        self.assertNotIn('```', result)

    def test_empty_array(self):
        raw = '[]'
        result = _extract_json_array(raw)
        self.assertEqual(result, '[]')

    def test_nested_objects_in_array(self):
        raw = '[{"description": "Step 1", "resources": {"tool": "python"}}]'
        result = _extract_json_array(raw)
        import json
        parsed = json.loads(result)
        self.assertEqual(parsed[0]['resources']['tool'], 'python')

    # integration: decompose_task handles messy LLM output
    def test_decompose_task_handles_fenced_output(self):
        maki = Maki("localhost", "11434", "llama3", 0.7)
        agent = Agent("A", maki, "researcher", "Be helpful")
        fenced = '```json\n[{"description": "Step 1", "resources": "none", "expected_outcome": "done"}]\n```'
        with patch.object(maki, 'request', return_value=_r(fenced)):
            subtasks = agent.decompose_task("big task")
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0]['description'], 'Step 1')

    def test_decompose_task_handles_preamble_output(self):
        maki = Maki("localhost", "11434", "llama3", 0.7)
        agent = Agent("A", maki, "researcher", "Be helpful")
        messy = 'Sure, here are the subtasks:\n[{"description": "Step 1", "resources": "none", "expected_outcome": "done"}]\nLet me know if you need changes!'
        with patch.object(maki, 'request', return_value=_r(messy)):
            subtasks = agent.decompose_task("big task")
        self.assertEqual(subtasks[0]['description'], 'Step 1')

    def test_decompose_task_still_raises_on_garbage(self):
        maki = Maki("localhost", "11434", "llama3", 0.7)
        agent = Agent("A", maki, "researcher", "Be helpful")
        with patch.object(maki, 'request', return_value=_r("I cannot decompose this task.")):
            with self.assertRaises(ValueError):
                agent.decompose_task("big task")


if __name__ == '__main__':
    unittest.main()
