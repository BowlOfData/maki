"""
Security-focused tests for the Maki framework.

Covers:
  1.  FTP path validation — remote traversal, null bytes, valid paths
  2.  Plugin security — ALLOWED_METHODS whitelist, private methods,
      argument count/length/type limits
  3.  Workflow condition safety — exceptions in conditions are caught
  4.  RateLimiter — construction, token acquisition, MakiLLama integration
  5.  LLM output parsing — _extract_json_array handles messy LLM output
"""

import unittest
from unittest.mock import MagicMock, patch

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
        self.maki = MagicMock()
        self.maki.chat.return_value = _r("mock response")
        self.agent = Agent("SecurityAgent", self.maki, "tester", "Test security")

    def _plain_plugin(self):
        """A minimal mock plugin whitelisting just safe_method."""
        plugin = MagicMock(spec=[])
        plugin.ALLOWED_METHODS = ["safe_method"]
        plugin.safe_method = MagicMock(return_value="ok")
        return plugin

    def _unlisted_plugin(self):
        """A mock plugin with no ALLOWED_METHODS attribute (exposes nothing)."""
        plugin = MagicMock(spec=[])
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

    def test_no_whitelist_blocks_all_methods(self):
        """Fail-closed: a plugin without ALLOWED_METHODS exposes nothing."""
        plugin = self._unlisted_plugin()
        error = self.agent._validate_plugin_call(plugin, "plug", "safe_method", {})
        self.assertIsNotNone(error)
        self.assertIn("no ALLOWED_METHODS", error)
        self.assertIn("plug", error)

    def test_no_whitelist_advertises_no_methods(self):
        """build_plugin_prompt_section lists nothing for an unlisted plugin."""
        plugin = self._unlisted_plugin()
        self.assertEqual(self.agent._allowed_methods(plugin), [])

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
        with patch.object(self.maki, 'chat', return_value=_r("final answer")):
            self.agent.handle_plugin_calls(directive, "task", None)
        plugin._secret.assert_not_called()

    def test_handle_plugin_calls_blocks_whitelisted_violation(self):
        """TOOL: directive for a method not in ALLOWED_METHODS is blocked,
        and the rejection is fed back into the synthesis prompt."""
        plugin = self._whitelisted_plugin(["safe_method"])
        self.agent.plugins["plug"] = plugin

        captured = []

        def fake_chat(prompt, **kwargs):
            captured.append(prompt)
            return _r("final answer")

        directive = 'TOOL: {"plugin": "plug", "method": "other_method", "args": {}}\ntext'
        with patch.object(self.maki, 'chat', side_effect=fake_chat):
            result = self.agent.handle_plugin_calls(directive, "task", None)
        plugin.other_method.assert_not_called()
        self.assertEqual(result, "final answer")
        self.assertEqual(len(captured), 1)
        self.assertIn("not in the allowed methods", captured[0])

    def test_handle_plugin_calls_feeds_unlisted_rejection_to_synthesis(self):
        """A call to a plugin without ALLOWED_METHODS is rejected and the
        error appears in the synthesis prompt rather than being dropped."""
        plugin = self._unlisted_plugin()
        self.agent.plugins["plug"] = plugin

        captured = []

        def fake_chat(prompt, **kwargs):
            captured.append(prompt)
            return _r("final answer")

        directive = 'TOOL: {"plugin": "plug", "method": "safe_method", "args": {}}\ntext'
        with patch.object(self.maki, 'chat', side_effect=fake_chat):
            self.agent.handle_plugin_calls(directive, "task", None)
        plugin.safe_method.assert_not_called()
        self.assertIn("no ALLOWED_METHODS", captured[0])


# ---------------------------------------------------------------------------
# 2b. Dangerous-method gating
# ---------------------------------------------------------------------------
class TestDangerousMethodGating(unittest.TestCase):
    """DANGEROUS_METHODS require Agent(allow_dangerous_tools=True)."""

    def setUp(self):
        self.maki = MagicMock()
        self.maki.chat.return_value = _r("mock response")

    def _writer_plugin(self):
        plugin = MagicMock(spec=[])
        plugin.ALLOWED_METHODS = ["read_thing", "write_thing"]
        plugin.DANGEROUS_METHODS = ["write_thing"]
        plugin.read_thing = MagicMock(return_value="data")
        plugin.write_thing = MagicMock(return_value="written")
        return plugin

    def test_dangerous_method_blocked_by_default(self):
        agent = Agent("A", self.maki, "tester", "t")
        error = agent._validate_plugin_call(self._writer_plugin(), "plug", "write_thing", {})
        self.assertIsNotNone(error)
        self.assertIn("dangerous", error.lower())
        self.assertIn("allow_dangerous_tools", error)

    def test_safe_method_still_allowed_by_default(self):
        agent = Agent("A", self.maki, "tester", "t")
        error = agent._validate_plugin_call(self._writer_plugin(), "plug", "read_thing", {})
        self.assertIsNone(error)

    def test_dangerous_method_allowed_with_opt_in(self):
        agent = Agent("A", self.maki, "tester", "t", allow_dangerous_tools=True)
        error = agent._validate_plugin_call(self._writer_plugin(), "plug", "write_thing", {})
        self.assertIsNone(error)

    def test_invalid_dangerous_methods_type_blocks_all_calls(self):
        agent = Agent("A", self.maki, "tester", "t", allow_dangerous_tools=True)
        plugin = self._writer_plugin()
        plugin.DANGEROUS_METHODS = {}        # invalid collection type
        error = agent._validate_plugin_call(plugin, "plug", "read_thing", {})
        self.assertIsNotNone(error)
        self.assertIn("DANGEROUS_METHODS", error)
        self.assertEqual(agent._allowed_methods(plugin), [])

    def test_prompt_section_hides_dangerous_methods_by_default(self):
        agent = Agent("A", self.maki, "tester", "t")
        methods = agent._allowed_methods(self._writer_plugin())
        self.assertEqual(methods, ["read_thing"])

    def test_prompt_section_shows_dangerous_methods_with_opt_in(self):
        agent = Agent("A", self.maki, "tester", "t", allow_dangerous_tools=True)
        methods = agent._allowed_methods(self._writer_plugin())
        self.assertEqual(methods, ["read_thing", "write_thing"])

    def test_handle_plugin_calls_blocks_dangerous_method(self):
        agent = Agent("A", self.maki, "tester", "t")
        plugin = self._writer_plugin()
        agent.plugins["plug"] = plugin

        directive = 'TOOL: {"plugin": "plug", "method": "write_thing", "args": {}}\ntext'
        with patch.object(self.maki, 'chat', return_value=_r("final")):
            agent.handle_plugin_calls(directive, "task", None)
        plugin.write_thing.assert_not_called()


# ---------------------------------------------------------------------------
# 2c. Built-in plugin whitelist contract
# ---------------------------------------------------------------------------
class TestBuiltinPluginWhitelists(unittest.TestCase):
    """Every registered plugin class must declare a class-level whitelist."""

    def test_every_registered_plugin_declares_allowed_methods(self):
        from maki.plugins import PLUGIN_REGISTRY, get_plugin_class

        for name in PLUGIN_REGISTRY:
            try:
                cls = get_plugin_class(name)
            except ImportError:
                continue   # optional extra not installed
            with self.subTest(plugin=name):
                allowed = cls.__dict__.get("ALLOWED_METHODS") or getattr(
                    cls, "ALLOWED_METHODS", None
                )
                self.assertIsInstance(
                    allowed, (list, set, tuple, frozenset),
                    f"{name} must declare a class-level ALLOWED_METHODS",
                )
                self.assertTrue(allowed, f"{name} whitelist must not be empty")
                for method in allowed:
                    self.assertTrue(
                        callable(getattr(cls, method, None)),
                        f"{name}.{method} is whitelisted but not a method",
                    )
                dangerous = getattr(cls, "DANGEROUS_METHODS", None)
                if dangerous is not None:
                    self.assertTrue(
                        set(dangerous) <= set(allowed),
                        f"{name}: DANGEROUS_METHODS must be a subset of ALLOWED_METHODS",
                    )

    def test_destructive_builtins_are_marked_dangerous(self):
        from maki.plugins.file_writer.file_writer import FileWriter
        from maki.plugins.ftp_client.ftp_client import FTPClient

        self.assertIn("write_file", FileWriter.DANGEROUS_METHODS)
        self.assertIn("append_to_file", FileWriter.DANGEROUS_METHODS)
        self.assertIn("write_file_lines", FileWriter.DANGEROUS_METHODS)
        self.assertIn("remove_directory", FTPClient.DANGEROUS_METHODS)
        self.assertIn("upload_file", FTPClient.DANGEROUS_METHODS)
        self.assertIn("download_file", FTPClient.DANGEROUS_METHODS)

    def test_load_plugin_warns_when_no_allowed_methods(self):
        """Loading a plugin without ALLOWED_METHODS logs a warning naming it."""
        import os
        import tempfile
        import textwrap

        agent = Agent("A", MagicMock(), "tester", "t")
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "bareplugin.py"), "w") as f:
                f.write(textwrap.dedent("""
                    class BarePlugin:
                        def __init__(self, maki=None):
                            self.maki = maki

                        def do_thing(self):
                            return "x"


                    def register_plugin(maki=None):
                        return BarePlugin(maki)
                """))
            with self.assertLogs("maki.agents.plugin_handler", level="WARNING") as cm:
                agent.load_plugin("bareplugin", plugin_path=tmp)
        self.assertTrue(any(
            "bareplugin" in m and "ALLOWED_METHODS" in m for m in cm.output
        ))


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

    def test_makillama_with_rate_limit_creates_limiter(self):
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3", rate_limit=60)
        self.assertIsNotNone(llm._rate_limiter)
        self.assertIsInstance(llm._rate_limiter, RateLimiter)

    def test_makillama_without_rate_limit_has_no_limiter(self):
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3")
        self.assertIsNone(llm._rate_limiter)

    def test_rate_limiter_called_on_chat(self):
        """Rate limiter acquire() is called before each chat request."""
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, '_verify_connection'):
            llm = MakiLLama(model="gemma3", rate_limit=60)
        llm._rate_limiter = MagicMock()

        fake_response = MagicMock()
        fake_response.iter_lines.return_value = iter([
            b'{"message": {"content": "hi"}, "done": true}'
        ])
        fake_response.raise_for_status = MagicMock()

        with patch.object(llm._http._session, 'post', return_value=fake_response):
            try:
                llm.chat("hello")
            except Exception:
                pass  # We only care that acquire was called

        llm._rate_limiter.acquire.assert_called_once()


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
        maki = MagicMock()
        agent = Agent("A", maki, "researcher", "Be helpful")
        fenced = '```json\n[{"description": "Step 1", "resources": "none", "expected_outcome": "done"}]\n```'
        with patch.object(maki, 'chat', return_value=_r(fenced)):
            subtasks = agent.decompose_task("big task")
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0]['description'], 'Step 1')

    def test_decompose_task_handles_preamble_output(self):
        maki = MagicMock()
        agent = Agent("A", maki, "researcher", "Be helpful")
        messy = 'Sure, here are the subtasks:\n[{"description": "Step 1", "resources": "none", "expected_outcome": "done"}]\nLet me know if you need changes!'
        with patch.object(maki, 'chat', return_value=_r(messy)):
            subtasks = agent.decompose_task("big task")
        self.assertEqual(subtasks[0]['description'], 'Step 1')

    def test_decompose_task_still_raises_on_garbage(self):
        maki = MagicMock()
        agent = Agent("A", maki, "researcher", "Be helpful")
        with patch.object(maki, 'chat', return_value=_r("I cannot decompose this task.")):
            with self.assertRaises(ValueError):
                agent.decompose_task("big task")


if __name__ == '__main__':
    unittest.main()
