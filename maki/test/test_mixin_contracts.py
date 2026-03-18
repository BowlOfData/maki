"""
Tests for PluginHandler and ReasoningEngine mixin contract enforcement.

Verifies that:
- Using either mixin without satisfying its protocol raises TypeError early.
- A correctly initialised Agent satisfies both protocols at runtime.
- The Protocol classes can be used for isinstance() checks.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock

from maki.maki import Maki
from maki.backend import LLMBackend
from maki.agents import Agent, PluginHostProtocol, ReasoningHostProtocol
from maki.agents.plugin_handler import PluginHandler
from maki.agents.reasoning import ReasoningEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_maki() -> LLMBackend:
    """Return a mock that satisfies isinstance(x, LLMBackend)."""
    m = MagicMock(spec=LLMBackend)
    return m


# ---------------------------------------------------------------------------
# PluginHandler contract
# ---------------------------------------------------------------------------

class TestPluginHandlerContract(unittest.TestCase):

    def test_missing_maki_raises_type_error(self):
        """_init_plugins must raise if 'maki' is absent."""

        class BadAgent(PluginHandler):
            def __init__(self):
                self.name = "bad"
                # deliberately omit self.maki
                self._init_plugins()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        self.assertIn("maki", str(ctx.exception))

    def test_missing_name_raises_type_error(self):
        """_init_plugins must raise if 'name' is absent."""

        class BadAgent(PluginHandler):
            def __init__(self):
                self.maki = _mock_maki()
                # deliberately omit self.name
                self._init_plugins()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        self.assertIn("name", str(ctx.exception))

    def test_missing_both_raises_type_error(self):
        """_init_plugins must report all missing attributes."""

        class BadAgent(PluginHandler):
            def __init__(self):
                self._init_plugins()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        msg = str(ctx.exception)
        self.assertIn("maki", msg)
        self.assertIn("name", msg)

    def test_correct_init_does_not_raise(self):
        """_init_plugins succeeds when both required attrs are present."""

        class GoodAgent(PluginHandler):
            def __init__(self):
                self.name = "good"
                self.maki = _mock_maki()
                self._init_plugins()

        agent = GoodAgent()
        self.assertEqual(agent.plugins, {})

    def test_error_message_names_the_class(self):
        """TypeError message includes the concrete class name for easy debugging."""

        class SpecificAgent(PluginHandler):
            def __init__(self):
                self._init_plugins()

        with self.assertRaises(TypeError) as ctx:
            SpecificAgent()

        self.assertIn("SpecificAgent", str(ctx.exception))


# ---------------------------------------------------------------------------
# ReasoningEngine contract
# ---------------------------------------------------------------------------

class TestReasoningEngineContract(unittest.TestCase):

    def test_missing_maki_raises_type_error(self):
        """_init_reasoning must raise if 'maki' is absent."""

        class BadAgent(ReasoningEngine):
            def __init__(self):
                self.reasoning_history = deque(maxlen=10)
                # deliberately omit self.maki
                self._init_reasoning()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        self.assertIn("maki", str(ctx.exception))

    def test_missing_reasoning_history_raises_type_error(self):
        """_init_reasoning must raise if 'reasoning_history' is absent."""

        class BadAgent(ReasoningEngine):
            def __init__(self):
                self.maki = _mock_maki()
                # deliberately omit self.reasoning_history
                self._init_reasoning()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        self.assertIn("reasoning_history", str(ctx.exception))

    def test_missing_both_raises_type_error(self):
        """_init_reasoning reports all missing attributes."""

        class BadAgent(ReasoningEngine):
            def __init__(self):
                self._init_reasoning()

        with self.assertRaises(TypeError) as ctx:
            BadAgent()

        msg = str(ctx.exception)
        self.assertIn("maki", msg)
        self.assertIn("reasoning_history", msg)

    def test_correct_init_does_not_raise(self):
        """_init_reasoning succeeds when both required attrs are present."""

        class GoodAgent(ReasoningEngine):
            def __init__(self):
                self.maki = _mock_maki()
                self.reasoning_history = deque(maxlen=10)
                self._init_reasoning()

        # Should not raise
        GoodAgent()

    def test_error_message_names_the_class(self):
        """TypeError message includes the concrete class name."""

        class AnotherAgent(ReasoningEngine):
            def __init__(self):
                self._init_reasoning()

        with self.assertRaises(TypeError) as ctx:
            AnotherAgent()

        self.assertIn("AnotherAgent", str(ctx.exception))


# ---------------------------------------------------------------------------
# Protocol isinstance checks
# ---------------------------------------------------------------------------

class TestProtocolIsInstance(unittest.TestCase):
    """runtime_checkable Protocols should work with isinstance()."""

    def _make_plugin_host(self):
        obj = MagicMock()
        obj.name = "test"
        obj.maki = _mock_maki()
        obj.plugins = {}
        return obj

    def _make_reasoning_host(self):
        obj = MagicMock()
        obj.maki = _mock_maki()
        obj.reasoning_history = deque()
        return obj

    def test_plugin_host_protocol_positive(self):
        obj = self._make_plugin_host()
        self.assertIsInstance(obj, PluginHostProtocol)

    def test_plugin_host_protocol_negative(self):
        obj = MagicMock(spec=[])  # no attributes
        self.assertNotIsInstance(obj, PluginHostProtocol)

    def test_reasoning_host_protocol_positive(self):
        obj = self._make_reasoning_host()
        self.assertIsInstance(obj, ReasoningHostProtocol)

    def test_reasoning_host_protocol_negative(self):
        obj = MagicMock(spec=[])
        self.assertNotIsInstance(obj, ReasoningHostProtocol)


# ---------------------------------------------------------------------------
# Agent integration: both protocols satisfied after __init__
# ---------------------------------------------------------------------------

class TestAgentSatisfiesProtocols(unittest.TestCase):

    def setUp(self):
        self.maki = Maki("localhost", "11434", "llama3", 0.7)

    def test_agent_satisfies_plugin_host_protocol(self):
        agent = Agent("Alice", self.maki, role="tester")
        self.assertIsInstance(agent, PluginHostProtocol)

    def test_agent_satisfies_reasoning_host_protocol(self):
        agent = Agent("Bob", self.maki, role="tester")
        self.assertIsInstance(agent, ReasoningHostProtocol)

    def test_agent_init_order_is_correct(self):
        """
        _init_reasoning and _init_plugins must be called after all required
        attributes are set; Agent.__init__ must enforce this order.
        """
        # If attribute order is wrong this would have raised during Agent.__init__
        agent = Agent("Charlie", self.maki)
        self.assertTrue(hasattr(agent, "plugins"))
        self.assertTrue(hasattr(agent, "reasoning_history"))


# ---------------------------------------------------------------------------
# __init_subclass__ enforcement — the previously-silent failure modes
# ---------------------------------------------------------------------------

class TestInitSubclassEnforcement(unittest.TestCase):
    """
    Verify that __init_subclass__ catches mixin contract violations
    automatically, even when _init_plugins() / _init_reasoning() are never
    called (e.g., because super().__init__() was forgotten).
    """

    # --- PluginHandler ---

    def test_plugin_handler_subclass_missing_attrs_raises_on_instantiation(self):
        """
        A PluginHandler subclass whose __init__ never sets 'name' or 'maki'
        must raise TypeError at instantiation time, not later.
        """
        class Broken(PluginHandler):
            def __init__(self):
                pass  # forgot to set anything; no super() call

        with self.assertRaises(TypeError) as ctx:
            Broken()
        msg = str(ctx.exception)
        self.assertIn("Broken", msg)
        self.assertIn("name", msg)
        self.assertIn("maki", msg)

    def test_plugin_handler_subclass_partial_attrs_raises(self):
        """Setting only 'name' but not 'maki' still raises."""
        class PartialPlugin(PluginHandler):
            def __init__(self):
                self.name = "partial"
                # maki never set

        with self.assertRaises(TypeError) as ctx:
            PartialPlugin()
        self.assertIn("maki", str(ctx.exception))

    def test_plugin_handler_subclass_auto_inits_plugins_dict(self):
        """
        A PluginHandler subclass that sets 'name' and 'maki' but never sets
        'plugins' should have it auto-initialized to {} after construction.
        """
        class MinimalPlugin(PluginHandler):
            def __init__(self):
                self.name = "minimal"
                self.maki = _mock_maki()
                # deliberately skip: self.plugins = {}

        obj = MinimalPlugin()
        self.assertEqual(obj.plugins, {})

    # --- ReasoningEngine ---

    def test_reasoning_engine_subclass_missing_attrs_raises_on_instantiation(self):
        """
        A ReasoningEngine subclass whose __init__ never sets required attrs
        must raise TypeError at instantiation time.
        """
        class Broken(ReasoningEngine):
            def __init__(self):
                pass  # forgot everything

        with self.assertRaises(TypeError) as ctx:
            Broken()
        msg = str(ctx.exception)
        self.assertIn("Broken", msg)
        self.assertIn("maki", msg)
        self.assertIn("reasoning_history", msg)

    def test_reasoning_engine_subclass_partial_attrs_raises(self):
        """Setting only 'maki' but not 'reasoning_history' still raises."""
        from collections import deque
        class PartialReasoning(ReasoningEngine):
            def __init__(self):
                self.maki = _mock_maki()
                # reasoning_history never set

        with self.assertRaises(TypeError) as ctx:
            PartialReasoning()
        self.assertIn("reasoning_history", str(ctx.exception))

    def test_reasoning_engine_subclass_correct_attrs_does_not_raise(self):
        """A ReasoningEngine subclass that sets all attrs without super() is OK."""
        from collections import deque
        class GoodReasoning(ReasoningEngine):
            def __init__(self):
                self.maki = _mock_maki()
                self.reasoning_history = deque(maxlen=10)

        obj = GoodReasoning()  # must not raise
        self.assertIsNotNone(obj)

    # --- Agent subclasses (the primary real-world failure mode) ---

    def test_agent_subclass_without_super_raises_on_instantiation(self):
        """
        The critical case: a subclass of Agent that overrides __init__ without
        calling super() must fail immediately at instantiation, not silently
        produce a broken agent.
        """
        maki = Maki("localhost", "11434", "llama3", 0.7)

        class BrokenAgent(Agent):
            def __init__(self, name):
                # forgot to call super().__init__(name, maki)
                self.name = name
                # self.maki, self.reasoning_history, self.plugins are all absent

        with self.assertRaises(TypeError) as ctx:
            BrokenAgent("alice")
        msg = str(ctx.exception)
        self.assertIn("BrokenAgent", msg)

    def test_agent_subclass_with_super_works_correctly(self):
        """A well-formed Agent subclass must initialize without error."""
        maki = Maki("localhost", "11434", "llama3", 0.7)

        class GoodAgent(Agent):
            def __init__(self, name, backend, extra="default"):
                super().__init__(name, backend)
                self.extra = extra

        agent = GoodAgent("alice", maki, extra="custom")
        self.assertEqual(agent.name, "alice")
        self.assertEqual(agent.extra, "custom")
        self.assertIsInstance(agent.plugins, dict)
        self.assertIsNotNone(agent.reasoning_history)

    def test_error_message_contains_super_hint(self):
        """The error message must tell the developer to call super().__init__()."""
        class Broken(PluginHandler):
            def __init__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            Broken()
        self.assertIn("super().__init__()", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
