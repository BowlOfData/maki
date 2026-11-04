"""Typed messages: the Message model, the agent's provider-neutral tool loop, memory and fallback."""
import json
import unittest

from maki import Agent, FallbackBackend, MemoryPolicy, ToolPolicy
from maki.backend import LLMBackend
from maki.callbacks import AgentCallback
from maki.exceptions import MakiNetworkError
from maki.objects import ConversationMemory, LLMResponse, Message, ToolCall, expand_tool_traces


def reply(text="", calls=None):
    return LLMResponse(content=text, model="m", prompt_tokens=3, completion_tokens=2, total_tokens=5,
                       elapsed_seconds=0.0, tool_calls=calls)


class TypedFake(LLMBackend):
    """A typed backend that answers from a script and records the messages of every round."""
    supports_native_tools = True
    supports_typed_messages = True
    supports_history = True

    def __init__(self, label="t", script=None):
        self.model, self.temperature, self.label = label, 0.0, label
        self.script = list(script or [])
        self.rounds = []       # (messages as sent, tool specs)
        self.choices = []      # tool_choice of each round
        self.legacy_calls = 0
        self.fail = None

    def request(self, prompt):
        return reply(self.label)

    def to_tool_schemas(self, specs):
        return specs

    def chat_with_tools(self, *a, **k):
        self.legacy_calls += 1
        raise AssertionError("the typed loop must not use chat_with_tools")

    def chat_turn(self, messages, tool_specs=None, *, system=None, config=None, tool_choice=None):
        self.choices.append(tool_choice)
        self.rounds.append(([Message.from_dict(m.to_dict()) for m in expand_tool_traces(messages)], tool_specs))
        if self.fail is not None and any(m.role == "tool" for m in messages):
            raise self.fail
        return self.script.pop(0) if self.script else reply(f"{self.label}-text")


class Echo:
    ALLOWED_METHODS = ["say"]
    DANGEROUS_METHODS = None

    def __init__(self):
        self.runs = []

    def say(self, text: str) -> str:
        """Echo.

        Args:
            text: what to echo
        """
        self.runs.append(text)
        return f"echo:{text}"


def agent_with(backend, **kw):
    agent = Agent("t", backend, **kw)
    agent.plugins["echo"] = Echo()
    return agent


def call(text="hi", cid="c1"):
    return ToolCall(id=cid, name="echo__say", args={"text": text})


class TestMessageModel(unittest.TestCase):
    def test_plain_messages_serialise_exactly_as_before(self):
        self.assertEqual(Message("user", "x").to_dict(), {"role": "user", "content": "x"})
        self.assertEqual(Message("user", "x", ["img"]).to_dict(), {"role": "user", "content": "x", "images": ["img"]})
        self.assertEqual(Message.from_dict({"role": "user", "content": "x"}), Message("user", "x"))

    def test_tool_fields_round_trip(self):
        trace = [Message("assistant", "", tool_calls=[call()]), Message("tool", "r", tool_call_id="c1", name="echo__say")]
        original = Message("assistant", "answer", tool_trace=trace)
        restored = Message.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)
        self.assertEqual(restored.tool_trace[0].tool_calls[0].args, {"text": "hi"})

    def test_size_counts_the_trace(self):
        plain = Message("assistant", "abcd")
        traced = Message("assistant", "abcd", tool_trace=[Message("assistant", "", tool_calls=[call()]),
                                                          Message("tool", "x" * 100, tool_call_id="c1")])
        self.assertEqual(plain.size(), 4)
        self.assertGreater(traced.size(), 104)

    def test_expand_puts_the_trace_before_its_answer_and_strips_it(self):
        trace = [Message("assistant", "", tool_calls=[call()]), Message("tool", "r", tool_call_id="c1")]
        out = expand_tool_traces([Message("user", "q"), Message("assistant", "a", tool_trace=trace)])
        self.assertEqual([m.role for m in out], ["user", "assistant", "tool", "assistant"])
        self.assertIsNone(out[-1].tool_trace)
        self.assertEqual(out[-1].content, "a")

    def test_response_serialises_tool_calls_only_when_present(self):
        self.assertNotIn("tool_calls", reply("x").to_dict())
        data = reply("", [call()]).to_dict()
        self.assertEqual(LLMResponse.from_dict(data).tool_calls, [call()])


class TestTypedLoop(unittest.TestCase):
    def test_a_tool_round_then_an_answer(self):
        backend = TypedFake(script=[reply("let me check", [call()]), reply("done")])
        agent = agent_with(backend)
        self.assertEqual(agent.execute_task("say hi", use_plugins=True), "done")
        self.assertEqual(agent.plugins["echo"].runs, ["hi"])
        self.assertEqual(backend.legacy_calls, 0)
        first, second = backend.rounds
        self.assertEqual([m.role for m in first[0]], ["user"])
        self.assertEqual([m.role for m in second[0]], ["user", "assistant", "tool"])
        assistant, tool = second[0][1], second[0][2]
        self.assertEqual((assistant.content, assistant.tool_calls[0].name), ("let me check", "echo__say"))
        self.assertEqual((tool.content, tool.tool_call_id, tool.name), ("echo:hi", "c1", "echo__say"))
        self.assertEqual(second[1][0]["name"], "echo__say")  # neutral specs, converted by the backend

    def test_several_rounds_and_parallel_calls_keep_order(self):
        backend = TypedFake(script=[reply("", [call("a", "1"), call("b", "2")]), reply("", [call("c", "3")]), reply("fin")])
        agent = agent_with(backend, tool_policy=ToolPolicy(parallel=True))
        self.assertEqual(agent.execute_task("go", use_plugins=True), "fin")
        last = backend.rounds[-1][0]
        self.assertEqual([m.role for m in last], ["user", "assistant", "tool", "tool", "assistant", "tool"])
        self.assertEqual([m.content for m in last if m.role == "tool"], ["echo:a", "echo:b", "echo:c"])

    def test_max_rounds_forces_a_text_answer_without_tools(self):
        backend = TypedFake(script=[reply("", [call("a", "1")]), reply("", [call("b", "2")]), reply("forced")])
        agent = agent_with(backend, tool_policy=ToolPolicy(max_rounds=2))
        self.assertEqual(agent.execute_task("go", use_plugins=True), "forced")
        self.assertEqual(backend.choices, [None, None, "none"])  # the last round lists tools but forbids them
        self.assertTrue(backend.rounds[-1][1])
        self.assertEqual(len(agent.plugins["echo"].runs), 2)

    def test_the_closing_round_carries_an_instruction_that_is_not_remembered(self):
        backend = TypedFake(script=[reply("", [call("a", "1")]), reply("closing answer")])
        agent = agent_with(backend, tool_policy=ToolPolicy(max_rounds=1), stateful=True,
                           memory=MemoryPolicy(history_mode="messages", tool_history=True))
        self.assertEqual(agent.execute_task("go", use_plugins=True), "closing answer")
        last = backend.rounds[-1][0]
        self.assertEqual((last[-1].role, "tool calls" in last[-1].content), ("user", True))
        self.assertEqual([m.role for m in last[:-1]], ["user", "assistant", "tool"])
        stored = agent._conversation_memory.messages()[1].tool_trace
        self.assertEqual([m.role for m in stored], ["assistant", "tool"])  # the instruction is not history

    def test_callbacks_see_typed_messages_and_the_real_response(self):
        seen = {"start": [], "end": []}

        class Spy(AgentCallback):
            def on_model_start(self, agent, messages):
                seen["start"].append([type(m).__name__ for m in messages])

            def on_model_end(self, agent, response, tool_calls):
                seen["end"].append((response.total_tokens, bool(tool_calls)))
        backend = TypedFake(script=[reply("", [call()]), reply("ok")])
        agent_with(backend, callbacks=[Spy()]).execute_task("x", use_plugins=True)
        self.assertEqual(seen["start"][0], ["Message"])
        self.assertEqual(seen["end"], [(5, True), (5, False)])

    def test_a_tool_error_reaches_the_model_as_a_tool_message(self):
        backend = TypedFake(script=[reply("", [ToolCall("e1", "echo__nope", {})]), reply("recovered")])
        self.assertEqual(agent_with(backend).execute_task("x", use_plugins=True), "recovered")
        tool = backend.rounds[1][0][-1]
        self.assertEqual((tool.role, tool.tool_call_id), ("tool", "e1"))
        self.assertTrue(tool.content)

    def test_token_usage_includes_tool_rounds(self):
        backend = TypedFake(script=[reply("", [call()]), reply("ok")])
        agent = agent_with(backend)
        agent.execute_task("x", use_plugins=True)
        self.assertEqual(agent.usage.calls, 2)
        self.assertEqual(agent.usage.total_tokens, 10)

    def test_untyped_native_backends_keep_the_provider_loop(self):
        class Legacy(TypedFake):
            supports_typed_messages = False

            def chat_with_tools(self, messages, tools, *, system=None, config=None):
                if not any(m.get("role") == "tool" for m in messages):
                    return None, [call()], list(messages)
                return reply("legacy answer"), None, list(messages)

            def append_tool_results(self, messages, results):
                return list(messages) + [{"role": "tool", "content": r} for _, r in results]
        agent = agent_with(Legacy())
        self.assertEqual(agent.execute_task("x", use_plugins=True), "legacy answer")


class TestToolHistory(unittest.TestCase):
    POLICY = dict(history_mode="messages", tool_history=True, tool_result_chars=30)

    def _agent(self, script, **policy):
        backend = TypedFake(script=script)
        agent = agent_with(backend, stateful=True, memory=MemoryPolicy(**{**self.POLICY, **policy}))
        return backend, agent

    def test_the_tool_loop_is_remembered_clipped_and_replayed(self):
        backend, agent = self._agent([reply("", [call("x" * 200)]), reply("first answer"), reply("second answer")])
        agent.execute_task("first", use_plugins=True)
        stored = agent._conversation_memory.messages()[1]
        self.assertEqual(stored.content, "first answer")
        self.assertEqual([m.role for m in stored.tool_trace], ["assistant", "tool"])
        self.assertIn("characters omitted", stored.tool_trace[1].content)  # clipped to tool_result_chars
        agent.execute_task("second", use_plugins=True)
        replayed = backend.rounds[-1][0]
        self.assertEqual([m.role for m in replayed], ["user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(replayed[3].content, "first answer")
        self.assertIsNone(replayed[3].tool_trace)

    def test_off_by_default(self):
        backend = TypedFake(script=[reply("", [call()]), reply("a"), reply("b")])
        agent = agent_with(backend, stateful=True, memory=MemoryPolicy(history_mode="messages"))
        agent.execute_task("one", use_plugins=True)
        self.assertIsNone(agent._conversation_memory.messages()[1].tool_trace)
        agent.execute_task("two", use_plugins=True)
        self.assertEqual([m.role for m in backend.rounds[-1][0]], ["user", "assistant", "user"])

    def test_long_tool_arguments_are_clipped_in_the_stored_trace(self):
        backend, agent = self._agent([reply("", [call("y" * 500)]), reply("answer")])
        agent.execute_task("q", use_plugins=True)
        stored_call = agent._conversation_memory.messages()[1].tool_trace[0].tool_calls[0]
        self.assertIn("characters omitted", stored_call.args["text"])
        self.assertEqual((stored_call.id, stored_call.name), ("c1", "echo__say"))

    def test_a_task_without_tools_leaves_no_trace_and_no_stale_one(self):
        backend, agent = self._agent([reply("", [call()]), reply("a1"), reply("a2")])
        agent.execute_task("with tools", use_plugins=True)
        agent.execute_task("no plugins", use_plugins=False)  # plain chat path
        self.assertIsNone(agent._conversation_memory.messages()[3].tool_trace)

    def test_a_trace_from_a_task_that_was_not_recorded_never_attaches_to_a_later_one(self):
        backend = TypedFake(script=[reply("", [call()]), reply("a1"), reply("a2")])
        agent = agent_with(backend, memory=MemoryPolicy(**self.POLICY))  # not stateful: nothing is recorded
        agent.execute_task("with tools", use_plugins=True)
        agent.stateful = True
        agent.execute_task("plain", use_plugins=False)
        self.assertIsNone(agent._conversation_memory.messages()[1].tool_trace)

    def test_older_turns_lose_their_trace(self):
        backend, agent = self._agent([reply("", [call()]), reply("a1"), reply("a2"), reply("a3")],
                                     recent_turns=1, older_turn_chars=50)
        agent.execute_task("one", use_plugins=True)
        agent.execute_task("two", use_plugins=True)
        agent.execute_task("three", use_plugins=True)
        roles = [m.role for m in backend.rounds[-1][0]]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant", "user"])  # no tool messages left

    def test_the_trace_counts_against_the_token_budget(self):
        mem = ConversationMemory(token_budget=200)
        big = [Message("assistant", "", tool_calls=[call()]), Message("tool", "z" * 2000, tool_call_id="c1")]
        for i in range(3):
            mem.append(Message("user", "q"))
            mem.append(Message("assistant", "a", tool_trace=big))
        self.assertEqual(len(mem), 2)  # the oldest turns were evicted; a bare 'a' would have fit them all

    def test_snapshot_round_trip_keeps_the_trace(self):
        backend, agent = self._agent([reply("", [call()]), reply("answer")])
        agent.execute_task("q", use_plugins=True)
        restored = Agent.from_dict(json.loads(json.dumps(agent.to_dict())), TypedFake())
        stored = restored._conversation_memory.messages()[1]
        self.assertEqual(stored.tool_trace[0].tool_calls[0].name, "echo__say")
        self.assertTrue(restored.memory_policy.tool_history)

    def test_the_summariser_hears_about_the_tools(self):
        backend, agent = self._agent([])
        prompts = []
        backend.chat = lambda prompt, **k: prompts.append(prompt) or reply("summary")
        trace = [Message("assistant", "", tool_calls=[call("needle")]),
                 Message("tool", "found it", tool_call_id="c1", name="echo__say")]
        agent._summarize("", [Message("user", "q"), Message("assistant", "a", tool_trace=trace)], 500)
        self.assertIn("tools used", prompts[0])
        self.assertIn("echo__say", prompts[0])
        self.assertIn("found it", prompts[0])

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            MemoryPolicy(tool_history=True)  # needs history_mode="messages"
        with self.assertRaises(ValueError):
            MemoryPolicy(history_mode="messages", tool_history="yes")
        self.assertTrue(MemoryPolicy.from_dict({"history_mode": "messages", "tool_history": True}).tool_history)


class TestFallbackTyped(unittest.TestCase):
    def test_a_failure_after_tools_ran_hands_the_same_messages_to_the_next_backend(self):
        a = TypedFake("a", [reply("", [call()])])
        a.fail = MakiNetworkError("down")
        b = TypedFake("b", [reply("from b")])
        fb = FallbackBackend([a, b], cooldown=0)
        self.assertTrue(fb.supports_typed_messages)
        agent = agent_with(fb)
        self.assertEqual(agent.execute_task("go", use_plugins=True), "from b")
        self.assertEqual(agent.plugins["echo"].runs, ["hi"], "the tool ran once")
        sent_to_b = b.rounds[0][0]
        self.assertEqual([m.role for m in sent_to_b], ["user", "assistant", "tool"])  # native, not flattened to text
        self.assertEqual(sent_to_b[2].content, "echo:hi")

    def test_mixed_chains_keep_the_transcript_path(self):
        class Untyped(TypedFake):
            supports_typed_messages = False
        self.assertFalse(FallbackBackend([TypedFake(), Untyped()]).supports_typed_messages)
        self.assertTrue(FallbackBackend([TypedFake(), TypedFake()]).supports_typed_messages)
        self.assertFalse(FallbackBackend([Untyped()]).supports_typed_messages)

    def test_a_chain_of_plain_backends_is_not_typed(self):
        class Plain(LLMBackend):
            model, temperature = "p", 0.0

            def request(self, prompt):
                return reply("p")
        self.assertFalse(FallbackBackend([Plain()]).supports_typed_messages)


class TestPlainChatIgnoresTraces(unittest.TestCase):
    def test_ollama_chat_history_carries_text_only(self):
        from maki.test.test_backend_conformance import OllamaAdapter
        case = unittest.TestCase()
        adapter = OllamaAdapter(case)
        backend = adapter.make()
        adapter.script_text(backend, "ok")
        trace = [Message("assistant", "", tool_calls=[call()]), Message("tool", "r", tool_call_id="c1")]
        backend.chat("now", history=[Message("user", "q"), Message("assistant", "a", tool_trace=trace)])
        raw = json.dumps(adapter.sent(backend).raw, default=str)
        self.assertNotIn("tool_trace", raw)
        self.assertNotIn("tool_calls", raw)


class TestWireFormats(unittest.TestCase):
    """The details the shared conformance suite can only check loosely."""

    def _backends(self):
        from maki.test.test_backend_conformance import AnthropicAdapter, OllamaAdapter, OpenAIAdapter
        case = unittest.TestCase()
        return {name: cls(case).make() for name, cls in
                (("ollama", OllamaAdapter), ("openai", OpenAIAdapter), ("anthropic", AnthropicAdapter))}

    MESSAGES = [Message("system", "ignored by anthropic"), Message("user", "q"),
                Message("assistant", "", tool_calls=[call("a", "id_a"), call("b", "id_b")]),
                Message("tool", "RA", tool_call_id="id_a", name="echo__say"),
                Message("tool", "RB", tool_call_id="id_b", name="echo__say")]

    def test_ollama(self):
        wire = self._backends()["ollama"].to_wire_messages(self.MESSAGES)
        self.assertEqual(wire[2]["tool_calls"][0], {"function": {"name": "echo__say", "arguments": {"text": "a"}}})
        self.assertEqual(wire[3], {"role": "tool", "content": "RA", "tool_name": "echo__say"})

    def test_openai(self):
        wire = self._backends()["openai"].to_wire_messages(self.MESSAGES)
        first = wire[2]["tool_calls"][0]
        self.assertEqual((first["id"], first["type"], first["function"]["name"]), ("id_a", "function", "echo__say"))
        self.assertEqual(json.loads(first["function"]["arguments"]), {"text": "a"})
        self.assertEqual(wire[3], {"role": "tool", "tool_call_id": "id_a", "content": "RA"})
        self.assertEqual(wire[4]["tool_call_id"], "id_b")

    def test_anthropic_flushes_results_that_end_the_conversation(self):
        wire = self._backends()["anthropic"].to_wire_messages(self.MESSAGES)
        self.assertEqual([m["role"] for m in wire], ["user", "assistant", "user"])
        self.assertEqual(len(wire[-1]["content"]), 2)

    def test_anthropic_batches_results_and_skips_system_and_empty_text(self):
        wire = self._backends()["anthropic"].to_wire_messages(self.MESSAGES + [Message("assistant", "  ")])
        self.assertEqual([m["role"] for m in wire], ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in wire[1]["content"]], ["tool_use", "tool_use"])  # no empty text block
        self.assertEqual(wire[1]["content"][0], {"type": "tool_use", "id": "id_a", "name": "echo__say",
                                                 "input": {"text": "a"}})
        results = wire[2]["content"]
        self.assertEqual([(b["type"], b["tool_use_id"], b["content"]) for b in results],
                         [("tool_result", "id_a", "RA"), ("tool_result", "id_b", "RB")])

    def test_anthropic_never_sends_an_empty_tool_result(self):
        wire = self._backends()["anthropic"].to_wire_messages(
            [Message("user", "q"), Message("assistant", "", tool_calls=[call()]),
             Message("tool", "", tool_call_id="c1", name="echo__say")])
        self.assertEqual(wire[-1]["content"][0]["content"], "(no output)")

    def test_images_survive_on_user_messages(self):
        for name, backend in self._backends().items():
            with self.subTest(backend=name):
                wire = backend.to_wire_messages([Message("user", "look", ["QUJD"])])
                self.assertIn("QUJD", json.dumps(wire))


if __name__ == "__main__":
    unittest.main()
