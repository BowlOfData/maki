"""Tests for schema-constrained generation (response_format + generate_structured)."""
import json
import unittest
from unittest.mock import MagicMock

from pydantic import BaseModel

from maki.exceptions import MakiValidationError
from maki.makiLLama import MakiLLama
from maki.objects import LLMResponse
from maki.structured import StructuredOutputError, generate_structured


class Verdict(BaseModel):
    ok: bool
    reason: str


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="m", prompt_tokens=1, completion_tokens=1,
                       total_tokens=2, elapsed_seconds=0.0, done=True)


class FakeBackend:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, prompt, history=None, config=None, system=None, response_format=None, think=None, **_):
        self.calls.append({"prompt": prompt, "history": history, "response_format": response_format,
                           "think": think, "system": system})
        return _resp(self.replies.pop(0))


class TestGenerateStructured(unittest.TestCase):
    def test_valid_first_try(self):
        b = FakeBackend([json.dumps({"ok": True, "reason": "prime"})])
        r = generate_structured(b, Verdict, "Is 7 prime?", system="sys")
        self.assertEqual(r.value, Verdict(ok=True, reason="prime"))
        self.assertEqual(r.attempts, 1)
        self.assertEqual(b.calls[0]["response_format"], Verdict.model_json_schema())
        self.assertIs(b.calls[0]["think"], False)
        self.assertEqual(b.calls[0]["system"], "sys")

    def test_retry_feeds_error_back(self):
        b = FakeBackend(['{"ok": "maybe"}', json.dumps({"ok": False, "reason": "no"})])
        r = generate_structured(b, Verdict, "q")
        self.assertEqual(r.attempts, 2)
        self.assertEqual(len(r.errors), 1)
        second = b.calls[1]
        self.assertIn("ok", second["prompt"])  # validation error names the field
        self.assertIn("rejected", second["prompt"])
        roles = [m.role for m in second["history"]]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(second["history"][1].content, '{"ok": "maybe"}')

    def test_malformed_json_retries(self):
        b = FakeBackend(["not json", json.dumps({"ok": True, "reason": "x"})])
        self.assertEqual(generate_structured(b, Verdict, "q").attempts, 2)

    def test_exhaustion_raises(self):
        b = FakeBackend(["bad", "bad", "bad"])
        with self.assertRaises(StructuredOutputError) as cm:
            generate_structured(b, Verdict, "q", max_retries=2)
        self.assertEqual(cm.exception.attempts, 3)
        self.assertEqual(cm.exception.last_raw, "bad")
        self.assertIsInstance(cm.exception, MakiValidationError)
        self.assertEqual(len(b.calls), 3)

    def test_zero_retries(self):
        b = FakeBackend(["bad", "bad"])
        with self.assertRaises(StructuredOutputError):
            generate_structured(b, Verdict, "q", max_retries=0)
        self.assertEqual(len(b.calls), 1)

    def test_negative_retries_rejected(self):
        with self.assertRaises(ValueError):
            generate_structured(FakeBackend([]), Verdict, "q", max_retries=-1)


class TestMakiLLamaPayload(unittest.TestCase):
    def test_schema_goes_into_format(self):
        llm = MakiLLama(model="gemma4:26b")
        schema = Verdict.model_json_schema()
        payload = llm._build_payload("hi", None, None, stream=False, response_format=schema, think=False)
        self.assertEqual(payload["format"], schema)
        self.assertIs(payload["think"], False)

    def test_json_format_flag_still_works_and_response_format_wins(self):
        llm = MakiLLama(model="m", json_format=True)
        self.assertEqual(llm._build_payload("hi", None, None, stream=False)["format"], "json")
        schema = {"type": "object"}
        self.assertEqual(
            llm._build_payload("hi", None, None, stream=False, response_format=schema)["format"], schema)

    def test_keep_alive_and_default_think(self):
        llm = MakiLLama(model="m", keep_alive="30m", think=True)
        p = llm._build_payload("hi", None, None, stream=False)
        self.assertEqual(p["keep_alive"], "30m")
        self.assertIs(p["think"], True)
        self.assertNotIn("format", p)

    def test_structured_reply_does_not_fall_back_to_thinking(self):
        llm = MakiLLama(model="m")
        data = {"message": {"content": "", "thinking": "reasoning..."}, "done": True}
        self.assertEqual(llm._parse_response(data, 0.1).content, "reasoning...")
        self.assertEqual(llm._parse_response(data, 0.1, thinking_fallback=False).content, "")

    def test_chat_sends_schema(self):
        llm = MakiLLama(model="m")
        llm._http = MagicMock()
        llm._http.post.return_value = MagicMock()
        from maki import makiLLama as mod
        original = mod.Connector.json_or_raise
        mod.Connector.json_or_raise = staticmethod(lambda r: {"message": {"content": '{"ok":true,"reason":"r"}'}, "done": True})
        try:
            r = llm.chat("q", response_format=Verdict.model_json_schema())
        finally:
            mod.Connector.json_or_raise = original
        sent = llm._http.post.call_args.kwargs["json"]
        self.assertEqual(sent["format"], Verdict.model_json_schema())
        self.assertEqual(Verdict.model_validate_json(r.content).reason, "r")


if __name__ == "__main__":
    unittest.main()
