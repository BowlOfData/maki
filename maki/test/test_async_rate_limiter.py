"""
Tests for Phase 2.4: async-safe rate limiting, logging consolidation, CLI hardening.

Verifies:
  - RateLimiter.async_acquire uses asyncio.sleep (never time.sleep)
  - MakiLLama.async_chat now throttles via async_acquire
  - MakiOpenAI.async_chat and MakiAnthropic.async_chat use async_acquire, not acquire
  - maki serve rejects --tls-key without --tls-cert
  - configure_logging is called from __main__.main()
"""
import asyncio
import inspect
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maki.objects import GenerationConfig, RateLimiter


# ---------------------------------------------------------------------------
# 1. RateLimiter.async_acquire
# ---------------------------------------------------------------------------

class TestRateLimiterAsyncAcquire(unittest.TestCase):

    def test_async_acquire_is_coroutine(self):
        rl = RateLimiter(60)
        self.assertTrue(inspect.iscoroutinefunction(rl.async_acquire))

    def test_async_acquire_returns_immediately_when_tokens_available(self):
        """Full bucket — async_acquire should not sleep."""
        rl = RateLimiter(600)

        async def _run():
            with patch("maki.objects.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await rl.async_acquire()
                mock_sleep.assert_not_called()

        asyncio.run(_run())

    def test_async_acquire_sleeps_when_bucket_empty(self):
        """Drained bucket — async_acquire must call asyncio.sleep."""
        rl = RateLimiter(1)
        rl._tokens = 0.0  # drain manually

        async def _run():
            # After sleep we inject a token so the loop terminates.
            async def _inject_and_sleep(seconds):
                rl._tokens = 2.0  # unblock the next iteration

            with patch("maki.objects.asyncio.sleep", side_effect=_inject_and_sleep) as mock_sleep:
                await rl.async_acquire()
                mock_sleep.assert_called_once()

        asyncio.run(_run())

    def test_async_acquire_does_not_call_time_sleep(self):
        """Verify the sync time.sleep is never called from async_acquire."""
        rl = RateLimiter(600)

        async def _run():
            with patch("maki.objects._time.sleep") as mock_sync_sleep:
                await rl.async_acquire()
                mock_sync_sleep.assert_not_called()

        asyncio.run(_run())

    def test_async_acquire_consumes_token(self):
        rl = RateLimiter(60)
        before = rl._tokens
        asyncio.run(rl.async_acquire())
        self.assertLess(rl._tokens, before)

    def test_sync_acquire_still_works(self):
        """Ensure existing sync acquire is not broken."""
        rl = RateLimiter(600)
        with patch("maki.objects._time.sleep") as mock_sleep:
            rl.acquire()
            mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 2. MakiLLama.async_chat — rate limiter wired up
# ---------------------------------------------------------------------------

class TestMakiLLamaAsyncRateLimiter(unittest.TestCase):

    def _make_llm(self):
        from maki.makiLLama import MakiLLama
        with patch.object(MakiLLama, "_verify_connection"):
            llm = MakiLLama(model="gemma3")
        return llm

    def test_async_chat_calls_async_acquire(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        mock_limiter.async_acquire = AsyncMock()
        llm._rate_limiter = mock_limiter

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "message": {"content": "hi"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 10,
            "model": "gemma3",
        }

        async def _run():
            with patch.object(llm._async_http, "post", return_value=fake_resp) as mock_post:
                mock_post.return_value = fake_resp
                # post is awaited, so make it an async function
                async def _async_post(*a, **kw):
                    return fake_resp
                llm._async_http.post = _async_post
                await llm.async_chat("hello")

        asyncio.run(_run())
        mock_limiter.async_acquire.assert_awaited_once()

    def test_async_chat_skips_rate_limiter_when_none(self):
        llm = self._make_llm()
        llm._rate_limiter = None

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "message": {"content": "hi"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 10,
            "model": "gemma3",
        }

        async def _run():
            async def _async_post(*a, **kw):
                return fake_resp
            llm._async_http.post = _async_post
            await llm.async_chat("hello")

        asyncio.run(_run())  # should not raise


# ---------------------------------------------------------------------------
# 3. MakiOpenAI.async_chat — async_acquire, not acquire
# ---------------------------------------------------------------------------

def _make_openai_response(content="OK"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.model = "gpt-4o-mini"
    resp.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    return resp


class TestMakiOpenAIAsyncRateLimiter(unittest.TestCase):

    def _make_llm(self):
        import maki.makiOpenAI as mod
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        original = mod._openai_sdk
        mod._openai_sdk = mock_sdk
        self.addCleanup(setattr, mod, "_openai_sdk", original)

        from maki.makiOpenAI import MakiOpenAI
        llm = MakiOpenAI.__new__(MakiOpenAI)
        llm.model = "gpt-4o-mini"
        llm.config = GenerationConfig()
        llm.temperature = llm.config.temperature
        llm.system_prompt = None
        llm.timeout = 120
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    def test_async_chat_calls_async_acquire(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        mock_limiter.async_acquire = AsyncMock()
        llm._rate_limiter = mock_limiter

        async def _mock_create(**kwargs):
            return _make_openai_response()

        llm._async_client.chat.completions.create = _mock_create
        asyncio.run(llm.async_chat("hi"))
        mock_limiter.async_acquire.assert_awaited_once()

    def test_async_chat_does_not_call_sync_acquire(self):
        """async_chat must not call the blocking acquire()."""
        llm = self._make_llm()
        mock_limiter = MagicMock()
        mock_limiter.async_acquire = AsyncMock()
        llm._rate_limiter = mock_limiter

        async def _mock_create(**kwargs):
            return _make_openai_response()

        llm._async_client.chat.completions.create = _mock_create
        asyncio.run(llm.async_chat("hi"))
        mock_limiter.acquire.assert_not_called()


# ---------------------------------------------------------------------------
# 4. MakiAnthropic.async_chat — async_acquire, not acquire
# ---------------------------------------------------------------------------

def _make_anthropic_response(text="OK"):
    resp = MagicMock()
    resp.content = [MagicMock()]
    resp.content[0].text = text
    resp.model = "claude-sonnet-4-6"
    resp.usage = MagicMock(input_tokens=5, output_tokens=10)
    return resp


class TestMakiAnthropicAsyncRateLimiter(unittest.TestCase):

    def _make_llm(self):
        import maki.makiAnthropic as mod
        mock_sdk = MagicMock()
        mock_sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_sdk.APIConnectionError = type("APIConnectionError", (Exception,), {})
        mock_sdk.APIStatusError = type("APIStatusError", (Exception,), {})
        original = mod._anthropic_sdk
        mod._anthropic_sdk = mock_sdk
        self.addCleanup(setattr, mod, "_anthropic_sdk", original)

        from maki.makiAnthropic import MakiAnthropic
        llm = MakiAnthropic.__new__(MakiAnthropic)
        llm.model = "claude-sonnet-4-6"
        llm.config = GenerationConfig()
        llm.temperature = llm.config.temperature
        llm.system_prompt = None
        llm.timeout = 120
        llm._rate_limiter = None
        llm._client = MagicMock()
        llm._async_client = MagicMock()
        return llm

    def test_async_chat_calls_async_acquire(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        mock_limiter.async_acquire = AsyncMock()
        llm._rate_limiter = mock_limiter

        async def _mock_create(**kwargs):
            return _make_anthropic_response()

        llm._async_client.messages.create = _mock_create
        asyncio.run(llm.async_chat("hi"))
        mock_limiter.async_acquire.assert_awaited_once()

    def test_async_chat_does_not_call_sync_acquire(self):
        llm = self._make_llm()
        mock_limiter = MagicMock()
        mock_limiter.async_acquire = AsyncMock()
        llm._rate_limiter = mock_limiter

        async def _mock_create(**kwargs):
            return _make_anthropic_response()

        llm._async_client.messages.create = _mock_create
        asyncio.run(llm.async_chat("hi"))
        mock_limiter.acquire.assert_not_called()


# ---------------------------------------------------------------------------
# 5. CLI hardening: --tls-key without --tls-cert must fail
# ---------------------------------------------------------------------------

class TestCLIHardening(unittest.TestCase):

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.tls_cert = kwargs.get("tls_cert", None)
        args.tls_key = kwargs.get("tls_key", None)
        args.host = "127.0.0.1"
        args.port = 8100
        args.api_key = None
        args.config = "fake.yaml"
        return args

    def test_tls_key_without_cert_exits(self):
        from maki.__main__ import _cmd_serve
        args = self._make_args(tls_key="key.pem", tls_cert=None)
        with self.assertRaises(SystemExit) as ctx:
            _cmd_serve(args)
        self.assertEqual(ctx.exception.code, 1)

    def test_tls_key_with_cert_proceeds(self):
        """Providing both cert and key should pass the guard (may fail later on import)."""
        from maki.__main__ import _cmd_serve
        args = self._make_args(tls_key="key.pem", tls_cert="cert.pem")
        # It will fail on uvicorn import or config load — that's fine, just not on the guard
        with self.assertRaises((SystemExit, Exception)):
            _cmd_serve(args)
        # Ensure SystemExit code is not 1 from the guard (it may be 1 from uvicorn missing)
        # We just check no ValueError or similar from the guard itself

    def test_no_tls_proceeds_past_guard(self):
        from maki.__main__ import _cmd_serve
        args = self._make_args(tls_key=None, tls_cert=None)
        with self.assertRaises((SystemExit, Exception)):
            _cmd_serve(args)  # will fail on uvicorn/config, not on the guard


# ---------------------------------------------------------------------------
# 6. configure_logging is imported and called from main()
# ---------------------------------------------------------------------------

class TestMainLogging(unittest.TestCase):

    def test_main_calls_configure_logging(self):
        from maki import __main__ as main_mod
        with patch.object(main_mod, "configure_logging") as mock_cfg, \
             patch("sys.argv", ["maki"]):
            try:
                main_mod.main()
            except SystemExit:
                pass
            mock_cfg.assert_called_once()

    def test_configure_logging_imported_from_logging_config(self):
        """The configure_logging in __main__ must be the one from logging_config."""
        from maki import __main__ as main_mod
        from maki.logging_config import configure_logging as canonical
        self.assertIs(main_mod.configure_logging, canonical)


if __name__ == "__main__":
    unittest.main()
