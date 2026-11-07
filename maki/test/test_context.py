"""Context-window awareness: estimation, truncation detection, the sticky num_ctx policy and the probes."""
import logging
import threading
import unittest
from unittest.mock import MagicMock, patch

from maki import Agent, FallbackBackend
from maki.backend import LLMBackend
from maki.context import (ContextAware, ContextInfo, MakiContextError, PromptSizeCheck, choose_num_ctx,
                          estimate_tokens, messages_chars)
from maki.exceptions import MakiError
from maki.objects import GenerationConfig, LLMResponse
from maki.test.test_backend_conformance import LlamaCppAdapter, OllamaAdapter, VllmAdapter

BIG = "word " * 2000          # 10,000 characters: about 2,500 estimated tokens
HUGE = "word " * 8000         # about 10,000 estimated tokens


class TestEstimation(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual((estimate_tokens(0), estimate_tokens(3), estimate_tokens(400), estimate_tokens(-5)),
                         (0, 0, 100, 0))

    def test_messages_chars_counts_text_blocks_and_tool_calls(self):
        self.assertEqual(messages_chars([{"role": "user", "content": "abcd"}, {"role": "assistant", "content": ""}]), 4)
        blocks = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        self.assertEqual(messages_chars(blocks), 5)
        with_image = [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 50_000}}]}]
        self.assertEqual(messages_chars(with_image), 4)  # an image's base64 is not prompt text
        with_calls = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}]
        self.assertGreater(messages_chars(with_calls), 20)
        self.assertEqual(messages_chars([{"role": "user", "content": None}]), 0)
        self.assertEqual(messages_chars([MagicMock(content="xyz")]), 3)


class TestPromptSizeCheck(unittest.TestCase):
    def _check(self, chars, reported, ctx=None, **kw):
        check = PromptSizeCheck(**kw)
        check.note_request(chars, ctx)
        return check.check(reported)

    def test_a_faithful_report_is_fine(self):
        self.assertIsNone(self._check(10_000, 2_500))
        self.assertIsNone(self._check(10_000, 3_900))  # dense text reports more than the estimate

    def test_a_big_shortfall_is_flagged_with_numbers_and_advice(self):
        message = self._check(28_000, 1_027, ctx=2048)
        self.assertIn("1027", message)
        self.assertIn("7000", message)
        self.assertIn("2048", message)
        self.assertIn("num_ctx", message)

    def test_small_prompts_are_never_judged(self):
        self.assertIsNone(self._check(1_000, 10))          # about 250 estimated tokens
        self.assertIsNone(self._check(10_000, 0))          # a server that reports nothing says nothing

    def test_the_reported_figure_is_the_whole_prompt(self):
        # Ollama reported prompt_eval_count=7043 with 7038 tokens cached: cached tokens are part of the total
        self.assertIsNone(self._check(28_000, 7_043))
        self.assertIsNotNone(self._check(28_000, 1_027))

    def test_filling_the_window_with_a_longer_prompt_is_flagged_even_above_the_ratio(self):
        self.assertIsNotNone(self._check(28_000, 4_000, ctx=4096))   # 4000 of ~7000 sent, window 4096
        self.assertIsNone(self._check(10_000, 2_500, ctx=4096))      # the prompt fits the window

    def test_thresholds_are_configurable(self):
        self.assertIsNone(self._check(10_000, 1_200, ratio=0.4))
        self.assertIsNotNone(self._check(10_000, 1_200, ratio=0.6))
        self.assertIsNotNone(self._check(1_000, 10, min_tokens=100))

    def test_concurrent_async_tasks_keep_their_own_request(self):
        import asyncio
        check = PromptSizeCheck()
        findings = {}

        async def call(name, chars, reported, pause):
            check.note_request(chars)
            await asyncio.sleep(pause)  # the other task notes its request while this one waits for the server
            findings[name] = check.check(reported)

        async def main():
            await asyncio.gather(call("big-cut", 40_000, 500, 0.02), call("small-fine", 4_000, 900, 0.0))
        asyncio.run(main())
        self.assertIsNotNone(findings["big-cut"])   # judged against its own 10,000 estimated tokens
        self.assertIsNone(findings["small-fine"])

    def test_each_thread_sees_its_own_request(self):
        check = PromptSizeCheck()
        check.note_request(40_000)
        seen = []
        worker = threading.Thread(target=lambda: seen.append(check.last_request_chars()))
        worker.start()
        worker.join()
        self.assertEqual((seen, check.last_request_chars()), ([0], 40_000))


class TestChooseNumCtx(unittest.TestCase):
    def test_an_already_loaded_window_that_covers_the_need_is_reused(self):
        self.assertEqual(choose_num_ctx(3000, loaded=16384), 16384)
        self.assertEqual(choose_num_ctx(3000, loaded=65536, cap=32768), 65536)  # no reload beats the cap

    def test_the_smallest_step_with_headroom(self):
        self.assertEqual(choose_num_ctx(1000), 8192)
        self.assertEqual(choose_num_ctx(6500), 8192)    # 6500 * 1.25 = 8125
        self.assertEqual(choose_num_ctx(6600), 16384)   # 8250 does not fit 8192
        self.assertEqual(choose_num_ctx(20000), 32768)

    def test_a_loaded_window_that_is_too_small_is_replaced(self):
        self.assertEqual(choose_num_ctx(7000, loaded=4096), 16384)

    def test_cap_and_model_maximum_limit_the_choice(self):
        self.assertEqual(choose_num_ctx(50_000), 32768)                     # default cap
        self.assertEqual(choose_num_ctx(50_000, cap=131072), 65536)
        self.assertEqual(choose_num_ctx(10_000, model_max=8192), 8192)
        self.assertEqual(choose_num_ctx(10_000, cap=4096), 4096)

    def test_degenerate_input(self):
        self.assertEqual(choose_num_ctx(0), 8192)
        self.assertEqual(choose_num_ctx(-10), 8192)


class Plain(ContextAware):
    pass


class TestContextAwareModes(unittest.TestCase):
    def _backend(self, mode):
        backend = Plain()
        backend._init_size_check(mode)
        backend._note_request([{"role": "user", "content": BIG}], 2048)
        return backend

    def test_warn_logs_and_records(self):
        backend = self._backend("warn")
        with self.assertLogs("maki.context", "WARNING") as logs:
            backend._report_prompt_size(300)
        self.assertIn("evaluated only 300", logs.output[0])
        self.assertIn("300", backend.last_truncation)

    def test_error_raises_a_maki_error(self):
        backend = self._backend("error")
        with self.assertRaises(MakiContextError) as caught:
            backend._report_prompt_size(300)
        self.assertIsInstance(caught.exception, MakiError)

    def test_ignore_does_nothing(self):
        backend = self._backend("ignore")
        backend._report_prompt_size(300)
        self.assertIsNone(backend.last_truncation)

    def test_a_good_report_clears_the_finding(self):
        backend = self._backend("warn")
        with self.assertLogs("maki.context", "WARNING"):
            backend._report_prompt_size(300)
        backend._note_request([{"role": "user", "content": BIG}], 8192)  # the next request fits its window
        backend._report_prompt_size(2500)
        self.assertIsNone(backend.last_truncation)

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            Plain()._init_size_check("shout")

    def test_works_without_init(self):
        backend = Plain()  # a test double or subclass that skipped __init__
        backend._note_request([{"role": "user", "content": BIG}])
        with self.assertLogs("maki.context", "WARNING"):
            backend._report_prompt_size(100)

    def test_re_exported_from_the_public_module(self):
        from maki.exceptions import MakiContextError as original
        self.assertIs(MakiContextError, original)


class OllamaCase(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaAdapter(self)

    def make(self, **kw):
        backend = self.adapter.make(**kw)
        backend.context_info = MagicMock(return_value=ContextInfo(max_length=262144, loaded_length=None))
        return backend

    def sent_options(self, backend):
        return self.adapter.sent(backend).raw["options"]


class TestOllamaNumCtx(OllamaCase):
    def test_default_sends_no_num_ctx(self):
        backend = self.make()
        self.adapter.script_text(backend, "ok")
        backend.chat("hi")
        self.assertNotIn("num_ctx", self.sent_options(backend))
        backend.context_info.assert_not_called()  # no probing unless asked

    def test_an_integer_sets_the_config(self):
        backend = self.make(num_ctx=12288)
        self.adapter.script_text(backend, "ok")
        backend.chat("hi")
        self.assertEqual(self.sent_options(backend)["num_ctx"], 12288)
        self.assertEqual(backend.config.num_ctx, 12288)

    def test_an_integer_does_not_change_the_callers_config_object(self):
        config = GenerationConfig()
        self.make(config=config, num_ctx=4096)
        self.assertIsNone(config.num_ctx)

    def test_invalid_values(self):
        for bad in (0, -1, "big", True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make(num_ctx=bad)
        for cap in (0, "x", True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.make(num_ctx="auto", num_ctx_cap=cap)
        with self.assertRaises(ValueError):
            self.make(on_truncation="loud")

    def test_auto_chooses_once_from_the_first_request_and_keeps_it(self):
        backend = self.make(num_ctx="auto")
        for text in ("short", BIG * 5):
            self.adapter.script_text(backend, "ok")
            backend.chat(text)
        first, second = backend._session.calls[0]["options"], backend._session.calls[1]["options"]
        self.assertEqual((first["num_ctx"], second["num_ctx"]), (8192, 8192))
        backend.context_info.assert_called_once()

    def test_the_agents_hint_sizes_the_window(self):
        backend = self.make(num_ctx="auto")
        backend.hint_context(7000)
        self.adapter.script_text(backend, "ok")
        backend.chat("short")
        self.assertEqual(self.sent_options(backend)["num_ctx"], 16384)  # 7000 plus the answer, with headroom
        backend.hint_context(50)
        self.assertEqual(backend._context_hint, 7000)  # hints only ever grow

    def test_a_user_set_window_wins_over_auto(self):
        backend = self.make(num_ctx="auto", config=GenerationConfig(num_ctx=3000))
        self.adapter.script_text(backend, "ok")
        backend.chat("hi")
        self.assertEqual(self.sent_options(backend)["num_ctx"], 3000)
        backend.context_info.assert_not_called()

    def test_auto_reuses_the_loaded_window_and_respects_the_model_maximum_and_cap(self):
        reuse = self.make(num_ctx="auto")
        reuse.context_info.return_value = ContextInfo(max_length=262144, loaded_length=65536)
        self.adapter.script_text(reuse, "ok")
        reuse.chat(BIG)
        self.assertEqual(self.sent_options(reuse)["num_ctx"], 65536)
        small = self.make(num_ctx="auto")
        small.context_info.return_value = ContextInfo(max_length=4096)
        small.hint_context(20000)
        self.adapter.script_text(small, "ok")
        small.chat("hi")
        self.assertEqual(self.sent_options(small)["num_ctx"], 4096)
        capped = self.make(num_ctx="auto", num_ctx_cap=16384)
        capped.hint_context(30000)
        self.adapter.script_text(capped, "ok")
        capped.chat("hi")
        self.assertEqual(self.sent_options(capped)["num_ctx"], 16384)

    def test_auto_also_applies_to_the_tool_path_and_streaming_collect(self):
        backend = self.make(num_ctx="auto")
        self.adapter.script_text(backend, "ok")
        backend.chat_with_tools([{"role": "user", "content": "q"}], [])
        self.assertEqual(self.sent_options(backend)["num_ctx"], 8192)
        self.adapter.script_stream(backend, ["a", "b"])
        backend.chat_collect("q")
        self.assertEqual(self.sent_options(backend)["num_ctx"], 8192)

    def test_an_unreachable_probe_still_gives_a_usable_window(self):
        backend = self.make(num_ctx="auto")
        backend.context_info.return_value = ContextInfo()
        self.adapter.script_text(backend, "ok")
        backend.chat("hi")
        self.assertEqual(self.sent_options(backend)["num_ctx"], 8192)


class TestOllamaTruncation(OllamaCase):
    def test_a_silent_cut_is_logged_and_recorded(self):
        backend = self.make(num_ctx=2048)
        self.adapter.script_text(backend, "wrong", prompt_tokens=1027)
        with self.assertLogs("maki.context", "WARNING") as logs:
            response = backend.chat(HUGE)
        self.assertEqual(response.content, "wrong")
        self.assertIn("evaluated only 1027", logs.output[0])
        self.assertIn("context window 2048", backend.last_truncation)

    def test_error_mode_raises(self):
        backend = self.make(num_ctx=2048, on_truncation="error")
        self.adapter.script_text(backend, "wrong", prompt_tokens=1027)
        with self.assertRaises(MakiContextError):
            backend.chat(HUGE)

    def test_a_faithful_report_is_silent(self):
        backend = self.make(num_ctx=16384)
        self.adapter.script_text(backend, "right", prompt_tokens=10_050)
        with self.assertNoLogs("maki.context", "WARNING"):
            backend.chat(HUGE)
        self.assertIsNone(backend.last_truncation)

    def test_every_response_path_is_checked(self):
        for name, call in (("chat_with_tools", lambda b: b.chat_with_tools([{"role": "user", "content": HUGE}], [])),
                           ("chat_turn", lambda b: b.chat_turn([__import__("maki").Message("user", HUGE)], None)),
                           ("chat_collect", lambda b: b.chat_collect(HUGE))):
            with self.subTest(path=name):
                backend = self.make(num_ctx=2048, on_truncation="error")
                if name == "chat_collect":
                    backend._session.queue.append(self._collect_reply(prompt_tokens=900))
                else:
                    self.adapter.script_text(backend, "x", prompt_tokens=900)
                with self.assertRaises(MakiContextError):
                    call(backend)

    def _collect_reply(self, prompt_tokens):
        import json

        from maki.test.test_backend_conformance import _OllamaResponse
        lines = [json.dumps({"message": {"content": "x"}, "done": False}).encode(),
                 json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": prompt_tokens,
                             "eval_count": 1}).encode()]
        return _OllamaResponse(lines=lines)

    def test_a_default_window_is_learned_after_a_big_response_and_then_the_cut_shows(self):
        """No num_ctx sent: 4,096 of about 7,500 tokens is above the ratio, so only the window gives it away."""
        backend = self.make()
        backend._http.get = MagicMock(return_value=MagicMock(
            status_code=200, text="{}", json=lambda: {"models": [{"name": "m", "context_length": 4096}]}))
        self.adapter.script_text(backend, "wrong", prompt_tokens=4096)
        with self.assertLogs("maki.context", "WARNING") as logs:
            backend.chat("word " * 6000)  # about 7,500 estimated tokens
        self.assertIn("context window 4096", logs.output[0])
        self.assertEqual(backend._loaded_window, 4096)
        backend._http.get.reset_mock()
        self.adapter.script_text(backend, "wrong again", prompt_tokens=4096)
        with self.assertLogs("maki.context", "WARNING"):
            backend.chat("word " * 6000)
        backend._http.get.assert_not_called()  # learned once, remembered

    def test_the_window_is_asked_for_only_after_a_big_response_and_only_once(self):
        backend = self.make()
        backend._http.get = MagicMock(return_value=MagicMock(
            status_code=200, text="{}", json=lambda: {"models": [{"name": "m", "context_length": 32768}]}))
        self.adapter.script_text(backend, "ok", prompt_tokens=20)
        backend.chat("a short question")
        backend._http.get.assert_not_called()                  # nothing big enough to judge
        for _ in range(2):
            self.adapter.script_text(backend, "ok", prompt_tokens=7_600)
            backend.chat("word " * 6000)                        # faithful, big, window unknown
        backend._http.get.assert_called_once()                  # learned on the first, remembered after
        self.assertIsNone(backend.last_truncation)

    def test_a_server_that_will_not_say_leaves_the_check_silent(self):
        backend = self.make()
        backend._http.get = MagicMock(side_effect=ConnectionError("no"))
        self.adapter.script_text(backend, "x", prompt_tokens=4096)
        backend.chat("word " * 6000)
        self.assertIsNone(backend.last_truncation)
        self.assertIsNone(getattr(backend, "_loaded_window", None))

    def test_a_fully_cached_prompt_is_not_mistaken_for_a_cut(self):
        from maki.test.test_backend_conformance import _OllamaResponse
        backend = self.make(num_ctx=16384, on_truncation="error")
        backend._session.queue.append(_OllamaResponse(
            {"model": "m", "message": {"role": "assistant", "content": "x"}, "done": True,
             "prompt_eval_count": 10_000, "prompt_eval_cached_count": 9_990, "eval_count": 2}))
        backend.chat(HUGE)  # the count is the whole prompt, cached part included

    def test_cached_tokens_are_not_added_to_a_truncated_report(self):
        from maki.test.test_backend_conformance import _OllamaResponse
        backend = self.make(num_ctx=2048, on_truncation="error")
        backend._session.queue.append(_OllamaResponse(
            {"model": "m", "message": {"role": "assistant", "content": "x"}, "done": True,
             "prompt_eval_count": 1_027, "prompt_eval_cached_count": 1_026, "eval_count": 2}))
        with self.assertRaises(MakiContextError):  # seen is 1027, not 2053
            backend.chat(HUGE)

    def test_ignore_mode_skips_all_bookkeeping(self):
        backend = self.make(num_ctx=2048, on_truncation="ignore")
        self.adapter.script_text(backend, "x", prompt_tokens=100)
        backend.chat(HUGE)
        self.assertIsNone(backend.last_truncation)


class TestOllamaProbe(unittest.TestCase):
    def _backend(self, show=None, ps=None, show_error=None, ps_error=None, model="gemma4:26b"):
        backend = OllamaAdapter(self).make()
        backend.model = model

        def reply(payload):
            return MagicMock(status_code=200, json=lambda: payload, text="{}")
        backend._http = MagicMock()
        if show_error:
            backend._http.post.side_effect = show_error
        else:
            backend._http.post.return_value = reply(show or {})
        if ps_error:
            backend._http.get.side_effect = ps_error
        else:
            backend._http.get.return_value = reply(ps or {"models": []})
        return backend

    def test_reads_the_model_maximum_and_the_loaded_window(self):
        backend = self._backend({"model_info": {"general.architecture": "gemma4", "gemma4.context_length": 262144}},
                                {"models": [{"name": "other:1b", "context_length": 4096},
                                            {"name": "gemma4:26b", "context_length": 16384}]})
        info = backend.context_info()
        self.assertEqual((info.max_length, info.loaded_length, info.effective), (262144, 16384, 16384))
        self.assertEqual(backend._max_context, 262144)

    def test_untagged_model_names_match_latest(self):
        backend = self._backend({"model_info": {}}, {"models": [{"name": "llama3:latest", "context_length": 8192}]},
                                model="llama3")
        self.assertEqual(backend.context_info().loaded_length, 8192)

    def test_either_probe_may_fail_independently(self):
        only_ps = self._backend(show_error=ConnectionError("x"), ps={"models": [{"name": "gemma4:26b", "context_length": 4096}]})
        self.assertEqual(only_ps.context_info(), ContextInfo(None, 4096, "ollama"))
        only_show = self._backend({"model_info": {"x.context_length": 1000}}, ps_error=ConnectionError("x"))
        self.assertEqual(only_show.context_info(), ContextInfo(1000, None, "ollama"))
        neither = self._backend(show_error=ConnectionError("x"), ps_error=ConnectionError("y"))
        self.assertEqual(neither.context_info(), ContextInfo(None, None, "ollama"))

    def test_probes_use_a_short_timeout(self):
        backend = self._backend({"model_info": {}}, {"models": []})
        backend.timeout = 300
        backend.context_info()
        self.assertEqual(backend._http.post.call_args.kwargs["timeout"], 5)
        self.assertEqual(backend._http.get.call_args.kwargs["timeout"], 5)

    def test_the_largest_context_length_key_wins(self):
        backend = self._backend({"model_info": {"a.context_length": 4096, "b.context_length": 8192, "c.other": 99}})
        self.assertEqual(backend.context_info().max_length, 8192)


class CompatCase(unittest.TestCase):
    adapter_class = LlamaCppAdapter

    def setUp(self):
        self.adapter = self.adapter_class(self)

    def make(self, window=None, **kw):
        backend = self.adapter.make(**kw)
        backend._probe_context = MagicMock(return_value=ContextInfo(loaded_length=window))
        return backend


class TestCompatTruncation(CompatCase):
    def test_a_short_report_for_a_big_prompt_is_flagged_with_the_probed_window(self):
        backend = self.make(window=4096)
        self.adapter.script_text(backend, "wrong", prompt_tokens=900)
        with self.assertLogs("maki.context", "WARNING") as logs:
            backend.chat(HUGE)
        self.assertIn("context window 4096", logs.output[0])

    def test_error_mode_and_a_faithful_report(self):
        backend = self.make(window=32768, on_truncation="error")
        self.adapter.script_text(backend, "right", prompt_tokens=10_100)
        backend.chat(HUGE)
        self.adapter.script_text(backend, "wrong", prompt_tokens=900)
        with self.assertRaises(MakiContextError):
            backend.chat(HUGE)

    def test_the_window_is_probed_once_and_only_for_big_prompts(self):
        backend = self.make(window=8192)
        self.adapter.script_text(backend, "ok")
        backend.chat("a short prompt")
        backend._probe_context.assert_not_called()
        for _ in range(2):
            self.adapter.script_text(backend, "ok", prompt_tokens=10_000)
            backend.chat(HUGE)
        backend._probe_context.assert_called_once()

    def test_a_vision_prompt_is_not_mistaken_for_a_cut(self):
        backend = self.make(window=32768, on_truncation="error")
        self.adapter.script_text(backend, "a cat", prompt_tokens=700)
        history = [__import__("maki").Message("user", "Describe this. " * 100, images=["A" * 400_000])]
        backend.chat("and now?", history=history)  # 700 tokens reported; the image is not text

    def test_a_failing_probe_never_breaks_the_request(self):
        backend = self.make()
        backend._probe_context.side_effect = RuntimeError("no /props")
        self.adapter.script_text(backend, "ok", prompt_tokens=10_000)
        self.assertEqual(backend.chat(HUGE).content, "ok")
        self.adapter.script_text(backend, "ok", prompt_tokens=10_000)
        backend.chat(HUGE)
        backend._probe_context.assert_called_once()  # the failure is cached too

    def test_the_tool_and_streaming_paths_are_checked(self):
        backend = self.make(window=4096, on_truncation="error")
        self.adapter.script_text(backend, "x", prompt_tokens=900)
        with self.assertRaises(MakiContextError):
            backend.chat_with_tools([{"role": "user", "content": HUGE}], [])

    def test_hint_context_is_accepted_and_ignored(self):
        self.make().hint_context(12345)


class TestCompatProbes(unittest.TestCase):
    def _llama(self, props):
        backend = LlamaCppAdapter(self).make()
        backend._http = MagicMock()
        backend._http.get.return_value = MagicMock(status_code=200, json=lambda: props, text="{}")
        return backend

    def test_llama_cpp_reads_props(self):
        backend = self._llama({"default_generation_settings": {"n_ctx": 16384}})
        self.assertEqual(backend.context_info(), ContextInfo(None, 16384, "llama.cpp"))
        self.assertIn("/props", backend._http.get.call_args.args[0])
        self.assertNotIn("/v1/props", backend._http.get.call_args.args[0])

    def test_llama_cpp_without_the_field(self):
        self.assertIsNone(self._llama({}).context_info().loaded_length)
        self.assertIsNone(self._llama({"default_generation_settings": {"n_ctx": 0}}).context_info().loaded_length)

    def test_vllm_reads_max_model_len_for_its_model(self):
        backend = VllmAdapter(self).make()
        backend._http = MagicMock()
        listing = {"data": [{"id": "other", "max_model_len": 99}, {"id": "m", "max_model_len": 32768}]}
        backend._http.get.return_value = MagicMock(status_code=200, json=lambda: listing, text="{}")
        self.assertEqual(backend.context_info(), ContextInfo(None, 32768, "vllm"))
        backend._http.get.return_value = MagicMock(status_code=200, json=lambda: {"data": [{"id": "m"}]}, text="{}")
        backend._context_cache = None
        self.assertIsNone(backend.context_info().loaded_length)


class TestAgentAndFallback(unittest.TestCase):
    class Hinted(LLMBackend):
        model, temperature = "h", 0.0

        def __init__(self):
            self.hints = []

        def request(self, prompt):
            return LLMResponse(content="x", model="h", prompt_tokens=1, completion_tokens=1, total_tokens=2,
                               elapsed_seconds=0.0)

        def hint_context(self, tokens):
            self.hints.append(tokens)

    def test_an_agent_tells_its_backend_what_it_needs(self):
        backend = self.Hinted()
        Agent("a", backend, stateful=True)
        from maki.objects import ConversationMemory
        self.assertEqual(backend.hints, [ConversationMemory.DEFAULT_TOKEN_BUDGET + 2048])

    def test_a_memory_budget_changes_the_hint(self):
        from maki import MemoryPolicy
        backend = self.Hinted()
        Agent("a", backend, stateful=True, memory=MemoryPolicy(token_budget=10_000))
        self.assertEqual(backend.hints, [12_048])

    def test_backends_without_the_method_are_fine(self):
        class Plain(LLMBackend):
            model, temperature = "p", 0.0

            def request(self, prompt):
                raise NotImplementedError
        Agent("a", Plain())

    def test_a_fallback_chain_forwards_the_hint_to_every_backend_that_has_one(self):
        a, b = self.Hinted(), self.Hinted()

        class Plain(LLMBackend):
            model, temperature = "p", 0.0

            def request(self, prompt):
                raise NotImplementedError
        chain = FallbackBackend([a, Plain(), b])
        chain.hint_context(9000)
        self.assertEqual((a.hints, b.hints), ([9000], [9000]))
        Agent("x", chain)
        self.assertEqual(len(a.hints), 2)


if __name__ == "__main__":
    unittest.main()
