import gc
import logging
import time
import threading
from typing import Generator, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from .maki import Maki
from .objects import GenerationConfig, LLMResponse


class HFBackend(Maki):
    """
    Runs any HuggingFace chat model directly via `transformers`.

    Supported model families:
        google/gemma-3-1b-it, google/gemma-3-4b-it, google/gemma-3-12b-it
        Qwen/Qwen2.5-0.5B-Instruct … Qwen/Qwen2.5-72B-Instruct
        meta-llama/Meta-Llama-3-8B-Instruct
        mistralai/Mistral-7B-Instruct-v0.3
        microsoft/Phi-3-mini-4k-instruct
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[str] = "auto",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(model=model_id)

        self._model_id = model_id

        # ── Device selection ──────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        self.logger.info("HFBackend · model=%s · device=%s", model_id, device)

        # ── Quantization ──────────────────────────────────────────────────
        bnb_config = None
        if load_in_4bit or load_in_8bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                self.logger.warning("BitsAndBytes quantization not available; loading in full precision.")

        # ── Load tokenizer ────────────────────────────────────────────────
        self.logger.info("Loading tokenizer …")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # ── Load model ────────────────────────────────────────────────────
        self.logger.info("Loading model weights (this may take a moment) …")
        load_kwargs: dict = dict(
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            device_map="auto" if device in ("cuda", "mps") else None,
        )
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        elif torch_dtype:
            load_kwargs["torch_dtype"] = (
                torch.float16 if torch_dtype == "float16"
                else torch.bfloat16 if torch_dtype == "bfloat16"
                else "auto"
            )

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if device == "cpu" and not bnb_config:
            self._model = self._model.to("cpu")

        self._model.eval()
        self.logger.info("Model loaded successfully.")

    @property
    def model_name(self) -> str:
        return self._model_id

    def _apply_chat_template(self, messages: list[dict]) -> str:
        """Use the model's own chat template if available, else fall back."""
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation for models without a chat template
            text = ""
            for m in messages:
                role, content = m["role"], m["content"]
                if role == "system":
                    text += f"<<SYS>>{content}<</SYS>>\n"
                elif role == "user":
                    text += f"[INST] {content} [/INST]\n"
                else:
                    text += f"{content}\n"
            return text

    def version(self) -> str:
        """Not supported — HFBackend runs locally with no remote API endpoint."""
        raise NotImplementedError("HFBackend does not expose a version endpoint; use model_name instead")

    def request_with_images(self, prompt: str, img: str) -> str:
        """Not supported via this interface — use generate() with a vision-capable model."""
        raise NotImplementedError(
            "HFBackend does not support request_with_images(); "
            "pass image tokens through generate() with an appropriate chat template"
        )

    def request(self, prompt: str) -> LLMResponse:
        """Override Maki.request to route through the local HF pipeline.

        Args:
            prompt: user prompt

        Returns:
            An LLMResponse containing the generated text and metadata

        Raises:
            ValueError: If prompt is not a valid string
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        if self._rate_limiter:
            self._rate_limiter.acquire()
        messages = [{"role": "user", "content": prompt.strip()}]
        return self.generate(messages, GenerationConfig())

    def generate(self, messages: list[dict], config: GenerationConfig) -> LLMResponse:
        """Run full generation and return an LLMResponse."""
        prompt = self._apply_chat_template(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs = dict(
            **inputs,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **config.to_hf_kwargs(),
        )

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**gen_kwargs)
        elapsed = time.perf_counter() - t0

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(new_tokens)

        return LLMResponse(
            content=text,
            model=self._model_id,
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
            elapsed_seconds=elapsed,
            backend="transformers",
        )

    def stream(self, messages: list[dict], config: GenerationConfig) -> Generator[str, None, None]:
        """Token-by-token streaming using a TextIteratorStreamer."""
        prompt = self._apply_chat_template(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
            **config.to_hf_kwargs(),
        )

        # Run generation in a background thread so we can yield from main thread.
        # inference_mode must be applied inside the thread — torch context managers
        # are per-thread and have no effect when entered from a different thread.
        def _generate():
            with torch.inference_mode():
                self._model.generate(**gen_kwargs)

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()

    def unload(self) -> None:
        """Release model from memory and free GPU cache."""
        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("HFBackend: model unloaded and memory freed.")
