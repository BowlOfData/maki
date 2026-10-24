# Backends

All backends implement the same `LLMBackend` contract, so agents, workflows and plugins work unchanged on any of them.

| Backend | Where inference runs | Class | Requires |
|---|---|---|---|
| Ollama | Your machine or LAN | `MakiLLama` | a running Ollama server |
| llama.cpp | Your machine or LAN | `MakiLlamaCpp` | `maki-framework[llamacpp]` and a running `llama serve` |
| vLLM | Your GPU server or LAN | `MakiVllm` | `maki-framework[vllm]` and a running `vllm serve` |
| HuggingFace Transformers | In-process | `HFBackend` | `torch`, `transformers`, `accelerate` |
| OpenAI | Hosted | `MakiOpenAI` | `maki-framework[openai]` |
| Anthropic | Hosted | `MakiAnthropic` | `maki-framework[anthropic]` |
| OpenRouter | Hosted, any vendor-namespaced model | `MakiOpenRouter` | `maki-framework[openrouter]` |

## Ollama

```python
from maki import MakiLLama

llm = MakiLLama(model="llama3.2")
print(llm.chat("Explain recursion in one sentence.").content)
```

### Streaming and async

```python
import asyncio
from maki import MakiLLama
from maki.objects import GenerationConfig

llm = MakiLLama(model="llama3.2", config=GenerationConfig(temperature=0.7, max_tokens=512))

for chunk in llm.stream("Write a short haiku about testing"):
    print(chunk, end="", flush=True)

async def main():
    response = await llm.async_chat("Summarize the benefits of type hints.")
    print(response.content)

asyncio.run(main())
```

### Stateful session

```python
session = llm.session(system="You are a concise engineering assistant.")
session.say("We are building a release checklist.")
print(session.say("What should we verify before publishing a Python package?").content)
```

## llama.cpp

`MakiLlamaCpp` talks to a [llama.cpp](https://llama.app/docs/introduction) server, which runs GGUF models.

```bash
pip install "maki-framework[llamacpp]"
llama serve -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0    # listens on http://127.0.0.1:8080
```

```python
from maki import MakiLlamaCpp
from maki.objects import GenerationConfig

llm = MakiLlamaCpp(config=GenerationConfig(temperature=0.7, top_k=40))
print(llm.health())                 # False while the model is still loading

response = llm.chat("What is the capital of France?")
print(response.content)
print(response.reasoning)           # thinking text from reasoning models, else None
```

- Everything `MakiOpenAI` does works unchanged: chat, streaming, async, vision, sessions and native tool calling.
- `top_k` and `repeat_penalty` from `GenerationConfig` are sent to the server.
- `chat_collect()` (used by `Agent(use_streaming=True)`) streams internally so the timeout applies per chunk.
- **Router mode:** start `llama serve` without a model and pass the name, e.g. `MakiLlamaCpp(model="ggml-org/gemma-3-4b-it-qat-GGUF:Q4_0")`.
- **Other servers:** pass `base_url="http://192.168.1.20:8080/v1"` for a LAN host and `api_key=` (or `LLAMACPP_API_KEY`) if the server uses `--api-key`.

Embeddings and reranking need a server started with `--embedding` or `--rerank`:

```python
from maki.plugins.rag_memory import RagMemory

embedder = MakiLlamaCpp()           # llama serve -hf unsloth/embeddinggemma-300m-GGUF --embedding
rag = RagMemory(dsn="memory://", embedder=embedder.embed)

reranker = MakiLlamaCpp()           # llama serve -m reranker.gguf --rerank
for hit in reranker.rerank("What is a panda?", ["hi", "The giant panda is a bear."], top_n=1):
    print(hit["relevance_score"], hit["document"])
```

## vLLM

`MakiVllm` talks to a [vLLM](https://docs.vllm.ai) server: high-throughput GPU serving with continuous batching. Use it when several agents or workflows hit the same model at once.

```bash
pip install "maki-framework[vllm]"
vllm serve Qwen/Qwen2.5-1.5B-Instruct       # listens on http://127.0.0.1:8000
```

```python
from maki import MakiVllm
from maki.objects import GenerationConfig

# `model` must match the name the server exposes (the model path, or --served-model-name);
# it defaults to the vLLM quickstart model, so this line is optional here
llm = MakiVllm(model="Qwen/Qwen2.5-1.5B-Instruct",
               config=GenerationConfig(temperature=0.2))
print(llm.health())                 # True once the engine is ready

print(llm.chat("What is the capital of France?").content)
```

Everything `MakiOpenAI` does works unchanged (chat, streaming, async, vision, sessions), so an existing agent switches backends by changing one line.

**Structured output.** The reply is constrained by the server, so it always parses. This uses vLLM's `structured_outputs` request field, which needs a recent vLLM (older releases used `guided_json`/`guided_regex`/`guided_choice`):

```python
import json
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

reply = llm.chat_json("Extract the person: Ada Lovelace, 36 years old.", Person)
person = Person(**json.loads(reply.content))          # also accepts a plain JSON-schema dict

print(llm.chat_choice("The food was great!", ["positive", "negative"]).content)
print(llm.chat_regex("Give me a 4-digit PIN.", r"\d{4}").content)
```

**Batch generation.** Prompts run concurrently and vLLM batches them on the GPU. Results come back in prompt order. `chat_batch()` uses a thread pool, so it is safe in scripts and in code that already runs an event loop:

```python
replies = llm.chat_batch(["Summarise report A", "Summarise report B", "Summarise report C"],
                         max_concurrency=16)
print([r.content for r in replies])

# async variant
replies = await llm.async_chat_batch(prompts)
```

**LoRA adapters.** Start the server with `--enable-lora --lora-modules sql-expert=/path/to/adapter`, then select the adapter by name. One server can back several agents with different adapters:

```python
print(llm.models())                                    # ["Qwen/Qwen2.5-1.5B-Instruct", "sql-expert"]
sql_agent_llm = MakiVllm(model="sql-expert")
```

**Other options**

- **Tool calling:** start the server with `--enable-auto-tool-choice --tool-call-parser <parser>`; without them tool requests fail with an HTTP 400.
- **Remote server:** pass `base_url="http://gpu-box:8000/v1"` and `api_key=` (or `VLLM_API_KEY`) if the server uses `--api-key`.
- **Reasoning models:** the thinking text is returned in `response.reasoning`.
- **Long generations:** `chat_collect()` streams internally so the timeout applies per chunk.
- **Embeddings and reranking** need a server running an embedding or reranker model (see [Memory](memory.md) for a RAG example): `MakiVllm(model="BAAI/bge-m3").embed("text")` (usable as `RagMemory(embedder=...)`) and `MakiVllm(model="BAAI/bge-reranker-v2-m3").rerank(query, documents, top_n=3)`.

## Hosted: OpenAI, Anthropic, OpenRouter

```python
from maki import MakiOpenAI, MakiAnthropic, MakiOpenRouter

llm = MakiOpenAI(model="gpt-4o")
llm = MakiAnthropic(model="claude-sonnet-4-5")
llm = MakiOpenRouter(model="meta-llama/llama-3.1-70b-instruct")

print(llm.chat("What is the capital of France?").content)
```

API keys are read from the vendor's usual environment variable (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) or can be passed explicitly. Hosted-model traffic goes through the vendors' own SDKs.

## HuggingFace Transformers

`HFBackend` runs models in-process, no server required.

```bash
pip install torch transformers accelerate   # add bitsandbytes for 4/8-bit quantization
```

```python
from maki import HFBackend

llm = HFBackend(model_id="mistralai/Mistral-7B-Instruct-v0.3", device="cuda")
print(llm.chat("Explain attention mechanisms.").content)
```

Supports quantization (`load_in_4bit`, `load_in_8bit`) and device selection (`cpu`, `cuda`, `mps`; auto-detected when omitted). Remote model code is not executed unless you pass `trust_remote_code=True`.
