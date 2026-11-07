# Evaluating models

`maki.eval` answers "which model should I use for *my* agent?". It runs your tasks through the whole agent, with its tools, memory and policies, on every model you name, several times each, and reports pass rate, time and tokens. It tests the agent, not a bare prompt, so it measures what you will actually run.

```python
from maki.eval import EvalCase, run_eval

cases = [
    EvalCase("capital", "What is the capital of France? One word.", contains="paris"),
    EvalCase("sum", "What is 17 + 25? Reply with the number only.", contains="42"),
    EvalCase("json", 'Reply with only {"ok": true}', check=lambda answer: '"ok"' in answer),
]
report = run_eval(cases, ["ollama:qwen3.5:4b", "ollama:gemma4:26b"], runs=3)
print(report.table())
```

```text
suite: custom
model                   pass   rate  median s  avg tokens  errors
ollama:qwen3.5:4b        7/9    78%       9.8         964       0
ollama:gemma4:26b        9/9   100%       3.1         212       0
```

(The numbers above show the shape of the output, not a benchmark.)

## Cases

An `EvalCase` is a task plus the checks its answer must pass. Give at least one:

| Check | Meaning |
|---|---|
| `contains="x"` or `["x", "y"]` | Every substring appears in the answer, ignoring case. |
| `check=fn` | `fn(answer)` or `fn(answer, agent)` returns a bool, or `(bool, "why it failed")`. Use the second form to inspect the agent, for example which tools it called; it is used when the second parameter is required or named `agent` (an optional second parameter is left alone). |
| `criteria="..."` | A plain-language requirement decided by a judge agent you pass as `judge=`. |

When you give several, all must pass. Other fields: `use_plugins=True` runs with tool calling, `setup_turns=[...]` runs earlier turns first on the same agent (only the last answer is scored), and `make_agent=` builds this case's agent when it needs its own memory policy or plugins.

## Running

```python
report = run_eval(
    suite_or_cases,
    targets,                  # "provider:model" strings, LLMBackend objects, or {"label": target}
    runs=3,                   # repetitions per case and model
    make_agent=lambda backend: Agent("eval", backend, memory=...),
    judge=judge_agent,        # only for criteria cases; must not be stateful
    warmup=True,              # one tiny request per model so load time stays out of the timings
    on_result=print,          # called with each finished run
)
```

- Every run builds a fresh agent, so runs never share history.
- Runs are grouped by model: all of one model's cases, then the next. Switching a local model costs a reload.
- A failing model call is recorded as an error for that run and the evaluation goes on. A check that raises is also an error, reported as a bug in the case rather than a model failure.
- Models are not deterministic, and a single run says little. Three or more is a minimum; set a low temperature for steadier comparisons.

The report offers `stats()`, `pass_rate(model, case=None)`, `failures(model=None)`, `table()`, `to_dict()` and `save(path)` (JSON, with every run's output, tokens, calls and error).

## Built-in suites

| Suite | What it measures |
|---|---|
| `basic` | Following an exact instruction, arithmetic, extraction, JSON output, a format rule. |
| `tool_calling` | A typed tool call, a two-step tool chain, and not calling a tool when none is needed. |
| `long_context` | Recalling a code word placed at the start or end of a prompt of about 3,000 tokens (`get_suite("long_context", tokens=8000)` changes the size). A model whose context window is too small fails the start case. |
| `memory` | Recalling facts after many turns on a small conversation budget, with and without the summary tier. Slow: ten model calls per run. |

```bash
python -m maki eval --list
python -m maki eval --suite basic --models ollama:qwen3.5:4b ollama:gemma4:26b --runs 3 --no-think
python -m maki eval --suite long_context --models ollama:gemma4:26b --json report.json
```

`--no-think` turns thinking off for `ollama:` models (thinking models are several times slower and spend many more tokens), `--judge provider:model` supplies a judge (an `ollama:` judge runs in JSON mode with thinking off), and `--json FILE` saves the report. The exit code is non-zero if any run errored.

## Limits

- Scores come from small samples. A 2/3 against a 3/3 is a hint, not a ranking.
- Judge verdicts depend on the judge model, and `criteria` cases are only as good as their wording.
- Cost is not scored: local runs are free, and token counts are reported instead.
