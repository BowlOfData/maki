# Checkpoints and resume

A local model can take minutes per task. A checkpoint lets a run that was killed (crash, power cut, closed laptop) continue at the last model round it finished, without running the tools it already ran.

```python
from maki import Agent

agent = Agent("helper", backend, stateful=True, checkpoint="helper.db")
agent.load_plugin("file_reader")

# first run: killed halfway through a task
agent.execute_task("Summarise the reports in ./reports", use_plugins=True)
```

```python
# later, in a new process
agent = Agent("helper", backend, stateful=True, checkpoint="helper.db")
agent.load_plugin("file_reader")
answer = agent.resume()     # the answer of the unfinished task, or None if nothing was running
```

## What is saved

The file is one sqlite database (standard library, no new dependency) with one row per agent name.

- **After every finished task** (also a streamed one): the agent's whole state, the same data as `Agent.to_dict()`: the conversation (history, summary, pinned facts), memory and task history. Its size is bounded by your `MemoryPolicy`.
- **After every completed tool round** of a native tool loop: the unfinished task and its messages so far, tool results included.

`resume()` restores the state, then continues the unfinished task from its last saved round. A tool whose round was saved is never run again: the model sees its recorded result. The task's callbacks, trace spans, task history and conversation memory behave as for a normal task, so the resumed answer ends up in the conversation once.

## Rules to know

- **An existing checkpoint is never overwritten silently.** If the file already holds a checkpoint for the agent's name, the agent raises `CheckpointError` on its first task until you call `resume()` (continue) or `discard_checkpoint()` (start over).
- **A round is saved when all its tools are done.** If the process dies while a round's tools are running, those tools may have run; on resume the model asks for that round again. For tools with side effects, make them idempotent or keep `parallel` off and rounds small.
- **Only native tool loops resume mid-task**, on backends with typed messages (Ollama, OpenAI, Anthropic, OpenRouter, llama.cpp, vLLM). A plain chat, a `TOOL:` directive task or a stream is saved when it finishes; if the process dies in the middle, `resume()` restores the last finished state and the task has to be asked again. If the backend of the new process cannot continue a tool loop, the task starts again.
- **`resume()` loads the file once.** The first call in a process restores the saved state; a later call (because the resumed task failed again) keeps the live state and only re-runs the unfinished task.
- **Resume uses the new agent's configuration.** Name, role, instructions, plugins and policies come from your code; only memory and conversation come from the file. Load the same plugins before calling `resume()`.
- **The file holds whole conversations.** It is created with `0600` permissions and carries a schema version (a file written by a newer maki is refused). Query-string credentials (`?api_key=...`) and string values under credential-looking keys (`api_key`, `access_token`, `secret`, `password`, `Authorization`, ...; not `keyword` or `tokenizer`) are replaced by `***` before anything is written, so a resumed model sees `***` where a tool result held a secret.
- **One agent name per file row.** Agents with different names can share one file; two live agents with the same name overwrite each other (last write wins). A saving error (disk full) is logged and does not lose the answer.

## Pruning

A checkpoint is one row per agent, replaced on every save, so it does not grow with time. Delete the file, or call `agent.discard_checkpoint()`, to forget a run.

## Limits

- Everything is stored as JSON. A value in `agent.memory` that is not JSON (an object) comes back as its `str()`.
- Images in a conversation are stored as they are (base64), so a vision-heavy conversation makes a large file.
