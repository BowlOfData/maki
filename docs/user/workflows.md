# Workflows

`AgentManager` coordinates multiple agents.

```python
from maki import MakiLLama
from maki.agents import AgentManager, WorkflowTask

llm = MakiLLama(model="llama3.2")
manager = AgentManager(llm)

manager.add_agent("Researcher", role="researcher")
manager.add_agent("Writer", role="writer")

workflow = [
    WorkflowTask(
        name="research",
        agent="Researcher",
        task="Find the main public-release risks for this repository.",
    ),
    WorkflowTask(
        name="summary",
        agent="Writer",
        task="Summarize the research into a release checklist.",
        dependencies=["research"],
    ),
]

results = manager.run_workflow(workflow)
print(results["summary"]["result"])
```

## Manager patterns

| Method | Behaviour |
|---|---|
| `assign_task()` | Route a single task to one named agent |
| `coordinate_agents()` | Sequential multi-agent pipeline with optional synthesis step |
| `collaborative_task()` | All agents work on the same task independently |
| `run_workflow()` | Dependency-aware execution with retries and optional parallel batches |

`run_workflow()` orders tasks topologically, retries failures, can run independent tasks in parallel, and can checkpoint progress so an interrupted run resumes where it stopped.

## Verified tasks (maker/checker)

A task with `verify=` is only `COMPLETED` once a checker accepts it. A `REVISE` verdict re-runs the task with the checker's feedback, up to `max_revisions`. If verification ultimately fails, the task fails and its dependents are skipped. See `examples/verified_workflow.py` in the repository and `maki/agents/verification.py` for `VerifySpec` and `agent_verifier`.

## Human approval

Tasks can be gated on human approval before they run; see `maki/agents/approvals.py` (`ApprovalSpec` and the approval stores).
