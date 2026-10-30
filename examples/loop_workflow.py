"""
A workflow whose "implement" task loops until its tests pass, then a downstream task runs.

The engineer agent edits ``slug.py`` through the file plugins; the loop re-runs pytest after every
iteration and a separate judge checks the style requirement. If the loop does not succeed the task
fails and "summarize" is skipped. Requires a local Ollama with ``gemma4:26b``.
The pytest command is not sandboxed: it runs the code the agent just wrote.
"""
import os
import sys
import tempfile

from maki.agents import AgentManager, LoopTask, Requirement, WorkflowTask
from maki.agents.agent import Agent
from maki.makiLLama import MakiLLama

MODEL = "gemma4:26b"
SLUG_PY = 'import re\n\n\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.lower())\n'
TESTS_PY = ('from slug import slugify\n\n\ndef test_basic():\n    assert slugify("Hello World") == "hello-world"\n\n\n'
            'def test_edges():\n    assert slugify("  A -- B!! ") == "a-b"\n')


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = os.path.realpath(tmp)
        for name, body in (("slug.py", SLUG_PY), ("test_slug.py", TESTS_PY)):
            with open(os.path.join(project, name), "w") as f:
                f.write(body)

        mgr = AgentManager(MakiLLama(model=MODEL, think=False))
        engineer = Agent("engineer", MakiLLama(model=MODEL, think=False, timeout=300), role="software engineer",
                         instructions="Edit files with the file tools. Keep changes minimal.",
                         stateful=True, allow_dangerous_tools=True)
        for plugin in ("file_reader", "file_writer"):
            engineer.load_plugin(plugin).base_dir = project
        judge = Agent("judge", MakiLLama(model=MODEL, think=False, json_format=True, timeout=300), role="reviewer")
        summarizer = Agent("summarizer", MakiLLama(model=MODEL, think=False, timeout=300), role="writer")
        mgr.agents.update(engineer=engineer, judge=judge, summarizer=summarizer)

        implement = WorkflowTask(
            "implement", "engineer", f"Make the tests in {project}/test_slug.py pass by fixing {project}/slug.py.",
            loop=LoopTask(
                judge="judge", use_plugins=True, max_iterations=5, cwd=project, context="shared",
                requirements=[Requirement("tests", "the test suite passes",
                                          command=[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
                              Requirement("style", "slugify has a docstring")],
                artifact_fn=lambda: open(os.path.join(project, "slug.py")).read()))
        summarize = WorkflowTask("summarize", "summarizer", "Describe in two sentences what changed in slug.py.",
                                 dependencies=["implement"])

        results = mgr.run_workflow([implement, summarize])
        for name, entry in results.items():
            print(f"\n== {name}: {mgr.last_workflow_state.tasks[name]['status'].value}")
            print((entry.get("result") or "")[:600])


if __name__ == "__main__":
    main()
