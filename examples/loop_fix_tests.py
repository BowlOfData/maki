"""
Agent loop example: fix a failing test suite until it is green.

An engineer agent edits ``slug.py`` through the file plugins; the loop re-runs pytest after
every iteration and a separate judge agent checks the style requirement. The loop stops when
everything passes, or after ``max_iterations`` / a stall.

Requires a local Ollama with ``gemma4:26b`` (any chat model works; change ``MODEL``).
The pytest command is not sandboxed: it runs the code the agent just wrote. The agent can
only write inside the temporary project directory.

Run:  python examples/loop_fix_tests.py [--approve]
"""
import os
import sys
import tempfile

from maki.agents import Agent, Decision, Requirement
from maki.makiLLama import MakiLLama

MODEL = "gemma4:26b"

SLUG_PY = '''import re


def slugify(text, max_length=0):
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text[:max_length] if max_length else text
'''

TESTS_PY = '''from slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_no_edge_hyphens():
    assert slugify("  A -- B!! ") == "a-b"


def test_accents():
    assert slugify("Café Crème") == "cafe-creme"


def test_truncation_has_no_trailing_hyphen():
    assert slugify("hello world foo", max_length=6) == "hello"
'''


def main() -> None:
    approve = "--approve" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        project = os.path.realpath(tmp)
        with open(os.path.join(project, "slug.py"), "w") as f:
            f.write(SLUG_PY)
        with open(os.path.join(project, "test_slug.py"), "w") as f:
            f.write(TESTS_PY)

        engineer = Agent("engineer", MakiLLama(model=MODEL, think=False, timeout=300), role="software engineer",
                         instructions="Edit files with the file tools. Keep changes minimal.",
                         stateful=True, allow_dangerous_tools=True)
        for name in ("file_reader", "file_writer"):
            engineer.load_plugin(name).base_dir = project  # confine the agent to the project

        judge = Agent("judge", MakiLLama(model=MODEL, think=False, json_format=True, timeout=300), role="reviewer")

        requirements = [
            Requirement("tests", "the test suite passes",
                        command=[sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
            Requirement("style", "slugify has a docstring and no print() calls"),
        ]

        shown = []

        def show(event):
            if event is None:
                print("starting loop")
                return Decision.CONTINUE
            print(event.summary())
            shown.append(event)
            if approve and input("continue? [Y/n] ").strip().lower() == "n":
                return Decision.STOP
            return Decision.CONTINUE

        result = engineer.run_loop(
            f"Make the tests in {project}/test_slug.py pass by fixing {project}/slug.py.",
            requirements, judge, on_iteration=show, use_plugins=True, max_iterations=6,
            cwd=project, artifact_fn=lambda: open(os.path.join(project, "slug.py")).read())

        for event in result.events[len(shown):]:  # the hook is not called for the iteration that ends the loop
            print(event.summary())
        print(f"\nresult: {result.stop_reason.value} after {result.iterations} iteration(s)")
        print(f"scratchpad: {result.scratchpad_path}")
        with open(os.path.join(project, "slug.py")) as f:
            print("\nfinal slug.py:\n" + f.read())


if __name__ == "__main__":
    main()
