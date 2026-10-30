"""
Example: an agent tidies a scratch folder with the file_mover plugin.

Needs a local Ollama with ``gemma4:26b`` (change MODEL for another chat model). The agent can only touch
the temporary folder: the operator sets the scope, the model cannot widen it.

Run:  python maki/plugins/file_mover/example_usage.py
"""
import os
import tempfile

from maki.agents import Agent
from maki.makiLLama import MakiLLama

MODEL = "gemma4:26b"


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryDirectory() as journal:
        for name, text in (("report.txt", "Q3 report"), ("notes.md", "meeting notes"), ("todo.txt", "buy milk")):
            with open(os.path.join(scratch, name), "w") as handle:
                handle.write(text)

        agent = Agent("organiser", MakiLLama(model=MODEL, think=False, timeout=300), role="file organiser",
                      instructions="You organise files with the file_mover tool. Preview with dry_run first, "
                                   "then perform the move. dst is always the full new path.",
                      allow_dangerous_tools=True)  # moving files is a dangerous tool: opt in explicitly
        # The journal lives outside the folder being tidied (by default in ~/.maki/moves).
        mover = agent.load_plugin("file_mover", base_dir=scratch, journal_dir=journal)
        print("Scope:", mover.scope.base_dir)

        print(agent.execute_task(
            "Move report.txt to archive/2026/report.txt (create the folders) and rename notes.md to "
            "NOTES-OLD.md. Then tell me what you did.", use_plugins=True))

        print("\nFiles now:")
        for root, _, files in os.walk(scratch):
            for name in sorted(files):
                print("  ", os.path.relpath(os.path.join(root, name), scratch))

        print("\nUndoing the last move through the plugin:", mover.undo_last())
        print("Journal:", [(m["action"], os.path.basename(m["src"]), os.path.basename(m["dst"]))
                           for m in mover.list_moves()["moves"]])


if __name__ == "__main__":
    main()
