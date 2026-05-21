"""
Example usage of the ObsidianMemory plugin.

The vault path defaults to ~/ObsidianVaults/tranding or can be set via
the TRANDING_VAULT_PATH environment variable.
"""

import tempfile
from pathlib import Path
from maki.plugins.obsidian_memory.obsidian_memory import ObsidianMemory


def main():
    # Use a temp dir so the example is self-contained
    with tempfile.TemporaryDirectory() as tmp:
        mem = ObsidianMemory(vault_path=tmp)
        print("ObsidianMemory plugin example usage")
        print("=====================================")

        # Example 1: Write a note
        print("\nExample 1: Write a note")
        path = mem.append_note(
            folder="trades",
            title="BTC buy 2024-01-01",
            frontmatter={"tags": ["trade", "btc"], "status": "open", "price": 40000},
            body="Bought 0.001 BTC at $40,000 (paper trade).",
        )
        print(f"  Written: {path}")

        # Example 2: Read it back
        print("\nExample 2: Read the note")
        note = mem.read_note(path)
        print(f"  frontmatter: {note['frontmatter']}")
        print(f"  body: {note['body'][:60]}")

        # Example 3: Update frontmatter
        print("\nExample 3: Update status to 'closed'")
        ok = mem.update_frontmatter(path, "status", "closed")
        print(f"  updated={ok}")

        # Example 4: Query notes
        print("\nExample 4: Query notes with tag 'btc'")
        results = mem.query_notes("trades", tags=["btc"])
        print(f"  found {len(results)} note(s)")

        # Example 5: JSONL log
        print("\nExample 5: Append and read JSONL")
        mem.append_to_jsonl("logs/events.jsonl", {"event": "order_filled", "price": 40500})
        records = mem.read_jsonl("logs/events.jsonl")
        print(f"  records: {records}")


if __name__ == "__main__":
    main()
