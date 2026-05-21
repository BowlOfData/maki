# Obsidian Memory Plugin

Reads and writes Markdown notes with YAML frontmatter into a local Obsidian vault. Also supports append-only JSONL logs for structured data.

## Configuration

Vault path resolved in priority order:

1. `TRANDING_VAULT_PATH` environment variable
2. `vault_path` constructor argument
3. Default: `~/ObsidianVaults/tranding`

## Usage

```python
from maki.plugins.obsidian_memory.obsidian_memory import ObsidianMemory

mem = ObsidianMemory()

# Write a note
path = mem.append_note(
    folder="trades",
    title="BTC buy 2024-01-01",
    frontmatter={"tags": ["trade", "btc"], "status": "open"},
    body="Bought 0.01 BTC at $40,000",
)

# Read it back
note = mem.read_note(path)
print(note["frontmatter"], note["body"])

# Update a frontmatter key
mem.update_frontmatter(path, "status", "closed")

# Query recent notes
results = mem.query_notes("trades", since_hours=24, tags=["btc"])

# JSONL log
mem.append_to_jsonl("logs/events.jsonl", {"event": "order_filled", "price": 40500})
records = mem.read_jsonl("logs/events.jsonl", limit=100)
```

## Methods

### `append_note(folder, title, frontmatter, body)`

Writes (or overwrites) a Markdown note with YAML frontmatter. Returns the relative vault path.

### `read_note(rel_path)`

Returns `{"frontmatter": dict, "body": str}` or `{}` if not found.

### `query_notes(folder, since_hours=None, tags=None, frontmatter_filters=None, limit=20)`

Lists matching notes from a folder, newest-first. Filters by modification time, tags, and arbitrary frontmatter key/value pairs.

### `update_frontmatter(rel_path, key, value)`

Updates a single frontmatter key in an existing note. Returns `True` on success.

### `list_folder(folder)`

Returns relative paths of all `.md` files in `folder`.

### `append_to_jsonl(rel_path, record)`

Appends a JSON record to a `.jsonl` file, creating it if missing.

### `read_jsonl(rel_path, limit=500)`

Returns the last `limit` records from a `.jsonl` file.
