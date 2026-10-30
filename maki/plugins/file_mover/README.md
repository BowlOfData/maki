# File Mover Plugin for Maki Framework

Moves and renames files and directories the same way on Linux, macOS and Windows. It is pure Python (no
shell), keeps a monthly journal of every move, can preview a move, and can undo the latest one.

## Features

- `move(src, dst)` for files, directories and symlinks (a link is moved as a link, never followed)
- `dry_run` preview of exactly what would happen (conflicts, case-only rename, cross-volume copy)
- Atomic rename on the same volume; across volumes it copies, verifies sizes and SHA-256, moves the copy
  into place and only then removes the source. A failure leaves the source untouched
- Case-only renames (`a.txt` to `A.txt`) on case-insensitive file systems
- `undo_last()` reverses the latest move, and refuses when it would be unsafe
- Operator-set scope: the model can never widen where the plugin may work

## Usage

```python
from maki.agents import Agent

agent = Agent(name="Organiser", maki_instance=llm, role="file organiser",
              allow_dangerous_tools=True)   # move and undo_last are dangerous tools
mover = agent.load_plugin("file_mover", base_dir="/path/to/project")

agent.execute_task("Move report.txt to archive/2026/report.txt, creating the folders.", use_plugins=True)
```

Without `allow_dangerous_tools=True` the model may only call `list_moves`.

### Operator options (code only, never tool arguments)

| Option | Default | Meaning |
|---|---|---|
| `base_dir` | current directory | Root that relative paths start from; always allowed |
| `allowed_roots` | none | Extra directories that are also allowed |
| `allow_any_path` | `False` | Allow every location. A move can then touch anything the process can reach, including system folders. Cannot be combined with `allowed_roots` |
| `journal_dir` | `~/.maki/moves` | Where the monthly journal files live |
| `retention_months` | `12` | Journal files older than this many calendar months are deleted when a new month starts; `None` keeps everything |
| `strict_names` | `False` | Apply the Windows file-name rules on every OS (see below) |

Options are passed to `register_plugin` by `agent.load_plugin("file_mover", **options)`.

## Tool methods

- `move(src, dst, overwrite=False, dry_run=False, create_dirs=False)`
  - `dst` is always the **full new path** of the item, never a folder to move it into. A trailing
    separator or an existing directory is an error.
  - A conflict is refused. `overwrite=True` replaces an existing **file** with a file; it never replaces a
    directory, and it **cannot be undone** (there is no trash).
  - `create_dirs=True` creates missing destination folders (undo recreates the original folder but never
    removes folders a move created).
  - The flags may be real booleans or the strings `"true"` / `"false"`.
- `undo_last()` moves the item of this plugin's latest move back. It refuses when there is nothing to undo, the latest
  entry is an undo (only one step), the move replaced a file, the item is gone or **changed since the move**
  (files: size and mtime; directories: entry count and newest mtime), or the original place is now taken.
- `list_moves(limit=10)` shows the latest journal entries, newest first. Only entries whose source and
  destination are both inside this plugin's scope are shown or undone, so several agents with different scopes
  can share one journal directory without seeing (or undoing) each other's moves.

Each method returns a dictionary with `success` and `error`.

## Per-OS behaviour

The API and the results are the same on every OS, except where the file systems themselves differ:

- **Windows-invalid names** (`CON`, `NUL`, `COM1`, a trailing dot or space, `<>:"|?*`) are refused only on a
  Windows host. Set `strict_names=True` to refuse them everywhere, so a tree you build on Linux can be copied
  to Windows. Without it, `aux.txt` works on Linux and fails on Windows.
- **Case**: `a.txt` and `A.txt` are one file on macOS/Windows (a case-only rename) and two files on most Linux
  file systems (a conflict). The plugin asks the file system, it does not guess.
- **Unicode**: names are not normalised (macOS may store the decomposed form).
- **No-clobber is best effort on POSIX**: a destination created by someone else between the check and the
  rename could be replaced. Windows refuses to rename onto an existing file.
- **Windows locks**: moving a file another program has open fails with an error and changes nothing.

## Safety notes

- Scope is checked for the source, the destination and the undo, by comparing file system objects (not
  strings), so it holds on case-insensitive disks and through symlinks.
- `allow_any_path=True` is the operator's decision. Do not turn it on for an agent that reads untrusted
  content (web pages, mail): it could be told to move files anywhere.
- A failed move removes the folders that `create_dirs` made for it.
- The journal is outside the folders being moved by default and is written after each move. If it cannot be
  written the move still happens and the result says it cannot be undone.
