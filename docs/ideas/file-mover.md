# File Mover Plugin

## Problem Statement
How might we let an LLM agent move and rename files and directories safely, with identical
behaviour on Linux, macOS and Windows, so the agent never needs to know which OS it runs on?

## Recommended Direction
"Safe mover": a `file_mover` plugin (class `FileMover`) built on a small cross-platform move
engine, with a preview mode and a journal so every move can be inspected before it happens and
reversed after, when reversal is possible.

It follows the existing plugin conventions (`ALLOWED_METHODS`, `DANGEROUS_METHODS`,
`PLUGIN_REGISTRY`). Mutating methods need `Agent(allow_dangerous_tools=True)`; the read-only
preview and journal listing do not.

Scope is decided by the operator, not the model. The default is the `base_dir` sandbox that
`file_writer` uses. Widening it is a constructor choice (`allowed_roots=[...]` or
`allow_any_path=True`, which allows every location with no carve-outs) that the LLM cannot change.

Conflicts are refused by default. `overwrite=True` replaces the destination and cannot be undone:
there is no trash, so the journal records that move as `irreversible`.

Every move is appended to a JSONL journal in `~/.maki/moves/`, one file per month. Files older than
`retention_months` (default 12) are deleted when a new month starts. `undo_last()` reverses only the
latest move, through the same scope checks, and refuses if the moved item has changed since.

## Key Assumptions to Validate
- [ ] The same API gives the same results on Linux, macOS and Windows: Ubuntu and macOS runners
      are required in CI; a Windows runner is best effort and non-blocking.
- [ ] Case-only renames (`a.txt` to `A.txt`), Windows reserved names, trailing dots and spaces, long
      paths and Unicode normalisation (macOS NFD vs NFC) are handled: table-driven tests per OS;
      unsupported cases return a clear error rather than a silent difference.
- [ ] Cross-filesystem moves are safe: copy, verify (size and hash), then delete the source, leaving
      the source intact if anything fails. Test with a temp dir on a second volume.
- [ ] Undo is correct after plain and cross-volume moves, and refuses cleanly when the item changed
      or when the move overwrote something: round-trip tests comparing directory trees.
- [ ] A tool-calling model uses `dry_run` and the error messages well: run with `gemma4:26b` on a
      scratch folder and read the transcripts.

## MVP Scope
In:
- `move(src, dst, overwrite=False, dry_run=False)` for files and directories (dangerous-gated;
  `dry_run=True` only reads and returns the exact plan: conflicts, case-only rename, cross-volume copy)
- `undo_last()` (dangerous-gated): reverses only the latest move; refuses if the item at the
  destination changed since the move (files: size and mtime; directories: a cheap fingerprint of
  entry count plus newest mtime), or if the move was an overwrite
- `list_moves(limit)` (read-only)
- Move engine: atomic `os.replace` on the same volume, verified copy-then-delete across volumes,
  case-only renames, reserved-name and trailing-dot checks, symlinks moved as links and never followed
- Conflict rule: refuse by default; `overwrite=True` replaces and is journaled as irreversible
- Scope: `base_dir` default; operator `allowed_roots` / `allow_any_path` (all locations, no
  carve-outs); enforced for source, destination and undo
- Journal in `~/.maki/moves/YYYY-MM.jsonl`; `retention_months` (default 12, None = keep all)
- `PLUGIN_REGISTRY` entry, README (with a warning about `allow_any_path`), tests, CI matrix

Out: trash folder, multi-step undo, content-hash change detection, batch moves, glob patterns,
remote (SFTP) moves, ownership and ACL preservation, a journal viewer UI.

## Not Doing (and Why)
- A trash folder: not wanted now. The cost is that overwrites cannot be undone, which the journal
  states plainly.
- Multi-step undo (`undo(n)`): one step is enough for the first release, and it keeps the staleness
  rule simple.
- Hash-based change detection for directories: the cheap fingerprint can miss an edit, but a hash
  is slow on large trees. Revisit if it bites.
- Batch `move_matching` with patterns: multiplies blast radius, and `fnmatch` case rules differ by
  OS. Add it after the single move is trusted.
- Local-to-remote moves through `ftp_client`: a different product with its own failure modes.
- Plan tokens (`plan_move` then `apply`): they need state between tool rounds; a `dry_run` flag is
  stateless and simpler for a model.
- Shelling out to `mv`, `move` or `robocopy`: they differ by OS. Pure Python (`pathlib`, `os`,
  `shutil`) only.
- Letting the model widen its own scope, or carving out protected paths under `allow_any_path`:
  scope is the operator's decision, and the README says what that means.
- Ownership, ACL and extended-attribute preservation: timestamps and permissions only for now.
- A blocking Windows CI job: a Windows-only failure does not block merges.

## Open Questions
- None blocking. Worth watching in the plan: how the cheap directory fingerprint behaves on big
  trees, and whether best-effort Windows CI ever shows real differences.
