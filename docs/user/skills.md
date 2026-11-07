# Skills

A skill is a folder of instructions an agent loads only when a task needs them. Maki reads the open Agent Skills format used by Claude Code, Codex and other agent tools, so a skill written for one of them works here and the other way round.

```text
release-notes/
├── SKILL.md          # required: name, description, then the instructions
├── references/       # optional: documents the instructions point to
├── scripts/          # optional: code (can be read, is never executed by maki)
└── assets/           # optional: templates and other files
```

```markdown
---
name: release-notes
description: Write release notes for a software version. Use for requests about changelogs or release announcements.
---
Release notes always start with the line "Codename: ..." and end with "Thanks, the team".
See references/style.md for tone.
```

## Using skills

```python
from maki import Agent, SkillRegistry

agent = Agent("helper", backend, skills=True)                  # your own skills (~/.agents/skills, ...)
agent = Agent("helper", backend, skills=["./team-skills"])     # specific folders
agent = Agent("helper", backend, skills=SkillRegistry.discover(trust_project=True))

agent.execute_task("Write release notes for version 3.2", use_plugins=True)
```

Skills work for tasks run with `use_plugins=True`. The model sees a short catalog (each skill's name and description, about 400 characters for three skills, capped at 4,000 by default) and two read-only tools:

- `skills.load(name)` returns the full instructions, plus the list of files bundled with the skill.
- `skills.read(name, path)` returns one bundled text file. It cannot leave the skill's folder (no `..`, absolute paths or symlinks that point outside), will not read binaries or files over 1 MB, and refuses `SKILL.md` (use `load`).

Output of both tools is capped at 30,000 characters. Skills are runtime configuration: like plugins, they are not saved by `Agent.to_dict()`. Skills are never executed, and `allowed-tools` in a skill is read but ignored: a skill cannot grant a tool.

## Where skills are found

First match wins:

1. Folders you pass explicitly (a folder of skills, or one skill folder itself; relative paths are relative to the working directory). These are trusted.
2. Project folders, from the working directory up to the git root (only the working directory when there is no git repository). In each, `.maki/skills`, then `.agents/skills`, then `.claude/skills`. **Loaded only with `trust_project=True`.**
3. User folders: `~/.maki/skills`, `~/.agents/skills`, then `~/.claude/skills`.

`.agents/skills` is the cross-tool convention (Codex reads it too), so a skill placed there is visible to maki and to those tools. A skill hidden by an earlier one of the same name is logged and listed in `registry.shadowed`; folders that fail to load are listed in `registry.problems`.

**Why project skills are opt-in.** A skill is text placed in front of your model. A repository you have just cloned can contain skills, so loading them is something you switch on for repositories you trust. When project skills are found and skipped, maki logs their names. `registry.untrusted` lists them.

## With `maki serve`

```yaml
skills:
  paths: [./skills]       # trusted folders that contain skill folders
  trust_project: false    # also load .maki/.agents/.claude skills from the working directory
  catalog_chars: 4000     # budget of the skill list shown to the model
```

`skills: true` loads user skills only.

## What to expect from models

Tested with a three-skill catalog on local models (one run each, so treat it as an observation): `gemma4:26b` and `qwen3.5:4b` both called `skills.load` for the matching task and neither called it for an unrelated question. The 26B model followed the instructions it loaded; the 4B model loaded the skill and then ignored one instruction in its answer. Small models pick the right skill more reliably than they obey it. Keep skills short, put the rule that matters first, and use [`maki eval`](eval.md) to check a skill on the model you will run.

Each skill's description is what the model decides on, so write it as when to use the skill, not only what it is.
