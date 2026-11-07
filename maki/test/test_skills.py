"""Agent Skills: parsing, validation, discovery order and trust, the catalog, the tools and the agent wiring."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maki import Agent
from maki.backend import LLMBackend
from maki.objects import LLMResponse, ToolCall
from maki.skills import (DEFAULT_CATALOG_CHARS, SKILL_FILE, Skill, SkillRegistry, SkillsPlugin, _simple_yaml,
                         coerce_registry, load_skill, parse_skill_text, project_roots)


def reply(text="", calls=None):
    return LLMResponse(content=text, model="m", prompt_tokens=3, completion_tokens=2, total_tokens=5,
                       elapsed_seconds=0.0, tool_calls=calls)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def skill(self, base, name, description="Does a thing.", body="Instructions here.", extra="", dirname=None):
        folder = Path(base) / (dirname or name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SKILL_FILE).write_text(f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n")
        return folder


class TestParsing(unittest.TestCase):
    def test_quoted_unquoted_and_nested_values(self):
        front, body = parse_skill_text(
            "---\nname: pdf-tools\ndescription: \"Work with PDFs\"\nlicense: MIT\ncompatibility: maki >= 0.5\n"
            "allowed-tools: Read Grep\nmetadata:\n  version: \"1.2\"\n  owner: team\n---\n# Title\nBody text\n")
        self.assertEqual((front["name"], front["description"], front["license"]), ("pdf-tools", "Work with PDFs", "MIT"))
        self.assertEqual(front["metadata"], {"version": "1.2", "owner": "team"})
        self.assertEqual(body, "# Title\nBody text")

    def test_block_scalars(self):
        front, _ = parse_skill_text("---\nname: a\ndescription: >-\n  first line\n  second line\nnotes: |\n  keep\n  lines\n---\nx")
        self.assertEqual(front["description"], "first line second line")

    def test_a_colon_in_an_unquoted_description_is_survived(self):
        front, _ = parse_skill_text("---\nname: a\ndescription: Use when: the user asks about X\n---\nbody")
        self.assertEqual(front["description"], "Use when: the user asks about X")

    def test_the_forgiving_reader_alone(self):
        data = _simple_yaml("name: x\n# comment\ndescription: >\n  folded\n  text\ntools:\n  - one\n  - 'two'\nmetadata:\n  k: v\n")
        self.assertEqual(data, {"name": "x", "description": "folded text", "tools": ["one", "two"], "metadata": {"k": "v"}})

    def test_works_without_pyyaml(self):
        with patch.dict(sys.modules, {"yaml": None}):
            front, body = parse_skill_text("---\nname: a\ndescription: 'quoted: still'\n---\nbody")
        self.assertEqual((front["description"], body), ("quoted: still", "body"))

    def test_no_front_matter_or_an_unterminated_one(self):
        self.assertEqual(parse_skill_text("just text"), ({}, "just text"))
        self.assertEqual(parse_skill_text("---\nname: a\nno end"), ({}, "---\nname: a\nno end"))
        self.assertEqual(parse_skill_text("﻿---\nname: a\ndescription: b\n---\nx")[0]["name"], "a")


class TestLoadSkill(TmpCase):
    def test_a_valid_skill(self):
        folder = self.skill(self.root, "release-notes", "Write release notes.", "Step 1.",
                            extra="license: MIT\nallowed-tools: Read Write\nmetadata:\n  version: '2'\n")
        loaded = load_skill(folder, "user")
        self.assertEqual((loaded.name, loaded.description, loaded.scope), ("release-notes", "Write release notes.", "user"))
        self.assertEqual((loaded.license, loaded.metadata, loaded.allowed_tools), ("MIT", {"version": "2"}, ("Read", "Write")))
        self.assertEqual(loaded.body(), "Step 1.")
        self.assertEqual(loaded.path, folder / SKILL_FILE)

    def test_a_multiline_description_is_flattened(self):
        folder = self.skill(self.root, "a", description=">-\n  line one\n  line two")
        self.assertEqual(load_skill(folder, "user").description, "line one line two")

    def test_invalid_skills_say_why(self):
        cases = {"no SKILL.md": lambda: self.root.joinpath("empty").mkdir() or self.root / "empty",
                 "missing name": lambda: self._raw("noname", "---\ndescription: d\n---\nx"),
                 "invalid name": lambda: self._raw("Bad_Name", "---\nname: Bad_Name\ndescription: d\n---\nx"),
                 "invalid name .a--b.": lambda: self._raw("a--b", "---\nname: a--b\ndescription: d\n---\nx"),
                 "invalid name .-a.": lambda: self._raw("-a", "---\nname: -a\ndescription: d\n---\nx"),
                 "at most 64 characters": lambda: self._raw("n" * 65, f"---\nname: {'n' * 65}\ndescription: d\n---\nx"),
                 "missing description": lambda: self._raw("nodesc", "---\nname: nodesc\n---\nx"),
                 "longer than": lambda: self._raw("long", f"---\nname: long\ndescription: {'d' * 1025}\n---\nx"),
                 "larger than": lambda: self._raw("big", "---\nname: big\ndescription: d\n---\n" + "x" * 300_000)}
        for reason, make in cases.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                load_skill(make(), "user")

    def _raw(self, dirname, text):
        folder = self.root / dirname
        folder.mkdir()
        (folder / SKILL_FILE).write_text(text)
        return folder

    def test_a_folder_name_that_differs_is_accepted_with_a_warning(self):
        folder = self.skill(self.root, "real-name", dirname="other-folder")
        with self.assertLogs("maki.skills", "WARNING") as logs:
            self.assertEqual(load_skill(folder, "user").name, "real-name")
        self.assertIn("other-folder", logs.output[0])

    def test_resources_are_listed_relative_and_without_hidden_files(self):
        folder = self.skill(self.root, "res")
        (folder / "references").mkdir()
        (folder / "references" / "guide.md").write_text("g")
        (folder / "scripts").mkdir()
        (folder / "scripts" / "run.sh").write_text("echo")
        (folder / ".hidden").write_text("h")
        (folder / "notes.txt").write_text("n")
        self.assertEqual(load_skill(folder, "user").resources(), ["notes.txt", "references/guide.md", "scripts/run.sh"])


class TestDiscovery(TmpCase):
    def setUp(self):
        super().setUp()
        self.home = self.root / "home"
        self.repo = self.root / "repo"
        self.cwd = self.repo / "services" / "api"
        self.cwd.mkdir(parents=True)
        (self.repo / ".git").mkdir()
        self.home.mkdir()

    def discover(self, **kw):
        kw.setdefault("cwd", self.cwd)
        kw.setdefault("home", self.home)
        return SkillRegistry.discover(**kw)

    def test_project_skills_need_trust(self):
        self.skill(self.cwd / ".agents" / "skills", "proj")
        self.skill(self.home / ".agents" / "skills", "mine")
        closed = self.discover()
        self.assertEqual(list(closed.skills), ["mine"])
        self.assertEqual([s.name for s in closed.untrusted], ["proj"])
        opened = self.discover(trust_project=True)
        self.assertEqual(sorted(opened.skills), ["mine", "proj"])
        self.assertEqual((opened.get("proj").scope, opened.get("mine").scope), ("project", "user"))
        self.assertEqual(opened.untrusted, [])

    def test_explicit_folders_are_trusted_and_come_first(self):
        extra = self.root / "extra"
        self.skill(extra, "tool")
        self.skill(self.home / ".claude" / "skills", "tool", description="user copy")
        registry = self.discover(paths=[extra])
        self.assertEqual(registry.get("tool").scope, "explicit")
        self.assertEqual([s[0] for s in registry.shadowed], ["tool"])
        self.discover(paths=[self.root / "does-not-exist"])  # a missing folder is not an error

    def test_an_explicit_path_may_be_one_skill_folder_itself(self):
        single = self.skill(self.root / "elsewhere", "solo")
        registry = self.discover(paths=[single])
        self.assertEqual(list(registry.skills), ["solo"])
        self.assertEqual(registry.get("solo").scope, "explicit")
        broken = self.root / "elsewhere" / "broken"
        broken.mkdir()
        (broken / SKILL_FILE).write_text("---\nname: broken\n---\nx")
        with self.assertLogs("maki.skills", "WARNING"):
            registry = self.discover(paths=[broken])
        self.assertEqual((len(registry.skills), len(registry.problems)), (0, 1))

    def test_precedence_nearer_then_folder_order_then_project_over_user(self):
        for base in (self.cwd, self.repo):
            self.skill(base / ".claude" / "skills", "dup", description=f"claude at {base.name}")
        self.skill(self.cwd / ".agents" / "skills", "dup", description="agents at api")
        self.skill(self.cwd / ".maki" / "skills", "dup", description="maki at api")
        self.skill(self.home / ".maki" / "skills", "dup", description="user")
        registry = self.discover(trust_project=True)
        self.assertEqual(registry.get("dup").description, "maki at api")
        self.assertEqual(len(registry.shadowed), 4)
        only_repo = self.root / "repo2"
        (only_repo / ".git").mkdir(parents=True)
        self.skill(only_repo / ".claude" / "skills", "near", description="repo root")
        (only_repo / "sub").mkdir()
        self.skill(only_repo / "sub" / ".claude" / "skills", "near", description="nearer")
        self.assertEqual(self.discover(cwd=only_repo / "sub", trust_project=True).get("near").description, "nearer")

    def test_it_walks_up_to_the_git_root_and_no_further(self):
        self.skill(self.repo / ".agents" / "skills", "at-root")
        self.skill(self.root / ".agents" / "skills", "above-root")
        names = set(self.discover(trust_project=True).skills)
        self.assertEqual(names, {"at-root"})

    def test_without_a_git_repo_only_the_working_directory_counts(self):
        loose = self.root / "loose" / "deep"
        loose.mkdir(parents=True)
        self.skill(loose / ".agents" / "skills", "here")
        self.skill(self.root / "loose" / ".agents" / "skills", "parent")
        self.assertEqual(list(self.discover(cwd=loose, trust_project=True).skills), ["here"])
        self.assertEqual(project_roots(loose), [loose])

    def test_the_home_folder_is_never_a_project(self):
        self.skill(self.home / ".agents" / "skills", "personal")
        registry = self.discover(cwd=self.home, trust_project=False)
        self.assertEqual(list(registry.skills), ["personal"])
        self.assertEqual(registry.untrusted, [])  # not reported as an untrusted project skill

    def test_broken_skills_are_reported_and_the_rest_still_load(self):
        base = self.home / ".agents" / "skills"
        self.skill(base, "good")
        (base / "broken").mkdir()
        (base / "broken" / SKILL_FILE).write_text("---\nname: broken\n---\nno description")
        (base / "not-a-skill").mkdir()
        (base / ".hidden").mkdir()
        (base / "stray-file.txt").write_text("x")
        with self.assertLogs("maki.skills", "WARNING"):
            registry = self.discover()
        self.assertEqual(list(registry.skills), ["good"])
        self.assertEqual(len(registry.problems), 1)
        self.assertIn("missing description", registry.problems[0][1])

    def test_catalog_size_is_validated(self):
        for bad in (0, 100, "big", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.discover(catalog_chars=bad)


class TestCatalog(TmpCase):
    def registry(self, count=3, description="Does a thing.", **kw):
        base = self.root / "s"
        for index in range(count):
            self.skill(base, f"skill-{index:02d}", description)
        return SkillRegistry.discover(paths=[base], cwd=self.root, home=self.root / "none", **kw)

    def test_lists_name_and_description_sorted_with_an_instruction(self):
        text = self.registry().catalog()
        self.assertIn("skills.load", text)
        self.assertEqual([line for line in text.splitlines() if line.startswith("- ")],
                         [f"- skill-0{i}: Does a thing." for i in range(3)])

    def test_empty_registry_has_no_catalog(self):
        self.assertEqual(SkillRegistry().catalog(), "")
        self.assertFalse(SkillRegistry())

    def test_long_descriptions_are_clipped(self):
        line = [x for x in self.registry(1, "word " * 200).catalog().splitlines() if x.startswith("- ")][0]
        self.assertLessEqual(len(line), 320)
        self.assertTrue(line.endswith("…"))

    def test_the_budget_is_kept_and_unlisted_skills_are_still_named(self):
        registry = self.registry(count=40, description="d" * 100, catalog_chars=1000)
        text = registry.catalog()
        self.assertLessEqual(len(text), 1300)  # budget plus the trailing names line
        self.assertIn("more skills, load by name:", text)
        listed = {x[2:].split(":")[0] for x in text.splitlines() if x.startswith("- ")}
        named = set(text.split("load by name: ")[1].rstrip(")").split(", "))
        self.assertEqual(listed | named, set(registry.skills))
        self.assertFalse(listed & named)

    def test_default_budget(self):
        self.assertEqual(self.registry().catalog_chars, DEFAULT_CATALOG_CHARS)


class TestSkillsPlugin(TmpCase):
    def setUp(self):
        super().setUp()
        self.folder = self.skill(self.root / "s", "demo", body="# Demo\nDo the thing.")
        (self.folder / "references").mkdir()
        (self.folder / "references" / "guide.md").write_text("the guide")
        (self.folder / "assets").mkdir()
        (self.folder / "assets" / "logo.bin").write_bytes(b"\xff\xfe\x00binary")
        self.plugin = SkillRegistry.discover(paths=[self.root / "s"], cwd=self.root, home=self.root / "none").plugin()

    def test_load_returns_the_body_and_lists_bundled_files(self):
        text = self.plugin.load("demo")
        self.assertIn("Do the thing.", text)
        self.assertIn("references/guide.md", text)
        self.assertIn("skills.read", text)
        self.assertNotIn("name: demo", text)  # the front matter is not part of the instructions

    def test_a_skill_without_files_has_no_file_footer(self):
        self.skill(self.root / "s", "plain")
        plugin = SkillRegistry.discover(paths=[self.root / "s"], cwd=self.root, home=self.root / "none").plugin()
        self.assertEqual(plugin.load("plain"), "Instructions here.")

    def test_unknown_skills_list_what_exists(self):
        with self.assertRaisesRegex(ValueError, "unknown skill 'nope'.*demo"):
            self.plugin.load("nope")
        with self.assertRaises(ValueError):
            self.plugin.read("nope", "x")

    def test_a_skill_without_instructions_says_so(self):
        self.skill(self.root / "s", "empty", body="")
        plugin = SkillRegistry.discover(paths=[self.root / "s"], cwd=self.root, home=self.root / "none").plugin()
        self.assertIn("no written instructions", plugin.load("empty"))

    def test_names_are_trimmed(self):
        self.assertIn("Do the thing.", self.plugin.load("  demo "))

    def test_read_returns_text_files(self):
        self.assertEqual(self.plugin.read("demo", "references/guide.md"), "the guide")

    def test_read_refuses_to_leave_the_skill_folder(self):
        outside = self.root / "secret.txt"
        outside.write_text("secret")
        os.symlink(outside, self.folder / "references" / "link.txt")
        for bad in ("../secret.txt", "references/../../secret.txt", str(outside), "references/link.txt", "/etc/passwd", ""):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                self.plugin.read("demo", bad)

    def test_read_refuses_skill_md_missing_files_directories_binaries_and_huge_files(self):
        for path, reason in ((SKILL_FILE, "skills.load"), ("nope.md", "no file"), ("references", "no file"),
                             ("assets/logo.bin", "not a text file")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, reason):
                self.plugin.read("demo", path)
        big = self.folder / "references" / "big.txt"
        big.write_text("x" * 1_100_000)
        with self.assertRaisesRegex(ValueError, "larger than"):
            self.plugin.read("demo", "references/big.txt")

    def test_long_instructions_are_clipped(self):
        self.skill(self.root / "s", "wordy", body="word " * 20_000)
        plugin = SkillRegistry.discover(paths=[self.root / "s"], cwd=self.root, home=self.root / "none").plugin()
        text = plugin.load("wordy")
        self.assertLess(len(text), 31_000)
        self.assertIn("more characters omitted", text)

    def test_the_plugin_only_exposes_read_only_methods(self):
        self.assertEqual(SkillsPlugin.ALLOWED_METHODS, ["load", "read"])
        self.assertIsNone(SkillsPlugin.DANGEROUS_METHODS)


class TestCoerce(TmpCase):
    def test_shapes(self):
        self.assertIsNone(coerce_registry(None))
        self.assertIsNone(coerce_registry(False))
        existing = SkillRegistry()
        self.assertIs(coerce_registry(existing), existing)
        base = self.root / "s"
        self.skill(base, "one")
        for spec in (str(base), [str(base)], (base,), {"paths": [str(base)]}):
            with self.subTest(spec=spec):
                self.assertIn("one", coerce_registry(spec).skills)
        self.assertIsInstance(coerce_registry(True), SkillRegistry)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            coerce_registry({"path": "x"})
        with self.assertRaises(TypeError):
            coerce_registry(42)

    def test_catalog_chars_default_applies_unless_given(self):
        self.assertEqual(coerce_registry(True, catalog_chars=900).catalog_chars, 900)
        self.assertEqual(coerce_registry({"catalog_chars": 700}, catalog_chars=900).catalog_chars, 700)


class RecordingBackend(LLMBackend):
    supports_native_tools = supports_typed_messages = supports_history = True

    def __init__(self, script):
        self.model, self.temperature, self.script = "r", 0.0, list(script)
        self.systems, self.rounds = [], []

    def request(self, prompt):
        return reply("plain")

    def chat(self, prompt, history=None, config=None, system=None, images=None):
        self.systems.append(system)
        return reply("plain")

    def to_tool_schemas(self, specs):
        return specs

    def chat_turn(self, messages, tool_specs=None, *, system=None, config=None, tool_choice=None):
        self.systems.append(system)
        self.rounds.append((list(messages), tool_specs))
        return self.script.pop(0)


class TestAgentWiring(TmpCase):
    def setUp(self):
        super().setUp()
        self.skill(self.root / "s", "release-notes", "Write release notes for a version.",
                   body="Always start with the codename PELICAN.")
        self.registry = SkillRegistry.discover(paths=[self.root / "s"], cwd=self.root, home=self.root / "none")

    def test_the_agent_gets_the_skills_plugin_and_a_catalog_only_for_tool_tasks(self):
        backend = RecordingBackend([reply("done")])
        agent = Agent("a", backend, skills=self.registry)
        self.assertIn("skills", agent.plugins)
        agent.execute_task("hello", use_plugins=True)
        self.assertIn("release-notes: Write release notes", backend.systems[0])
        self.assertNotIn("Skills you can load", agent._build_system_message())  # off unless tools are in play
        agent.execute_task("hello")
        self.assertNotIn("release-notes", backend.systems[-1])

    def test_no_skills_means_no_plugin_and_an_empty_registry_adds_nothing(self):
        self.assertNotIn("skills", Agent("a", RecordingBackend([])).plugins)
        self.assertNotIn("skills", Agent("a", RecordingBackend([]), skills=SkillRegistry()).plugins)

    def test_the_model_loads_a_skill_through_the_tool_loop(self):
        backend = RecordingBackend([reply("", [ToolCall("c1", "skills__load", {"name": "release-notes"})]),
                                    reply("The codename is PELICAN")])
        agent = Agent("a", backend, skills=self.registry)
        self.assertEqual(agent.execute_task("write release notes for 2.0", use_plugins=True), "The codename is PELICAN")
        tool_message = backend.rounds[1][0][-1]
        self.assertEqual((tool_message.role, tool_message.name), ("tool", "skills__load"))
        self.assertIn("PELICAN", tool_message.content)
        names = {spec["name"] for spec in backend.rounds[0][1]}
        self.assertTrue({"skills__load", "skills__read"} <= names)

    def test_a_wrong_skill_name_comes_back_as_an_error_the_model_can_fix(self):
        backend = RecordingBackend([reply("", [ToolCall("c1", "skills__load", {"name": "nope"})]), reply("ok")])
        Agent("a", backend, skills=self.registry).execute_task("go", use_plugins=True)
        self.assertIn("release-notes", backend.rounds[1][0][-1].content)  # the error names what exists

    def test_tool_schemas_describe_both_tools_with_their_parameters(self):
        agent = Agent("a", RecordingBackend([]), skills=self.registry)
        specs = {s["name"]: s for s in agent._build_tool_specs()}
        self.assertEqual(specs["skills__load"]["parameters"]["required"], ["name"])
        self.assertEqual(sorted(specs["skills__read"]["parameters"]["required"]), ["name", "path"])
        self.assertIn("skill", specs["skills__load"]["description"].lower())

    def test_works_on_the_text_directive_path_too(self):
        class TextOnly(LLMBackend):
            model, temperature = "t", 0.0

            def __init__(self):
                self.prompts = []

            def request(self, prompt):
                return self.chat(prompt)

            def chat(self, prompt, history=None, config=None, system=None, images=None):
                self.prompts.append((prompt, system))
                return reply("plain")
        backend = TextOnly()
        agent = Agent("a", backend, skills=self.registry)
        agent.execute_task("hi", use_plugins=True)
        prompt, system = backend.prompts[0]
        self.assertIn("release-notes", system)
        self.assertIn("skills", prompt)  # the TOOL: prompt section lists the plugin's methods


class TestConfig(TmpCase):
    def _load(self, body):
        from unittest.mock import MagicMock

        from maki.distributed.config_loader import load_agent_from_config
        path = self.root / "agent.yaml"
        path.write_text(body)
        with patch("maki.distributed.config_loader._build_backend",
                   return_value=MagicMock(spec=LLMBackend, model="m")):
            return load_agent_from_config(str(path))

    def test_skills_section(self):
        base = self.root / "s"
        self.skill(base, "one")
        agent = self._load(f"name: a\nskills:\n  paths: ['{base}']\n  catalog_chars: 900\n")
        self.assertIn("one", agent.skills.skills)
        self.assertEqual(agent.skills.catalog_chars, 900)
        self.assertIn("skills", agent.plugins)

    def test_no_section_and_bad_sections(self):
        self.assertIsNone(self._load("name: a\n").skills)
        self.assertIsNone(self._load("name: a\nskills: false\n").skills)
        for body in ("skills: {nope: 1}\n", "skills: 5\n"):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "Invalid 'skills' section"):
                self._load("name: a\n" + body)


if __name__ == "__main__":
    unittest.main()
