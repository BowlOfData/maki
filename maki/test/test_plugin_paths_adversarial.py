"""Adversarial paths against every file plugin: nothing outside ``base_dir`` may be read, written or listed.

A table of hostile paths (traversal, absolute paths, symlinks out, a sibling directory that shares the base's
prefix, NUL bytes, huge names, odd separators) is thrown at every method of FileReader, DirectoryReader,
FileWriter and JsonReader. Readers must never return content from outside; writers must leave the whole tree
outside ``base_dir`` byte-for-byte unchanged. Legitimate paths (including a symlink that stays inside) must
still work, so the guard is shown not to be a blanket refusal.

Every hostile path stays inside a temporary sandbox on purpose (no ``/`` or ``/etc``): if the guard were ever
broken, this test must not walk or write the real filesystem while it fails.
"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from maki.plugins.directory_reader.directory_reader import DirectoryReader
from maki.plugins.file_reader.file_reader import FileReader
from maki.plugins.file_writer.file_writer import FileWriter
from maki.plugins.json_reader.json_reader import JsonReader

SECRET = "TOP-SECRET-OUTSIDE"


def tree_state(root: Path) -> dict:
    """Every file and directory under *root* with a content hash, to prove nothing changed."""
    state = {}
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            state[rel] = ("link", os.readlink(path))
        elif path.is_dir():
            state[rel] = ("dir",)
        else:
            state[rel] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return state


class Sandbox(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.base = self.root / "base"
        self.outside = self.root / "outside"
        self.evil_sibling = self.root / "base-evil"       # shares the prefix "base": startswith() traps
        for folder in (self.base / "sub", self.outside, self.evil_sibling):
            folder.mkdir(parents=True)
        (self.base / "ok.txt").write_text("fine\nline two\n")
        (self.base / "sub" / "inner.txt").write_text("inner")
        (self.base / "data.json").write_text('[{"a": 1, "b": 2}]')
        (self.outside / "secret.txt").write_text(SECRET)
        (self.outside / "secret.json").write_text('[{"a": "' + SECRET + '"}]')
        (self.evil_sibling / "secret2.txt").write_text(SECRET)
        os.symlink(self.outside, self.base / "link_out")                       # directory link out
        os.symlink(self.outside / "secret.txt", self.base / "link_file")       # file link out
        os.symlink(self.base / "sub", self.base / "link_in")                   # link that stays inside
        os.symlink(self.base / "loop", self.base / "loop")                     # a link loop
        self.outside_before = tree_state(self.outside)
        self.evil_before = tree_state(self.evil_sibling)

    @property
    def hostile(self):
        out = str(self.outside / "secret.txt")
        return [
            "../outside/secret.txt", "sub/../../outside/secret.txt", "./../outside/secret.txt",
            "../../../../../../../../etc/passwd", out, str(self.outside), "link_out/secret.txt", "link_out",
            "link_file", "link_in/../../outside/secret.txt", "../base-evil/secret2.txt", "../base-evil",
            "sub/../..", "..", "../", str(self.root), str(self.root / "base-evil"), "loop", "loop/x",
            "~/../../outside/secret.txt",
            "a\x00b", "sub\x00/../../outside/secret.txt", "x" * 5000, "../" * 300 + "outside/secret.txt",
            "..\\outside\\secret.txt", "%2e%2e/outside/secret.txt", "....//outside/secret.txt", "sub//..//..//outside",
            "‥/outside/secret.txt", "．．/outside/secret.txt",
        ]

    def assert_outside_untouched(self):
        self.assertEqual(tree_state(self.outside), self.outside_before, "a file outside base_dir changed")
        self.assertEqual(tree_state(self.evil_sibling), self.evil_before, "the prefix-sharing sibling changed")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["base", "base-evil", "outside"],
                         "something new appeared next to base_dir")

    def leaked(self, result) -> bool:
        return SECRET in repr(result)


class TestReadersCannotEscape(Sandbox):
    def test_file_reader_methods(self):
        plugin = FileReader(base_dir=str(self.base))
        for path in self.hostile:
            for name in ("read_file", "read_file_as_lines", "get_file_info"):
                with self.subTest(method=name, path=path[:40]):
                    result = getattr(plugin, name)(path)
                    self.assertFalse(self.leaked(result), "secret content returned")
                    self.assertFalse(result["success"] and name != "get_file_info", result)
                    if name == "get_file_info":  # a link out must not even confirm what is there
                        self.assertNotEqual(result.get("size"), len(SECRET), result)

    def test_directory_reader_methods(self):
        plugin = DirectoryReader(base_dir=str(self.base))
        for path in self.hostile:
            for name in ("read_directory", "read_directory_as_text"):
                for recursive in (False, True):
                    with self.subTest(method=name, path=path[:40], recursive=recursive):
                        result = getattr(plugin, name)(path, recursive=recursive)
                        self.assertFalse(self.leaked(result), "secret content returned")
                        listed = " ".join(str(entry) for entry in result.get("files", []))
                        for outside in (str(self.outside), str(self.evil_sibling)):
                            self.assertNotIn(outside, listed, "a file outside base_dir was listed")
                        self.assertNotIn("secret.txt", listed)

    def test_a_recursive_listing_does_not_follow_a_link_out(self):
        plugin = DirectoryReader(base_dir=str(self.base))
        for name in ("read_directory", "read_directory_as_text"):
            with self.subTest(method=name):
                result = getattr(plugin, name)(".", recursive=True)
                self.assertFalse(self.leaked(result), "a symlinked directory leaked its files")

    def test_json_reader(self):
        plugin = JsonReader(base_dir=str(self.base))
        for path in self.hostile + ["link_out/secret.json", "../outside/secret.json"]:
            with self.subTest(path=path[:40]):
                self.assertFalse(self.leaked(plugin.read_json_fields(path, ["a"])))

    def test_non_string_paths_are_refused_cleanly(self):
        for plugin, call in ((FileReader(base_dir=str(self.base)), lambda p, x: p.read_file(x)),
                             (DirectoryReader(base_dir=str(self.base)), lambda p, x: p.read_directory(x)),
                             (FileWriter(base_dir=str(self.base)), lambda p, x: p.write_file(x, "c"))):
            for bad in (None, 5, [], {}, b"bytes", "", "   "):
                with self.subTest(plugin=type(plugin).__name__, bad=bad), self.assertRaises(ValueError):
                    call(plugin, bad)


class TestWritersCannotEscape(Sandbox):
    def test_file_writer_methods_leave_everything_outside_unchanged(self):
        plugin = FileWriter(base_dir=str(self.base))
        for path in self.hostile:
            for name, args in (("write_file", ("PWNED",)), ("append_to_file", ("PWNED",)),
                               ("write_file_lines", (["PWNED"],))):
                for create_dirs in (True, False):
                    with self.subTest(method=name, path=path[:40], create_dirs=create_dirs):
                        kwargs = {"create_dirs": create_dirs} if name != "append_to_file" else {}
                        try:
                            getattr(plugin, name)(path, *args, **kwargs)
                        except ValueError:
                            pass  # an input the API itself rejects (a NUL byte, say) is also a refusal
                        self.assert_outside_untouched()

    def test_new_directories_are_never_created_outside(self):
        plugin = FileWriter(base_dir=str(self.base))
        for path in ("../outside/new/dir/file.txt", "link_out/new/file.txt", "../newdir/file.txt",
                     "../base-evil/new/file.txt", str(self.root / "elsewhere" / "file.txt")):
            with self.subTest(path=path):
                result = plugin.write_file(path, "PWNED", create_dirs=True)
                self.assertFalse(result["success"], result)
                self.assert_outside_untouched()

    def test_the_symlink_target_is_not_overwritten_through_a_link(self):
        plugin = FileWriter(base_dir=str(self.base))
        for name in ("write_file", "append_to_file"):
            with self.subTest(method=name):
                self.assertFalse(getattr(plugin, name)("link_file", "PWNED")["success"])
                self.assertEqual((self.outside / "secret.txt").read_text(), SECRET)

    def test_a_prefix_sharing_sibling_is_not_inside(self):
        plugin = FileWriter(base_dir=str(self.base))
        self.assertFalse(plugin.write_file("../base-evil/planted.txt", "x")["success"])
        self.assertFalse((self.evil_sibling / "planted.txt").exists())


class TestLegitimatePathsStillWork(Sandbox):
    def test_reading_inside_the_base_including_through_an_inside_link(self):
        reader = FileReader(base_dir=str(self.base))
        for path, expected in (("ok.txt", "fine"), ("sub/inner.txt", "inner"), ("link_in/inner.txt", "inner"),
                               (str(self.base / "ok.txt"), "fine"), ("./sub/../ok.txt", "fine")):
            with self.subTest(path=path):
                result = reader.read_file(path)
                self.assertTrue(result["success"], result)
                self.assertIn(expected, result["content"])

    def test_listing_and_json_inside_the_base(self):
        listing = DirectoryReader(base_dir=str(self.base)).read_directory("sub")
        self.assertTrue(listing["success"], listing)
        self.assertIn("inner", repr(listing))
        self.assertTrue(JsonReader(base_dir=str(self.base)).read_json_fields("data.json", ["a"])["success"])

    def test_writing_inside_the_base(self):
        writer = FileWriter(base_dir=str(self.base))
        for path, call in (("new/deep/file.txt", lambda p: writer.write_file(p, "hello")),
                           ("sub/appended.txt", lambda p: writer.append_to_file(p, "more")),
                           ("lines.txt", lambda p: writer.write_file_lines(p, ["a", "b"]))):
            with self.subTest(path=path):
                self.assertTrue(call(path)["success"])
                self.assertTrue((self.base / path).exists())
        self.assert_outside_untouched()

    def test_base_dir_that_is_itself_a_symlink_still_confines(self):
        alias = self.root / "alias"
        os.symlink(self.base, alias)
        reader = FileReader(base_dir=str(alias))
        self.assertTrue(reader.read_file("ok.txt")["success"])
        self.assertFalse(self.leaked(reader.read_file("../outside/secret.txt")))
        self.assertFalse(self.leaked(reader.read_file("link_out/secret.txt")))


if __name__ == "__main__":
    unittest.main()
