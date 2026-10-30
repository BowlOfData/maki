"""file_mover: portable names and scope."""
import os
import tempfile
import unittest

from maki.plugins.file_mover._names import check_name, check_path
from maki.plugins.file_mover._scope import Scope, ScopeError


class NameTests(unittest.TestCase):
    def test_reserved_names_with_and_without_extension_and_any_case(self):
        for name in ("CON", "con", "NUL.txt", "aux.tar.gz", "COM1", "lpt9.log", "Con .txt"):
            self.assertIsNotNone(check_name(name, windows_rules=True), name)
        for name in ("CONSOLE", "com10", "aux_file", "nul-x", "a.con"):
            self.assertIsNone(check_name(name, windows_rules=True), name)

    def test_windows_forbidden_characters_trailing_dot_and_space(self):
        for name in ("a<b", "a>b", 'a"b', "a:b", "a|b", "a?b", "a*b", "a\\b", "a\x01b", "name.", "name "):
            self.assertIsNotNone(check_name(name, windows_rules=True), repr(name))
        self.assertIsNone(check_name("fine-name_1.txt", windows_rules=True))

    def test_without_windows_rules_only_universal_limits_apply(self):
        for name in ("CON", "aux.txt", "a:b", "name.", "name ", "a*b"):
            self.assertIsNone(check_name(name, windows_rules=False), repr(name))
        self.assertIsNotNone(check_name("a\x00b", windows_rules=False))
        self.assertIsNotNone(check_name("x" * 256, windows_rules=False))
        self.assertIsNone(check_name("x" * 255, windows_rules=False))

    def test_check_path_checks_every_component_but_not_the_anchor_or_dots(self):
        self.assertIsNotNone(check_path("ok/aux/file.txt", windows_rules=True))
        self.assertIsNotNone(check_path("ok/fine/file.", windows_rules=True))
        self.assertIsNone(check_path("ok/../fine/./file.txt", windows_rules=True))
        self.assertIsNone(check_path("/abs/path/file.txt", windows_rules=True))
        self.assertIsNone(check_path("ok/aux/file.txt", windows_rules=False))


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        self.base = os.path.join(self.root, "base")
        self.other = os.path.join(self.root, "other")
        self.outside = os.path.join(self.root, "outside")
        for d in (self.base, self.other, self.outside):
            os.makedirs(d)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_scope_is_base_dir_only(self):
        scope = Scope(base_dir=self.base)
        self.assertEqual(scope.require("sub/file.txt"), os.path.join(self.base, "sub", "file.txt"))
        self.assertEqual(scope.require(self.base), self.base)
        for bad in ("../outside/x", self.outside, os.path.join(self.base, "..", "outside"), "/"):
            with self.assertRaises(ScopeError, msg=bad):
                scope.require(bad)

    def test_a_sibling_with_the_same_prefix_is_not_inside(self):
        sibling = self.base + "-evil"
        os.makedirs(sibling)
        with self.assertRaises(ScopeError):
            Scope(base_dir=self.base).require(sibling)

    def test_allowed_roots_extend_the_base_dir(self):
        scope = Scope(base_dir=self.base, allowed_roots=[self.other])
        scope.require(os.path.join(self.other, "x"))
        scope.require("y")
        with self.assertRaises(ScopeError):
            scope.require(self.outside)

    def test_any_path_allows_everything_and_cannot_mix_with_roots(self):
        scope = Scope(base_dir=self.base, allow_any_path=True)
        scope.require(self.outside)
        scope.require("/")
        with self.assertRaises(ValueError):
            Scope(base_dir=self.base, allowed_roots=[self.other], allow_any_path=True)

    def test_symlink_pointing_out_is_followed_for_the_target_but_located_by_the_link(self):
        link = os.path.join(self.base, "link")
        try:
            os.symlink(self.outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        scope = Scope(base_dir=self.base)
        with self.assertRaises(ScopeError):
            scope.require("link/file.txt")  # a path through the link leaves the scope
        self.assertEqual(scope.require("link", follow_final=False), link)  # the link itself is inside

    def test_destination_that_does_not_exist_yet_is_checked_by_its_nearest_parent(self):
        scope = Scope(base_dir=self.base)
        scope.require("a/b/c/new.txt")
        with self.assertRaises(ScopeError):
            scope.require(os.path.join(self.outside, "a", "b", "new.txt"))

    def test_case_insensitive_aliases_of_the_root_are_recognised(self):
        probe = os.path.join(self.root, "CaseProbe")
        os.makedirs(probe)
        if not os.path.exists(os.path.join(self.root, "caseprobe")):
            self.skipTest("file system is case-sensitive")
        scope = Scope(base_dir=probe)
        scope.require(os.path.join(self.root, "caseprobe", "x.txt"))  # same directory, different case


if __name__ == "__main__":
    unittest.main()
