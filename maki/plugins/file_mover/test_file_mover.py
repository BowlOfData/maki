"""FileMover: the tool surface (scope, names, dry run, journal, undo)."""
import os
import tempfile
import unittest
from datetime import datetime, timezone

from maki.plugins import get_plugin_class, list_plugins
from maki.plugins.file_mover import FileMover, register_plugin


def write(path, text="data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.realpath(self.tmp.name)
        self.work = os.path.join(self.d, "work")
        self.outside = os.path.join(self.d, "outside")
        os.makedirs(self.work)
        os.makedirs(self.outside)
        self.journal_dir = os.path.join(self.d, "journal")

    def tearDown(self):
        self.tmp.cleanup()

    def mover(self, **kwargs):
        kwargs.setdefault("journal_dir", self.journal_dir)
        kwargs.setdefault("base_dir", self.work)
        return FileMover(**kwargs)

    def w(self, *parts):
        return os.path.join(self.work, *parts)


class RegistrationTests(unittest.TestCase):
    def test_it_is_a_built_in_plugin_with_a_gated_tool_surface(self):
        self.assertIn("file_mover", list_plugins())
        cls = get_plugin_class("file_mover")
        self.assertIs(cls, FileMover)
        self.assertEqual(sorted(cls.ALLOWED_METHODS), ["list_moves", "move", "undo_last"])
        self.assertEqual(sorted(cls.DANGEROUS_METHODS), ["move", "undo_last"])
        self.assertIsInstance(register_plugin(None, base_dir=tempfile.gettempdir(),
                                              journal_dir=tempfile.gettempdir()), FileMover)

    def test_scope_options_are_constructor_only(self):
        import inspect
        for name in ("move", "undo_last", "list_moves"):
            params = set(inspect.signature(getattr(FileMover, name)).parameters) - {"self"}
            self.assertFalse(params & {"base_dir", "allowed_roots", "allow_any_path", "journal_dir"}, name)

    def test_conflicting_scope_options_are_rejected(self):
        with self.assertRaises(ValueError):
            FileMover(allowed_roots=["/x"], allow_any_path=True)


class MoveTests(Base):
    def test_move_renames_inside_base_dir_and_journals_it(self):
        write(self.w("a.txt"), "hello")
        result = self.mover().move("a.txt", "docs/b.txt", create_dirs=True)
        self.assertTrue(result["success"], result)
        self.assertEqual((result["kind"], result["method"], result["undoable"]), ("file", "rename", True))
        self.assertEqual(read(self.w("docs", "b.txt")), "hello")
        self.assertIn("journal_id", result)

    def test_dry_run_reports_the_plan_and_changes_nothing(self):
        write(self.w("a.txt"))
        mover = self.mover()
        result = mover.move("a.txt", "b.txt", dry_run=True)
        self.assertTrue(result["success"] and result["dry_run"])
        self.assertEqual(result["plan"]["kind"], "file")
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertFalse(os.path.exists(self.w("b.txt")))
        self.assertEqual(mover.list_moves()["count"], 0)
        write(self.w("b.txt"))
        failing = mover.move("a.txt", "b.txt", dry_run=True)
        self.assertFalse(failing["success"])
        self.assertEqual(failing["plan"]["reason"], "conflict")

    def test_conflict_is_refused_and_overwrite_replaces_a_file_irreversibly(self):
        write(self.w("a.txt"), "new")
        write(self.w("b.txt"), "old")
        mover = self.mover()
        refused = mover.move("a.txt", "b.txt")
        self.assertFalse(refused["success"])
        self.assertEqual(refused["reason"], "conflict")
        result = mover.move("a.txt", "b.txt", overwrite=True)
        self.assertTrue(result["success"])
        self.assertFalse(result["undoable"])
        self.assertEqual(read(self.w("b.txt")), "new")
        self.assertIn("cannot be restored", mover.undo_last()["error"])

    def test_scope_is_enforced_for_both_paths(self):
        write(self.w("a.txt"))
        write(os.path.join(self.outside, "x.txt"))
        mover = self.mover()
        for src, dst in (("a.txt", os.path.join(self.outside, "a.txt")), ("../outside/x.txt", "x.txt"),
                         (os.path.join(self.outside, "x.txt"), "x.txt")):
            result = mover.move(src, dst)
            self.assertFalse(result["success"], (src, dst))
            self.assertIn("outside the allowed", result["error"])
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.outside, "x.txt")))

    def test_operator_can_widen_scope_with_allowed_roots_or_any_path(self):
        write(os.path.join(self.outside, "x.txt"))
        self.assertTrue(self.mover(allowed_roots=[self.outside]).move(
            os.path.join(self.outside, "x.txt"), "x.txt")["success"])
        write(os.path.join(self.outside, "y.txt"))
        self.assertTrue(self.mover(allow_any_path=True).move(
            os.path.join(self.outside, "y.txt"), os.path.join(self.outside, "z.txt"))["success"])

    def test_destination_names_follow_the_windows_rules_when_strict(self):
        write(self.w("a.txt"))
        lenient = self.mover(strict_names=False)
        if os.name != "nt":
            self.assertTrue(lenient.move("a.txt", "aux.txt")["success"])
            os.rename(self.w("aux.txt"), self.w("a.txt"))
        strict = self.mover(strict_names=True)
        for bad in ("aux.txt", "dir/NUL", "name.", "a:b.txt", "q?.txt"):
            result = strict.move("a.txt", bad)
            self.assertFalse(result["success"], bad)
            self.assertIn("Invalid destination name", result["error"])
        self.assertTrue(os.path.exists(self.w("a.txt")))

    def test_trailing_separator_and_bad_arguments_are_refused_with_a_message(self):
        write(self.w("a.txt"))
        os.makedirs(self.w("folder"))
        mover = self.mover()
        self.assertEqual(mover.move("a.txt", "folder/")["reason"], "dst_is_folder")
        self.assertEqual(mover.move("a.txt", "folder")["reason"], "conflict")
        for args in (("", "x"), ("a.txt", ""), (None, "x"), ("a.txt", 5)):
            self.assertFalse(mover.move(*args)["success"], args)
        for flag in ("overwrite", "dry_run", "create_dirs"):
            result = mover.move("a.txt", "b.txt", **{flag: "yes"})
            self.assertFalse(result["success"])
            self.assertIn(flag, result["error"])
        self.assertTrue(os.path.exists(self.w("a.txt")))

    def test_flags_may_arrive_as_true_false_strings_as_tool_schemas_describe_them(self):
        write(self.w("a.txt"))
        mover = self.mover()
        preview = mover.move("a.txt", "b.txt", dry_run="true", overwrite="False", create_dirs="FALSE")
        self.assertTrue(preview["success"] and preview["dry_run"])
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertTrue(mover.move("a.txt", "n/b.txt", create_dirs="true")["success"])

    def test_a_missing_journal_directory_does_not_hide_a_completed_move(self):
        write(self.w("a.txt"))
        blocker = os.path.join(self.d, "blocker")
        write(blocker)  # a file where the journal directory should be
        result = self.mover(journal_dir=os.path.join(blocker, "journal")).move("a.txt", "b.txt")
        self.assertTrue(result["success"])
        self.assertFalse(result["undoable"])
        self.assertIn("not journaled", result["warning"])
        self.assertTrue(os.path.exists(self.w("b.txt")))


class SharedJournalTests(Base):
    """Two agents with different scopes can share one journal directory without seeing or undoing each other."""

    def setUp(self):
        super().setUp()
        self.a = os.path.join(self.d, "projA")
        self.b = os.path.join(self.d, "projB")
        for folder, name in ((self.a, "a.txt"), (self.b, "b.txt")):
            write(os.path.join(folder, name))
        self.ma = FileMover(base_dir=self.a, journal_dir=self.journal_dir)
        self.mb = FileMover(base_dir=self.b, journal_dir=self.journal_dir)

    def test_list_moves_shows_only_moves_inside_the_scope(self):
        self.mb.move("b.txt", "b2.txt")
        self.ma.move("a.txt", "a2.txt")
        listed = self.mb.list_moves()["moves"]
        self.assertEqual([m["src"] for m in listed], [os.path.join(self.b, "b.txt")])
        everything = FileMover(base_dir=self.a, allow_any_path=True, journal_dir=self.journal_dir).list_moves()
        self.assertEqual(everything["count"], 2)

    def test_undo_last_undoes_this_scopes_latest_move_even_if_another_scope_moved_later(self):
        self.mb.move("b.txt", "b2.txt")
        self.ma.move("a.txt", "a2.txt")
        result = self.mb.undo_last()
        self.assertTrue(result["success"], result)
        self.assertTrue(os.path.exists(os.path.join(self.b, "b.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.a, "a2.txt")))  # A's move is untouched
        self.assertTrue(self.ma.undo_last()["success"])  # and A can still undo its own

    def test_a_scope_with_no_moves_of_its_own_has_nothing_to_undo(self):
        self.ma.move("a.txt", "a2.txt")
        self.assertIn("nothing to undo", self.mb.undo_last()["error"])
        self.assertTrue(os.path.exists(os.path.join(self.a, "a2.txt")))


class UndoAndListTests(Base):
    def test_undo_last_reverses_the_latest_move_and_list_moves_shows_both(self):
        write(self.w("a.txt"), "x")
        mover = self.mover()
        mover.move("a.txt", "b.txt")
        result = mover.undo_last()
        self.assertTrue(result["success"], result)
        self.assertEqual(read(self.w("a.txt")), "x")
        listing = mover.list_moves(limit=5)
        self.assertEqual([m["action"] for m in listing["moves"]], ["undo", "move"])
        self.assertFalse(mover.undo_last()["success"])  # an undo is not itself undone

    def test_undo_refuses_when_the_item_changed(self):
        write(self.w("a.txt"), "x")
        mover = self.mover()
        mover.move("a.txt", "b.txt")
        write(self.w("b.txt"), "edited afterwards")
        result = mover.undo_last()
        self.assertFalse(result["success"])
        self.assertIn("changed", result["error"])

    def test_list_moves_validates_the_limit(self):
        mover = self.mover()
        for bad in (0, -1, 101, True, "0", "abc", 1.5):
            self.assertFalse(mover.list_moves(bad)["success"], bad)
        self.assertEqual(mover.list_moves()["count"], 0)
        self.assertTrue(mover.list_moves("5")["success"])  # a model may send the number as a string

    def test_the_journal_rotates_by_month(self):
        clock = [datetime(2026, 1, 5, tzinfo=timezone.utc)]
        mover = self.mover(retention_months=2, clock=lambda: clock[0])
        write(self.w("a.txt"))
        mover.move("a.txt", "b.txt")
        clock[0] = datetime(2026, 3, 5, tzinfo=timezone.utc)
        mover.move("b.txt", "c.txt")
        self.assertEqual(sorted(os.listdir(self.journal_dir)), ["2026-03.jsonl"])


if __name__ == "__main__":
    unittest.main()
