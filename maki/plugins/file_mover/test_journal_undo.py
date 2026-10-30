"""file_mover: journal (rotation, fingerprints) and undo."""
import os
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from maki.plugins.file_mover import _engine
from maki.plugins.file_mover._engine import execute_move, plan_move
from maki.plugins.file_mover._journal import Journal, fingerprint
from maki.plugins.file_mover._scope import Scope
from maki.plugins.file_mover._undo import UndoRefused, undo_latest


def write(path, text="data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class Clock:
    def __init__(self, year=2026, month=10, day=15):
        self.now = datetime(year, month, day, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.realpath(self.tmp.name)
        self.jdir = os.path.join(self.d, "journal")
        self.work = os.path.join(self.d, "work")
        os.makedirs(self.work)

    def tearDown(self):
        self.tmp.cleanup()

    def w(self, *parts):
        return os.path.join(self.work, *parts)

    def files(self, journal):
        return journal._month_files()


class JournalTests(Base):
    def test_entries_go_to_a_file_per_month_newest_first(self):
        clock = Clock(2026, 9, 30)
        journal = Journal(self.jdir, clock=clock)
        journal.record("move", src="a", dst="b")
        clock.now = datetime(2026, 10, 1, tzinfo=timezone.utc)
        journal.record("move", src="c", dst="d")
        self.assertEqual(self.files(journal), ["2026-09.jsonl", "2026-10.jsonl"])
        self.assertEqual([e["src"] for e in journal.entries()], ["c", "a"])
        self.assertEqual(journal.last_record()["src"], "c")
        self.assertEqual(len(journal.entries(limit=1)), 1)

    def test_starting_a_new_month_deletes_files_older_than_the_retention(self):
        clock = Clock(2025, 1, 10)
        journal = Journal(self.jdir, retention_months=3, clock=clock)
        for month in (1, 2, 3):
            clock.now = datetime(2025, month, 10, tzinfo=timezone.utc)
            journal.record("move", src=str(month), dst="x")
        self.assertEqual(len(self.files(journal)), 3)
        clock.now = datetime(2025, 4, 1, tzinfo=timezone.utc)
        journal.record("move", src="4", dst="x")
        self.assertEqual(self.files(journal), ["2025-02.jsonl", "2025-03.jsonl", "2025-04.jsonl"])

    def test_retention_across_a_year_boundary_and_none_keeps_everything(self):
        clock = Clock(2025, 11, 5)
        journal = Journal(self.jdir, retention_months=2, clock=clock)
        journal.record("move", src="nov", dst="x")
        clock.now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        journal.record("move", src="jan", dst="x")
        self.assertEqual(self.files(journal), ["2026-01.jsonl"])  # 2025-11 is two months back: gone
        keep = Journal(os.path.join(self.d, "keep"), retention_months=None, clock=Clock(2020, 1, 1))
        keep.record("move", src="old", dst="x")
        keep.clock = Clock(2030, 1, 1)
        keep.record("move", src="new", dst="x")
        self.assertEqual(len(self.files(keep)), 2)

    def test_retention_must_be_a_positive_integer_or_none(self):
        for bad in (0, -1, True, 1.5, "12"):
            with self.assertRaises(ValueError):
                Journal(self.jdir, retention_months=bad)

    def test_bad_lines_are_skipped_and_an_empty_journal_has_no_last_record(self):
        journal = Journal(self.jdir, clock=Clock())
        self.assertIsNone(journal.last_record())
        journal.record("move", src="a", dst="b")
        with open(os.path.join(self.jdir, "2026-10.jsonl"), "a") as f:
            f.write("not json\n[1, 2]\n")
        self.assertEqual(journal.last_record()["src"], "a")

    def test_concurrent_appends_keep_every_line_intact(self):
        journal = Journal(self.jdir, clock=Clock())

        def burst(i):
            for n in range(25):
                journal.record("move", src=f"{i}-{n}", dst="x")

        threads = [threading.Thread(target=burst, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(journal.entries()), 200)

    def test_fingerprints_change_when_the_item_changes(self):
        write(self.w("f.txt"), "one")
        before = fingerprint(self.w("f.txt"))
        self.assertEqual(before["kind"], "file")
        write(self.w("f.txt"), "two!")
        self.assertNotEqual(before, fingerprint(self.w("f.txt")))
        write(self.w("tree", "a.txt"))
        tree = fingerprint(self.w("tree"))
        self.assertEqual((tree["kind"], tree["entries"]), ("dir", 1))
        write(self.w("tree", "sub", "b.txt"))
        self.assertNotEqual(tree, fingerprint(self.w("tree")))
        self.assertEqual(fingerprint(self.w("tree")), fingerprint(self.w("tree")))  # stable


class UndoTests(Base):
    def setUp(self):
        super().setUp()
        self.scope = Scope(base_dir=self.work)
        self.journal = Journal(self.jdir, clock=Clock())

    def move(self, src, dst, **kwargs):
        plan = plan_move(src, dst, **kwargs)
        result = execute_move(plan)
        return self.journal.record("move", src=src, dst=dst, kind=plan.kind, method=result["method"],
                                   irreversible=plan.will_overwrite, fingerprint=fingerprint(dst))

    def test_undo_moves_the_latest_item_back_and_records_the_undo(self):
        write(self.w("a.txt"), "x")
        self.move(self.w("a.txt"), self.w("b.txt"))
        result = undo_latest(self.journal, self.scope)
        self.assertEqual(result["restored"], self.w("a.txt"))
        self.assertTrue(os.path.exists(self.w("a.txt")))
        self.assertFalse(os.path.exists(self.w("b.txt")))
        self.assertEqual(self.journal.last_record()["action"], "undo")

    def test_a_second_undo_in_a_row_is_refused(self):
        write(self.w("a.txt"))
        self.move(self.w("a.txt"), self.w("b.txt"))
        undo_latest(self.journal, self.scope)
        with self.assertRaisesRegex(UndoRefused, "is an undo"):
            undo_latest(self.journal, self.scope)

    def test_only_the_latest_move_is_undone(self):
        write(self.w("a.txt"))
        write(self.w("c.txt"))
        self.move(self.w("a.txt"), self.w("b.txt"))
        self.move(self.w("c.txt"), self.w("d.txt"))
        undo_latest(self.journal, self.scope)
        self.assertTrue(os.path.exists(self.w("c.txt")))
        self.assertTrue(os.path.exists(self.w("b.txt")))  # the earlier move stays

    def test_directory_move_is_undone_and_missing_original_folder_is_recreated(self):
        write(self.w("old", "tree", "x.txt"))
        self.move(self.w("old", "tree"), self.w("tree"))
        os.rmdir(self.w("old"))
        undo_latest(self.journal, self.scope)
        self.assertTrue(os.path.exists(self.w("old", "tree", "x.txt")))

    def test_refusals(self):
        with self.assertRaisesRegex(UndoRefused, "nothing to undo"):
            undo_latest(self.journal, self.scope)

        write(self.w("a.txt"), "x")
        self.move(self.w("a.txt"), self.w("b.txt"))
        write(self.w("b.txt"), "changed since the move")
        with self.assertRaisesRegex(UndoRefused, "changed"):
            undo_latest(self.journal, self.scope)
        self.assertTrue(os.path.exists(self.w("b.txt")))  # untouched

        os.remove(self.w("b.txt"))
        with self.assertRaisesRegex(UndoRefused, "no longer at"):
            undo_latest(self.journal, self.scope)

    def test_an_overwriting_move_cannot_be_undone(self):
        write(self.w("a.txt"), "new")
        write(self.w("b.txt"), "old")
        self.move(self.w("a.txt"), self.w("b.txt"), overwrite=True)
        with self.assertRaisesRegex(UndoRefused, "cannot be restored"):
            undo_latest(self.journal, self.scope)

    def test_something_now_in_the_original_place_blocks_the_undo(self):
        write(self.w("a.txt"), "x")
        self.move(self.w("a.txt"), self.w("b.txt"))
        write(self.w("a.txt"), "newcomer")
        with self.assertRaisesRegex(UndoRefused, "already exists"):
            undo_latest(self.journal, self.scope)
        self.assertEqual(open(self.w("a.txt")).read(), "newcomer")

    def test_undo_obeys_the_scope(self):
        outside = os.path.join(self.d, "outside")
        os.makedirs(outside)
        write(os.path.join(outside, "a.txt"))
        wide = Scope(base_dir=self.work, allow_any_path=True)
        plan = plan_move(os.path.join(outside, "a.txt"), self.w("a.txt"))
        execute_move(plan)
        self.journal.record("move", src=os.path.join(outside, "a.txt"), dst=self.w("a.txt"), kind="file",
                            method="rename", irreversible=False, fingerprint=fingerprint(self.w("a.txt")))
        with self.assertRaisesRegex(UndoRefused, "nothing to undo"):
            undo_latest(self.journal, self.scope)  # narrow scope: that move is not this scope's own
        undo_latest(self.journal, wide)  # the operator's wider scope allows it
        self.assertTrue(os.path.exists(os.path.join(outside, "a.txt")))

    def test_cross_volume_move_can_be_undone(self):
        write(self.w("a.txt"), "payload")
        real = _engine._rename_direct
        calls = {"n": 0}

        def first_call_crosses(src, dst, overwrite):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(18, "cross-device")
            return real(src, dst, overwrite)

        with patch.object(_engine, "_rename_direct", side_effect=first_call_crosses):
            entry = self.move(self.w("a.txt"), self.w("b.txt"))
        self.assertEqual(entry["method"], "copy")
        undo_latest(self.journal, self.scope)
        self.assertEqual(open(self.w("a.txt")).read(), "payload")


if __name__ == "__main__":
    unittest.main()
