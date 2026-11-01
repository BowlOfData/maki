"""file_mover engine: planning, local execution and the cross-volume fallback."""
import errno
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from maki.plugins.file_mover import _engine
from maki.plugins.file_mover._engine import MoveError, execute_move, plan_move


def write(path, text="data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


def case_insensitive(folder):
    probe = os.path.join(folder, "CaseProbe")
    write(probe)
    try:
        return os.path.exists(os.path.join(folder, "caseprobe"))
    finally:
        os.remove(probe)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.realpath(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, *parts):
        return os.path.join(self.d, *parts)


class PlanTests(Base):
    def test_plain_file_move_is_planned_as_a_rename(self):
        write(self.p("a.txt"))
        plan = plan_move(self.p("a.txt"), self.p("b.txt"))
        self.assertTrue(plan.ok)
        self.assertEqual((plan.kind, plan.same_volume, plan.case_only, plan.will_overwrite), ("file", True, False, False))
        self.assertIn("rename within the volume", plan.steps)
        self.assertTrue(os.path.exists(self.p("a.txt")))  # planning changes nothing

    def test_errors_carry_a_reason(self):
        write(self.p("a.txt"))
        write(self.p("b.txt"))
        os.makedirs(self.p("dir", "inner"))
        cases = {
            "missing_source": plan_move(self.p("nope"), self.p("x")),
            "same_path": plan_move(self.p("a.txt"), self.p("a.txt")),
            "conflict": plan_move(self.p("a.txt"), self.p("b.txt")),
            "into_itself": plan_move(self.p("dir"), self.p("dir", "inner", "dir")),
            "dst_is_folder": plan_move(self.p("a.txt"), self.p("sub"), dst_trailing_separator=True),
            "missing_parent": plan_move(self.p("a.txt"), self.p("no", "such", "a.txt")),
            "dir_overwrite": plan_move(self.p("a.txt"), self.p("dir"), overwrite=True),
            "kind_mismatch": plan_move(self.p("dir"), self.p("b.txt"), overwrite=True),
        }
        for reason, plan in cases.items():
            self.assertFalse(plan.ok, reason)
            self.assertEqual(plan.reason, reason)
            self.assertTrue(plan.error)

    def test_overwrite_file_over_file_is_planned_and_flagged(self):
        write(self.p("a.txt"))
        write(self.p("b.txt"))
        plan = plan_move(self.p("a.txt"), self.p("b.txt"), overwrite=True)
        self.assertTrue(plan.ok)
        self.assertTrue(plan.will_overwrite)
        self.assertTrue(any("cannot be undone" in s for s in plan.steps))

    def test_missing_parent_is_allowed_with_create_dirs(self):
        write(self.p("a.txt"))
        plan = plan_move(self.p("a.txt"), self.p("new", "deep", "a.txt"), create_dirs=True)
        self.assertTrue(plan.ok)
        self.assertIn("create the missing destination folders", plan.steps)

    def test_a_hard_link_is_an_existing_destination_not_a_case_rename(self):
        write(self.p("a.txt"))
        try:
            os.link(self.p("a.txt"), self.p("b.txt"))
        except (OSError, NotImplementedError):
            self.skipTest("hard links not available")
        plan = plan_move(self.p("a.txt"), self.p("b.txt"))
        self.assertEqual(plan.reason, "conflict")

    def test_symlink_source_is_planned_as_the_link(self):
        write(self.p("target.txt"))
        try:
            os.symlink(self.p("target.txt"), self.p("ln"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        plan = plan_move(self.p("ln"), self.p("target.txt"), overwrite=True)  # destination is the link's own target
        self.assertEqual(plan.kind, "symlink")
        self.assertTrue(plan.ok)

    def test_case_only_rename_is_detected_on_a_case_insensitive_file_system(self):
        if not case_insensitive(self.d):
            self.skipTest("file system is case-sensitive")
        write(self.p("a.txt"))
        plan = plan_move(self.p("a.txt"), self.p("A.txt"))
        self.assertTrue(plan.ok)
        self.assertTrue(plan.case_only)

    def test_on_a_case_sensitive_file_system_different_case_is_a_different_file(self):
        if case_insensitive(self.d):
            self.skipTest("file system is case-insensitive")
        write(self.p("a.txt"))
        write(self.p("A.txt"))
        self.assertEqual(plan_move(self.p("a.txt"), self.p("A.txt")).reason, "conflict")


class LocalExecuteTests(Base):
    def test_file_and_directory_moves(self):
        write(self.p("a.txt"), "hello")
        write(self.p("tree", "x", "y.txt"), "deep")
        self.assertEqual(execute_move(plan_move(self.p("a.txt"), self.p("b.txt"))), {"method": "rename"})
        self.assertEqual(read(self.p("b.txt")), "hello")
        self.assertFalse(os.path.exists(self.p("a.txt")))
        execute_move(plan_move(self.p("tree"), self.p("moved")))
        self.assertEqual(read(self.p("moved", "x", "y.txt")), "deep")
        self.assertFalse(os.path.exists(self.p("tree")))

    def test_overwrite_replaces_the_file(self):
        write(self.p("a.txt"), "new")
        write(self.p("b.txt"), "old")
        execute_move(plan_move(self.p("a.txt"), self.p("b.txt"), overwrite=True))
        self.assertEqual(read(self.p("b.txt")), "new")
        self.assertFalse(os.path.exists(self.p("a.txt")))

    def test_create_dirs_makes_the_destination_folders(self):
        write(self.p("a.txt"))
        execute_move(plan_move(self.p("a.txt"), self.p("n", "m", "a.txt"), create_dirs=True))
        self.assertTrue(os.path.exists(self.p("n", "m", "a.txt")))

    def test_symlink_moves_as_a_link_even_when_dangling(self):
        try:
            os.symlink(self.p("missing-target"), self.p("ln"))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available")
        execute_move(plan_move(self.p("ln"), self.p("ln2")))
        self.assertTrue(os.path.islink(self.p("ln2")))
        # Windows may report the target with a \\?\ prefix
        self.assertEqual(os.readlink(self.p("ln2")).removeprefix("\\\\?\\"), self.p("missing-target"))
        self.assertFalse(os.path.lexists(self.p("ln")))

    def test_a_failure_leaves_everything_as_it_was(self):
        write(self.p("a.txt"), "keep")
        plan = plan_move(self.p("a.txt"), self.p("b.txt"))
        with patch.object(_engine, "_rename_direct", side_effect=PermissionError(errno.EACCES, "denied")):
            with self.assertRaises(MoveError) as ctx:
                execute_move(plan)
        self.assertFalse(ctx.exception.partial)
        self.assertEqual(read(self.p("a.txt")), "keep")
        self.assertFalse(os.path.exists(self.p("b.txt")))

    def test_a_destination_that_appears_after_planning_is_not_clobbered(self):
        if os.name == "nt":
            self.skipTest("Windows rename already refuses an existing destination")
        write(self.p("a.txt"), "mine")
        plan = plan_move(self.p("a.txt"), self.p("b.txt"))
        write(self.p("b.txt"), "theirs")
        with self.assertRaises(MoveError):
            execute_move(plan)
        self.assertEqual(read(self.p("b.txt")), "theirs")
        self.assertEqual(read(self.p("a.txt")), "mine")

    def test_case_only_rename_goes_through_a_temporary_name_and_rolls_back(self):
        write(self.p("a.txt"), "x")
        plan = _engine.MovePlan(src=self.p("a.txt"), dst=self.p("A.txt"), kind="file", case_only=True)
        real_rename = os.rename
        calls = []

        def flaky(src, dst):
            calls.append((os.path.basename(src), os.path.basename(dst)))
            if len(calls) == 2:
                raise PermissionError(errno.EACCES, "denied")
            return real_rename(src, dst)

        with patch.object(_engine.os, "rename", side_effect=flaky):
            with self.assertRaises(MoveError):
                execute_move(plan)
        self.assertEqual(read(self.p("a.txt")), "x")  # restored
        self.assertEqual(len(calls), 3)

    def test_real_case_only_rename(self):
        if not case_insensitive(self.d):
            self.skipTest("file system is case-sensitive")
        write(self.p("report.txt"), "r")
        execute_move(plan_move(self.p("report.txt"), self.p("Report.txt")))
        self.assertEqual([n for n in os.listdir(self.d)], ["Report.txt"])
        self.assertEqual(read(self.p("Report.txt")), "r")

    def test_temporary_names_stay_within_the_file_name_limit(self):
        for name in ("x" * 255, "y" * 240, "short.txt"):
            temp = os.path.basename(_engine._temp_sibling(os.path.join(self.d, name)))
            self.assertLessEqual(len(temp), 255, name)
        self.assertNotEqual(_engine._temp_sibling(self.p("a")), _engine._temp_sibling(self.p("a")))

    def test_folders_created_for_a_move_that_then_fails_are_removed_again(self):
        write(self.p("a.txt"), "keep")
        os.makedirs(self.p("existing"))
        plan = plan_move(self.p("a.txt"), self.p("existing", "new", "deep", "a.txt"), create_dirs=True)
        with patch.object(_engine, "_rename_direct", side_effect=PermissionError(errno.EACCES, "denied")):
            with self.assertRaises(MoveError):
                execute_move(plan)
        self.assertTrue(os.path.isdir(self.p("existing")))  # was there before
        self.assertFalse(os.path.exists(self.p("existing", "new")))
        self.assertEqual(read(self.p("a.txt")), "keep")

    def test_executing_a_failed_plan_raises(self):
        with self.assertRaises(MoveError):
            execute_move(plan_move(self.p("nope"), self.p("x")))


class CrossVolumeTests(Base):
    @staticmethod
    def fake_exdev(real):
        calls = {"n": 0}

        def wrapper(src, dst, overwrite):
            calls["n"] += 1
            if calls["n"] == 1:  # the first attempt is the direct rename; later ones are the engine's own
                raise OSError(errno.EXDEV, "Invalid cross-device link")
            return real(src, dst, overwrite)
        return wrapper

    def run_move(self, src, dst, **kwargs):
        plan = plan_move(src, dst, **kwargs)
        with patch.object(_engine, "_rename_direct", side_effect=self.fake_exdev(_engine._rename_direct)):
            return execute_move(plan)

    def test_file_is_copied_verified_and_the_source_removed(self):
        write(self.p("a.txt"), "payload")
        self.assertEqual(self.run_move(self.p("a.txt"), self.p("b.txt")), {"method": "copy"})
        self.assertEqual(read(self.p("b.txt")), "payload")
        self.assertFalse(os.path.exists(self.p("a.txt")))
        self.assertEqual([n for n in os.listdir(self.d) if "maki-moving" in n], [])

    def test_directory_with_nested_files_and_a_link(self):
        write(self.p("tree", "x", "y.txt"), "deep")
        write(self.p("tree", "z.txt"), "top")
        try:
            os.symlink("z.txt", self.p("tree", "ln"))
            has_link = True
        except (OSError, NotImplementedError):
            has_link = False
        self.assertEqual(self.run_move(self.p("tree"), self.p("moved"))["method"], "copy")
        self.assertEqual(read(self.p("moved", "x", "y.txt")), "deep")
        if has_link:
            self.assertEqual(os.readlink(self.p("moved", "ln")), "z.txt")
        self.assertFalse(os.path.exists(self.p("tree")))

    def test_overwrite_across_volumes(self):
        write(self.p("a.txt"), "new")
        write(self.p("b.txt"), "old")
        self.run_move(self.p("a.txt"), self.p("b.txt"), overwrite=True)
        self.assertEqual(read(self.p("b.txt")), "new")
        self.assertFalse(os.path.exists(self.p("a.txt")))

    def test_windows_not_same_device_error_also_triggers_the_fallback(self):
        class WinError(OSError):
            winerror = 17
        write(self.p("a.txt"), "w")
        plan = plan_move(self.p("a.txt"), self.p("b.txt"))
        real = _engine._rename_direct
        state = {"n": 0}

        def wrapper(src, dst, overwrite):
            state["n"] += 1
            if state["n"] == 1:
                raise WinError(0, "The system cannot move the file to a different disk drive")
            return real(src, dst, overwrite)

        with patch.object(_engine, "_rename_direct", side_effect=wrapper):
            self.assertEqual(execute_move(plan), {"method": "copy"})
        self.assertEqual(read(self.p("b.txt")), "w")

    def test_a_mismatching_copy_is_discarded_and_the_source_survives(self):
        write(self.p("a.txt"), "payload")
        write(self.p("b.txt"), "theirs")
        with patch.object(_engine, "_sha256", side_effect=["aaa", "bbb"]):
            with self.assertRaises(MoveError):
                self.run_move(self.p("a.txt"), self.p("b.txt"), overwrite=True)
        self.assertEqual(read(self.p("a.txt")), "payload")
        self.assertEqual(read(self.p("b.txt")), "theirs")
        self.assertEqual([n for n in os.listdir(self.d) if "maki-moving" in n], [])

    def test_a_copy_that_fails_midway_is_cleaned_up(self):
        write(self.p("tree", "x.txt"), "x")
        with patch.object(_engine.shutil, "copy2", side_effect=OSError(errno.ENOSPC, "No space left")):
            with self.assertRaises(MoveError):
                self.run_move(self.p("tree"), self.p("moved"))
        self.assertTrue(os.path.exists(self.p("tree", "x.txt")))
        self.assertFalse(os.path.exists(self.p("moved")))
        self.assertEqual([n for n in os.listdir(self.d) if "maki-moving" in n], [])

    def test_a_source_that_cannot_be_removed_is_reported_as_partial(self):
        write(self.p("a.txt"), "payload")
        with patch.object(_engine, "_remove", side_effect=PermissionError(errno.EACCES, "locked")):
            with self.assertRaises(MoveError) as ctx:
                self.run_move(self.p("a.txt"), self.p("b.txt"))
        self.assertTrue(ctx.exception.partial)
        self.assertEqual(read(self.p("b.txt")), "payload")
        self.assertEqual(read(self.p("a.txt")), "payload")

    @unittest.skipUnless(os.environ.get("MAKI_TEST_OTHER_VOLUME"), "set MAKI_TEST_OTHER_VOLUME to a folder on another volume")
    def test_real_move_to_another_volume(self):
        other = tempfile.mkdtemp(dir=os.environ["MAKI_TEST_OTHER_VOLUME"])
        try:
            write(self.p("a.txt"), "real")
            plan = plan_move(self.p("a.txt"), os.path.join(other, "a.txt"))
            self.assertFalse(plan.same_volume)
            self.assertEqual(execute_move(plan)["method"], "copy")
            self.assertEqual(read(os.path.join(other, "a.txt")), "real")
        finally:
            shutil.rmtree(other, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
