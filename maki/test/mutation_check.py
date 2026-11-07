"""
Mutation check: do the tests notice when the code is subtly wrong?

    python -m maki.test.mutation_check maki/utils.py \\
        --tests maki/test/test_ssrp_protection.py maki/test/test_connector.py \\
        [--functions _validate_ip,_validate_domain] [--lines 40-90] [--workers 4] [--timeout 90] [--limit 200]

For every mutation site in the target file (a comparison flipped, ``and``/``or`` swapped, a ``not`` dropped,
a boolean or integer constant changed, a ``raise`` turned into ``pass``, a returned value replaced by None) the
tests you name are run against a copy of the package with just that one change. A mutant is

    killed     a test failed (a hang that hits the timeout counts: a test noticed that something is wrong);
    survived   every test still passed: the tests do not check what that line does.

Mutants run in throw-away copies of the ``maki`` package in a temporary directory, never in the working tree,
so a mutant that loops forever or writes files cannot damage the repository. The score is killed / (killed +
survived). A surviving mutant is not always a gap (an equivalent mutation changes nothing observable), but
each one is worth reading.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import copy
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_SWAP_COMPARE = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE,
                 ast.GtE: ast.Lt, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
_SWAP_BINOP = {ast.Add: ast.Sub, ast.Sub: ast.Add}


@dataclass
class Site:
    index: int
    line: int
    kind: str
    detail: str


def _eligible(node: ast.AST) -> Optional[str]:
    """The kind of mutation applicable to *node*, or None."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _SWAP_COMPARE:
        return "compare"
    if isinstance(node, ast.BoolOp):
        return "boolop"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "not"
    if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or (isinstance(node.value, int) and abs(node.value) < 10**9)):
        return "constant"
    if isinstance(node, ast.BinOp) and type(node.op) in _SWAP_BINOP:
        return "binop"
    if isinstance(node, ast.Raise):
        return "raise"
    if isinstance(node, ast.Return) and node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None):
        return "return"
    return None


class _Mutator(ast.NodeTransformer):
    """Applies the mutation at one eligible site (counted in walk order); everything else is left alone."""

    def __init__(self, target: int, allowed: "set[int] | None"):
        self.target, self.allowed, self.seen, self.done = target, allowed, -1, None

    def generic_visit(self, node):
        kind = _eligible(node)
        if kind and (self.allowed is None or getattr(node, "lineno", -1) in self.allowed):
            self.seen += 1
            if self.seen == self.target:
                return self._mutate(node, kind)
        return super().generic_visit(node)

    def _mutate(self, node, kind):
        self.done = f"{kind} at line {node.lineno}"
        if kind == "compare":
            node.ops = [_SWAP_COMPARE[type(node.ops[0])]()]
        elif kind == "boolop":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif kind == "not":
            return node.operand
        elif kind == "constant":
            node.value = (not node.value) if isinstance(node.value, bool) else node.value + 1
        elif kind == "binop":
            node.op = _SWAP_BINOP[type(node.op)]()
        elif kind == "raise":
            return ast.copy_location(ast.Pass(), node)
        elif kind == "return":
            node.value = ast.copy_location(ast.Constant(value=None), node)
        return node


def _scope_lines(tree: ast.AST, functions: Sequence[str], lines: Optional[Tuple[int, int]]) -> "set[int] | None":
    if not functions and not lines:
        return None
    keep: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions:
            keep.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if lines:
        keep.update(range(lines[0], lines[1] + 1))
    return keep


def find_sites(source: str, functions: Sequence[str] = (), lines: Optional[Tuple[int, int]] = None) -> List[Site]:
    tree = ast.parse(source)
    allowed = _scope_lines(tree, functions, lines)
    sites: List[Site] = []
    counter = -1
    probe = _Probe(allowed)
    probe.visit(tree)
    for line, kind in probe.found:
        counter += 1
        sites.append(Site(counter, line, kind, ""))
    return sites


class _Probe(ast.NodeVisitor):
    def __init__(self, allowed):
        self.allowed, self.found = allowed, []

    def generic_visit(self, node):
        kind = _eligible(node)
        if kind and (self.allowed is None or getattr(node, "lineno", -1) in self.allowed):
            self.found.append((node.lineno, kind))
        super().generic_visit(node)


def mutate(source: str, index: int, functions: Sequence[str] = (), lines: Optional[Tuple[int, int]] = None) -> Tuple[str, str]:
    tree = ast.parse(source)
    mutator = _Mutator(index, _scope_lines(tree, functions, lines))
    new_tree = mutator.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new_tree)
    if mutator.done is None:
        raise IndexError(index)
    return ast.unparse(new_tree), mutator.done


def _run_one(workdir: Path, relative: str, mutated: str, tests: Sequence[str], timeout: float) -> str:
    target = workdir / relative
    original = target.read_text(encoding="utf-8")
    target.write_text(mutated, encoding="utf-8")
    try:
        env = {**os.environ, "PYTHONPATH": str(workdir), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests],
                                  cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "killed (timeout)"
        return "survived" if proc.returncode == 0 else "killed"
    finally:
        target.write_text(original, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maki.test.mutation_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("target", help="the source file to mutate, e.g. maki/utils.py")
    parser.add_argument("--tests", nargs="+", required=True, help="test files or directories to run against each mutant")
    parser.add_argument("--functions", default="", help="comma-separated function names to restrict to")
    parser.add_argument("--lines", default="", help="START-END line range to restrict to")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds per mutant")
    parser.add_argument("--limit", type=int, default=0, help="run at most this many mutants (0 = all)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    target = (root / args.target).resolve()
    relative = str(target.relative_to(root))
    functions = [f for f in args.functions.split(",") if f]
    lines = tuple(int(x) for x in args.lines.split("-")) if args.lines else None
    source = target.read_text(encoding="utf-8")
    sites = find_sites(source, functions, lines)
    if args.limit:
        sites = sites[:args.limit]
    print(f"{relative}: {len(sites)} mutants, tests: {' '.join(args.tests)}", flush=True)

    with tempfile.TemporaryDirectory(prefix="maki-mutants-") as tmp:
        copies = []
        for i in range(max(1, args.workers)):
            workdir = Path(tmp) / f"w{i}"
            workdir.mkdir()
            shutil.copytree(root / "maki", workdir / "maki", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if (root / "pyproject.toml").exists():
                shutil.copy(root / "pyproject.toml", workdir / "pyproject.toml")
            copies.append(workdir)
        baseline = _run_one(copies[0], relative, source, args.tests, args.timeout)
        if baseline != "survived":
            print("the tests do not pass on the unmutated code; fix that first", file=sys.stderr)
            return 2

        results = []

        def work(item):
            n, site = item
            workdir = copies[n % len(copies)]
            mutated, what = mutate(source, site.index, functions, lines)
            return site, what, _run_one(workdir, relative, mutated, args.tests, args.timeout)

        # one mutant per worker copy at a time: copies are reused, so serialise per copy
        buckets = [[] for _ in copies]
        for n, site in enumerate(sites):
            buckets[n % len(copies)].append((n, site))

        def run_bucket(bucket):
            return [work(item) for item in bucket]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(copies)) as pool:
            for done in pool.map(run_bucket, buckets):
                results.extend(done)

    results.sort(key=lambda r: r[0].index)
    lines_of_source = source.splitlines()
    killed = sum(1 for _, _, status in results if status.startswith("killed"))
    survived = [(s, w) for s, w, status in results if status == "survived"]
    total = killed + len(survived)
    print(f"killed {killed}/{total} ({100 * killed / total:.0f}%)" if total else "no mutants")
    for site, what in survived:
        print(f"  SURVIVED {what}: {lines_of_source[site.line - 1].strip()[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
