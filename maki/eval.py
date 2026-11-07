"""
Compare models on your own tasks, through the whole agent (tools, memory, policies), not a bare prompt.

    from maki.eval import EvalCase, run_eval

    cases = [
        EvalCase("capital", "Capital of France? One word.", contains="paris"),
        EvalCase("json", "Reply with only JSON: {\\"ok\\": true}", check=lambda out: '"ok"' in out),
    ]
    report = run_eval(cases, ["ollama:qwen3.5:4b", "ollama:gemma4:26b"], runs=3)
    print(report.table())

Every run gets a fresh agent, so runs never see each other's history. A case passes when every check it
declares passes: ``contains`` (all substrings, case-insensitive), ``check`` (a callable on the answer, and
optionally on the agent), and ``criteria`` (text judged by a judge agent). Per run the report keeps the time,
tokens, model calls and any error; per model it shows pass rate, median time and tokens.

Runs are grouped by model (all of one model's cases, then the next), because switching a local model costs a
reload. A warm-up call before each model keeps that load out of the timings (``warmup=False`` to include it).

Built-in suites live in :mod:`maki.eval_suites`; from the shell::

    python -m maki eval --suite basic --models ollama:qwen3.5:4b ollama:gemma4:26b --runs 2
"""

from __future__ import annotations

import inspect
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agents.agent import Agent
from .agents.loop_judge import build_judge_prompt
from .backend import LLMBackend

CheckResult = Union[bool, Tuple[bool, str]]
Check = Callable[..., CheckResult]

_OUTPUT_KEEP = 300  # characters of each answer kept in the report


@dataclass
class EvalCase:
    """One task and how to tell whether the answer is good.

    Attributes:
        name: Short unique label.
        task: What the agent is asked (the last turn, which is the one scored).
        contains: Substring or substrings that must all appear in the answer (case-insensitive).
        check: ``check(answer)`` or ``check(answer, agent)`` returning a bool or ``(bool, detail)``.
        criteria: A plain-language requirement a judge agent decides on (needs ``judge=`` in :func:`run_eval`).
        use_plugins: Run with the agent's plugins and tool calling.
        context: Passed to ``execute_task`` as its context.
        setup_turns: Earlier turns run first on the same agent (their answers are not scored); for memory tests.
        make_agent: Builds the agent for this case, overriding the suite's (for cases that need their own
            memory policy or plugins).
    """
    name: str
    task: str
    contains: Union[str, Sequence[str], None] = None
    check: Optional[Check] = None
    criteria: Optional[str] = None
    use_plugins: bool = False
    context: Optional[Dict[str, Any]] = None
    setup_turns: Sequence[str] = ()
    make_agent: Optional[Callable[[LLMBackend], Agent]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("EvalCase.name must be a non-empty string")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError(f"EvalCase '{self.name}': task must be a non-empty string")
        if isinstance(self.contains, str):
            self.contains = (self.contains,)
        if self.contains is not None:
            self.contains = tuple(self.contains)
            if not self.contains or not all(isinstance(c, str) and c for c in self.contains):
                raise ValueError(f"EvalCase '{self.name}': contains needs non-empty strings")
        if self.check is not None and not callable(self.check):
            raise ValueError(f"EvalCase '{self.name}': check must be callable")
        if self.criteria is not None and (not isinstance(self.criteria, str) or not self.criteria.strip()):
            raise ValueError(f"EvalCase '{self.name}': criteria must be a non-empty string")
        if self.contains is None and self.check is None and self.criteria is None:
            raise ValueError(f"EvalCase '{self.name}': give at least one of contains, check or criteria")
        self.setup_turns = tuple(self.setup_turns)


@dataclass
class EvalSuite:
    """A named set of cases plus how to build the agent they run on."""
    name: str
    cases: Sequence[EvalCase]
    make_agent: Optional[Callable[[LLMBackend], Agent]] = None
    description: str = ""


@dataclass
class RunResult:
    target: str
    case: str
    run: int
    passed: bool
    detail: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    error: str = ""
    output: str = ""

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("target", "case", "run", "passed", "detail", "seconds",
                                              "prompt_tokens", "completion_tokens", "calls", "error", "output")}


@dataclass
class TargetStats:
    target: str
    runs: int
    passed: int
    errors: int
    median_seconds: float
    mean_tokens: float
    warmup_seconds: Optional[float] = None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0


@dataclass
class EvalReport:
    """All runs of one :func:`run_eval`, with summaries."""
    suite: str
    results: List[RunResult] = field(default_factory=list)
    warmups: Dict[str, float] = field(default_factory=dict)

    @property
    def targets(self) -> List[str]:
        return list(dict.fromkeys(r.target for r in self.results))

    @property
    def cases(self) -> List[str]:
        return list(dict.fromkeys(r.case for r in self.results))

    def stats(self) -> List[TargetStats]:
        out = []
        for target in self.targets:
            rows = [r for r in self.results if r.target == target]
            out.append(TargetStats(
                target=target, runs=len(rows), passed=sum(r.passed for r in rows),
                errors=sum(bool(r.error) for r in rows),
                median_seconds=statistics.median(r.seconds for r in rows),
                mean_tokens=statistics.fmean(r.tokens for r in rows),
                warmup_seconds=self.warmups.get(target)))
        return out

    def pass_rate(self, target: str, case: Optional[str] = None) -> float:
        rows = [r for r in self.results if r.target == target and (case is None or r.case == case)]
        return sum(r.passed for r in rows) / len(rows) if rows else 0.0

    def failures(self, target: Optional[str] = None) -> List[RunResult]:
        return [r for r in self.results if not r.passed and (target is None or r.target == target)]

    def table(self) -> str:
        """A plain-text summary: one row per model, then pass counts per case."""
        stats = self.stats()
        width = max([len("model")] + [len(s.target) for s in stats])
        lines = [f"suite: {self.suite}",
                 f"{'model':<{width}}  {'pass':>9}  {'rate':>5}  {'median s':>8}  {'avg tokens':>10}  {'errors':>6}"]
        for s in stats:
            lines.append(f"{s.target:<{width}}  {f'{s.passed}/{s.runs}':>9}  {s.pass_rate:>5.0%}  "
                         f"{s.median_seconds:>8.1f}  {s.mean_tokens:>10.0f}  {s.errors:>6}")
        cases = self.cases
        if len(cases) > 1:
            name_width = max(len(c) for c in cases)
            lines += ["", f"{'case':<{name_width}}  " + "  ".join(f"{s.target[:18]:>18}" for s in stats)]
            for case in cases:
                cells = []
                for s in stats:
                    rows = [r for r in self.results if r.target == s.target and r.case == case]
                    cells.append(f"{sum(r.passed for r in rows)}/{len(rows)}".rjust(18))
                lines.append(f"{case:<{name_width}}  " + "  ".join(cells))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"suite": self.suite, "warmups": self.warmups,
                "results": [r.to_dict() for r in self.results],
                "stats": [{"target": s.target, "runs": s.runs, "passed": s.passed, "pass_rate": s.pass_rate,
                           "errors": s.errors, "median_seconds": s.median_seconds, "mean_tokens": s.mean_tokens}
                          for s in self.stats()]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


# --------------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------------

def _resolve_targets(targets: Any) -> List[Tuple[str, Union[str, LLMBackend]]]:
    if isinstance(targets, (str, LLMBackend)):
        targets = [targets]
    items = list(targets.items()) if isinstance(targets, Mapping) else [(None, t) for t in targets]
    if not items:
        raise ValueError("run_eval needs at least one model to evaluate")
    named: List[Tuple[str, Union[str, LLMBackend]]] = []
    for label, target in items:
        if not isinstance(target, (str, LLMBackend)):
            raise TypeError(f"targets must be 'provider:model' strings or LLMBackend instances, got {type(target).__name__}")
        name = label or (target if isinstance(target, str) else f"{type(target).__name__}({target.model})")
        named.append((name, target))
    names = [n for n, _ in named]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    return named


def _takes_agent(check: Check) -> bool:
    """True when *check* wants the agent as a second argument: a required second positional parameter, or
    one named ``agent``. An optional second parameter (``lambda a, strict=False``) is left alone."""
    try:
        positional = [p for p in inspect.signature(check).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(positional) >= 2 and (positional[1].default is positional[1].empty or positional[1].name == "agent")


def _run_check(check: Check, answer: str, agent: Agent) -> Tuple[bool, str]:
    takes_agent = _takes_agent(check)
    outcome = check(answer, agent) if takes_agent else check(answer)
    if isinstance(outcome, tuple):
        return bool(outcome[0]), str(outcome[1])
    return bool(outcome), ""


_JUDGE_ATTEMPTS = 2
_CRITERIA_ID = "c1"


def _verdicts(raw: str) -> Optional[List[dict]]:
    """The verdict entries (dicts with a boolean ``pass``) of a judge reply, or None if it has none."""
    for text in (raw, raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw else ""):
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            continue
        entries = data.get("results") if isinstance(data, dict) else None
        if isinstance(entries, list) and entries and all(isinstance(e, dict) and isinstance(e.get("pass"), bool)
                                                         for e in entries):
            return entries
    return None


def _judge(judge: Agent, goal: str, criteria: str, answer: str) -> Optional[Tuple[bool, str]]:
    """``(passed, feedback)`` from a judge agent, or None when it never gave a usable reply.

    One criterion is judged, so the verdict entries' ids do not matter: models often split a compound
    criterion ("polite and short") into entries with ids of their own. The criterion passes only if every
    entry passes. (Loops, which judge several named requirements, keep the strict id check.)
    """
    prompt = build_judge_prompt(goal, [(_CRITERIA_ID, criteria)], answer)
    for _ in range(_JUDGE_ATTEMPTS):
        try:
            raw = judge.execute_task_with_retry(prompt)
        except Exception:  # fail closed: a judge that errors never passes a case
            continue
        entries = _verdicts(raw or "")
        if entries is not None:
            feedback = "; ".join(str(e.get("feedback") or "") for e in entries if not e["pass"] and e.get("feedback"))
            return all(e["pass"] for e in entries), feedback
    return None


def _score(case: EvalCase, answer: str, agent: Agent, judge: Optional[Agent]) -> Tuple[bool, str, str]:
    """``(passed, detail, error)`` for one answer; an error means the scoring itself failed."""
    failed: List[str] = []
    if case.contains:
        missing = [c for c in case.contains if c.lower() not in answer.lower()]
        if missing:
            failed.append(f"missing {missing}")
    if case.check is not None:
        try:
            ok, detail = _run_check(case.check, answer, agent)
        except Exception as exc:  # a broken check is a broken case, not a model failure
            return False, "", f"check raised {type(exc).__name__}: {exc}"
        if not ok:
            failed.append(detail or "check failed")
    if case.criteria:
        verdict = _judge(judge, case.task, case.criteria, answer)
        if verdict is None:
            return False, "", "the judge gave no usable verdict"
        if not verdict[0]:
            failed.append(verdict[1] or "judge: criteria not met")
    return not failed, "; ".join(failed), ""


def _one_run(name: str, case: EvalCase, run: int, backend: LLMBackend,
             make_agent: Callable[[LLMBackend], Agent], judge: Optional[Agent]) -> RunResult:
    result = RunResult(target=name, case=case.name, run=run, passed=False)
    started = time.perf_counter()
    agent = None
    try:
        agent = (case.make_agent or make_agent)(backend)
        before = agent.usage.snapshot()
        for turn in case.setup_turns:
            agent.execute_task(turn, use_plugins=case.use_plugins)
        answer = agent.execute_task(case.task, context=case.context, use_plugins=case.use_plugins)
        spent = agent.usage - before
        result.prompt_tokens, result.completion_tokens, result.calls = (
            spent.prompt_tokens, spent.completion_tokens, spent.calls)
        result.output = (answer or "")[:_OUTPUT_KEEP]
        result.seconds = time.perf_counter() - started
        result.passed, result.detail, result.error = _score(case, answer or "", agent, judge)
    except Exception as exc:  # a failing model call is a result, not a crash of the whole evaluation
        result.seconds = time.perf_counter() - started
        result.error = f"{type(exc).__name__}: {exc}"
    if result.error:
        result.passed = False
    return result


def _unpack(suite: Any, make_agent: Optional[Callable[[LLMBackend], Agent]]) -> Tuple[List[EvalCase], str, Any]:
    if isinstance(suite, EvalSuite):
        return list(suite.cases), suite.name, make_agent or suite.make_agent
    return list(suite), "custom", make_agent


def _validate(cases: List[EvalCase], runs: Any, judge: Optional[Agent]) -> None:
    if not cases:
        raise ValueError("run_eval needs at least one case")
    if len({c.name for c in cases}) != len(cases):
        raise ValueError("case names must be unique")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if any(c.criteria for c in cases) and judge is None:
        raise ValueError("cases with criteria need a judge agent (judge=)")
    if judge is not None and getattr(judge, "stateful", False):
        raise ValueError("the judge agent must not be stateful")


def _warm_up(backend: LLMBackend, label: str, report: "EvalReport") -> None:
    """One tiny request so a local model's load time is not charged to its first case."""
    started = time.perf_counter()
    try:
        backend.chat("Reply with the single word: OK")
        report.warmups[label] = time.perf_counter() - started
    except Exception:  # a dead model shows up in the runs themselves
        pass


def run_eval(
    suite: Union[EvalSuite, Sequence[EvalCase]],
    targets: Any,
    *,
    runs: int = 1,
    judge: Optional[Agent] = None,
    make_agent: Optional[Callable[[LLMBackend], Agent]] = None,
    warmup: bool = True,
    on_result: Optional[Callable[[RunResult], None]] = None,
) -> EvalReport:
    """Run every case *runs* times on every model and return the report.

    Args:
        suite: An :class:`EvalSuite` or a plain list of :class:`EvalCase`.
        targets: ``"provider:model"`` strings, ``LLMBackend`` instances, or a ``{label: target}`` mapping.
        runs: Repetitions per case and model. Models are not deterministic; 3 or more says more than 1.
        judge: A stateless agent that decides ``criteria`` cases. Required when any case has ``criteria``.
        make_agent: ``make_agent(backend) -> Agent``, called for every run. Defaults to the suite's, else a
            plain ``Agent``. Build plugins, memory and policies here.
        warmup: Send one tiny request to each model first so a local model's load time stays out of the
            timings.
        on_result: Called with each finished :class:`RunResult` (for live progress).

    Raises:
        ValueError: For no cases, ``runs < 1``, duplicate model names, or ``criteria`` without a judge.
    """
    cases, name, factory = _unpack(suite, make_agent)
    _validate(cases, runs, judge)
    named = _resolve_targets(targets)
    factory = factory or (lambda backend: Agent("eval", backend))

    report = EvalReport(suite=name)
    for label, target in named:
        backend = target
        if isinstance(target, str):
            from .backend_factory import init_backend
            backend = init_backend(target)
        if warmup:
            _warm_up(backend, label, report)
        for case in cases:
            for run in range(1, runs + 1):
                result = _one_run(label, case, run, backend, factory, judge)
                report.results.append(result)
                if on_result is not None:
                    on_result(result)
    return report


# --------------------------------------------------------------------------------------------------
# Command line (python -m maki eval ...)
# --------------------------------------------------------------------------------------------------

def add_cli_arguments(parser: Any) -> None:
    parser.add_argument("--suite", default="basic", help="built-in suite name (see --list), default: basic")
    parser.add_argument("--models", nargs="+", metavar="SPEC", help="models as provider:model, e.g. ollama:qwen3.5:4b")
    parser.add_argument("--runs", type=int, default=1, help="repetitions per case and model (default 1)")
    parser.add_argument("--judge", metavar="SPEC", help="model that judges 'criteria' cases (provider:model)")
    parser.add_argument("--json", metavar="FILE", help="also write the full report as JSON")
    parser.add_argument("--no-warmup", action="store_true", help="include model load time in the timings")
    parser.add_argument("--no-think", action="store_true",
                        help="turn model thinking off for ollama: models (thinking models are much slower)")
    parser.add_argument("--verbose", action="store_true", help="show maki's own log lines while running")
    parser.add_argument("--list", action="store_true", help="list the built-in suites and exit")


def _cli_models(args: Any) -> Tuple[Optional[Agent], Any]:
    """The judge agent and the models for the command line, built (and so validated) before anything runs."""
    from .backend_factory import init_backend, split_spec

    # A judge answers best in JSON mode with thinking off (see maki.agents.loop_judge).
    judge_options = {"think": False, "json_format": True} if (args.judge or "").startswith("ollama:") else {}
    judge = Agent("judge", init_backend(args.judge, **judge_options), role="reviewer") if args.judge else None
    if not args.no_think:
        for spec in args.models:  # fail on a bad spec before anything runs
            split_spec(spec)
        return judge, args.models
    return judge, {spec: init_backend(spec, think=False) if spec.startswith("ollama:") else spec
                   for spec in args.models}


def _print_result(result: RunResult) -> None:
    verdict = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
    note = result.error or result.detail
    print(f"  [{result.target}] {result.case} #{result.run}: {verdict} {result.seconds:.1f}s"
          f"{' - ' + note[:100] if note else ''}", flush=True)


def _cli_suite(args: Any) -> "EvalSuite":
    """The suite named on the command line, after checking the rest of the arguments."""
    from .eval_suites import get_suite

    if not args.models:
        raise ValueError("give at least one model with --models (provider:model)")
    if isinstance(args.runs, int) and args.runs < 1:
        raise ValueError("--runs must be at least 1")
    try:
        return get_suite(args.suite)
    except KeyError as exc:
        raise ValueError(exc.args[0]) from None


def run_cli(args: Any) -> int:
    import logging

    from .eval_suites import SUITES

    if args.list:
        for key, factory in SUITES.items():
            print(f"{key:<14} {factory().description}")
        return 0
    if not args.verbose:
        logging.getLogger("maki").setLevel(logging.WARNING)  # the progress lines below are the output
    try:
        suite = _cli_suite(args)
        judge, models = _cli_models(args)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}")
        return 2
    report = run_eval(suite, models, runs=args.runs, judge=judge, warmup=not args.no_warmup, on_result=_print_result)
    print()
    print(report.table())
    if args.json:
        report.save(args.json)
        print(f"\nreport written to {args.json}")
    return 0 if all(not r.error for r in report.results) else 1
