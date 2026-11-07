"""
Load and soak check for ``maki serve``.

    python -m maki.test.load_check --duration 120 --concurrency 32 [--latency-ms 5] [--json report.json]
    python -m maki.test.load_check --url http://127.0.0.1:8100 --api-key KEY --duration 60 --concurrency 8

By default it starts the real server stack (uvicorn over real sockets, API key on) in this process with a fake
backend that answers after ``--latency-ms``, so the server, not a model, is what is measured. With ``--url`` it
drives an already running server instead (only ``/execute`` and the cheap endpoints, so a slow model is not
overwhelmed on purpose).

The workload mixes valid calls (execute, memory set/get/delete, history, health, info) with the things a
network brings: malformed JSON, oversized bodies, missing and wrong credentials, unknown paths and wrong methods.
Every response must be one of the statuses that request is allowed to produce; anything else (a 500, a hang, a
dropped connection that was not an oversized upload) is an error. Process memory, threads and open file
descriptors are sampled every second, and the run fails when memory keeps climbing, when threads or descriptors
leak, or when any error occurred.

Exit status 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import requests

KEY = "load-check-key"


@dataclass
class Sample:
    t: float
    rss_mb: float
    threads: int
    fds: int


@dataclass
class Result:
    latencies: Dict[str, List[float]] = field(default_factory=dict)
    statuses: Dict[Tuple[str, object], int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def record(self, kind: str, status: object, seconds: float, allowed: Sequence[object]) -> None:
        self.latencies.setdefault(kind, []).append(seconds)
        self.statuses[(kind, status)] = self.statuses.get((kind, status), 0) + 1
        if status not in allowed and len(self.errors) < 50:
            self.errors.append(f"{kind}: unexpected {status!r}")


def process_rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 if out.isdigit() else 0.0


def process_threads(pid: int) -> int:
    """Threads of process *pid* (Linux: /proc; macOS: ``ps -M``)."""
    status = f"/proc/{pid}/status"
    if os.path.exists(status):
        for line in open(status):
            if line.startswith("Threads:"):
                return int(line.split()[1])
    out = subprocess.run(["ps", "-M", "-p", str(pid)], capture_output=True, text=True).stdout.splitlines()
    return max(0, len(out) - 1)


def open_fds(pid: int) -> int:
    """Open file descriptors of process *pid* (Linux: /proc; elsewhere ``lsof``); -1 when it cannot be read."""
    path = f"/proc/{pid}/fd"
    if os.path.isdir(path):
        return len(os.listdir(path))
    try:
        out = subprocess.run(["lsof", "-p", str(pid), "-n", "-P"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return -1
    return max(0, len(out.splitlines()) - 1)


def start_server(latency_ms: float, max_body_bytes: int):
    """The real server in a background thread, on a free port, with a fake backend."""
    import uvicorn

    from ..agents import Agent
    from ..distributed.server import create_app
    from ..backend import LLMBackend
    from ..objects import LLMResponse

    class Fake(LLMBackend):
        model, temperature, supports_history = "fake", 0.0, True

        def request(self, prompt):
            return self.chat(prompt)

        def chat(self, prompt, history=None, config=None, system=None, images=None):
            if latency_ms:
                time.sleep(latency_ms / 1000)
            return LLMResponse(content="ok", model="fake", prompt_tokens=3, completion_tokens=2, total_tokens=5,
                               elapsed_seconds=0.0)

    app = create_app(Agent("load", Fake()), api_key=KEY, max_body_bytes=max_body_bytes)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("the server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def serve_forever(latency_ms: float, max_body_bytes: int) -> None:
    """Server-only mode (run in its own process by the check): prints ``PORT n`` and serves until stdin closes."""
    server, thread, base = start_server(latency_ms, max_body_bytes)
    print(f"PORT {base.rsplit(':', 1)[1]}", flush=True)
    sys.stdin.read()  # the parent closes our stdin when it is done
    server.should_exit = True
    thread.join(10)


def launch_server_process(latency_ms: float, max_body_bytes: int):
    """The server in a separate process: ``(process, base_url)``. Its memory is then the server's alone."""
    cmd = [sys.executable, "-m", "maki.test.load_check", "--serve-only", "--latency-ms", str(latency_ms),
           "--max-body", str(max_body_bytes)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line.startswith("PORT "):
            return proc, f"http://127.0.0.1:{line.split()[1]}"
        if proc.poll() is not None:
            break
    proc.kill()
    raise RuntimeError("the server process did not start")


def workload(rng: random.Random, big: bytes, api_key: str):
    """One request: ``(kind, method, path, kwargs, allowed statuses)``."""
    roll = rng.random()
    auth = {"Authorization": f"Bearer {api_key}"}
    key = f"k{rng.randrange(150)}"
    if roll < 0.40:
        return "execute", "POST", "/execute", {"json": {"task": f"task {rng.random()}"}, "headers": auth}, (200,)
    if roll < 0.52:
        return "memory_set", "POST", "/memory/set", {"json": {"key": key, "value": rng.random()}, "headers": auth}, (200,)
    if roll < 0.60:
        return "memory_get", "GET", f"/memory/{key}", {"headers": auth}, (200, 404)
    if roll < 0.64:
        return "memory_delete", "DELETE", f"/memory/{key}", {"headers": auth}, (200, 404)
    if roll < 0.72:
        return "health", "GET", "/health", {}, (200,)
    if roll < 0.78:
        return "info", "GET", "/info", {"headers": auth}, (200,)
    if roll < 0.82:
        return "history", "GET", "/history", {"headers": auth}, (200,)
    if roll < 0.86:
        return "bad_json", "POST", "/execute", {"data": b"{not json", "headers": {**auth, "Content-Type": "application/json"}}, (400, 422)
    if roll < 0.89:
        return "no_auth", "POST", "/execute", {"json": {"task": "x"}}, (401,)
    if roll < 0.92:
        return "wrong_auth", "GET", "/info", {"headers": {"Authorization": "Bearer nope"}}, (401,)
    if roll < 0.95:
        return "unknown_path", "GET", f"/{rng.choice(['nope', '../etc/passwd', 'x' * 300, '%00'])}", {"headers": auth}, (404, 405)
    if roll < 0.97:
        return "wrong_method", "GET", "/execute", {"headers": auth}, (405,)
    return "oversize", "POST", "/execute", {"data": big, "headers": {**auth, "Content-Type": "application/json"}}, (413, "dropped")


def worker(base: str, deadline: float, result: Result, lock: threading.Lock, seed: int, big: bytes, only_cheap: bool,
           api_key: str):
    rng = random.Random(seed)
    session = requests.Session()
    while time.time() < deadline:
        kind, method, path, kwargs, allowed = workload(rng, big, api_key)
        if only_cheap and kind not in ("execute", "health", "info"):
            continue
        started = time.perf_counter()
        try:
            response = session.request(method, base + path, timeout=60, **kwargs)
            status: object = response.status_code
        except requests.exceptions.ConnectionError:
            status = "dropped"  # the server may close a refused upload before reading it
        except requests.exceptions.RequestException as exc:
            status = type(exc).__name__
        elapsed = time.perf_counter() - started
        with lock:
            result.record(kind, status, elapsed, allowed)


def sampler(deadline: float, samples: List[Sample], stop: threading.Event, pid: int):
    t0 = time.time()
    while not stop.is_set() and time.time() < deadline + 1:
        samples.append(Sample(time.time() - t0, process_rss_mb(pid), process_threads(pid), open_fds(pid)))
        stop.wait(1.0)


class _AllocationTrace:
    """tracemalloc snapshots at a quarter of the run and at the end, to name what keeps growing."""

    def __init__(self, duration: float):
        import tracemalloc
        self.tm = tracemalloc
        tracemalloc.start(12)
        self.first = None
        timer = threading.Timer(duration / 4, self._take)
        timer.daemon = True
        timer.start()

    def _take(self):
        self.first = self.tm.take_snapshot()

    def report(self, top: int = 12):
        if self.first is None:
            return
        last = self.tm.take_snapshot()
        stats = last.compare_to(self.first, "lineno")
        print(f"\nallocation growth since the first quarter (top {top} by size):")
        for stat in stats[:top]:
            frame = stat.traceback[0]
            print(f"  {stat.size_diff / 1024:+9.1f} KiB  {stat.count_diff:+7d} blocks  {frame.filename.split('site-packages/')[-1]}:{frame.lineno}")
        total = sum(s.size_diff for s in stats)
        print(f"  total traced growth: {total / 1024 / 1024:+.2f} MiB")


def percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def analyse(samples: List[Sample], args) -> List[str]:
    """The failed checks, as messages."""
    problems = []
    if len(samples) < 8:
        return problems
    quarter = max(2, len(samples) // 4)
    early, late = samples[quarter:2 * quarter], samples[-quarter:]  # skip warm-up, compare steady states
    growth = statistics.median(s.rss_mb for s in late) - statistics.median(s.rss_mb for s in early)
    if growth > args.max_rss_growth_mb:
        problems.append(f"memory grew {growth:.1f} MB between the first and last quarter (limit {args.max_rss_growth_mb})")
    half = samples[len(samples) // 2:]
    mean_t, mean_r = statistics.fmean(x.t for x in half), statistics.fmean(x.rss_mb for x in half)
    slope = sum((x.t - mean_t) * (x.rss_mb - mean_r) for x in half) / max(1e-9, sum((x.t - mean_t) ** 2 for x in half)) * 60
    if slope > args.max_rss_slope:
        problems.append(f"memory is still climbing at {slope:.1f} MB/min over the second half (limit {args.max_rss_slope})")
    thread_growth = max(s.threads for s in late) - max(s.threads for s in early)
    if thread_growth > args.max_thread_growth:
        problems.append(f"thread count grew by {thread_growth} (limit {args.max_thread_growth})")
    fd_growth = max(s.fds for s in late) - max(s.fds for s in early)
    if fd_growth > args.max_fd_growth:
        problems.append(f"open file descriptors grew by {fd_growth} (limit {args.max_fd_growth})")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maki.test.load_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--duration", type=float, default=60.0, help="seconds of load (default 60)")
    parser.add_argument("--concurrency", type=int, default=16, help="parallel clients (default 16)")
    parser.add_argument("--latency-ms", type=float, default=3.0, help="fake backend latency (default 3)")
    parser.add_argument("--url", help="drive this running server instead of an in-process one")
    parser.add_argument("--api-key", default=KEY, help="bearer key for --url")
    parser.add_argument("--max-rss-growth-mb", type=float, default=40.0)
    parser.add_argument("--max-rss-slope", type=float, default=3.0, help="MB/min over the second half of the run")
    parser.add_argument("--max-thread-growth", type=int, default=5)
    parser.add_argument("--max-fd-growth", type=int, default=10)
    parser.add_argument("--json", help="write the report here")
    parser.add_argument("--trace-malloc", action="store_true",
                        help="compare allocation snapshots from 1/4 and the end of the run and list what grew (slower; "
                             "implies --in-process)")
    parser.add_argument("--in-process", action="store_true",
                        help="run the server in this process (default: a separate process whose memory is sampled)")
    parser.add_argument("--serve-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-body", type=int, default=64 * 1024, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.serve_only:
        serve_forever(args.latency_ms, args.max_body)
        return 0

    server = proc = None
    if args.url:
        base, watch_pid = args.url.rstrip("/"), None
    elif args.in_process or args.trace_malloc:
        server, _, base = start_server(args.latency_ms, max_body_bytes=64 * 1024)
        watch_pid = os.getpid()
    else:
        proc, base = launch_server_process(args.latency_ms, 64 * 1024)
        watch_pid = proc.pid
    big = b'{"task": "' + b"x" * (256 * 1024) + b'"}'  # over the 64 KiB limit used above
    result, lock, samples, stop = Result(), threading.Lock(), [], threading.Event()
    deadline = time.time() + args.duration
    print(f"load: {base}, {args.concurrency} clients for {args.duration:.0f}s", flush=True)
    threads = [threading.Thread(target=worker, args=(base, deadline, result, lock, i, big, bool(args.url), args.api_key), daemon=True)
               for i in range(args.concurrency)]
    watcher = threading.Thread(target=sampler, args=(deadline, samples, stop, watch_pid), daemon=True) \
        if watch_pid else None
    if watcher:
        watcher.start()
    tracer = _AllocationTrace(args.duration) if args.trace_malloc else None
    for t in threads:
        t.start()
    for t in threads:
        t.join(args.duration + 90)
    stop.set()
    if watcher:
        watcher.join(5)
    stuck = sum(t.is_alive() for t in threads)
    if tracer is not None:
        tracer.report()

    total = sum(result.statuses.values())
    print(f"\n{total} requests, {total / args.duration:.0f}/s, {len(result.errors)} unexpected responses, "
          f"{stuck} clients stuck")
    print(f"{'kind':14} {'count':>7} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'max ms':>8}  statuses")
    for kind, lat in sorted(result.latencies.items()):
        statuses = ", ".join(f"{s}:{n}" for (k, s), n in sorted(result.statuses.items(), key=str) if k == kind)
        print(f"{kind:14} {len(lat):7d} {1000 * percentile(lat, .5):8.1f} {1000 * percentile(lat, .95):8.1f} "
              f"{1000 * percentile(lat, .99):8.1f} {1000 * max(lat):8.1f}  {statuses}")
    problems = analyse(samples, args)
    if samples:
        first, last = samples[0], samples[-1]
        where = "server process" if watch_pid != os.getpid() else "this process (server and clients)"
        print(f"\n{where}: RSS {first.rss_mb:.0f} -> {last.rss_mb:.0f} MB, threads {first.threads} -> {last.threads}, "
              f"fds {first.fds} -> {last.fds}")
    if stuck:
        problems.append(f"{stuck} clients never finished (a hang)")
    problems += result.errors[:10]
    for p in problems:
        print("FAIL:", p)
    print("PASS" if not problems else f"{len(problems)} problem(s)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"requests": total, "errors": result.errors, "problems": problems,
                       "samples": [s.__dict__ for s in samples]}, handle)
    if server is not None:
        server.should_exit = True
    if proc is not None:
        proc.stdin.close()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
