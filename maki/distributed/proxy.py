"""
AgentProxy: a drop-in replacement for Agent that dispatches every call to a
remote AgentServer over HTTP.

From AgentManager's perspective an AgentProxy is indistinguishable from a
local Agent: both expose execute_task(), execute_task_with_retry(),
stream_task(), remember(), recall(), and clear_memory().

All HTTP goes through the hardened ``maki.connector.Connector`` layer,
which owns URL validation, timeouts, and the mapping of transport/status
failures onto the Maki exception tree.

Circuit breaker  — After failure_threshold consecutive transient failures
                   the proxy stops attempting requests and raises immediately.
                   Resets automatically after recovery_timeout seconds.

Distributed tracing — Every execute_task / stream_task call generates a UUID
                   trace_id sent as X-Maki-Trace-Id.  The server echoes it back
                   in the response body and all server-side logs.  Stored as
                   last_trace_id for inspection after each call.

mTLS             — Pass ssl_verify=False or a CA-bundle path, and/or
                   cert=(certfile, keyfile) for mutual TLS.

Exception mapping (performed by Connector)
------------------------------------------
    transport timeout          → MakiTimeoutError
    connection error           → MakiNetworkError
    HTTP 408 / 504             → MakiTimeoutError
    HTTP 5xx                   → MakiNetworkError
    HTTP 4xx                   → MakiAPIError
    circuit breaker open       → MakiNetworkError (fail-fast, no HTTP call)
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Union

from ..agents.loop import Decision, Requirement
from ..connector import Connector
from ..exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .circuit_breaker import CircuitBreaker, CircuitState
from .loop_jobs import LOOP_ID_PATTERN

logger = logging.getLogger(__name__)

_RETRYABLE = (MakiNetworkError, MakiTimeoutError)

TRACE_HEADER = "X-Maki-Trace-Id"

_LOOP_OPTIONS = ("max_iterations", "max_seconds", "stall_limit", "context", "use_plugins", "max_tokens", "require_approval")
_MAX_STREAM_RECONNECTS = 5


@dataclass
class RemoteLoopResult:
    """Outcome of a loop run on a remote server (``events`` are the server's event dicts)."""
    loop_id: str
    stop_reason: str
    iterations: int
    detail: str = ""
    error: str = ""
    artifact: str = ""
    tokens: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == "succeeded"


class AgentProxy:
    """
    Client-side handle for a remote Maki agent.

    Args:
        endpoint:           Base URL of the remote AgentServer.
        api_key:            Bearer token (None = open access).
        timeout:            Per-request HTTP timeout in seconds.
        failure_threshold:  Consecutive transient failures before circuit opens.
        recovery_timeout:   Seconds before a HALF_OPEN probe is attempted.
        ssl_verify:         True (default), False, or path to a CA-bundle file.
        cert:               Client certificate: path string or (certfile, keyfile) tuple.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        ssl_verify: Union[bool, str] = True,
        cert: Optional[Any] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.last_trace_id: Optional[str] = None

        # Operator-configured endpoint: LAN/private addresses are legitimate.
        self._http = Connector(
            timeout=timeout,
            allow_private=True,
            verify=ssl_verify,
            cert=cert,
        )

        # Populated lazily on first call to connect() or execute_task().
        self.agent_id: str = ""
        self.name: str = ""
        self.role: str = ""
        self.plugins: dict = {}
        self._connected = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, trace_id: Optional[str] = None) -> dict:
        h: dict = {}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if trace_id:
            h[TRACE_HEADER] = trace_id
        return h

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path}"

    def _get(self, path: str, trace_id: Optional[str] = None):
        return self._http.get(self._url(path), headers=self._headers(trace_id))

    def _post(self, path: str, payload: dict, trace_id: Optional[str] = None):
        return self._http.post(
            self._url(path), json=payload, headers=self._headers(trace_id)
        )

    def _delete(self, path: str, trace_id: Optional[str] = None):
        return self._http.delete(self._url(path), headers=self._headers(trace_id))

    def connect(self) -> None:
        """Fetch agent metadata from /info and mark the proxy as connected.

        Called automatically on the first execute_task() / stream_task() so that
        constructing an AgentProxy does not require the remote server to be up.
        Call it explicitly if you need the proxy's name/role/plugins fields
        populated before the first task.

        Raises:
            MakiNetworkError: if the remote /info endpoint is unreachable.
        """
        r = self._get("/info")
        data = r.json()
        self.agent_id = data.get("agent_id", "")
        self.name = data.get("name", "")
        self.role = data.get("role", "")
        self.plugins = {p: None for p in data.get("plugins", [])}
        self._connected = True
        logger.info("AgentProxy connected: %s at %s", self.name, self.endpoint)

    def _refresh_info(self) -> None:
        """Backward-compat alias for connect()."""
        self.connect()

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def execute_task(
        self,
        task: str,
        context: Optional[Dict] = None,
        use_plugins: bool = False,
        trace_id: Optional[str] = None,
    ) -> str:
        """Dispatch *task* to the remote agent and return its result.

        Raises MakiNetworkError immediately if the circuit breaker is open.
        """
        if not self._connected:
            self.connect()

        if not self._circuit_breaker.allow_request():
            raise MakiNetworkError(
                f"Circuit breaker OPEN for {self.endpoint}: agent is temporarily unavailable. "
                f"({self._circuit_breaker.failure_count} consecutive failures)"
            )

        request_trace = trace_id or str(uuid.uuid4())
        payload: dict = {"task": task, "use_plugins": use_plugins}
        if context:
            payload["context"] = context

        try:
            r = self._post("/execute", payload, trace_id=request_trace)
            self._circuit_breaker.record_success()
            body = r.json()
            self.last_trace_id = body.get("trace_id", request_trace)
            logger.debug("[trace=%s] execute_task OK (%s)", self.last_trace_id, self.name)
            return body["result"]
        except _RETRYABLE:
            self._circuit_breaker.record_failure()
            raise

    def execute_task_with_retry(
        self,
        task: str,
        context: Optional[Dict] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """Execute *task* with retry on transient failures.

        If the circuit breaker opens mid-retry the loop aborts immediately
        rather than sleeping and retrying again.
        """
        for attempt in range(max_retries):
            if not self._circuit_breaker.allow_request():
                raise MakiNetworkError(
                    f"Circuit breaker OPEN for {self.endpoint}: aborting retries"
                )
            try:
                return self.execute_task(task, context)
            except _RETRYABLE as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "AgentProxy '%s': attempt %d/%d failed: %s — retrying in %.1fs",
                    self.name, attempt + 1, max_retries, e, retry_delay,
                )
                time.sleep(retry_delay)
            except Exception:
                raise
        raise RuntimeError(f"Task '{task}' failed after {max_retries} attempts")

    def stream_task(
        self,
        task: str,
        context: Optional[Dict] = None,
        use_plugins: bool = False,
        trace_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Yield text chunks streamed from the remote agent via SSE.

        The circuit-breaker check runs immediately (before the generator is
        returned), so callers get a fast-fail even if they haven't started
        consuming chunks yet.
        """
        if not self._connected:
            self.connect()

        if not self._circuit_breaker.allow_request():
            raise MakiNetworkError(
                f"Circuit breaker OPEN for {self.endpoint}: streaming unavailable"
            )
        request_trace = trace_id or str(uuid.uuid4())
        return self._stream_generator(task, context, use_plugins, request_trace)

    def _stream_generator(
        self,
        task: str,
        context: Optional[Dict],
        use_plugins: bool,
        trace_id: str,
    ) -> Generator[str, None, None]:
        payload: dict = {"task": task, "use_plugins": use_plugins}
        if context:
            payload["context"] = context
        headers = self._headers(trace_id)

        try:
            response = self._http.post(
                self._url("/stream"),
                json=payload,
                headers=headers,
                stream=True,
                raise_on_status=False,
            )
        except _RETRYABLE:
            self._circuit_breaker.record_failure()
            raise

        try:
            # Maps non-2xx onto the Maki tree; 4xx (MakiAPIError) does not
            # trip the breaker, matching execute_task semantics.
            Connector.raise_for_response(response)
            self._circuit_breaker.record_success()
            for raw in Connector.iter_lines(response):
                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                if not line.startswith("data: "):
                    continue
                data_part = line[len("data: "):]
                if data_part == "[DONE]":
                    return
                data = json.loads(data_part)
                if "error" in data:
                    raise MakiNetworkError(
                        f"Stream error from remote agent: {data['error']}"
                    )
                if "chunk" in data:
                    yield data["chunk"]
        except _RETRYABLE:
            self._circuit_breaker.record_failure()
            raise
        finally:
            try:
                response.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    def run_loop(
        self,
        goal: str,
        requirements: Sequence[Union[str, Dict[str, Any], Requirement]],
        on_iteration: Optional[Callable[[Dict[str, Any]], Optional[Decision]]] = None,
        **options: Any,
    ) -> RemoteLoopResult:
        """Run a loop on the remote server and follow it until it stops.

        Requirements are text only (strings, ``{"id", "text"}`` dicts, or ``Requirement``
        objects without a command or check): the server never executes anything a caller
        sends, and the judge is the server's own ``judge:`` agent. ``options`` may hold
        ``max_iterations``, ``max_seconds``, ``max_tokens``, ``stall_limit``, ``context``, ``use_plugins``.

        ``on_iteration`` is called with each iteration event dict (see
        :mod:`maki.distributed.loop_jobs`); returning ``Decision.STOP`` cancels the remote
        loop (it stops after the iteration in flight). Notes cannot be sent to a remote loop.

        Raises:
            ValueError: for unsupported requirements or options.
            MakiAPIError: the server rejected the request (e.g. 400, 409 busy).
            MakiNetworkError: unreachable, loops disabled (503), or the stream kept dropping.
        """
        unknown = set(options) - set(_LOOP_OPTIONS)
        if unknown:
            raise ValueError(f"unsupported loop option(s) for a remote loop: {sorted(unknown)}")
        payload: Dict[str, Any] = {"goal": goal, "requirements": [self._remote_requirement(r) for r in requirements]}
        payload.update({k: v for k, v in options.items() if v is not None})
        loop_id = self._loop_call("post", "/loop", payload)["id"]
        return self._follow_loop(loop_id, on_iteration)

    def resume_loop(
        self,
        loop_id: str,
        on_iteration: Optional[Callable[[Dict[str, Any]], Optional[Decision]]] = None,
        **options: Any,
    ) -> RemoteLoopResult:
        """Continue a stopped or interrupted remote loop from its checkpoint and follow it."""
        unknown = set(options) - {"max_iterations", "max_seconds", "max_tokens"}
        if unknown:
            raise ValueError(f"unsupported resume option(s): {sorted(unknown)}")
        body = {k: v for k, v in options.items() if v is not None}
        self._loop_call("post", f"/loop/{self._check_loop_id(loop_id)}/resume", body)
        return self._follow_loop(loop_id, on_iteration)

    def approve_loop(
        self,
        loop_id: str,
        approve: bool = True,
        note: str = "",
        resume: bool = False,
        on_iteration: Optional[Callable[[Dict[str, Any]], Optional[Decision]]] = None,
    ) -> Union[Dict[str, Any], RemoteLoopResult]:
        """Answer a suspended remote loop's pending approval.

        Approve (a note is shown to the agent) lets the next iteration run; reject with a
        note continues with that note; reject without a note stops the loop. Returns the
        loop's status, or with ``resume=True`` resumes it and follows it to its next stop.

        Raises:
            MakiAPIError: 404 unknown loop, 409 nothing pending.
        """
        snapshot = self._loop_call("post", f"/loop/{self._check_loop_id(loop_id)}/approval",
                                   {"approve": bool(approve), "note": note})
        if not resume:
            return snapshot
        return self.resume_loop(loop_id, on_iteration=on_iteration)

    def loop_status(self, loop_id: str) -> Dict[str, Any]:
        """The remote loop's status snapshot."""
        return self._loop_call("get", f"/loop/{self._check_loop_id(loop_id)}")

    def cancel_loop(self, loop_id: str) -> None:
        """Ask the remote loop to stop after its current iteration."""
        self._loop_call("delete", f"/loop/{self._check_loop_id(loop_id)}")

    @staticmethod
    def _check_loop_id(loop_id: str) -> str:
        if not isinstance(loop_id, str) or not LOOP_ID_PATTERN.match(loop_id):
            raise ValueError(f"not a loop id: {loop_id!r}")
        return loop_id

    @staticmethod
    def _remote_requirement(req: Union[str, Dict[str, Any], Requirement]) -> Union[str, Dict[str, Any]]:
        if isinstance(req, Requirement):
            if req.kind != "agent":
                raise ValueError(f"requirement '{req.id}' has a {req.kind} check; remote loops accept text "
                                 "requirements only")
            return {"id": req.id, "text": req.text}
        return req

    def _loop_call(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        if not self._connected:
            self.connect()
        if not self._circuit_breaker.allow_request():
            raise MakiNetworkError(f"Circuit breaker OPEN for {self.endpoint}: agent is temporarily unavailable")
        try:
            if method == "post":
                r = self._post(path, payload or {})
            elif method == "delete":
                r = self._delete(path)
            else:
                r = self._get(path)
            self._circuit_breaker.record_success()
            return r.json()
        except _RETRYABLE:
            self._circuit_breaker.record_failure()
            raise

    def _follow_loop(self, loop_id: str, on_iteration: Optional[Callable]) -> RemoteLoopResult:
        events: List[Dict[str, Any]] = []
        last_index = -1
        cancelled = False
        failures = 0
        while True:
            try:
                for index, event in self._loop_events(loop_id, last_index):
                    failures = 0
                    last_index = index
                    events.append(event)
                    if event.get("type") == "stopped":
                        return RemoteLoopResult(
                            loop_id, str(event.get("stop_reason", "")), int(event.get("iterations", 0)),
                            str(event.get("detail", "")), str(event.get("error", "")),
                            str(event.get("artifact", "")), int(event.get("tokens", 0)), events)
                    if event.get("type") == "iteration" and on_iteration is not None and not cancelled:
                        try:
                            decision = on_iteration(event)
                        except BaseException:  # don't leave the remote loop running unattended
                            self._cancel_quietly(loop_id)
                            raise
                        cancelled = self._apply_loop_decision(loop_id, decision)
            except _RETRYABLE:
                self._circuit_breaker.record_failure()
            # Reaching here means the stream dropped or ended without a stopped event:
            # reconnect and replay from the last event received.
            failures += 1
            if failures > _MAX_STREAM_RECONNECTS:
                raise MakiNetworkError(f"Loop event stream for {loop_id} kept ending before the loop stopped")
            time.sleep(min(0.5 * failures, 3.0))

    def _cancel_quietly(self, loop_id: str) -> None:
        try:
            self.cancel_loop(loop_id)
        except MakiAPIError as e:
            if e.status_code != 409:  # 409: the loop had already stopped
                logger.warning("could not cancel remote loop %s: %s", loop_id, e)
        except Exception as e:  # noqa: BLE001 - best effort; the caller's own error is what matters
            logger.warning("could not cancel remote loop %s: %s", loop_id, e)

    def _apply_loop_decision(self, loop_id: str, decision: Optional[Decision]) -> bool:
        """Act on a hook decision; return True if the remote loop was asked to stop."""
        if decision is None:
            return False
        if decision.note:
            logger.warning("a reviewer note cannot be sent to a remote loop; ignored")
        if not decision.stop:
            return False
        try:
            self.cancel_loop(loop_id)
        except MakiAPIError as e:
            if e.status_code != 409:  # 409: the loop had already stopped
                raise
        return True

    def _loop_events(self, loop_id: str, after: int) -> Generator[tuple, None, None]:
        """Yield ``(index, event)`` from the server's SSE stream, replaying from ``after``."""
        if not self._circuit_breaker.allow_request():
            raise MakiNetworkError(f"Circuit breaker OPEN for {self.endpoint}: loop stream unavailable")
        path = f"/loop/{loop_id}/events" + (f"?after={after}" if after >= 0 else "")
        response = self._http.get(self._url(path), headers=self._headers(), stream=True, raise_on_status=False)
        try:
            Connector.raise_for_response(response)
            self._circuit_breaker.record_success()
            index = after
            for raw in Connector.iter_lines(response):
                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                if line.startswith("id: ") and line[4:].strip().isdigit():
                    index = int(line[4:])
                elif line == "data: [DONE]":
                    return
                elif line.startswith("data: "):
                    yield index, json.loads(line[len("data: "):])
        finally:
            try:
                response.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any) -> None:
        self._post("/memory/set", {"key": key, "value": value})

    def recall(self, key: str) -> Any:
        try:
            r = self._get(f"/memory/{key}")
        except MakiAPIError as e:
            if e.status_code == 404:
                return None
            raise
        return r.json()["value"]

    def clear_memory(self) -> None:
        self._delete("/memory")

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def reset_conversation(self) -> None:
        self._delete("/history")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @property
    def circuit_state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        return self._circuit_breaker.state

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AgentProxy(name={self.name!r}, endpoint={self.endpoint!r}, "
            f"circuit={self._circuit_breaker.state.value})"
        )

    def __del__(self) -> None:
        try:
            http = getattr(self, "_http", None)
            if http is not None:
                http.close()
        except Exception:
            pass
