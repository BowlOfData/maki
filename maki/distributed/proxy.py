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
from typing import Any, Dict, Generator, Optional, Union

from ..connector import Connector
from ..exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError
from .circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)

_RETRYABLE = (MakiNetworkError, MakiTimeoutError)

TRACE_HEADER = "X-Maki-Trace-Id"


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
