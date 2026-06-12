"""
AgentProxy: a drop-in replacement for Agent that dispatches every call to a
remote AgentServer over HTTP.

From AgentManager's perspective an AgentProxy is indistinguishable from a
local Agent: both expose execute_task(), execute_task_with_retry(),
stream_task(), remember(), recall(), and clear_memory().

New in Phase 5
--------------
Circuit breaker  — After failure_threshold consecutive transient failures
                   the proxy stops attempting requests and raises immediately.
                   Resets automatically after recovery_timeout seconds.

Distributed tracing — Every execute_task / stream_task call generates a UUID
                   trace_id sent as X-Maki-Trace-Id.  The server echoes it back
                   in the response body and all server-side logs.  Stored as
                   last_trace_id for inspection after each call.

mTLS             — Pass ssl_verify=False or a CA-bundle path, and/or
                   cert=(certfile, keyfile) for mutual TLS.

Exception mapping
-----------------
    httpx timeout              → MakiTimeoutError
    httpx connection error     → MakiNetworkError
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

try:
    import httpx
except ImportError as _e:
    raise ImportError(
        "AgentProxy requires httpx. "
        'Install it with: pip install "maki[distributed]"'
    ) from _e

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

        # Build httpx.Client with optional SSL overrides.
        client_kwargs: dict = {"timeout": timeout}
        if ssl_verify is not True:
            client_kwargs["verify"] = ssl_verify
        if cert is not None:
            client_kwargs["cert"] = cert
        self._session = httpx.Client(**client_kwargs)

        # Populated from /info on construction.
        self.agent_id: str = ""
        self.name: str = ""
        self.role: str = ""
        self.plugins: dict = {}

        self._refresh_info()

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

    def _raise_for_response(self, response: "httpx.Response") -> None:
        if response.is_success:
            return
        code = response.status_code
        detail = response.text
        if code in (408, 504):
            raise MakiTimeoutError(f"Remote agent timeout (HTTP {code}): {detail}")
        if code >= 500:
            raise MakiNetworkError(f"Remote agent server error (HTTP {code}): {detail}")
        raise MakiAPIError(f"Remote agent rejected request (HTTP {code}): {detail}")

    def _get(self, path: str, trace_id: Optional[str] = None) -> "httpx.Response":
        try:
            r = self._session.get(self._url(path), headers=self._headers(trace_id))
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on GET {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on GET {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _post(self, path: str, payload: dict, trace_id: Optional[str] = None) -> "httpx.Response":
        try:
            r = self._session.post(self._url(path), json=payload, headers=self._headers(trace_id))
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on POST {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on POST {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _delete(self, path: str, trace_id: Optional[str] = None) -> "httpx.Response":
        try:
            r = self._session.delete(self._url(path), headers=self._headers(trace_id))
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on DELETE {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on DELETE {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _refresh_info(self) -> None:
        """Fetch agent metadata from /info (called at construction, no circuit check)."""
        r = self._get("/info")
        data = r.json()
        self.agent_id = data.get("agent_id", "")
        self.name = data.get("name", "")
        self.role = data.get("role", "")
        self.plugins = {p: None for p in data.get("plugins", [])}
        logger.info("AgentProxy connected: %s at %s", self.name, self.endpoint)

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
            with self._session.stream(
                "POST", self._url("/stream"), json=payload, headers=headers,
            ) as response:
                if not response.is_success:
                    # A streaming body must be read before .text is available;
                    # otherwise httpx raises ResponseNotRead instead of the
                    # mapped Maki error.
                    response.read()
                self._raise_for_response(response)
                self._circuit_breaker.record_success()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        return
                    data = json.loads(payload)
                    if "error" in data:
                        raise MakiNetworkError(
                            f"Stream error from remote agent: {data['error']}"
                        )
                    if "chunk" in data:
                        yield data["chunk"]
        except _RETRYABLE:
            self._circuit_breaker.record_failure()
            raise
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure()
            raise MakiTimeoutError("Timeout during streaming") from e
        except httpx.ConnectError as e:
            self._circuit_breaker.record_failure()
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure()
            raise MakiNetworkError(f"Stream request failed: {e}") from e

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any) -> None:
        self._post("/memory/set", {"key": key, "value": value})

    def recall(self, key: str) -> Any:
        try:
            r = self._get(f"/memory/{key}")
        except MakiAPIError as e:
            if "404" in str(e):
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
            self._session.close()
        except Exception:
            pass
