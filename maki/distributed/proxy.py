"""
AgentProxy: a drop-in replacement for Agent that dispatches every call to a
remote AgentServer over HTTP.

From AgentManager's perspective an AgentProxy is indistinguishable from a
local Agent: both expose execute_task(), execute_task_with_retry(),
stream_task(), remember(), recall(), and clear_memory().

Usage
-----
    from maki.distributed.proxy import AgentProxy

    proxy = AgentProxy(endpoint="http://worker-node:8100", api_key="secret")
    result = proxy.execute_task("summarise the report")

Exception mapping
-----------------
    httpx timeout              → MakiTimeoutError
    httpx connection error     → MakiNetworkError
    HTTP 408 / 504             → MakiTimeoutError
    HTTP 5xx                   → MakiNetworkError
    HTTP 4xx                   → MakiAPIError
"""
import json
import logging
import time
from typing import Any, Dict, Generator, Optional

try:
    import httpx
except ImportError as _e:
    raise ImportError(
        "AgentProxy requires httpx. "
        'Install it with: pip install "maki[distributed]"'
    ) from _e

from ..exceptions import MakiAPIError, MakiNetworkError, MakiTimeoutError

logger = logging.getLogger(__name__)

# Errors that are worth retrying (mirrors agent.py)
_RETRYABLE = (MakiNetworkError, MakiTimeoutError)


class AgentProxy:
    """
    Client-side handle for a remote Maki agent.

    Implements the same calling interface as Agent so that AgentManager
    (and any other orchestration code) can treat local and remote agents
    identically.

    Args:
        endpoint:  Base URL of the remote AgentServer, e.g. "http://host:8100".
        api_key:   Bearer token expected by the server (None = open access).
        timeout:   Per-request timeout in seconds (default 60).
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._session = httpx.Client(timeout=timeout)

        # Populated from /info on construction.
        self.agent_id: str = ""
        self.name: str = ""
        self.role: str = ""
        self.plugins: dict = {}

        self._refresh_info()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

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

    def _get(self, path: str) -> "httpx.Response":
        try:
            r = self._session.get(self._url(path), headers=self._headers())
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on GET {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on GET {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _post(self, path: str, payload: dict) -> "httpx.Response":
        try:
            r = self._session.post(self._url(path), json=payload, headers=self._headers())
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on POST {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on POST {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _delete(self, path: str) -> "httpx.Response":
        try:
            r = self._session.delete(self._url(path), headers=self._headers())
        except httpx.TimeoutException as e:
            raise MakiTimeoutError(f"Timeout on DELETE {path}") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Request failed on DELETE {path}: {e}") from e
        self._raise_for_response(r)
        return r

    def _refresh_info(self) -> None:
        """Fetch agent metadata from the remote /info endpoint."""
        r = self._get("/info")
        data = r.json()
        self.agent_id = data.get("agent_id", "")
        self.name = data.get("name", "")
        self.role = data.get("role", "")
        self.plugins = {p: None for p in data.get("plugins", [])}
        logger.info("AgentProxy connected: %s at %s", self.name, self.endpoint)

    # ------------------------------------------------------------------
    # Task execution — mirrors Agent's public interface
    # ------------------------------------------------------------------

    def execute_task(
        self,
        task: str,
        context: Optional[Dict] = None,
        use_plugins: bool = False,
    ) -> str:
        """Dispatch *task* to the remote agent and return its result."""
        payload: dict = {"task": task, "use_plugins": use_plugins}
        if context:
            payload["context"] = context
        r = self._post("/execute", payload)
        return r.json()["result"]

    def execute_task_with_retry(
        self,
        task: str,
        context: Optional[Dict] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """Execute *task* with retry on transient network/timeout errors."""
        for attempt in range(max_retries):
            try:
                return self.execute_task(task, context)
            except _RETRYABLE as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "AgentProxy '%s': task failed (attempt %d/%d): %s — retrying in %.1fs",
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
    ) -> Generator[str, None, None]:
        """Yield text chunks streamed from the remote agent via SSE."""
        params = {"task": task, "use_plugins": use_plugins}
        try:
            with self._session.stream(
                "GET", self._url("/stream"),
                params=params, headers=self._headers(),
            ) as response:
                self._raise_for_response(response)
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
        except httpx.TimeoutException as e:
            raise MakiTimeoutError("Timeout during streaming") from e
        except httpx.ConnectError as e:
            raise MakiNetworkError(f"Cannot connect to {self.endpoint}: {e}") from e
        except httpx.RequestError as e:
            raise MakiNetworkError(f"Stream request failed: {e}") from e

    # ------------------------------------------------------------------
    # Memory — proxied to the remote agent's /memory endpoints
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any) -> None:
        """Store *key*/*value* in the remote agent's memory."""
        self._post("/memory/set", {"key": key, "value": value})

    def recall(self, key: str) -> Any:
        """Retrieve *key* from the remote agent's memory (None if absent)."""
        try:
            r = self._get(f"/memory/{key}")
        except MakiAPIError as e:
            if "404" in str(e):
                return None
            raise
        return r.json()["value"]

    def clear_memory(self) -> None:
        """Clear all memory in the remote agent."""
        self._delete("/memory")

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def reset_conversation(self) -> None:
        """Clear the remote agent's conversation history."""
        self._delete("/history")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"AgentProxy(name={self.name!r}, endpoint={self.endpoint!r})"

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
