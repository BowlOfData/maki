"""
Circuit breaker for AgentProxy.

Tracks consecutive transient failures (MakiNetworkError, MakiTimeoutError) on a
single remote endpoint and temporarily blocks requests after a configurable
failure threshold, giving the remote node time to recover.

States
------
CLOSED    Normal operation; requests are allowed through.
OPEN      Too many consecutive failures; requests fail immediately.
HALF_OPEN Recovery window open; one probe request is allowed.
          Success → CLOSED.  Failure → OPEN (timer reset).

Thread-safety
-------------
State transitions and counters are guarded by a single threading.Lock.
The OPEN → HALF_OPEN transition is evaluated lazily on each allow_request() call
once recovery_timeout seconds have elapsed since the last failure.
"""
import enum
import threading
import time


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Args:
        failure_threshold:  Consecutive failures required to open the circuit.
        recovery_timeout:   Seconds to wait before entering HALF_OPEN.
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be >= 0")
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state, with lazy OPEN → HALF_OPEN transition on timeout."""
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def allow_request(self) -> bool:
        """Return True if the caller may proceed with the HTTP request."""
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Reset the breaker to CLOSED after a successful call."""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a transient failure; open the circuit if threshold is reached."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == CircuitState.HALF_OPEN or self._failure_count >= self._threshold:
                self._state = CircuitState.OPEN

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self._state.value}, "
            f"failures={self._failure_count}/{self._threshold})"
        )
