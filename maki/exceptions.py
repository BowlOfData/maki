"""
Custom exception classes for the Maki framework
"""

import datetime
import email.utils
import time
from typing import Any, Optional


class MakiError(Exception):
    """Base exception for all Maki framework errors"""
    pass


class MakiValidationError(MakiError):
    """Exception raised for validation errors"""
    pass


class MakiNetworkError(MakiError):
    """Exception raised for network-related errors"""
    pass


class MakiTimeoutError(MakiNetworkError):
    """Exception raised for timeout errors"""
    pass


class MakiAPIError(MakiError):
    """Exception raised for API response errors"""

    def __init__(self, message: str = "", status_code: int = 0, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after  # seconds the server asked callers to wait (429), when it said


def parse_retry_after(value: Any) -> Optional[float]:
    """Seconds from a ``Retry-After`` header value (delta-seconds or an HTTP date); None if absent or invalid."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:  # "-0000" means UTC with no zone information; naive would read as local time
            when = when.replace(tzinfo=datetime.timezone.utc)
        seconds = when.timestamp() - time.time()
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return None
    return max(0.0, seconds)


def retry_after_of(response_or_error: Any) -> Optional[float]:
    """The ``Retry-After`` of an HTTP response, or of an SDK error that carries one (``error.response``)."""
    response = getattr(response_or_error, "response", response_or_error)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
        if value is None and isinstance(headers, dict):  # a plain dict is case-sensitive; HTTP headers are not
            value = next((v for k, v in headers.items() if str(k).lower() == "retry-after"), None)
        return parse_retry_after(value)
    except Exception:
        return None


def error_for_status(status: int, message: str, retry_after: Optional[float] = None) -> MakiError:
    """The Maki exception for an HTTP error status, the same for every backend and the Connector.

    408 and 504 are timeouts and any 5xx is a transient server failure (both retryable);
    everything else, 4xx included, is a request the caller must change (``MakiAPIError``).
    """
    if status in (408, 504):
        return MakiTimeoutError(message)
    if isinstance(status, int) and status >= 500:
        return MakiNetworkError(message)
    return MakiAPIError(message, status_code=status if isinstance(status, int) else 0, retry_after=retry_after)
