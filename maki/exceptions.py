"""
Custom exception classes for the Maki framework
"""


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

    def __init__(self, message: str = "", status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def error_for_status(status: int, message: str) -> MakiError:
    """The Maki exception for an HTTP error status, the same for every backend and the Connector.

    408 and 504 are timeouts and any 5xx is a transient server failure (both retryable);
    everything else, 4xx included, is a request the caller must change (``MakiAPIError``).
    """
    if status in (408, 504):
        return MakiTimeoutError(message)
    if isinstance(status, int) and status >= 500:
        return MakiNetworkError(message)
    return MakiAPIError(message, status_code=status if isinstance(status, int) else 0)
