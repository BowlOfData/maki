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